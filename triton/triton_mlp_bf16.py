import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

import os
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
os.environ["TORCH_USE_CUDA_DSA"]="1"

triton.Config.debug = True

# MLP 모델 정의
class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(MLP, self).__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.silu = F.silu

    def forward(self, input):
        return self.down_proj(self.silu(self.gate_proj(input)) * self.up_proj(input))

# Triton 커널 정의
@triton.jit
def triton_tem_fused_mm_0(arg_A, arg_B, out_ptr0,
                          M, N, K,
                          stride_am, stride_ak, stride_bk, stride_bn,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                          GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = arg_A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = arg_B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(K, 0, -BLOCK_K):
        a = tl.load(A)
        b = tl.load(B)
        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)
    tl.store(out_ptr0 + (idx_n + N * idx_m), acc, mask=mask)

@triton.jit
def triton_tem_fused_mm_mul_silu_1(arg_A, arg_B, in_ptr2, out_ptr0, out_ptr1,
                                   M, N, K,
                                   stride_am, stride_ak, stride_bk, stride_bn,
                                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                                   GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = arg_A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = arg_B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(K, 0, -BLOCK_K):
        a = tl.load(A)
        b = tl.load(B)
        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)
    
    xindex = idx_n + (N * idx_m)
    tl.store(out_ptr0 + xindex, acc, mask)
    tmp0 = tl.load(in_ptr2 + xindex, mask)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp4 * acc
    tl.store(out_ptr1 + xindex, tmp5, mask)

@triton.jit
def triton_tem_fused_mm_2(arg_A, arg_B, out_ptr0,
                          M, N, K,
                          stride_am, stride_ak, stride_bk, stride_bn,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                          GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = arg_A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = arg_B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    c = accumulator.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_ptrs = out_ptr0 + offs_cm[:, None] * N + offs_cn[None, :]
    tl.store(out_ptrs, c, mask=offs_cm[:, None] < M and offs_cn[None, :] < N)


# Triton 구현
def triton_mlp(gate_proj, up_proj, down_proj, input):
    batch_size, seq_len, hidden_size = input.shape
    input_reshaped = input.view(batch_size * seq_len, hidden_size)

    M, K, N = batch_size * seq_len, hidden_size, 14336
    
    buf0 = torch.empty((batch_size * seq_len, 14336), device='cuda', dtype=torch.bfloat16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    triton_tem_fused_mm_0[grid](input_reshaped, gate_proj.t(), buf0, 
                                M, N, K, 
                                input_reshaped.stride(0), input_reshaped.stride(1),
                                gate_proj.stride(0), gate_proj.stride(1),
                                BLOCK_M=16, BLOCK_N=64, BLOCK_K=32, GROUP_M=8)

    buf1 = torch.empty((batch_size * seq_len, 14336), device='cuda', dtype=torch.bfloat16)
    buf2 = torch.empty((batch_size * seq_len, 14336), device='cuda', dtype=torch.bfloat16)
    triton_tem_fused_mm_mul_silu_1[grid](input_reshaped, up_proj.t(), buf0, buf1, buf2,
                                         M, N, K,
                                         input_reshaped.stride(0), input_reshaped.stride(1),
                                         up_proj.stride(0), up_proj.stride(1),
                                         BLOCK_M=16, BLOCK_N=64, BLOCK_K=32, GROUP_M=8)

    M, K, N = batch_size * seq_len, 14336, 4096
    buf3 = torch.empty((batch_size * seq_len, 4096), device='cuda', dtype=torch.bfloat16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    triton_tem_fused_mm_2[grid](buf2, down_proj.t(), buf3,
                            M, N, K,
                            buf2.stride(0), buf2.stride(1),
                            down_proj.t().stride(0), down_proj.t().stride(1),
                            BLOCK_M=16, BLOCK_N=64, BLOCK_K=32, GROUP_M=8)

    return buf3.view(batch_size, seq_len, 4096)

# 하이퍼파라미터 설정
hidden_size = 4096
intermediate_size = 14336

# MLP 모델 인스턴스 생성 및 GPU로 이동
model = MLP(hidden_size, intermediate_size).cuda().to(torch.bfloat16)

# 입력 데이터 준비
input_prefill = torch.randn(1, 39, hidden_size, dtype=torch.bfloat16, device='cuda')
input_decoding = torch.randn(1, 1, hidden_size, dtype=torch.bfloat16, device='cuda')

# MLP 모델 실행
with torch.no_grad():
    for i in range(5):
        print(f"Torch의 Prefill 단계 {i+1} 시작...")
        start_time = time.time()
        output_prefill = model(input_prefill)
        end_time = time.time()
        print(f"Torch의 Prefill 단계 {i+1} 완료. 소요 시간: {end_time - start_time:.8f}초")

    for i in range(5):
        print(f"Triton의 Prefill 단계 {i+1} 시작...")
        start_time = time.time()
        output_prefill_triton = triton_mlp(model.gate_proj.weight, model.up_proj.weight, model.down_proj.weight, input_prefill)
        end_time = time.time()
        print(f"Triton의 Prefill 단계 {i+1} 완료. 소요 시간: {end_time - start_time:.8f}초")

    for i in range(5):
        print(f"Torch의 Decoding 단계 {i+1} 시작...")
        start_time = time.time()
        output_decoding = model(input_decoding)
        end_time = time.time()
        print(f"Torch의 Decoding 단계 {i+1} 완료. 소요 시간: {end_time - start_time:.8f}초")

    for i in range(5):
        print(f"Triton의 Decoding 단계 {i+1} 시작...")
        start_time = time.time()
        output_decoding_triton = triton_mlp(model.gate_proj.weight, model.up_proj.weight, model.down_proj.weight, input_decoding)
        end_time = time.time()
        print(f"Triton의 Decoding 단계 {i+1} 완료. 소요 시간: {end_time - start_time:.8f}초")

torch.cuda.synchronize()

# 결과 비교 (Prefill)
is_close_prefill = torch.allclose(output_prefill, output_prefill_triton, rtol=1e-2, atol=1e-2)
max_diff_prefill = torch.max(torch.abs(output_prefill - output_prefill_triton))

print(f"Prefill results are close: {is_close_prefill}")
print(f"Prefill maximum difference: {max_diff_prefill.item()}")

# 결과 비교 (Decoding)
is_close_decoding = torch.allclose(output_decoding, output_decoding_triton, rtol=1e-2, atol=1e-2)
max_diff_decoding = torch.max(torch.abs(output_decoding - output_decoding_triton))

print(f"Decoding results are close: {is_close_decoding}")
print(f"Decoding maximum difference: {max_diff_decoding.item()}")
