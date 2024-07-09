import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# # 환경 변수 설정
os.environ["TORCH_COMPILE_DEBUG"] = "1"

# 모델 정의
class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(MLP, self).__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.silu = F.silu

    def forward(self, input):
        return self.down_proj(self.silu(self.gate_proj(input)) * self.up_proj(input))

# 하이퍼파라미터 설정
hidden_size = 4096
intermediate_size = 14336
batch_size = 32

# 모델 인스턴스 생성 및 GPU로 이동
model = MLP(hidden_size, intermediate_size).cuda().to(torch.bfloat16)

# 입력 데이터 준비
input_prefill = torch.randn(1, 39, hidden_size, dtype=torch.bfloat16).cuda()
input_decoding = torch.randn(1, 1, hidden_size, dtype=torch.bfloat16).cuda()

# 모델 컴파일 (동적 shape 지원)
# compiled_model = torch.compile(model, mode="reduce-overhead")
compiled_model = torch.compile(model, mode="max-autotune")

# Prefill 단계 실행 (2번 실행)
for i in range(2):
    print(f"Prefill 단계 {i+1} 시작...")
    start_time = time.time()
    output_prefill = compiled_model(input_prefill)
    end_time = time.time()
    print(f"Prefill 단계 {i+1} 완료. 소요 시간: {end_time - start_time:.4f}초")

# Decoding 단계 실행 (3번 실행)
for i in range(3):
    print(f"Decoding 단계 {i+1} 시작...")
    start_time = time.time()
    output_decoding = compiled_model(input_decoding)
    end_time = time.time()
    print(f"Decoding 단계 {i+1} 완료. 소요 시간: {end_time - start_time:.4f}초")

# 결과 출력
print(f"Prefill Output shape: {output_prefill.shape}")
print(f"Decoding Output shape: {output_decoding.shape}")
