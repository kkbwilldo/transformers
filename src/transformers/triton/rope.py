import torch
import triton
import triton.language as tl
import torch.autograd as autograd


class RoPEKernel:

    """
    RoPEKernel은 RoPE의 forward와 backward 연산을 수행하는 Triton 커널입니다.
    """

    @staticmethod
    @triton.jit
    def forward(
        input_ptr,
        input_seq_len_stride,
        input_batch_stride,
        output_ptr,
        cos_ptr,
        sin_ptr,
        cos_stride,
        sin_stride,
        seq_len,
        head_dim,
        BLOCKSIZE: tl.constexpr,
        BATCH_SIZE: tl.constexpr        
    ):

        # 블록의 오프셋을 계산
        # grid = (seq_len, num_heads)
        block_id_seq = tl.program_id(axis=0)
        block_id_head = tl.program_id(axis=1)

        head_dim_offset = tl.arange(0, BLOCKSIZE)
        head_dim_mid = head_dim // 2

        mask = head_dim_offset < head_dim_mid

        # cos_offset = block_id_seq * cos_stride + head_dim_offset
        cos_offset = (block_id_seq % seq_len) * cos_stride + head_dim_offset
        sin_offset = (block_id_seq % seq_len) * sin_stride + head_dim_offset

        # 메모리 경계를 넘어가면 0.0을 반환
        cos = tl.load(cos_ptr + cos_offset, mask=mask, other=0.0)
        sin = tl.load(sin_ptr + cos_offset, mask=mask, other=0.0)

        # thread 1개로 여러 샘플을 처리
        # - 병렬 처리 효율화
        # - cos, sin을 재사용하여 메모리 접근을 줄임
        for batch_idx in tl.static_range(0, BATCH_SIZE):
            front_offset = (block_id_seq * input_seq_len_stride) + \
                           (batch_idx * input_batch_stride) + \
                           (block_id_head * head_dim) + \
                           head_dim_offset
            back_offset = (block_id_seq * input_seq_len_stride) + \
                          (batch_idx * input_batch_stride) + \
                          (block_id_head * head_dim) + \
                          head_dim_mid + head_dim_offset

            front = tl.load(input_ptr + front_offset, mask=mask, other=0.0)
            back = tl.load(input_ptr + back_offset, mask=mask, other=0.0)

            front_val = front * cos - back * sin
            back_val = front * sin + back * cos

            tl.store(output_ptr + front_offset, front_val, mask=mask)
            tl.store(output_ptr + back_offset, back_val, mask=mask)
        
        return


class RoPEFunction(autograd.Function):
    
    """
    RoPEFunction은 RoPE의 forward와 backward 연산을 수행하는 PyTorch autograd 함수입니다.
    """

    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):

        # from (batch_size, seq_len, num_heads, head_dim)
        # to (seq_len, batch_size, num_heads, head_dim)
        t = t.transpose(0,1)

        seq_len, batch_size, num_heads, head_dim = t.shape
        output = torch.empty_like(t)

        BLOCK_SIZE = triton.next_power_of_2(head_dim//2)

        grid = (seq_len, num_heads)

        RoPEKernel.forward[grid](
            t,
            t.stride(0), # seq_len dim
            t.stride(1), # batch_size dim
            output,
            cos,
            sin,
            cos.stride(0), # seq_len dim
            sin.stride(0), # seq_len dim
            seq_len,
            head_dim,
            BLOCK_SIZE,
            batch_size
        )

        ctx.cos = cos
        ctx.sin = sin
        ctx.BLOCKS_SIZE = BLOCK_SIZE

        return output

    @staticmethod
    def backward(
        ctx,
        grad_output
    ):

        raise NotImplementedError("backward 함수는 아직 구현되지 않았습니다.")
   
 
def kbkim_apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor = None,
    unsqueeze_dim: int = 1
    ):
    
    """
    kbkim_apply_rotary_pos_emb 함수는 query tensor와 key tensor에
    Rotary Position Embedding을 적용하는 함수입니다.
    """
   
    # from (batch_size, seq_len, head_dim)
    # to (seq_len, 1, 1, head_dim)
    # 이는 모든 배치 내 샘플이 seq_len 축을 따라 같은 theta를 가지기 때문에
    # broadcasting하여 공유하기 위함입니다.
    cos = cos[0].unsqueeze(0).unsqueeze(0).contiguous()
    sin = sin[0].unsqueeze(0).unsqueeze(0).contiguous()
 
    q_embed = RoPEFunction.apply(q, cos, sin)
    k_embed = RoPEFunction.apply(k, cos, sin)

    return q_embed, k_embed
