#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// CUDA 커널 함수
template <typename scalar_t>
__global__
void fused_kernel(
    const scalar_t* __restrict__ x, 
    const scalar_t* __restrict__ gate_proj_weights,
    const scalar_t* __restrict__ up_proj_weights, 
    const scalar_t* __restrict__ down_proj_weights,
    scalar_t* __restrict__ output, 
    int batch_size, 
    int seq_len, 
    int hidden_size, 
    int intermediate_size) {

    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int hidden_idx = threadIdx.x;

    if (hidden_idx < hidden_size) {
        int input_idx = (batch_idx * seq_len + seq_idx) * hidden_size + hidden_idx;
        scalar_t gate_proj_val = 0.0;
        scalar_t up_proj_val = 0.0;

        for (int i = 0; i < intermediate_size; ++i) {
            gate_proj_val += x[input_idx] * gate_proj_weights[hidden_idx * intermediate_size + i];
            up_proj_val += x[input_idx] * up_proj_weights[hidden_idx * intermediate_size + i];
        }

        // SiLU activation function
        gate_proj_val = gate_proj_val / (1.0 + exp(-gate_proj_val));

        scalar_t down_proj_val = 0.0;
        for (int i = 0; i < intermediate_size; ++i) {
            down_proj_val += gate_proj_val * up_proj_val * down_proj_weights[i * hidden_size + hidden_idx];
        }

        output[input_idx] = down_proj_val;
    }
}

// Python에서 호출 가능한 인터페이스 함수
void fused_mlp(
    torch::Tensor x, 
    torch::Tensor gate_proj_weights, 
    torch::Tensor up_proj_weights, 
    torch::Tensor down_proj_weights, 
    torch::Tensor output, 
    int batch_size, 
    int seq_len, 
    int hidden_size, 
    int intermediate_size, 
    int block_size) {
    
    const dim3 blocks(batch_size, seq_len);

    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, x.scalar_type(), "fused_mlp", ([&] {
        fused_kernel<scalar_t><<<blocks, block_size>>>(
            x.data_ptr<scalar_t>(),
            gate_proj_weights.data_ptr<scalar_t>(),
            up_proj_weights.data_ptr<scalar_t>(),
            down_proj_weights.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, seq_len, hidden_size, intermediate_size
        );
    }));
}

// Python 모듈 초기화
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_mlp", &fused_mlp, "Fused MLP kernel");
}
