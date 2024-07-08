#include <torch/extension.h>
#include <cuda_runtime.h>
#include <tuple>
#include <cmath>

#define MIN(a,b) ((a)<(b)?(a):(b))

// CUDA 커널 함수
template <typename scalar_t>
__global__
void fused_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ gate_proj_weights,
    const scalar_t* __restrict__ up_proj_weights,
    const scalar_t* __restrict__ down_proj_weights,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ gate_proj_output,
    scalar_t* __restrict__ up_proj_output,
    scalar_t* __restrict__ intermediate_output,
    int batch_size,
    int seq_len,
    int hidden_size,
    int intermediate_size
) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int inter_idx = threadIdx.x + blockIdx.z * blockDim.x;

    if (inter_idx < intermediate_size) {
        int input_offset = (batch_idx * seq_len + seq_idx) * hidden_size;
        int intermediate_offset = (batch_idx * seq_len + seq_idx) * intermediate_size;

        scalar_t gate_proj_val = 0.0;
        scalar_t up_proj_val = 0.0;

        for (int i = 0; i < hidden_size; ++i) {
            gate_proj_val += x[input_offset + i] * gate_proj_weights[inter_idx * hidden_size + i];
            up_proj_val += x[input_offset + i] * up_proj_weights[inter_idx * hidden_size + i];
        }

        // SiLU activation function: x * sigmoid(x)
        scalar_t silu_gate = gate_proj_val / (1.0 + exp(-gate_proj_val)) * gate_proj_val;

        // Store intermediate results
        gate_proj_output[intermediate_offset + inter_idx] = gate_proj_val;
        up_proj_output[intermediate_offset + inter_idx] = up_proj_val;
        
        scalar_t intermediate = silu_gate * up_proj_val;
        intermediate_output[intermediate_offset + inter_idx] = intermediate;

        // Compute final output
        scalar_t down_proj_val = 0.0;
        for (int i = 0; i < hidden_size; ++i) {
            down_proj_val += intermediate * down_proj_weights[i * intermediate_size + inter_idx];
        }

        output[input_offset + (inter_idx%hidden_size)] = down_proj_val;
        gate_proj_output[intermediate_offset + inter_idx] = gate_proj_val;
        up_proj_output[intermediate_offset + inter_idx] = up_proj_val;
        intermediate_output[intermediate_offset + inter_idx] = intermediate;

        if(inter_idx < hidden_size){
            output[input_offset + inter_idx] = down_proj_val;
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> fused_mlp(
    torch::Tensor x,
    torch::Tensor gate_proj_weights,
    torch::Tensor up_proj_weights,
    torch::Tensor down_proj_weights,
    int block_size
) {
    auto batch_size = x.size(0);
    auto seq_len = x.size(1);
    auto hidden_size = x.size(2);
    auto intermediate_size = gate_proj_weights.size(0);

    auto output = torch::zeros_like(x);
    auto gate_proj_output = torch::zeros({batch_size, seq_len, intermediate_size}, x.options());
    auto up_proj_output = torch::zeros({batch_size, seq_len, intermediate_size}, x.options());
    auto intermediate_output = torch::zeros({batch_size, seq_len, intermediate_size}, x.options());

    const int threads = 256;  // 또는 다른 적절한 값
    const int blocks_z = (intermediate_size + threads - 1) / threads;
    const dim3 blocks(batch_size, seq_len, blocks_z);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "fused_mlp", ([&] {
        fused_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            gate_proj_weights.data_ptr<scalar_t>(),
            up_proj_weights.data_ptr<scalar_t>(),
            down_proj_weights.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            gate_proj_output.data_ptr<scalar_t>(),
            up_proj_output.data_ptr<scalar_t>(),
            intermediate_output.data_ptr<scalar_t>(),
            batch_size, seq_len, hidden_size, intermediate_size
        );
    }));

    return std::make_tuple(output, gate_proj_output, up_proj_output, intermediate_output);
}

// Python 모듈 초기화
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_mlp", &fused_mlp, "Fused MLP kernel",
        pybind11::arg("x"),
        pybind11::arg("gate_proj_weights"),
        pybind11::arg("up_proj_weights"),
        pybind11::arg("down_proj_weights"),
        pybind11::arg("block_size")
    );
}
