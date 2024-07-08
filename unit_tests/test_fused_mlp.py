import torch
import unittest
import itertools
import kbkim_kernels
import torch.nn as nn
import torch.nn.functional as F

from parameterized import parameterized

_FLOAT_TYPES_ = [torch.float16, torch.float32, torch.float64, torch.bfloat16]

ACT2FN = {
    "relu": F.relu,
    "gelu": F.gelu,
    "silu": F.silu,
    "tanh": torch.tanh,
}

class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, activation_fn):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[activation_fn]

    def forward(self, x):
        return self.down_proj(
            self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        )

class TestFusedMLP(unittest.TestCase):
    def setUp(self):

        # configuration settings of llama3
        self.hidden_size = 4096
        self.intermediate_size = 14336
        self.activation_fn = "silu"
        self.layer = MLP(self.hidden_size, self.intermediate_size, self.activation_fn).cuda()
        self.layer.eval()

    def generate_input(self, batch_size, seq_len, dtype):
        return torch.randn(batch_size, seq_len, self.hidden_size, dtype=dtype).cuda()

    @parameterized.expand(itertools.product([1], [8], [256], _FLOAT_TYPES_))
    def test_fused_mlp(self, batch_size, seq_len, block_size, dtype):
        input_tensor = self.generate_input(batch_size, seq_len, dtype).contiguous()
        self.layer.to(dtype)

        with torch.no_grad():
#             torch_output = self.layer(input_tensor)
            gate_proj = self.layer.gate_proj(input_tensor)
            up_proj = self.layer.up_proj(input_tensor)
            gate_proj_activated = self.layer.act_fn(gate_proj)
            intermediate = gate_proj_activated * up_proj
            torch_output = self.layer.down_proj(intermediate)

        fused_mlp_output, fused_gate_proj, fused_up_proj, fused_intermediate = kbkim_kernels.fused_mlp(
            input_tensor,
            self.layer.gate_proj.weight.contiguous(),
            self.layer.up_proj.weight.contiguous(),
            self.layer.down_proj.weight.contiguous(),
            block_size
        )
        print(f"gate_proj weight shape: {self.layer.gate_proj.weight.shape}")

        print(f"gate_proj shape: {gate_proj.shape}")
        print(f"fused_gate_proj shape: {fused_gate_proj.shape}")
        print(f"up_proj shape: {up_proj.shape}")
        print(f"fused_up_proj shape: {fused_up_proj.shape}")
        print(f"down_proj shape: {torch_output.shape}")
        print(f"fused_down_proj shape: {fused_mlp_output.shape}")
        print(f"intermediate shape: {intermediate.shape}")
        print(f"fused_intermediate shape: {fused_intermediate.shape}")
        
        print(f"Max difference in gate_proj: {(gate_proj - fused_gate_proj).abs().max().item()}")
        print(f"Max difference in up_proj: {(up_proj - fused_up_proj).abs().max().item()}")
        print(f"Max difference in intermediate: {(intermediate - fused_intermediate).abs().max().item()}")
        print(f"Max difference in final output: {(torch_output - fused_mlp_output).abs().max().item()}")

        self.assertTrue(
            torch.allclose(torch_output, fused_mlp_output, atol=1e-5),
            f"The output is not close for batch_size={batch_size}, seq_len={seq_len}, block_size={block_size} dtype={dtype}"
        )

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
