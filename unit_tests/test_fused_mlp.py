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
        input_tensor = self.generate_input(batch_size, seq_len, dtype)
        self.layer.to(dtype)

        with torch.no_grad():
            torch_output = self.layer(input_tensor)

        fused_mlp_output = kbkim_kernels.fused_mlp(
            input_tensor,
            self.layer.gate_proj.weight,
            self.layer.up_proj.weight,
            self.layer.down_proj.weight,
            block_size
        )
        
        self.assertTrue(
            torch.allclose(torch_output, fused_mlp_output, atol=1e-5),
            f"The output is not close for batch_size={batch_size}, seq_len={seq_len}, block_size={block_size} dtype={dtype}"
        )

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
