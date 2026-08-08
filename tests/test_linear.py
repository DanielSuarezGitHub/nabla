from typing import cast

import numpy as np
import torch

from nabla.nn.linear import Linear
from nabla.tensor import Tensor


def test_linear_creates_weight_with_correct_shape() -> None:
    x = Linear(3, 5)
    assert np.ndim(x.w.data) == 2
    assert x.w.data.shape == (3, 5)


def test_linear_creates_bias_with_correct_shape() -> None:
    x = Linear(3, 5)
    assert np.ndim(x.b.data) == 1
    assert x.b.data.shape == (5,)

# torch does x @ w.t transpose thats why we transpose our weights 
def test_linear_backward_matches_pytorch() -> None:
    x_data = [[1.0, 2.0, 3.0]]

    linear = Linear(3, 5)
    torch_linear = torch.nn.Linear(3, 5)
    
    with torch.no_grad():
        torch_linear.weight.copy_(torch.tensor(linear.w.data.T, dtype=torch.float64))
        torch_linear.bias.copy_(torch.tensor(linear.b.data, dtype=torch.float64))

    x = Tensor(x_data, requires_grad=True)

    torch_x = torch.tensor(
        x_data,
        dtype=torch.float32,
        requires_grad=True,
    )

    our_output = linear(x)
    torch_output = torch_linear(torch_x)

    our_output.sum().backward()
    torch_output.sum().backward()

    assert x.grad is not None
    assert linear.w.grad is not None
    assert linear.b.grad is not None
    assert torch_x.grad is not None
    assert torch_linear.weight.grad is not None
    assert torch_linear.bias.grad is not None

    assert np.allclose(
        x.grad,
        torch_x.grad.numpy(),
    )

    assert np.allclose(
        linear.w.grad,
        torch_linear.weight.grad.numpy().T,
    )

    assert np.allclose(
        linear.b.grad,
        torch_linear.bias.grad.numpy(),
    )
