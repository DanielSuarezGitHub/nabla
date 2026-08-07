import numpy as np
import torch

from nabla.tensor import Tensor


def test_add_forward_matches_torch() -> None:
    data_a = [[1.0, 2.0], [3.0, 4.0]]
    data_b = [[5.0, 6.0], [7.0, 8.0]]

    a = Tensor(data_a)
    b = Tensor(data_b)

    torch_a = torch.tensor(data_a, dtype=torch.float64)
    torch_b = torch.tensor(data_b, dtype=torch.float64)

    out = a + b
    torch_out = torch_a + torch_b

    assert np.allclose(
        out.data,
        torch_out.numpy(),
    )


def test_matmul_backward_matches_torch() -> None:
    x_data = [[1.0, 2.0], [3.0, 4.0]]
    w_data = [[5.0, 6.0], [7.0, 8.0]]

    x = Tensor(x_data, requires_grad=True)
    w = Tensor(w_data, requires_grad=True)

    tx = torch.tensor(x_data, dtype=torch.float64, requires_grad=True)
    tw = torch.tensor(w_data, dtype=torch.float64, requires_grad=True)

    (x @ w).sum().backward()
    (tx @ tw).sum().backward()

    assert x.grad is not None
    assert w.grad is not None
    assert tx.grad is not None
    assert tw.grad is not None

    assert np.allclose(x.grad, tx.grad.numpy())
    assert np.allclose(w.grad, tw.grad.numpy())