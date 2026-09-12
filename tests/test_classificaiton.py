import numpy as np
import torch
import torch.nn.functional as F

from nabla.nn.cross_entropy import CrossEntropyLoss
from nabla.tensor import Tensor

def test_cross_entropy_forward_matches_pytorch() -> None:
    logits_data = np.array(
        [
            [2.0, 1.0, 0.1],
            [0.5, 2.5, 1.0],
            [1.0, 0.2, 3.0],
        ]
    )
    targets = np.array([0, 1, 2])

    logits = Tensor(logits_data, requires_grad=True)
    loss = CrossEntropyLoss()(logits, targets)

    torch_logits = torch.tensor(
        logits_data,
        dtype=torch.float64,
        requires_grad=True,
    )
    torch_targets = torch.tensor(targets, dtype=torch.long)
    torch_loss = F.cross_entropy(torch_logits, torch_targets)

    np.testing.assert_allclose(
        loss.data,
        torch_loss.detach().numpy(),
        rtol=1e-7,
        atol=1e-7,
    )


def test_cross_entropy_backward_matches_pytorch() -> None:
    logits_data = np.array(
        [
            [2.0, 1.0, 0.1],
            [0.5, 2.5, 1.0],
            [1.0, 0.2, 3.0],
        ]
    )
    targets = np.array([0, 1, 2])

    logits = Tensor(logits_data, requires_grad=True)
    loss = CrossEntropyLoss()(logits, targets)
    loss.backward()

    torch_logits = torch.tensor(
        logits_data,
        dtype=torch.float64,
        requires_grad=True,
    )
    torch_targets = torch.tensor(targets, dtype=torch.long)
    torch_loss = F.cross_entropy(torch_logits, torch_targets)
    torch_loss.backward()
    assert logits.grad is not None
    assert torch_logits.grad is not None
    np.testing.assert_allclose(
        logits.grad,
        torch_logits.grad.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )
