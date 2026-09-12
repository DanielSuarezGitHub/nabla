import numpy as np
from numpy.typing import NDArray

from nabla._utils import accumulate_grad
from nabla.nn.module import Module
from nabla.nn.parameter import Parameter
from nabla.tensor import Tensor


class CrossEntropyLoss(Module):
    def forward(
        self,
        logits: Tensor,
        targets: NDArray[np.integer],
    ) -> Tensor:
        z = logits.data
        B = z.shape[0]

        m = np.max(z, axis=1, keepdims=True)
        shifted = z - m

        exp_shifted = np.exp(shifted)
        probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

        logsumexp = m.squeeze(1) + np.log(np.sum(exp_shifted, axis=1))
        correct_logits = z[np.arange(B), targets]

        losses = -correct_logits + logsumexp
        loss_value = losses.mean()

        out = Tensor(
            loss_value,
            requires_grad=logits.requires_grad,
            _prev=(logits,),
            _op="cross_entropy",
        )

        def _backward() -> None:
            if out.grad is None:
                return

            if logits.requires_grad:
                grad = probs.copy()
                grad[np.arange(B), targets] -= 1
                grad /= B

                contribution = out.grad * grad
                logits.grad = accumulate_grad(logits.grad, contribution)

        out._backward = _backward
        return out

    def parameters(self) -> list[Parameter]:
        return []