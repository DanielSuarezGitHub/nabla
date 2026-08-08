


from nabla.nn.module import Module
from nabla.tensor import Tensor


class MSELoss(Module):
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        error = pred - target
        return (error ** 2).mean()