from nabla.nn.module import Module
from nabla.nn.parameter import Parameter
from nabla.tensor import Tensor


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        out = x.relu()
        return out

    def parameters(self) -> list[Parameter]:
        return []
