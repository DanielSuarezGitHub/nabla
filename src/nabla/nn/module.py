from abc import ABC, abstractmethod

from nabla.nn.parameter import Parameter
from nabla.tensor import Tensor


class Module(ABC):
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        ...

    @abstractmethod
    def parameters(self) -> list[Parameter]:
        ...

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)