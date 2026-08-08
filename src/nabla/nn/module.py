from abc import ABC, abstractmethod
from typing import Any

from nabla.nn.parameter import Parameter
from nabla.tensor import Tensor


class Module(ABC):
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        ...

    @abstractmethod
    def parameters(self) -> list[Parameter]:
        ...

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)