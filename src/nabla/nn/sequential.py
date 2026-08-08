from nabla.nn.module import Module
from nabla.nn.parameter import Parameter
from nabla.tensor import Tensor


class Sequential(Module):
    modules: tuple[Module, ...]

    def __init__(self, *modules: Module):
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.modules:
            out = layer(out)
        return out

    def parameters(self) -> list[Parameter]:
        params: list[Parameter] = []

        for module in self.modules:
            params.extend(module.parameters())

        return params
