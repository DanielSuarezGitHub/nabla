from collections.abc import Iterable

from nabla.nn.parameter import Parameter


class SGD:
    parameters: tuple[Parameter, ...]
    lr: float

    def __init__(
        self,
        parameters: Iterable[Parameter],
        learning_rate: float,
    ):
        self.parameters = tuple(parameters)
        self.lr = learning_rate

    def zero_grad(self) -> None:
        for paramater in self.parameters:
            paramater.zero_grad()

    def step(self) -> None:
        for paramater in self.parameters:
            if paramater.grad is not None:
                paramater.data -= paramater.grad * self.lr
