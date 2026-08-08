import numpy as np

from nabla.nn.module import Module
from nabla.nn.paramater import Parameter
from nabla.tensor import Tensor


class Linear(Module):
    w: Parameter
    b: Parameter
    in_features: int
    out_features: int
    
    def __init__(self, in_features: int, out_features: int):
        self.w = Parameter(
            np.random.normal(
                loc=0.0,
                scale=0.1,
                size=(in_features, out_features),
            )
        )
        self.b = Parameter(np.zeros(out_features))
        self.in_features = in_features
        self.out_features = out_features
        

    def forward(self, x: Tensor) -> Tensor:
        out = (x @ self.w) + self.b
        return out

    def parameters(self) -> list[Parameter]:
        return [self.w, self.b]
