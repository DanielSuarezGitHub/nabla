from __future__ import annotations

from collections.abc import Callable
from numbers import Real

import numpy as np
import numpy.typing as npt


class Tensor:
    data: np.ndarray
    grad: np.ndarray | None
    requires_grad: bool
    _prev: tuple[Tensor, ...]
    _op: str
    _backward: Callable[[], None]

    def __init__(
        self,
        data: npt.ArrayLike,
        *,
        requires_grad: bool = False,
        _prev: tuple[Tensor, ...] = (),
        _op: str = "",
    ) -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None


        self.requires_grad = requires_grad
        self._prev = _prev
        self._op = _op

        self._backward = lambda: None

        def __add__(self, other: Tensor | npt.ArrayLike) -> Tensor:
            raise NotImplementedError
        
        def __radd__(self, other: Tensor | npt.ArrayLike) -> Tensor:
            raise NotImplementedError
        
        def __mul__(self, other: Tensor | npt.ArrayLike) -> Tensor:
            raise NotImplementedError
        
        def __rmul__(self, other: Tensor | npt.ArrayLike) -> Tensor:
            raise NotImplementedError

        def __neg__(self) -> Tensor:
            raise NotImplementedError
        
        def __sub__(self) -> Tensor:
            raise NotImplementedError
        
        def __rsub__(self) -> Tensor:
            raise NotImplementedError

        def __pow__(self, exponent: Real) -> Tensor:
            raise NotImplementedError
        
        def zero_grad(self) -> None:
            self.grad = None
        
        def backward(self, grad: npt.ArrayLike | None = None) -> None:
            raise NotImplementedError\
        
        def relu(self) -> Tensor:
            raise NotImplementedError
        
       