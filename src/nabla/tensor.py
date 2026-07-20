from __future__ import annotations

from collections.abc import Callable
from numbers import Real

import numpy as np
import numpy.typing as npt
from nabla._utils import accumulate_grad, sum_with_shape

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
        other_tensor = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(
            self.data + other_tensor.data,
            requires_grad=self.requires_grad or other_tensor.requires_grad,
            _prev=(self, other_tensor),
            _op="add",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            
            if self.requires_grad:
                contribution = sum_with_shape(out.grad, self.data.shape)
                self.grad = accumulate_grad(self.grad, contribution)

            if other_tensor.requires_grad:
                contribution = sum_with_shape(out.grad, other_tensor.data.shape)
                other_tensor.grad = accumulate_grad(other_tensor.grad, contribution)
            
                    

        out._backward = _backward;

        return out

    def __radd__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        return self + other

    def __mul__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        raise NotImplementedError

    def __rmul__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        raise NotImplementedError

    def __neg__(self) -> Tensor:
        raise NotImplementedError

    def __sub__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        raise NotImplementedError

    def __rsub__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        raise NotImplementedError

    def __pow__(self, exponent: Real) -> Tensor:
        raise NotImplementedError

    def zero_grad(self) -> None:
        self.grad = None

    def backward(self, grad: npt.ArrayLike | None = None) -> None:
        raise NotImplementedError

    def relu(self) -> Tensor:
        raise NotImplementedError