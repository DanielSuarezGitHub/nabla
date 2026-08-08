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

        out._backward = _backward

        return out

    def __radd__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        return self + other

    def __mul__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        other_tensor = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(
            self.data * other_tensor.data,
            requires_grad=self.requires_grad or other_tensor.requires_grad,
            _prev=(self, other_tensor),
            _op="mul",
        )

        def _backward() -> None:
            if out.grad is None:
                return

            if self.requires_grad:
                contribution = sum_with_shape(out.grad * other_tensor.data, self.data.shape)
                self.grad = accumulate_grad(self.grad, contribution)

            if other_tensor.requires_grad:
                contribution = sum_with_shape(out.grad * self.data, other_tensor.data.shape)
                other_tensor.grad = accumulate_grad(other_tensor.grad, contribution)

        out._backward = _backward

        return out

    def __rmul__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        return self * other

    def __neg__(self) -> Tensor:
        return self * -1

    def __sub__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        other_tensor = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other_tensor)

    def __rsub__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        other_tensor = other if isinstance(other, Tensor) else Tensor(other)
        return other_tensor + (-self)

    def __pow__(self, exponent: int | float) -> Tensor:
        out = Tensor(
            self.data**exponent,
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="pow",
        )

        def _backward() -> None:
            if out.grad is None:
                return

            if self.requires_grad:
                contribution = out.grad * exponent * self.data ** (exponent - 1)
                self.grad = accumulate_grad(self.grad, contribution)

        out._backward = _backward
        return out

    def __matmul__(self, other: Tensor | npt.ArrayLike) -> Tensor:
        return self.matmul(other)
        pass

    def matmul(self, other: Tensor | npt.ArrayLike) -> Tensor:
        other_tensor = other if isinstance(other, Tensor) else Tensor(other)

        if not (self.data.ndim == 2 and other_tensor.data.ndim == 2):
            raise ValueError("unsupported currently for tensors that arent 2d")

        out = Tensor(
            self.data @ other_tensor.data,
            requires_grad=self.requires_grad or other_tensor.requires_grad,
            _prev=(self, other_tensor),
            _op="matmul",
        )

        def _backward() -> None:
            if out.grad is None:
                return

            if self.requires_grad:
                contribution = out.grad @ other_tensor.data.T
                self.grad = accumulate_grad(self.grad, contribution)

            if other_tensor.requires_grad:
                contribution = self.data.T @ out.grad
                other_tensor.grad = accumulate_grad(other_tensor.grad, contribution)

        out._backward = _backward

        return out

    def zero_grad(self) -> None:
        self.grad = None

    def backward(self) -> None:
        if not self.requires_grad:
            raise RuntimeError("Operation .backward() needs this to require grad be set to true")
        if self.data.size != 1:
            raise RuntimeError(".backward must be called on scalar output")
        self.grad = np.ones_like(self.data)

        visited: set[int] = set()
        topo: list[Tensor] = []

        def topologicalSort(tensor: Tensor) -> None:
            tensor_id = id(tensor)

            if tensor_id in visited:
                return

            visited.add(tensor_id)

            for parent in tensor._prev:
                topologicalSort(parent)

            topo.append(tensor)

        topologicalSort(self)

        for tensor in reversed(topo):
            tensor._backward()

    def relu(self) -> Tensor:
        out = Tensor(
            np.maximum(0.0, self.data),
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="relu",
        )

        def _backward() -> None:
            if out.grad is None:
                return

            if self.requires_grad:
                contribution = out.grad * (self.data > 0)
                self.grad = accumulate_grad(self.grad, contribution)

        out._backward = _backward
        return out

    def sum(self) -> Tensor:
        out = Tensor(
                np.sum(self.data),
                requires_grad=self.requires_grad,
                _prev=(self,),
                 _op='sum'
                 )

        def _backward() -> None:
            if out.grad is None:
                return

            if self.requires_grad:
                contribution = out.grad * np.ones_like(self.data)
                self.grad = accumulate_grad(self.grad, contribution)

        out._backward = _backward
        return out

    def mean(self) -> Tensor:
        return self.sum() * (1 / self.data.size)
