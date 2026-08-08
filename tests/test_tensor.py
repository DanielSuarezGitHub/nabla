from typing import cast

import numpy as np

from nabla.nn.paramater import Parameter
from nabla.tensor import Tensor


def test_tensor_stores_data() -> None:
    tensor = Tensor([1.0, 2.0, 3.0])

    expected = np.array([1.0, 2.0, 3.0])

    assert np.array_equal(tensor.data, expected)


def test_add_forward() -> None:
    a = Tensor([1.0, 2.0])
    b = Tensor([3.0, 4.0])

    out = a + b

    assert np.array_equal(out.data, [4.0, 6.0])
    assert out.data.shape == (2,)


def test_add_requires_grad() -> None:
    a = Tensor([1.0, 2.0], requires_grad=False)
    b = Tensor([3.0, 4.0], requires_grad=True)

    out = a + b

    assert out.requires_grad is True


def test_add_backward() -> None:
    a = Tensor([1.0, 2.0], requires_grad=True)
    b = Tensor([3.0, 4.0], requires_grad=True)

    out = a + b

    out.grad = np.array([5.0, 7.0])
    out._backward()

    assert np.array_equal(cast(np.ndarray, a.grad), [5.0, 7.0])
    assert np.array_equal(cast(np.ndarray, b.grad), [5.0, 7.0])


def test_add_backward_same_tensor() -> None:
    x = Tensor([1.0, 2.0], requires_grad=True)

    out = x + x

    out.grad = np.array([3.0, 5.0])
    out._backward()

    assert np.array_equal(cast(np.ndarray, x.grad), [6.0, 10.0])


def test_add_backward_with_broadcasting() -> None:
    a = Tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        requires_grad=True,
    )
    b = Tensor([10.0, 20.0], requires_grad=True)

    out = a + b

    out.grad = np.array(
        [
            [5.0, 8.0],
            [9.0, 24.0],
        ]
    )
    out._backward()

    assert np.array_equal(
        cast(np.ndarray, a.grad),
        [
            [5.0, 8.0],
            [9.0, 24.0],
        ],
    )
    assert np.array_equal(cast(np.ndarray, b.grad), [14.0, 32.0])


def test_mul_forward() -> None:
    a = Tensor([1.0, 2.0])
    b = Tensor([3.0, 4.0])

    out = a * b

    assert np.array_equal(out.data, [3.0, 8.0])
    assert out.data.shape == (2,)


def test_mul_backward_same_shape() -> None:
    a = Tensor([1.0, 2.0], requires_grad=True)
    b = Tensor([3.0, 4.0], requires_grad=True)

    out = a * b

    out.grad = np.array([5.0, 7.0])
    out._backward()

    assert np.array_equal(cast(np.ndarray, a.grad), [15.0, 28.0])
    assert np.array_equal(cast(np.ndarray, b.grad), [5.0, 14.0])


def test_mul_backward_with_broadcasting() -> None:
    a = Tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        requires_grad=True,
    )
    b = Tensor([10.0, 20.0], requires_grad=True)

    out = a * b

    out.grad = np.array(
        [
            [5.0, 8.0],
            [9.0, 24.0],
        ]
    )
    out._backward()

    assert np.array_equal(
        cast(np.ndarray, a.grad),
        [
            [50.0, 160.0],
            [90.0, 480.0],
        ],
    )
    assert np.array_equal(cast(np.ndarray, b.grad), [32.0, 112.0])


def test_sum_forward_returns_scalar_total() -> None:
    a = Tensor([[12, 6], [2, 1]], requires_grad=True)

    out = a.sum()
    assert np.array_equal(out.data, np.array(21.0))


def test_sum_backward_produces_ones() -> None:
    a = Tensor([[12, 6], [2, 1]], requires_grad=True)
    out = a.sum()
    out.backward()
    assert np.array_equal(cast(np.ndarray, a.grad), [[1, 1], [1, 1]])


def test_sum_backward_end_to_end() -> None:
    a = Tensor([[1, 4], [14, 8]], requires_grad=True)
    (a * 3).sum().backward()
    assert np.array_equal(cast(np.ndarray, a.grad), [[3, 3], [3, 3]])


def test_mamtmul_forward() -> None:
    a = Tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    b = Tensor(
        [
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )

    o = a @ b
    assert np.array_equal(cast(np.ndarray, o.data), [[19, 22.0], [43, 50]])


def test_matmul_backward_both_operands() -> None:
    a = Tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        requires_grad=True,
    )

    b = Tensor(
        [
            [5.0, 6.0],
            [7.0, 8.0],
        ],
        requires_grad=True,
    )

    o = a @ b
    o.sum().backward()
    assert np.array_equal(cast(np.ndarray, a.grad), [[11, 15], [11, 15]])
    assert np.array_equal(cast(np.ndarray, b.grad), [[4, 4], [6, 6]])


def test_linear_computation_path_backward_all_inputs() -> None:
    x = Tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        requires_grad=True,
    )

    w = Tensor(
        [
            [5.0, 6.0],
            [7.0, 8.0],
        ],
        requires_grad=True,
    )

    b = Tensor([10.0, 20.0], requires_grad=True)  # behaves like row

    out = (x @ w + b).sum()
    out.backward()
    assert np.array_equal(
        cast(np.ndarray, x.grad),
        [
            [11.0, 15.0],
            [11.0, 15.0],
        ],
    )

    assert np.array_equal(
        cast(np.ndarray, w.grad),
        [
            [4.0, 4.0],
            [6.0, 6.0],
        ],
    )

    assert np.array_equal(
        cast(np.ndarray, b.grad),
        [2.0, 2.0],
    )


def test_parameter_requires_grad_by_default() -> None:
    p = Parameter([1.0, 2.0])

    assert p.requires_grad is True
    assert isinstance(p, Tensor)