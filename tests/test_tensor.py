import numpy as np
from typing import cast
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

    assert np.array_equal(cast(np.ndarray,x.grad), [6.0, 10.0])


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

    out.grad = np.array([
        [5.0, 8.0],
        [9.0, 24.0],
    ])
    out._backward()

    assert np.array_equal(cast(np.ndarray, a.grad), [
        [5.0, 8.0],
        [9.0, 24.0],
    ])
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

    out.grad = np.array([
        [5.0, 8.0],
        [9.0, 24.0],
    ])
    out._backward()

    assert np.array_equal(cast(np.ndarray, a.grad), [
        [50.0, 160.0],
        [90.0, 480.0],
    ])
    assert np.array_equal(cast(np.ndarray, b.grad), [32.0, 112.0])
