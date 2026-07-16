import numpy as np

from nabla.tensor import Tensor


def test_tensor_stores_data() -> None:
    tensor = Tensor([1.0, 2.0, 3.0])

    expected = np.array([1.0, 2.0, 3.0])

    assert np.array_equal(tensor.data, expected)