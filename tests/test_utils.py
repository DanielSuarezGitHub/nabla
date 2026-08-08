import numpy as np
import pytest

from nabla._utils import accumulate_grad, sum_with_shape


def test_accumulate_grad_initializes_none() -> None:
    contribution = np.array([1.0, 2.0, 3.0])

    result = accumulate_grad(None, contribution)

    assert np.array_equal(result, [1.0, 2.0, 3.0])


def test_accumulate_grad_adds_to_existing_gradient() -> None:
    current = np.array([1.0, 2.0, 3.0])
    contribution = np.array([10.0, 20.0, 30.0])

    result = accumulate_grad(current, contribution)

    assert np.array_equal(result, [11.0, 22.0, 33.0])



@pytest.mark.parametrize(
    ("grad", "target_shape", "expected"),
    [
        # Same shape: no reduction needed
        (
            np.array([
                [1.0, 2.0],
                [3.0, 4.0],
            ]),
            (2, 2),
            np.array([
                [1.0, 2.0],
                [3.0, 4.0],
            ]),
        ),

        # Matrix -> scalar (0D)
        (
            np.array([
                [5.0, 8.0],
                [9.0, 24.0],
            ]),
            (),
            np.array(46.0),
        ),

        #  Matrix -> 1D array
        (
            np.array([
                [5.0, 10.0],
                [7.0, 14.0],
            ]),
            (2,),
            np.array([12.0, 24.0]),
        ),

        # Matrix -> column vector
        (
            np.array([
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]),
            (2, 1),
            np.array([
                [6.0],
                [15.0],
            ]),
        ),

        # Matrix -> row vector
        (
            np.array([
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]),
            (1, 3),
            np.array([
                [5.0, 7.0, 9.0],
            ]),
        ),

        # Matrix -> 1D array of length 1
        (
            np.array([
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]),
            (1,),
            np.array([21.0]),
        ),
    ],
)


def test_sum_with_shape( grad: np.ndarray, target_shape: tuple[int, ...], expected: np.ndarray,) -> None:  # noqa: E501
    result = sum_with_shape(grad, target_shape)

    assert np.array_equal(result, expected)
    assert result.shape == target_shape