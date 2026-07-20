from __future__ import annotations

import numpy as np



def sum_with_shape(grad: np.ndarray, target_shape: tuple [int, ...]) -> np.ndarray:
    result = grad
    while result.ndim > len(target_shape):
        result = result.sum(axis=0)

    for axis, target_size in enumerate(target_shape):
        if target_size == 1 and result.shape[axis] != 1:
            result = result.sum(axis=axis, keepdims=True)
    
    return result.reshape(target_shape)



def accumulate_grad(current_grad: np.ndarray | None, contribution: np.ndarray) -> np.ndarray:
    if current_grad is None:
        return contribution.copy()
    return current_grad + contribution