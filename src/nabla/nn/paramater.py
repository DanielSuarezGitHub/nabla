from nabla.tensor import Tensor


class Parameter(Tensor):
    def __init__(self, data) -> None:
        super().__init__(data, requires_grad=True)