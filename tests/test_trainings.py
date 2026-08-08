import numpy as np

from nabla.nn.linear import Linear
from nabla.nn.mse import MSELoss
from nabla.nn.relu import ReLU
from nabla.nn.sequential import Sequential
from nabla.optim.sgd import SGD
from nabla.tensor import Tensor


def test_linear_model_learns_simple_function() -> None:
    x = Tensor(
        np.array([
            [-2.0],
            [-1.0],
            [0.0],
            [1.0],
            [2.0],
        ])
    )

    y = Tensor(3 * x.data + 2)

    model = Linear(1, 1)
    loss_fn = MSELoss()
    optimizer = SGD(model.parameters(), learning_rate=0.05)

    initial_pred = model(x)
    initial_loss = loss_fn(initial_pred, y).data.item()

    for _ in range(200):
        optimizer.zero_grad()

        pred = model(x)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()

    final_pred = model(x)
    final_loss = loss_fn(final_pred, y).data.item()

    assert final_loss < initial_loss
    assert final_loss < 1e-3

    assert np.allclose(model.w.data, [[3.0]], atol=0.05)
    assert np.allclose(model.b.data, [2.0], atol=0.05)





def test_mlp_learns_nonlinear_absolute_value() -> None:
    x_data = np.linspace(-2.0, 2.0, 100).reshape(-1, 1)
    y_data = np.abs(x_data)

    x = Tensor(x_data)
    y = Tensor(y_data)

    model = Sequential(
        Linear(1, 8),
        ReLU(),
        Linear(8, 1),
    )

    loss_fn = MSELoss()
    optimizer = SGD(
        model.parameters(),
        learning_rate=0.01,
    )

    initial_loss = loss_fn(model(x), y).data.item()

    for _ in range(2000):
        optimizer.zero_grad()

        prediction = model(x)
        loss = loss_fn(prediction, y)

        loss.backward()
        optimizer.step()

    final_loss = loss_fn(model(x), y).data.item()

    assert final_loss < initial_loss
    assert final_loss < 0.02