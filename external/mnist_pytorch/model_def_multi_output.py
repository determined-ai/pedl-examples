"""
This example shows how to interact with the PEDL PyTorch interface to build a
multi prediction MNIST network (with both binary and digit labels).  For an
implementation of the standard MNIST digit prediction (predicting only digit
labels), see model_def.py in the same folder.

Predictions are calculated in the `forward` method of the `MultiNet` class.
The `MultiMNistTrial` class contains methods for calculating the losses,
training metrics, and validation metrics.
"""

from typing import Any, cast, Dict, Tuple

import torch
from torch import nn

import pedl
from pedl.frameworks.pytorch import PyTorchTrial, reset_parameters
from pedl.frameworks.pytorch.data import TorchData
from pedl.frameworks.pytorch.util import error_rate

from layers import Flatten, Squeeze  # noqa: I100


class MultiNet(nn.Module):
    """
    MNIST network that takes
    input: data
    output: digit predictions, binary predictions
    """

    def __init__(self) -> None:
        super().__init__()
        # Set hyperparameters that influence the model architecture.
        self.n_filters1 = pedl.get_hyperparameter("n_filters1")
        self.n_filters2 = pedl.get_hyperparameter("n_filters2")
        self.dropout = pedl.get_hyperparameter("dropout")

        # Define the central model.
        self.model = nn.Sequential(
            nn.Conv2d(1, self.n_filters1, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(self.n_filters1, self.n_filters2, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(16 * self.n_filters2, 50),
            nn.ReLU(),
            nn.Dropout2d(self.dropout),
        )  # type: nn.Sequential
        # Predict digit labels from self.model.
        self.digit = nn.Sequential(nn.Linear(50, 10), nn.Softmax(dim=0))
        # Predict binary labels from self.model.
        self.binary = nn.Sequential(nn.Linear(50, 1), nn.Sigmoid(), Squeeze())

    def forward(self, *args: TorchData, **kwargs: Any) -> TorchData:
        assert len(args) == 1
        assert isinstance(args[0], dict)
        # The name "data" is defined by the return value of the
        # `MultiMNistPyTorchDatasetAdapter.get_batch()` method.
        model_out = self.model(args[0]["data"])

        # Define two prediction outputs for a multi-output network.
        digit_predictions = self.digit(model_out)
        binary_predictions = self.binary(model_out)

        # Return the two outputs as a dict of outputs. This dict will become
        # the `predictions` input to the `MultiMNistTrial.losses()` function.
        return {"digit_predictions": digit_predictions, "binary_predictions": binary_predictions}


def compute_loss(predictions: TorchData, labels: TorchData) -> torch.Tensor:
    assert isinstance(predictions, dict)
    assert isinstance(labels, dict)

    labels["binary_labels"] = labels["binary_labels"].type(torch.float32)  # type: ignore

    # Calculate loss functions.
    loss_digit = torch.nn.functional.nll_loss(
        predictions["digit_predictions"], labels["digit_labels"]
    )
    loss_binary = torch.nn.functional.binary_cross_entropy(
        predictions["binary_predictions"], labels["binary_labels"]
    )

    # Rudimentary example of how loss functions may be combined for
    # multi-output training.
    loss = loss_binary + loss_digit
    return loss


class MultiMNistTrial(PyTorchTrial):
    def build_model(self) -> nn.Module:
        model = MultiNet()

        # If loading backbone weights, do not call reset_parameters() or
        # call before loading the backbone weights.
        reset_parameters(model)
        return model

    def optimizer(self, model: nn.Module) -> torch.optim.Optimizer:  # type: ignore
        return torch.optim.SGD(
            model.parameters(), lr=pedl.get_hyperparameter("learning_rate"), momentum=0.9
        )

    def train_batch(
        self, batch: Any, model: nn.Module, epoch_idx: int, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        batch = cast(Tuple[TorchData, Dict[str, torch.Tensor]], batch)
        data, labels = batch

        output = model(data)
        loss = compute_loss(output, labels)
        error = error_rate(output["digit_predictions"], labels["digit_labels"])

        return {"loss": loss, "classification_error": error}

    def evaluate_batch(self, batch: Any, model: nn.Module) -> Dict[str, Any]:
        batch = cast(Tuple[TorchData, Dict[str, torch.Tensor]], batch)
        data, labels = batch

        output = model(data)
        error = error_rate(output["digit_predictions"], labels["digit_labels"])

        return {"validation_error": error}
