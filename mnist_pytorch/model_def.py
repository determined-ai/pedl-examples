"""
This example shows how to interact with the pedl PyTorch interface to
build a basic MNIST network.

The method `build_model` returns the model to be trained, in this case an
instance of `nn.Sequential`. This model is single-input and single-output. For
an example of a multi-output model, see the `build_model` method in the
definition of `MultiMNistTrial` in model_def_multi_output.py. In that case,
`build_model` returns an instance of a custom `nn.Module`.

Predictions are the output of the `forward` method of the model (for
`nn.Sequential`, that is automatically defined). The predictions are then fed
directly into the `losses` method and the `validation_metrics` method.

The method `MNistTrial.losses` calculates the loss of the training, which for
this model is a single tensor value.

The output of `losses` is then fed directly into `validation_metrics`, which
returns a dictionary mapping metric names to metric values.
"""

from typing import Any, Dict, Sequence, Tuple, Union, cast

import torch
from torch import nn

import pedl
from layers import Flatten  # noqa: I100
from pedl.frameworks.pytorch import PyTorchTrial, reset_parameters
from pedl.frameworks.pytorch.util import error_rate

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


class MNistTrial(PyTorchTrial):
    def build_model(self) -> nn.Module:
        model = nn.Sequential(
            nn.Conv2d(1, pedl.get_hyperparameter("n_filters1"), kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(
                pedl.get_hyperparameter("n_filters1"),
                pedl.get_hyperparameter("n_filters2"),
                kernel_size=5,
            ),
            nn.MaxPool2d(2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(16 * pedl.get_hyperparameter("n_filters2"), 50),
            nn.ReLU(),
            nn.Dropout2d(pedl.get_hyperparameter("dropout")),
            nn.Linear(50, 10),
            nn.LogSoftmax(),
        )

        # If loading backbone weights, do not call reset_parameters() or
        # call before loading the backbone weights.
        reset_parameters(model)
        return model

    def optimizer(self, model: nn.Module) -> torch.optim.Optimizer:  # type: ignore
        return torch.optim.SGD(
            model.parameters(), lr=pedl.get_hyperparameter("learning_rate"), momentum=0.9
        )

    def train_batch(
        self, batch: TorchData, model: nn.Module, epoch_idx: int, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        batch = cast(Tuple[torch.Tensor, torch.Tensor], batch)
        data, labels = batch

        output = model(data)
        loss = torch.nn.functional.nll_loss(output, labels)
        error = error_rate(output, labels)

        return {"loss": loss, "train_error": error}

    def evaluate_batch(self, batch: TorchData, model: nn.Module) -> Dict[str, Any]:
        batch = cast(Tuple[torch.Tensor, torch.Tensor], batch)
        data, labels = batch

        output = model(data)
        error = error_rate(output, labels)

        return {"validation_error": error}
