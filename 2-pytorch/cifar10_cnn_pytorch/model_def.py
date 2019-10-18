"""
CNN on Cifar10 from Keras example:
https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
- const.yaml, ~50% accuracy after 50 steps
- adaptive.yaml, accuracy comparable to Keras example after 50 steps
"""

from typing import Any, Dict, Sequence, Union

import torch
from torch import nn

from pedl.frameworks.pytorch import PyTorchTrial
from pedl.frameworks.pytorch.util import error_rate

# Constants about the data set.
IMAGE_SIZE = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


class Flatten(nn.Module):
    def forward(self, *args: TorchData, **kwargs: Any) -> torch.Tensor:
        assert len(args) == 1
        x = args[0]
        assert isinstance(x, torch.Tensor)
        return x.view(x.size(0), -1)


class CIFARTrial(PyTorchTrial):
    def build_model(self, hparams: Dict[str, Any]) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, IMAGE_SIZE, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(hparams["layer1_dropout"]),
            nn.Conv2d(32, 64, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(hparams["layer2_dropout"]),
            Flatten(),
            nn.Linear(2304, 512),
            nn.ReLU(),
            nn.Dropout2d(hparams["layer3_dropout"]),
            nn.Linear(512, NUM_CLASSES),
            nn.Softmax(dim=0),
        )

    def losses(self, predictions: TorchData, labels: TorchData) -> torch.Tensor:
        assert isinstance(predictions, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        return torch.nn.functional.nll_loss(predictions, labels)

    def optimizer(self, model: nn.Module, hparams: Dict[str, Any]) -> torch.optim.Optimizer:
        return torch.optim.RMSprop(
            model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["learning_rate_decay"],
            alpha=0.9,
        )

    def validation_metrics(
        self, predictions: TorchData, labels: TorchData, losses: TorchData
    ) -> Dict[str, torch.Tensor]:
        assert isinstance(predictions, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        return {"validation_error": error_rate(predictions, labels)}
