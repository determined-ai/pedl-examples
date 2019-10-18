from typing import Any, Dict, Tuple

import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


def download_data(experiment_config: Dict[str, Any], hparams: Dict[str, Any]) -> str:
    data_dir = "./data"
    _ = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
    _ = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)

    return data_dir


def get_data(train: bool, data_dir: str) -> torchvision.datasets.CIFAR10:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=train, download=False, transform=transform
    )

    return trainset


def make_data_loaders(
    experiment_config: Dict[str, Any], hparams: Dict[str, Any], download_data_dir: str
) -> Tuple[DataLoader, DataLoader]:

    batch_size = hparams["batch_size"]
    return (
        DataLoader(get_data(True, download_data_dir), batch_size=batch_size),
        DataLoader(get_data(False, download_data_dir), batch_size=batch_size),
    )
