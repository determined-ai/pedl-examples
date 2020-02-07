import logging
import os
import shutil
import urllib.parse
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from pedl import get_download_data_dir
from pedl.frameworks.pytorch.data import DataLoader


def get_dataset(data_dir: str, train: bool) -> Any:
    return datasets.MNIST(
        data_dir,
        train=train,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                # These are the precomputed mean and standard deviation of the
                # MNIST data; this normalizes the data to have zero mean and unit
                # standard deviation.
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )


def download_data(experiment_config: Dict[str, Any], hparams: Dict[str, Any]) -> str:
    download_directory = "/tmp/work_dir/MNIST"
    url = experiment_config["data"]["url"]
    url_path = urllib.parse.urlparse(url).path
    basename = url_path.rsplit("/", 1)[1]

    if not os.path.exists(download_directory):
        os.makedirs(download_directory)

    filepath = os.path.join(download_directory, basename)
    if not os.path.exists(filepath):
        logging.info("Downloading {}".format(url))

        r = requests.get(url, stream=True)
        with open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    shutil.unpack_archive(filepath, download_directory)

    return os.path.dirname(download_directory)


def make_data_loaders(
    experiment_config: Dict[str, Any], hparams: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader]:
    download_data_dir = get_download_data_dir()
    train_data = get_dataset(download_data_dir, True)
    validation_data = get_dataset(download_data_dir, False)
    batch_size = hparams["batch_size"]
    return (
        DataLoader(train_data, batch_size=batch_size),
        DataLoader(validation_data, batch_size=batch_size),
    )


# The methods below are used for data loading for the multi-output model
# only (see model_def_multi_output.py).


class MultiMNISTPyTorchDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Tuple:
        data_and_labels = self._dataset[index]
        data = data_and_labels[0]
        digit_label = np.array(data_and_labels[1])
        binary_label = (digit_label >= 5).astype(np.int)
        return data, (digit_label, binary_label)


def get_multi_dataset(data_dir: str, train: bool) -> Dataset:
    dataset = get_dataset(data_dir, train)
    return MultiMNISTPyTorchDataset(dataset)


def collate_fn(batch: List[Tuple]) -> Tuple:
    data = []
    digit_labels = []
    binary_labels = []
    for i in range(len(batch)):
        datum, (digit_label, binary_label) = batch[i]
        data.append(datum)
        digit_labels.append(digit_label)
        binary_labels.append(binary_label)
    data = np.stack(data, 0)
    digit_labels = np.stack(digit_labels, 0)
    binary_labels = np.stack(binary_labels, 0)
    return {"data": data}, {"binary_labels": binary_labels, "digit_labels": digit_labels}


def make_multi_data_loaders(
    experiment_config: Dict[str, Any], hparams: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader]:
    download_data_dir = get_download_data_dir()
    train_data = get_multi_dataset(download_data_dir, True)
    validation_data = get_multi_dataset(download_data_dir, False)
    batch_size = hparams["batch_size"]
    return (
        DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn),
        DataLoader(validation_data, batch_size=batch_size, collate_fn=collate_fn),
    )
