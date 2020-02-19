"""
This file loads the training and validation data for model_def
"""
import logging
import os
import shutil
import tarfile
from typing import Any, Dict

import numpy as np
import randomNAS_files.data_util as data
import torch
import wget

from torch.utils.data import Dataset

import pedl
from pedl.frameworks.pytorch.data import DataLoader


class PadSequence:
    def __call__(self, batch):
        features = batch[:-1]
        labels = batch[1:]

        features = torch.stack(features)
        labels = torch.stack(labels).contiguous().view(-1)

        return features, labels


class BatchSamp:
    def __init__(self, dataset, valid=False):
        self.valid = valid
        self.data_length = len(dataset) - 1 - 1

    def _calculate_seq_len(self, i):
        bptt = (
            pedl.get_hyperparameter("bptt")
            if np.random.random() < 0.95
            else pedl.get_hyperparameter("bptt") / 2.0
        )
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        seq_len = min(
            seq_len,
            pedl.get_hyperparameter("bptt") + pedl.get_hyperparameter("max_seq_length_delta"),
        )
        seq_len = min(
            pedl.get_hyperparameter("bptt") if self.valid else seq_len, self.data_length - 1 - i
        )
        return seq_len

    def __len__(self):
        return self.data_length

    def __iter__(self):
        seq_len = 0 if not self.valid else pedl.get_hyperparameter("bptt")
        i = 0
        while i < self.data_length:
            seq_len = self._calculate_seq_len(i)
            start = i
            end = i + seq_len
            # sometimes the seq_len is 0
            # this means we have reached the end of the data
            if seq_len == 0:
                break
            yield list(range(start, end + 1))
            i += seq_len


class PTBData(Dataset):
    def __init__(self, data, seq_len, batch_size, valid=False):
        self.batch_size = batch_size
        self.data = self.batchify(data)
        self.max_seq_len = pedl.get_hyperparameter("bptt") + pedl.get_hyperparameter(
            "max_seq_length_delta"
        )
        self.valid = valid

    def batchify(self, data):
        nbatch = data.size(0) // self.batch_size
        data = data.narrow(0, 0, nbatch * self.batch_size)
        data = data.view(self.batch_size, -1).t().contiguous()  # returns [29049, 32]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def download_data(experiment_config: Dict[str, Any], hparams: Dict[str, Any]) -> str:
    """
    Downloads the data if the data does not exist in the provided data_loc
    Returns: string of path to the data.
    """
    data_loc = pedl.get_data_config().get("data_loc")

    if os.path.exists(data_loc + "train.txt") and os.path.exists(data_loc + "valid.txt"):
        # Exit if the data already exists
        return data_loc

    if not os.path.isdir(data_loc):
        os.makedirs(data_loc)
    logging.info("downloading and extracting %s...")

    url = "http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz"
    data_file = "simple-examples.tgz"

    wget.download(url, data_loc + data_file)

    tf = tarfile.open(data_loc + data_file)
    tf.extractall()
    tf.close()

    shutil.move("simple-examples/data/ptb.train.txt", data_loc + "train.txt")
    shutil.move("simple-examples/data/ptb.valid.txt", data_loc + "valid.txt")
    shutil.move("simple-examples/data/ptb.test.txt", data_loc + "test.txt")

    logging.info("\tcompleted")
    return data_loc


def make_data_loaders(experiment_config: Dict[str, Any], hparams: Dict[str, Any]):
    """
    Required method to load in the datasets
    returns: PEDL DataLoader
    """
    corpus = data.Corpus(pedl.get_data_config().get("data_loc"))

    train_dataset = PTBData(
        corpus.train, pedl.get_hyperparameter("seq_len"), pedl.get_hyperparameter("batch_size")
    )
    test_dataset = PTBData(
        corpus.valid, pedl.get_hyperparameter("seq_len"), pedl.get_hyperparameter("eval_batch_size")
    )

    return (
        DataLoader(train_dataset, batch_sampler=BatchSamp(train_dataset), collate_fn=PadSequence()),
        DataLoader(
            test_dataset,
            batch_sampler=BatchSamp(test_dataset, valid=True),
            collate_fn=PadSequence(),
        ),
    )
