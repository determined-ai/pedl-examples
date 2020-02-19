"""
This example is to shows a possible way to create a random NAS search using PEDL.
The flags and configurations can be found under const.yaml and random.yaml.

Using PEDL, we are able to skip step 1 because we can search through using pedl capabilities

This implementation is based on:
https://github.com/liamcli/randomNAS_release/tree/6513a0a6a781ed1f0009ccd9bae622ae7f0a961d

Paper for reference: https://arxiv.org/pdf/1902.07638.pdf

"""
import logging
import math
import os
import pickle as pkl
from typing import Dict, Sequence, Union

import numpy as np
import randomNAS_files.genotypes as genotypes
import torch
from randomNAS_files.model import RNNModel
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler

import pedl
from pedl.frameworks.pytorch import LRScheduler
from pedl.frameworks.pytorch import PyTorchTrial

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]
PTB_NUMBER_TOKENS = 10000


class MyLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        """
        Custom LR scheudler for the LR to be adjusted based on the batch size
        """
        self.seq_len = pedl.get_hyperparameter("bptt")
        self.start_lr = pedl.get_hyperparameter("learning_rate")
        super(MyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        ret = list(self.base_lrs)
        self.base_lrs = [
            self.start_lr * self.seq_len / pedl.get_hyperparameter("bptt")
            for base_lr in self.base_lrs
        ]
        return ret

    def set_seq_len(self, seq_len):
        self.seq_len = seq_len


class NASModel(PyTorchTrial):
    def optimizer(self, model: nn.Module):
        """
        Required Method. Sets the optimizer to use
        Returns: optimizer
        """
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=pedl.get_hyperparameter("learning_rate"),
            weight_decay=pedl.get_hyperparameter("wdecay"),
        )
        return optimizer

    def create_lr_scheduler(self, optimizer: torch.optim.Optimizer):
        """
        Required Method to use a learning rate scheduler
        Returns: PEDL scheduler object
        PEDL will handle the learning rate scheduler update based on the PEDL LRScheduler parameters
        If step_every_batch or step_every_epoch is True, PEDL will handle the .step().
        If both are false, the user will be in charge of calling .step().
        """
        self.myLR = MyLR(optimizer)
        step_every_batch = pedl.get_hyperparameter("step_every_batch")
        step_every_epoch = pedl.get_hyperparameter("step_every_epoch")
        return LRScheduler(
            self.myLR, step_every_batch=step_every_batch, step_every_epoch=step_every_epoch
        )

    def sample_arch(self):
        """
        Required: Method to build the Optimizer
        Returns: PyTorch Optimizer
        """
        n_nodes = genotypes.STEPS
        n_ops = len(genotypes.PRIMITIVES)
        arch = []
        for i in range(n_nodes):
            op = np.random.choice(range(1, n_ops))
            node_in = np.random.choice(range(i + 1))
            arch.append((genotypes.PRIMITIVES[op], node_in))
        concat = range(1, 9)
        genotype = genotypes.Genotype(recurrent=arch, concat=concat)
        return genotype

    def build_model(self) -> nn.Module:
        """
        Required Method that builds the model
        Returns: PyTorch Model
        """
        arch_to_use = pedl.get_hyperparameter("arch_to_use")

        if hasattr(genotypes, arch_to_use):
            self.arch = getattr(genotypes, arch_to_use)
            logging.info("using genotype.{0}".format(self.arch))
        else:
            self.arch = self.sample_arch()
            logging.info("using random arch.{0}".format(self.arch))

        model = RNNModel(
            PTB_NUMBER_TOKENS,
            pedl.get_hyperparameter("emsize"),
            pedl.get_hyperparameter("nhid"),
            pedl.get_hyperparameter("nhidlast"),
            pedl.get_hyperparameter("dropout"),
            pedl.get_hyperparameter("dropouth"),
            pedl.get_hyperparameter("dropoutx"),
            pedl.get_hyperparameter("dropouti"),
            pedl.get_hyperparameter("dropoute"),
            genotype=self.arch,
        )

        # Made for stacking multiple cells, by default the depth is set to 1
        # which will not run this for loop
        for _ in range(
            pedl.get_hyperparameter("depth") - 1
        ):  # minus 1 because 1 gets auto added by the main model
            new_cell = model.cell_cls(
                pedl.get_hyperparameter("emsize"),
                pedl.get_hyperparameter("nhid"),
                pedl.get_hyperparameter("dropouth"),
                pedl.get_hyperparameter("dropoutx"),
                self.arch,
                pedl.get_hyperparameter("init_op"),
            )
            model.rnns.append(new_cell)

        model.batch_size = pedl.get_hyperparameter("batch_size")

        return model

    def update_and_step_lr(self, seq_len):
        """
        Updates and steps the learning rate
        """
        self.myLR.set_seq_len(seq_len)
        self.myLR.step()

    def train_batch(self, batch: TorchData, model: nn.Module, epoch_idx: int, batch_idx: int):
        """
        Trains the provided batch.
        Returns: Dictionary of the calculated Metrics
        """

        features, labels = batch
        self.update_and_step_lr(features.shape[0])

        # set hidden if it's the first run
        if batch_idx == 0:
            self.hidden = model.init_hidden(pedl.get_hyperparameter("batch_size"))

        # detach to prevent backpropagating to far
        for i in range(len(self.hidden)):
            self.hidden[i] = self.hidden[i].detach()

        log_prob, self.hidden, rnn_hs, dropped_rnn_hs = model(features, self.hidden, return_h=True)

        loss = nn.functional.nll_loss(
            log_prob.view(-1, log_prob.size(2)), labels.contiguous().view(-1)
        )
        if pedl.get_hyperparameter("alpha") > 0:
            loss = loss + sum(
                pedl.get_hyperparameter("alpha") * dropped_rnn_h.pow(2).mean()
                for dropped_rnn_h in dropped_rnn_hs[-1:]
            )

        loss = (
            loss
            + sum(
                pedl.get_hyperparameter("beta") * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean()
                for rnn_h in rnn_hs[-1:]
            )
        ) * 1.0

        try:
            perplexity = math.exp(loss / len(features))
        except Exception as e:
            logging.error("Calculating perplexity failed with error: %s", e)
            perplexity = 100000

        if math.isnan(perplexity):
            perplexity = 100000

        return {"loss": loss, "perplexity": perplexity}

    def evaluate_full_dataset(self, data_loader: torch.utils.data.DataLoader, model: nn.Module):
        """
        Determines if multiple architectures should be evaluated and sends to approprate path
        Returns: the results of the evaluated dataset or the best result from multiple evaluations
        """
        eval_same_arch = pedl.get_hyperparameter("eval_same_arch")

        if eval_same_arch:  # evaluate the same architecture
            res = self.evaluate_dataset(data_loader, model, self.arch)
        else:
            res = self.evaluate_multiple_archs(data_loader, model)

        return res

    def evaluate_multiple_archs(self, data_loader, model):
        """
        Helper that randomly selects architectures and evaluates their performance
        This function is only called if eval_same_arch is False and should not be used for
        the primary NAS search
        """
        num_archs_to_eval = pedl.get_hyperparameter("num_archs_to_eval")

        sample_vals = []
        for _ in range(num_archs_to_eval):
            arch = self.sample_arch()

            res = self.evaluate_dataset(data_loader, model, arch)
            perplexity = res["perplexity"]
            loss = res["loss"]

            sample_vals.append((arch, perplexity, loss))

        sample_vals = sorted(sample_vals, key=lambda x: x[1])

        logging.info("best arch found: ", sample_vals[0])
        self.save_archs(sample_vals)

        return {"loss": sample_vals[0][2], "perplexity": sample_vals[0][1]}

    def evaluate_dataset(self, data_loader, model, arch, split=None):
        """
        Evaluates the full dataset against the given arch
        """
        hidden = model.init_hidden(pedl.get_hyperparameter("eval_batch_size"))

        model = self.set_model_arch(arch, model)

        total_loss = 0
        num_samples_seen = 0
        for i, batch in enumerate(data_loader):
            features, targets = batch
            features, targets = features.cuda(), targets.cuda()

            log_prob, hidden = model(features, hidden)
            loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets).data
            total_loss += loss * len(features)

            for i in range(len(hidden)):
                hidden[i] = hidden[i].detach()
            num_samples_seen += features.shape[0]

        try:
            perplexity = math.exp(total_loss.item() / num_samples_seen)
        except Exception as e:
            logging.error("Calculating perplexity failed with error: %s", e)
            perplexity = 100000

        if math.isnan(perplexity):
            perplexity = 100000

        if math.isnan(loss):
            loss = 100000

        return {"loss": total_loss, "perplexity": perplexity}

    def save_archs(self, data):
        out_file = pedl.get_data_config().get("out_file") + pedl.get_hyperparameter("seed")

        with open(os.path.join(out_file), "wb+") as f:
            pkl.dump(data, f)

    def set_model_arch(self, arch, model):
        for rnn in model.rnns:
            rnn.genotype = arch
        return model
