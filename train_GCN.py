import abc
from abc import ABC
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gnet import *
from calibrator.calibrator import * 
from config import parse_args
from CIFAR10 import train_loader,test_loader, val_loader


class NodewiseNLL(nn.Module):
    # edge_index - shape: (2, E), dtype: long
    def __init__(self, node_index):
        super().__init__()
        self.node_index = node_index

    def forward(self, logits, gts):
        raise NotImplementedError
    
    def forward(self, logits, labels):
        nodelogits = logits[self.node_index]
        nodelabels = labels[self.node_index]
        return F.cross_entropy(nodelogits, nodelabels)

class Metrics(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def acc(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def nll(self) -> float:
        raise NotImplementedError
    
class NodewiseMetrics(Metrics):
    def __init__(
            self, logits, labels, index,
            bins: int = 15, scheme: str = 'equal_width', norm=1):
        self.node_index = index
        self.logits = logits
        self.labels = labels
        self.nll_fn = NodewiseNLL(index)

    def acc(self) -> float:
        preds = torch.argmax(self.logits, dim=1)[self.node_index]
        return torch.mean(
            (preds == self.gts[self.node_index]).to(torch.get_default_dtype())
        ).item()

    def nll(self) -> float:
        return self.nll_fn(self.logits, self.labels).item()


def main(args):
    # Evaluation
    val_result = {'acc': [], 'nll': []}
    test_result = {'acc': [], 'nll': []}

    parser = argparse.ArgumentParser(description = "PyTorch Graph CNN Training")
    args = parser.parse_args()

    # Early stopping
    patience = 100
    vlss_mn = float('inf')
    vacc_mx = 0.0
    state_dict_early_model = None
    curr_step = 0
    best_result = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.005)
            
    criterion = torch.nn.CrossEntropyLoss()




