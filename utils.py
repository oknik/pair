import os
import pprint
import torch
import numpy as np


def ensure_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

class CircleNum():
    def __init__(self, id, upedge):
        self.upedge = upedge
        self.id = id
    def add(self):
        if self.id + 1 >= self.upedge:
            self.id = 0
        else:
            self.id += 1
    def val(self):
        return self.id

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

def count_acc_cosine(logits, label, cosine_weight):
    score = logits
    threshold = 0.5
    pred = (score > threshold).float()
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

def count_acc(logits, label):
    logits = torch.sigmoid(logits)
    threshold = 0.5
    pred = (logits > threshold).float()
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm
