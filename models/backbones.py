import torch
import torch.nn as nn
from functools import partial
from transformers import ViTModel

class BackBone(nn.Module):
    def __init__(self, args):
        super(BackBone, self).__init__()
        from .vit import vit_small
        self.encoder = vit_small()

    def forward(self, support, query):
        # feature extraction
        support = self.encoder(support)
        query = self.encoder(query)
        return support, query

