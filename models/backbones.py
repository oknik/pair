import torch
import torch.nn as nn
from functools import partial
from transformers import ViTModel

class BackBone(nn.Module):
    def __init__(self, args):
        super(BackBone, self).__init__()
        from .vit import vit_small
        # 因为双模态修改了
        self.encoder = vit_small(in_chans=6)

    def forward(self, support, query):
        # feature extraction
        support = self.encoder(support)
        query = self.encoder(query)
        return support, query

