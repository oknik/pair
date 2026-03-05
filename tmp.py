import torch

weights = torch.load("pytorch_model.bin")
torch.save({'teacher': weights}, "pretrained.pth")