import sys
import os
sys.path.append(os.getcwd())
print(sys.path)
from gans.utils import data_loader_cifar, sample_noise, plot_batch_images, show_cifar
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
from torch.autograd import Variable

num_train=50000
num_val=5000
noise_dim=96
batch_size=128
loader_train, loader_val = data_loader_cifar()
dtype = torch.FloatTensor

class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """

    def __init__(self, N=-1, C=128, H=7, W=7):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)


def generator_func(noise=noise_dim):
    return nn.Sequential(
        nn.Linear(noise_dim, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 128*7*7),
        nn.ReLU(),
        nn.BatchNorm1d(128*7*7),
        Unflatten(128, 128, 7, 7),
        nn.ConvTranspose2d(128, 64, 6, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
        nn.Tanh(),
    )

def generator_loss(score_discriminator):
    """Calculate loss for the generator
    The objective is to maximise the error rate of the discriminator
    """
    bce_loss = nn.BCEWithLogitsLoss()
    labels = Variable(torch.ones(score_discriminator.size()))
    loss = bce_loss(score_discriminator, labels)
    return loss