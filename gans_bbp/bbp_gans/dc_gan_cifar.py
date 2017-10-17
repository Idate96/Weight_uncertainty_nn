from gans.utils import data_loader_cifar, sample_noise, plot_batch_images
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

def initialize_weights(m):
    """m is a layer"""
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform(m.weight.data)

class Flatten(nn.Module):
    def forward(self, x):
        #         print("flattening tensor ", x.size())
        N, C, H, W = x.size()  # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


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


def discriminator():
    """
    Build and return a PyTorch model for the DCGAN discriminator implementing
    the architecture above.
    """
    return nn.Sequential(
        # Unflatten(batch_size, 3, 32, 32),
        nn.Conv2d(3, 32, 5, stride=1),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(32, 64, 5, stride=1),
        nn.MaxPool2d(2, stride=2),
        Flatten(),
        nn.Linear(1600, 1600),
        nn.ReLU(),
        nn.Linear(1600, 1)
    )


def generator(noise=noise_dim):
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


def ls_discriminator_loss(scores_real, scores_fake):
    loss = 1 / 2 * torch.mean((scores_real - 1) ** 2 + (scores_fake) ** 2)
    return loss


def ls_generator_loss(scores_fake):
    loss = 1 / 2 * torch.mean((scores_fake - 1) ** 2)
    return loss


def get_optimizer(model):
    """Return optimizer"""
    optimizer = optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))
    return optimizer

def train(discriminator, generator, show_every = 250, num_epochs = 10):
    iter_count = 0
    dis_optimizer = get_optimizer(discriminator)
    gen_optimizer = get_optimizer(generator)
    for epoch in range(num_epochs):
        for x, _ in loader_train:
            if len(x) != batch_size:
                continue
            dis_optimizer.zero_grad()
            real_data = Variable(x).type(torch.FloatTensor)
            logits_real = discriminator(real_data).type(torch.FloatTensor)

            g_fake_seed = Variable(sample_noise(batch_size, noise_dim)).type(torch.FloatTensor)
            fake_images = generator(g_fake_seed)
            logits_fake = discriminator(fake_images.detach())

            d_total_error = ls_discriminator_loss(logits_real, logits_fake)
            d_total_error.backward()
            dis_optimizer.step()

            gen_optimizer.zero_grad()
            g_fake_seed = Variable(sample_noise(batch_size, noise_dim)).type(torch.FloatTensor)
            fake_images = generator(g_fake_seed)

            gen_logits_fake = discriminator(fake_images)
            g_loss = ls_generator_loss(gen_logits_fake)
            g_loss.backward()
            gen_optimizer.step()

            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,d_total_error.data[0],g_loss.data[0]))
                imgs_numpy = fake_images.data.cpu().numpy()
                plot_batch_images(imgs_numpy[0:16], iter_num=iter_count, cifar=True)
                print()
            iter_count += 1

if __name__ == '__main__':
    generator = generator()
    discriminator = discriminator()
    train(discriminator, generator)
