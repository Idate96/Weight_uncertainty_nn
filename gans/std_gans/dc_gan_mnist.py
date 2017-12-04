import sys
import os
sys.path.append(os.getcwd())
print(sys.path)
from gans.utils import data_loader, sample_noise, show_images
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
loader_train, loader_val = data_loader()
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


def discriminator_func():
    """
    Build and return a PyTorch model for the DCGAN discriminator implementing
    the architecture above.
    """
    return nn.Sequential(
        Unflatten(batch_size, 1, 28, 28),
        nn.Conv2d(1, 32, 5, stride=1),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(32, 64, 5, stride=1),
        nn.MaxPool2d(2, stride=2),
        Flatten(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1)
    )


def generator_func(noise=noise_dim):
    return nn.Sequential(
        nn.Linear(noise_dim, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 128*7*7),
        nn.ReLU(),
        nn.BatchNorm1d(128*7*7),
        Unflatten(128, 128, 7, 7),
        nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
        nn.Tanh(),
    )

def discriminator_loss(logits_real, logits_fake):
    """Calculate loss for discriminator
    objective : min : loss = - <log(d(x))>  - <log(1 - d(g(z))>
    x coming from data distribution and z from uniform noise distribution
    To do so we will employ the standard binary cross entropy loss :
    bce_loss = y * log(d(x)) + (1-y) * log(d(g(z)))
    where y = 1 for real images and 0 for fake
    :param logits_real: output of discriminator for images coming form the train set
    :param logits_fake: output of discriminator for fake images
    :return: loss
    """
    bce_loss = nn.BCEWithLogitsLoss()
    labels_real = Variable(torch.ones(logits_real.size()), requires_grad=False).type(torch.FloatTensor)
    labels_fake = Variable(torch.zeros(logits_fake.size()), requires_grad=False).type(torch.FloatTensor)
    loss = bce_loss(logits_real, labels_real) + bce_loss(logits_fake, labels_fake)
    return loss

def generator_loss(score_discriminator):
    """Calculate loss for the generator
    The objective is to maximise the error rate of the discriminator
    """
    bce_loss = nn.BCEWithLogitsLoss()
    labels = Variable(torch.ones(score_discriminator.size()))
    loss = bce_loss(score_discriminator, labels)
    return loss

def ls_discriminator_loss(scores_real, scores_fake):
    loss = 1 / 2 * torch.mean((scores_real - 1) ** 2 + (scores_fake) ** 2)
    return loss


def ls_generator_loss(scores_fake):
    loss = 1 / 2 * torch.mean((scores_fake - 1) ** 2)
    return loss


def optimizer_generator(model):
    """Return optimizer"""
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))
    return optimizer

def optimizer_discriminator(model):
    """Return optimizer"""
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))
    return optimizer

def init_networks(resume=False, label=''):
    if resume:
        generator_n = torch.load('../../results/models/generator' + label + '.pt')
        discriminator_n = torch.load(
            '../../results/models/discriminator' + label + '.pt')
    else:
        generator_n = generator_func()
        generator_n.apply(initialize_weights)
        discriminator_n = discriminator_func()
        discriminator_n.apply(initialize_weights)
    generator_n.label = label
    print(generator_n.label)
    discriminator_n.label = label
    return generator_n, discriminator_n

def train(discriminator, generator, show_every= 25, num_epochs= 10, resume=False,
          save_every = 2000):
    iter_count = 0
    dis_optimizer = optimizer_discriminator(discriminator)
    gen_optimizer = optimizer_generator(generator)
    for epoch in range(num_epochs):
        for x, _ in loader_train:
            if len(x) != batch_size:
                continue
            if iter_count == 0:
                weights_average = {}
                weights_average_gen = {}
                for name, value in discriminator.state_dict().items():
                    weights_average[name] = torch.mean(value)
                for name, value in generator.state_dict().items():
                    weights_average_gen[name] = torch.mean(value)
                print("Average value of initialized weights dis : \n", weights_average)
                print("Average value of initialized weights gen : \n", weights_average_gen)

            dis_optimizer.zero_grad()
            real_data = Variable(x).type(torch.FloatTensor)
            logits_real = discriminator(2 * (real_data.view(batch_size, -1) - 0.5)).type(
                torch.FloatTensor)

            g_fake_seed = Variable(sample_noise(batch_size, noise_dim)).type(torch.FloatTensor)
            fake_images = generator(g_fake_seed).detach()
            logits_fake = discriminator(fake_images.view(batch_size, -1))

            d_total_error = discriminator_loss(logits_real, logits_fake)
            d_total_error.backward()
            dis_optimizer.step()

            gen_optimizer.zero_grad()
            g_fake_seed = Variable(sample_noise(batch_size, noise_dim)).type(torch.FloatTensor)
            fake_images = generator(g_fake_seed)

            gen_logits_fake = discriminator(fake_images.view(batch_size, -1))
            g_loss = generator_loss(gen_logits_fake)
            g_loss.backward()
            gen_optimizer.step()

            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_total_error.data[0],
                                                           g_loss.data[0]))
                imgs = fake_images[:16].data.numpy()
                show_images(imgs, iter_num=iter_count, save=True, show=False, model=generator.label)
            iter_count += 1
            if iter_count % save_every == 0:
                torch.save(discriminator, 'results/weights/discriminator' +
                           discriminator.label + '.pt')
                torch.save(generator, 'results/weights/generator' + generator.label
                           + '.pt')


if __name__ == '__main__':
    generator, discriminator = init_networks(label='dc_mnist_01')
    train(discriminator, generator)
