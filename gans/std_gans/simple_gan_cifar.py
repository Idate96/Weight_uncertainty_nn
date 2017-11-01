import sys
import os
sys.path.append(os.getcwd())
from gans.utils import data_loader_cifar, sample_noise, plot_batch_images, show_cifar
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
from torch.autograd import Variable

num_train = 45000
num_val = 5000
noise_dim = 96
batch_size = 128
loader_train, loader_val = data_loader_cifar()

def initialize_weights(m):
    """m is a layer"""
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform(m.weight.data)

def discriminator_func():
    """Discrinator nn"""
    model = nn.Sequential(
        nn.Linear(32**2*3, 2**11),
        nn.LeakyReLU(0.01),
        # nn.Dropout(),
        nn.Linear(2**11, 2**11),
        nn.LeakyReLU(0.01),
        # nn.Dropout(),
        nn.Linear(2**11, 1)
    )
    return model.type(torch.FloatTensor)

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


def generator_func(noise_dimension=noise_dim):
    model = nn.Sequential(
        # spherical shape might be better
        nn.Linear(noise_dimension, 2**7),
        nn.ReLU(),
        nn.Linear(2**7, 2**7),
        nn.ReLU(),
        nn.Linear(2**7, 32**2*3),
        nn.Tanh()
    )
    return model.type(torch.FloatTensor)

def generator_loss(score_discriminator):
    """Calculate loss for the generator
    The objective is to maximise the error rate of the discriminator
    """
    bce_loss = nn.BCEWithLogitsLoss()
    labels = Variable(torch.ones(score_discriminator.size()))
    loss = bce_loss(score_discriminator, labels)
    return loss

def get_optimizer(model):
    """Return optimizer"""
    optimizer = optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))
    return optimizer


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    """
    loss = 1 / 2 * torch.mean((scores_real - 1) ** 2 + scores_fake ** 2)
    return loss


def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.
    """
    loss = 1 / 2 * torch.mean((scores_fake - 1) ** 2)
    return loss

def init_networks(resume=False, label=''):
    if resume:
        generator_n = torch.load('../../results/models/generator' + label + '.pt')
        discriminator_n = torch.load(
            '../../results/models/discriminator' + label + '.pt')
    else:
        generator_n = generator_func()
        discriminator_n = discriminator_func()
    generator_n.label = label
    print(generator_n.label)
    discriminator_n.label = label
    return generator_n, discriminator_n

def train(discriminator, generator, show_every= 25, num_epochs= 10, resume=False,
          save_every = 1500):
    iter_count = 0
    dis_optimizer = get_optimizer(discriminator)
    gen_optimizer = get_optimizer(generator)
    for epoch in range(num_epochs):
        for x, _ in loader_train:
            if len(x) != batch_size:
                continue
            dis_optimizer.zero_grad()
            real_data = Variable(x).type(torch.FloatTensor)
            # noise = Variable(sample_noise(batch_size, 32**2*3)).type(torch.FloatTensor)
            logits_real = discriminator(real_data.view([-1, 32**2*3])).type(
                torch.FloatTensor)

            g_fake_seed = Variable(sample_noise(batch_size, noise_dim)).type(torch.FloatTensor)
            fake_images = generator(g_fake_seed)
            logits_fake = discriminator(fake_images.view([-1, 32**2*3])).detach()

            d_total_error = discriminator_loss(logits_real, logits_fake)
            d_total_error.backward()
            dis_optimizer.step()

            gen_optimizer.zero_grad()
            g_fake_seed = Variable(sample_noise(batch_size, noise_dim)).type(torch.FloatTensor)
            fake_images = generator(g_fake_seed)

            gen_logits_fake = discriminator(fake_images)
            g_loss = generator_loss(gen_logits_fake)
            g_loss.backward()
            gen_optimizer.step()

            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,d_total_error.data[0],g_loss.data[0]))
                fake_images = fake_images.view([-1, 3, 32, 32])
                imgs = fake_images[:64].data
                show_cifar(imgs, iter_num=iter_count, save=True, show=False, name=generator.label)
            iter_count += 1
            if iter_count % save_every == 0:
                torch.save(discriminator, '../../results/weights/discriminator' +
                           discriminator.label + '.pt')
                torch.save(generator, '../../results/weights/generator' + generator.label
                           + '.pt')

if __name__ == '__main__':
    generator, discriminator = init_networks(label= 'vanilla_gan_cifar_ls_00')
    train(discriminator, generator)