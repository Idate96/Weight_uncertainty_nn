import sys
import os
sys.path.append(os.getcwd())
from gans.utils import data_loader_cifar, sample_noise, plot_batch_images, show_cifar
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
from torch.autograd import Variable
from gans.bbp_gans.generator_bbp_cifar import Generator
from gans.bbp_gans.discriminator_bbp_cifar import Discriminator

num_train = 50000
num_val = 5000
noise_dim = 96
batch_size = 128
loader_train, loader_val = data_loader_cifar()

# def initialize_weights(m):
#     """m is a layer"""
#     if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
#         init.xavier_uniform(m.weight.data)
#
#
# def train(discriminator, generator, show_every=250, num_epochs=20):
#     iter_count = 0
#     discriminator.add_optimizer()
#     generator.add_optimizer()
#     for epoch in range(num_epochs):
#         for x, _ in loader_train:
#             if len(x) != batch_size:
#                 continue
#             discriminator.optimizer.zero_grad()
#             real_data = Variable(x).type(torch.FloatTensor)
#             logits_real = discriminator.forward(2 * (real_data.view(batch_size, -1) - 0.5)).type(torch.FloatTensor)
#
#             g_fake_seed = Variable(sample_noise(batch_size, noise_dim)).type(torch.FloatTensor)
#             fake_images = generator.forward(g_fake_seed)
#             logits_fake = discriminator.forward(fake_images.detach().view(batch_size, -1))
#
#             d_total_error = discriminator.loss(logits_real, logits_fake)
#             d_total_error.backward()
#             discriminator.optimizer.step()
#
#             generator.optimizer.zero_grad()
#             g_fake_seed = Variable(sample_noise(batch_size, noise_dim)).type(torch.FloatTensor)
#             fake_images = generator.forward(g_fake_seed)
#
#             gen_logits_fake = discriminator.forward(fake_images.view(batch_size, -1))
#             g_loss = generator.loss(gen_logits_fake)
#             g_loss.backward()
#             generator.optimizer.step()
#
#             if (iter_count % show_every == 0):
#                 print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_total_error.data[0],
#                                                            g_loss.data[0]))
#                 imgs = fake_images[:64]
#                 show_cifar(imgs, iter_num=iter_count, save=True, show=True)
#                 print()
#             iter_count += 1


def init_networks(resume=False, label=''):
    if resume:
        generator_n = torch.load('../../results/models/generator' + label + '.pt')
        discriminator_n = torch.load('../../results/models/discriminator' + label + '.pt')
    else:
        generator_n = Generator(10**-3, [2**10, 2**10], label='bbp_02')
        discriminator_n = Discriminator(10**-3, [256, 256], label='bbp_02')
    generator_n.label = label
    discriminator_n.label = label
    return generator_n, discriminator_n

def train(discriminator, generator, show_every=25, num_epochs=20, save_every=2000):
    iter_count = 0
    discriminator.add_optimizer()
    generator.add_optimizer()
    for epoch in range(num_epochs):
        for x, _ in loader_train:
            if len(x) != batch_size:
                continue
            discriminator.optimizer.zero_grad()
            real_data = Variable(x).type(torch.FloatTensor)
            noise = Variable(sample_noise(batch_size, 32**2*3)).type(torch.FloatTensor)
            logits_real = discriminator.forward(real_data.view([-1, 32 ** 2 * 3]) + noise).type(
                torch.FloatTensor)

            g_fake_seed = Variable(sample_noise(batch_size, noise_dim)).type(torch.FloatTensor)
            fake_images = generator.forward(g_fake_seed)
            logits_fake = discriminator.forward(fake_images.view([-1, 32**2*3]) + noise).detach()

            d_total_error = discriminator.loss(logits_real, logits_fake)
            d_total_error.backward()
            discriminator.optimizer.step()

            generator.optimizer.zero_grad()
            g_fake_seed = Variable(sample_noise(batch_size, noise_dim)).type(torch.FloatTensor)
            fake_images = generator.forward(g_fake_seed)

            gen_logits_fake = discriminator.forward(fake_images.view(batch_size, -1))
            g_loss = generator.loss(gen_logits_fake)
            g_loss.backward()
            generator.optimizer.step()

            if iter_count % show_every == 0:
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,d_total_error.data[0],g_loss.data[0]))
                fake_images = fake_images.view([-1, 3, 32, 32])
                imgs = fake_images[:64].data
                show_cifar(imgs, iter_num=iter_count, save=True, show=False, name=generator.label)
            iter_count += 1
            if iter_count % save_every == 0:
                torch.save(discriminator, 'results/weights/discriminator' +
                           discriminator.label + '.pt')
                torch.save(generator, 'results/weights/generator' + generator.label
                           + '.pt')


if __name__ == '__main__':
    generator, discriminator = init_networks(label='vanilla_bbp_cifar_00')
    train(discriminator, generator)

