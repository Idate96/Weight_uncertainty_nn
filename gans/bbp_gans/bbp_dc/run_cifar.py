"""
Contatains DC nn with bayes by backprop.
Time per batch of 128 is 10 secs
To see results I need 8 epochs each epoch is 390 batches.
This means I need : 31200 secs => 9 hours at full speed
"""

from utils import data_loader, sample_noise, plot_batch_images
import torch
import time
from gans.utils import data_loader_cifar, sample_noise, plot_batch_images, show_cifar
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
from gans.bbp_gans.bbp_dc.generator import Generator
from gans.bbp_gans.bbp_dc.discriminator import Discriminator
from torch.autograd import Variable


num_train = 50000
num_val = 5000
noise_dim = 96
batch_size = 128
loader_train, loader_val = data_loader_cifar()
dtype = torch.FloatTensor


def train(discriminator, generator, show_every=25, num_epochs=20, save_every = 2000):
    start_t = time.time()
    iter_count = 0
    discriminator.add_optimizer()
    generator.add_optimizer()
    for epoch in range(num_epochs):
        for x, _ in loader_train:
            if len(x) != batch_size:
                continue
            discriminator.optimizer.zero_grad()
            real_data = Variable(x).type(torch.FloatTensor)
            logits_real = discriminator.forward(real_data).type(torch.FloatTensor)

            g_fake_seed = Variable(sample_noise(batch_size, noise_dim)).type(torch.FloatTensor)
            fake_images = generator.forward(g_fake_seed)
            logits_fake = discriminator.forward(fake_images.detach())

            d_total_error = discriminator.loss(logits_real, logits_fake)
            d_total_error.backward()
            discriminator.optimizer.step()

            generator.optimizer.zero_grad()
            g_fake_seed = Variable(sample_noise(batch_size, noise_dim)).type(torch.FloatTensor)
            fake_images = generator.forward(g_fake_seed)

            gen_logits_fake = discriminator.forward(fake_images)
            g_loss = generator.loss(gen_logits_fake)
            g_loss.backward()
            generator.optimizer.step()

            if (iter_count % show_every == 0):
                checkpt_t = time.time()
                print("time : {:.2f} sec" .format(checkpt_t - start_t))
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_total_error.data[0],
                                                           g_loss.data[0]))
                print("real logits average ", torch.mean(logits_real).data)
                print("average output generator : ", torch.mean(fake_images).data)
                imgs = fake_images[:64].data
                show_cifar(imgs, iter_num=iter_count, save=True, show=False, name=generator.label)
            iter_count += 1
            if iter_count % save_every == 0:
                torch.save(discriminator, 'results/weights/discriminator' +
                           discriminator.label + '.pt')
                torch.save(generator, 'results/weights/generator' + generator.label
                           + '.pt')

if __name__ == '__main__':
    generator = Generator(10**-3, [1024, 1024], label='bbp_dc_cifar_00')
    discriminator = Discriminator(10**-3, [1600, 1600], label='bbp_dc_cifar_00')
    train(discriminator, generator)