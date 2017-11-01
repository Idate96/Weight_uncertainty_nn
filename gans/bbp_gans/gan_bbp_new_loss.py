from utils import data_loader, sample_noise, plot_batch_images
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
from torch.autograd import Variable
from generator_bbp import Generator
from discriminator_bbp import Discriminator

num_train = 50000
num_val = 5000
noise_dim = 96
batch_size = 128
loader_train, loader_val = data_loader()

def initialize_weights(m):
    """m is a layer"""
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform(m.weight.data)


def train(discriminator, generator, show_every=250, num_epochs=20):
    iter_count = 0
    discriminator.add_optimizer()
    generator.add_optimizer()
    for epoch in range(num_epochs):
        for x, _ in loader_train:
            if len(x) != batch_size:
                continue
            discriminator.optimizer.zero_grad()
            real_data = Variable(x).type(torch.FloatTensor)
            logits_real = discriminator.forward(2 * (real_data.view(batch_size, -1) - 0.5)).type(torch.FloatTensor)

            g_fake_seed = Variable(sample_noise(batch_size, noise_dim)).type(torch.FloatTensor)
            fake_images = generator.forward(g_fake_seed)
            logits_fake = discriminator.forward(fake_images.detach().view(batch_size, -1))

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

            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,d_total_error.data[0],g_loss.data[0]))
                imgs_numpy = fake_images.data.cpu().numpy()
                plot_batch_images(imgs_numpy[0:16], iter_num=iter_count)
                print()
            iter_count += 1

if __name__ == '__main__':
    generator = Generator(10**-3, [1024, 1024])
    discriminator = Discriminator(10**-3, [256, 256])
    train(discriminator, generator)