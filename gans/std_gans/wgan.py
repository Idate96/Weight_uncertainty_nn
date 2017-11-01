from gans.utils import data_loader, sample_noise, show_images, show_cifar
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
from torch.autograd import Variable

num_train=50000
num_val=5000
noise_dim=96
batch_size=128
loader_train, loader_val = data_loader()

def initialize_weights(m):
    """m is a layer"""
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform(m.weight.data)

def discriminator():
    """Discrinator nn"""
    model = nn.Sequential(
        nn.Linear(28**2, 256),
        nn.LeakyReLU(0.01),
        nn.Linear(256, 256),
        nn.LeakyReLU(0.01),
        nn.Linear(256, 1)
    )
    return model.type(torch.FloatTensor)

def discriminator_loss(logits_real, logits_fake):
    loss = -torch.mean(logits_real) + torch.mean(logits_fake)
    return loss


def generator(noise_dimension=noise_dim):
    model = nn.Sequential(
        # spherical shape might be better
        nn.Linear(noise_dimension, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 28**2),
        nn.Tanh()
    )
    return model.type(torch.FloatTensor)

def generator_loss(score_discriminator):
    """Calculate loss for the generator
    The objective is to maximise the error rate of the discriminator
    """
    loss = - torch.mean(score_discriminator)
    return loss

def optimizer_gen(model):
    """Return optimizer"""
    optimizer = optim.RMSprop(model.parameters(), lr=5e-5)
    return optimizer

def optimizer_dis(model):
    """Return optimizer"""
    optimizer = optim.RMSprop(model.parameters(), lr=5e-5)
    return optimizer

def init_networks(resume=False, label=''):
    if resume:
        generator_n = torch.load('../../results/models/generator' + label + '.pt')
        discriminator_n = torch.load(
            '../../results/models/discriminator' + label + '.pt')
    else:
        generator_n = generator()
        discriminator_n = discriminator()
    generator_n.label = label
    discriminator_n.label = label
    return generator_n, discriminator_n


def train(discriminator, generator, show_every=250, num_epochs=100, save_every=2000):
    iter_count = 0
    dis_optimizer = optimizer_dis(discriminator)
    gen_optimizer = optimizer_gen(generator)
    for epoch in range(num_epochs):
        for x, _ in loader_train:
            if len(x) != batch_size:
                continue
            dis_optimizer.zero_grad()
            real_data = Variable(x).type(torch.FloatTensor)
            logits_real = discriminator(2 * (real_data.view(batch_size, -1) - 0.5)).type(torch.FloatTensor)

            # g_fake_seed = Variable(sample_noise(batch_size, noise_dim)).type(torch.FloatTensor)
            # fake_images = generator(g_fake_seed)

            for _ in range(5):
                g_fake_seed = Variable(sample_noise(batch_size, noise_dim)).type(torch.FloatTensor)
                fake_images = generator(g_fake_seed)
                logits_fake = discriminator(fake_images.detach().view(batch_size, -1))

                d_total_error = discriminator_loss(logits_real, logits_fake)

                d_total_error.backward(retain_graph=True)
                dis_optimizer.step()

                # weight clipping
                for p in discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)

            gen_optimizer.zero_grad()
            gen_logits_fake = discriminator(fake_images.view(batch_size, -1))
            g_loss = generator_loss(gen_logits_fake)
            g_loss.backward()
            gen_optimizer.step()

            if iter_count % show_every == 0:
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_total_error.data[0],
                                                           g_loss.data[0]))
                imgs_numpy = fake_images.data.cpu().numpy()
                show_images(imgs_numpy[0:16], iter_count, save=True, model=generator.label)

            iter_count += 1
            if iter_count % save_every == 0:
                torch.save(discriminator, 'results/weights/discriminator' +
                           discriminator.label + '.pt')
                torch.save(generator, 'results/weights/generator' + generator.label
                           + '.pt')
            iter_count += 1

if __name__ == '__main__':
    generator, discriminator = init_networks(label='wgan_mnist_00')
    train(discriminator, generator)