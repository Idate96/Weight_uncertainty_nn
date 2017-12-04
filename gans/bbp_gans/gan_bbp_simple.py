from gans.utils import data_loader, sample_noise, show_images
import torch
import torch.nn as nn
import time
import torch.optim as optim
from torch.nn import init
from torch.autograd import Variable
from gans.bbp_gans.discriminator_bbp import Discriminator
from gans.bbp_gans.generator_bbp import Generator

num_train=50000
num_val=5000
noise_dim=96
batch_size=128
loader_train, loader_val = data_loader()

def initialize_weights(m):
    """m is a layer"""
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform(m.weight.data)

# def discriminator():
#     """Discrinator nn"""
#     model = nn.Sequential(
#         nn.Linear(28**2, 256),
#         nn.LeakyReLU(0.01),
#         nn.Linear(256, 256),
#         nn.LeakyReLU(0.01),
#         nn.Linear(256, 1)
#     )
#     return model.type(torch.FloatTensor)

def discriminator_loss_std(logits_real, logits_fake):
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

def discriminator_loss_lsq(logits_real, logits_fake):
    loss = 1/2 * torch.mean((logits_real- 1)**2 + (logits_fake)**2)
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

def generator_loss_std(score_discriminator):
    """Calculate loss for the generator
    The objective is to maximise the error rate of the discriminator
    """
    labels = Variable(torch.ones(score_discriminator.size()), requires_grad=False).type(torch.FloatTensor)
    bce_loss = nn.BCEWithLogitsLoss()
    loss = bce_loss(score_discriminator, labels)
    return loss

def generator_loss_lsq(scores_fake):
    return 1/2 * torch.mean((scores_fake - 1)**2)

def get_optimizer(model):
    """Return optimizer"""
    optimizer = optim.Adam(model.parameters(), lr=10**-3, betas=(0.5, 0.999))
    return optimizer

def train(discriminator, generator, discriminator_loss, generator_loss, show_every=250, num_epochs=20):
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
            logits_real = discriminator.forward(2 * (real_data.view(batch_size, -1) - 0.5)).type(torch.FloatTensor)

            g_fake_seed = Variable(sample_noise(batch_size, noise_dim)).type(torch.FloatTensor)
            fake_images = generator.forward(g_fake_seed)
            logits_fake = discriminator.forward(fake_images.detach().view(batch_size, -1))

            d_total_error = discriminator_loss(logits_real, logits_fake)
            d_total_error.backward()
            discriminator.optimizer.step()

            generator.optimizer.zero_grad()
            g_fake_seed = Variable(sample_noise(batch_size, noise_dim)).type(torch.FloatTensor)
            fake_images = generator.forward(g_fake_seed)

            gen_logits_fake = discriminator.forward(fake_images.view(batch_size, -1))
            g_loss = generator_loss(gen_logits_fake)
            g_loss.backward()
            generator.optimizer.step()

            if (iter_count % show_every == 0):
                checkpt_t = time.time()
                print("time : {:.2f} sec" .format(checkpt_t - start_t))
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_total_error.data[0],
                                                           g_loss.data[0]))
                print("real logits average ", torch.mean(logits_real).data)
                print("average output generator : ", torch.mean(fake_images).data)
                print("fake logits average ", torch.mean(gen_logits_fake).data)
                imgs = fake_images[:16].data.numpy()
                show_images(imgs, iter_num=iter_count, save=True, show=False, model=generator.label)
            iter_count += 1

if __name__ == '__main__':
    generator = Generator(10**-3, [1024, 1024])
    discriminator = Discriminator(10**-3, [256, 256])
    train(discriminator, generator, discriminator_loss_lsq, generator_loss_lsq)