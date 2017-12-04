"""
Contatains DC nn with bayes by backprop.
Time per batch of 128 is 10 secs
To see results I need 8 epochs each epoch is 390 batches.
This means I need : 31200 secs => 9 hours at full speed
"""
import numpy as np
import torch
import time
from gans.utils import data_loader_cifar, sample_noise, plot_batch_images, show_cifar
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
from gans.mixed_bbp_gen_dc_deltaq.generator import Generator
from gans.mixed_bbp_gen_dc_deltaq.discriminator import discriminator_func, discriminator_loss, \
    optimizer_discriminator
from torch.autograd import Variable


num_train = 50000
num_val = 5000
noise_dim = 96
batch_size = 128
loader_train, loader_val = data_loader_cifar()
dtype = torch.FloatTensor


def train(discriminator, generator, show_every=100, num_epochs=20, save_every=2000):
    start_t = time.time()
    iter_count = 0
    optimizer_dis = optimizer_discriminator(discriminator)
    generator.add_optimizer()
    for epoch in range(num_epochs):
        for x, _ in loader_train:
            if len(x) != batch_size:
                continue
            # weights_init = discriminator.state_dict()["7.weight"][:]
            if iter_count == 0:
                weights_average = {}
                weights_average_gen = {}
                for name, value in discriminator.state_dict().items():
                    weights_average[name] = torch.mean(value)
                for name, value in generator.state_dict().items():
                    weights_average_gen[name] = torch.mean(value)
                print("Average value of initialized weights dis : \n", weights_average)
                print("Average value of initialized weights gen : \n", weights_average_gen)

            optimizer_dis.zero_grad()
            real_data = Variable(x).type(torch.FloatTensor)
            logits_real = discriminator(real_data).type(torch.FloatTensor)

            g_fake_seed = Variable(sample_noise(batch_size, noise_dim)).type(torch.FloatTensor)
            fake_images = generator.forward(g_fake_seed)
            logits_fake = discriminator(fake_images.detach())

            # analysed = batch_size
            # fooled = np.sum(logits_fake.data.numpy() > 0.5)
            # print("average logits real ", torch.mean(logits_real))
            print("average logits fake ", torch.mean(logits_fake))
            # print("fooled : ", fooled)
            # print("analysed ", analysed)
            # print("guess ratio {0:.4f}" .format(fooled/analysed))
            # if fooled/analysed > 0.5 or iter_count == 0:
            d_total_error = discriminator_loss(logits_real, logits_fake)
            d_total_error.backward()
            optimizer_dis.step()

            generator.optimizer.zero_grad()
            g_fake_seed = Variable(sample_noise(batch_size, noise_dim)).type(torch.FloatTensor)
            fake_images = generator.forward(g_fake_seed)
            print("average output generator :", torch.mean(fake_images).data.numpy())

            gen_logits_fake = discriminator(fake_images)
            g_loss = generator.loss(gen_logits_fake)
            g_loss.backward()
            generator.optimizer.step()


            # weights_after = discriminator.state_dict()["7.weight"]
            # if iter_count > 0:
                # assert weights_after.numpy().all() == weights_init.numpy().all(), "Discriminator has " \
                #                                                           "trained"


            if (iter_count % show_every == 0):
                checkpt_t = time.time()
                print("time : {:.2f} sec" .format(checkpt_t - start_t))
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_total_error.data[0],
                                                           g_loss.data[0]))
                print("real logits average ", torch.mean(logits_real).data)
                print("average output generator : ", torch.mean(fake_images).data)
                print("fake logits average ", torch.mean(gen_logits_fake).data)
                imgs = fake_images[:64].data
                show_cifar(imgs, iter_num=iter_count, save=True, show=False, name=generator.label)
            iter_count += 1
            if iter_count % save_every == 0:
                torch.save(discriminator, 'results/weights/discriminator' +
                           discriminator.label + '.pt')
                torch.save(generator, 'results/weights/generator' + generator.label
                           + '.pt')

if __name__ == '__main__':
    generator = Generator(10**-3, [1024, 1024], label='mixed_dc_cifar_09')
    discriminator = discriminator_func()
    train(discriminator, generator)
