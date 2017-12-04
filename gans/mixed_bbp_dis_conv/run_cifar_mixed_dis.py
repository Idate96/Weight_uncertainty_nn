import torch
import time
from gans.utils import data_loader_cifar, sample_noise, plot_batch_images, show_cifar
from torch.autograd import Variable
from gans.mixed_bbp_dis_conv.discriminator import Discriminator
from gans.mixed_bbp_dis_conv.generator import generator_func, generator_loss, optimizer_generator

num_train = 50000
num_val = 5000
noise_dim = 96
batch_size = 128
loader_train, loader_val = data_loader_cifar()
dtype = torch.FloatTensor

def train(discriminator, generator, show_every=50, num_epochs=20, save_every=2000):
    start_t = time.time()
    iter_count = 0
    optimizer_gen = optimizer_generator(generator)
    discriminator.add_optimizer()
    for epoch in range(num_epochs):
        for x, _ in loader_train:
            if len(x) != batch_size:
                continue
            discriminator.optimizer.zero_grad()
            real_data = Variable(x).type(torch.FloatTensor)
            # noise = Variable(sample_noise(batch_size, 32**2*3)).type(torch.FloatTensor)
            logits_real = discriminator.forward(real_data).type(torch.FloatTensor)

            g_fake_seed = Variable(sample_noise(batch_size, noise_dim)).type(torch.FloatTensor)
            fake_images = generator(g_fake_seed)
            logits_fake = discriminator.forward(fake_images).detach()

            d_total_error = discriminator.loss(logits_real, logits_fake)
            d_total_error.backward()
            discriminator.optimizer.step()

            optimizer_gen.zero_grad()
            g_fake_seed = Variable(sample_noise(batch_size, noise_dim)).type(torch.FloatTensor)
            fake_images = generator(g_fake_seed)

            gen_logits_fake = discriminator.forward(fake_images)
            g_loss = generator_loss(gen_logits_fake)
            g_loss.backward()
            optimizer_gen.step()

            if (iter_count % show_every == 0):
                checkpt_t = time.time()
                print("time : {:.2f} sec" .format(checkpt_t - start_t))
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_total_error.data[0],
                                                           g_loss.data[0]))
                print("real logits average ", torch.mean(logits_real).data)
                print("average output generator : ", torch.mean(fake_images).data)
                print("fake logits average ", torch.mean(gen_logits_fake).data)
                imgs = fake_images[:64].data
                show_cifar(imgs, iter_num=iter_count, save=True, show=False, name=discriminator.label)
            iter_count += 1
            if iter_count % save_every == 0:
                torch.save(discriminator, 'results/weights/discriminator' +
                           discriminator.label + '.pt')
                torch.save(generator, 'results/weights/generator' + generator.label
                           + '.pt')
if __name__ == '__main__':
    discriminator = Discriminator(2e-4, [1600, 1600], label='mixed_dc_cifar_dis_01')
    generator = generator_func()
    train(discriminator, generator)
