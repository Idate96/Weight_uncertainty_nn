import torch
import torch.nn as nn
import os
from torch.nn import init
print(os.getcwd())
from torchvision import transforms
from torch.autograd import Variable
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
loader_train, loader_val = data_loader_cifar()

#
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def show_images(images, iter_num = None, save=True,  model=''):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg, sqrtimg]))
    if iter_num:
        title = 'iter_' + str(iter_num)
        if save:
            plt.savefig('results/' + model + '/' + title +'.png', bbox_inches='tight')
    return

def preprocess_img(x):
    return 2 * x - 1.0

def deprocess_img(x):
    return (x + 1.0) / 2.0

class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples

def data_loader(num_train=50000, num_val=5000, noise_dim=96, batch_size=128):
    custom_trans = transforms.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    mnist_train = dset.MNIST('datasets/MNIST_data', train=True, download=True,
                               transform=custom_trans)
    loader_train = DataLoader(mnist_train, batch_size=batch_size,
                              sampler=ChunkSampler(num_train, 0))

    mnist_val = dset.MNIST('datasets/MNIST_data', train=True, download=True,
                               transform=custom_trans)
    loader_val = DataLoader(mnist_val, batch_size=batch_size,
                            sampler=ChunkSampler(num_val, num_train))
    return loader_train, loader_val


def show_cifar(image, iter_num=None, save=True, name='', show=False):
    image_grid = torchvision.utils.make_grid(image, nrow=8)
    npimg = (image_grid.numpy() * 0.5) + 0.5
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    if save and iter_num is not None:
        title = 'iter_' + str(iter_num)
        plt.savefig('results/' + name + '/' + title + '.png',
                    bbox_inches='tight')
    if show:
        plt.draw()
        plt.pause(0.001)


def data_loader_cifar(num_train=45000, num_val=5000, noise_dim = 96, batch_size = 128):
    """Cifar 10 data loader.
    
    Num pixel in cifar = (3*32*32),
    noise dimension = 96
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    cifar_train = dset.CIFAR10('datasets/CIFAR10_data', train=True, download=True,
                               transform=transform)
    loader_train = DataLoader(cifar_train, batch_size=batch_size, sampler=ChunkSampler(num_train,
                                                                                       0))
    cifar_val = dset.CIFAR10('datasets/CIFAR10_data', train=True, download=True,
                             transform=transform)
    loader_val = DataLoader(cifar_val, batch_size=batch_size, sampler=ChunkSampler(num_val, num_train))
    return loader_train, loader_val


def plot_batch_images(images, iter_num=None, save=True, cifar=False, name=''):
    # loader_train, loader_test = data_loader()
    # imgs = loader_train.__iter__().next()[0].view(loader_train.batch_size, 784).numpy().squeeze()
    if cifar:
        # show_images_cifar(images, iter_num, save, name = '')
        pass
    else:
        show_images(images, iter_num, save, name='')
    plt.draw()
    plt.pause(0.001)

def sample_noise(batch_size, dim):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, dim) containing uniform
      random noise in the range (-1, 1).
    """
    return 2 * torch.rand(batch_size, dim) - 1

def xavier_init(shape):
    std = (2/(shape[0]+shape[1]))**0.5
    # print("variance weights : ", var)
    return torch.from_numpy(std * np.random.randn(*shape))


def initialize_weights(m):
    """m is a layer"""
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform(m.weight.data)




if __name__=='__main__':
    for x, _ in loader_train:
        print(np.shape(x))
        plot_batch_images(x, iter_num=10, cifar=True, name='test_images', save=False)