from gans.utils import data_loader, sample_noise, plot_batch_images
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from gans.utils import xavier_init
from torch.nn import init
from torch.autograd import Variable

noise_dim = 96
batch_size = 128
num_batches = int(50000/128)

class Discriminator(nn.Module):
    def __init__(self, learning_rate, hidden_dims, label=''):
        super().__init__()
        self.label = label
        self.learning_rate = learning_rate
        self.hidden_dims = hidden_dims
        self.weight_std = 10**-3
        self.sigma_1_prior = Variable(torch.Tensor([0.001]), requires_grad=False)
        self.sigma_2_prior = Variable(torch.Tensor([10**-7]), requires_grad=False)
        self.prior_weight = 0.5
        self.num_samples = 1

        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)

        # # convlutional layers
        # self.W1_conv_mu = nn.Parameter(xavier_init((32, 1, 5, 5)).type(torch.FloatTensor),
        #                   requires_grad=True)
        # self.W1_conv_rho = nn.Parameter(xavier_init((32, 1, 5, 5)).type(
        #     torch.FloatTensor), requires_grad=True)
        # self.W2_conv_mu = nn.Parameter(xavier_init((64, 32, 5, 5)).type(torch.FloatTensor),
        #                                requires_grad=True)
        # self.W2_conv_rho = nn.Parameter(xavier_init((64, 32, 5, 5)).type(torch.FloatTensor),
        #                                requires_grad=True)

        # linear layers
        self.W1_mu = nn.Parameter(xavier_init((self.hidden_dims[0], self.hidden_dims[1])).type(
            torch.FloatTensor), requires_grad=True)
        self.W1_rho = nn.Parameter(xavier_init((self.hidden_dims[0], self.hidden_dims[1])).type(
            torch.FloatTensor), requires_grad=True)
        # self.W2_mu = nn.Parameter(xavier_init((self.hidden_dims[0], self.hidden_dims[1])).type(torch.FloatTensor), requires_grad=True)
        # self.W2_rho = nn.Parameter(xavier_init((self.hidden_dims[0], self.hidden_dims[1])).type(torch.FloatTensor), requires_grad=True)
        self.W3_mu = nn.Parameter(xavier_init((self.hidden_dims[1], 1)).type(torch.FloatTensor),
                                  requires_grad=True)
        self.W3_rho = nn.Parameter(xavier_init((self.hidden_dims[1], 1)).type(torch.FloatTensor),
                                   requires_grad=True)


    def compute_parameters(self):
        # print('w1_mu', self.W1_mu)
        # print('w1_rho', self.W1_rho)
        self.W1_sigma = torch.log(1 + torch.exp(self.W1_rho))
        self.W1 = (self.W1_mu+ self.W1_sigma * \
                   Variable(self.weight_std * torch.randn(self.W1_mu.size()), requires_grad=False))

        # self.W2_sigma = torch.log(1 + torch.exp(self.W2_rho))
        # self.W2 = (self.W2_mu + self.W2_sigma * \
        #            Variable(self.weight_std * torch.randn(self.W2_mu.size()), requires_grad=False))

        self.W3_sigma = torch.log(1 + torch.exp(self.W3_rho))
        self.W3 = (self.W3_mu + self.W3_sigma * \
                   Variable(self.weight_std * torch.randn(self.W3_mu.size()), requires_grad=False))

        print("Mean conv W1 {0}, var {1}".format(torch.mean(self.conv1.weight.data), torch.std(
            self.conv1.weight.data)))        # self.W3_sigma = torch.log(1 + torch.exp(
        print("Mean conv W2 {0}, var {1}".format(torch.mean(self.conv2.weight.data), torch.std(
            self.conv2.weight.data)))

        # self.W1_conv_sigma = torch.log(1 + torch.exp(self.W1_conv_rho))
        # self.W1_conv = (self.W1_conv_mu + self.W1_conv_sigma *
        #                 Variable(self.weight_std * torch.randn(self.W1_conv_sigma.size()),
        #                          requires_grad=False))
        # self.W2_conv_sigma = torch.log(1 + torch.exp(self.W2_conv_rho))
        # self.W2_conv = (self.W2_conv_mu + self.W2_conv_sigma *
        #                 Variable(self.weight_std * torch.randn(self.W2_conv_sigma.size()),
        #                          requires_grad=False))

    def forward(self, input=None):
        self.compute_parameters()
        if input is not None:
            input = input.view(batch_size, 1, 28, 28)
            x = self.pool(self.conv1(input))
            x = self.pool(self.conv2(x))
            reshaped_input = x.view(x.size(0), -1)
            h1 = f.leaky_relu(torch.matmul(reshaped_input, self.W1), negative_slope=0.01)
            preds = torch.matmul(h1, self.W3)
            return preds

    def log_gaussian(self, x, mu, sigma):
        prob = - torch.log(torch.abs(sigma)) - (x - mu)**2 / (2 * sigma**2)
        return prob

    def gaussian(self, x, mu, sigma):
        prob = 1/sigma * torch.exp(-(x - mu)**2/(2*sigma**2)) + 10**-8
        return prob

    def log_prior(self, x):
        prob = torch.log(self.prior_weight * self.gaussian(x, 0, self.sigma_1_prior)
                            + (1-self.prior_weight)*self.gaussian(x, 0, self.sigma_2_prior))
        return prob

    def log_likelyhood(self, x, predicted):
        prob = (- torch.log(self.sigma_1_prior) -
               (x - predicted)**2 / (2 * self.sigma_1_prior**2))*self.prior_weight
        return prob

    def log_posterior(self, x, mu, sigma):
        sigma = sigma
        prob = self.log_gaussian(x, mu, sigma)
        return prob

    def loss(self, logits_real, logits_fake):
        target_real = 1
        target_fake = 0
        log_qw, log_pw, log_likelyhood_gauss = 0, 0, 0

        for _ in range(self.num_samples):
            log_pw += 1/self.num_samples*(self.log_prior(self.W1).sum() +
                                          # self.log_prior(self.W2).sum() +
                                          self.log_prior(self.W3).sum())



            log_qw += 1 / self.num_samples * (self.log_posterior(self.W1, self.W1_mu, self.W1_sigma).sum() +
                                              # self.log_posterior(self.W2, self.W2_mu, self.W2_sigma).sum() +
                                              self.log_posterior(self.W3, self.W3_mu, self.W3_sigma).sum())
          # t2 = time.time()

            log_likelyhood_gauss += 1/self.num_samples * self.log_likelyhood(logits_real,
                                                                             target_real).sum()
            log_likelyhood_gauss += 1/self.num_samples * self.log_likelyhood(logits_fake,
                                                                             target_fake).sum()

        loss = 1/batch_size * (1/num_batches * (log_qw - log_pw) - log_likelyhood_gauss)
        return loss

    def add_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
