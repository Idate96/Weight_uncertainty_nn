from utils import data_loader, sample_noise, plot_batch_images
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

class Generator(nn.Module):
    def __init__(self, learning_rate, hidden_dims, label=''):
        super().__init__()
        self.label = label
        self.learning_rate = learning_rate
        self.hidden_dims = hidden_dims
        self.weight_std = 10**-3
        self.sigma_1_prior = Variable(torch.Tensor([0.001]), requires_grad=False)
        self.sigma_2_prior = Variable(torch.Tensor([10**-7]), requires_grad=False)
        self.prior_weight = 0.5
        self.num_samples = 2
        # layers
        self.W1_mu = nn.Parameter(xavier_init((noise_dim, self.hidden_dims[0])).type(torch.FloatTensor), requires_grad=True)
        self.W1_rho = nn.Parameter(xavier_init((noise_dim, self.hidden_dims[0])).type(torch.FloatTensor), requires_grad=True)
        self.W2_mu = nn.Parameter(xavier_init((self.hidden_dims[0], 128*7*7)).type(
            torch.FloatTensor), requires_grad=True)
        self.W2_rho = nn.Parameter(xavier_init((self.hidden_dims[0], 128*7*7)).type(
            torch.FloatTensor),
                                   requires_grad=True)
        # self.W3_mu = nn.Parameter(xavier_init((self.hidden_dims[1], 32**2*3)).type(torch.FloatTensor),
        #                           requires_grad=True)
        # self.W3_rho = nn.Parameter(xavier_init((self.hidden_dims[1], 32**2*3)).type(torch.FloatTensor),
        #                            requires_grad=True)
        # # deconv layers
        self.W1_deconv_mu = nn.Parameter(xavier_init((128, 64, 6, 6)).type(torch.FloatTensor),
                          requires_grad=True)
        self.W1_deconv_rho = nn.Parameter(xavier_init((128, 64, 6, 6)).type(
            torch.FloatTensor), requires_grad=True)
        self.W2_deconv_mu = nn.Parameter(xavier_init((64, 3, 4, 4)).type(torch.FloatTensor),
                                       requires_grad=True)
        self.W2_deconv_rho = nn.Parameter(xavier_init((64, 3, 4, 4)).type(torch.FloatTensor),
                                       requires_grad=True)

    def compute_parameters(self):
        # print('w1_mu', self.W1_mu)
        # print('w1_rho', self.W1_rho)
        self.W1_sigma = torch.log(1 + torch.exp(self.W1_rho))
        self.W1 = (self.W1_mu + self.W1_sigma * \
                   Variable(self.weight_std * torch.randn(self.W1_mu.size()), requires_grad=False))

        self.W2_sigma = torch.log(1 + torch.exp(self.W2_rho))
        self.W2 = (self.W2_mu + self.W2_sigma * \
                   Variable(self.weight_std * torch.randn(self.W2_mu.size()), requires_grad=False))

        # self.W3_sigma = torch.log(1 + torch.exp(self.W3_rho))
        # self.W3 = (self.W3_mu + self.W3_sigma * \
        #            Variable(self.weight_std * torch.randn(self.W3_mu.size()), requires_grad=False))

        self.W1_deconv_sigma = torch.log(1 + torch.exp(self.W1_deconv_rho))
        self.W1_deconv = (self.W1_deconv_mu + self.W1_deconv_sigma *
                        Variable(self.weight_std * torch.randn(self.W1_deconv_sigma.size()),
                                 requires_grad=False))
        self.W2_deconv_sigma = torch.log(1 + torch.exp(self.W2_deconv_rho))
        self.W2_deconv = (self.W2_deconv_mu + self.W2_deconv_sigma *
                        Variable(self.weight_std * torch.randn(self.W2_deconv_sigma.size()),
                                 requires_grad=False))

    def forward(self, input=None):
        self.compute_parameters()
        if input is not None:
            input = input.view(input.size(0), -1)
            h1 = f.leaky_relu(torch.matmul(input, self.W1))
            norm_layer1 = nn.BatchNorm1d(self.hidden_dims[0])
            h1 = norm_layer1(h1)
            h2 = f.leaky_relu(torch.matmul(h1, self.W2))
            norm_layer2 = nn.BatchNorm1d(128*7*7)
            h2 = norm_layer2(h2)
            # batch-norm and convolution
            x = h2.view(batch_size, 128, 7, 7)
            output_deconv_1 = f.leaky_relu(f.conv_transpose2d(x, self.W1_deconv, stride=2, padding=1))
            norm_layer3 = nn.BatchNorm2d(64)
            output_deconv_1 = norm_layer3(output_deconv_1)
            output_deconv_2 = f.conv_transpose2d(output_deconv_1, self.W2_deconv, stride=2,
                                                 padding=1)
            preds = f.tanh(output_deconv_2)
            return preds

    def log_gaussian(self, x, mu, sigma):
        prob = - torch.log(torch.abs(sigma)) - (x - mu)**2 / (2 * sigma**2)
        return prob

    def gaussian(self, x, mu, sigma):
        prob = 1/sigma * torch.exp(-(x-mu)**2/(2*sigma**2)) + 10**-8
        return prob

    def log_prior(self, x):
        prob = torch.log(self.prior_weight * self.gaussian(x, 0, self.sigma_1_prior)
                            + (1-self.prior_weight)*self.gaussian(x, 0, self.sigma_2_prior))
        return prob

    def log_likelyhood(self, x, predicted):
        prob = (- torch.log(self.sigma_1_prior) -
               (x - predicted)**2 / (2 * self.sigma_1_prior**2))
        return prob

    def log_posterior(self, x, mu, sigma):
        sigma = sigma
        prob = self.log_gaussian(x, mu, sigma)
        return prob

    def loss(self, scores_fake):
        # we want ot fool the discriminator
        target = 1
        log_pw = 1/self.num_samples*(self.log_prior(self.W1).sum() + self.log_prior(self.W2).sum())
                                      # self.log_prior(self.W3).sum())

        log_pw += 1/self.num_samples*(self.log_prior(self.W1_deconv).sum() + self.log_prior(
                                        self.W2_deconv).sum())

        log_qw = 1 / self.num_samples * (self.log_posterior(self.W1, self.W1_mu, self.W1_sigma).sum() +
                                          self.log_posterior(self.W2, self.W2_mu,
                                                             self.W2_sigma).sum())
                                          # self.log_posterior(self.W3, self.W3_mu, self.W3_sigma).sum())
    # t2 = time.time()
        log_qw += 1/ self.num_samples * (self.log_posterior(self.W1_deconv, self.W1_deconv_mu,
                                                            self.W1_deconv_sigma).sum() +
                    self.log_posterior(self.W2_deconv, self.W2_deconv_mu,
                                       self.W2_deconv_sigma).sum())

        log_likelyhood_gauss = 1/self.num_samples * self.log_likelyhood(scores_fake, target).sum()

        loss = 1/batch_size * (1/num_batches * (log_qw - log_pw) - log_likelyhood_gauss)
        return loss

    def add_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4, betas=(0.5, 0.999))
