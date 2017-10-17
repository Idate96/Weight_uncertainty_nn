import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import time
batch_size = 100
num_batches = 55000/batch_size
kl_weight = 1

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)

def xavier_init(shape):
    var = 2/(sum(shape))
    # print("variance weights : ", var)
    return torch.from_numpy(var * np.random.randn(*shape))

# class SimpleBayes(nn.Module):
#     def __init__(self, learning_rate, hidden_dims):
#         super().__init__()
#         self.learning_rate = learning_rate
#         self.hidden_dims = hidden_dims
#         self.weight_var = 10**-3
#         self.sigma_1_prior = Variable(torch.Tensor([10**-1]), requires_grad=False)
#         self.sigma_2_prior = Variable(torch.Tensor([10**-7]), requires_grad=False)
#         self.prior_weight = 1
#         self.num_samples = 10
#         # layers
#         self.W1_mu = nn.Parameter(xavier_init((28**2, self.hidden_dims[0])).type(torch.FloatTensor), requires_grad=True)
#         self.W1_rho = nn.Parameter(xavier_init((28**2, self.hidden_dims[0])).type(torch.FloatTensor), requires_grad=True)
#         self.W2_mu = nn.Parameter(xavier_init((hidden_dims[0], 10)).type(torch.FloatTensor), requires_grad=True)
#         self.W2_rho = nn.Parameter(xavier_init((hidden_dims[0], 10)).type(torch.FloatTensor), requires_grad=True)
#         self.W1_common = nn.Parameter(xavier_init((28**2, self.hidden_dims[0])).type(torch.FloatTensor), requires_grad=True)
#         self.W2_common = nn.Parameter(xavier_init((self.hidden_dims[0], 10)).type(torch.FloatTensor), requires_grad=True)
#
#     def forward(self, input):
#         input = input.view(input.size(0), -1)
#         self.W1_sigma = torch.log(1 + torch.exp(self.W1_rho))
#         # print("simgma weights :", self.W1_sigma)
#         self.W1 = (self.W1_mu + self.W1_sigma * \
#                                Variable(self.weight_var*torch.randn(self.W1_mu.size()), requires_grad=False))
#         # print("weihgts ;", self.W1)
#         h1 = f.relu(input @ self.W1)
#         self.W2_sigma = torch.log(1 + torch.exp(self.W2_rho))
#         self.W2 = (self.W2_mu + self.W2_sigma * \
#                     Variable(self.weight_var*torch.randn(self.W2_mu.size()), requires_grad=False))
#         preds = f.softmax(h1 @ self.W2)
#         return preds
#
#     def log_gaussian(self, x, mu, sigma):
#         # print("shape x : {0}, shape mu : {1}, shape sigma {2}".format(sigma.size(), 0, 0))
#         # print("log gaussian called")
#         # print("sigma  ", sigma)
#         # print("weights : ", x)
#         prob = -0.5 * (torch.log(sigma) + (x - mu)**2 / sigma**2)
#         # print("prob : ", prob)
#         # print("done gaussian")
#         return prob
#
#     def log_prior(self, x):
#         prob = self.prior_weight * self.log_gaussian(x, 0, self.sigma_1_prior) + \
#                (1-self.prior_weight)* self.log_gaussian(x, 0, self.sigma_2_prior)
#         return prob
#
#     def loss(self, output, target):
#         # print("calculating loss...")
#         log_pw = self.log_prior(self.W1).sum() + self.log_prior(self.W2).sum()
#         # print("prior value : ", log_pw.data)
#         # print("calculated prior")
#         log_qw = self.log_gaussian(self.W1, self.W1_mu, self.W1_sigma).sum() +\
#                  self.log_gaussian(self.W2, self.W2_mu, self.W2_sigma).sum()
#         # print("posterior :", log_qw.data)
#         # print("calculated posterior")
#         # print(target.size())
#         log_likelyhood = f.cross_entropy(output, target)
#         # print("likelihood : ", log_likelyhood.data)
#         # print("calculated likelihood")
#         loss = 1/50000 * (-log_qw + log_pw) + log_likelyhood
#         return loss
#
#     def add_optimizer(self):
#         parameters = [self.W1_mu, self.W1_rho, self.W2_mu, self.W2_rho]
#         parameters_common = [self.W1_common, self.W2_common]
#         # print(list(self.parameters()))
#         self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
#
# class MultisampleBayes(nn.Module):
#     def __init__(self, learning_rate, hidden_dims):
#         super().__init__()
#         self.learning_rate = learning_rate
#         self.hidden_dims = hidden_dims
#         self.weight_var = 10**-3
#         self.sigma_1_prior = Variable(torch.Tensor([10**-1]), requires_grad=False)
#         self.sigma_2_prior = Variable(torch.Tensor([10**-7]), requires_grad=False)
#         self.prior_weight = 0.5
#         self.num_samples = 2
#         # layers
#         self.W1_mu = nn.Parameter(xavier_init((28**2, self.hidden_dims[0])).type(torch.FloatTensor), requires_grad=True)
#         self.W1_rho = nn.Parameter(xavier_init((28**2, self.hidden_dims[0])).type(torch.FloatTensor), requires_grad=True)
#         self.W2_mu = nn.Parameter(xavier_init((hidden_dims[0], 10)).type(torch.FloatTensor), requires_grad=True)
#         self.W2_rho = nn.Parameter(xavier_init((hidden_dims[0], 10)).type(torch.FloatTensor), requires_grad=True)
#         self.W1_common = nn.Parameter(xavier_init((28**2, self.hidden_dims[0])).type(torch.FloatTensor), requires_grad=True)
#         self.W2_common = nn.Parameter(xavier_init((self.hidden_dims[0], 10)).type(torch.FloatTensor), requires_grad=True)
#
#     def forward(self, input=None):
#         if input is None:
#             self.W1_sigma = torch.log(1 + torch.exp(self.W1_rho))
#             # print("simgma weights :", self.W1_sigma)
#             self.W1 = (self.W1_mu + self.W1_sigma * \
#                        Variable(self.weight_var * torch.randn(self.W1_mu.size()), requires_grad=False))
#             self.W2_sigma = torch.log(1 + torch.exp(self.W2_rho))
#             self.W2 = (self.W2_mu + self.W2_sigma * \
#                        Variable(self.weight_var * torch.randn(self.W2_mu.size()), requires_grad=False))
#         else:
#             input = input.view(input.size(0), -1)
#             self.W1_sigma = torch.log(1 + torch.exp(self.W1_rho))
#             # print("simgma weights :", self.W1_sigma)
#             self.W1 = (self.W1_mu + self.W1_sigma * \
#                                    Variable(self.weight_var*torch.randn(self.W1_mu.size()), requires_grad=False))
#             # print("weihgts ;", self.W1)
#             h1 = f.relu(input @ self.W1)
#             self.W2_sigma = torch.log(1 + torch.exp(self.W2_rho))
#             self.W2 = (self.W2_mu + self.W2_sigma * \
#                         Variable(self.weight_var*torch.randn(self.W2_mu.size()), requires_grad=False))
#             preds = f.softmax(h1 @ self.W2)
#             return preds
#
#     def log_gaussian(self, x, mu, sigma):
#         # print("shape x : {0}, shape mu : {1}, shape sigma {2}".format(sigma.size(), 0, 0))
#         # print("log gaussian called")
#         # print("sigma  ", sigma)
#         # print("weights : ", x)
#         prob = -0.5 * (torch.log(sigma) + (x - mu)**2 / sigma**2)
#         # print("prob : ", prob)
#         # print("done gaussian")
#         return prob
#
#     def log_prior(self, x):
#         prob = self.prior_weight * self.log_gaussian(x, 0, self.sigma_1_prior) + \
#                (1-self.prior_weight)* self.log_gaussian(x, 0, self.sigma_2_prior)
#         return prob
#
#     def loss(self, output, target):
#         log_qw, log_pw, log_likelyhood = 0, 0, 0
#         for sample_idx in range(self.num_samples):
#             if sample_idx > 0:
#                 self.forward()
#         # print("calculating loss...")
#             log_pw += 1/self.num_samples*(self.log_prior(self.W1).sum() + self.log_prior(self.W2).sum())
#             # print("prior value : ", log_pw.data)
#             # print("calculated prior")
#             log_qw += 1/self.num_samples*(self.log_gaussian(self.W1, self.W1_mu, self.W1_sigma).sum() +\
#                      self.log_gaussian(self.W2, self.W2_mu, self.W2_sigma).sum())
#             # print("posterior :", log_qw.data)
#             # print("calculated posterior")
#             # print(target.size())
#             log_likelyhood += 1/self.num_samples * f.cross_entropy(output, target)
#         # print("likelihood : ", log_likelyhood.data)
#         # print("calculated likelihood")
#         loss = 1/50000 * (-log_qw + log_pw) + log_likelyhood
#         return loss
#
#     def add_optimizer(self):
#         parameters = [self.W1_mu, self.W1_rho, self.W2_mu, self.W2_rho]
#         parameters_common = [self.W1_common, self.W2_common]
#         # print(list(self.parameters()))
#         self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
#
# class SimpleBayeswBias(nn.Module):
#     def __init__(self, learning_rate, hidden_dims):
#         super().__init__()
#         self.learning_rate = learning_rate
#         self.hidden_dims = hidden_dims
#         self.weight_var = 10**-3
#         self.sigma_1_prior = Variable(torch.Tensor([10**-1]), requires_grad=False)
#         self.sigma_2_prior = Variable(torch.Tensor([10**-7]), requires_grad=False)
#         self.prior_weight = 1
#         self.num_samples = 10
#         # layers weights
#         self.W1_mu = nn.Parameter(xavier_init((28**2, self.hidden_dims[0])).type(torch.FloatTensor), requires_grad=True)
#         self.W1_rho = nn.Parameter(xavier_init((28**2, self.hidden_dims[0])).type(torch.FloatTensor), requires_grad=True)
#         self.W2_mu = nn.Parameter(xavier_init((hidden_dims[0], 10)).type(torch.FloatTensor), requires_grad=True)
#         self.W2_rho = nn.Parameter(xavier_init((hidden_dims[0], 10)).type(torch.FloatTensor), requires_grad=True)
#         # layer bias
#         self.b1_mu = nn.Parameter(xavier_init((self.hidden_dims[0],)).type(torch.FloatTensor),
#                                   requires_grad=True)
#         self.b1_rho = nn.Parameter(xavier_init((self.hidden_dims[0], )).type(torch.FloatTensor),
#                                    requires_grad=True)
#         self.b2_mu = nn.Parameter(xavier_init((10, )).type(torch.FloatTensor), requires_grad=True)
#         self.b2_rho = nn.Parameter(xavier_init((10, )).type(torch.FloatTensor), requires_grad=True)
#
#     def forward(self, input):
#         input = input.view(input.size(0), -1)
#         self.W1_sigma = torch.log(1 + torch.exp(self.W1_rho))
#         self.b1_sigma = torch.log(1 + torch.exp(self.b1_rho))
#         # print("simgma weights :", self.W1_sigma)
#         self.W1 = (self.W1_mu + self.W1_sigma * \
#                                Variable(self.weight_var*torch.randn(self.W1_mu.size()), requires_grad=False))
#         self.b1 = (self.b1_mu + self.b1_sigma * \
#                                Variable(self.weight_var*torch.randn(self.b1_mu.size()), requires_grad=False))
#         # print("weihgts ;", self.W1)
#         h1 = f.relu(input @ self.W1 + self.b1)
#         self.b2_sigma = torch.log(1 + torch.exp(self.b2_rho))
#         self.W2_sigma = torch.log(1 + torch.exp(self.W2_rho))
#         self.W2 = (self.W2_mu + self.W2_sigma * \
#                     Variable(self.weight_var*torch.randn(self.W2_mu.size()), requires_grad=False))
#         self.b2 = (self.b2_mu + self.b2_sigma * \
#                    Variable(self.weight_var * torch.randn(self.b2_mu.size()), requires_grad=False))
#         preds = f.softmax(h1 @ self.W2 + self.b2)
#         return preds
#
#     def log_gaussian(self, x, mu, sigma):
#         # print("shape x : {0}, shape mu : {1}, shape sigma {2}".format(sigma.size(), 0, 0))
#         # print("log gaussian called")
#         # print("sigma  ", sigma)
#         # print("weights : ", x)
#         prob = -0.5 * (torch.log(sigma) + (x - mu)**2 / sigma**2)
#         # print("prob : ", prob)
#         # print("done gaussian")
#         return prob
#
#     def log_prior(self, x):
#         prob = self.prior_weight * self.log_gaussian(x, 0, self.sigma_1_prior) + \
#                (1-self.prior_weight)* self.log_gaussian(x, 0, self.sigma_2_prior)
#         return prob
#
#     def loss(self, output, target):
#         # print("calculating loss...")
#         log_pw = self.log_prior(self.W1).sum() + self.log_prior(self.W2).sum()
#         log_pw += self.log_prior(self.b1).sum() + self.log_prior(self.b2).sum()
#         # print("prior value : ", log_pw.data)
#         # print("calculated prior")
#         log_qw = self.log_gaussian(self.W1, self.W1_mu, self.W1_sigma).sum() +\
#                  self.log_gaussian(self.W2, self.W2_mu, self.W2_sigma).sum()
#         log_qw += self.log_gaussian(self.b1, self.b1_mu, self.b1_sigma).sum() +\
#                  self.log_gaussian(self.b2, self.b2_mu, self.b2_sigma).sum()
#         # print("posterior :", log_qw.data)
#         # print("calculated posterior")
#         # print(target.size())
#         log_likelyhood = f.cross_entropy(output, target)
#         # print("likelihood : ", log_likelyhood.data)
#         # print("calculated likelihood")
#         loss = 1/50000 * (-log_qw + log_pw) + log_likelyhood
#         return loss
#
#     def add_optimizer(self):
#         self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

class BayesBackprop(nn.Module):
    def __init__(self, learning_rate, hidden_dims):
        super().__init__()
        self.learning_rate = learning_rate
        self.hidden_dims = hidden_dims
        self.weight_var = 10**-3
        self.sigma_1_prior = Variable(torch.Tensor([10**-3]), requires_grad=False)
        self.sigma_2_prior = Variable(torch.Tensor([10**-7]), requires_grad=False)
        self.prior_weight = 1
        self.num_samples = 3
        # layers
        self.W1_mu = nn.Parameter(xavier_init((28**2, self.hidden_dims[0])).type(torch.FloatTensor), requires_grad=True)
        self.W1_rho = nn.Parameter(xavier_init((28**2, self.hidden_dims[0])).type(torch.FloatTensor), requires_grad=True)
        self.W2_mu = nn.Parameter(xavier_init((self.hidden_dims[0], self.hidden_dims[1])).type(torch.FloatTensor), requires_grad=True)
        self.W2_rho = nn.Parameter(xavier_init((self.hidden_dims[0], self.hidden_dims[1])).type(torch.FloatTensor), requires_grad=True)
        self.W3_mu = nn.Parameter(xavier_init((self.hidden_dims[1], 10)).type(torch.FloatTensor),
                                  requires_grad=True)
        self.W3_rho = nn.Parameter(xavier_init((self.hidden_dims[1], 10)).type(torch.FloatTensor),
                                   requires_grad=True)
        # layer bias
        self.b1_mu = nn.Parameter(xavier_init((self.hidden_dims[0],)).type(torch.FloatTensor),
                                  requires_grad=True)
        self.b1_rho = nn.Parameter(xavier_init((self.hidden_dims[0], )).type(torch.FloatTensor),
                                   requires_grad=True)
        self.b2_mu = nn.Parameter(xavier_init((self.hidden_dims[1], )).type(torch.FloatTensor), requires_grad=True)
        self.b2_rho = nn.Parameter(xavier_init((self.hidden_dims[1], )).type(torch.FloatTensor), requires_grad=True)
        self.b3_mu = nn.Parameter(xavier_init((10,)).type(torch.FloatTensor), requires_grad=True)
        self.b3_rho = nn.Parameter(xavier_init((10,)).type(torch.FloatTensor), requires_grad=True)

    def compute_parameters(self):
        self.W1_sigma = torch.log(1 + torch.exp(self.W1_rho))
        self.W1 = (self.W1_mu + self.W1_sigma * \
                   Variable(self.weight_var * torch.randn(self.W1_mu.size()), requires_grad=False))
        self.b1_sigma = torch.log(1 + torch.exp(self.b1_rho))
        self.b1 = (self.b1_mu + self.b1_sigma * \
                   Variable(self.weight_var * torch.randn(self.b1_mu.size()), requires_grad=False))
        self.W2_sigma = torch.log(1 + torch.exp(self.W2_rho))
        self.W2 = (self.W2_mu + self.W2_sigma * \
                   Variable(self.weight_var * torch.randn(self.W2_mu.size()), requires_grad=False))
        self.b2_sigma = torch.log(1 + torch.exp(self.b2_rho))
        self.b2 = (self.b2_mu + self.b2_sigma * \
                   Variable(self.weight_var * torch.randn(self.b2_mu.size()), requires_grad=False))
        self.W3_sigma = torch.log(1 + torch.exp(self.W3_rho))
        self.W3 = (self.W3_mu + self.W3_sigma * \
                   Variable(self.weight_var * torch.randn(self.W3_mu.size()), requires_grad=False))
        self.b3_sigma = torch.log(1 + torch.exp(self.b3_rho))
        self.b3 = (self.b3_mu + self.b3_sigma * \
                   Variable(self.weight_var * torch.randn(self.b3_mu.size()), requires_grad=False))

    def forward(self, input=None):
        self.compute_parameters()
        if input is not None:
            input = input.view(input.size(0), -1)
            h1 = f.relu(torch.matmul(input, self.W1) + self.b1)
            h2 = f.relu(torch.matmul(h1, self.W2) + self.b2)
            preds = f.softmax(torch.matmul(h2, self.W3) + self.b3)
            return preds

    def log_gaussian(self, x, mu, sigma):
        # print("shape x : {0}, shape mu : {1}, shape sigma {2}".format(sigma.size(), 0, 0))
        # print("log gaussian called")
        # print("sigma  ", sigma)
        # print("weights : ", x)
        prob = -0.5 * np.log(2 * np.pi) - 0.5 * (torch.log(sigma) - (x - mu)**2 / 2 * sigma**2)
        # print("prob : ", prob)
        # print("done gaussian")
        return prob

    def log_prior(self, x):
        prob = self.prior_weight * self.log_gaussian(x, 0, self.sigma_1_prior) + \
               (1-self.prior_weight)* self.log_gaussian(x, 0, self.sigma_2_prior)
        return prob

    def loss(self, output, target):
        log_qw, log_pw, log_likelyhood = 0, 0, 0
        for sample_idx in range(self.num_samples):
            if sample_idx > 0:
                self.forward()
        # print("calculating loss...")
            log_pw += 1/self.num_samples*(self.log_prior(self.W1).sum() + self.log_prior(self.W2).sum() +
                                          self.log_prior(self.W3).sum())
            log_pw += 1/self.num_samples*(self.log_prior(self.b1).sum() + self.log_prior(self.b2).sum() +
                                          self.log_prior(self.b3).sum())
            # print("prior value : ", log_pw.data)
            # print("calculated prior")
            log_qw += 1/self.num_samples*(self.log_gaussian(self.W1, self.W1_mu, self.weight_var*self.W1_sigma).sum() +
                     self.log_gaussian(self.W2, self.W2_mu, self.weight_var*self.W2_sigma).sum() +
                                          self.log_gaussian(self.W3, self.W3_mu, self.weight_var*self.W3_sigma)).sum()
            log_qw += 1/self.num_samples*(self.log_gaussian(self.b1, self.b1_mu, self.weight_var*self.b1_sigma).sum() +
                     self.log_gaussian(self.b2, self.b2_mu, self.weight_var*self.b2_sigma).sum() +
                                          self.log_gaussian(self.b3, self.b3_mu, self.weight_var*self.b3_sigma)).sum()
            # print("posterior :", log_qw.data)
            # print("calculated posterior")
            # print(target.size())
            # print('output :', output)
            # print('target hot', target)
            log_likelyhood += 1/self.num_samples * f.cross_entropy(output, target)
        # print("likelihood : ", log_likelyhood.data)
        # print("calculated likelihood")
        loss = 1/num_batches * (-log_qw + log_pw) + log_likelyhood/batch_size
        return loss

    def add_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

class BayesBackpropGauss(nn.Module):
    def __init__(self, learning_rate, hidden_dims):
        super().__init__()
        self.learning_rate = learning_rate
        self.hidden_dims = hidden_dims
        self.weight_std = 10**-3
        self.sigma_1_prior = Variable(torch.Tensor([0.001]), requires_grad=False)
        self.sigma_2_prior = Variable(torch.Tensor([10**-7]), requires_grad=False)
        self.prior_weight = 0.25
        self.num_samples = 2
        # layers
        self.W1_mu = nn.Parameter(xavier_init((28**2, self.hidden_dims[0])).type(torch.FloatTensor), requires_grad=True)
        self.W1_rho = nn.Parameter(xavier_init((28**2, self.hidden_dims[0])).type(torch.FloatTensor), requires_grad=True)
        self.W2_mu = nn.Parameter(xavier_init((self.hidden_dims[0], self.hidden_dims[1])).type(torch.FloatTensor), requires_grad=True)
        self.W2_rho = nn.Parameter(xavier_init((self.hidden_dims[0], self.hidden_dims[1])).type(torch.FloatTensor), requires_grad=True)
        self.W3_mu = nn.Parameter(xavier_init((self.hidden_dims[1], 10)).type(torch.FloatTensor),
                                  requires_grad=True)
        self.W3_rho = nn.Parameter(xavier_init((self.hidden_dims[1], 10)).type(torch.FloatTensor),
                                   requires_grad=True)
        # layer bias
        self.b1_mu = nn.Parameter(xavier_init((self.hidden_dims[0],)).type(torch.FloatTensor),
                                  requires_grad=True)
        self.b1_rho = nn.Parameter(xavier_init((self.hidden_dims[0], )).type(torch.FloatTensor),
                                   requires_grad=True)
        self.b2_mu = nn.Parameter(xavier_init((self.hidden_dims[1], )).type(torch.FloatTensor), requires_grad=True)
        self.b2_rho = nn.Parameter(xavier_init((self.hidden_dims[1], )).type(torch.FloatTensor), requires_grad=True)
        self.b3_mu = nn.Parameter(xavier_init((10,)).type(torch.FloatTensor), requires_grad=True)
        self.b3_rho = nn.Parameter(xavier_init((10,)).type(torch.FloatTensor), requires_grad=True)

    def compute_parameters(self):
        # print('w1_mu', self.W1_mu)
        # print('w1_rho', self.W1_rho)
        self.W1_sigma = torch.log(1 + torch.exp(self.W1_rho))
        self.W1 = (self.W1_mu + self.W1_sigma * \
                   Variable(self.weight_std * torch.randn(self.W1_mu.size()), requires_grad=False))
        self.b1_sigma = torch.log(1 + torch.exp(self.b1_rho))
        self.b1 = (self.b1_mu + self.b1_sigma * \
                   Variable(self.weight_std * torch.randn(self.b1_mu.size()), requires_grad=False))
        self.W2_sigma = torch.log(1 + torch.exp(self.W2_rho))
        self.W2 = (self.W2_mu + self.W2_sigma * \
                   Variable(self.weight_std * torch.randn(self.W2_mu.size()), requires_grad=False))
        self.b2_sigma = torch.log(1 + torch.exp(self.b2_rho))
        self.b2 = (self.b2_mu + self.b2_sigma * \
                   Variable(self.weight_std * torch.randn(self.b2_mu.size()), requires_grad=False))
        self.W3_sigma = torch.log(1 + torch.exp(self.W3_rho))
        self.W3 = (self.W3_mu + self.W3_sigma * \
                   Variable(self.weight_std * torch.randn(self.W3_mu.size()), requires_grad=False))
        self.b3_sigma = torch.log(1 + torch.exp(self.b3_rho))
        self.b3 = (self.b3_mu + self.b3_sigma * \
                   Variable(self.weight_std * torch.randn(self.b3_mu.size()), requires_grad=False))

    def forward(self, input=None):
        self.compute_parameters()
        if input is not None:
            input = input.view(input.size(0), -1)
            h1 = f.relu(torch.matmul(input, self.W1) + self.b1)
            h2 = f.relu(torch.matmul(h1, self.W2) + self.b2)
            preds = f.softmax(torch.matmul(h2, self.W3) + self.b3)
            return preds

    def log_gaussian(self, x, mu, sigma):
        # print("shape x : {0}, shape mu : {1}, shape sigma {2}".format(sigma.size(), 0, 0))
        # print("log gaussian called")
        # print("sigma  ", sigma)
        # print("weights : ", x)
        prob = - 0.5 * torch.log(torch.abs(sigma)) - (x - mu)**2 / (2 * sigma**2)
        # print("prob : ", prob)
        # print("done gaussian")
        return prob

    def gaussian(self, x, mu, sigma):
        # print('x - my  ' + 89*"-", torch.min(-(x-mu)**2/(2*sigma**2)))
        prob = 1/sigma * torch.exp(-(x-mu)**2/(2*sigma**2)) + 10**-8
        return prob

    def log_prior(self, x):
        prob = torch.log(self.prior_weight * self.gaussian(x, 0, self.sigma_1_prior)
                            + (1-self.prior_weight)*self.gaussian(x, 0, self.sigma_2_prior))
        return prob

    def log_likelyhood(self, x, predicted):
        prob = (- 0.5 * torch.log(self.sigma_1_prior) -
               (x - predicted)**2 / (2 * self.sigma_1_prior**2))*self.prior_weight
        prob += (- 0.5 * torch.log(self.sigma_2_prior) -\
               (x - predicted)**2 / (2 * self.sigma_2_prior**2))*(1-self.prior_weight)
        return prob

    def log_posterior(self, x, mu, sigma):
        sigma = sigma
        prob = self.log_gaussian(x, mu, sigma)
        return prob

    def loss(self, input, output, target):
        log_qw, log_pw, log_likelyhood_gauss = 0, 0, 0
        for sample_idx in range(self.num_samples):
            if sample_idx > 0:
                output = self.forward(input)
                # print(output)
        # print("calculating loss...")
            log_pw += 1/self.num_samples*(self.log_prior(self.W1).sum() + self.log_prior(self.W2).sum() +
                                          self.log_prior(self.W3).sum())
            log_pw += 1/self.num_samples*(self.log_prior(self.b1).sum() + self.log_prior(self.b2).sum() +
                                          self.log_prior(self.b3).sum())
            # t1 = time.time()
            # print("prior value : ", log_pw.data)
            # print("calculated prior")
            # log_qw += 1/self.num_samples*(self.log_gaussian(self.W1, self.W1_mu, self.W1_rho).sum() +
            #          self.log_gaussian(self.W2, self.W2_mu, self.W2_rho).sum() +
            #                               self.log_gaussian(self.W3, self.W3_mu, self.W3_rho)).sum()
            # log_qw += 1/self.num_samples*(self.log_gaussian(self.b1, self.b1_mu, self.b1_rho).sum() +
            #          self.log_gaussian(self.b2, self.b2_mu, self.b2_rho).sum() +
            #                               self.log_gaussian(self.b3, self.b3_mu, self.b3_rho)).sum()

            log_qw += 1 / self.num_samples * (self.log_posterior(self.W1, self.W1_mu, self.W1_sigma).sum() +
                                              self.log_posterior(self.W2, self.W2_mu, self.W2_sigma).sum() +
                                              self.log_posterior(self.W3, self.W3_mu, self.W3_sigma).sum())
            log_qw += 1 / self.num_samples * (self.log_posterior(self.b1, self.b1_mu, self.b1_sigma).sum() +
                                              self.log_posterior(self.b2, self.b2_mu, self.b2_sigma).sum() +
                                              self.log_posterior(self.b3, self.b3_mu, self.b3_sigma).sum())         # t2 = time.time()
            # print("posterior :", log_qw.data)
            # print("calculated posterior")
            # print(target.size())
            # log_likelyhood += 1/self.num_samples * f.cross_entropy(output, target)
        # print("likelihood : ", log_likelyhood.data)
            log_likelyhood_gauss += 1/self.num_samples * self.log_likelyhood(output, target).sum()
            # t3 = time.time()
            # print('time prior ', t1-t0)
            # print('time post ', t2-t1)
            # print('time likely ', t3-t2)
        # print("calculated likelihood")
        # print('likelihood : ', log_likelyhood_gauss)
        # print('prior : ', log_pw)
        # print('post :', log_qw)
        # print("kl div : ",  1/num_batches * (log_qw - log_pw))
        loss = 1/batch_size * (1/num_batches * kl_weight * (log_qw - log_pw) - log_likelyhood_gauss)
        return loss

    def add_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

class BayesBackpropMixGauss(nn.Module):
    def __init__(self, learning_rate, hidden_dims):
        super().__init__()
        self.learning_rate = learning_rate
        self.hidden_dims = hidden_dims
        self.weight_std = 10**-3
        self.sigma_1_prior = Variable(torch.Tensor([0.001]), requires_grad=False)
        self.sigma_2_prior = Variable(torch.Tensor([10**-7]), requires_grad=False)
        self.prior_weight = 0.5
        self.num_samples = 2
        # layers
        self.W1_mu = nn.Parameter(xavier_init((28**2, self.hidden_dims[0])).type(torch.FloatTensor), requires_grad=True)
        self.W1_rho = nn.Parameter(xavier_init((28**2, self.hidden_dims[0])).type(torch.FloatTensor), requires_grad=True)
        self.W2_mu = nn.Parameter(xavier_init((self.hidden_dims[0], self.hidden_dims[1])).type(torch.FloatTensor), requires_grad=True)
        self.W2_rho = nn.Parameter(xavier_init((self.hidden_dims[0], self.hidden_dims[1])).type(torch.FloatTensor), requires_grad=True)
        self.W3_mu = nn.Parameter(xavier_init((self.hidden_dims[1], 10)).type(torch.FloatTensor),
                                  requires_grad=True)
        self.W3_rho = nn.Parameter(xavier_init((self.hidden_dims[1], 10)).type(torch.FloatTensor),
                                   requires_grad=True)
        # layer bias
        self.b1_mu = nn.Parameter(xavier_init((self.hidden_dims[0],)).type(torch.FloatTensor),
                                  requires_grad=True)
        self.b1_rho = nn.Parameter(xavier_init((self.hidden_dims[0],)).type(torch.FloatTensor),
                                   requires_grad=True)
        self.b2_mu = nn.Parameter(xavier_init((self.hidden_dims[1],)).type(torch.FloatTensor), requires_grad=True)
        self.b2_rho = nn.Parameter(xavier_init((self.hidden_dims[1],)).type(torch.FloatTensor), requires_grad=True)
        self.b3_mu = nn.Parameter(xavier_init((10,)).type(torch.FloatTensor), requires_grad=True)
        self.b3_rho = nn.Parameter(xavier_init((10,)).type(torch.FloatTensor), requires_grad=True)

    def compute_parameters(self):
        # print('w1_mu', self.W1_mu)
        # print('w1_rho', self.W1_rho)
        self.W1_sigma = torch.log(1 + torch.exp(self.W1_rho))
        self.W1 = (self.W1_mu+ self.W1_sigma * \
                   Variable(self.weight_std * torch.randn(self.W1_mu.size()), requires_grad=False))
        self.b1_sigma = torch.log(1 + torch.exp(self.b1_rho))
        self.b1 = (self.b1_mu + self.b1_sigma * \
                   Variable(self.weight_std * torch.randn(self.b1_mu.size()), requires_grad=False))
        self.W2_sigma = torch.log(1 + torch.exp(self.W2_rho))
        self.W2 = (self.W2_mu + self.W2_sigma * \
                   Variable(self.weight_std * torch.randn(self.W2_mu.size()), requires_grad=False))
        self.b2_sigma = torch.log(1 + torch.exp(self.b2_rho))
        self.b2 = (self.b2_mu + self.b2_sigma * \
                   Variable(self.weight_std * torch.randn(self.b2_mu.size()), requires_grad=False))
        self.W3_sigma = torch.log(1 + torch.exp(self.W3_rho))
        self.W3 = (self.W3_mu + self.W3_sigma * \
                   Variable(self.weight_std * torch.randn(self.W3_mu.size()), requires_grad=False))
        self.b3_sigma = torch.log(1 + torch.exp(self.b3_rho))
        self.b3 = (self.b3_mu + self.b3_sigma * \
                   Variable(self.weight_std * torch.randn(self.b3_mu.size()), requires_grad=False))

    def forward(self, input=None):
        self.compute_parameters()
        if input is not None:
            input = input.view(input.size(0), -1)
            h1 = f.relu(torch.matmul(input, self.W1) + self.b1)
            h2 = f.relu(torch.matmul(h1, self.W2) + self.b2)
            preds = f.softmax(torch.matmul(h2, self.W3) + self.b3)
            return preds

    def log_gaussian(self, x, mu, sigma):
        # print("shape x : {0}, shape mu : {1}, shape sigma {2}".format(sigma.size(), 0, 0))
        # print("log gaussian called")
        # print("sigma  ", sigma)
        # print("weights : ", x)
        prob = - torch.log(torch.abs(sigma)) - (x - mu)**2 / (2 * sigma**2)
        # print("prob : ", prob)
        # print("done gaussian")
        return prob

    def gaussian(self, x, mu, sigma):
        # print('x - my  ' + 89*"-", torch.min(-(x-mu)**2/(2*sigma**2)))
        prob = 1/sigma * torch.exp(-(x-mu)**2/(2*sigma**2)) + 10**-8
        return prob

    def log_prior(self, x):
        prob = torch.log(self.prior_weight * self.gaussian(x, 0, self.sigma_1_prior)
                            + (1-self.prior_weight)*self.gaussian(x, 0, self.sigma_2_prior))
        # print(torch.max(self.gaussian(x, 0, self.sigma_2_prior)))
        return prob

    def log_likelyhood(self, x, predicted):
        prob = (- torch.log(self.sigma_1_prior) -
               (x - predicted)**2 / (2 * self.sigma_1_prior**2))*self.prior_weight
        # prob += (- torch.log(self.sigma_2_prior) -\
        #        (x - predicted)**2 / (2 * self.sigma_2_prior**2))*(1-self.prior_weight)
        # max, index = torch.max(predicted, 1)
        # print(index)
        # prob = torch.log(torch.index_select(x, 1, index))
        return prob

    def log_posterior(self, x, mu, sigma):
        sigma = sigma
        prob = self.log_gaussian(x, mu, sigma)
        return prob

    def loss(self, input, output, target):
        log_qw, log_pw, log_likelyhood_gauss = 0, 0, 0
        for sample_idx in range(self.num_samples):
            if sample_idx > 0:
                output = self.forward(input)
                # print(output)
        # print("calculating loss...")
            log_pw += 1/self.num_samples*(self.log_prior(self.W1).sum() + self.log_prior(self.W2).sum() +
                                          self.log_prior(self.W3).sum())
            log_pw += 1/self.num_samples*(self.log_prior(self.b1).sum() + self.log_prior(self.b2).sum() +
                                          self.log_prior(self.b3).sum())
            # t1 = time.time()
            # print("prior value : ", log_pw.data)
            # print("calculated prior")
            # log_qw += 1/self.num_samples*(self.log_gaussian(self.W1, self.W1_mu, self.W1_rho).sum() +
            #          self.log_gaussian(self.W2, self.W2_mu, self.W2_rho).sum() +
            #                               self.log_gaussian(self.W3, self.W3_mu, self.W3_rho)).sum()
            # log_qw += 1/self.num_samples*(self.log_gaussian(self.b1, self.b1_mu, self.b1_rho).sum() +
            #          self.log_gaussian(self.b2, self.b2_mu, self.b2_rho).sum() +
            #                               self.log_gaussian(self.b3, self.b3_mu, self.b3_rho)).sum()

            log_qw += 1 / self.num_samples * (self.log_posterior(self.W1, self.W1_mu, self.W1_sigma).sum() +
                                              self.log_posterior(self.W2, self.W2_mu, self.W2_sigma).sum() +
                                              self.log_posterior(self.W3, self.W3_mu, self.W3_sigma).sum())
            log_qw += 1 / self.num_samples * (self.log_posterior(self.b1, self.b1_mu, self.b1_sigma).sum() +
                                              self.log_posterior(self.b2, self.b2_mu, self.b2_sigma).sum() +
                                              self.log_posterior(self.b3, self.b3_mu, self.b3_sigma).sum())         # t2 = time.time()
            # print("posterior :", log_qw.data)
            # print("calculated posterior")
            # print(target.size())
            # log_likelyhood += 1/self.num_samples * f.cross_entropy(output, target)
        # print("likelihood : ", log_likelyhood.data)
            log_likelyhood_gauss += 1/self.num_samples * self.log_likelyhood(output, target).sum()
            # t3 = time.time()
            # print('time prior ', t1-t0)
            # print('time post ', t2-t1)
            # print('time likely ', t3-t2)
        # print("calculated likelihood")
        # print('likelihood : ', log_likelyhood_gauss)
        # print('prior : ', log_pw)
        # print('post :', log_qw)
        # print("kl div : ",  1/num_batches * (log_qw - log_pw))
        loss = 1/batch_size * (1/num_batches * kl_weight * (log_qw - log_pw) - log_likelyhood_gauss)
        return loss

    def add_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

def preprocess_target(target):
    # print('start prepro')
    target_gauss = target.unsqueeze(1)
    target_hot = torch.FloatTensor(*target.size(), 10).zero_()
    target_hot.scatter_(1, target_gauss.data, 1)
    target_hot_variable = Variable(target_hot)
    # print('end prepro')
    return target_hot_variable

def train(model, epoch):
    log_interval = 60
    model.train()
    model.add_optimizer()
    for batch_idx, (data, target) in enumerate(train_loader):
        global kl_weight
        kl_weight = 2**(num_batches - batch_idx)/(2**num_batches - 1)
        # print("batch id : ", batch_idx)
        data, target = Variable(data), Variable(target)
        target = preprocess_target(target)
        model.optimizer.zero_grad()
        output = model(data)
        # print("found/ output")
        loss = model.loss(data, output, target)
        loss.backward()
        model.optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        target_gauss = preprocess_target(target)
        output = model(data)
        test_loss += model.loss(data, output, target_gauss).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({})\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__=="__main__":
    net = BayesBackpropMixGauss(10**-3, [50, 75])
    num_epochs = 100
    for epoch in range(num_epochs):
        train(net, epoch)
        test(net)

