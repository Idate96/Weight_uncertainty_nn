from torch import nn, optim
from torch.autograd import Variable
import torch

class Flatten(nn.Module):
    def forward(self, x):
        #         print("flattening tensor ", x.size())
        N, C, H, W = x.size()  # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image



def discriminator_func():
    """
    Build and return a PyTorch model for the DCGAN discriminator implementing
    the architecture above.
    """
    return nn.Sequential(
        # Unflatten(batch_size, 3, 32, 32),
        nn.Conv2d(3, 32, 5, stride=1),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(32, 32, 5, stride=1),
        nn.MaxPool2d(2, stride=2),
        Flatten(),
        nn.Linear(800, 800),
        nn.ReLU(),
        nn.Linear(800, 1)
    )
#
# def discriminator_func():
#     """
#     Build and return a PyTorch model for the DCGAN discriminator implementing
#     the architecture above.
#     """
#     return nn.Sequential(
#         # Unflatten(batch_size, 3, 32, 32),
#         nn.Conv2d(3, 32, 5, stride=1),
#         nn.MaxPool2d(2, stride=2),
#         nn.Conv2d(32, 64, 5, stride=1),
#         nn.MaxPool2d(2, stride=2),
#         Flatten(),
#         nn.Linear(1600, 1600),
#         nn.ReLU(),
#         nn.Linear(1600, 1)
#     )

def discriminator_loss(logits_real, logits_fake):
    """Calculate loss for discriminator
    objective : min : loss = - <log(d(x))>  - <log(1 - d(g(z))>
    x coming from data distribution and z from uniform noise distribution
    To do so we will employ the standard binary cross entropy loss :
    bce_loss = y * log(d(x)) + (1-y) * log(1 - d(g(z)))
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


def optimizer_discriminator(model):
    """Return optimizer"""
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.999))
    return optimizer
