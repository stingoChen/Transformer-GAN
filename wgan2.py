import argparse
import os
import numpy as np
import math
import sys
from data import data
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("model", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=10000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
# parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
# parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
# parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
# parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=14, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.09, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

# img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.fn = nn.Linear(11, 32)
        self.fn1 = nn.Linear(32, 64)
        self.fn2 = nn.Linear(64, 32)
        self.fn3 = nn.Linear(32, 11)

    def forward(self, input_):
        output_ = self.fn(input_)
        output_ = self.relu(output_)

        output_ = self.fn1(output_)
        output_ = self.relu(output_)

        output_ = self.fn2(output_)
        output_ = self.relu(output_)

        output_ = self.fn3(output_)
        output_ = self.sigmoid(output_)
        return output_


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.fn = nn.Linear(11, 64)
        self.fn1 = nn.Linear(64, 128)
        self.fn2 = nn.Linear(128, 64)
        self.fn3 = nn.Linear(64, 6)

        self.fn4 = nn.Linear(6, 32)

        self.fn5 = nn.Linear(32, 64)
        self.fn6 = nn.Linear(64, 12)

    def forward(self, input_):
        output_ = self.fn(input_)
        output_ = self.relu(output_)

        output_ = self.fn1(output_)
        output_ = self.relu(output_)

        output_ = self.fn2(output_)
        output_ = self.relu(output_)

        output_ = self.fn3(output_)
        output_ = self.relu(output_)

        output_ = self.fn4(output_)
        output_ = self.relu(output_)

        output_ = self.fn5(output_)
        output_ = self.relu(output_)

        output_ = self.fn6(output_)
        return output_


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# Configure data loader
train_ds = data.train_data
dataloader = torch.utils.data.DataLoader(train_ds, batch_size=opt.batch_size, drop_last=True, shuffle=True)

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

mse_loss = nn.MSELoss()

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(opt.n_epochs):

    for i, data in enumerate(dataloader):

        # Configure input
        real_data = Variable(data.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()
        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, 11))))

        # Generate a batch of images
        fake_data = generator(z).detach()
        # Adversarial loss
        x1 = discriminator(real_data)
        x2 = discriminator(fake_data)
        loss_D = -torch.mean(x1[:, 0]) + torch.mean(x2[:, 0])
        mse_ = mse_loss(x1[:, 1:], real_data) + mse_loss(x1[:, 1:], fake_data)

        loss = loss_D + mse_
        loss.backward()
        optimizer_D.step()

        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)

        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:
            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_data = generator(z)
            # Adversarial loss
            # print(discriminator(gen_data))
            loss_G = -torch.mean(discriminator(gen_data)[:, 0])

            loss_G.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
            )

            batches_done += 1

torch.save({
    'netG_state_dict': generator.state_dict(),
    'netD_state_dict': discriminator.state_dict(),
    'optimizerD_state_dict': optimizer_D.state_dict(),
    'optimizerG_state_dict': optimizer_G.state_dict()},
    "./model/model%d" % opt.n_epochs)
