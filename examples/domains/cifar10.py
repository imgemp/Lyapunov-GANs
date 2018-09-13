# https://github.com/LMescheder/TheNumericsOfGANs/blob/master/consensus_gan/models/dcgan4_nobn_cf.py

import os
import urllib
import gzip
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from lyapunov.core import Data, GNet, DNet

import matplotlib.pyplot as plt
import seaborn as sns


class CIFAR10(Data):
    def __init__(self, replace=True, **kwargs):
        super(CIFAR10, self).__init__()
        self.data = self.load()
        self.dataiter = iter(self.data)

    def sample(self, batch_size):
        try:
            return self.dataiter.next()[0].view(batch_size,-1)
        except:
            self.dataiter = iter(self.data)
            return self.dataiter.next()[0].view(batch_size,-1)

    def plot_current(self, train, params, i):
        # images = train.m.get_fake(64, params['z_dim']).view(3, 32, 32).cpu().data.numpy()
        images = train.m.get_fake(64, params['z_dim']).view(-1, 3, 32, 32)
        img = torchvision.utils.make_grid(images)
        img = img / 2 + 0.5     # unnormalize
        plt.imshow(np.transpose(img.cpu().data.numpy(), (1, 2, 0)))
        plt.xticks([]); plt.yticks([])
        plt.savefig(params['saveto']+'samples_{}.png'.format(i))
        plt.close()

    def plot_series(self, np_samples, params):
        np_samples_ = np.array(np_samples) / 2 + 0.5
        # np_samples_ = np_samples[::1]  # np_samples: viz_every x n_viz x 3*32*32
        # np_samples_ = np.array(np_samples).reshape((len(np_samples_),))
        # np_samples_ = np.array(np_samples_) / 2 + 0.5
        # np_samples_ = np.transpose(np_samples_, (1, 2, 0))
        cols = len(np_samples_)
        fig = plt.figure(figsize=(2*cols, 2*params['n_viz']))
        for i, samps in enumerate(np_samples_):
            if i == 0:
                ax = plt.subplot(1,cols,1)
            else:
                plt.subplot(1,cols,i+1, sharex=ax, sharey=ax)
            thissamp = samps.reshape((params['n_viz'],3,32,32)).transpose((0,2,3,1))
            ax2 = plt.imshow(thissamp.reshape(-1, 32, 3))
            plt.xticks([]); plt.yticks([])
            plt.title('step %d'%(i*params['viz_every']))
        plt.gcf().tight_layout()
        fig.savefig(params['saveto']+'series.pdf')
        plt.close()

    def load(self, path='examples/domains/cifar10'):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./examples/domains/cifar10', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                                  shuffle=True, num_workers=2)

        return trainloader


class Generator(GNet):
    def __init__(self, input_dim, n_hidden=64, output_dim=32, n_layer=None, nonlin=None):
        super(Generator, self).__init__()
        n_hidden = 64
        preprocess = nn.Linear(input_dim, output_dim//16 * output_dim//16 * n_hidden)

        deconv = nn.Sequential(
            nn.ConvTranspose2d(n_hidden, n_hidden, kernel_size=[5,5], stride=[2,2], padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True,negative_slope=0.2),
            nn.ConvTranspose2d(n_hidden, n_hidden, kernel_size=[5,5], stride=[2,2], padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True,negative_slope=0.2),
            nn.ConvTranspose2d(n_hidden, n_hidden, kernel_size=[5,5], stride=[2,2], padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True,negative_slope=0.2),
            nn.ConvTranspose2d(n_hidden, 3, kernel_size=[5,5], stride=[2,2], padding=2, output_padding=1)
        )
        
        self.preprocess = preprocess
        self.deconv = deconv
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.tanh = nn.Tanh()

        self.input_dim = input_dim
        self.n_hidden = n_hidden
        self.output_dim = output_dim

    def forward(self, x):
        output = self.preprocess(x)
        output = output.view(-1, self.n_hidden, self.output_dim//16, self.output_dim//16)
        output = self.lrelu(output)
        output = self.deconv(output)
        output = self.tanh(output)
        return output.view(-1, 3*self.output_dim**2)

    def init_weights(self):
        nn.init.xavier_uniform_(self.preprocess.weight.data, gain=1)
        self.preprocess.bias.data.zero_()
        for layer in self.deconv:
            if hasattr(layer,'weight'):
                nn.init.xavier_uniform_(layer.weight.data, gain=1)
                layer.bias.data.zero_()


class Discriminator(DNet):
    # assumption: output_dim = 1
    def __init__(self, input_dim, output_dim=32, n_hidden=64, n_layer=None, nonlin=None, quad=False):
        super(Discriminator, self).__init__()
        n_hidden = 64
        main = nn.Sequential(
            nn.Conv2d(3, n_hidden, kernel_size=[5,5], stride=[2,2], padding=2),
            nn.LeakyReLU(inplace=True,negative_slope=0.2),
            nn.Conv2d(n_hidden, n_hidden, kernel_size=[5,5], stride=[2,2], padding=2),
            nn.LeakyReLU(inplace=True,negative_slope=0.2),
            nn.Conv2d(n_hidden, n_hidden, kernel_size=[5,5], stride=[2,2], padding=2),
            nn.LeakyReLU(inplace=True,negative_slope=0.2),
            nn.Conv2d(n_hidden, n_hidden, kernel_size=[5,5], stride=[2,2], padding=2),
            nn.LeakyReLU(inplace=True,negative_slope=0.2)
        )

        output = nn.Linear(output_dim//16 * output_dim//16 * n_hidden, 1)

        self.main = main
        self.output = output
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden = n_hidden

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        out = self.main(x)
        out = out.view(-1, self.output_dim//16 * self.output_dim//16 * self.n_hidden)
        out = self.output(out)
        return out.view(-1)

    def init_weights(self):
        for layer in self.main:
            if hasattr(layer,'weight'):
                nn.init.xavier_uniform_(layer.weight.data, gain=1)
                layer.bias.data.zero_()
        nn.init.xavier_uniform_(self.output.weight.data, gain=1)
        self.output.bias.data.zero_()
