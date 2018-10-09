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


class MNIST(Data):
    def __init__(self, replace=True, batch_size=64, **kwargs):
        super(MNIST, self).__init__()
        self.data = self.load(batch_size)
        self.dataiter = iter(self.data)
        self.batch_size = batch_size
        self.warned = False

    def sample(self, batch_size):
        q, r = divmod(batch_size, self.batch_size)
        if (q < 1 or r != 0) and not self.warned:
            print('WARNING: Only positive integer multiples of {} allowed'.format(self.batch_size))
            self.warned = True
        n_batches = max(1, q)
        batches = []
        for b in range(n_batches):
            try:
                batches += [self.dataiter.next()[0].view(batch_size,-1)]
            except:
                self.dataiter = iter(self.data)
                batches += [self.dataiter.next()[0].view(batch_size,-1)]
        if n_batches > 1:
            return torch.cat(batches, dim=0)
        else:
            return batches[0]

    def plot_current(self, train, params, i):
        images = train.m.get_fake(64, params['z_dim']).view(-1, 1, 28, 28)
        img = torchvision.utils.make_grid(images)
        img = img / 2 + 0.5     # unnormalize
        plt.imshow(np.transpose(img.cpu().data.numpy(), (1, 2, 0)))
        plt.xticks([]); plt.yticks([])
        plt.savefig(params['saveto']+'samples_{}.png'.format(i))
        plt.close()

    def plot_series(self, np_samples, params):
        np_samples_ = np.array(np_samples) / 2 + 0.5
        cols = len(np_samples_)
        fig = plt.figure(figsize=(2*cols, 2*params['n_viz']))
        for i, samps in enumerate(np_samples_):
            if i == 0:
                ax = plt.subplot(1,cols,1)
            else:
                plt.subplot(1,cols,i+1, sharex=ax, sharey=ax)
            thissamp = samps.reshape((params['n_viz'],1,28,28)).transpose((0,2,3,1))
            ax2 = plt.imshow(thissamp.reshape(-1,28), cmap=plt.get_cmap('binary'))
            plt.xticks([]); plt.yticks([])
            plt.title('step %d'%(i*params['viz_every']))
        plt.gcf().tight_layout()
        fig.savefig(params['saveto']+'series.pdf')
        plt.close()

    def load(self, path='examples/domains/mnist2', batch_size=64):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.MNIST(root='./examples/domains/mnist2', train=True,
                                              download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)

        return trainloader


class Generator(GNet):
    def __init__(self, input_dim, n_hidden=64, output_dim=28, n_layer=None, nonlin=None):
        super(Generator, self).__init__()
        n_hidden = 64
        preprocess = nn.Linear(input_dim, output_dim//14 * output_dim//14 * n_hidden)

        deconv = nn.Sequential(
            nn.ConvTranspose2d(n_hidden, n_hidden, kernel_size=[5,5], stride=[2,2], padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True,negative_slope=0.2),
            nn.ConvTranspose2d(n_hidden, n_hidden, kernel_size=[5,5], stride=[2,2], padding=2, output_padding=0),
            nn.LeakyReLU(inplace=True,negative_slope=0.2),
            nn.ConvTranspose2d(n_hidden, n_hidden, kernel_size=[5,5], stride=[2,2], padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True,negative_slope=0.2),
            nn.ConvTranspose2d(n_hidden, 1, kernel_size=[5,5], stride=[2,2], padding=2, output_padding=1)
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
        output = output.view(-1, self.n_hidden, self.output_dim//14, self.output_dim//14)
        output = self.lrelu(output)
        output = self.deconv(output)
        output = self.tanh(output)
        return output.view(-1, self.output_dim**2)

    def init_weights(self, filepath=''):
        try:
            assert filepath != ''
            self.init_weights_from_file(filepath)
        except:
            nn.init.xavier_uniform_(self.preprocess.weight.data, gain=1)
            self.preprocess.bias.data.zero_()
            for layer in self.deconv:
                if hasattr(layer,'weight'):
                    nn.init.xavier_uniform_(layer.weight.data, gain=1)
                    layer.bias.data.zero_()


class Discriminator(DNet):
    # assumption: output_dim = 1
    def __init__(self, input_dim, output_dim=28, n_hidden=64, n_layer=None, nonlin=None, quad=False):
        super(Discriminator, self).__init__()
        n_hidden = 64
        main = nn.Sequential(
            nn.Conv2d(1, n_hidden, kernel_size=[5,5], stride=[2,2], padding=2),
            nn.LeakyReLU(inplace=True,negative_slope=0.2),
            nn.Conv2d(n_hidden, n_hidden, kernel_size=[5,5], stride=[2,2], padding=2),
            nn.LeakyReLU(inplace=True,negative_slope=0.2),
            nn.Conv2d(n_hidden, n_hidden, kernel_size=[5,5], stride=[2,2], padding=2),
            nn.LeakyReLU(inplace=True,negative_slope=0.2),
            nn.Conv2d(n_hidden, n_hidden, kernel_size=[5,5], stride=[2,2], padding=2),
            nn.LeakyReLU(inplace=True,negative_slope=0.2)
        )

        output = nn.Linear(output_dim//14 * output_dim//14 * n_hidden, 1)

        self.main = main
        self.output = output
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden = n_hidden

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        out = self.main(x)
        out = out.view(-1, self.output_dim//14 * self.output_dim//14 * self.n_hidden)
        out = self.output(out)
        return out.view(-1)

    def init_weights(self, filepath=''):
        try:
            assert filepath != ''
            self.init_weights_from_file(filepath)
        except:
            for layer in self.main:
                if hasattr(layer,'weight'):
                    nn.init.xavier_uniform_(layer.weight.data, gain=1)
                    layer.bias.data.zero_()
            nn.init.xavier_uniform_(self.output.weight.data, gain=1)
            self.output.bias.data.zero_()
