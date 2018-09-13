# https://github.com/LMescheder/TheNumericsOfGANs/blob/master/consensus_gan/models/resnet_cf.py
# https://github.com/LMescheder/TheNumericsOfGANs/blob/master/experiments/celebA/conopt.py

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


class CELEBA(Data):
    def __init__(self, replace=True, **kwargs):
        super(CELEBA, self).__init__()
        self.data = self.load()
        self.dataiter = iter(self.data)

    def sample(self, batch_size):
        try:
            return self.dataiter.next()[0].view(batch_size,-1)
        except:
            self.dataiter = iter(self.data)
            return self.dataiter.next()[0].view(batch_size,-1)

    def plot_current(self, train, params, i):
        images = train.m.get_fake(64, params['z_dim']).view(-1, 3, 64, 64)
        img = torchvision.utils.make_grid(images)
        img = img / 2 + 0.5     # unnormalize
        plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
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
            thissamp = samps.reshape((params['n_viz'],3,64,64)).transpose((0,2,3,1))
            ax2 = plt.imshow(thissamp.reshape(-1, 64, 3))
            plt.xticks([]); plt.yticks([])
            plt.title('step %d'%(i*params['viz_every']))
        plt.gcf().tight_layout()
        fig.savefig(params['saveto']+'series.pdf')
        plt.close()

    def load(self, path='examples/domains/celebA'):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.ImageFolder(root='./examples/domains/celebA', transform=transform)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                                  shuffle=True, num_workers=2)

        return trainloader


class Generator(GNet):
    def __init__(self, input_dim, n_hidden=128, output_dim=64, n_layer=None, nonlin=None):
        super(Generator, self).__init__()
        n_hidden = 128
        preprocess = nn.Linear(input_dim, output_dim//16 * output_dim//16 * n_hidden)

        self.convt_1 = nn.ConvTranspose2d(n_hidden, n_hidden, kernel_size=[5,5], stride=[2,2], padding=2, output_padding=1)
        self.conv_1a = nn.Conv2d(n_hidden, n_hidden//2, kernel_size=[3,3], stride=[1,1], padding=1)
        self.conv_1b = nn.Conv2d(n_hidden//2, n_hidden, kernel_size=[3,3], stride=[1,1], padding=1)

        self.convt_2 = nn.ConvTranspose2d(n_hidden, n_hidden, kernel_size=[5,5], stride=[2,2], padding=2, output_padding=1)
        self.conv_2a = nn.Conv2d(n_hidden, n_hidden//2, kernel_size=[3,3], stride=[1,1], padding=1)
        self.conv_2b = nn.Conv2d(n_hidden//2, n_hidden, kernel_size=[3,3], stride=[1,1], padding=1)

        self.convt_3 = nn.ConvTranspose2d(n_hidden, n_hidden, kernel_size=[5,5], stride=[2,2], padding=2, output_padding=1)
        self.conv_3a = nn.Conv2d(n_hidden, n_hidden//2, kernel_size=[3,3], stride=[1,1], padding=1)
        self.conv_3b = nn.Conv2d(n_hidden//2, n_hidden, kernel_size=[3,3], stride=[1,1], padding=1)

        self.convt_4 = nn.ConvTranspose2d(n_hidden, 3, kernel_size=[5,5], stride=[2,2], padding=2, output_padding=1)
        
        self.preprocess = preprocess
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.tanh = nn.Tanh()

        self.input_dim = input_dim
        self.n_hidden = n_hidden
        self.output_dim = output_dim

    def forward(self, x):
        output = self.preprocess(x)
        output = output.view(-1, self.n_hidden, self.output_dim//16, self.output_dim//16)

        print(output.shape)
        output = self.convt_1(self.lrelu(output))
        print(output.shape)
        doutput = self.conv_1a(self.lrelu(output))
        print(doutput.shape)
        output = output + 1e-1*self.conv_1b(self.lrelu(doutput))

        print(output.shape)
        output = self.convt_2(self.lrelu(output))
        print(output.shape)
        doutput = self.conv_2a(self.lrelu(output))
        print(doutput.shape)
        output = output + 1e-1*self.conv_2b(self.lrelu(doutput))

        print(output.shape)
        output = self.convt_3(self.lrelu(output))
        print(output.shape)
        doutput = self.conv_3a(self.lrelu(output))
        print(doutput.shape)
        output = output + 1e-1*self.conv_3b(self.lrelu(doutput))

        print(output.shape)
        output = self.convt_4(self.lrelu(output))

        print(output.shape)
        output = self.tanh(output)
        return output.view(-1, 3*self.output_dim**2)

    def init_weights(self):
        nn.init.xavier_uniform_(self.preprocess.weight.data, gain=1)
        self.preprocess.bias.data.zero_()
        for layer in [self.convt_1,self.conv_1a,self.conv_1b,
                      self.convt_2,self.conv_2a,self.conv_2b,
                      self.convt_3,self.conv_3a,self.conv_3b,
                      self.convt_4]:
            if hasattr(layer,'weight'):
                nn.init.xavier_uniform_(layer.weight.data, gain=1)
                layer.bias.data.zero_()


class Discriminator(DNet):
    # assumption: output_dim = 1
    def __init__(self, input_dim, output_dim=64, n_hidden=128, n_layer=None, nonlin=None, quad=False):
        super(Discriminator, self).__init__()
        n_hidden = 128

        self.conv_1a = nn.Conv2d(3, n_hidden, kernel_size=[5,5], stride=[2,2], padding=2, output_padding=1)
        self.conv_1b = nn.Conv2d(3, n_hidden, kernel_size=[3,3], stride=[1,1], padding=1)
        self.conv_1c = nn.Conv2d(n_hidden, n_hidden, kernel_size=[3,3], stride=[1,1], padding=1)

        self.conv_2a = nn.Conv2d(n_hidden, n_hidden, kernel_size=[5,5], stride=[2,2], padding=2, output_padding=1)
        self.conv_2b = nn.Conv2d(n_hidden, n_hidden//2, kernel_size=[3,3], stride=[1,1], padding=1)
        self.conv_2c = nn.Conv2d(n_hidden//2, n_hidden, kernel_size=[3,3], stride=[1,1], padding=1)

        self.conv_3a = nn.Conv2d(n_hidden, n_hidden, kernel_size=[5,5], stride=[2,2], padding=2, output_padding=1)
        self.conv_3b = nn.Conv2d(n_hidden, n_hidden//2, kernel_size=[3,3], stride=[1,1], padding=1)
        self.conv_3c = nn.Conv2d(n_hidden//2, n_hidden, kernel_size=[3,3], stride=[1,1], padding=1)

        self.conv_4a = nn.Conv2d(n_hidden, n_hidden, kernel_size=[5,5], stride=[2,2], padding=2, output_padding=1)
        self.conv_4b = nn.Conv2d(n_hidden, n_hidden//2, kernel_size=[3,3], stride=[1,1], padding=1)
        self.conv_4c = nn.Conv2d(n_hidden//2, n_hidden, kernel_size=[3,3], stride=[1,1], padding=1)

        output = nn.Linear(output_dim//16 * output_dim//16 * n_hidden, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

        self.output = output
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden = n_hidden

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        print(x.shape)

        out = self.conv_1a(x)
        print(out.shape)
        dout = self.conv_1b(self.lrelu(out))
        print(dout.shape)
        out += 1e-1*self.conv_1c(self.lrelu(dout))
        print(out.shape)

        out = self.conv_2a(x)
        print(out.shape)
        dout = self.conv_2b(self.lrelu(out))
        print(dout.shape)
        out += 1e-1*self.conv_2c(self.lrelu(dout))
        print(out.shape)

        out = self.conv_3a(x)
        print(out.shape)
        dout = self.conv_3b(self.lrelu(out))
        print(dout.shape)
        out += 1e-1*self.conv_3c(self.lrelu(dout))
        print(out.shape)

        out = self.conv_4a(x)
        print(out.shape)
        dout = self.conv_4b(self.lrelu(out))
        print(dout.shape)
        out += 1e-1*self.conv_4c(self.lrelu(dout))
        print(out.shape)

        out = out.view(-1, self.output_dim//16 * self.output_dim//16 * self.n_hidden)
        out = self.output(out)
        return out.view(-1)

    def init_weights(self):
        for layer in [self.conv_1a,self.conv_1b,self.conv_1c,
                      self.conv_2a,self.conv_2b,self.conv_2c,
                      self.conv_3a,self.conv_3b,self.conv_3c,
                      self.conv_4a,self.conv_4b,self.conv_4c]:
            if hasattr(layer,'weight'):
                nn.init.xavier_uniform_(layer.weight.data, gain=1)
                layer.bias.data.zero_()
        nn.init.xavier_uniform_(self.output.weight.data, gain=1)
        self.output.bias.data.zero_()
