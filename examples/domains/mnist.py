# https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py

import os
import urllib
import gzip
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lyapunov.core import Data, GNet, DNet

import matplotlib.pyplot as plt
import seaborn as sns

from IPython import embed
class MNIST(Data):
    def __init__(self, replace=True, **kwargs):
        super(MNIST, self).__init__()
        self.replace = replace
        self.data = self.load()
        self.N = self.data.shape[0]

    def sample(self, batch_size):
        chosen = np.random.choice(self.N, size=batch_size, replace=self.replace)
        samples = self.data[chosen]
        return torch.from_numpy(samples)

    def plot_current(self, train, params, i):
        samples = train.m.get_fake(1, params['z_dim']).view(-1, 28).cpu().data.numpy()
        images = samples
        plt.imshow(images, cmap=plt.get_cmap('binary'))
        plt.xticks([]); plt.yticks([])
        plt.savefig(params['saveto']+'samples_{}.png'.format(i))
        plt.close()

    def plot_series(self, np_samples, params):
        np_samples_ = np_samples[::1]
        cols = len(np_samples_)
        fig = plt.figure(figsize=(2*cols, 2*params['n_viz']))
        for i, samps in enumerate(np_samples_):
            if i == 0:
                ax = plt.subplot(1,cols,1)
            else:
                plt.subplot(1,cols,i+1, sharex=ax, sharey=ax)
            ax2 = plt.imshow(samps.reshape(-1,28), cmap=plt.get_cmap('binary'))
            plt.xticks([]); plt.yticks([])
            plt.title('step %d'%(i*params['viz_every']))
        plt.gcf().tight_layout()
        fig.savefig(params['saveto']+'series.pdf')
        plt.close()

    def load(self, path='examples/domains/mnist'):
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        filepath = path + '/mnist.pkl.gz'

        if not os.path.isfile(filepath):
            print("Couldn't find MNIST dataset in "+path+", downloading...")
            urllib.request.urlretrieve(url, filepath)

        with gzip.open(filepath, 'rb') as f:
            train_data, dev_data, test_data = pickle.load(f, encoding='latin1')

        images, targets = train_data

        return images


class Generator(GNet):
    def __init__(self, input_dim, n_hidden=64, output_dim=784, n_layer=None, nonlin=None):
        super(Generator, self).__init__()

        preprocess = nn.Sequential(
            nn.Linear(input_dim, 4*4*4*n_hidden),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4*n_hidden, 2*n_hidden, 5),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2*n_hidden, n_hidden, 5),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(n_hidden, 1, 8, stride=2)

        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

        self.input_dim = input_dim
        self.n_hidden = n_hidden
        self.output_dim = output_dim

    def forward(self, x):
        output = self.preprocess(x)
        output = output.view(-1, 4*self.n_hidden, 4, 4)
        output = self.block1(output)
        output = output[:, :, :7, :7]
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        return output.view(-1, self.output_dim)

    def init_weights(self):
        return None


class Discriminator(DNet):
    # assumption: output_dim = 1
    def __init__(self, input_dim, n_hidden=64, n_layer=None, nonlin=None, quad=False):
        super(Discriminator, self).__init__()

        main = nn.Sequential(
            nn.Conv2d(1, n_hidden, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(n_hidden, 2*n_hidden, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(2*n_hidden, 4*n_hidden, 5, stride=2, padding=2),
            nn.ReLU(True),
        )
        self.main = main
        self.output = nn.Linear(4*4*4*n_hidden, 1)

        self.input_dim = input_dim
        self.n_hidden = n_hidden

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        out = self.main(x)
        out = out.view(-1, 4*4*4*self.n_hidden)
        out = self.output(out)
        return out.view(-1)

    def init_weights(self):
        return None
