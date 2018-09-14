import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lyapunov.core import Data, GNet, DNet

from sklearn.datasets import make_swiss_roll

import matplotlib.pyplot as plt
import seaborn as sns


class Synthetic(Data):
    def __init__(self):
        super(Synthetic, self).__init__()

    def plot_current(self, train, params, i):
        xx = train.m.get_fake(params['batch_size'], params['z_dim']).cpu().data.numpy()
        yy = train.m.get_real(params['batch_size']).cpu().data.numpy()

        fig = plt.figure(figsize=(5,5))
        if xx.shape[1] == 1:
            xx1 = 0*xx[:,0]
            yy1 = 0*yy[:,0]
        else:
            xx1 = xx[:,1]
            yy1 = yy[:,1]
        plt.scatter(xx[:, 0], xx1, c='r', edgecolor='none', zorder=2)
        plt.scatter(yy[:, 0], yy1, c='k', edgecolor='none', zorder=1)
        ax = plt.gca()

        x = np.linspace(*ax.get_xlim())
        y = np.linspace(*ax.get_xlim())
        X, Y = np.meshgrid(x, y)
        if xx.shape[1] == 1:
            xy = X.reshape(-1,1)
        else:
            xy = np.hstack((X.reshape(-1,1),Y.reshape(-1,1)))
        decision = train.m.get_decisions([xy])[0].cpu().data.numpy()
        CS = plt.contour(X,Y,decision.reshape(X.shape), zorder=0)
        plt.clabel(CS, inline=1, fontsize=10)

        plt.axis('off')
        fig.savefig(params['saveto']+'fig'+str(i)+'.pdf') 
        plt.close(fig)

    def plot_series(self, np_samples, params):
        xmax = 3
        np_samples_ = np_samples[::1]
        cols = len(np_samples_)
        bg_color  = sns.color_palette('Greens', n_colors=256)[0]
        fig = plt.figure(figsize=(2*cols, 2))
        for i, samps in enumerate(np_samples_):
            if i == 0:
                ax = plt.subplot(1,cols,1)
            else:
                plt.subplot(1,cols,i+1, sharex=ax, sharey=ax)
            samps += 0.1*np.random.rand(*samps.shape) # to circumvent kdeplot error when samps is constant
            if samps.shape[1] == 1:
                samps1 = 0.1*np.random.rand(*samps.shape) # to circumvent kdeplot error when samps is constant
            else:
                samps1 = samps[:,1]
            ax2 = sns.kdeplot(samps[:, 0], samps1, shade=True, cmap='Greens', n_levels=20, clip=[[-xmax,xmax]]*2)
            ax2.set_facecolor(bg_color)
            ax2.set_xlim([-xmax,xmax])
            ax2.set_ylim([-xmax,xmax])
            plt.xticks([]); plt.yticks([])
            plt.title('step %d'%(i*params['viz_every']))
        plt.gcf().tight_layout()
        fig.savefig(params['saveto']+'series.pdf')


class SwissRoll(Synthetic):
    def __init__(self, noise=0.25, factor=7.5, **kwargs):
        super(SwissRoll, self).__init__()
        self.noise = noise
        self.factor = 7.5

    def sample(self, batch_size):
        samples = make_swiss_roll(n_samples=batch_size,noise=self.noise)[0]
        samples = samples.astype('float32')[:, [0, 2]]
        samples /= self.factor
        return torch.from_numpy(samples)


class Gaussian(Synthetic):
    def __init__(self, dim=2, Cov=None, **kwargs):
        super(Gaussian, self).__init__()
        if Cov is not None:
            assert len(Cov.shape) == 2
            assert np.linalg.norm(Cov-Cov.T) < 1e-20
            assert np.all(np.linalg.eigvals(Cov)>0.)
            self.Cov = Cov
            self.dim = Cov.shape[0]
        else:
            self.dim = dim
            L = 10*np.random.rand(dim,dim)
            L[range(dim),range(dim)] = np.clip(L[range(dim),range(dim)],1e-1,np.inf)
            L = np.tril(L)
            self.Cov = np.dot(L,L.T)
        if self.dim > 2:
            self.plot_current = lambda train, params, i: None
            self.plot_series = lambda np_samples, params: None
        self.mu = np.zeros(self.dim)

    def sample(self, batch_size):
        return torch.from_numpy(np.random.multivariate_normal(self.mu,self.Cov,size=batch_size).astype('float32'))


class MOG(Synthetic):
    def __init__(self, xs, ys, std):
        super(MOG, self).__init__()
        mus = [torch.from_numpy(np.array([xi,yi])) for xi,yi in zip(xs.ravel(), ys.ravel())]
        std = torch.from_numpy(np.array([std,std]))
        self.cat = torch.distributions.Categorical(probs=torch.ones(self.n_mixture))
        self.comps = [torch.distributions.Normal(loc=mu, scale=std) for mu in mus]

    def sample(self, batch_size):
        samples = []
        cat_onehot = torch.FloatTensor(1,self.n_mixture)
        for b in range(batch_size):
            cat_onehot.zero_()
            all_comps = torch.cat([c.sample().view(-1,1).float() for c in self.comps],dim=1)
            cat_onehot.scatter_(1,self.cat.sample().view(1,-1),1)
            samples += [torch.sum(all_comps*cat_onehot,dim=1,keepdim=True)]
        return torch.cat(samples,dim=1).t()


class MOG_Circle(MOG):
    def __init__(self, n_mixture=8, std=0.02, radius=2.0, **kwargs):
        self.n_mixture = n_mixture
        self.std = std
        self.radius = radius
        thetas = np.linspace(0, 2 * np.pi * (n_mixture-1)/float(n_mixture), n_mixture)
        xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
        super(MOG_Circle, self).__init__(xs, ys, std)


class MOG_Grid(MOG):
    def __init__(self, n_mixture=25, **kwargs):
        self.n_mixture = n_mixture
        rng = int(np.sqrt(n_mixture))
        xs, ys = np.linspace(-rng/2.828, rng/2.828, rng), np.linspace(-rng/2.828, rng/2.828, rng)
        X, Y = np.meshgrid(xs, ys)
        xs, ys = X.ravel(), Y.ravel()
        std = 0.05/2.828
        super(MOG_Grid, self).__init__(xs, ys, std)


class Generator(GNet):
    def __init__(self, input_dim, n_hidden=128, output_dim=2, n_layer=2, nonlin='relu'):
        super(Generator, self).__init__()
        hidden_fcs = []
        in_dim = input_dim
        for l in range(n_layer):
            hidden_fcs += [nn.Linear(in_dim, n_hidden)]
            in_dim = n_hidden
        self.hidden_fcs = nn.ModuleList(hidden_fcs)
        self.final_fc = nn.Linear(in_dim, output_dim)
        self.layers = hidden_fcs + [self.final_fc]
        if nonlin == 'leaky_relu':
            self.nonlin = F.leaky_relu
        elif nonlin == 'relu':
            self.nonlin = F.relu
        elif nonlin == 'tanh':
            self.nonlin = F.tanh
        elif nonlin == 'sigmoid':
            self.nonlin = F.sigmoid
        else:
            self.nonlin = lambda x: x

    def forward(self,x):
        h = x
        for hfc in self.hidden_fcs:
            h = self.nonlin(hfc(h))
        return self.final_fc(h)

    def init_weights(self):
        for layer in self.layers:
            nn.init.orthogonal_(layer.weight.data, gain=0.8)
            # layer.weight.data.normal_(0.0, 0.02)
            layer.bias.data.zero_()


class Discriminator(DNet):
    # assumption: output_dim = 1
    def __init__(self, input_dim, n_hidden=128, n_layer=1, nonlin='relu', quad=False):
        super(Discriminator, self).__init__()
        hidden_fcs = []
        in_dim = input_dim
        for l in range(n_layer):
            hidden_fcs += [nn.Linear(in_dim, n_hidden)]
            in_dim = n_hidden
        self.hidden_fcs = nn.ModuleList(hidden_fcs)
        self.quad = quad
        if quad:
            self.final_fc = nn.Linear(in_dim, in_dim)  #W z + b
        else:
            self.final_fc = nn.Linear(in_dim, 1)
        self.layers = hidden_fcs + [self.final_fc]
        if nonlin == 'leaky_relu':
            self.nonlin = F.leaky_relu
        elif nonlin == 'relu':
            self.nonlin = F.relu
        elif nonlin == 'tanh':
            self.nonlin = F.tanh
        elif nonlin == 'sigmoid':
            self.nonlin = F.sigmoid
        else:
            self.nonlin = lambda x: x

    def forward(self,x):
        # h = x/4.0
        h = x
        for hfc in self.hidden_fcs:
            h = self.nonlin(hfc(h))
        if self.quad:
            return torch.sum(h*self.final_fc(h),dim=1)
        else:
            return self.final_fc(h)

    def init_weights(self):
        for layer in self.layers:
            nn.init.orthogonal_(layer.weight.data, gain=0.8)
            layer.bias.data.zero_()


class Generator_C(GNet):
    def __init__(self, input_dim, n_hidden=128, output_dim=2, n_layer=2, nonlin='relu'):
        super(Generator_C, self).__init__()
        assert input_dim == output_dim
        # b0 = -1+2*np.random.rand(output_dim).astype('float32')
        b0 = -20*np.ones(output_dim).astype('float32')
        self.b = nn.Parameter(torch.from_numpy(b0))
            
    def forward(self,x):
        return self.b*torch.ones_like(x)

    def init_weights(self):
        pass


class Discriminator_L(DNet):
    # assumption: output_dim = 1
    def __init__(self, input_dim, n_hidden=128, n_layer=1, nonlin='relu', quad=False):
        super(Discriminator_L, self).__init__()
        # self.w1 = nn.Parameter(torch.zeros(input_dim))
        self.w1 = nn.Parameter(10*torch.ones(input_dim))

    def forward(self,x):
        return torch.matmul(x, self.w1)

    def init_weights(self):
        pass


class Generator_L(GNet):
    def __init__(self, input_dim, n_hidden=128, output_dim=2, n_layer=2, nonlin='relu'):
        super(Generator_L, self).__init__()
        assert input_dim == output_dim
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self,x):
        return self.linear(x)

    def init_weights(self):
        nn.init.orthogonal_(self.linear.weight.data, gain=0.8)
        self.linear.bias.data = torch.from_numpy(-1+2*np.random.rand(output_dim).astype('float32'))


class Discriminator_Q(DNet):
    # assumption: output_dim = 1
    def __init__(self, input_dim, n_hidden=128, n_layer=1, nonlin='relu', quad=False):
        super(Discriminator_Q, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self,x):
        return torch.sum(x*self.linear(x),dim=1)

    def init_weights(self):
        nn.init.orthogonal_(self.linear.weight.data, gain=0.8)
        self.linear.bias.data.zero_()
