import os
import shutil
import argparse
import datetime
import resource
import numpy as np
import gc
import torch
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('../')

from lyapunov.core import Manager
from lyapunov.utils import gpu_helper, save_weights, load_weights

from tqdm import tqdm
from IPython import embed

def parse_params():
    parser = argparse.ArgumentParser(description='GANs in PyTorch')
    parser.add_argument('-dom','--domain', type=str, default='MO8G', help='domain to run', required=False)
    parser.add_argument('-desc','--description', type=str, default='', help='description for the experiment', required=False)
    parser.add_argument('-alg','--algorithm', type=str, default='simgd', help='algorithm to use for training: simgd, cc, con, reg, reg_alt', required=False)
    parser.add_argument('-bs','--batch_size', type=int, default=512, help='batch_size for training', required=False)
    parser.add_argument('-div','--divergence', type=str, default='JS', help='divergence measure, i.e. V, for training', required=False)
    parser.add_argument('-d_opt','--disc_optim', type=str, default='Adam', help='discriminator training algorithm', required=False)
    parser.add_argument('-d_lr','--disc_learning_rate', type=float, default=1e-4, help='discriminator learning rate', required=False)
    parser.add_argument('-d_l2','--disc_weight_decay', type=float, default=0., help='discriminator weight decay', required=False)
    parser.add_argument('-d_nh','--disc_n_hidden', type=int, default=128, help='# of hidden units for discriminator', required=False)
    parser.add_argument('-d_nl','--disc_n_layer', type=int, default=1, help='# of hidden layers for discriminator', required=False)
    parser.add_argument('-d_nonlin','--disc_nonlinearity', type=str, default='relu', help='type of nonlinearity for discriminator', required=False)
    parser.add_argument('-d_quad','--disc_quadratic_layer', type=lambda x: (str(x).lower() == 'true'), default=False, help='whether to use a quadratic final layer', required=False)
    parser.add_argument('-g_opt','--gen_optim', type=str, default='Adam', help='generator training algorithm', required=False)
    parser.add_argument('-g_lr','--gen_learning_rate', type=float, default=1e-4, help='generator learning rate', required=False)
    parser.add_argument('-g_l2','--gen_weight_decay', type=float, default=0., help='generator weight decay', required=False)
    parser.add_argument('-g_nh','--gen_n_hidden', type=int, default=128, help='# of hidden units for generator', required=False)
    parser.add_argument('-g_nl','--gen_n_layer', type=int, default=2, help='# of hidden layers for generator', required=False)
    parser.add_argument('-g_nonlin','--gen_nonlinearity', type=str, default='relu', help='type of nonlinearity for generator', required=False)
    parser.add_argument('-betas','--betas', type=float, nargs=2, default=(0.5,0.999), help='beta params for Adam', required=False)
    parser.add_argument('-eps','--epsilon', type=float, default=1e-8, help='epsilon param for Adam', required=False)
    parser.add_argument('-mx_it','--max_iter', type=int, default=100001, help='max # of training iterations', required=False)
    parser.add_argument('-viz_every','--viz_every', type=int, default=1000, help='skip viz_every iterations between plotting current results', required=False)
    parser.add_argument('-series_every','--series_every', type=int, default=25000, help='skip series_every iterations between plotting series plot', required=False)
    parser.add_argument('-w_every','--weights_every', type=int, default=25000, help='skip weights_every iterations between saving weights', required=False)
    parser.add_argument('-n_viz','--n_viz', type=int, default=5120, help='number of samples for series plot', required=False)
    parser.add_argument('-zdim','--z_dim', type=int, default=256, help='dimensionality of p(z) - unit normal', required=False)
    parser.add_argument('-xdim','--x_dim', type=int, default=2, help='dimensionality of p(x) - data distribution', required=False)
    parser.add_argument('-gam','--gamma', type=float, default=10., help='gamma parameter for consensus, reg, reg_alt, and cc', required=False)
    parser.add_argument('-gamT','--gammaT', type=float, default=-1e11, help='gamma parameter for JTF in cc algorithm', required=False)
    parser.add_argument('-kap','--kappa', type=float, default=0., help='kappa parameter for F in cc algorithm', required=False)
    parser.add_argument('-step','--step', type=float, default=1e-3, help='step used to compute F(x_k+1) for cc', required=False)
    parser.add_argument('-saveto','--saveto', type=str, default='', help='path prefix for saving results', required=False)
    parser.add_argument('-gpu','--gpu', type=int, default=-2, help='if/which gpu to use (-1: all, -2: None)', required=False)
    parser.add_argument('-verb','--verbose', type=lambda x: (str(x).lower() == 'true'), default=False, help='whether to print progress to stdout', required=False)
    args = vars(parser.parse_args())

    if args['domain'] == 'MO8G':
        from examples.domains.synthetic import MOG_Circle as Domain
        from examples.domains.synthetic import Generator, Discriminator
    elif args['domain'] == 'MO25G':
        from examples.domains.synthetic import MOG_Grid as Domain
        from examples.domains.synthetic import Generator, Discriminator
    elif args['domain'] == 'SwissRoll':
        from examples.domains.synthetic import SwissRoll as Domain
        from examples.domains.synthetic import Generator, Discriminator
    elif 'Gaussian' in args['domain']:
        from examples.domains.synthetic import Gaussian as Domain
        if args['domain'][:2] == 'CL':
            from examples.domains.synthetic import Generator_C as Generator
            from examples.domains.synthetic import Discriminator_L as Discriminator
        elif args['domain'][:2] == 'LQ':
            from examples.domains.synthetic import Generator_L as Generator
            from examples.domains.synthetic import Discriminator_Q as Discriminator
    elif args['domain'] == 'MNIST':
        from examples.domains.mnist import MNIST as Domain
        from examples.domains.mnist import Generator, Discriminator
    elif args['domain'] == 'MNIST2':
        from examples.domains.mnist2 import MNIST as Domain
        from examples.domains.mnist2 import Generator, Discriminator
    elif args['domain'] == 'CIFAR10':
        from examples.domains.cifar10 import CIFAR10 as Domain
        from examples.domains.cifar10 import Generator, Discriminator
    else:
        raise NotImplementedError(args['domain'])

    if args['algorithm'] == 'simgd':
        from lyapunov.train_ops.train_simgd import SimGD as Train
    elif args['algorithm'] == 'cc':
        from lyapunov.train_ops.train_cc import CrossCurl as Train
        if args['gammaT'] < -1e10:
            args['gammaT'] == args['gamma']
    elif args['algorithm'] == 'consensus':
        # from lyapunov.train_ops.consensus import Consensus as Train
        from lyapunov.core import Train
    elif args['algorithm'] == 'reg':
        from lyapunov.train_ops.train_reg import Regularized as Train
    elif args['algorithm'] == 'reg_alt':
        from lyapunov.train_ops.train_reg_alt import Regularized_Alt as Train
    else:
        raise NotImplementedError(args['alg'])

    if args['saveto'] == '':
        args['saveto'] = 'examples/results/' + args['domain'] + '/' + args['algorithm'] + '/' + args['description']

    if args['description'] == '':
        args['description'] = args['domain'] + '-' + args['algorithm']
    elif args['description'].isdigit():
        args['description'] = args['domain'] + '-' + args['algorithm'] + '-' + args['description']

    saveto = args['saveto'] + '/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S/{}').format('')
    if not os.path.exists(saveto):
        os.makedirs(saveto)
    shutil.copy(os.path.realpath('lyapunov/run.py'), os.path.join(saveto, 'run.py'))
    shutil.copy(os.path.realpath('lyapunov/core.py'), os.path.join(saveto, 'core.py'))
    train_file = args['algorithm']+'.py'
    shutil.copy(os.path.realpath('lyapunov/train_ops/'+train_file), os.path.join(saveto, train_file))
    with open(saveto+'args.txt', 'w') as file:
        for key, val in args.items():
            file.write('--'+str(key)+' '+str(val)+'\n')
    args['saveto'] = saveto

    cuda_available = torch.cuda.is_available()
    if args['gpu'] >= -1 and cuda_available:
        torch.cuda.device(args['gpu'])
        args['description'] += ' (gpu'+str(torch.cuda.current_device())+')'
    else:
        args['description'] += ' (cpu)'

    from lyapunov.train_ops.consensus import Consensus
    from lyapunov.train_ops.rmsprop import RMSProp
    from lyapunov.train_ops.simgd import SimGD
    # args['maps'] = [Consensus, RMSProp, SimGD]
    args['maps'] = [RMSProp, Consensus, SimGD]
    # args['maps'] = [Consensus, SimGD]
    # args['maps'] = [RMSProp, SimGD]
    # args['maps'] = [SimGD]
    args['K'] = 2
    args['psi_epsilon'] = 1e-4
    args['gs_freq'] = 1
    args['start_lam_it'] = 4500

    # python lyapunov/run.py $(cat examples/args/MO8G/con/exp-con-25.txt) -alg consensus -verb True -bs 512 -d_lr 1e-3 -g_lr 1e-3 -gam 1 -d_nonlin leaky_relu -g_nonlin leaky_relu -mx_it 5001 -dom CLGaussian -xdim 1 -zdim 1 -mx_it 10000 -d_lr 1e-6 -g_lr 1e-6
    # python lyapunov/run.py $(cat examples/args/MO8G/con/exp-con-25.txt) -alg consensus -verb True -bs 512 -d_lr 1e-3 -g_lr 1e-3 -gam 1 -d_nonlin leaky_relu -g_nonlin leaky_relu -mx_it 5001

    return Train, Domain, Generator, Discriminator, args


def run_experiment(Train, Domain, Generator, Discriminator, params):
    to_gpu = gpu_helper(params['gpu'])

    data = Domain(dim=params['x_dim'])
    G = Generator(input_dim=params['z_dim'],output_dim=params['x_dim'],n_hidden=params['gen_n_hidden'],
                  n_layer=params['gen_n_layer'],nonlin=params['gen_nonlinearity'])
    D = Discriminator(input_dim=params['x_dim'],n_hidden=params['disc_n_hidden'],n_layer=params['disc_n_layer'],
                      nonlin=params['disc_nonlinearity'],quad=params['disc_quadratic_layer'])
    G.init_weights()
    D.init_weights()
    G = to_gpu(G)
    D = to_gpu(D)

    m = Manager(data, D, G, params, to_gpu)

    train = Train(manager=m)

    fs = []
    frames = []
    np_samples = []
    ds = [] # first gradients 
    gs = []
    viz_every = params['viz_every']

    iterations = range(params['max_iter'])
    if params['verbose']:
        iterations = tqdm(iterations,desc=params['description'])

    for i in iterations:
        
        lams, d, g, f = train.train_op(i)
        
        if params['verbose']:
            iterations.set_postfix({'Lambda':lams,'||F_D||^2':d,'||F_G||^2':g,'V':f})

        fs.append(f)
        ds.append(d)
        gs.append(g)

        if viz_every > 0 and i % viz_every == 0:
            if params['n_viz'] > 0:
                np_samples.append(train.m.get_fake(params['n_viz'], params['z_dim']).cpu().data.numpy())
            data.plot_current(train, params, i)

        if params['series_every'] > 0 and params['n_viz'] > 0 and i % params['series_every'] == 0:
            data.plot_series(np_samples, params)

        if params['weights_every'] > 0 and i % params['weights_every'] == 0:
            save_weights(m.D,params['saveto']+'D_'+str(i)+'.pkl')
            save_weights(m.G,params['saveto']+'G_'+str(i)+'.pkl')
                 

    np.savetxt(params['saveto']+'d_norm.out',np.array(ds))
    np.savetxt(params['saveto']+'g_norm.out',np.array(gs))
    np.savetxt(params['saveto']+'loss.out',np.array(fs))

    fig = plt.figure()
    ax = plt.subplot(111)
    plt.plot(range(len(ds)), ds)
    ax.set_ylabel('Discriminator Gradient L2 Norm')
    ax.set_xlabel('Iteration')
    fig.savefig(params['saveto']+'d_norm.pdf')

    fig = plt.figure()
    ax = plt.subplot(111)
    plt.plot(range(len(gs)), gs)
    ax.set_ylabel('Generator Gradient L2 Norm')
    ax.set_xlabel('Iteration')
    fig.savefig(params['saveto']+'g_norm.pdf')

    if params['n_viz'] > 0:
        data.plot_series(np_samples, params)

    fig = plt.figure()
    ax = plt.subplot(111)
    plt.plot(range(len(fs)),np.array(fs))
    ax.set_ylabel('Loss')
    ax.set_xlabel('Iteration')
    fig.savefig(params['saveto']+'loss.pdf')


if __name__ == '__main__':
    Train, Domain, Generator, Discriminator, params = parse_params()
    run_experiment(Train, Domain, Generator, Discriminator, params)
