import os
import psutil
import shutil
import argparse
import datetime
import resource
import pickle
import numpy as np
import torch
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import seaborn as sns

from sklearn.decomposition import IncrementalPCA

import sys
sys.path.append('../')

from lyapunov.core import Manager
from lyapunov.utils import gpu_helper, save_weights, load_weights

from tqdm import tqdm

from IPython import embed


process = psutil.Process(os.getpid())

def parse_params():
    parser = argparse.ArgumentParser(description='GANs in PyTorch')
    parser.add_argument('-dom','--domain', type=str, default='MO8G', help='domain to run', required=False)
    parser.add_argument('-desc','--description', type=str, default='', help='description for the experiment', required=False)
    parser.add_argument('-bs','--batch_size', type=int, default=512, help='batch_size for training', required=False)
    parser.add_argument('-div','--divergence', type=str, default='JS', help='divergence measure, i.e. V, for training', required=False)
    parser.add_argument('-d_lr','--disc_learning_rate', type=float, default=1e-4, help='discriminator learning rate', required=False)
    parser.add_argument('-d_l2','--disc_weight_decay', type=float, default=0., help='discriminator weight decay', required=False)
    parser.add_argument('-d_nh','--disc_n_hidden', type=int, default=128, help='# of hidden units for discriminator', required=False)
    parser.add_argument('-d_nl','--disc_n_layer', type=int, default=1, help='# of hidden layers for discriminator', required=False)
    parser.add_argument('-d_nonlin','--disc_nonlinearity', type=str, default='relu', help='type of nonlinearity for discriminator', required=False)
    parser.add_argument('-d_quad','--disc_quadratic_layer', type=lambda x: (str(x).lower() == 'true'), default=False, help='whether to use a quadratic final layer', required=False)
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
    parser.add_argument('-maps','--map_strings', type=str, nargs='+', default=[''], help='string names of optimizers to use for generator and discriminator', required=False)
    parser.add_argument('-gam','--gamma', type=float, default=10., help='gamma parameter for consensus, reg, reg_alt, and cc', required=False)
    parser.add_argument('-gamT','--gammaT', type=float, default=-1e11, help='gamma parameter for JTF in cc algorithm', required=False)
    parser.add_argument('-kap','--kappa', type=float, default=0., help='kappa parameter for F in cc algorithm', required=False)
    parser.add_argument('-K','--K', type=int, default=2, help='number of lyapunov exponents to compute', required=False)
    parser.add_argument('-psi_epsilon','--psi_epsilon', type=float, default=0., help='epsilon to use for finite difference approximation of Jacobian vector product', required=False)
    parser.add_argument('-LE_freq','--LE_freq', type=int, default=5, help='number of steps to wait inbetween computing LEs', required=False)
    parser.add_argument('-LE_batch_mult','--LE_batch_mult', type=int, default=10, help='batch_size multiplier to reduce variance when computing LEs', required=False)
    parser.add_argument('-start_lam_it','--start_lam_it', type=int, default=-1, help='number of steps to wait inbetween computing LEs', required=False)
    parser.add_argument('-det','--deterministic', type=lambda x: (str(x).lower() == 'true'), default=False, help='whether to compute loss always using same samples', required=False)
    parser.add_argument('-saveto','--saveto', type=str, default='', help='path prefix for saving results', required=False)
    parser.add_argument('-gpu','--gpu', type=int, default=-2, help='if/which gpu to use (-1: all, -2: None)', required=False)
    parser.add_argument('-verb','--verbose', type=lambda x: (str(x).lower() == 'true'), default=False, help='whether to print progress to stdout', required=False)
    args = vars(parser.parse_args())

    if args['psi_epsilon'] <= 0.:
        args['psi_epsilon'] = 0.1*min(args['disc_learning_rate'],args['gen_learning_rate'])
    if args['start_lam_it'] < 0.:
        args['start_lam_it'] = int(0.9*args['max_iter'])

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

    from lyapunov.core import Train
    args['maps'] = []
    for mp in args['map_strings']:
        if mp.lower() == 'consensus':
            from lyapunov.train_ops.consensus import Consensus
            args['maps'] += [Consensus]
        elif mp.lower() == 'rmsprop':
            from lyapunov.train_ops.rmsprop import RMSProp
            args['maps'] += [RMSProp]
        else:
            raise NotImplementedError(mp)
    from lyapunov.train_ops.simgd import SimGD
    args['maps'] += [SimGD]

    if args['saveto'] == '':
        args['saveto'] = 'examples/results/' + args['domain'] + '/' + '-'.join(args['map_strings']) + '/' + args['description']

    if args['description'] == '':
        args['description'] = args['domain'] + '-' + '-'.join(args['map_strings'])
    elif args['description'].isdigit():
        args['description'] = args['domain'] + '-' + '-'.join(args['map_strings']) + '-' + args['description']

    saveto = args['saveto'] + '/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S/{}').format('')
    if not os.path.exists(saveto):
        os.makedirs(saveto)
        os.makedirs(saveto+'/samples')
        os.makedirs(saveto+'/weights')
    shutil.copy(os.path.realpath('lyapunov/run.py'), os.path.join(saveto, 'run.py'))
    shutil.copy(os.path.realpath('lyapunov/core.py'), os.path.join(saveto, 'core.py'))
    for mp in args['map_strings']:
        train_file = mp+'.py'
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

    # python lyapunov/run.py $(cat examples/args/MO8G/con/00.txt) -dom CLGaussian -xdim 1 -zdim 1 -mx_it 10000 -d_lr 1e-6 -g_lr 1e-6
    
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
    ls = []
    viz_every = params['viz_every']

    iterations = range(params['max_iter'])
    if params['verbose']:
        iterations = tqdm(iterations,desc=params['description'])

    for i in iterations:
        
        lams, d, g, f = train.train_op(i)
        
        if params['verbose']:
            iterations.set_postfix({'Lambda':lams,'||F_D||^2':d,'||F_G||^2':g,'V':f, 'Mem': process.memory_info().rss})

        fs.append(f)
        ds.append(d)
        gs.append(g)

        if i >= params['start_lam_it']:
            ls.append(lams)
            save_weights(m.D,params['saveto']+'weights/D_'+str(i)+'.pkl')
            save_weights(m.G,params['saveto']+'weights/G_'+str(i)+'.pkl')
            if train.req_aux:
                aux_d = []
                for a in train.aux_d:
                    aux_d += [a.cpu().data.numpy()]
                aux_g = []
                for a in train.aux_g:
                    aux_g += [a.cpu().data.numpy()]
                pickle.dump(aux_d,open(params['saveto']+'weights/D_aux_'+str(i)+'.pkl','wb'))
                pickle.dump(aux_g,open(params['saveto']+'weights/G_aux_'+str(i)+'.pkl','wb'))

        if viz_every > 0 and i % viz_every == 0:
            if params['n_viz'] > 0:
                np.save(params['saveto']+'samples/'+str(i), train.m.get_fake(params['n_viz'], params['z_dim']).cpu().data.numpy())
            data.plot_current(train, params, i)
            if i >= params['start_lam_it']:
                fig = plt.figure()
                plt.plot(np.vstack(ls))
                fig.savefig(params['saveto']+'lyapunov_exponents.pdf') 
                plt.close(fig)

        if params['weights_every'] > 0 and i % params['weights_every'] == 0:
            save_weights(m.D,params['saveto']+'D_'+str(i)+'.pkl')
            save_weights(m.G,params['saveto']+'G_'+str(i)+'.pkl')
                 

    np.savetxt(params['saveto']+'d_norm.out',np.array(ds))
    np.savetxt(params['saveto']+'g_norm.out',np.array(gs))
    np.savetxt(params['saveto']+'loss.out',np.array(fs))

    print('Plotting gradient norms...')
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

    print('Plotting loss...')
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.plot(range(len(fs)),np.array(fs))
    ax.set_ylabel('Loss')
    ax.set_xlabel('Iteration')
    fig.savefig(params['saveto']+'loss.pdf')

    print('Plotting PCA of trajectory...')
    weights = []
    for w_i in range(params['start_lam_it'],params['max_iter']):
        w_D = flatten_nested(pickle.load(open(params['saveto']+'weights/D_'+str(w_i)+'.pkl','rb')))
        w_G = flatten_nested(pickle.load(open(params['saveto']+'weights/G_'+str(w_i)+'.pkl','rb')))
        weights.append(np.hstack([w_D,w_G]))
    weights = np.vstack(weights)
    ipca = IncrementalPCA(n_components=2, batch_size=10)
    X_ipca = ipca.fit_transform(weights)
    fig, ax = plt.subplots()
    path = mpath.Path(X_ipca)
    verts = path.interpolated(steps=3).vertices
    x, y = verts[:, 0], verts[:, 1]
    z = np.linspace(0, 1, len(x))
    colorline(x, y, z, cmap=plt.get_cmap('Greys'), linewidth=2)
    fig.savefig(params['saveto']+'weights_pca.pdf') 
    plt.close(fig)

    print('Plotting norm of weights over trajectory...')
    w_norms = np.linalg.norm(weights, axis=1)
    fig = plt.figure()
    plt.plot(w_norms)
    fig.savefig(params['saveto']+'weight_norms.pdf') 
    plt.close(fig)

    print('Plotting sample series over epochs...')
    if params['n_viz'] > 0:
        np_samples = []
        for viz_i in range(0,params['max_iter'],viz_every):
            np_samples.append(np.load(params['saveto']+'samples/'+str(viz_i)+'.npy'))
        data.plot_series(np_samples, params)

    print('Complete.')


def flatten_nested(a):
    return np.concatenate([ai.flatten() for ai in a])


def colorline(
    x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    ax.set_xlim([x.min(),x.max()])
    ax.set_ylim([y.min(),y.max()])

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


if __name__ == '__main__':
    Train, Domain, Generator, Discriminator, params = parse_params()
    run_experiment(Train, Domain, Generator, Discriminator, params)
