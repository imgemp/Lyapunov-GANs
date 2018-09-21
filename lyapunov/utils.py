"""
Image grid saver, based on color_grid_vis from github.com/Newmu
"""

import pickle
import numpy as np
import torch
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll


def gpu_helper(gpu):
    if gpu >= -1:
        def to_gpu(x):
            x = x.cuda()
            return x
        return to_gpu
    else:
        def no_op(x):
            return x
        return no_op

def save_weights(module,file):
    weights = []
    for p in module.parameters():
        weights += [p.cpu().data.numpy()]
    pickle.dump(weights,open(file,'wb'))

def load_weights(module,file):
    weights = pickle.load(open(file,'rb'))
    for p,w in zip(module.parameters(),weights):
        p.data = torch.from_numpy(w)
    return weights

def detach_all(a):
    detached = []
    for ai in a:
        if isinstance(ai, list) or isinstance(ai, tuple):
            detached += [detach_all(ai)]
        elif ai is not None:
            detached += [ai.detach()]
        else:
            detached += [None]
    return detached

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

    ax.set_xlim([np.nanmin(x),np.nanmax(x)])
    ax.set_ylim([np.nanmin(y),np.nanmax(y)])

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


def intersection(x, y, sub=0):
    lo = max(x[0], y[0]) - sub
    hi = min(x[-1], y[-1]) - sub
    if lo <= hi:
        return range(lo, hi+1)
    else:
        return None

def shift_range(rng, shift=0):
    if rng is not None:
        return range(rng.start+shift, rng.stop+shift, rng.step)
    else:
        return None