__doc__ = "a method to support some basic and not-so-basic plotting functionality"
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import numpy as np

### non-standard libraries
from sddr import utils

#-------------------------------------------------

def kde2fit(points, logkde, priors, fields, b=utils.DEFAULT_B):
    """
    generate something like a corner plot for logkde
    assumes
        points.shape = Nfields, Npoints
        logkde.shape = (Npoints,)*Nfields
        priors.shape = Nfields, 2
        len(fields) = Nfields
    b is a scalar

    returns the associated figure object
    """
    ### sanity-check input
    Nfields, Npoints = points.shape
    nfields, nbounds = priors.shape
    assert nfields==Nfields
    assert nbounds==2
    assert len(fields)==Nfields
    shape = logkde.shape
    assert len(shape)==Nfields
    assert np.all(n==Npoints for n in shape)

    ### plot this stuff!
    fig = plt.figure()

    raise NotImplementedError('figure out how to plot this thing!')

    ### iterate through pairs of fields, marginalizing away as needed
#    marg = utils.nd_marg(points, logkde) ### get's rid of the last axis
