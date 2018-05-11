__doc__ = "a module holding utility functions for sddr estimation"
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

import numpy as np

from scipy.special import erf
from scipy.special import digamma
from scipy.special import polygamma

from scipy.stats import beta

import h5py

#-------------------------------------------------

LOG3 = np.log(3)
DEFAULT_NUM_POINTS = 101

DEFAULT_B = 0.1
DEFAULT_B_RANGE = (1e-6, 1e+2)
DEFAULT_RTOL = 1e-4
DEFAULT_DLOGL = 10

DEFAULT_PRIOR_MIN = -10
DEFAULT_PRIOR_MAX = -4
DEFAULT_FIT_MAX = -9

DEFAULT_NUM_SUBSETS = 10

#-------------------------------------------------

def load(paths, verbose=False):
    """
    load in samples from files
    pulls out only log10NLTides_A0
    """
    samples = None
    for path in paths:
        if verbose:
            print('reading samples from: '+path)

        if path.endswith('hdf5'):
            new = load_hdf5(path, verbose=verbose)
        elif path.endswidth('dat'):
            new = load_dat(path, verbose=verbose)
        else:
            raise ValueError, 'do not know how to load: '+path

        if samples is None:
            samples = new
        else:
            samples = np.concatenate((samples, new))

    if verbose:
        print('retained %d samples in all'%len(samples))
    return samples

def load_hdf5(path, verbose=False):
    with h5py.File(path, 'r') as file_obj:
        new = file_obj['lalinference/lalinference_mcmc/posterior_samples']['log10NLTides_A0']
    if verbose:
        print('    found %d samples'%len(new))

    new = new[len(new)/2:] ### FIXME: 
                           ###    throw out the burn in more intelligently?
                           ###    also downsample to uncorrelated samples?

    if verbose:
        print('    retained %d samples'%len(new))
    return new

def load_dat(path, verbose=False):
    new = np.genfromtxt(path, names=True)['log10NLTides_A0']
    if verbose:
        print('    found %d samples'%len(new))
    return new

#------------------------

def partition(data, num_subsets=DEFAULT_NUM_SUBSETS):
    '''
    partition samples into separate subsets
    '''
    N=len(data)
    subsets = [np.zeros(N, dtype=bool) for _ in xrange(num_subsets)]
    for i in xrange(N):
        n = i%num_subsets
        subsets[n][i] = True

    return [data[truth] for truth in subsets]

#-------------------------------------------------

def hist(data, b=None, prior_min=DEFAULT_PRIOR_MIN, prior_max=DEFAULT_PRIOR_MAX):
    '''
    compute a simple histogram and return the normalized count in the first bin
    '''
    if b is None:
        b = int(len(data)**0.5)

    n, b = np.histogram(data, b, range=(prior_min, prior_max))
    return np.log(n[0]) - np.log(b[1]-b[0]) - np.log(np.sum(n))

def chist(data, b=None, deg=2, fit_max=DEFAULT_FIT_MAX, prior_min=DEFAULT_PRIOR_MIN, prior_max=DEFAULT_PRIOR_MAX):
    '''
    compute a cumulative histogram and fit it with a low-order polynomial, returning the slope at the prior_min
    '''
    if b is None:
        b = 2*len(data)

    n, b = np.histogram(data, b, range=(prior_min, prior_max))
    c = np.cumsum(n).astype(float)/np.sum(n) ### make a cumulative histogram

    ### fit to a low-order polynomial
    B = 0.5*(b[1:]+b[:-1])
    truth = B<=fit_max
    params = np.polyfit(B[truth], c[truth], deg)

    return np.log(np.sum([params[i]*(deg-i)*(prior_min)**(deg-i-1) for i in xrange(deg)]))

def kde(data, b=DEFAULT_B, prior_min=DEFAULT_PRIOR_MIN, prior_max=DEFAULT_PRIOR_MAX):
    return _compute_logkde(prior_min, data, b=b, prior_min=prior_min, prior_max=prior_max)

def max_kde(data, (min_b, max_b), prior_min=DEFAULT_PRIOR_MIN, prior_max=DEFAULT_PRIOR_MAX, rtol=DEFAULT_RTOL, verbose=False):
    b = optimize_bandwidth(data, min_b, max_b, rtol=rtol, verbose=verbose) ### find the ~best bandwidth
    return _compute_logkde(prior_min, data, b=b, prior_min=prior_min, prior_max=prior_max)

def marg_kde(data, (min_b, max_b), prior_min=DEFAULT_PRIOR_MIN, prior_max=DEFAULT_PRIOR_MAX, rtol=DEFAULT_RTOL, dlogl=DEFAULT_DLOGL, num_points=DEFAULT_NUM_POINTS, verbose=False):
    bs, weights = marginalize_bandwidth(data, min_b, max_b, rtol=rtol, dlogl=dlogl, num_points=num_points, verbose=verbose) ### marginalize over Bs
    return np.log(np.sum([weight*np.exp(_compute_logkde(prior_min, data, b=b, prior_min=prior_min, prior_max=prior_max)) for b, weight in zip(bs, weights)]))

def _compute_logkde(x, data, b=DEFAULT_B, prior_min=DEFAULT_PRIOR_MIN, prior_max=DEFAULT_PRIOR_MAX):
    '''
    compute the log(kde) @ x given data with the specified bandwidth normalized within the prior bounds with reflecting boundary conditions

    ONLY SUPPORTS SCALAR x
    '''
    N = len(data)
    twobsqrd = 2*b**2
    bsqrt2 = b*2**0.5

    # compute log(kernel) for each data sample, incorporating reflecting boundaries
    logkde = np.concatenate((
        -(x-data)**2/twobsqrd,
        -(x-2*prior_min+data)**2/twobsqrd,
        -(x-2*prior_max+data)**2/twobsqrd
    )) - 0.5*np.log(2*np.pi*b**2) ### subtract the normalization for the Gaussian kernel

    # sum all contributions together, retaining high precision
    m = np.max(logkde)
    logkde = np.log(np.sum(np.exp(logkde-m))) + m

    # compute the integral between the prior bounds
    norm = 0.5*np.sum( \
        erf((prior_max-data)/bsqrt2) - erf((prior_min-data)/bsqrt2) \
      + erf((prior_max-2*prior_min+data)/bsqrt2) - erf((data-prior_min)/bsqrt2) \
      + erf((data-prior_max)/bsqrt2) - erf((prior_min-2*prior_max+data)/bsqrt2) \
    )

    return logkde - np.log(norm)

def _compute_loglike(data, b=DEFAULT_B): #, prior_min=DEFAULT_PRIOR_MIN, prior_max=DEFAULT_PRIOR_MAX):
    '''
    compute the leave-one-out loglikelihood
    '''
    N = len(data)
    truth = np.ones(N, dtype=bool)

    bsqrd2 = 1./(2*b**2)
    norm = np.log(N-1) + 0.5*np.log(2*np.pi*b**2)
    logL = 0.
    for i in xrange(N): ### FIXME: can I do this without just iterating (which is kinda slow)?
        truth[i-1] = True
        truth[i] = False
        z = -(data[i]-data[truth])**2*bsqrd2
        m = np.max(z)
        logL += np.log(np.sum(np.exp(z-m))) + m - norm
    logL /= N

    return logL

def _array_compute_loglike(data, b=DEFAULT_B): #, prior_min=DEFAULT_PRIOR_MIN, prior_max=DEFAULT_PRIOR_MAX):
    '''
    compute the leave-one-out loglikelihood
    not actually faster...
    '''
    N = len(data)
    ones = np.ones(N)
    z = np.outer(data, ones) ### allocating this much memory is slow?
    z = -(z - z.transpose())**2/(2*b**2) + np.diag(-np.infty*ones) ### zero out the diagonal elements
    m = np.max(z, axis=1)
    return np.sum(np.log(np.sum(np.exp(z-m), axis=1)) + m - np.log(N-1) - 0.5*np.log(2*np.pi*b**2))/N

def _compute_gradloglike(data, b=DEFAULT_B):
    '''
    compute dloglike/dlogb
    '''
    N = len(data)
    truth = np.ones(N, dtype=bool)

    bsqrd2 = 1./(2*b**2)
    grad = 0.
    for i in xrange(N):
        truth[i-1] = True
        truth[i] = False
        z = -(data[i]-data[truth])**2*bsqrd2
        exp = np.exp(z-np.max(z))
        grad += np.sum((-2*z - 1)*exp) / np.sum(exp)
    grad /= N

    return grad

def optimize_bandwidth(data, min_b, max_b, rtol=1e-4, verbose=False):
    '''
    perform a bisection search in log(b) for the zero-crossing of gradloglike
    '''
    _min = _compute_gradloglike(data, b=min_b)
    if _min <= 0:
        return min_b

    _max = _compute_gradloglike(data, b=max_b)
    if _max >= 0:
        return max_b

    mid_b = (max_b*min_b)**0.5
    while max_b-min_b > rtol*mid_b:
        _mid = _compute_gradloglike(data, b=mid_b)
        if verbose:
            print (min_b, _min), (mid_b, _mid), (max_b, _max)

        if _mid > 0: ### replace min
            min_b = mid_b
            _min = _mid
        else: ### replace max
            max_b = mid_b
            _max = _mid
        mid_b = (max_b*min_b)**0.5
        
    return mid_b

def marginalize_bandwidth(data, min_b, max_b, rtol=DEFAULT_RTOL, dlogl=DEFAULT_DLOGL, num_points=DEFAULT_NUM_POINTS, verbose=False):
    '''
    perform a series of bisection searches to define a grid in bandwidth over which we iterate and compute weights
    '''
    ### first, perform a bisection search to find the maximum likelihood
    ### leverage this to find the b-range corresponding to loglike >= max(loglike)-dlogl
    _min = _compute_gradloglike(data, b=min_b)
    if _min <= 0: ### one of the boundaries is the maximum allowed, so we have a short-cut
        max_b = _find_b(data, _compute_loglike(data, min_b) - dlogl, min_b, max_b, rtol=rtol)

    else:
        _max = _compute_gradloglike(data, b=max_b)
        if _max >= 0: ### the other boundary gives another short-cut
            min_b = _find_b(data, _compute_loglike(data, max_b) - dlogl, min_b, max_b, rtol=rtol)

        else: ### we have to do a bisection search
            _min_b = min_b ### remember these for later
            _max_b = max_b

            mid_b = (_max_b*_min_b)**0.5
            while _max_b-_min_b > rtol*mid_b:
                _mid = _compute_gradloglike(data, b=mid_b)

                if verbose:
                    print (_min_b, _min), (mid_b, _mid), (_max_b, _max)
        
                if _mid > 0: ### replace min
                    _min_b = mid_b
                    _min = _mid
                else: ### replace max
                    _max_b = mid_b
                    _max = _mid
                mid_b = (_max_b*_min_b)**0.5

            ### now search through bmap to find the left/right edges of this
            target = _compute_loglike(data, b=mid_b) - dlogl ### this logl defines the boundaries of our marginalization

            min_b = _find_b(data, target, min_b, mid_b, rtol=rtol, verbose=verbose) # find the lower b
            max_b = _find_b(data, target, mid_b, max_b, rtol=rtol, verbose=verbose) # find the upper b

    ### we now have min_b, max_b defining the ranges we want, so let's define our grid
    bs = np.logspace(np.log10(min_b), np.log10(max_b), num_points) ### choose logarithmic spacing -> flat prior in log(b)

    weights = np.array([_compute_loglike(data, b) for b in bs]) ### NOTE: this is expensive... 
    weights = np.exp(weights-np.max(weights))
    weights /= np.sum(weights)

    if verbose:
        print 'grid is as follows:'
        for b, w in zip(bs, weights):
            print '\tb=', b, '\tw=', w

    return bs, weights

def _find_b(data, target, min_b, max_b, rtol=DEFAULT_RTOL, verbose=False):
    '''
    find the b corresponding to loglike=target between min_b, max_b

    NOTE: we expect target to be smaller than the loglike at at least one of the specified min_b, max_b
    things may break if that is not true...
    '''
    _min = _compute_loglike(data, min_b)
    _max = _compute_loglike(data, max_b)

    if _min > _max: ### swap these
        max_b, min_b = min_b, max_b
        _max, _min = _min, _max

    mid_b = (min_b*max_b)**0.5
    while abs(max_b-min_b) > rtol*mid_b:
        _mid = _compute_loglike(data, mid_b)

        if verbose:
            print (min_b, _min), (mid_b, _mid), (max_b, _max)

        if target > _mid:
            min_b = mid_b
            _min = _mid
        else:
            max_b = mid_b
            _max = _mid
        mid_b = (max_b*min_b)**0.5

    return mid_b
