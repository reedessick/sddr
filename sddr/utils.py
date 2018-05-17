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
DEFAULT_NUM_QUANTILES = 1001
DEFAULT_NUM_SUBSETS = 10

DEFAULT_B = 0.1
DEFAULT_B_RANGE = (1e-6, 1e+2)
DEFAULT_RTOL = 1e-4
DEFAULT_DLOGL = 10
DEFAULT_B_PRIOR = 'log'

DEFAULT_PRIOR_MIN = -10
DEFAULT_PRIOR_MAX = -4
DEFAULT_FIT_MAX = -9

DEFAULT_FIELD = 'log10NLTides_A0'

#-------------------------------------------------

def load(paths, verbose=False, field=DEFAULT_FIELD):
    """
    load in samples from files
    pulls out only log10NLTides_A0
    """
    samples = None
    for path in paths:
        if verbose:
            print('reading samples from: '+path)

        if path.endswith('hdf5'):
            new = load_hdf5(path, field=field, verbose=verbose)

        elif path.endswith('dat'):
            new = load_dat(path, field=field, verbose=verbose)

        else:
            raise ValueError, 'do not know how to load: '+path

        if samples is None:
            samples = new
        else:
            samples = np.concatenate((samples, new))

    if verbose:
        print('retained %d samples in all'%len(samples))
    return samples

def load_hdf5(path, field=DEFAULT_FIELD, verbose=False):
    with h5py.File(path, 'r') as file_obj:
        new = file_obj['lalinference/lalinference_mcmc/posterior_samples'][field]
    if verbose:
        print('    found %d samples'%len(new))

    new = new[len(new)/2:] ### FIXME: 
                           ###    throw out the burn in more intelligently?
                           ###    also downsample to uncorrelated samples?

    if verbose:
        print('    retained %d samples'%len(new))
    return new

def load_dat(path, field=DEFAULT_FIELD, verbose=False):
    new = np.genfromtxt(path, names=True)[field]
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

def _get_hist_bins(N):
    return max(10, int(N**0.5/10))

def hist(x, data, b=None, prior_min=DEFAULT_PRIOR_MIN, prior_max=DEFAULT_PRIOR_MAX):
    '''
    compute a simple histogram and return the normalized count in the first bin
    '''
    if b is None:
        b = _get_hist_bins(len(data))

    n, b = np.histogram(data, b, range=(prior_min, prior_max))
    ind = int(np.interp(x, b, np.arange(len(b)))) ### do this backflip to allow for arbitrary x...
    return np.log(n[ind]) - np.log(b[ind+1]-b[ind]) - np.log(np.sum(n))

def chist(x, data, b=None, deg=2, fit_max=DEFAULT_FIT_MAX, prior_min=DEFAULT_PRIOR_MIN, prior_max=DEFAULT_PRIOR_MAX):
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

    return np.log(np.sum([params[i]*(deg-i)* x**(deg-i-1) for i in xrange(deg)]))

def kde(x, data, b=DEFAULT_B, prior_min=DEFAULT_PRIOR_MIN, prior_max=DEFAULT_PRIOR_MAX):
    return _compute_logkde(x, data, b=b, prior_min=prior_min, prior_max=prior_max)

def max_kde(x, data, (min_b, max_b), prior_min=DEFAULT_PRIOR_MIN, prior_max=DEFAULT_PRIOR_MAX, rtol=DEFAULT_RTOL, verbose=False):
    b = optimize_bandwidth(data, min_b, max_b, rtol=rtol, verbose=verbose) ### find the ~best bandwidth
    return _compute_logkde(x, data, b=b, prior_min=prior_min, prior_max=prior_max)

def marg_kde(x, data, (min_b, max_b), prior_min=DEFAULT_PRIOR_MIN, prior_max=DEFAULT_PRIOR_MAX, rtol=DEFAULT_RTOL, dlogl=DEFAULT_DLOGL, num_points=DEFAULT_NUM_POINTS, prior=DEFAULT_B_PRIOR, verbose=False):
    bs, weights = marginalize_bandwidth(data, min_b, max_b, rtol=rtol, dlogl=dlogl, num_points=num_points, prior=prior, verbose=verbose) ### marginalize over bandwidth
    return np.log(np.sum([weight*np.exp(_compute_logkde(x, data, b=b, prior_min=prior_min, prior_max=prior_max)) for b, weight in zip(bs, weights)]))

def marg_betakde(x, data, (min_b, max_b), prior_min=DEFAULT_PRIOR_MIN, prior_max=DEFAULT_PRIOR_MAX, rtol=DEFAULT_RTOL, dlogl=DEFAULT_DLOGL, num_points=DEFAULT_NUM_POINTS, prior=DEFAULT_B_PRIOR, num_quantiles=DEFAULT_NUM_QUANTILES, verbose=False):
    bs, weights = marginalize_bandwidth(data, min_b, max_b, rtol=rtol, dlogl=dlogl, num_points=num_points, prior=prior, verbose=verbose) ### marginalize over bandwidth
    params = [_compute_betadistrib(x, data, b=b, prior_min=prior_min, prior_max=prior_max) for b in bs]

    ### figure out boundaries for our initial gridding
    lq = 1./num_quantiles
    low = np.infty
    hgh = -np.infty
    for _alpha, _beta, _scale in params:
        l = np.log(beta.ppf(lq, _alpha, _beta)*_scale)
        h = np.log(_scale) ### just take the biggest possible value allowed by the beta distribution

        low = min(l, low) ### this skips nans, which is pretty convenient
        hgh = max(h, hgh)

    ### compute the cdf along the initial grid
    logkde = np.linspace(low, hgh, num_quantiles) ### our gridding in logkde
    cdf = _compute_cdf(logkde, params, weights)

    ### re-structure the grid to give something closer to a reasonable set of quantiles
#    if np.any(cdf<=lq):
#        low = logkde[cdf<=lq][-1]
#
#    if np.any(cdf>=1.-lq):
#        hgh = logkde[cdf>=1.-lq][0]
#    logkde = np.linspace(low, hgh, num_quantiles)

    logkde = np.interp(np.linspace(lq, 1.-lq, num_quantiles), cdf, logkde) ### interpolate to approximate even sampling in probability distrib
    cdf = _compute_cdf(logkde, params, weights) ### recompute at the new grid placement

    return logkde, cdf

def _compute_cdf(logkde, params, weights):
    exp = np.exp(logkde)
    cdf = np.array([beta.cdf(exp/_scale, _alpha, _beta) for _alpha, _beta, _scale in params]) ### cdf for each param separately
    return np.sum( cdf.transpose()*weights, axis=1 )

def _compute_logkde(x, data, b=DEFAULT_B, prior_min=DEFAULT_PRIOR_MIN, prior_max=DEFAULT_PRIOR_MAX):
    '''
    compute the log(kde) @ x given data with the specified bandwidth normalized within the prior bounds with reflecting boundary conditions

    ONLY SUPPORTS SCALAR x
    '''
    twobsqrd = 2*b**2
    bsqrt2 = b*2**0.5

    # compute log(kernel) for each data sample, incorporating reflecting boundaries
    logkde = np.concatenate((
        -(x-data)**2/twobsqrd,
        -(x-2*prior_min+data)**2/twobsqrd,
        -(x-2*prior_max+data)**2/twobsqrd
    )) - 0.5*np.log(np.pi*twobsqrd) ### subtract the normalization for the Gaussian kernel

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

def _compute_betadistrib(x, data, b=DEFAULT_B, prior_min=DEFAULT_PRIOR_MIN, prior_max=DEFAULT_PRIOR_MAX):
    '''
    compute the best-fit beta distribution for kde(x|b)
    return alpha, beta, scale

    we fit f = ((2*np.pi*b**2)**0.5 * norm / 3)

    '''
    N = len(data)
    logN = np.log(N)
    twobsqrd = 2*b**2
    bsqrt2 = b*2**0.5

    # compute log(kernel) for each data sample, incorporating reflecting boundaries
    logf = np.array([
        -(x-data)**2/twobsqrd,
        -(x-2*prior_min+data)**2/twobsqrd,
        -(x-2*prior_max+data)**2/twobsqrd
    ])
    # sum over reflecting boundaries, retaining high precision
    m = np.max(logf, axis=0)
    logf = np.log(np.sum(np.exp(logf - m), axis=0)) + m -np.log(3) ### normalization by 3 guarantees f \in [0,1]

    # compute the second moment
    logf2 = logf*2

    # sum all contributions for different samples together, retaining high precision
    m = np.max(logf)
    logf = np.log(np.sum(np.exp(logf-m))) + m - logN

    # repeat for second moment
    m = np.max(logf2)
    logf2 = np.log(np.sum(np.exp(logf2-m))) + m - logN

    # compute the beta-distrib parameters that match these moments
    e = np.exp(logf)               ### expected value
    v = (np.exp(logf2) - e**2)/N   ### variance

    x = ((1.-e)*e - v)
    _alpha = (e/v)*x                ### beta distribution parameters
    _beta = ((1.-e)/v)*x

    ### FIXME: check for unphysical variances here?

    # compute the scale by which we multiply the beta distrib via an integral between the prior bounds
    norm = 0.5*np.sum( \
        erf((prior_max-data)/bsqrt2) - erf((prior_min-data)/bsqrt2) \
      + erf((prior_max-2*prior_min+data)/bsqrt2) - erf((data-prior_min)/bsqrt2) \
      + erf((data-prior_max)/bsqrt2) - erf((prior_min-2*prior_max+data)/bsqrt2) \
    ) / N
    scale = 3/(norm*(np.pi*twobsqrd)**0.5)

    return _alpha, _beta, scale

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

def marginalize_bandwidth(data, min_b, max_b, rtol=DEFAULT_RTOL, dlogl=DEFAULT_DLOGL, num_points=DEFAULT_NUM_POINTS, prior=DEFAULT_B_PRIOR, verbose=False):
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
    if prior=='log':
        bs = np.logspace(np.log10(min_b), np.log10(max_b), num_points) ### choose logarithmic spacing -> flat prior in log(b)
    elif prior=='lin':
        bs = np.linspace(min_b, max_b, num_points)
    else:
        raise ValueError, 'prior=%s not understood!'%prior

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
