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
DEFAULT_DELTALOGP = 6.5
DEFAULT_DOWNSAMPLE = 1
DEFAULT_INITIAL_BURNIN = 0

#-------------------------------------------------

def load(paths, field=DEFAULT_FIELD, deltaLogP=DEFAULT_DELTALOGP, downsample=DEFAULT_DOWNSAMPLE, initial_burnin=DEFAULT_INITIAL_BURNIN, verbose=False):
    """
    load in samples from files
    pulls out only log10NLTides_A0
    """
    samples = None
    for path in paths:
        if verbose:
            print('reading samples from: '+path)

        if path.endswith('hdf5'):
            new = load_hdf5(path, field=field, deltaLogP=deltaLogP, downsample=downsample, initial_burnin=initial_burnin, verbose=verbose)

        elif path.endswith('dat'):
            new = load_dat(path, field=field, downsample=downsample, verbose=verbose)

        else:
            raise ValueError, 'do not know how to load: '+path

        if samples is None:
            samples = new
        else:
            samples = np.concatenate((samples, new))

    if verbose:
        print('retained %d samples in all'%len(samples))
    return samples

def load_hdf5(path, field=DEFAULT_FIELD,  deltaLogP=DEFAULT_DELTALOGP, downsample=DEFAULT_DOWNSAMPLE, initial_burnin=DEFAULT_INITIAL_BURNIN, verbose=False):
    with h5py.File(path, 'r') as file_obj:
        new = file_obj['lalinference/lalinference_mcmc/posterior_samples'][...]
    if verbose:
        print('    found %d samples'%len(new))

    # find the first time the chain fluctuates above threshold
    ind = np.arange(len(new))
    ind = ind[ind>=initial_burnin] ### throw away the first few burn-in points
    if len(ind): 
        ind = ind[np.max(new['logpost'][ind])-new['logpost'][ind] < deltaLogP] ### filter the rest by deltaLogP
        if not len(ind): ### no samples survive this cut!
            raise ValueError, 'no samples survived the deltaLogP cut!'
        ind = ind[0] ### take the first one

        # keep everything after that point
        new = new[ind:]

    ### downsample based on a fixed spacing provided by the user
    new = new[::downsample] ### FIXME

    if verbose:
        print('    retained %d samples'%len(new))
    if field is None:
        return new
    else:
        return new[field] ### only return the requested field

def load_dat(path, field=DEFAULT_FIELD, downsample=DEFAULT_DOWNSAMPLE, verbose=False):
    new = np.genfromtxt(path, names=True)
    if verbose:
        print('    found %d samples'%len(new))

    new = new[::downsample]
    if verbose:
        print('    retained %d samples'%len(new))
    if field is None:
        return new
    else:
        return new[field]

#------------------------

def partition(data, weights, num_subsets=DEFAULT_NUM_SUBSETS):
    '''
    partition samples into separate subsets
    '''
    N=len(data)
    subsets = [np.zeros(N, dtype=bool) for _ in xrange(num_subsets)]
    for i in xrange(N):
        n = i%num_subsets
        subsets[n][i] = True

    return [(data[truth], weights[truth]) for truth in subsets]

#-------------------------------------------------

def _get_hist_bins(N):
    return max(10, int(N**0.5/10))

def hist(x, data, weights, b=None, prior_min=DEFAULT_PRIOR_MIN, prior_max=DEFAULT_PRIOR_MAX):
    '''
    compute a simple histogram and return the normalized count in the first bin
    '''
    if b is None:
        b = _get_hist_bins(len(data))

    n, b = np.histogram(data, b, range=(prior_min, prior_max), weights=weights)
    ind = int(np.interp(x, b, np.arange(len(b)))) ### do this backflip to allow for arbitrary x...
    return np.log(n[ind]) - np.log(b[ind+1]-b[ind]) - np.log(np.sum(n))

def chist(x, data, weights, b=None, deg=2, fit_max=DEFAULT_FIT_MAX, prior_min=DEFAULT_PRIOR_MIN, prior_max=DEFAULT_PRIOR_MAX):
    '''
    compute a cumulative histogram and fit it with a low-order polynomial, returning the slope at the prior_min
    '''
    if b is None:
        b = 2*len(data)

    n, b = np.histogram(data, b, range=(prior_min, prior_max), weights=weights)
    c = np.cumsum(n).astype(float)/np.sum(n) ### make a cumulative histogram

    ### fit to a low-order polynomial
    B = 0.5*(b[1:]+b[:-1])
    truth = B<=fit_max
    params = np.polyfit(B[truth], c[truth], deg)

    return np.log(np.sum([params[i]*(deg-i)* x**(deg-i-1) for i in xrange(deg)]))

def kde(x, data, weights, b=DEFAULT_B, prior_min=DEFAULT_PRIOR_MIN, prior_max=DEFAULT_PRIOR_MAX):
    return _compute_logkde(x, data, weights, b=b, prior_min=prior_min, prior_max=prior_max)

def max_kde(x, data, weights, (min_b, max_b), prior_min=DEFAULT_PRIOR_MIN, prior_max=DEFAULT_PRIOR_MAX, rtol=DEFAULT_RTOL, verbose=False):
    b = optimize_bandwidth(data, weights, min_b, max_b, rtol=rtol, verbose=verbose) ### find the ~best bandwidth
    if verbose:
        print('    max b = %.3e'%b)
    return _compute_logkde(x, data, weights, b=b, prior_min=prior_min, prior_max=prior_max)

def marg_kde(x, data, weights, (min_b, max_b), prior_min=DEFAULT_PRIOR_MIN, prior_max=DEFAULT_PRIOR_MAX, rtol=DEFAULT_RTOL, dlogl=DEFAULT_DLOGL, num_points=DEFAULT_NUM_POINTS, prior=DEFAULT_B_PRIOR, verbose=False):
    bs, ws = marginalize_bandwidth(data, weights, min_b, max_b, rtol=rtol, dlogl=dlogl, num_points=num_points, prior=prior, verbose=verbose) ### marginalize over bandwidth
    return np.log(np.sum([w*np.exp(_compute_logkde(x, data, weights, b=b, prior_min=prior_min, prior_max=prior_max)) for b, w in zip(bs, ws)]))

def max_betakde(x, data, weights, (min_b, max_b), prior_min=DEFAULT_PRIOR_MIN, prior_max=DEFAULT_PRIOR_MAX, rtol=DEFAULT_RTOL, num_quantiles=DEFAULT_NUM_QUANTILES, verbose=False):
    b = optimize_bandwidth(data, weights, min_b, max_b, rtol=rtol, verbose=verbose) ### find the ~best bandwidth
    if verbose:
        print('    max b = %.3e'%b)
    _alpha, _beta, _scale = _compute_betadistrib(x, data, weights, b=b, prior_min=prior_min, prior_max=prior_max)

    lq = 1./num_quantiles
    low = np.log(beta.ppf(lq, _alpha, _beta)*_scale)
    hgh = np.log(_scale)
    logkde = np.linspace(low, hgh, num_quantiles) ### our gridding in logkde
    cdf = _compute_cdf(logkde, [(_alpha, _beta, _scale)], [1])

    logkde = np.interp(np.linspace(lq, 1.-lq, num_quantiles), cdf, logkde) ### interpolate to approximate even sampling in probability distrib
    cdf = _compute_cdf(logkde, [(_alpha, _beta, _scale)], [1]) ### recompute at the new grid placement

    return logkde, cdf

def marg_betakde(x, data, weights, (min_b, max_b), prior_min=DEFAULT_PRIOR_MIN, prior_max=DEFAULT_PRIOR_MAX, rtol=DEFAULT_RTOL, dlogl=DEFAULT_DLOGL, num_points=DEFAULT_NUM_POINTS, prior=DEFAULT_B_PRIOR, num_quantiles=DEFAULT_NUM_QUANTILES, verbose=False):
    bs, ws = marginalize_bandwidth(data, weights, min_b, max_b, rtol=rtol, dlogl=dlogl, num_points=num_points, prior=prior, verbose=verbose) ### marginalize over bandwidth
    params = [_compute_betadistrib(x, data, weights, b=b, prior_min=prior_min, prior_max=prior_max) for b in bs]

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
    cdf = _compute_cdf(logkde, params, ws)

    ### re-structure the grid to give something closer to a reasonable set of quantiles
#    if np.any(cdf<=lq):
#        low = logkde[cdf<=lq][-1]
#
#    if np.any(cdf>=1.-lq):
#        hgh = logkde[cdf>=1.-lq][0]
#    logkde = np.linspace(low, hgh, num_quantiles)

    logkde = np.interp(np.linspace(lq, 1.-lq, num_quantiles), cdf, logkde) ### interpolate to approximate even sampling in probability distrib
    cdf = _compute_cdf(logkde, params, ws) ### recompute at the new grid placement

    return logkde, cdf

def _compute_cdf(logkde, params, weights):
    exp = np.exp(logkde)
    cdf = np.array([beta.cdf(exp/_scale, _alpha, _beta) for _alpha, _beta, _scale in params]) ### cdf for each param separately
    return np.sum( cdf.transpose()*weights, axis=1 )

def _compute_logkde(x, data, weights, b=DEFAULT_B, prior_min=DEFAULT_PRIOR_MIN, prior_max=DEFAULT_PRIOR_MAX):
    '''
    compute the log(kde) @ x given data with the specified bandwidth normalized within the prior bounds with reflecting boundary conditions

    ONLY SUPPORTS SCALAR x
    '''
    twobsqrd = 2*b**2
    bsqrt2 = b*2**0.5

    # compute the integral between the prior bounds
    norm = 0.5*np.sum( weights*(\
        erf((prior_max-data)/bsqrt2) - erf((prior_min-data)/bsqrt2) \
      + erf((prior_max-2*prior_min+data)/bsqrt2) - erf((data-prior_min)/bsqrt2) \
      + erf((data-prior_max)/bsqrt2) - erf((prior_min-2*prior_max+data)/bsqrt2)) \
    )

    # compute log(kernel) for each data sample, incorporating reflecting boundaries
    logkde = np.concatenate((
        -(x-data)**2/twobsqrd,
        -(x-2*prior_min+data)**2/twobsqrd,
        -(x-2*prior_max+data)**2/twobsqrd
    )) - 0.5*np.log(np.pi*twobsqrd) ### subtract the normalization for the Gaussian kernel
    weights = np.concatenate((weights, weights, weights)) ### concatenate this crap so the weights match the sample points

    # sum all contributions together, retaining high precision
    m = np.max(logkde)
    logkde = np.log(np.sum(np.exp(logkde-m)*weights)) + m

    return logkde - np.log(norm)

def _compute_betadistrib(x, data, weights, b=DEFAULT_B, prior_min=DEFAULT_PRIOR_MIN, prior_max=DEFAULT_PRIOR_MAX):
    '''
    compute the best-fit beta distribution for kde(x|b)
    return alpha, beta, scale

    we fit f = ((2*np.pi*b**2)**0.5 * norm / 3)

    '''
#    N = len(data)
    N = np.sum(weights)
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
    logf = np.log(np.sum(np.exp(logf-m)*weights)) + m - logN

    # repeat for second moment
    m = np.max(logf2)
    logf2 = np.log(np.sum(np.exp(logf2-m)*weights)) + m - logN

    # compute the beta-distrib parameters that match these moments
    e = np.exp(logf)               ### expected value
    v = (np.exp(logf2) - e**2)*np.sum(weights**2)/N   ### variance

    x = ((1.-e)*e - v)
    _alpha = (e/v)*x                ### beta distribution parameters
    _beta = ((1.-e)/v)*x

    ### FIXME: check for unphysical variances here?

    # compute the scale by which we multiply the beta distrib via an integral between the prior bounds
    norm = 0.5*np.sum(weights*( \
        erf((prior_max-data)/bsqrt2) - erf((prior_min-data)/bsqrt2) \
      + erf((prior_max-2*prior_min+data)/bsqrt2) - erf((data-prior_min)/bsqrt2) \
      + erf((data-prior_max)/bsqrt2) - erf((prior_min-2*prior_max+data)/bsqrt2)) \
    ) / N
    scale = 3/(norm*(np.pi*twobsqrd)**0.5)

    return _alpha, _beta, scale

def _compute_loglike(data, weights, b=DEFAULT_B): #, prior_min=DEFAULT_PRIOR_MIN, prior_max=DEFAULT_PRIOR_MAX):
    '''
    compute the leave-one-out loglikelihood
    '''
    N = np.sum(weights)
    truth = np.ones_like(data, dtype=bool)

    bsqrd2 = 1./(2*b**2)
    norm = 0.5*np.log(2*np.pi*b**2)
    logL = 0.
    for i in xrange(len(data)): ### FIXME: can I do this without just iterating (which is kinda slow)?
        truth[i-1] = True
        truth[i] = False
        z = -(data[i]-data[truth])**2*bsqrd2
        m = np.max(z)
        logL += (np.log(np.sum(np.exp(z-m)*weights[truth])) - np.log(np.sum(weights[truth])) + m - norm) * weights[i]
    logL /= N

    return logL

### not actually used, so this is DEPRECATED
#def _array_compute_loglike(data, weights, b=DEFAULT_B): #, prior_min=DEFAULT_PRIOR_MIN, prior_max=DEFAULT_PRIOR_MAX):
#    '''
#    compute the leave-one-out loglikelihood
#    not actually faster...
#    '''
#
#    raise NotImplementedError('incorporate weights')
#
#    N = len(data)
#    ones = np.ones(N)
#    z = np.outer(data, ones) ### allocating this much memory is slow?
#    z = -(z - z.transpose())**2/(2*b**2) + np.diag(-np.infty*ones) ### zero out the diagonal elements
#    m = np.max(z, axis=1)
#    return np.sum(np.log(np.sum(np.exp(z-m), axis=1)) + m - np.log(N-1) - 0.5*np.log(2*np.pi*b**2))/N

def _compute_gradloglike(data, weights, b=DEFAULT_B):
    '''
    compute dloglike/dlogb
    '''
    N = np.sum(weights)
    truth = np.ones_like(data, dtype=bool)

    bsqrd2 = 1./(2*b**2)
    grad = 0.
    for i in xrange(len(data)):
        truth[i-1] = True
        truth[i] = False
        z = -(data[i]-data[truth])**2*bsqrd2
        exp = np.exp(z-np.max(z))*weights[truth]
        grad += np.sum((-2*z - 1)*exp) / np.sum(exp) * weights[i]
    grad /= N

    return grad

def optimize_bandwidth(data, weights, min_b, max_b, rtol=1e-4, verbose=False):
    '''
    perform a bisection search in log(b) for the zero-crossing of gradloglike
    '''
    _min = _compute_gradloglike(data, weights, b=min_b)
    if _min <= 0:
        return min_b

    _max = _compute_gradloglike(data, weights, b=max_b)
    if _max >= 0:
        return max_b

    mid_b = (max_b*min_b)**0.5
    while max_b-min_b > rtol*mid_b:
        _mid = _compute_gradloglike(data, weights, b=mid_b)
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

def marginalize_bandwidth(data, weights, min_b, max_b, rtol=DEFAULT_RTOL, dlogl=DEFAULT_DLOGL, num_points=DEFAULT_NUM_POINTS, prior=DEFAULT_B_PRIOR, verbose=False):
    '''
    perform a series of bisection searches to define a grid in bandwidth over which we iterate and compute weights
    '''
    ### first, perform a bisection search to find the maximum likelihood
    ### leverage this to find the b-range corresponding to loglike >= max(loglike)-dlogl
    _min = _compute_gradloglike(data, weights, b=min_b)
    if _min <= 0: ### one of the boundaries is the maximum allowed, so we have a short-cut
        max_b = _find_b(data, weights, _compute_loglike(data, weights, min_b) - dlogl, min_b, max_b, rtol=rtol)

    else:
        _max = _compute_gradloglike(data, weights, b=max_b)
        if _max >= 0: ### the other boundary gives another short-cut
            min_b = _find_b(data, weights, _compute_loglike(data, weights, max_b) - dlogl, min_b, max_b, rtol=rtol)

        else: ### we have to do a bisection search
            _min_b = min_b ### remember these for later
            _max_b = max_b

            mid_b = (_max_b*_min_b)**0.5
            while _max_b-_min_b > rtol*mid_b:
                _mid = _compute_gradloglike(data, weights, b=mid_b)

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
            target = _compute_loglike(data, weights, b=mid_b) - dlogl ### this logl defines the boundaries of our marginalization

            min_b = _find_b(data, weights, target, min_b, mid_b, rtol=rtol, verbose=verbose) # find the lower b
            max_b = _find_b(data, weights, target, mid_b, max_b, rtol=rtol, verbose=verbose) # find the upper b

    ### we now have min_b, max_b defining the ranges we want, so let's define our grid
    if prior=='log':
        bs = np.logspace(np.log10(min_b), np.log10(max_b), num_points) ### choose logarithmic spacing -> flat prior in log(b)
    elif prior=='lin':
        bs = np.linspace(min_b, max_b, num_points)
    else:
        raise ValueError, 'prior=%s not understood!'%prior

    ws = np.array([_compute_loglike(data, weights, b) for b in bs]) ### NOTE: this is expensive... 
    ws = np.exp(ws-np.max(ws))
    ws /= np.sum(ws)

    if verbose:
        print 'grid is as follows:'
        for b, w in zip(bs, ws):
            print '\tb=', b, '\tw=', w

    return bs, ws

def _find_b(data, weights, target, min_b, max_b, rtol=DEFAULT_RTOL, verbose=False):
    '''
    find the b corresponding to loglike=target between min_b, max_b

    NOTE: we expect target to be smaller than the loglike at at least one of the specified min_b, max_b
    things may break if that is not true...
    '''
    _min = _compute_loglike(data, weights, min_b)
    _max = _compute_loglike(data, weights, max_b)

    if _min > _max: ### swap these
        max_b, min_b = min_b, max_b
        _max, _min = _min, _max

    mid_b = (min_b*max_b)**0.5
    while abs(max_b-min_b) > rtol*mid_b:
        _mid = _compute_loglike(data, weights, mid_b)

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

#---------------------------------------------------------------------------------------------------
### multi-dimensional routines

def whiten(points, samples, weights, priors, low=0.1, high=0.9, verbose=False):
    """
    assumes
        points.shape = Nfields, Npoints
        samples.shape = Nfields, Nsamp
        weights.shape = Nsamp,
        priors.shape = Nfields, 2
    and b should be a scalar
    """
    medians = np.median(samples, axis=1)
    iqrs = np.percentile(samples, 100*high, axis=1) - np.percentile(samples, 100*low, axis=1)

    if verbose:
        print('whitening marginal distributions')
        if len(samples.shape)==1:
            print('  median = %+.3e'%(medians))
            print('  IQR[%.2f,%.2f] = %+.3e'%(low, high, iqrs))

        else:
            for i, (m, s) in enumerate(zip(medians, iqrs)):
                print('  median(%01d) = %+.3e'%(i, m))
                print('  IQR[%.2f,%.2f](%01d) = %+.3e'%(low, high, i, s))

    wpoints = np.transpose((np.transpose(points) - medians)/iqrs)
    wsamples = np.transpose((np.transpose(samples) - medians)/iqrs)
    wpriors = np.transpose((np.transpose(priors) - medians)/iqrs)

    return (wpoints, wsamples, wpriors), (medians, iqrs)

def nd_kde(points, samples, weights, priors, b=DEFAULT_B, reflect=True):
    return _compute_nd_logkde(points, samples, weights, priors, b, reflect=reflect)

def _compute_nd_logkde(points, samples, weights, priors, b=DEFAULT_B, reflect=True):
    """
    given (Nfields, Nsamp, Npoints), we expect
        points.shape = Nfields, Npoints
        samples.shape = Nfields, Nsamp
        weights.shape = Nsamp,
        priors.shape = Nfields, 2
    and b should be a scalar

    returns logkde evaluated at the meshgrid result of points with ordering='ij'
    this includes reflecting each sample around the prior bounds
    """
    ### sanity check input
    Nfields, Npoints = points.shape
    nfields, Nsamp = samples.shape
    assert nfields==Nfields
    assert len(weights) == Nsamp
    assert len(weights.shape) == 1
    nfields, nbounds = priors.shape
    assert nfields==Nfields
    assert nbounds==2
    assert isinstance(b, (int, float))

    ### set up arrays
    Neval = Npoints**Nfields
    flat_logkde = np.empty((2, Neval), dtype='float')
    flat_logkde[0,:] = -np.infty

    flat_points = np.transpose(np.array([_.flatten() for _ in np.meshgrid(*points, indexing='ij')]))

    m = np.empty(Neval, dtype='float') ### used when computing a maximum

    ### set up kernel function on the fly
    f = -0.5/b**2
    g = Nfields*(0.5*np.log(2*np.pi) + np.log(b))
    _nd_logkernel = lambda vector: f*np.sum((flat_points - vector)**2, axis=1) - g

    ### iterate over samples
    for samp_ind, logw in enumerate(np.log(weights)):
        vect = samples[:,samp_ind] ### extract the location of this sample

        ### add in the contribution from that sample
        flat_logkde[1,:] = _nd_logkernel(vect) + logw
        m[:] = np.max(flat_logkde, axis=0)
        flat_logkde[0,:] = np.log(np.sum(np.exp(flat_logkde-m), axis=0))+m

        ### iterate over fields, adding reflected contribution for each field
        if reflect: ### add in points for reflecting boundary conditions
            for field_ind in xrange(Nfields):
                x = vect[field_ind] # remember this becuase it is used repeatedly

                # iterate over limits, adding a mirror image across each boundary
                for lim in priors[field_ind]:
                    vect[field_ind] = 2*lim - x
                    flat_logkde[1,:] = _nd_logkernel(vect) + logw
                    m[:] = np.max(flat_logkde, axis=0)
                    flat_logkde[0,:] = np.log(np.sum(np.exp(flat_logkde-m), axis=0))+m

                vect[field_ind] = x ### re-set the value to what it was initially

    ### we're done with this, so we normalize by the number of samples
    ### NOTE: assuming the whole integral is N will only be reasonable if the bandwidth used is much smaller than the width of the prior
    flat_logkde -= np.log(Nsamp) + np.log(np.sum(weights))

    ### reshape and return
    return np.reshape(flat_logkde[0,:], (Npoints,)*Nfields)

def nd_marg(points, logkde, axis=-1):
    """
    return a marginalized form of logkde, marginalizing over the last axis of both points and logkde
    assumes
        points.shape = Nfields, Npoints
        logkde.shape = (Npoints,)*Nfields
    """
    ### make sure axis is the last axis (-1)
    N = len(points)
    trans = range(N)
    trans.pop(axis) ### remove this
    trans.append(axis) ### add it
    logkde = logkde.transpose(*trans)
    t = np.ones(N, dtype=bool)
    t[axis] = False

    m = np.max(logkde, axis=-1)
    return points[t], np.log(np.trapz(np.exp(logkde-np.outer(m, np.ones(logkde.shape[0])).reshape(logkde.shape)), points[axis], axis=-1)) + m

def nd_marg_leave1(points, logkde):
    return nd_marg(*nd_marg_leave2(points, logkde))

def nd_marg_leave2(points, logkde):
    N = len(points) ### number of fields
    if N > 2:
        return nd_marg_leave2(*nd_marg(points, logkde))
    else:
        return points, logkde

def nd_norm(points, logkde):
    """
    compute the norm of a multi-dimensional kde
    does this by direct marginalization vi numpy.trapz

    assumes
        points.shape = Nfields, Npoints
        logkde.shape = (Npoints,)*Nfields
    """
    return nd_marg(*nd_marg_leave1(points, logkde))[1] ### only return the final thing

def logkde2limits(points, logkde, level):
    """
    computes the upper limit derived from a kde at level
    assumes points and logkde are 1-dimensional
    """
    kde = np.exp(logkde-np.max(logkde))
    integral = np.concatenate((np.array([0]), np.cumsum(0.5*(kde[1:]+kde[:-1])*np.diff(points))))
    integral /= integral[-1] ### make sure this is normalized appropriately
    return np.interp(1-level, integral, points), np.interp(level, integral, points)

