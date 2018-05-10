__doc__ = "a module holding utility functions for sddr estimation"
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------

import numpy as np

from scipy.special import erf
from scipy.special import digamma
from scipy.special import polygamma

from scipy.stats import beta

#-------------------------------------------------

LOG3 = np.log(3)
DEFAULT_NUM_POINTS = 101

DEFAULT_B = 0.1
DEFAULT_PRIOR_MIN = -10
DEFAULT_PRIOR_MAX = -4

DEFAULT_NUM_SUBSETS

#-------------------------------------------------

def load_hdf5(paths, verbose=False):
    """
    load in samples from an hdf5 file
    """
    samples = None
    for path in args:
        if verbose:
            print('reading samples from: '+path)
        with h5py.File(path, 'r') as file_obj:
            new = file_obj['lalinference/lalinference_mcmc/posterior_samples'][...]
        if verbose:
            print('    found %d samples'%len(new))
        new = new[len(new)/2:] ### throw out the burn in...
        if verbose:
            print('    retained %d samples'%len(new))

        if samples is None:
            samples = new
        else:
            samples = np.concatenate((samples, new))

    if verbose:
        print('retained %d samples in all'%len(samples))

    return samples

def partition(data, num_subsets=DEFAULT_NUM_SUBSETS):
    '''
    partition samples into separate subsets
    '''
    N=len(samples)
    subsets = [np.zeros(N, dtype=bool) for _ in xrange(opts.num_subsets)]
    for i in xrange(N):
        n = i%opts.num_subsets
        subsets[n][i] = True

    return [data[truth] for truth in subsets]

#-------------------------------------------------

def hist(data, b=None, prior_min=DEFAULT_PRIOR_MIN, prior_max=DEFAULT_PRIOR_MAX):
    '''
    compute a simple histogram and return the normalized count in the first bin
    '''
    if b is None:
        b = 10/len(data)**0.5

    n, b = np.histogram(data, b, range=(prior_min, prior_max))
    return np.log(n[0]) - np.log(b[1]-b[0]) - np.log(np.sum(n))

def chist(data, b=None, deg=2, fit_max=DEFAULT_PRIOR_MAX, prior_min=DEFAULT_PRIOR_MIN, prior_max=DEFAULT_PRIOR_MAX):
    '''
    compute a cumulative histogram and fit it with a low-order polynomial, returning the slope at the prior_min
    '''
    if b is None:
        b = 2*len(data)

    n, b = np.histogram(data, b, range=(prior_min, prior_max))
    c = np.cumsum(n) ### make a cumulative histogram
    c /= c[-1] ### normalize this to 1

    ### fit to a low-order polynomial
    B = 0.5*(b[1:]+b[:-1])
    truth = B<=fit_max
    params = np.polyfit(B[truth], c[truth], deg)

    return np.log(np.sum([(deg-i)(prior_min)**(deg-i-1)*params[i] for i in xrange(deg+1)]))

def kde(data, b=DEFAULT_B, prior_min=DEFAULT_PRIOR_MIN, prior_max=DEFAULT_PRIOR_MAX):
    '''
    compute a kde at prior_min
    '''
    """
    obj = FixedBandwidth1DKDE(
        data,
        num_points=DEFAULT_NUM_POINTS,
        b=DEFAULT_B,
        compute=False,
        prior_min=prior_min,
        prior_max=prior_max,
    )
    obj.optimize()
    obj.compute()

    return obj.logpdf(prior_min)
    """

    return _compute_logkde(prior_min, data, b=b, prior_min=prior_min, prior_max=prior_max)

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

#-------------------------------------------------

### PROBABLY DEPRECATED BELOW THIS POINT!!!

#-------------------------------------------------

class FixedBandwidth1DKDE(object):
    """
    an object that represents the calibration mapping between SupervisedClassifier output (ranks \in [0, 1]) and probabilistic statements
    essentially a fancy interpolation device

    SupervisedClassifier and IncrementalSupervisedClassifier objects retain a reference to one of these
    functionality should be universal for all classifiers!

    We accomplish this with a standard fixed-bandwidth Gaussian KDE, represented internally as a vector that's referenced via interpolation for rapid execution.
    """

    def __init__(
            self,
            observations,
            num_points=DEFAULT_NUM_POINTS,
            b=DEFAULT_B,
            compute=True,
            prior_min=DEFAULT_PRIOR_MIN,
            prior_max=DEFAULT_PRIOR_MAX,
        ):
        self._obs = np.array([])
        self._interp_x = np.linspace(prior_min, prior_max, num_points)
        self._prior_min = prior_min
        self._prior_max = prior_max
        self._b = b
        self.append(observations, compute=compute) ### automatically calls self.compute() after appending samples

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, b):
        self._b = b

    @property
    def obs(self):
        return self._obs

    @property
    def interp_x(self):
        return self._interp_x

    @property
    def interp_logpdf(self):
        return self._interp_logpdf

    @property
    def interp_cdf(self):
        return self._interp_cdf

    @property
    def interp_pdf_alpha(self):
        return self._interp_pdf_alpha

    @property
    def interp_pdf_beta(self):
        return self._interp_pdf_beta

    @property
    def interp_cdf_alpha(self):
        return self._interp_cdf_alpha

    @property
    def interp_cdf_beta(self):
        return self._interp_cdf_beta

    def compute(self):
        """
        compute the interpolation arrays used for fast execution later
        this includes error estimates, which we compute by fitting beta distributions to the first 2 moments of the pdf and cdf, respectively
        """
        ### iterate over observations to compute KDE
        num_points = len(self._interp_x)
        ib2 = self.b**-2
        ibsqrt2 = 1./(2**0.5 * self.b)
        f = 0.5*ib2

        ### set up things for pdf
        self._interp_logpdf = np.empty((num_points, 4), dtype='float') ### we do this to avoid re-allocating memory repeatedly
        self._interp_logpdf[:,0] = -np.infty

        _interp_2logpdf = np.empty((num_points, 2), dtype='float')
        _interp_2logpdf[:,0] = -np.infty

        ### set up things for cdf
        _interp_erf = 0.
        _interp_2erf = 0.

        count = 0.
        ### iterate
        priormin2 = 2*self._prior_min
        priormax2 = 2*self._prior_max

        for rank in self.obs:
            x0 = self._interp_x-rank
            x1 = self._interp_x-priormin2+rank
            x2 = self._interp_x-priormax2+rank
            z0 = -x0**2
            z1 = -x1**2
            z2 = -x2**2

            ### compute pdf
            self._interp_logpdf[:,1] = f*z0
            self._interp_logpdf[:,2] = f*z1
            self._interp_logpdf[:,3] = f*z2
            m = np.max(self._interp_logpdf[:,1:], axis=1)
            self._interp_logpdf[:,1] = np.log(np.sum(np.exp(self._interp_logpdf[:,1:].transpose()-m), axis=0))+m - LOG3 ### NOTE, we divide pdf by 2 so that the thing we fit with a beta-distribution is bound between 0 and 1.

            m = np.max(self._interp_logpdf[:,:2], axis=1)
            self._interp_logpdf[:,0] = np.log(np.sum(np.exp(self._interp_logpdf[:,:2].transpose()-m), axis=0))+m

            ### compute 2nd moment for pdf error estimates
            _interp_2logpdf[:,1] = self._interp_logpdf[:,1]*2
            m = np.max(_interp_2logpdf, axis=1)
            _interp_2logpdf[:,0] = np.log(np.sum(np.exp(_interp_2logpdf.transpose()-m), axis=0))+m

            ### compute 2nd momemt for cdf error estimates
            erf0 = erf((self._prior_min-rank)*ibsqrt2)
            erf1 = erf((rank-self._prior_min)*ibsqrt2)
            erf2 = erf((prior_min-2*self._prior_max+rank)*ibsqrt2)

            c = 0.5*(erf(x0*ibsqrt2) - erf0 + erf(x1*ibsqrt2) - erf1 + erf(x2*ibsqrt2) - erf2)
            _interp_erf += c
            _interp_2erf += c**2

            count += 0.5*(\
                erf((prior_max-rank)*ibsqrt2) - erf0 \
              + erf((prior_max-2*prior_min+rank)*ibsqrt2) - erf1 \
              + erf((-prior_max+rank)*ibsqrt2) - erf2\
            )

        self._interp_logpdf = self._interp_logpdf[:,0] ### only retain the final sum
        _interp_2logpdf = _interp_2logpdf[:,0]

        ### compute parameters for error estimates
        ### NOTE: if the distributions don't really fit withih rank \in [0,1], doing this may bias our error estimates!

        # NOTE: for a beta distribution described by 2 parameters: alpha, beta
        #  pdf(x|alpha, beta) = Gamma(alpha+beta)/(Gamma(alpha)*Gamma(beta)) * x**(alpha-1) * (1-x)**(beta-1)
        #  E[x] = alpha/(alpha+beta)
        #  Var[x] = alpha*beta/((alpha+beta)**2 * (alpha+beta+1))

        inum_obs = 1./count

        # pdf
        Ep = np.exp(self._interp_logpdf)*inum_obs # expected value
        Vp = (np.exp(_interp_2logpdf)*inum_obs - Ep**2)*inum_obs # variance
        ### handle numerical error explicitly by imposing a minimum variance
        ###   assume samples are distributed according to a Gaussian with stdv ~0.1*self.b, which is much smaller than we can resolve
        ###   minVp = 0.5*(0.1)**4 / num_obs
        minVp = Ep*inum_obs
        Vp[Vp<minVp] = minVp ### handle numerical error explicitly

        self._interp_pdf_alpha, self._interp_pdf_beta = self._cumulants2params(Ep, Vp)

        # cdf
        Ec = _interp_erf*inum_obs ### expected value
        Vc = (_interp_2erf*inum_obs - Ec**2)*inum_obs ### variacne

        ### handle numerical error explicitly by imposing a minimum variance
        ### NOTE: we use the same mimimum as Vp without much justification...
        minVc = Ec*inum_obs
        Vc[Vc<minVp] = minVc ### handle numerical error explicitly

        self._interp_cdf_alpha, self._interp_cdf_beta = self._cumulants2params(Ec, Vc)

        ### normalize the pdf, compute the cdf
        ### NOTE: if the distributions are entirely contained within rank \in [0,1], these should be equivalent to Ep and Ec, respectively

        self._interp_logpdf -= 0.5*np.log(2*np.pi*self.b**2) + np.log(count) - LOG3 ### NOTE: we do NOT normalize pdfs so they integrate to one with rank \in [0, 1]. USER BEWARE!

        self._interp_cdf = Ec

        ### handle numerical stability issues explicitly
        self._interp_cdf[self._interp_cdf<0] = 0
        self._interp_cdf[self._interp_cdf>1] = 1

    def _cumulants2params(self, E, V, safe=False):
        """
        compute the beta-distribution parameters corresponding to this expected value and variance
        """
        ### now actually compute things
        x = ((1.-E)*E - V)
        alpha = (E/V)*x
        beta = ((1.-E)/V)*x
        
        bad = alpha<=0
        if np.any(bad):
            if safe:
                raise Warning, 'alpha<=0'
            alpha[bad] = 1e-8 ### FIXME, this hard-coding could be bad, but we're talking about when my model breaks down anyway...
            beta[bad] = 1e-2

        bad = beta <= 0
        if np.any(bad):
            if safe:
                raise Warning, 'beta<=0'
            beta[bad] = 1e-8 ### FIXME, this hard-coding could be bad, but we're talking about when my model breaks down anyway...
            alpha[bad] = 1e-2

        return alpha, beta

    def _grad(self, b):
        r"""
        computes dlogL/dlogb at b

            dlogL/dlogb = \sum_i \frac{sum_{j \neq i} ((x_i-x_j)**2/b**2 - 1)*exp(-0.5*(x_i-x_j)**2/b**2)}{\sum_{j \neq i} exp(-0.5*(x_i-x_j)**2/b**2)}
        """
        b2 = b**2
        dlogL = 0.

        truth = np.ones_like(self.obs, dtype=bool)
        for ind, rank in enumerate(self.obs):
            truth[ind-1] = True
            truth[ind] = False

            z = (rank-self.obs[truth])**2/b2

            logden = -0.5*z
            m = np.max(logden)
            weights = np.exp(logden-m)

            dlogL += np.sum(weights*(z - 1)) / np.sum(weights)

        return dlogL

    def optimize(self, minb=1e-4, maxb=1.0, tol=1e-6, **kwargs):
        """
        looks for the maximum of logL via Newton's method for the zeros of dlogL/dlogb
        expects dlogL/dlogb to be monotonic in logb, which will likely be true. However, if it is not then the logic in this loop may fail.
        """
        ### check if we're already within tolerance
        if maxb - minb <= minb*tol:
            self.b = (minb*maxb)**0.5
            return None

        ### check if end points are optima
        mdlogL = self._grad(minb)
        if mdlogL <= 0:
            self.b = minb
            return None

        MdlogL = self._grad(maxb)
        if MdlogL >=0:
            self.b = maxb
            return None

        # iterate through bisection search until termination condition is reached
        while maxb - minb > minb*tol:
            midb = (minb*maxb)**0.5
            dlogL = self._grad(midb)
            if dlogL == 0:
                self.b = midb
                return None

            elif dlogL > 0:
                minb = midb

            else: # dlogL < 0
                maxb = midb

        self.b = (minb*maxb)**0.5
        return None

    def append(self, newobservations, compute=True):
        """
        appends new observations and recomputes interpolation arrays
        """
        if isinstance(newobservations, (int, float)):
            self._obs = np.concatenate((self._obs, np.array([newobservations])))
        else:
            self._obs = np.concatenate((self._obs, newobservations))
        self.compute()

        if compute:
            self.compute()

    def cdf(self, ranks):
        return np.interp(ranks, self._interp_x, self._interp_cdf)

    def cdf_alpha(self, ranks):
        '''
        NOTE: because force the cdf to go through (0,0) and (1,1), the error estimates are messed up near the end points because I can't represent that with a beta function. This turns out to also affect the second-to-last point as well because of how np.interp works (the nan from the last point get mixed in)
        '''
        return np.interp(ranks, self._interp_x, self._interp_cdf_alpha)

    def cdf_beta(self, ranks):
        '''
        NOTE: because force the cdf to go through (0,0) and (1,1), the error estimates are messed up near the end points because I can't represent that with a beta function. This turns out to also affect the second-to-last point as well because of how np.interp works (the nan from the last point get mixed in)
        '''
        return np.interp(ranks, self._interp_x, self._interp_cdf_beta)

    def cdf_std(self, ranks):
        return np.array([beta.std(self.cdf_alpha(rank), self.cdf_beta(rank)) for rank in ranks])

    def cdf_var(self, ranks):
        return np.array([beta.var(self.cdf_alpha(rank), self.cdf_beta(rank)) for rank in ranks])

    def cdf_quantile(self, ranks, q):
        '''
        NOTE: because force the cdf to go through (0,0) and (1,1), the error estimates are messed up near the end points because I can't represent that with a beta function. This turns out to also affect the second-to-last point as well because of how np.interp works (the nan from the last point get mixed in)
        '''
        return np.array([beta.ppf(q, self.cdf_alpha(rank), self.cdf_beta(rank)) for rank in ranks])

    def quantile(self, q):
        return np.interp(q, self._interp_cdf, self._interp_x)

    def pdf(self, ranks):
        return np.exp(self.logpdf(ranks))

    def pdf_alpha(self, ranks):
        return np.interp(ranks, self._interp_x, self._interp_pdf_alpha)

    def pdf_beta(self, ranks):
        return np.interp(ranks, self._interp_x, self._interp_pdf_beta)

    def pdf_std(self, ranks):
        return np.array([beta.std(self.pdf_alpha(rank), self.pdf_beta(rank)) for rank in ranks])

    def pdf_var(self, ranks):
        return np.array([beta.var(self.pdf_alpha(rank), self.pdf_beta(rank)) for rank in ranks])

    @property
    def _pdf_scale(self):
        return 3./((2*np.pi)**0.5 * self.b)

    def pdf_quantile(self, ranks, q):
        scale = self._pdf_scale
        return np.array([beta.ppf(q, self.pdf_alpha(rank), self.pdf_beta(rank), loc=0, scale=scale) for rank in ranks])

    def logpdf(self, ranks):
        '''
        NOTE: this returns log(E[pdf]), not E[log(pdf)]
        '''
        return np.interp(ranks, self._interp_x, self._interp_logpdf)

    def logpdf_mean(self, ranks):
        '''
        NOTE: this returns E[log(pdf)] whereas logpdf returns log(E[pdf])
        '''
        scale = np.log(self._pdf_scale)
        return np.array([digamma(self.pdf_alpha(rank)) - digamma(self.pdf_alpha(rank)+self.pdf_beta(rank)) for rank in ranks]) - scale

    def logpdf_std(self, ranks):
        return self.logpdf_var(ranks)**0.5

    def logpdf_var(self, ranks):
        scale = np.log(self._pdf_scale)
        return np.array([polygamma(1, self.pdf_alpha(rank)) - polygamma(1, self.pdf_alpha(rank)+self.pdf_beta(rank)) for rank in ranks]) - scale

    def logpdf_quantile(self, ranks, q):
        return np.log(self.pdf_quantile(ranks, q))

def loglike(data, b=DEFAULT_B, prior_min=DEFAULT_PRIOR_MIN, prior_max=DEFAULT_PRIOR_MAX):
    '''
    compute the leave-one-out loglikelihood
    '''
    N = len(data)
    truth = np.ones(N, dtype=bool)

    logL = 0.
    for i in xrange(N): ### FIXME: can I do this without just iterating (which is kinda slow)?
        truth[i-1] = True
        truth[i] = False
        logL += logkde(data[i], data[truth], b=b, prior_min=prior_min, prior_max=prior_max)
    logL /= N

    return N
