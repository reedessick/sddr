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

def kde2fig(points, logkde, priors, fields, sanitycheck_tuple=None, levels=[], log=True):
    """
    generate something like a corner plot for logkde
    assumes
        points.shape = Nfields, Npoints
        logkde.shape = (Npoints,)*Nfields
        priors.shape = Nfields, 2
        len(fields) = Nfields

    sanitycheck_tuple is either None or (samples, weights, wpoints, wsamples, wpriors, b)

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


    sanitycheck = sanitycheck_tuple is not None
    if sanitycheck:
        samples, weights, wpoints, wsamples, wpriors, b = sanitycheck_tuple
        Nsamp = samples.shape[1]
    
    ### set up helper arrays
    tmp = np.empty(Npoints, dtype='float')
    tmp_field = None

    # make a copy so I can mess around without screwing up original array
    dummy_logkde = np.empty_like(logkde, dtype='float')
    dummy_points = np.empty((Nfields, Npoints), dtype='float')

    ### plot this stuff!
    fig = plt.figure()

    for row in xrange(Nfields):
        for col in xrange(row+1):
            ax = plt.subplot(Nfields, Nfields, row*Nfields+col+1)

            trans = range(Nfields)
            dummy_logkde[...] = logkde[...]
            dummy_points[...] = points[...]

            if row==col: ### marginalize away everything except for this row, plot result

                # put col as the first index
                trans[0] = col
                trans[col] = 0
                dummy_logkde[...] = dummy_logkde.transpose(*trans)[...]

                tmp[:] = dummy_points[col,:]
                dummy_points[col,:] = dummy_points[0,:]
                dummy_points[0,:] = tmp[:]

                # actually plot
                tmp[:] = utils.nd_marg_leave1(dummy_points, dummy_logkde)[1]
                tmp -= np.max(tmp)
                tmp -= np.log(np.sum(np.exp(tmp)))

                if log:
                    ax.plot(dummy_points[0], tmp, color='b')

                else:
                    ax.plot(dummy_points[0], np.exp(tmp), color='b')

                # decorate
                ax.set_xlim(priors[row])

                if sanitycheck:
                    ### add direct 1D marginalization
                    y = np.array([utils._compute_logkde(wp, wsamples[col], weights, b=b, prior_min=wpriors[col][0], prior_max=wpriors[col][1]) for wp in wpoints[col]])
                    y -= np.max(y)
                    y -= np.log(np.sum(np.exp(y)))
                    ax.plot(points[col], np.exp(y), color='k', linestyle='dashed')

                    ### add a histogram
                    ax.hist(samples[col], bins=points[col], histtype='step', color='g', weights=np.ones(Nsamp, dtype=float)/Nsamp)

            else: ### marginalize everything besides these rows and columns

                if sanitycheck:
                    ### add data points
                    counts, xedges, yedges, _ = ax.hist2d(samples[col], samples[row], bins=max(10, int(0.1*len(samples[col]))**0.5), cmap='Greens') #, norm=matplotlib.colors.LogNorm())

                    ax.scatter(samples[col], samples[row], marker='.', s=1, color='k', alpha=0.1)

                    # add marginalized histograms to diagonal axes
                    margx = 1.*np.sum(counts, axis=1)/Nsamp * (len(xedges)-1)/(len(points[row])-1)
                    plt.figure(fig.number)
                    plt.subplot(Nfields, Nfields, Nfields*col+col+1).plot((xedges[1:]+xedges[:-1])/2., margx, color='m')

                    margy = 1.*np.sum(counts, axis=0)/Nsamp * (len(yedges)-1)/(len(points[col])-1)
                    plt.figure(fig.number)
                    plt.subplot(Nfields, Nfields, Nfields*row+row+1).plot((yedges[1:]+yedges[:-1])/2., margy, color='m')

                # put the column in the first index
                trans[0] = col
                trans[col] = 0

                x = trans[1]
                trans[1] = row
                trans[row] = x

                dummy_logkde[...] = dummy_logkde.transpose(*trans)[...]

                tmp[:] = dummy_points[col,:]
                dummy_points[col,:] = dummy_points[0,:]
                dummy_points[0,:] = tmp[:]

                tmp[:] = dummy_points[row,:]
                dummy_points[row,:] = dummy_points[1,:]
                dummy_points[1,:] = tmp[:]

                # actually plot
                ans = utils.nd_marg_leave2(dummy_points, dummy_logkde)[1].transpose() ### FIXME: fill in a place-holder
                if log:
                    if levels:
                        ax.contour(dummy_points[0], dummy_points[1], ans, levels=np.log(np.kde2levels(np.exp(ans), levels)), colors='b')
                    else:
                        ax.contour(dummy_points[0], dummy_points[1], ans, colors='b')
                else:
                    if levels:
                        ax.contour(dummy_points[0], dummy_points[1], np.exp(ans), levels=kde2levels(np.exp(ans), levels), colors='b')
                    else:
                        ax.contour(dummy_points[0], dummy_points[1], np.exp(ans), colors='b')

                # decorate
                ax.set_xlim(priors[col])
                ax.set_ylim(priors[row])

            if col > 0:
                plt.setp(ax.get_yticklabels(), visible=False)
            else:
                ax.set_ylabel(fields[row])

            if row < Nfields-1:
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                ax.set_xlabel(fields[col])
            
    plt.subplots_adjust(
        hspace=0.05,
        wspace=0.05,
    )
    return fig

def kde2levels(kde, levels):
    kde = kde.flatten()

    order = kde.argsort()[::-1] ### largest to smallest
    ckde = np.cumsum(kde[order]) ### cumulative distribution
    ckde /= np.sum(kde)

    ans = []
    for level in levels: ### iterate through levels, returning the kde value associated with that confidence
                         ### assume kde spacing is close enough that interpolation isn't worth while...
        ans.append(kde[order[ckde<=level][-1]])

    return ans
