from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# std
from collections import OrderedDict

# Third-party
from astropy import log as logger
import astropy.units as u
import h5py
import matplotlib.pyplot as pl
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import scoreatpercentile

def sky_cut(ra, dec, lims):
    idx = ((ra > lims['ra'][0]) & (ra < lims['ra'][1]) &
           (dec > lims['dec'][0]) & (dec < lims['dec'][1]))
    return idx

def plot_cmd(X, g_lim, weights=None, nbins=64, smooth=0.02):
    x = X[:,1]
    y = X[:,0]
    xbins = np.linspace(0., 0.7, nbins)
    ybins = np.linspace(g_lim[0], g_lim[1], nbins)

    H,xedges,yedges = np.histogram2d(x, y,
                                     bins=(xbins,ybins), weights=weights)
    xx,yy = np.meshgrid((xedges[1:]+xedges[:-1])/2., (yedges[1:]+yedges[:-1])/2.)

    if smooth:
        H = gaussian_filter(H, sigma=smooth)

    fig,ax = pl.subplots(1,1,figsize=(4,6))
    ax.pcolormesh(xx, yy, H.T, cmap='Blues')
    ax.set_ylim(*g_lim[::-1])
    ax.set_xlim(0., 0.7)
    return fig

def main(XCov_filename, bin_size, threshold=None):
    sky_lim = dict(ra=(220,240), dec=(-32,-12))
    g_lim = (17., 20.6)

    with h5py.File(XCov_filename, mode='r') as f:
        allX = f['all']['X'][:]
        pre_filter_ix = np.ones_like(allX[:,0]).astype(bool)

        # magnitude cut ----------------------------------------
        pre_filter_ix &= (allX[:,0] > g_lim[0]) & (allX[:,0] < g_lim[1])

        # sky cut
        pre_filter_ix &= sky_cut(f['all']['ra'][:], f['all']['dec'][:],
                                 lims=sky_lim)

        ra = f['all']['ra'][:][pre_filter_ix]
        dec = f['all']['dec'][:][pre_filter_ix]
        allX = allX[pre_filter_ix]

        non_ll = f['all']['noncluster_log_likelihood'][:][pre_filter_ix]
        iso_lls = OrderedDict()
        for dm in np.arange(14., 17+0.2, 0.2):
            str_dm = "{:.2f}".format(dm)
            longname = 'isochrone_{}_log_likelihood'.format(str_dm)
            if longname in f['all'].keys():
                iso_lls[str_dm] = f['all'][longname][:][pre_filter_ix]
        # ------------------------------------------------------

        # compute weights
        all_ll = OrderedDict()
        all_weights = OrderedDict()
        for name,ll in iso_lls.items():
            all_ll[name] = ll - non_ll
            all_weights[name] = np.exp(all_ll[name])
            all_weights[name][all_ll[name] > threshold] = np.exp(threshold)

        # bin by position on the sky
        ra_bins = np.arange(ra.min(), ra.max(), step=bin_size.to(u.degree).value)
        dec_bins = np.arange(dec.min(), dec.max(), step=bin_size.to(u.degree).value)
        logger.debug("{} RA bins, {} Dec bins".format(len(ra_bins), len(dec_bins)))

        vmin = None
        vmax = None
        for name in iso_lls.keys():
            logger.debug("Making histogram for {}".format(name))
            H,ra_edges,dec_edges = np.histogram2d(ra, dec,
                                                  bins=(ra_bins, dec_bins),
                                                  weights=all_weights[name], normed=True)
            ra_mesh,dec_mesh = np.meshgrid((ra_edges[1:]+ra_edges[:-1])/2,
                                           (dec_edges[1:]+dec_edges[:-1])/2)

            if vmin is None:
                Hrav = H.ravel()
                vmin, vmax = scoreatpercentile(Hrav, [15,75])

            pl.figure(figsize=(8,8))
            pl.pcolormesh(ra_mesh, dec_mesh, H.T,
                          cmap='Greys', vmin=vmin, vmax=vmax)
            pl.xlim(sky_lim['ra'][1], sky_lim['ra'][0])
            pl.ylim(sky_lim['dec'][0], sky_lim['dec'][1])
            pl.savefig("plots/ngc5897/iso_{}_thresh_{}_eq.png".format(name, threshold), dpi=300)

            fig = plot_cmd(allX, g_lim, weights=all_weights[name])
            fig.savefig("plots/ngc5897/cmd_{}_thresh_{}.png".format(name, threshold), dpi=300)

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")

    parser.add_argument("-f", "--xcov-filename", dest="XCov_filename", required=True,
                        type=str, help="Full path to XCov file")
    parser.add_argument("--bin-size", dest="bin_size", default="8 arcmin",
                        type=str, help="Value and unit for sky bin size.")
    parser.add_argument("-t", "--thresh", dest="threshold", required=True,
                        type=float, help="")
    # parser.add_argument("-n", "--normalize", action="store_true", dest="normalize",
    #                     default=False, help="Normalize the histogram.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    val,unit = args.bin_size.split()
    bin_size = float(val) * u.Unit(unit)

    main(args.XCov_filename, bin_size=bin_size, threshold=args.threshold) #, normalize=args.normalize)
