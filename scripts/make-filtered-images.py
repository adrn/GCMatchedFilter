from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
from astropy import log as logger
import astropy.units as u
import h5py
import matplotlib.pyplot as pl
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import threshold

def sky_cut(ra, dec, lims):
    idx = ((ra > lims['ra'][0]) & (ra < lims['ra'][1]) &
           (dec > lims['dec'][0]) & (dec < lims['dec'][1]))
    return idx

def color_cut(X, lims):
    idx = np.ones(X.shape[0]).astype(bool)
    #for i,(lo,hi) in enumerate(lims):
    #    idx &= (X[:,i] > lo) & (X[:,i] < hi)

    # use diagonal lines instead of a box
    gr_line = lambda x: 10.*x + 15.3
    gi_line = lambda x: 7.*x + 15.25
    #gz_line = lambda x: 4.*x + 16.75
    idx &= (X[:,0] > gr_line(X[:,1])) & (X[:,0] < (gr_line(X[:,1])+5))
    idx &= (X[:,0] > gi_line(X[:,2])) & (X[:,0] < (gi_line(X[:,2])+5))

    # remove the disk instead of boxing the cluster
    #disk_box = X[:,0] < 18.
    #lims = [(0.3,0.42), (0.38, 0.55), (0.4, 0.6)]
    #for i,(lo,hi) in enumerate(lims):
    #    disk_box &= (X[:,i+1] > lo) & (X[:,i+1] < hi)
    #idx &= np.logical_not(disk_box)

    idx &= (X[:,0] > 17.5) & (X[:,0] < 21.)
    
    return idx

def plot_cmd(X, weights=None, nbins=64, smooth=0.02):
    x = X[:,1]
    y = X[:,0]
    xbins = np.linspace(-0.5, 1.75, nbins)
    ybins = np.linspace(14, 22, nbins)

    H,xedges,yedges = np.histogram2d(x, y,
                                     bins=(xbins,ybins), weights=weights)
    xx,yy = np.meshgrid((xedges[1:]+xedges[:-1])/2., (yedges[1:]+yedges[:-1])/2.)

    if smooth:
        H = gaussian_filter(H, sigma=smooth)
    
    fig,ax = pl.subplots(1,1,figsize=(6,6))
    ax.pcolormesh(xx, yy, H.T, cmap='Blues')
    ax.set_ylim(22, 14)
    ax.set_xlim(-0.5, 1.75)
    return fig

def main(XCov_filename, bin_size, normalize=False):
    # HACK: start with a small region, and a color cut
    sky_lims = {'ra': (225,235), 'dec': (-24,-18)}
    # color_lims = [(15.5,20.5), (-0.4,0.6), (-0.65,0.9), (-0.9,0.9)] # includes HB
    # color_lims = [(17.5,20.75), (-0.1,0.6), (-0.2,0.9), (-0.2,0.9)]
    color_lims = [(17.5,20.75), (-0.1,0.6), (-0.2,0.9), (-0.2,0.9)]

    with h5py.File(XCov_filename, mode='r') as f:
        ra = f['all']['ra'][:]
        dec = f['all']['dec'][:]
        X = f['all']['X'][:]

        # c_ll = f['all']['cluster_log_likelihood'][:]
        c_ll = f['all']['isochrone_log_likelihood'][:]
        nc_ll = f['all']['noncluster_log_likelihood'][:]
        ll = c_ll - nc_ll
        #ll = c_ll

        ll = threshold(ll, threshmax=50)

        if normalize:
            weights = np.exp(ll - ll.max())
        else:
            weights = np.exp(ll)
        
        weights[np.isinf(weights)] = 0.

        if np.isnan(weights).any() or np.isinf(weights).any():
            logger.warning("NaN or Inf weights!")

        ix = np.ones_like(ra).astype(bool)
        #ix &= sky_cut(ra, dec, lims=sky_lims)
        ix &= color_cut(X, lims=color_lims)
        X = X[ix]
        c_ll = ll[ix]
        nc_ll = nc_ll[ix]
        ll = ll[ix]
        ra = ra[ix]
        dec = dec[ix]
        weights = weights[ix]

        #ix = np.isfinite(weights)
        #X = X[ix]
        #weights=weights[ix]
        #c_ll = c_ll[ix]
        #nc_ll = nc_ll[ix]
        #ll = ll[ix]
        #ra = ra[ix]
        #dec = dec[ix]

        #pl.hist(weights)
        #pl.savefig("/u/10/a/amp2217/public_html/plots/test.png", dpi=300)

        #fig,ax=pl.subplots(1,1)
        #ax.hist(np.exp(nc_ll)[np.isfinite(nc_ll)], bins=128)
        #fig.savefig("/u/10/a/amp2217/public_html/plots/test_hist.png", dpi=300)
        #return

        # plot CMD weighted by LL before color cuts
        fig = plot_cmd(X, weights=weights, nbins=128)
        fig.savefig("/u/10/a/amp2217/public_html/plots/cmd_gi_ll_diff.png", dpi=300)
        fig = plot_cmd(X, weights=np.exp(c_ll), nbins=128)
        fig.savefig("/u/10/a/amp2217/public_html/plots/cmd_gi_ll_c.png", dpi=300)
        fig = plot_cmd(X[np.isfinite(nc_ll)], weights=np.exp(nc_ll[np.isfinite(nc_ll)]), nbins=128)
        fig.savefig("/u/10/a/amp2217/public_html/plots/cmd_gi_ll_nc.png", dpi=300)

        ra_bins = np.arange(ra.min(), ra.max(), step=bin_size.to(u.degree).value)
        dec_bins = np.arange(dec.min(), dec.max(), step=bin_size.to(u.degree).value)
        logger.debug("{} RA bins, {} Dec bins".format(len(ra_bins), len(dec_bins)))
        
        H,ra_edges,dec_edges = np.histogram2d(ra, dec,
                                              bins=(ra_bins,dec_bins), weights=weights,
                                              normed=normalize)
        H_sigma = (3*u.arcmin / bin_size).decompose().value
        H = gaussian_filter(H, sigma=H_sigma)

        H = np.fliplr(H.T)

        from astropy.io import fits
        hdu = fits.PrimaryHDU(H)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto("/u/10/a/amp2217/public_html/plots/test.fits", clobber=True)

        fig,axes = pl.subplots(1, 2, figsize=(10,5), sharey=True)
        for i in range(2):
            axes[i].plot(X[:,i+1], X[:,0], marker=',', linestyle='none', alpha=0.02)
        axes[0].set_xlim(-0.5, 1.1)
        axes[1].set_xlim(-0.6, 1.8)
        axes[0].set_ylim(axes[0].get_ylim()[::-1])
        fig.savefig("/u/10/a/amp2217/public_html/plots/test_cmd.png", dpi=300)

        return

        pl.figure(figsize=(10,8))
        pl.hist(H.ravel(), bins=128)
        pl.savefig("/u/10/a/amp2217/public_html/plots/test_hist.png", dpi=300)


        pl.figure(figsize=(10,8))
        pl.imshow(H.T, extent=[ra_bins.min(), ra_bins.max(), dec_bins.min(), dec_bins.max()],
                  cmap='Greys', interpolation='nearest')
        pl.xlim(pl.xlim()[::-1])
        pl.savefig("/u/10/a/amp2217/public_html/plots/test.png", dpi=300)

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
    parser.add_argument("--bin-size", dest="bin_size", default="15 arcmin",
                        type=str, help="Value and unit for sky bin size.")
    parser.add_argument("-n", "--normalize", action="store_true", dest="normalize",
                        default=False, help="Normalize the histogram.")

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

    main(args.XCov_filename, bin_size=bin_size, normalize=args.normalize)
