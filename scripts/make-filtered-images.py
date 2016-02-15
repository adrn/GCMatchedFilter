from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
from astropy import log as logger
import astropy.units as u
import h5py
import numpy as np

def sky_cut(ra, dec, lims):
    idx = ((ra > lims['ra'][0]) & (ra < lims['ra'][1]) &
           (dec > lims['dec'][0]) & (dec < lims['dec'][1]))
    return idx

def color_cut(X, lims):
    idx = np.ones(X.shape[0]).astype(bool)
    for i,(lo,hi) in enumerate(lims):
        idx &= (X[:,i] > lo) & (X[:,i] < hi)
    return idx

def main(XCov_filename, bin_size, normalize=False):
    # HACK: start with a small region, and a color cut
    sky_lims = {'ra': (225,235), 'dec': (-24,-18)}
    color_lims = [(17.5,20.), (0,0.7), (0,0.8), (0,0.8)]

    with h5py.File(XCov_filename, mode='r') as f:
        ra = f['all']['ra']
        dec = f['all']['dec']
        X = f['all']['X']

        ix = sky_cut(ra, dec, lims=sky_lims)
        ix &= color_cut(X, lims=color_lims)

        ra = ra[ix]
        dec = dec[ix]
        ll = ll = f['all']['cluster_log_likelihoog']

        ra_bins = np.arange(ra.min(), ra.max(), step=bin_size.to(u.degree).value)
        dec_bins = np.arange(dec.min(), dec.max(), step=bin_size.to(u.degree).value)

        if normalize:
            weights = np.exp(ll - ll.max())
        else:
            weights = np.exp(ll)

        print("NaN or Inf weights: {}".format(np.isnan(weights).any() or np.isinf(weights).any()))
        return

        # good_idx = np.isfinite(weights)
        # good_targets = this_targets[good_idx]

        # H,ra_edges,dec_edges = np.histogram2d(good_targets['ra'], good_targets['dec'],
        #                                       bins=(ra_bins,dec_bins), weights=weights[good_idx],
        #                                       normed=True)

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
