""" Turn a catalog of photometry from PS1 into an HDF5 file """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
import astropy.coordinates as coord
from astropy import log as logger
import astropy.units as u
import numpy as np
import h5py

# -------------------------------------------------------------------
# HACK: should be able to set this at command line?
cluster_c = coord.SkyCoord(ra=229.352*u.degree,
                           dec=-21.01*u.degree)
cluster_pad = {
    'inner': 7*u.arcmin,
    'outer': 12*u.arcmin
}
search_region = dict(
    ra=(224.,234),
    dec=(-26,-16)
)
control_regions = [
    dict(ra=(223,235), dec=(-16,-15)),
    dict(ra=(223,235), dec=(-27,-26)),
    dict(ra=(223,224), dec=(-26,-16)),
    dict(ra=(234,235), dec=(-26,-16))
]
# -------------------------------------------------------------------

def data_to_X_cov(data):
    X = np.vstack([data['dered_{}'.format(band)] for band in 'griz']).T
    Xerr = np.vstack([data['{}Err'.format(band)] for band in 'griz']).T

    # HACK: this is fixed...colors should be customizable too, eh?
    # mixing matrix W
    W = np.array([[1, 0, 0, 0],    # g magnitude
                  [0, 0, 1, 0],    # i magnitude
                  [1, -1, 0, 0],   # g-r color
                  [1, 0, -1, 0],   # g-i
                  [1, 0, 0, -1]])  # g-z
    X = np.dot(X, W.T)

    # compute error covariance with mixing matrix
    Xcov = np.zeros(Xerr.shape + Xerr.shape[-1:])
    Xcov[:, range(Xerr.shape[1]), range(Xerr.shape[1])] = Xerr ** 2

    # each covariance C = WCW^T
    Xcov = np.tensordot(np.dot(Xcov, W.T), W, (-2, -1))

    return X[:,1:], Xcov[:,1:,1:] # ignore g magnitude

def between(arr, lims):
    return (arr >= lims[0]) & (arr < lims[1])

def sky_cut(ra, dec, lims):
    idx = ((ra > lims['ra'][0]) & (ra < lims['ra'][1]) &
           (dec > lims['dec'][0]) & (dec < lims['dec'][1]))
    return idx

def main(ps1_filename, out_filename=None, overwrite=False):
    ps1_filename = os.path.abspath(ps1_filename)
    base_filename = os.path.splitext(ps1_filename)[0]

    # save files
    if out_filename is None:
        out_filename = "{}_XCov.h5".format(base_filename)

    if os.path.exists(out_filename) and overwrite:
        logger.debug("Clobbering existing file: {}".format(out_filename))
        os.remove(out_filename)

    if os.path.exists(out_filename):
        logger.debug("File already exists: {}".format(out_filename))
        return out_filename

    # Load PS1 photometry
    ps1 = np.load(ps1_filename)
    mask = np.ones(len(ps1)).astype(bool)

    # only finite (non-nan or inf) values
    mask &= np.isfinite(ps1['dered_g'])
    mask &= np.isfinite(ps1['dered_r'])
    mask &= np.isfinite(ps1['dered_i'])
    mask &= np.isfinite(ps1['dered_z'])
    ps1 = ps1[mask]

    # cut out color region
    color_idx = np.ones(len(ps1)).astype(bool)
    color_idx &= between(ps1['dered_i'], (15,21))
    color_idx &= between(ps1['dered_g']-ps1['dered_r'], (0,0.7))
    color_idx &= between(ps1['dered_g']-ps1['dered_i'], (0,1))
    color_idx &= between(ps1['dered_g']-ps1['dered_z'], (0,1))
    ps1 = ps1[color_idx]
    logger.debug("{} stars after color cuts".format(len(ps1)))

    # cut out search region
    search_idx = sky_cut(ps1['ra'], ps1['dec'], lims=search_region)
    search_ps1 = ps1[search_idx]
    logger.debug("{} stars in search field".format(len(search_ps1)))

    # control fields
    control_idx = np.zeros_like(search_idx).astype(bool)
    for region in control_regions:
        control_idx |= sky_cut(ps1['ra'], ps1['dec'], lims=region)
    control_ps1 = ps1[control_idx]
    logger.debug("{} stars in control fields".format(len(control_ps1)))

    # randomize ps1 array so i can do simple slicing later
    np.random.shuffle(search_ps1)
    np.random.shuffle(control_ps1)

    # output hdf5 file
    # TODO: right now, this saves multiple versions of the data because I SUCK
    with h5py.File(out_filename, mode='w') as f:

        # feature and covariance matrices for all stars
        g = f.create_group('search')
        search_X, search_Cov = data_to_X_cov(search_ps1)
        g.create_dataset('X', search_X.shape, dtype='f', data=search_X)
        g.create_dataset('Cov', search_Cov.shape, dtype='f', data=search_Cov)
        g.create_dataset('ra', search_ps1['ra'].shape, dtype='f', data=search_ps1['ra'])
        g.create_dataset('dec', search_ps1['dec'].shape, dtype='f', data=search_ps1['dec'])

        # define coordinates object for all stars
        ps1_c = coord.ICRS(ra=ps1['ra']*u.degree, dec=ps1['dec']*u.degree)

        # feature and covariance matrices for cluster stars
        cluster_idx = ps1_c.separation(cluster_c) < cluster_pad['inner']
        cluster = ps1[cluster_idx]
        logger.debug("{} stars in cluster".format(len(cluster)))

        cluster_X, cluster_Cov = data_to_X_cov(cluster)
        g = f.create_group('cluster')
        g.create_dataset('X', cluster_X.shape, dtype='f', data=cluster_X)
        g.create_dataset('Cov', cluster_Cov.shape, dtype='f', data=cluster_Cov)

        # feature and covariance matrices for NON-cluster stars
        g = f.create_group('control')
        control_X, control_Cov = data_to_X_cov(control_ps1)
        g.create_dataset('X', control_X.shape, dtype='f', data=control_X)
        g.create_dataset('Cov', control_Cov.shape, dtype='f', data=control_Cov)
        g.create_dataset('ra', control_ps1['ra'].shape, dtype='f', data=control_ps1['ra'])
        g.create_dataset('dec', control_ps1['dec'].shape, dtype='f', data=control_ps1['dec'])

        f.flush()
        logger.debug("Saved {}".format(out_filename))

    return out_filename

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite",
                        default=False, help="DESTROY.")

    parser.add_argument("--input", dest="ps1_filename", required=True,
                        type=str, help="Path to PS1 catalog file (a .npy file)")
    parser.add_argument("--output", dest="out_filename", default=None,
                        type=str, help="Full path to output file HDF5 file. Default "
                                       " is to put in same path as input as XCov.h5")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    XCov_filename = main(ps1_filename=args.ps1_filename,
                         out_filename=args.out_filename,
                         overwrite=args.overwrite)
