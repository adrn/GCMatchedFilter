""" Turn a catalog of photometry from PS1 into an HDF5 file """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
import astropy.coordinates as coord
from astropy import log as logger
import astropy.units as u
import numpy as np
import h5py

# Global configuration stuff
# HACK: should be able to set this at command line?
cluster_c = coord.SkyCoord(ra=229.352*u.degree,
                           dec=-21.01*u.degree)
cluster_pad = {
    'inner': 0.08*u.degree,
    'outer': 0.2*u.degree
}

def data_to_X_cov(data):
    X = np.vstack([data['dered_{}'.format(band)] for band in 'griz']).T
    Xerr = np.vstack([data['{}Err'.format(band)] for band in 'griz']).T

    # mixing matrix W
    W = np.array([[1, 0, 0, 0],    # g magnitude
                  [1, -1, 0, 0],   # g-r color
                  [1, 0, -1, 0],   # g-i color
                  [1, 0, 0, -1]])  # g-z color
    X = np.dot(X, W.T)

    # compute error covariance with mixing matrix
    Xcov = np.zeros(Xerr.shape + Xerr.shape[-1:])
    Xcov[:, range(Xerr.shape[1]), range(Xerr.shape[1])] = Xerr ** 2

    # each covariance C = WCW^T
    Xcov = np.tensordot(np.dot(Xcov, W.T), W, (-2, -1))

    return X, Xcov

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
    ps1 = ps1[np.isfinite(ps1['dered_g']) & np.isfinite(ps1['dered_r']) &
              np.isfinite(ps1['dered_i']) & np.isfinite(ps1['dered_z'])]

    # output hdf5 file
    # TODO: right now, this saves multiple versions of the data because I SUCK
    with h5py.File(out_filename, mode='w') as f:

        # feature and covariance matrices for all stars
        allX, allCov = data_to_X_cov(ps1)
        g = f.create_group('all')
        g.create_dataset('X', allX.shape, dtype='f', data=allX)
        g.create_dataset('Cov', allCov.shape, dtype='f', data=allCov)
        g.create_dataset('ra', ps1['ra'].shape, dtype='f', data=ps1['ra'])
        g.create_dataset('dec', ps1['dec'].shape, dtype='f', data=ps1['dec'])

        # define coordinates object for all stars
        ps1_c = coord.ICRS(ra=ps1['ra']*u.degree, dec=ps1['dec']*u.degree)

        # feature and covariance matrices for cluster stars
        cluster_idx = ps1_c.separation(cluster_c) < cluster_pad['inner']
        cluster = ps1[cluster_idx]

        clusterX, clusterCov = data_to_X_cov(cluster)
        g = f.create_group('cluster')
        g.create_dataset('X', clusterX.shape, dtype='f', data=clusterX)
        g.create_dataset('Cov', clusterCov.shape, dtype='f', data=clusterCov)

        # feature and covariance matrices for NON-cluster stars
        g = f.create_group('noncluster')
        ncX = allX[~cluster_idx]
        ncCov = allCov[~cluster_idx]
        g.create_dataset('X', ncX.shape, dtype='f', data=ncX)
        g.create_dataset('Cov', ncCov.shape, dtype='f', data=ncCov)
        g.create_dataset('ra', ncX.shape[0:1], dtype='f', data=ps1[~cluster_idx]['ra'])
        g.create_dataset('dec', ncX.shape[0:1], dtype='f', data=ps1[~cluster_idx]['dec'])

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
