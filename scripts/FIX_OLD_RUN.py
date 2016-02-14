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

def main(XCov_filename, results_filename, ps1_filename):

    # Load PS1 photometry
    ps1 = np.load(ps1_filename)
    ps1 = ps1[np.isfinite(ps1['dered_g']) & np.isfinite(ps1['dered_r']) &
              np.isfinite(ps1['dered_i']) & np.isfinite(ps1['dered_z'])]

    with h5py.File(results_filename, mode='r') as f:
        ll = f['cluster_log_likelihood']

    with h5py.File(XCov_filename, mode='r+') as f:

        # feature and covariance matrices for all stars
        g = f['all']
        g.create_dataset('ra', ps1['ra'].shape, dtype='f', data=ps1['ra'])
        g.create_dataset('dec', ps1['dec'].shape, dtype='f', data=ps1['dec'])
        g.create_dataset('cluster_log_likelihood', ll.shape, dtype='f', data=ll)

        # define coordinates object for all stars
        ps1_c = coord.ICRS(ra=ps1['ra']*u.degree, dec=ps1['dec']*u.degree)

        # feature and covariance matrices for cluster stars
        cluster_idx = ps1_c.separation(cluster_c) < cluster_pad['inner']

        # feature and covariance matrices for NON-cluster stars
        g = f.create_group('noncluster')
        ncX, ncCov = data_to_X_cov(ps1[~cluster_idx])
        g.create_dataset('X', ncX.shape, dtype='f', data=ncX)
        g.create_dataset('Cov', ncCov.shape, dtype='f', data=ncCov)
        g.create_dataset('ra', ncX.shape[0:1], dtype='f', data=ps1[~cluster_idx]['ra'])
        g.create_dataset('dec', ncX.shape[0:1], dtype='f', data=ps1[~cluster_idx]['dec'])

if __name__ == "__main__":
    import sys
    XCov_filename, results_filename, ps1_filename = sys.argv[1:]
    main(XCov_filename, results_filename, ps1_filename)
