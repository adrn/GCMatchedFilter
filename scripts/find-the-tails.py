""" Find stuff around NGC 5897 """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
import astropy.coordinates as coord
from astropy import log as logger
import astropy.units as u
from astroML.utils import log_multivariate_gaussian
import numpy as np
from scipy.misc import logsumexp
import filelock
import h5py

# Global configuration stuff
cluster_c = coord.SkyCoord(ra=229.352*u.degree,
                           dec=-21.01*u.degree)

def color_cut(d, lims):
    g = d['dered_g']
    gr = g-d['dered_r']
    gi = g-d['dered_i']
    gz = g-d['dered_z']

    ix = ((g > lims['g'][0]) & (g < lims['g'][1]) &
          (gr > lims['g-r'][0]) & (gr < lims['g-r'][1]) &
          (gi > lims['g-i'][0]) & (gi < lims['g-i'][1]) &
          (gz > lims['g-z'][0]) & (gz < lims['g-z'][1]))
    return d[ix]

def sky_cut(data, lims):
    idx = ((data['ra'] > lims['ra'][0]) & (data['ra'] < lims['ra'][1]) &
           (data['dec'] > lims['dec'][0]) & (data['dec'] < lims['dec'][1]))
    return data[idx]

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

def worker(allX, allCov, clusterX, clusterCov):
    V = allCov[:,np.newaxis,:,:] + clusterCov
    ll = log_multivariate_gaussian(allX[:,np.newaxis,:], clusterX, V)
    ll = logsumexp(ll, axis=-1) # NOTE: could also max here
    return ll

def initialize(ps1_catalog_file, overwrite=False):
    cluster_pad = {
        'inner': 0.08*u.degree,
        'outer': 0.2*u.degree
    }

    ps1_catalog_file = os.path.abspath(ps1_catalog_file)
    basepath = os.path.split(ps1_catalog_file)[0]

    # save files
    XCov_filename = os.path.join(basepath, "XCov.h5")
    results_filename = os.path.join(basepath, "results.h5")

    # Load PS1 photometry
    ps1 = np.load(ps1_catalog_file)
    ps1 = ps1[np.isfinite(ps1['dered_g']) & np.isfinite(ps1['dered_r']) &
              np.isfinite(ps1['dered_i']) & np.isfinite(ps1['dered_z'])]

    if os.path.exists(XCov_filename) and overwrite:
        logger.debug("Clobbering existing file: {}".format(XCov_filename))
        os.remove(XCov_filename)

    if os.path.exists(results_filename) and overwrite:
        os.remove(results_filename)

    if os.path.exists(XCov_filename) and os.path.exists(results_filename):
        logger.debug("Files already exist: {}, {}".format(XCov_filename, results_filename))
        return XCov_filename, results_filename

    # output hdf5 file
    f = h5py.File(XCov_filename, mode='w')

    # feature and covariance matrices for all stars
    allX, allCov = data_to_X_cov(ps1)
    g = f.create_group('all')
    dsetX = g.create_dataset('X', allX.shape, dtype='f')
    dsetCov = g.create_dataset('Cov', allCov.shape, dtype='f')
    dsetX[...] = allX
    dsetCov[...] = allCov

    # define coordinates object for all stars
    ps1_c = coord.ICRS(ra=ps1['ra']*u.degree, dec=ps1['dec']*u.degree)

    # Save feature and covariance matrices for cluster
    cluster_idx = ps1_c.separation(cluster_c) < cluster_pad['inner']
    cluster = ps1[cluster_idx]

    clusterX, clusterCov = data_to_X_cov(cluster)
    g = f.create_group('cluster')
    dsetX = g.create_dataset('X', clusterX.shape, dtype='f')
    dsetCov = g.create_dataset('Cov', clusterCov.shape, dtype='f')
    dsetX[...] = clusterX
    dsetCov[...] = clusterCov

    logger.debug("Saved {}".format(XCov_filename))

    f.close()

    with h5py.File(results_filename, mode='w') as f:
        dset = f.create_dataset('cluster_log_likelihood', (allX.shape[0],), dtype='f')
        dset[:] = np.ones(allX.shape[0]) + np.nan

    return XCov_filename, results_filename

def main(XCov_filename, results_filename, chunk_index, n_per_chunk):
    if chunk_index is None:
        raise ValueError("-i, --chunk-index is required")

    slc = slice(chunk_index*n_per_chunk, (chunk_index+1)*n_per_chunk)
    with h5py.File(results_filename, mode='r') as results_f:
        ll = results_f['cluster_log_likelihood'][slc]
        if np.isfinite(ll).all():
            logger.debug("All log-likelihoods already computed for Chunk {} ({}:{})"
                         .format(chunk_index,slc.start,slc.stop))
            return

    logger.debug("Computing likelihood for Chunk {} ({}:{})".format(chunk_index,slc.start,slc.stop))
    with h5py.File(XCov_filename, mode='r') as f:
        logger.debug("{} total stars, {} cluster stars".format(f['all']['X'].shape[0],
                                                               f['cluster']['X'].shape[0]))
        X = f['all']['X'][slc]
        Cov = f['all']['Cov'][slc]
        ll = worker(X, Cov, f['cluster']['X'], f['cluster']['Cov'])
        logger.debug("Log-likelihoods computed")

        lock = filelock.FileLock("results.lock")
        try:
            with lock.acquire(timeout=90):
                logger.debug("File lock acquired - writing to results")
                with h5py.File(results_filename, mode='r+') as results_f:
                    results_f['cluster_log_likelihood'][slc] = ll
                    results_f.flush()

        except filelock.Timeout:
            logger.error("Timed out trying to acquire file lock.")
            sys.exit(1)

def status(results_filename):
    with h5py.File(results_filename, mode='r') as results_f:
        ll = results_f['cluster_log_likelihood'][:]
        ndone = np.isfinite(ll).sum()
        nnot = np.isnan(ll).sum()
        logger.info("{} done, {} not done".format(ndone, nnot))

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

    parser.add_argument("--initialize", dest="initialize", action="store_true", default=False,
                        help="Create HDF5 file with X and Cov.")
    parser.add_argument("--status", dest="status", action="store_true", default=False,
                        help="Check status of results file.")

    parser.add_argument("-f", dest="filename", default=None, required=True,
                        type=str, help="Path to PS1 catalog file (a .npy file)")
    parser.add_argument("-n", "--nperchunk", dest="n_per_chunk", default=1000,
                        type=int, help="Number of stars per chunk.")
    parser.add_argument("-i", "--chunk-index", dest="index", default=None,
                        type=int, help="Index of the chunk to process.")
    parser.add_argument("--mpi", dest="mpi", action="store_true", default=False,
                        help="Run with MPI.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    XCov_filename, results_filename = initialize(args.filename, overwrite=args.overwrite)

    if args.status:
        status(results_filename)
        sys.exit(0)

    if not args.initialize:
        main(XCov_filename, results_filename,
             chunk_index=args.index, n_per_chunk=args.n_per_chunk)
