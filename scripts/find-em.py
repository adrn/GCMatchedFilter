""" Find stuff around NGC 5897 """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
import astropy.coordinates as coord
from astropy import log as logger
import astropy.units as u
from astroML.utils import log_multivariate_gaussian
import numpy as np
from scipy.misc import logsumexp
from gary.util import get_pool

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

def worker(task):
    filenames, metadata, n, offset = task

    clusterX = np.memmap(filenames['cluster']['X'], mode='r',
                         dtype='float64', shape=metadata['cluster']['X']['shape'])
    clusterCov = np.memmap(filenames['cluster']['Cov'], mode='r',
                           dtype='float64', shape=metadata['cluster']['Cov']['shape'])

    allX = np.memmap(filenames['all']['X'], mode='r',
                     dtype='float64', shape=metadata['all']['X']['shape'])
    allCov = np.memmap(filenames['all']['Cov'], mode='r',
                       dtype='float64', shape=metadata['all']['Cov']['shape'])
    allX = allX[offset:offset+n]
    allCov = allCov[offset:offset+n]

    V = allCov[:,np.newaxis,:,:] + clusterCov
    ll = log_multivariate_gaussian(allX[:,np.newaxis,:], clusterX, V)
    ll = logsumexp(ll, axis=-1) # NOTE: could also max here

    result = dict()
    result['ll'] = ll
    result['n'] = n
    result['offset'] = offset
    result['output_filename'] = filenames['output']
    result['output_metadata'] = metadata['output']

    return result

def callback(result):
    fp = np.memmap(result['output_filename'], dtype='float64',
                   shape=result['output_metadata']['shape'], mode='r+')
    fp[result['offset']:result['offset']+result['n']] = result['ll']
    fp.flush()

def main(ps1_catalog_file, index, n_per_chunk, mpi=False):
    lims = {
        'g': (17.,20.2),
        'g-r': (0,0.7),
        'g-i': (0,0.9),
        'g-z': (0,0.85),
    }
    cluster_pad = {
        'inner': 0.08*u.degree,
        'outer': 0.2*u.degree
    }

    # MPI or serial pool
    pool = get_pool(mpi=mpi)

    ps1_catalog_file = os.path.abspath(ps1_catalog_file)
    basepath = os.path.split(ps1_catalog_file)[0]
    output_filename = os.path.join(basepath, "results.mmap")

    # Load PS1 photometry
    ps1 = np.load(ps1_catalog_file)
    ps1 = ps1[np.isfinite(ps1['dered_g']) & np.isfinite(ps1['dered_r']) &
              np.isfinite(ps1['dered_i']) & np.isfinite(ps1['dered_z'])]

    # Save feature and covariance matrices
    allX, allCov = data_to_X_cov(ps1)
    all_X_filename = os.path.join(basepath, "allX.mmap")
    all_Cov_filename = os.path.join(basepath, "allCov.mmap")

    if not os.path.exists(all_X_filename) or not os.path.exists(all_Cov_filename):
        X_fp = np.memmap(all_X_filename, dtype=allX.dtype, mode='w+', shape=allX.shape)
        X_fp[:,:] = allX

        Cov_fp = np.memmap(all_Cov_filename, dtype=allCov.dtype, mode='w+', shape=allCov.shape)
        Cov_fp[:,:] = allCov

        logger.debug("Saved {}".format(all_X_filename))

    # define coordinates object for all stars
    ps1_c = coord.ICRS(ra=ps1['ra']*u.degree, dec=ps1['dec']*u.degree)

    # Save feature and covariance matrices for cluster
    cluster_idx = ps1_c.separation(cluster_c) < cluster_pad['inner']
    cluster = ps1[cluster_idx]

    clusterX, clusterCov = data_to_X_cov(cluster)
    cluster_X_filename = os.path.join(basepath, "clusterX.mmap")
    cluster_Cov_filename = os.path.join(basepath, "clusterCov.mmap")

    if not os.path.exists(cluster_X_filename) or not os.path.exists(cluster_Cov_filename):
        X_fp = np.memmap(cluster_X_filename, dtype=clusterX.dtype, mode='w+', shape=clusterX.shape)
        X_fp[:,:] = clusterX

        Cov_fp = np.memmap(cluster_Cov_filename, dtype=clusterCov.dtype, mode='w+', shape=clusterCov.shape)
        Cov_fp[:,:] = clusterCov

        logger.debug("Saved {}".format(cluster_X_filename))

    logger.info("{} total stars, {} cluster stars".format(len(ps1), len(clusterX)))

    # for MPI
    tasks = list()

    filenames = dict(all=dict(), cluster=dict())
    filenames['all']['X'] = all_X_filename
    filenames['all']['Cov'] = all_Cov_filename
    filenames['cluster']['X'] = cluster_X_filename
    filenames['cluster']['Cov'] = cluster_Cov_filename
    filenames['output'] = output_filename

    metadata = dict(all=dict(X=dict(),Cov=dict()), cluster=dict(X=dict(),Cov=dict()), output=dict())
    metadata['all']['X']['shape'] = allX.shape
    metadata['all']['Cov']['shape'] = allCov.shape
    metadata['cluster']['X']['shape'] = clusterX.shape
    metadata['cluster']['Cov']['shape'] = clusterCov.shape
    metadata['output']['shape'] = (allX.shape[0],)

    for i in range(len(ps1) // n_per_chunk):
        tasks.append([filenames, metadata, n_per_chunk, i*n_per_chunk])
    if (len(tasks) * n_per_chunk) < len(ps1):
        tasks += [[filenames, metadata, n_per_chunk, (i+1)*n_per_chunk]]

    result = worker(tasks[index])
    callback(result)

    # logger.debug("{} tasks".format(len(tasks)))
    # # pool.map(worker, tasks, callback=callback) # HACK: slice
    # # pool.close()

    # fp = np.memmap(filenames['output'], dtype='float64',
    #                shape=metadata['output']['shape'], mode='r')

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")

    parser.add_argument("-f", dest="filename", default=None, required=True,
                        type=str, help="Path to PS1 catalog file (a .npy file)")
    parser.add_argument("-t", "--test", dest="test", action="store_true", default=False,
                        help="Run in test/development mode.")
    parser.add_argument("-n", "--nperchunk", dest="n_per_chunk", default=1000,
                        type=int, help="Number of stars per chunk.")
    parser.add_argument("-i", "--index", dest="index", required=True,
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

    main(ps1_catalog_file=args.filename, index=args.index, n_per_chunk=args.n_per_chunk, mpi=args.mpi)
