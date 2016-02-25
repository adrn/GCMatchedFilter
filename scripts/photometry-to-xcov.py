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

# Project
from globber.core import ps1_data_to_X_cov

# HACK: name should be customizable
from globber.ngc5897 import (cluster_c, cluster_pad, mixing_matrix,
                             color_lims, magnitude_lims)

def between(arr, lims):
    return (arr >= lims[0]) & (arr < lims[1])

def sky_cut(ra, dec, lims):
    idx = ((ra >= lims['ra'][0]) & (ra <= lims['ra'][1]) &
           (dec >= lims['dec'][0]) & (dec <= lims['dec'][1]))
    return idx

def main(ps1_filename, ra_lims=None, dec_lims=None, out_filename=None, overwrite=False):
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

    # define coordinates object for all stars
    ps1_c = coord.ICRS(ra=ps1['ra']*u.degree, dec=ps1['dec']*u.degree)

    # sky limits
    if ra_lims is None:
        ra_lims = [ps1['ra'].min(), ps1['ra'].max()]
    if dec_lims is None:
        dec_lims = [ps1['dec_lims'].min(), ps1['dec_lims'].max()]
    search_region = dict(ra=ra_lims, dec=dec_lims)

    # only finite (non-nan or inf) values (no constraint on y because dat shi cray)
    mask &= np.isfinite(ps1['dered_g'])
    mask &= np.isfinite(ps1['dered_r'])
    mask &= np.isfinite(ps1['dered_i'])
    mask &= np.isfinite(ps1['dered_z'])
    ps1 = ps1[mask]

    # cut out color region
    color_idx = np.ones(len(ps1)).astype(bool)
    for m_name,lim in magnitude_lims.items():
        color_idx &= between(ps1['dered_{}'.format(m_name)], lim)

    for (f1,f2),lim in color_lims.items():
        color_idx &= between(ps1['dered_{}'.format(f1)]-ps1['dered_{}'.format(f2)], lim)

    ps1 = ps1[color_idx]
    logger.debug("{} stars after color cuts".format(len(ps1)))

    # cut out search region
    search_idx = sky_cut(ps1['ra'], ps1['dec'], lims=search_region)
    search_ps1 = ps1[search_idx]
    logger.debug("{} stars in search field".format(len(search_ps1)))

    # feature and covariance matrices for cluster stars
    cluster_idx = ps1_c.separation(cluster_c) < cluster_pad['inner']
    cluster = ps1[cluster_idx]
    logger.debug("{} stars in cluster".format(len(cluster)))

    # control fields
    control_idx = ps1_c.separation(cluster_c) > cluster_pad['outer']
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
        search_X, search_Cov = ps1_data_to_X_cov(search_ps1, mixing_matrix)
        g.create_dataset('X', search_X.shape, dtype='f', data=search_X)
        g.create_dataset('Cov', search_Cov.shape, dtype='f', data=search_Cov)
        g.create_dataset('ra', search_ps1['ra'].shape, dtype='f', data=search_ps1['ra'])
        g.create_dataset('dec', search_ps1['dec'].shape, dtype='f', data=search_ps1['dec'])

        cluster_X, cluster_Cov = ps1_data_to_X_cov(cluster, mixing_matrix)
        g = f.create_group('cluster')
        g.create_dataset('X', cluster_X.shape, dtype='f', data=cluster_X)
        g.create_dataset('Cov', cluster_Cov.shape, dtype='f', data=cluster_Cov)

        # feature and covariance matrices for NON-cluster stars
        g = f.create_group('control')
        control_X, control_Cov = ps1_data_to_X_cov(control_ps1, mixing_matrix)
        g.create_dataset('X', control_X.shape, dtype='f', data=control_X)
        g.create_dataset('Cov', control_Cov.shape, dtype='f', data=control_Cov)
        g.create_dataset('ra', control_ps1['ra'].shape, dtype='f', data=control_ps1['ra'])
        g.create_dataset('dec', control_ps1['dec'].shape, dtype='f', data=control_ps1['dec'])

        # also create a group for the likelihood calculations
        g = f.create_group('log_likelihood')

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
