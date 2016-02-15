""" Turn a catalog of photometry from PS1 into an HDF5 file """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
import astropy.coordinates as coord
from astropy import log as logger
import astropy.units as u
from astropy.io import ascii
import numpy as np
import h5py

def iso_to_XCov(data, smooth=0.1):
    X = np.vstack([data['{}P1'.format(band)] for band in 'griz']).T

    # mixing matrix W
    W = np.array([[1, 0, 0, 0],    # g magnitude
                  [1, -1, 0, 0],   # g-r color
                  [1, 0, -1, 0],   # g-i color
                  [1, 0, 0, -1]])  # g-z color
    X = np.dot(X, W.T)

    H = np.diag([smooth]*X.shape[1])
    H = H[np.newaxis]

    # compute error covariance with mixing matrix
    Cov = np.zeros(H.shape + H.shape[-1:])
    Cov[:, range(H.shape[1]), range(H.shape[1])] = H ** 2

    # each covariance C = WCW^T
    Cov = np.tensordot(np.dot(Cov, W.T), W, (-2, -1))

    return X, Cov

def main(iso_filename, XCov_filename, smooth, overwrite=False):

    iso = ascii.read(iso_filename, header_start=13)

    # output hdf5 file
    with h5py.File(XCov_filename, mode='r+') as f:

        # feature and covariance matrices for all stars
        X,Cov = iso_to_XCov(iso, smooth=smooth)
        g = f.create_group('isochrone')
        g.create_dataset('X', X.shape, dtype='f', data=X)
        g.create_dataset('Cov', Cov.shape, dtype='f', data=Cov)

        f.flush()
        logger.debug("Saved isochrone to {}".format(XCov_filename))

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

    parser.add_argument("-i", "--iso-file", dest="iso_filename", required=True,
                        type=str, help="Path to isochrone file.")
    parser.add_argument("-x", "--xcov-file", dest="XCov_filename", required=True,
                        type=str, help="Path to XCov file.")
    parser.add_argument("-s", "--smooth", dest="smooth", default=0.1,
                        type=float, help="Bandwidth of KDE.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(iso_filename=args.iso_filename,
         XCov_filename=args.XCov_filename,
         smooth=args.smooth)
