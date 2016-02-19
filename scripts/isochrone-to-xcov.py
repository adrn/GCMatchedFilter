""" Turn a catalog of photometry from PS1 into an HDF5 file """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
from astropy import log as logger
from astropy.io import ascii
import numpy as np
import h5py
from scipy.interpolate import splprep, splev

def iso_to_XCov(data, smooth=0.1, interpolate=False):
    X = np.vstack([data['{}P1'.format(band)] for band in 'griz']).T

    # mixing matrix W
    W = np.array([[1, 0, 0, 0],    # g magnitude
                  [1, -1, 0, 0],   # g-r color
                  [1, 0, -1, 0],   # g-i
                  [1, 0, 0, -1]])  # g-z
    X = np.dot(X, W.T)

    if interpolate:
        # interpolate
        t = np.linspace(0, 1, X.shape[0])
        tck, u = splprep(X.T, s=0, u=t)

        tnew = np.linspace(0, 1, 4096)
        Xinterp = np.array(splev(tnew, tck)).T
        X = Xinterp

    # compute error covariance with mixing matrix
    Cov = np.zeros(X.shape + (X.shape[-1],))
    for i in range(X.shape[1]):
        Cov[:,i,i] = smooth**2

    # each covariance C = WCW^T
    Cov = np.tensordot(np.dot(Cov, W.T), W, (-2, -1))

    # HACK:
    DM = 15.62
    X[:,0] += DM

    # HACK: slicing to ignore z
    return X[:,:3], Cov[:,:3,:3]

def main(iso_filename, XCov_filename, smooth, interpolate=False, overwrite=False):

    # FOR PARSEC ISOCHRONE
    # iso = ascii.read(iso_filename, header_start=13)
    # iso[114:] = iso[114:][::-1]

    # FOR DARTMOTH ISOCHRONE
    iso = ascii.read(iso_filename, header_start=8)

    # output hdf5 file
    with h5py.File(XCov_filename, mode='r+') as f:

        # feature and covariance matrices for all stars
        X,Cov = iso_to_XCov(iso, smooth=smooth, interpolate=interpolate)

        if 'isochrone' in f and overwrite:
            f.__delitem__('isochrone')
            logger.debug("Overwriting isochrone data")

        if 'isochrone' not in f:
            g = f.create_group('isochrone')
        else:
            g = f['isochrone']

        if 'X' not in f['isochrone'] or 'Cov' not in f['isochrone']:
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
         smooth=args.smooth, overwrite=args.overwrite)
