""" Turn a catalog of photometry from PS1 into an HDF5 file """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
from astropy import log as logger
from astropy.io import ascii
import h5py
import numpy.lib.recfunctions as nprf

# Project
from globber.core import ps1_isoc_to_XCov

# HACK: name should be customizable
from globber.ngc5897 import mixing_matrix

def main(iso_filename, XCov_filename, interpolate=True, overwrite=False):

    # FOR PARSEC ISOCHRONE (reversing it for interpolation)
    iso = ascii.read(iso_filename, header_start=13)[:114][::-1]
    iso = nprf.stack_arrays((iso[:25], iso[27:]),usemask=False) # because of stupid red clump turnaround

    # FOR DARTMOUTH ISOCHRONE (reversing it for interpolation)
    # iso = ascii.read(iso_filename, header_start=8)[::-1]

    # output hdf5 file
    with h5py.File(XCov_filename, mode='r+') as f:

        # feature and covariance matrices for all stars
        X = ps1_isoc_to_XCov(iso, W=mixing_matrix, interpolate=interpolate)

        if 'isochrone' in f and overwrite:
            f.__delitem__('isochrone')
            logger.debug("Overwriting isochrone data")

        if 'isochrone' not in f:
            g = f.create_group('isochrone')
        else:
            g = f['isochrone']

        if 'X' not in f['isochrone']:
            g.create_dataset('X', X.shape, dtype='f', data=X)

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
         overwrite=args.overwrite)
