""" Write out a binary file containing the PS1 data with dereddened photometry """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
import astropy.coordinates as coord
import astropy.table as table
import astropy.units as u
import numpy as np

# Project
from globmfer.dust import getval

# HACK: needed for Branimir's SFD code
os.environ['DUST_DIR'] = "/Users/adrian/projects/globmfer/data/dust"

def main(input_file):

    filepath,ext = os.path.splitext(input_file)
    output_file = "{}_dered.npy".format(filepath)

    # read the catalog file
    ps1 = table.Table(np.genfromtxt(input_file, dtype=None, names=True))

    # only take photometry that is finite in g,r,i
    good_idx = np.ones(len(ps1)).astype(bool)
    for band in 'gri':
        good_idx &= np.isfinite(ps1['{}'.format(band)])
    ps1 = ps1[good_idx]

    # make an astropy coordinates object for the stars
    ps1_c = coord.ICRS(ra=ps1['ra']*u.degree, dec=ps1['dec']*u.degree)
    ps1_g = ps1_c.transform_to(coord.Galactic)

    # get SFD extinction E(B-V) for all PS1 catalog positions
    SFD_EBV = getval(ps1_g.l.wrap_at(180*u.degree).degree, ps1_g.b.degree)

    # HACK: a correction based on Branimir's fit to the NGC 5897 data
    SFD_EBV = SFD_EBV - 0.0033 # Brani's correction

    # deredden the photometry and save as new columns
    for band, c in zip('grizy', [3.172, 2.271, 1.682, 1.322, 1.087]):
        ps1["dered_{}".format(band)] = ps1[band] - c*SFD_EBV

    ps1_arr = ps1.as_array()
    np.save(output_file, ps1_arr)

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")

    parser.add_argument("-f", "--file", dest="input_file", required=True,
                        type=str, help="The input catalog file from PS1.")

    args = parser.parse_args()

    main(args.input_file)
