"""
Use Extreme Deconvolution to fit for density of star in color-magnitude
space in the control field.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import pickle
import sys

# Third-party
from astropy import log as logger
from astropy.io import ascii
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from astroML.density_estimation import XDGMM

# HACK: need to make cluster-independent
#       for now, hard code info for NGC 5897
DM = 15.62 # Brani's fit
cluster_c = coord.SkyCoord(ra=229.352*u.degree,
                           dec=-21.01*u.degree) # cluster sky position
color_lim = {
    'g-r': (0,0.7),
    'g-i': (0,0.9),
    'g-z': (0,0.85),
}

def cut_func(d, g_lim):
    cp = 'dered_{band}'

    g = d[cp.format(band='g')]
    gr = g-d[cp.format(band='r')]
    gi = g-d[cp.format(band='i')]
    gz = g-d[cp.format(band='z')]

    ix = (((g) > (g_lim[0])) & ((g) < (g_lim[1])) &
          (gr > (color_lim['g-r'][0])) & (gr < (color_lim['g-r'][1])) &
          (gi > (color_lim['g-i'][0])) & (gi < (color_lim['g-i'][1])) &
          (gz > (color_lim['g-z'][0])) & (gz < (color_lim['g-z'][1])))
    return d[ix]

def data_to_X_cov(data):
    X = np.vstack([data['dered_{}'.format(band)] for band in 'griz']).T
    Xerr = np.vstack([data['{}Err'.format(band)] for band in 'griz']).T

    # mixing matrix W
    W = np.array([[1, 0, 0, 0],    # g magnitude
                  [1, -1, 0, 0],   # g-r color
                  [1, 0, -1, 0],   # g-i color
                  [1, 0, 0, -1]])  # g-z color

    X = np.dot(X, W.T)

    # compute error covariance from mixing matrix
    Xcov = np.zeros(Xerr.shape + Xerr.shape[-1:])
    Xcov[:, range(Xerr.shape[1]), range(Xerr.shape[1])] = Xerr ** 2

    # each covariance C = WCW^T
    # best way to do this is with a tensor dot-product
    Xcov = np.tensordot(np.dot(Xcov, W.T), W, (-2, -1))

    return X, Xcov

def main(ps1_file, g_lim):
    try:
        ps1 = np.load(ps1_file)
    except:
        ps1 = ascii.read(ps1_file)

    cut_ps1 = cut_func(ps1, g_lim=g_lim)
    ps1_c = coord.SkyCoord(ra=cut_ps1['ra']*u.degree,
                           dec=cut_ps1['dec']*u.degree)

    cut_ps1 = cut_ps1[ps1_c.separation(cluster_c) > (0.12*u.degree)]

    # feature and covariance matrices
    X,Xcov = data_to_X_cov(cut_ps1)

    n_clusters = 8
    n_iter = 512

    xd_clf = XDGMM(n_clusters, n_iter=n_iter, tol=1E-4, verbose=True)
    xd_clf.fit(X[::100], Xcov[::100])

    # pickle this thing! xd_clf
    with open("xd_control_clf.pickle", "wb") as f:
        pickle.dump(xd_clf, f)

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")

    parser.add_argument("-f", "--ps1file", dest="ps1_file", required=True,
                        type=str, help="Path to the PS1 photometry file (npy or txt file).")
    parser.add_argument("-g", "--glim", dest="g_lim", default="17.5,20.5",
                        type=str, help="Range of g-band magnitudes to consider.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    g_lim = list(map(float, args.g_lim.split(",")))
    main(ps1_file=args.ps1_file, g_lim=g_lim)
