from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
import astropy.units as u
import numpy as np

# Project
# ...

def main():
    pass

if __name__ == "__main__":
    main()


def ps1_data_to_X_cov(data, W):
    """
    Given Pan-STARRS1 photometry -- which must contain keys like
    'dered_g' and 'gErr', etc. -- and a mixing matrix, `W`, create
    a feature / data matrix `X` and a covariance matrix `V`.
    """
    X = np.vstack([data['dered_{}'.format(band)] for band in 'grizy']).T
    Xerr = np.vstack([data['{}Err'.format(band)] for band in 'grizy']).T

    X = np.einsum('nj,mj->nm', X, W)

    # compute error covariance with mixing matrix
    V = np.zeros(Xerr.shape + Xerr.shape[-1:])
    for i in range(Xerr.shape[1]):
        V[:,i,i] = Xerr[:,i]**2

    # each covariance C = WCW^T
    V = np.einsum('mj,njk->nmk', W, V)
    V = np.einsum('lk,nmk->nml', W, V)

    return X, V
