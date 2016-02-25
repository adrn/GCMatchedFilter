from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
from scipy.interpolate import splprep, splev

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

def ps1_isoc_to_XCov(data, W, interpolate=False, n_interpolate=1024):
    """
    Given an isochrone in Pan-STARRS1 filters -- which must contain keys like
    'gP1', etc. -- and a mixing matrix, `W`, create a feature / data matrix `X`.
    """
    X = np.vstack([data['{}P1'.format(band)] for band in 'grizy']).T
    X = np.einsum('nj,mj->nm', X, W)

    if interpolate:
        tck, u = splprep(X[:,1:].T, u=X[:,0], k=3, s=1E-4)
        u_fine = np.linspace(u.min(), u.max(), n_interpolate)
        X = np.vstack([u_fine] + splev(u_fine, tck)).T

    return X
