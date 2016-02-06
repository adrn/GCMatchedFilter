""" ...explain... """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.coordinates as coord
import astropy.units as u
from astropy.tests.helper import remote_data
import numpy as np

from ..core import get_sfd_ebv

# @remote_data
# HACK: right now, have to copy SFD files in to build dir...
# def test_SFD():
#     c = coord.SkyCoord(ra=np.random.uniform(0,360,size=128)*u.degree,
#                        dec=np.random.uniform(-15,15,size=128)*u.degree)
#     ebv = get_sfd_ebv(c)
#     print(ebv)

