# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Stellar streams around globular clusters.
"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    from . import dust

    # modify URL of data repository:
    #   in this case, the only
    from astropy.utils.data import conf
    conf.dataurl = "http://data.adrian.pw/dust/"
