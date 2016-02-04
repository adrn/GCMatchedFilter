""" Note: this is from Branimir Sesar """

import os, numpy
from astropy.io import fits as pyfits
from scipy.ndimage import map_coordinates
import pdb
from astropy import wcs

def getval(l, b, map='sfd', size=None, order=1):
    """Return SFD at the Galactic coordinates l, b.

    Example usage:
    h, w = 1000, 4000
    b, l = numpy.mgrid[0:h,0:w]
    l = 180.-(l+0.5) / float(w) * 360.
    b = 90. - (b+0.5) / float(h) * 180.
    ebv = dust.getval(l, b)
    imshow(ebv, aspect='auto', norm=matplotlib.colors.LogNorm())
    """
    l = numpy.atleast_1d(l)
    b = numpy.atleast_1d(b)
    if map == 'sfd':
        map = 'dust'
    if map in ['dust', 'd100', 'i100', 'i60', 'mask', 'temp', 'xmap']:
        fname = 'SFD_'+map
    else:
        fname = map
    maxsize = { 'd100':1024, 'dust':4096, 'i100':4096, 'i60':4096,
                'mask':4096 }
    if size is None and map in maxsize:
        size = maxsize[map]
    if size is not None:
        fname = fname + '_%d' % size
    fname = os.path.join(os.environ['DUST_DIR'], fname)
    if not os.access(fname+'_ngp.fits', os.F_OK):
        raise Exception('Map file %s not found' % (fname+'_ngp.fits'))
    if l.shape != b.shape:
        raise ValueError('l.shape must equal b.shape')
    out = numpy.zeros_like(l, dtype='f4')
    for pole in ['ngp', 'sgp']:
        m = (b >= 0) if pole == 'ngp' else b < 0
        if numpy.any(m):
            hdulist = pyfits.open(fname+'_%s.fits' % pole)
            w = wcs.WCS(hdulist[0].header)
            x, y = w.wcs_world2pix(l[m], b[m], 0)
            out[m] = map_coordinates(hdulist[0].data, [y, x], order=order, mode='nearest')
    return out
