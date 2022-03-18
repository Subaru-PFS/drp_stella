import lsst.utils

import astropy.coordinates
import astropy.io.fits
import astropy.units
import astropy.wcs
import numpy as np
import scipy.interpolate

import os

__all__ = ("DustMap",)


class DustMap:
    """Dust extinction map E(B-V)

    Parameters
    ----------
    filenames : `list` of `str`, optional
        Paths to FITS files containing E(B-V).
        Must be of length 2 if not None,
        one file for the northern hemisphere
        and the other for the southern hemisphere.
        Uses ``dustmaps_cachedata`` package by default.
    kx : `int`, optional
        Degrees of the bivariate spline. Default is 1.
    ky : `int`, optional
        Degrees of the bivariate spline. Default is 1.
    """
    def __init__(self, filenames=None, kx=1, ky=1):
        if not filenames:
            dataDir = lsst.utils.getPackageDir("dustmaps_cachedata")
            filenames = [
                os.path.join(dataDir, "data", "sfd", "SFD_dust_4096_ngp.fits"),
                os.path.join(dataDir, "data", "sfd", "SFD_dust_4096_sgp.fits"),
            ]
        if len(filenames) != 2:
            raise ValueError(
                f"Two and only two dustmap files must be given, but {len(filenames)} file(s) are given."
            )
        with astropy.io.fits.open(filenames[0]) as hdus:
            hdu = hdus[0]
            self.ngpWcs = astropy.wcs.WCS(hdu.header)
            self.ngpImage = scipy.interpolate.RectBivariateSpline(
                np.linspace(1, hdu.data.shape[0], hdu.data.shape[0], endpoint=True, dtype=float),
                np.linspace(1, hdu.data.shape[1], hdu.data.shape[1], endpoint=True, dtype=float),
                hdu.data.astype(float),
                kx=kx,
                ky=ky,
                s=0,
            )
        with astropy.io.fits.open(filenames[1]) as hdus:
            hdu = hdus[0]
            self.sgpWcs = astropy.wcs.WCS(hdu.header)
            self.sgpImage = scipy.interpolate.RectBivariateSpline(
                np.linspace(1, hdu.data.shape[0], hdu.data.shape[0], endpoint=True, dtype=float),
                np.linspace(1, hdu.data.shape[1], hdu.data.shape[1], endpoint=True, dtype=float),
                hdu.data.astype(float),
                kx=kx,
                ky=ky,
                s=0,
            )

    def __call__(self, ra, dec):
        """Get E(B-V) at (ra, dec) in ICRS coordinates

        Parameters
        ----------
        ra : `float` or `numpy.array` or `astropy.units.quantity.Quantity`
            R.A. in degrees by default.
        dec : `float` or `numpy.array` or `astropy.units.quantity.Quantity`
            Dec. in degrees by default.

        Returns
        -------
        ebv : `numpy.array`
            E(B-V)
        """
        # convert (ra, dec) to (l, b) (galactic coordinates)
        if not isinstance(ra, astropy.units.quantity.Quantity):
            ra = np.asarray(ra, dtype=float) * astropy.units.degree
        if not isinstance(dec, astropy.units.quantity.Quantity):
            dec = np.asarray(dec, dtype=float) * astropy.units.degree
        gal = astropy.coordinates.SkyCoord(ra, dec, frame="icrs").galactic
        gall = np.asarray(gal.l.degree)
        galb = np.asarray(gal.b.degree)

        # allocate an array in which to store E(B-V)
        ebv = np.full(shape=gall.shape, fill_value=np.nan, dtype=float)

        # northern hemisphere
        index = (galb >= 0)
        if np.any(index):
            x, y = self.ngpWcs.wcs_world2pix(gall[index], galb[index], 1.0)
            ebv[index] = self.ngpImage(y, x, grid=False)
        # southern hemisphere
        index = (galb < 0)
        if np.any(index):
            x, y = self.sgpWcs.wcs_world2pix(gall[index], galb[index], 1.0)
            ebv[index] = self.sgpImage(y, x, grid=False)

        return ebv
