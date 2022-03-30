import numpy as np
from scipy import interpolate

import abc

__all__ = ("ExtinctionCurve", "F99ExtinctionCurve")


class ExtinctionCurve(metaclass=abc.ABCMeta):
    """Dust extinction curve.

    This is an abstract base class.
    ``_extinction()`` method must be overridden in subclasses.
    """

    def attenuation(self, wavelength, ebv):
        """A(lambda) converted to fractional extinction.

        To get a spectrum attenuated, multiply it by this.
        To correct an attenuated spectrum, divide it by this.

        Parameters
        ----------
        wavelength : `numpy.array`
            Wavelength in nm.
        ebv : `float`
            E(B-V).

        Returns
        -------
        att : `numpy.array`
            Fractional extinction
        """
        att = 10**(-0.4 * self._extinction(wavelength, ebv))
        return att

    @abc.abstractmethod
    def _extinction(self, wavelength, ebv):
        """Get the extinction as a function of wavelength, A(lambda)
        using a specific extinction curve model.

        Subclasses must override this method.

        Parameters
        ----------
        wavelength : `numpy.array`
            Wavelength in nm.
        ebv : `float`
            E(B-V).

        Returns
        -------
        ax : `numpy.array'
            Extinction as a function of wavelength, A(lambda) (magnitude).
        """
        raise NotImplementedError()


class F99ExtinctionCurve(ExtinctionCurve):
    """Fitzpatrick (1999) extinction curve.
    (Bibcode: 1999PASP..111...63F)

    Parameters
    ----------
    Rv : `float`
        Ratio of total to selective extinction at V, Rv = A(V)/E(B-V).
        Default is 3.1.
    """

    xAnchorF99 = np.array([0., 0.377, 0.820, 1.667, 1.828, 2.141, 2.433, 3.704, 3.846], dtype=float)
    """spline anchors taken from F99, micron^-1, from IR to UV
    """

    def __init__(self, Rv=3.1):
        super().__init__()
        self._Rv = Rv

        yAnchorUV = self._fm90(self.xAnchorF99[7:])
        yAnchorIR = np.array([0., 0.265, 0.829], dtype=float) * (Rv / 3.1)
        yAnchorOpt = np.array([
            -0.426 + 1.0044 * Rv,
            -0.050 + 1.0016 * Rv,
            0.701 + 1.0016 * Rv,
            1.208 + 1.0032 * Rv - 0.00033 * Rv**2
        ], dtype=float)
        yAnchorF99 = np.concatenate((yAnchorIR, yAnchorOpt, yAnchorUV))

        self.interpolator = interpolate.splrep(self.xAnchorF99, yAnchorF99, k=3, s=0)

    def _extinction(self, wavelength, ebv):
        """Get the extinction as a function of wavelength, A(lambda).

        Parameters
        ----------
        wavelength : `numpy.array` of `float`
            Wavelength in nm.
        ebv : `float`
            E(B-V).

        Returns
        -------
        ax : `numpy.array` of `float`
            Extinction as a function of wavelength, A(lambda) (magnitude).
        """
        x = 1. / (wavelength*1e-3)  # micron^(-1)
        axebv = interpolate.splev(x, self.interpolator)
        ax = axebv * ebv
        return ax

    def _fm90(self, x):
        """UV extinction curve of Fitzpatrick & Massa (1990)
        (Bibcode: 1990ApJS...72..163F)

        Parameters
        ----------
        x : `numpy.array` of `float`
            1/wavelength (micron^-1).

        Returns
        -------
        uvCurveAxebv : `numpy.array` of `float`
            Extinction curve, A(lamba)/E(B-V).
        """
        x = np.asarray(x, dtype='float')
        c2 = -0.824 + 4.717 / self._Rv
        c1 = 2.030 - 3.007 * c2
        x0 = 4.596
        gamma = 0.99
        c3 = 3.23
        c4 = 0.41

        linearTerm = c1 + c2 * x
        drudeTerm = x**2 / ((x**2 - x0**2)**2 + (x**2) * gamma**2)

        # no term for Far-UV because x must be less than 5.9 micron^-1
        fuvTerm = 0.

        uvCurve = (
            linearTerm +
            c3 * drudeTerm +
            c4 * fuvTerm
        )  # E(lambda-V)/E(B-V)

        # convert E(lambda-V)/E(B-V) to A(lambda)/E(B-V)
        return uvCurve + self._Rv

    @property
    def Rv(self):
        """Ratio of total to selective extinction at V, Rv = A(V)/E(B-V)

        This property is read-only.

        Returns
        -------
        Rv : `float`
            Rv = A(V)/E(B-V).
        """
        return self._Rv
