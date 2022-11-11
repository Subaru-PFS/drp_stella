#
# Merged from various files which form part of matplotlib 3.6.0
#
import matplotlib.colors

__all__ = ["AsinhNorm"]

try:
    from matplotlib.colors import AsinhNorm
except ImportError:
    AsinhNorm = None

if AsinhNorm is None:

    import matplotlib.scale
    import matplotlib.ticker
    import math
    import numpy as np

    class AsinhTransform(matplotlib.scale.Transform):
        """Inverse hyperbolic-sine transformation used by `.AsinhScale`"""
        input_dims = output_dims = 1

        def __init__(self, linear_width):
            super().__init__()
            if linear_width <= 0.0:
                raise ValueError("Scale parameter 'linear_width' " +
                                 "must be strictly positive")
            self.linear_width = linear_width

        def transform_non_affine(self, a):
            return self.linear_width * np.arcsinh(a / self.linear_width)

        def inverted(self):
            return InvertedAsinhTransform(self.linear_width)

    class InvertedAsinhTransform(matplotlib.scale.Transform):
        """Hyperbolic sine transformation used by `.AsinhScale`"""
        input_dims = output_dims = 1

        def __init__(self, linear_width):
            super().__init__()
            self.linear_width = linear_width

        def transform_non_affine(self, a):
            return self.linear_width * np.sinh(a / self.linear_width)

        def inverted(self):
            return AsinhTransform(self.linear_width)

    class AsinhLocator(matplotlib.ticker.Locator):
        """
        An axis tick locator specialized for the inverse-sinh scale
        This is very unlikely to have any use beyond
        the `~.scale.AsinhScale` class.
        .. note::
           This API is provisional and may be revised in the future
           based on early user feedback.
        """
        def __init__(self, linear_width, numticks=11, symthresh=0.2,
                     base=10, subs=None):
            """
            Parameters
            ----------
            linear_width : float
                The scale parameter defining the extent
                of the quasi-linear region.
            numticks : int, default: 11
                The approximate number of major ticks that will fit
                along the entire axis
            symthresh : float, default: 0.2
                The fractional threshold beneath which data which covers
                a range that is approximately symmetric about zero
                will have ticks that are exactly symmetric.
            base : int, default: 10
                The number base used for rounding tick locations
                on a logarithmic scale. If this is less than one,
                then rounding is to the nearest integer multiple
                of powers of ten.
            subs : tuple, default: None
                Multiples of the number base, typically used
                for the minor ticks, e.g. (2, 5) when base=10.
            """
            super().__init__()
            self.linear_width = linear_width
            self.numticks = numticks
            self.symthresh = symthresh
            self.base = base
            self.subs = subs

        def set_params(self, numticks=None, symthresh=None,
                       base=None, subs=None):
            """Set parameters within this locator."""
            if numticks is not None:
                self.numticks = numticks
            if symthresh is not None:
                self.symthresh = symthresh
            if base is not None:
                self.base = base
            if subs is not None:
                self.subs = subs if len(subs) > 0 else None

        def __call__(self):
            vmin, vmax = self.axis.get_view_interval()
            if (vmin * vmax) < 0 and abs(1 + vmax / vmin) < self.symthresh:
                # Data-range appears to be almost symmetric, so round up:
                bound = max(abs(vmin), abs(vmax))
                return self.tick_values(-bound, bound)
            else:
                return self.tick_values(vmin, vmax)

        def tick_values(self, vmin, vmax):
            # Construct a set of "on-screen" locations
            # that are uniformly spaced:
            ymin, ymax = self.linear_width * np.arcsinh(np.array([vmin, vmax])
                                                            / self.linear_width)
            ys = np.linspace(ymin, ymax, self.numticks)
            zero_dev = np.abs(ys / (ymax - ymin))
            if (ymin * ymax) < 0:
                # Ensure that the zero tick-mark is included,
                # if the axis straddles zero
                ys = np.hstack([ys[(zero_dev > 0.5 / self.numticks)], 0.0])

            # Transform the "on-screen" grid to the data space:
            xs = self.linear_width * np.sinh(ys / self.linear_width)
            zero_xs = (ys == 0)

            # Round the data-space values to be intuitive base-n numbers,
            # keeping track of positive and negative values separately,
            # but giving careful treatment to the zero value:
            if self.base > 1:
                log_base = math.log(self.base)
                powers = (
                    np.where(zero_xs, 0, np.sign(xs)) *
                    np.power(self.base,
                             np.where(zero_xs, 0.0,
                                      np.floor(np.log(np.abs(xs) + zero_xs*1e-6)
                                                    / log_base)))
                )
                if self.subs:
                    qs = np.outer(powers, self.subs).flatten()
                else:
                    qs = powers
            else:
                powers = (
                    np.where(xs >= 0, 1, -1) *
                    np.power(10, np.where(zero_xs, 0.0,
                                          np.floor(np.log10(np.abs(xs)
                                                            + zero_xs*1e-6))))
                )
                qs = powers * np.round(xs / powers)
            ticks = np.array(sorted(set(qs)))

            if len(ticks) >= 2:
                return ticks
            else:
                return np.linspace(vmin, vmax, self.numticks)


    class AsinhScale(matplotlib.scale.ScaleBase):
        """
        A quasi-logarithmic scale based on the inverse hyperbolic sine (asinh)
        For values close to zero, this is essentially a linear scale,
        but for large magnitude values (either positive or negative)
        it is asymptotically logarithmic. The transition between these
        linear and logarithmic regimes is smooth, and has no discontinuities
        in the function gradient in contrast to
        the `.SymmetricalLogScale` ("symlog") scale.
        Specifically, the transformation of an axis coordinate :math:`a` is
        :math:`a \\rightarrow a_0 \\sinh^{-1} (a / a_0)` where :math:`a_0`
        is the effective width of the linear region of the transformation.
        In that region, the transformation is
        :math:`a \\rightarrow a + \\mathcal{O}(a^3)`.
        For large values of :math:`a` the transformation behaves as
        :math:`a \\rightarrow a_0 \\, \\mathrm{sgn}(a) \\ln |a| + \\mathcal{O}(1)`.
        .. note::
           This API is provisional and may be revised in the future
           based on early user feedback.
        """

        name = 'asinh'

        auto_tick_multipliers = {
            3: (2, ),
            4: (2, ),
            5: (2, ),
            8: (2, 4),
            10: (2, 5),
            16: (2, 4, 8),
            64: (4, 16),
            1024: (256, 512)
        }

        def __init__(self, axis, *, linear_width=1.0,
                     base=10, subs='auto', **kwargs):
            """
            Parameters
            ----------
            linear_width : float, default: 1
                The scale parameter (elsewhere referred to as :math:`a_0`)
                defining the extent of the quasi-linear region,
                and the coordinate values beyond which the transformation
                becomes asymptotically logarithmic.
            base : int, default: 10
                The number base used for rounding tick locations
                on a logarithmic scale. If this is less than one,
                then rounding is to the nearest integer multiple
                of powers of ten.
            subs : sequence of int
                Multiples of the number base used for minor ticks.
                If set to 'auto', this will use built-in defaults,
                e.g. (2, 5) for base=10.
            """
            super().__init__(axis)
            self._transform = AsinhTransform(linear_width)
            self._base = int(base)
            if subs == 'auto':
                self._subs = self.auto_tick_multipliers.get(self._base)
            else:
                self._subs = subs

        linear_width = property(lambda self: self._transform.linear_width)

        def get_transform(self):
            return self._transform

        def set_default_locators_and_formatters(self, axis):
            axis.set(major_locator=AsinhLocator(self.linear_width,
                                                base=self._base),
                     minor_locator=AsinhLocator(self.linear_width,
                                                base=self._base,
                                                subs=self._subs),
                     minor_formatter=matplotlib.ticker.NullFormatter())
            if self._base > 1:
                axis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation(self._base))
            else:
                axis.set_major_formatter('{x:.3g}'),


    @matplotlib.colors.make_norm_from_scale(
        AsinhScale,
        init=lambda linear_width=1, vmin=None, vmax=None, clip=False: None)
    class AsinhNorm(matplotlib.colors.Normalize):
        """
        The inverse hyperbolic sine scale is approximately linear near
        the origin, but becomes logarithmic for larger positive
        or negative values. Unlike the `SymLogNorm`, the transition between
        these linear and logarithmic regions is smooth, which may reduce
        the risk of visual artifacts.
        .. note::
           This API is provisional and may be revised in the future
           based on early user feedback.
        Parameters
        ----------
        linear_width : float, default: 1
            The effective width of the linear region, beyond which
            the transformation becomes asymptotically logarithmic
        """

        @property
        def linear_width(self):
            return self._scale.linear_width

        @linear_width.setter
        def linear_width(self, value):
            self._scale.linear_width = value
