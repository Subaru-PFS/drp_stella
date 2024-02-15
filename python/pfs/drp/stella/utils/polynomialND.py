import numpy as np

import math

from typing import List, Union
from typing import overload


__all__ = ["NormalizedPolynomialND"]


class NormalizedPolynomialND:
    """``nVars``-variate polynomial whose domain is normalized to ``[-1, 1]^nVars``.

    The suffix "D" ("Double") is after NormalizedPolynomial1D and -2D.

    ``nVars`` is guessed from ``min`` and ``max``. If both ``min`` and ``max``
    are scalars, then all appearances of ``nVars`` in docstrings of this class
    should be read "", or the empty string. For example, ``(nVars,)`` should
    be read ``()`` and ``(a, b, nVars)`` should be read ``(a, b)``.

    Parameters
    ----------
    params : `np.ndarray|int`
        Parameters (coefficients) of a polynomial.
        If this argument is of `int` type, it is interpreted as a polynomial
        order ``order`` and a zero polynomial of order ``order`` is created.
    min : `np.ndarray`, shape ``(nVars,)``
        Any vertex of the hyperrectangular domain.
    max : `np.ndarray`, shape ``(nVars,)``
        The vertex, of the hyperrectangular domain, that is the farthest
        from ``min``.
    """

    @overload
    def __init__(self, order: int, min: Union[np.ndarray, float], max: Union[np.ndarray, float]) -> None:
        ...

    @overload
    def __init__(
        self, params: np.ndarray, min: Union[np.ndarray, float], max: Union[np.ndarray, float]
    ) -> None:
        ...

    def __init__(self, params, min, max):
        self._min = np.minimum(min, max, dtype=float)
        self._max = np.maximum(min, max, dtype=float)

        if self._min.ndim == 0:
            self._isScalarDomain = True
            self._min = self._min.reshape(1)
            self._max = self._max.reshape(1)
        elif self._min.ndim == 1:
            self._isScalarDomain = False
        else:
            raise ValueError("`min` and `max` must be at most one-dimensional.")

        nVars = len(self._min)
        if nVars == 0:
            raise ValueError("0-variate polynomial is not allowed.")

        params_arr = np.asarray(params)
        if params_arr.ndim == 0:
            # `params` is not parameters but a polynomial order in fact.
            order = int(params)
            params = np.zeros(shape=(math.comb(nVars + order, nVars),), dtype=float)
        elif params_arr.ndim == 1:
            # `params` are parameters indeed.
            params = np.asarray(params_arr, dtype=float)
            order = getPolynomialOrder(nVars, len(params))
        else:
            raise ValueError("`params` must be one-dimensional.")

        self._order = order
        self._params = params
        self._scale = 1.0 / (self._max - self._min)
        self._exponents = getExponents(order, nVars)

    def __call__(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Evaluate the polynomial at ``x``.

        Parameters
        ----------
        x : `np.ndarray` of `float`, shape ``(..., nVars)``
            Point at which to evaluate the polynomial.

        Returns
        -------
        y : `np.ndarray` of `float`, shape ``(...)``
            ``f(x)``. The shape of the return value is that of ``x`` with
            the last dimension stripped.
        """
        xToExponents = self.getDFuncDParameters(x)
        return xToExponents @ self._params

    def getDFuncDParameters(self, x: Union[np.ndarray, float]) -> np.ndarray:
        """Get ``df / d params`` (it is not ``df/dx``) at ``x``.

        Because the polynomial is
            f = sum(
                params[i] * x[0]**expons[i][0] * ... * params[i] * x[nVars-1]**expons[i][nVars-1]
                for i in range(len(params))
            )
        ``df / d params[i]`` is
        ``x[0]**expons[i][0] * ... * params[i] * x[nVars-1]**expons[i][nVars-1]``.

        Parameters
        ----------
        x : `np.ndarray` of `float`, shape ``(..., nVars)``
            Point at which to evaluate ``df / d params``

        Returns
        -------
        dFuncDParams : `np.ndarray` of `float`, shape ``(..., nParams)``
            List of ``df / d params[i]`` for all [i].
        """
        x = np.array(x, dtype=float, copy=True)
        nVars = len(self._min)

        if self._isScalarDomain:
            x = x.reshape(x.shape + (1,))
        else:
            if x.ndim == 0 or x.shape[-1] != nVars:
                raise ValueError(f"Size of the last dimension of `x` must be {nVars}")

        min = self._min.reshape((1,) * (x.ndim - 1) + (nVars,))
        scale = self._scale.reshape((1,) * (x.ndim - 1) + (nVars,))

        x -= min
        x *= scale

        # Note: If x.shape is (a, b, ..., c, nVars),
        # then xShape will be (a, b, ..., c, 1, nVars)
        xShape = x.shape[:-1] + (1,) + x.shape[-1:]
        # Note: If _exponents.shape is (nParams, nVars),
        # then exponentsShape will be (1, 1, ..., 1, nParams, nVars).
        exponentsShape = (1,) * (x.ndim - 1) + self._exponents.shape

        # xToExponents has shape (a, b, ..., c, nParams).
        xToExponents = np.prod(x.reshape(xShape) ** self._exponents.reshape(exponentsShape), axis=(-1,))
        return xToExponents

    def getOrder(self) -> int:
        """Get the polynomial order

        Returns
        -------
        order : `int`
            Polynomial order.
        """
        return self._order

    def getMin(self) -> Union[np.ndarray, float]:
        """Get the bottom-left vertex of the hyperrectangular domain.

        Returns
        -------
        min : `np.ndarray` of `float`, shape ``(nVars,)``
            Bottom-left vertex.
        """
        if self._isScalarDomain:
            return float(self._min)
        else:
            return self._min

    def getMax(self) -> Union[np.ndarray, float]:
        """Get the top-bottom vertex of the hyperrectangular domain.

        Returns
        -------
        max : `np.ndarray` of `float`, shape ``(nVars,)``
            top-bottom vertex.
        """
        if self._isScalarDomain:
            return float(self._max)
        else:
            return self._max

    def getParams(self) -> np.ndarray:
        """Get the parameters of this polynomial.

        Returns
        -------
        params : `np.ndarray` of `float`, shape ``(nParams,)``
            Parameters.
        """
        return self._params

    @staticmethod
    def getParamsFromLowerVariatePoly(params: np.ndarray, variables: List[Union[int, None]]) -> np.ndarray:
        """Get the parameters for N-variate polynomial that is equal to the
        M-variate polynomial whose parameters are ``params`` (N >= M).

        Parameters
        ----------
        params : `np.ndarray`
            Parameters for M-variate polynomial.
        variables : `List[int|None]`
            Mapping of variables from N-variate one to M-variate one.
            ``i``-th variable of N-variate one is identified with
            ``variables[i]``-th variable of M-variate one. If ``variables[i]``
            is ``None``, then ``i``-th variable of N-variate one does not
            appear in M-variate one.

        Returns
        -------
        paramsN : `np.ndarray`
            Parameters for N-variate polynomial.
        """
        nVarsM = max(i for i in variables if i is not None) + 1
        nParamsM = len(params)
        order = getPolynomialOrder(nVarsM, nParamsM)
        nVarsN = len(variables)
        nParamsN = math.comb(nVarsN + order, nVarsN)

        exponentsM = getExponents(order, nVarsM)
        exponentsN = getExponents(order, nVarsN)

        exponentsN_inverse = {tuple(v): k for k, v in enumerate(exponentsN)}
        paramsN = np.zeros(shape=(nParamsN,), dtype=float)
        for indexM, exponsM in enumerate(exponentsM):
            exponsN = tuple((exponsM[i] if i is not None else 0) for i in variables)
            indexN = exponentsN_inverse[exponsN]
            paramsN[indexN] = params[indexM]

        return paramsN


def getPolynomialOrder(nVars: int, nParams: int) -> int:
    """Find the ``order`` of an ``nVars``-variate polynomial
    that has ``nParams`` parameters.

    It is ``order`` that satisfies ``math.comb(nVars + order, nVars) == nParams``.
    """
    if nVars <= 0:
        raise ValueError("nVars must be >= 1")
    if nParams <= 0:
        raise ValueError("nParams must be >= 1")

    # We don't expect ``order`` is very large. Let us brute-force it.
    step = 5
    startOrder = 0
    stopOrder = startOrder + step
    while True:
        n = math.comb(nVars + stopOrder, nVars)
        if n >= nParams:
            break
        startOrder = stopOrder
        stopOrder += step

    if n == nParams:
        return stopOrder

    for order in range(startOrder, stopOrder):
        n = math.comb(nVars + order, nVars)
        if n >= nParams:
            if n == nParams:
                return order
            else:
                break

    raise ValueError(f"There cannot be a {nVars}-variate polynomial with {nParams} parameters")


def getExponents(order: int, nVars: int) -> np.ndarray:
    """Get list of exponents ``(p[0],.., p[nVars-1])`` of all possible terms
    ``x[0]**p[0] * ... * x[nVars-1]**p[nVars-1]`` in an ``nVars``-variate
    polynomial.

    The returned list is sorted so that
      - the elements are arranged in descending order of ``sum(p)``.
      - Ties are broken according reversely to tuple order of ``(p[0],...,p[nVars-1])``

    Parameters
    ----------
    order : `int`
        Polynomial order.
    nVars : `int`
        Number of variables of the polynomial.

    Returns
    -------
    exponents : `np.ndarray` of `int`, shape ``(nTerms, nVars)``
        List of exponents ``(p[0],.., p[nVars])`` for all possible terms
        ``x[0]**p[0] * ... * x[nVars-1]**p[nVars-1]``.
    """

    def key(expons: List[int]) -> List[int]:
        """Key for sorting.

        Parameters
        ----------
        expons : `List[int]`
            A tuple of exponents``(p[0],.., p[nVars-1])`` as described
            in the docstring of the parent function.

        Returns
        -------
        key : `List[int]`
            Key by which to sort the list.
        """
        return [-sum(expons)] + [-p for p in expons]

    exponents = _getExponents(order, nVars)
    exponents.sort(key=key)
    return np.array(exponents, dtype=int)


def _getExponents(order: int, nVars: int) -> List[List[int]]:
    """Get list of exponents ``(p[0],.., p[nVars-1])`` of all possible terms
    ``x[0]**p[0] * ... * x[nVars-1]**p[nVars-1]`` in an ``nVars``-variate
    polynomial.

    Parameters
    ----------
    order : `int`
        Polynomial order.
    nVars : `int`
        Number of variables of the polynomial.

    Returns
    -------
    exponents : `List[List[int]]`, shape ``(nTerms, nVars)``
        List of exponents ``(p[0],.., p[nVars])`` for all possible terms
        ``x[0]**p[0] * ... * x[nVars-1]**p[nVars-1]``.
    """
    if nVars >= 2:
        exponents: List[List[int]] = []
        for p in range(order + 1):
            subexponents = _getExponents(order - p, nVars - 1)
            for subexpons in subexponents:
                subexpons.append(p)
            exponents.extend(subexponents)
        return exponents
    else:
        return [[p] for p in range(order + 1)]
