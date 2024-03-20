#ifndef PFS_DRP_STELLA_MATH_LINEARINTERPOLATION_H
#define PFS_DRP_STELLA_MATH_LINEARINTERPOLATION_H

#include "ndarray.h"
#include "ndarray/eigen.h"

#include "lsst/pex/exceptions.h"

#include "pfs/drp/stella/utils/checkSize.h"

namespace pfs {
namespace drp {
namespace stella {
namespace math {

/// Linear interpolation
template <typename T, int C1, int C2>
class LinearInterpolator {
  public:
    using XArray = ndarray::Array<T, 1, C1>;
    using YArray = ndarray::Array<T, 1, C2>;

    /// Ctor
    ///
    /// Input ordinates (x values) must be monotonic increasing.
    ///
    /// @param xFrom : x values (monotonic increasing)
    /// @param yFrom : y values
    /// @param yLow : value to return for x < min(xFrom)
    /// @param yHigh : value to return for x > max(xFrom)
    LinearInterpolator(XArray const& xFrom, YArray const& yFrom, T yLow=0, T yHigh=0)
      : _xFrom(xFrom), _yFrom(yFrom), _size(xFrom.size()), _yLow(yLow), _yHigh(yHigh) {
        utils::checkSize(xFrom.size(), yFrom.size(), "x vs y");
        if (xFrom.size() == 0) {
            throw LSST_EXCEPT(lsst::pex::exceptions::LengthError, "Interpolation requires non-empty x array");
        }
        for (std::size_t ii = 1; ii < xFrom.size(); ++ii) {
            if (xFrom[ii] <= xFrom[ii - 1]) {
                std::ostringstream os;
                os << "Non-monotonic-increasing ordinates:  x[ " << ii << "] == " << xFrom[ii] <<
                    " < x[" << ii - 1 << "] == " << xFrom[ii - 1] << std::endl;
                throw LSST_EXCEPT(lsst::pex::exceptions::LengthError, os.str());
            }
        }
    }

    LinearInterpolator(LinearInterpolator const&) = default;
    LinearInterpolator(LinearInterpolator&&) = default;
    LinearInterpolator& operator=(LinearInterpolator const&) = default;
    LinearInterpolator& operator=(LinearInterpolator&&) = default;
    ~LinearInterpolator() = default;

    //@{
    /// Interpolate
    T operator()(T xTo) const {
        auto const xFrom = ndarray::asEigenArray(_xFrom);
        auto const yFrom = ndarray::asEigenArray(_yFrom);
        if (xTo < xFrom[0]) {
            return _yLow;
        }
        if (xTo > xFrom[_size - 1]) {
            return _yHigh;
        }
        if (_size == 1) {
            return yFrom[0];
        }
        auto const it = std::lower_bound(xFrom.data(), xFrom.data() + _size, xTo);
        if (it == xFrom.data() + _size) {
            return yFrom[_size - 1];
        }
        if (it == xFrom.data()) {
            return yFrom[0];
        }
        auto const index = it - xFrom.data();

        T const yPrev = yFrom[index - 1];
        T const yNext = yFrom[index];
        T const xPrev = xFrom[index - 1];
        T const xNext = xFrom[index];
        return yPrev + (yNext - yPrev)*(xTo - xPrev)/(xNext - xPrev);
    }
    ndarray::Array<T, 1, 1> operator()(XArray const& xTo) const {
        ndarray::Array<T, 1, 1> yTo = ndarray::allocate(xTo.size());
        for (std::size_t ii = 0; ii < xTo.size(); ++ii) {
            yTo[ii] = (*this)(xTo[ii]);
        }
        return yTo;
    }
    YArray operator()(XArray const& xTo, YArray & yTo) const {
        checkSize(xTo.size(), yTo.size(), "xTo vs yTo");
        for (std::size_t ii = 0; ii < xTo.size(); ++ii) {
            yTo[ii] = (*this)(xTo[ii]);
        }
        return yTo;
    }
    //@}

  private:
    XArray const _xFrom;  ///< x values
    YArray const _yFrom;  ///< y values
    std::size_t const _size;  ///< size of xFrom and yFrom
    T const _yLow;  ///< value to return for x < min(_xFrom)
    T const _yHigh;  ///< value to return for x > max(_xFrom)
};

template <typename T, int C1, int C2>
T linearInterpolate(
    ndarray::Array<T, 1, C1> const& xFrom,
    ndarray::Array<T, 1, C2> const& yFrom,
    T xTo,
    T yLow=0,
    T yHigh=0
) {
    return LinearInterpolator(xFrom, yFrom, yLow, yHigh)(xTo);
}
template <typename T, int C1, int C2, int C3>
ndarray::Array<T, 1, 1> linearInterpolate(
    ndarray::Array<T, 1, C1> const& xFrom,
    ndarray::Array<T, 1, C2> const& yFrom,
    ndarray::Array<T, 1, C3> const& xTo,
    T yLow=0,
    T yHigh=0
) {
    return LinearInterpolator(xFrom, yFrom, yLow, yHigh)(xTo);
}
template <typename T, int C1, int C2, int C3, int C4>
ndarray::Array<T, 1, C4> linearInterpolate(
    ndarray::Array<T, 1, C1> const& xFrom,
    ndarray::Array<T, 1, C2> const& yFrom,
    ndarray::Array<T, 1, C3> const& xTo,
    ndarray::Array<T, 1, C4> yTo,
    T yLow=0,
    T yHigh=0
) {
    return LinearInterpolator(xFrom, yFrom, yLow, yHigh)(xTo, yTo);
}


}}}}  // namespace pfs::drp::stella::math

#endif
