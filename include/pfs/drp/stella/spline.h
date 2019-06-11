#if !defined(DRP_STELLA_MATH_SPLINE_H)
#define DRP_STELLA_MATH_SPLINE_H 1

#include "ndarray_fwd.h"

namespace pfs {
namespace drp {
namespace stella {
namespace math {

///
// \brief cubic spline interpolation
//
template<typename T>
class Spline {
public:
    enum InterpolationTypes { CUBIC_NOTAKNOT, CUBIC_NATURAL };
    using Array = ndarray::Array<T, 1, 1>;
    using ConstArray = ndarray::Array<T const, 1, 1>;

    Spline(ConstArray const& x,                            ///< positions of knots
           ConstArray const& y,                            ///< values of function at knots
           InterpolationTypes interpolationType=CUBIC_NOTAKNOT ///< spline boundary conditions
          );
    Spline(
        std::vector<T> const& x,
        std::vector<T> const& y,
        InterpolationTypes interpolationType=CUBIC_NOTAKNOT
    ) : Spline(
        ConstArray(ndarray::external(x.data(), ndarray::makeVector(x.size()))),
        ConstArray(ndarray::external(y.data(), ndarray::makeVector(y.size()))),
        interpolationType
    ) {}

    // That ctor disables the move ctors
    Spline(Spline const&) = default;
    Spline(Spline &&) = default;
    Spline& operator=(Spline const &) = default;
    Spline& operator=(Spline &&) = default;

    T operator() (T const x) const;
    Array operator() (Array const array) const;

    ConstArray const getX() const { return _x; }
    ConstArray const getY() const { return _y; }
    
private:
    Array _x, _y;              // x,y coordinates of points
    Array _k;                  // slope at points, used for interpolation
    //
    // These are only used if all the intervals in _x are equal
    //
    bool _xIsUniform;                   // are all the intervals the same?
    int _m0;                            // index of point at/near the middle of the array
    double _dmdx;                       // amount x increases for between points; == x[1] - x[0]

    int _findIndex(T z) const;          // point whose index we want
};

}}}}

#endif
