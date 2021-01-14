#if !defined(DRP_STELLA_MATH_SPLINE_H)
#define DRP_STELLA_MATH_SPLINE_H 1

#include "ndarray_fwd.h"
#include "lsst/afw/table/io/Persistable.h"

#include "pfs/drp/stella/utils/math.h"

namespace pfs {
namespace drp {
namespace stella {
namespace math {

///
// \brief cubic spline interpolation
//
template<typename T>
class Spline : public lsst::afw::table::io::Persistable {
public:
    enum InterpolationTypes { CUBIC_NOTAKNOT, CUBIC_NATURAL };
    using InternalT = double;
    using Array = ndarray::Array<InternalT, 1, 1>;
    using ConstArray = ndarray::Array<InternalT const, 1, 1>;

    Spline(ConstArray const& x,                            ///< positions of knots
           ConstArray const& y,                            ///< values of function at knots
           InterpolationTypes interpolationType=CUBIC_NOTAKNOT ///< spline boundary conditions
          );
    Spline(
        ndarray::Array<T, 1, 1> const& x,
        ndarray::Array<T, 1, 1> const& y,
        InterpolationTypes interpolationType=CUBIC_NOTAKNOT ///< spline boundary conditions
    ) : Spline(utils::convertArray<InternalT>(x), utils::convertArray<InternalT>(y),
               interpolationType) {}
    Spline(
        std::vector<InternalT> const& x,
        std::vector<InternalT> const& y,
        InterpolationTypes interpolationType=CUBIC_NOTAKNOT
    ) : Spline(ConstArray(utils::vectorToArray(x)), ConstArray(utils::vectorToArray(y)),
                          interpolationType) {}
    Spline(
        std::vector<T> const& x,
        std::vector<T> const& y,
        InterpolationTypes interpolationType=CUBIC_NOTAKNOT
    ) : Spline(utils::convertArray<InternalT>(utils::vectorToArray(x)),
               utils::convertArray<InternalT>(utils::vectorToArray(y)),
               interpolationType) {}

    // That ctor disables the move ctors
    Spline(Spline const&) = default;
    Spline(Spline &&) = default;
    Spline& operator=(Spline const &) = default;
    Spline& operator=(Spline &&) = default;

    T operator() (T const x) const;
    ndarray::Array<T, 1, 1> operator() (ndarray::Array<T, 1, 1> const& array) const;

    ConstArray const getX() const { return _x; }
    ConstArray const getY() const { return _y; }
    InterpolationTypes getInterpolationType() const { return _interpolationType; }

    bool isPersistable() const noexcept { return true; }

    class Factory;

  protected:
    std::string getPersistenceName() const { return "Spline"; }
    std::string getPythonModule() const { return "pfs.drp.stella"; }
    void write(lsst::afw::table::io::OutputArchiveHandle & handle) const;

  private:
    Array _x, _y;              // x,y coordinates of points
    Array _k;                  // slope at points, used for interpolation
    //
    // These are only used if all the intervals in _x are equal
    //
    bool _xIsUniform;                   // are all the intervals the same?
    int _m0;                            // index of point at/near the middle of the array
    double _dmdx;                       // amount x increases for between points; == x[1] - x[0]
    InterpolationTypes _interpolationType;  // type of spline interpolation

    int _findIndex(T z) const;          // point whose index we want
};

}}}}

#endif
