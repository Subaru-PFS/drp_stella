#if !defined(DRP_STELLA_MATH_SPLINE_H)
#define DRP_STELLA_MATH_SPLINE_H 1

namespace pfs { namespace drp { namespace stella { namespace math { 
///
// \brief cubic spline interpolation
//
template<typename T>
class spline {
public:
    enum InterpolationTypes { CUBIC_NOTAKNOT, CUBIC_NATURAL };

    spline() = default;                                        // needed as we have vectors<spline>
    spline(std::vector<T> const& x,                            ///< positions of knots
           std::vector<T> const& y,                            ///< values of function at knots
           InterpolationTypes interpolationType=CUBIC_NOTAKNOT ///< spline boundary conditions
          );
    // That ctor disables the move ctors
    spline(const spline &) = default;
    spline(spline &&) = default;
    spline& operator=(const spline &) = default;
    spline& operator=(spline &&) = default;

    T operator() (T const x) const;
private:
    std::vector<T> _x, _y;              // x,y coordinates of points
    std::vector<T> _k;                  // slope at points, used for interpolation
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
