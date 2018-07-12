#include <iostream>
#include <sstream>

#include "ndarray.h"
#include "lsst/pex/exceptions.h"

#include "pfs/drp/stella/spline.h"

namespace pfs {
namespace drp {
namespace stella {
namespace math {

/************************************************************************************************************/
namespace {
/*
 * solves tridiagonal linear system .
 *
 * Diagonal elements m_{i,i} = a[i]; subdiag m_{i,i-1} = b[i]; superdiag m_{i,i+1} = c[i]
 *
 * it is understood that b[0] and c[n-1] are zero, but are not referenced. f is rhs
 *
 * N.b. all arrays are trashed, and the result is returned in the storage formerly
 * occupied by f
 */
template<typename T>
void
tridi(ndarray::Array<T, 1, 1> &b,
      ndarray::Array<T, 1, 1> &a,
      ndarray::Array<T, 1, 1> &c,
      ndarray::Array<T, 1, 1> &f
) {
    std::size_t const n = a.size();

    c[0] /= a[0];
    for (std::size_t i = 1; i != n; ++i) {
        c[i] /= (a[i] - b[i]*c[i-1]);
    }
    f[0] /= a[0];

    for (std::size_t i = 1; i != n; ++i) {
        f[i] = (f[i] - b[i]*f[i-1])/(a[i] - b[i]*c[i-1]);
    }
    //
    // Put the answer into f
    //
    // N.b. note that the code is written with assignments to "f" on the LHS
    // to make it clear that f is now being reused as the result
    //
    auto &x = f;
    x[n - 1] = f[n - 1];
    for(int i = n - 2;i >= 0;i--) {
        x[i] = f[i] - c[i]*x[i+1];
    }
}

} // anonymous namespace

template<typename T>
Spline<T>::Spline(
    ndarray::Array<T const, 1, 1> const& x,
    ndarray::Array<T const, 1, 1> const& y,
    InterpolationTypes interpolationType
) {
    std::size_t const n = x.size();
    if (n < 3) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthError, "At least 3 points are needed to fit a spline");
    }
    //
    // Check if x is sorted, and see if all the intervals are the same
    //
    {
        T xOld = x[0];
        T const dxOld = x[1] - x[0];
        _xIsUniform = true;              // innocent until proven guilty
        for (std::size_t i = 1; i != n; ++i) {
            T const dx = x[i] - xOld;
            if (dx <= 0) {
                std::ostringstream os;
                os << "Non-monotonic-increasing ordinates:  x[ " << i << "] == " << x[i] <<
                    " < x[" << i - 1 << "] == " << xOld << std::endl;
                throw LSST_EXCEPT(lsst::pex::exceptions::LengthError, os.str());
            }
            if (dx != dxOld) {
                _xIsUniform = false;
            }
            xOld = x[i];
        }
    }
    //
    // If all the intervals are exactly the same we can make the interpolation a bit faster
    //
    if (_xIsUniform) {
        _m0 = n/2;
        _dmdx = (n - 1.0)/(x[n - 1] - x[0]);
    }
    
    /*********************************************************************************************************/

    _x = ndarray::allocate(n);
    _y = ndarray::allocate(n);
    _x.deep() = x;
    _y.deep() = y;
    _k = ndarray::allocate(n);
    /*
     * We store the tridigonal system M in the three arrays dm1, d, and dp1 where:
     *   Diagonal elements M_{i,i}   = dia[i]
     *   subdiag           M_{i,i-1} = dm1[i]
     *   superdiag         M_{i,i+1} = dp1[i]
     *
     * I.e.
     *  ( dia[0]  dp1[0]  0       0       0       0                                             )
     *  ( dm1[1]  dia[1]  dp1[1]  0       0       0                                             )
     *  ( 0       dm1[2]  dia[2]  dp1[2]  0       0                                             )
     *  ( 0       0       dm1[3]  dia[3]  dp1[3]  0  ...                                        )
     *  (                                            ...                                        )
     *  (                                         0  ...  dm1[n-3] dia[n-3]  dp1[n-3]  0        )
     *  (                                         0  ...  0        dm1[n-2]  dia[n-2]  dp1[n-2] )
     *  (                                         0  ...  0        0         dm1[n-1]  dia[n-1] )
     *  (                                              
     *
     * Note that dm1[0] and dp1[n-1] are not referenced
     */
    Array dm1 = ndarray::allocate(n);       // below diagonal; was b
    Array dia = ndarray::allocate(n);       // diagonal; was a
    Array dp1 = ndarray::allocate(n);       // above diagonal; was c
    Array &rhs = _k;           // the rhs; shares storage with _k

    T h_p =       x[1] - x[0];         // previous value of h
    T delta_p =  (y[1] - y[0])/h_p;    // previous value of delta
    for (std::size_t i = 1; i != n - 1; ++i) { // n.b. no first or last column, we'll set them soon
        T const h =      x[i + 1] - x[i];
        T const delta = (y[i + 1] - y[i])/h;
        dm1[i] = h;
        dia[i] = 2*(h + h_p);
        dp1[i] = h_p;
        rhs[i] = 3.0*(h*delta_p + h_p*delta);
            
        delta_p = delta;
        h_p = h;
    }

   switch (interpolationType) {
     case CUBIC_NOTAKNOT:
       {
           T h_p =      x[1] - x[0];
           T delta_p = (y[1] - y[0])/h_p;
           T h =        x[2] - x[1];
           T delta =   (y[2] - y[1])/h;
           
           dia[0] =   h;
           dp1[0] =   h + h_p;
           rhs[0] = ((2*h*h + 3*h*h_p)*delta_p + h_p*h_p*delta)/(h + h_p);
           
           h_p =      x[n - 2] - x[n - 3];
           delta_p = (y[n - 2] - y[n - 3])/h_p;
           h =        x[n - 1] - x[n - 2];
           delta =   (y[n - 1] - y[n - 2])/h;
           
           dm1[n - 1] = h + h_p;
           dia[n - 1] = h_p;
           rhs[n - 1] = (h*h*delta_p + (2*h_p*h_p + 3*h*h_p)*delta)/(h + h_p);
       }
       break;
     case CUBIC_NATURAL:
       {
           T h_p =      x[1] - x[0];
           T delta_p = (y[1] - y[0])/h_p;
           
           dia[0] = 2*h_p;
           dp1[0] =   h_p;
           rhs[0] = 3.0*h_p*delta_p;
           
           h_p =      x[n - 1] - x[n - 2];
           delta_p = (y[n - 1] - y[n - 2])/h_p;
           
           dm1[n - 1] =   h_p;
           dia[n - 1] = 2*h_p;
           rhs[n - 1] = 3.0*h_p*delta_p;
       }
       break;
   }

   tridi(dm1, dia, dp1, rhs);           // n.b. rhs is aliased to _k
}

/*********************************************************/
/*
 * Evaluate our spline
 */
template<typename T>
T Spline<T>::operator()(T const z) const
{
    int const m = _findIndex(z);         // index of the next LARGEST point in _x; clipped in [1, n-1]

    T const h =      _x[m] - _x[m-1];
    T const delta = (_y[m] - _y[m-1])/h;
    
    T const dx = z - _x[m-1];
    T const t = dx/h;
    T const a = (_k[m-1] - delta)*(1 - t);
    T const b = (_k[m]   - delta)*t;
    
    return t*_y[m] + (1 - t)*_y[m-1] + h*t*(1 - t)*(a - b);
}

/*
 * Find the index of the element in x which comes after z; return 1 or n-1 if out of range
 */
template<typename T>
int Spline<T>::_findIndex(T z) const    // point whose index we want
{
    int const n = _x.size();
    if (z <= _x[1]) {
        return 1;
    } else if(z >= _x[n - 1]) {
        return n - 1;
    }
    
    if (_xIsUniform) {
        return 1 + _m0 + (z - _x[_m0])*_dmdx;
    }

    int lo = 0;
    int hi = n - 1;

    while (hi - lo > 1) {
        int mid = (lo + hi)/2;
        if (z < _x[mid]) {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    return hi;
}
    
/************************************************************************************************************/

template class Spline<float>;

}}}}
