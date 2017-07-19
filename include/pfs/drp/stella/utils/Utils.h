#ifndef __PFS_DRP_STELLA_UTILS_H__
#define __PFS_DRP_STELLA_UTILS_H__

#include <iostream>
#include <vector>

#include "lsst/afw/fits.h"
#include "lsst/afw/image.h"
#include "ndarray.h"
#include "pfs/drp/stella/math/CurveFitting.h"
#include "pfs/drp/stella/math/Math.h"

namespace afwImage = lsst::afw::image;

using namespace std;
namespace pfs { namespace drp { namespace stella { namespace utils{
  int KeyWord_Set(vector<string> const& S_A1_In,
                  string const& str_In);
    
  template<typename T>
  ndarray::Array<T, 2, 1> get2DndArray(T nRows, T nCols);
    
  template<typename T>
  ndarray::Array<T, 1, 1> get1DndArray(T size);

  template<typename T>
  ndarray::Array<T, 1, 1> vectorToNdArray(std::vector<T> & vec);
  template<typename T>
  ndarray::Array<T const, 1, 1> vectorToNdArray(std::vector<T> const& vec);

  template<typename T, typename U>
  ndarray::Array<U, 1, 1> typeCastNdArray(ndarray::Array<T const, 1, 1> const& arr, U const& newType);

  /*
   * @brief: Test functionality of PolyFit
   * We can't include the tests in Python as the keyword arguments (vector of void pointers)
   * does not get swigged.
   */
  void testPolyFit();
}}}}
#endif
