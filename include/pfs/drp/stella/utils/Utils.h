#ifndef __PFS_DRP_STELLA_UTILS_H__
#define __PFS_DRP_STELLA_UTILS_H__

#include <vector>

#include "ndarray.h"

namespace pfs { namespace drp { namespace stella { namespace utils{
  int KeyWord_Set(std::vector<std::string> const& S_A1_In,
                  std::string const& str_In);
    
  template<typename T>
  ndarray::Array<T, 2, 1> get2DndArray(T nRows, T nCols);
    
  template<typename T>
  ndarray::Array<T, 1, 1> get1DndArray(T size);

  template<typename T>
  ndarray::Array<T, 1, 1> vectorToNdArray(std::vector<T> & vec);
  template<typename T>
  ndarray::Array<T const, 1, 1> vectorToNdArray(std::vector<T> const& vec);

  template<typename T, typename U>
  ndarray::Array<U, 1, 1> typeCastNdArray(ndarray::Array<T, 1, 1> const& arr, U const& newType);

  /*
   * @brief: Test functionality of PolyFit
   * We can't include the tests in Python as the keyword arguments (vector of void pointers)
   * does not get swigged.
   */
  void testPolyFit();

  /**
   * @brief check a vector for a number
   * @param vec : vector to search for number
   * @param number : number to search for
   * @return -1 if not found, otherwise the position in vec where number was found
   */
  template<typename T>
  int find(std::vector<T> const& vec, T number);
}// End namespace utils
}}}
#endif
