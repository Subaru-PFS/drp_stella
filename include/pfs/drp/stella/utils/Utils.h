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

  /// removes leading and/or trailing spaces from str
  /// mode == 0: remove leading spaces
  /// mode == 1: remove trailing spaces
  /// mode == 2: remove leading and trailing spaces
  bool trimString(string &str, const int mode);

  /// removes leading and/or trailing 'chr' from str
  /// mode == 0: remove leading 'chr'
  /// mode == 1: remove trailing 'chr'
  /// mode == 2: remove leading and trailing 'chr'
  bool trimString(string &str, const char chr, const int mode);

  //  int sToI(const string &str);
  bool sToI(const string &str, int &I_Out);

  /**
    *       function int CountLines(const string &fnc: in)
    *       Returns number of lines of file fnc.
    **/
  long countLines(const string &fnc);

  /**
    *       function int CountDataLines(const string &fnc: in)
    *       Returns number of lines which do not start with '#' of file fnc.
    **/
  long countDataLines(const string &fnc);

  /**
    *      function int CountCols(const string &fileName_In: in, const string &delimiter: in)
    *      Returns number of columns of file fileName_In.
    **/
  long countCols(const string &fileName_In, const string &delimiter_In);

  bool FileAccess(const string &fn);
    
  template<typename T>
  ndarray::Array<T, 2, 1> get2DndArray(T nRows, T nCols);
    
  template<typename T>
  ndarray::Array<T, 1, 1> get1DndArray(T size);
  
  template<typename T>
  std::vector<T> copy(const std::vector<T> &vecIn);
  
  template< typename T > 
  std::string numberToString_dotToUnderscore( T number, int accuracy = -1 );
  
  std::string dotToUnderscore( std::string number, int accuracy = -1 );

  template< typename T >
  PTR( T ) getPointer( T & );

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
