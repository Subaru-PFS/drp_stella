#ifndef __PFS_DRP_STELLA_UTILS_H__
#define __PFS_DRP_STELLA_UTILS_H__
#include <vector>
#include <iostream>
//#include "../blitz.h"
#include <fitsio.h>
#include <fitsio2.h>
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/image.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/fits.h"

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

  template< typename PixelT, int C >
  inline void fits_write_ndarray( lsst::afw::fits::Fits & fitsfile,
                                  ndarray::Array< PixelT, 2, C > const& array,
                                  CONST_PTR(lsst::daf::base::PropertySet) metadata_i){
    ndarray::Array< PixelT const, 2, ( C > 2 ? 2 : C) > tempArr(array);
    cout << "writing ndarray" << endl;
    PTR(lsst::daf::base::PropertySet) metadata;
    PTR(lsst::daf::base::PropertySet) wcsAMetadata = lsst::afw::image::detail::createTrivialWcsAsPropertySet( lsst::afw::image::detail::wcsNameForXY0, 0, 0);
    if ( metadata_i ) {
        metadata = metadata_i->deepCopy();
        metadata->combine( wcsAMetadata );
    } else {
        metadata = wcsAMetadata;
    }
    fitsfile.createImage< PixelT >( array.getShape() );
    if ( metadata ) {
        fitsfile.writeMetadata( *metadata );
    }
    fitsfile.writeImage( tempArr );
  }
  
  template< typename PixelT, int C >
  inline void fits_write_ndarray( lsst::afw::fits::Fits & fitsfile,
                                  ndarray::Array< PixelT, 3, C > const& array,
                                  CONST_PTR(lsst::daf::base::PropertySet) metadata_i){
    ndarray::Array< PixelT, 2, C > arr;
    arr = ndarray::allocate( array.getShape()[ 0 ] * array.getShape()[ 1 ], array.getShape()[ 2 ] );
    int iRow = 0;
    for ( int i = 0; i < array.getShape()[ 0 ]; ++i ){
      for ( int j = 0; j < array.getShape()[ 1 ]; ++j, ++iRow ){
        arr[ ndarray::view( iRow )() ] = array[ ndarray::view( i )( j )() ];
      }
    }
    fits_write_ndarray( fitsfile, arr, metadata_i );
  }
  
  template< typename T >
  PTR( T ) getPointer( T & );
}}}}
#endif
