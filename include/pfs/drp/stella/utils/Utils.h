#ifndef __PFS_DRP_STELLA_UTILS_H__
#define __PFS_DRP_STELLA_UTILS_H__
#include <vector>
#include <iostream>
#include "../blitz.h"
#include <fitsio.h>
#include <fitsio2.h>
#include "lsst/afw/image/MaskedImage.h"

namespace afwImage = lsst::afw::image;

using namespace std;
namespace pfs { namespace drp { namespace stella { namespace utils{

  /**
    *       Returns Position of <str_In> in Array of strings <S_A1_In>, if <S_A1_In> contains string <str_In>, else returns -1.
    **/
  int KeyWord_Set(const blitz::Array<string, 1> &S_A1_In,
                  const string &str_In);

  template<typename T>
  bool WriteFits(const blitz::Array<T,2>* image_In, const string &fileName_In);

  template<typename T>
  bool WriteFits(ndarray::Array<T, 2, 2> const& image_In, const string &fileName_In);

  template<typename T>
  bool WriteFits(const blitz::Array<T,1>* image_In, const string &fileName_In);

  /**
    *      task: Writes Array <I_A1_In> to file <CS_FileName_In>
    *      CS_Mode: [binary, ascii]
    **/
  template<typename T, int N>
  bool WriteArrayToFile(const blitz::Array<T, N> &I_A1_In,
                        const string &S_FileName_In,
                        const string &S_Mode);

  /**
            task: Writes Array <D_A2_In> to file <CS_FileName_In>
            CS_Mode: [binary, ascii]
    **/
//    template<typename T, int N>
//    bool WriteArrayToFile(const blitz::Array<T, 2> &D_A2_In,
//                          const string &S_FileName_In,
//                          const string &S_Mode);

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

  /// converts str to double if possible and returns true, otherwise returns false
  bool sToD(const string &str, double &D_Out);
  bool sToD(const blitz::Array<string, 1> &S_A1_In, blitz::Array<double, 1> &D_A1_Out);

  //  int sToI(const string &str);
  bool sToI(const string &str, int &I_Out);

  /**
    *       function int CountLines(const string &fnc: in)
    *       Returns number of lines of file <fnc>.
    **/
  long countLines(const string &fnc);

  /**
    *       function int CountDataLines(const string &fnc: in)
    *       Returns number of lines which do not start with '#' of file <fnc>.
    **/
  long countDataLines(const string &fnc);

  /**
    *      function int CountCols(const string &fnc: in, const string &delimiter: in)
    *      Returns number of columns of file <fnc>.
    **/
  long countCols(const string &fileName_In, const string &delimiter_In);

  bool FileAccess(const string &fn);

  bool readFileToStrArr(const string &S_FileName_In,
                        blitz::Array<string, 2> &S_A2_Out,
                        const string &S_Delimiter);

  bool readFileToDblArr(const string &S_FileName_In,
                        blitz::Array<double, 2> &D_A2_Out,
                        const string &S_Delimiter);

  bool readFileLinesToStrArr(const string &S_FileName_In,
                              blitz::Array<string, 1> &S_A1_Out);
    
  template<typename T>
  blitz::Array<T, 2> get2DBlitzArray(T nRows, T nCols);
    
  template<typename T>
  blitz::Array<T, 1> get1DBlitzArray(T size);
    
  template<typename T>
  ndarray::Array<T, 2, 2> get2DndArray(T nRows, T nCols);
    
  template<typename T>
  ndarray::Array<T, 1, 1> get1DndArray(T size);
  
  template<typename T>
  std::vector<T> copy(const std::vector<T> &vecIn);
  
}}}}
#endif
