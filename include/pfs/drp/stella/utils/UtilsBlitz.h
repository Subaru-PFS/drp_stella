#ifndef __PFS_DRP_STELLA_UTILSBLITZ_H__
#define __PFS_DRP_STELLA_UTILSBLITZ_H__
#include <vector>
#include <iostream>
#include "../blitz.h"
#include <fitsio.h>
#include <fitsio2.h>
#include "lsst/afw/image/MaskedImage.h"
#include "Utils.h"

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
  bool WriteFits(const blitz::Array<T,1>* image_In, const string &fileName_In);

  /**
    *      task: Writes Array <I_A1_In> to file <CS_FileName_In>
    *      CS_Mode: [binary, ascii]
    **/
  template<typename T, int N>
  bool WriteArrayToFile(const blitz::Array<T, N> &I_A1_In,
                        const string &S_FileName_In,
                        const string &S_Mode);

  /// converts str to double if possible and returns true, otherwise returns false
  bool sToD(const string &str, double &D_Out);
  bool sToD(const blitz::Array<string, 1> &S_A1_In, blitz::Array<double, 1> &D_A1_Out);

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
}}}}
#endif
