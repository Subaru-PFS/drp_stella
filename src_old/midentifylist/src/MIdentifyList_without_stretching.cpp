/*
author: Andreas Ritter
created: 05/08/2012
last edited: 05/08/2012
compiler: g++ 4.4
basis machine: Arch Linux
*/

#include "MIdentifyList.h"

using namespace std;

int main(int argc, char *argv[])
{
  #ifdef __DEBUG_IDENTIFYLIST__
    cout << "MIdentifyList::main: argc = " << argc << endl;
  #endif
  if (argc < 10)
  {
    cout << "MIdentifyList::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: identify char[] FitsFileName_Ec_List_In, char[] FitsFileName_Ref_In, char[] TextFileName_EcD_List_Out, char[] TextFileName_Coeffs_List_Out, char[] LineList_In, int Radius_XCor_In, int Radius_GaussFit_In, double FWHM_In, int Order_In" << endl;
    exit(EXIT_FAILURE);
  }

  char *P_CharArr_Ec_List_In = (char*)argv[1];
//  #ifdef __DEBUG_IDENTIFYLIST__
    cout << "MIdentifyList::main: P_CharArr_Ec_List_In set to " << P_CharArr_Ec_List_In << endl;
//  #endif

  char *P_CharArr_Spec_Ref_In = (char*)argv[2];
//  #ifdef __DEBUG_IDENTIFYLIST__
    cout << "MIdentifyList::main: P_CharArr_Spec_Ref_In set to " << P_CharArr_Spec_Ref_In << endl;
//  #endif

  char *P_CharArr_Ec_List_Out = (char*)argv[3];
//  #ifdef __DEBUG_IDENTIFYLIST__
    cout << "MIdentifyList::main: P_CharArr_Ec_List_Out set to " << P_CharArr_Ec_List_Out << endl;
//  #endif

  char *P_CharArr_Coeffs_List_Out = (char*)argv[4];
//  #ifdef __DEBUG_IDENTIFYLIST__
    cout << "MIdentifyList::main: P_CharArr_Coeff_Out set to " << P_CharArr_Coeffs_List_Out << endl;
//  #endif

  char *P_CharArr_LineList_In = (char*)argv[5];
//  #ifdef __DEBUG_IDENTIFYLIST__
    cout << "MIdentifyList::main: P_CharArr_LineList_In set to " << P_CharArr_LineList_In << endl;
//  #endif

  char *P_CharArr_Radius_XCor_In = (char*)argv[6];
//  #ifdef __DEBUG_IDENTIFYLIST__
    cout << "MIdentifyList::main: P_CharArr_Radius_XCor_In set to " << P_CharArr_Radius_XCor_In << endl;
//  #endif

  char *P_CharArr_Radius_GaussFit_In = (char*)argv[7];
//  #ifdef __DEBUG_IDENTIFYLIST__
    cout << "MIdentifyList::main: P_CharArr_Radius_GaussFit_In set to " << P_CharArr_Radius_GaussFit_In << endl;
//  #endif

  char *P_CharArr_FWHM_In = (char*)argv[8];
//  #ifdef __DEBUG_IDENTIFYLIST__
    cout << "MIdentifyList::main: P_CharArr_FWHM_In set to " << P_CharArr_FWHM_In << endl;
//  #endif

  char *P_CharArr_Order_In = (char*)argv[9];
//  #ifdef __DEBUG_IDENTIFYLIST__
    cout << "MIdentifyList::main: P_CharArr_Order_In set to " << P_CharArr_Order_In << endl;
//  #endif

  /// read parameters
  CString CS_FitsFileName_Ec_List_In;
  CS_FitsFileName_Ec_List_In.Set(P_CharArr_Ec_List_In);
//  #ifdef __DEBUG_IDENTIFYLIST__
    cout << "MIdentifyList::main: CS_FitsFileName_Ec_List_In set to " << CS_FitsFileName_Ec_List_In << endl;
//  #endif

  ///  char *P_CharArr_Spec_Ref_In = (char*)argv[2];
  CString CS_FitsFileName_SpecRef_In;
  CS_FitsFileName_SpecRef_In.Set(P_CharArr_Spec_Ref_In);
//  #ifdef __DEBUG_IDENTIFYLIST__
    cout << "MIdentifyList::main: CS_FitsFileName_SpecRef_In set to " << CS_FitsFileName_SpecRef_In << endl;
//  #endif

  CString CS_TextFileName_Ec_List_Out;
  CS_TextFileName_Ec_List_Out.Set(P_CharArr_Ec_List_Out);
//  #ifdef __DEBUG_IDENTIFYLIST__
    cout << "MIdentifyList::main: CS_TextFileName_Ec_List_Out set to " << CS_TextFileName_Ec_List_Out << endl;
//  #endif

  CString CS_TextFileName_Coeffs_List_Out;
  CS_TextFileName_Coeffs_List_Out.Set(P_CharArr_Coeffs_List_Out);
//  #ifdef __DEBUG_IDENTIFYLIST__
    cout << "MIdentifyList::main: CS_TextFileName_Coeffs_List_Out set to " << CS_TextFileName_Coeffs_List_Out << endl;
//  #endif

  CString CS_LineList_In;
  CS_LineList_In.Set(P_CharArr_LineList_In);
//  #ifdef __DEBUG_IDENTIFYLIST__
    cout << "MIdentifyList::main: CS_LineList_In set to " << CS_LineList_In << endl;
//  #endif

  int I_Radius_XCor_In = atoi(P_CharArr_Radius_XCor_In);
//  #ifdef __DEBUG_IDENTIFYLIST__
    cout << "MIdentifyList::main: I_Radius_In set to " << I_Radius_XCor_In << endl;
//  #endif

  int I_Radius_GaussFit_In = atoi(P_CharArr_Radius_GaussFit_In);
//  #ifdef __DEBUG_IDENTIFYLIST__
    cout << "MIdentifyList::main: I_Radius_GaussFit_In set to " << I_Radius_GaussFit_In << endl;
//  #endif

  CString CS_FWHM_In;
  CS_FWHM_In.Set(P_CharArr_FWHM_In);
  double D_FWHM_In = CS_FWHM_In.AToD();
//  #ifdef __DEBUG_IDENTIFYLIST__
    cout << "MIdentifyList::main: D_FWHM_In set to " << D_FWHM_In << endl;
//  #endif

  int I_Order_In = atoi(P_CharArr_Order_In);
//  #ifdef __DEBUG_IDENTIFYLIST__
    cout << "MIdentifyList::main: I_Order_In set to " << I_Order_In << endl;
//  #endif

  CFits F_Image, F_ImageRef;

  /// Read FitsFileName_Ec_List_In
  Array<CString, 1> CS_A1_FitsFiles_Ec(2);
  if (!F_Image.ReadFileLinesToStrArr(CS_FitsFileName_Ec_List_In, CS_A1_FitsFiles_Ec)){
    cout << "MIdentifyList::main: ERROR: F_Image.ReadFileLinesToStrArr(" << CS_FitsFileName_Ec_List_In << ", CS_A1_FitsFiles_Ec) returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Read TextFileName_EcD_List_Out
  Array<CString, 1> CS_A1_TextFiles_Ec_Out(2);
  if (!F_Image.ReadFileLinesToStrArr(CS_TextFileName_Ec_List_Out, CS_A1_TextFiles_Ec_Out)){
    cout << "MIdentifyList::main: ERROR: F_Image.ReadFileLinesToStrArr(" << CS_TextFileName_Ec_List_Out << ", CS_A1_TextFiles_Ec_Out) returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Read Coeffs_List_Out
  Array<CString, 1> CS_A1_Coeffs_List_Out(2);
  if (!F_Image.ReadFileLinesToStrArr(CS_TextFileName_Coeffs_List_Out, CS_A1_Coeffs_List_Out)){
    cout << "MIdentifyList::main: ERROR: F_Image.ReadFileLinesToStrArr(" << CS_TextFileName_Coeffs_List_Out << ", CS_A1_TextFiles_Ec_Out) returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  if (CS_A1_FitsFiles_Ec.size() != CS_A1_TextFiles_Ec_Out.size()){
    cout << "MIdentifyList::main: ERROR: CS_A1_FitsFiles_Ec.size() != CS_A1_TextFiles_Ec_Out.size()" << endl;
    exit(EXIT_FAILURE);
  }

  if (CS_A1_FitsFiles_Ec.size() != CS_A1_Coeffs_List_Out.size()){
    cout << "MIdentifyList::main: ERROR: CS_A1_FitsFiles_Ec.size() != CS_A1_Coeffs_List_Out.size()" << endl;
    exit(EXIT_FAILURE);
  }

  /// Read LineList_In
  Array<CString, 2> CS_A2_LineList(2,2);
  if (!F_Image.ReadFileToStrArr(CS_LineList_In, CS_A2_LineList, CString(" "))){
    cout << "MIdentifyList::main: ERROR: F_Image.ReadFileToStrArr(" << CS_LineList_In << ", CS_A2_LineList, CString(' ')) returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Convert LineList_In to Array<double, 2>
  Array<double, 2> D_A2_LineList(CS_A2_LineList.rows(), CS_A2_LineList.cols());
  for (int i_row=0; i_row<CS_A2_LineList.rows(); i_row++){
    for (int i_col=0; i_col<CS_A2_LineList.cols(); i_col++){
      D_A2_LineList(i_row, i_col) = double((CS_A2_LineList(i_row, i_col)).AToD());
    }
  }
  #ifdef __DEBUG_IDENTIFYLIST__
    cout << "MIdentifyList::main: D_A2_LineList = " << D_A2_LineList << endl;
  #endif

  /// Read reference spectrum
  if (!F_ImageRef.SetFileName(CS_FitsFileName_SpecRef_In)){
    cout << "MIdentifyList::main: ERROR: F_ImageRef.SetFileName(" << CS_FitsFileName_SpecRef_In << ") returned FALSE" << endl;
    exit(EXIT_FAILURE);
  }
  if (!F_ImageRef.ReadArray()){
    cout << "MIdentifyList::main: ERROR: F_ImageRef.ReadArray() returned FALSE" << endl;
    exit(EXIT_FAILURE);
  }

  Array<double, 1> D_A1_Spec(1);
  Array<double, 1> D_A1_SpecRef(1);
  Array<double, 1> D_A1_SpecTemp(1);
  Array<double, 1> D_A1_SpecRefTemp(1);
  Array<double, 2> D_A2_SpecCalib_Out(2,2);
  Array<double, 1> D_A1_PolyFitCoeffs_Out(I_Order_In+1);
  Array<double, 2> D_A2_LineList_Temp(D_A2_LineList.rows(),D_A2_LineList.cols());
  Array<double, 2> D_A2_LineList_Ident(D_A2_LineList.rows(),D_A2_LineList.cols());
  double D_RMS_Out;
  int I_PixShift, I_LinePos, i_line_temp, i_nlines;
  for (int i_file=0; i_file < CS_A1_FitsFiles_Ec.size(); i_file++){
    i_line_temp = 0;
    #ifdef __DEBUG_IDENTIFYLIST__
      cout << "MIdentifyList::main: Starting F_Image.SetFileName(" << (CS_A1_FitsFiles_Ec(i_file)).Get() << ")" << endl;
    #endif
    if (!F_Image.SetFileName((CS_A1_FitsFiles_Ec)(i_file)))
    {
      cout << "MIdentifyList::main: ERROR: F_Image.SetFileName(" << (CS_A1_FitsFiles_Ec(i_file)).Get() << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    /// Read FitsFile
    #ifdef __DEBUG_IDENTIFYLIST__
      cout << "MIdentifyList::main: Starting F_Image.ReadArray()" << endl;
    #endif
    if (!F_Image.ReadArray())
    {
      cout << "MIdentifyList::main: ERROR: F_Image.ReadArray() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    /// Write aperture header information
    F_Image.WriteApHead(CString("aphead_")+CS_A1_FitsFiles_Ec(i_file)+CString(".head"));
    #ifdef __DEBUG_IDENTIFYLIST__
      cout << "MIdentifyList::main: F_Image.GetNRows() = " << F_Image.GetNRows() << endl;
      cout << "MIdentifyList::main: F_Image.GetNCols() = " << F_Image.GetNCols() << endl;
    #endif

    /// Read F_Image.PixArray to Array<double, 1>
    if (F_Image.GetNRows() == 1){
      D_A1_Spec.resize(F_Image.GetNCols());
      D_A1_Spec = (F_Image.GetPixArray())(0,Range::all());
    }
    else if (F_Image.GetNCols() == 1){
      D_A1_Spec.resize(F_Image.GetNRows());
      D_A1_Spec = (F_Image.GetPixArray())(Range::all(),0);
    }
    else{
      cout << "MIdentifyList::main: ERROR: Input spectrum not one-dimensional" << endl;
      exit(EXIT_FAILURE);
    }

    /// Read F_ImageRef.PixArray to Array<double, 1>
    if (F_ImageRef.GetNRows() == 1){
      D_A1_SpecRef.resize(F_ImageRef.GetNCols());
      D_A1_SpecRef = (F_ImageRef.GetPixArray())(0,Range::all());
    }
    else if (F_ImageRef.GetNCols() == 1){
      D_A1_SpecRef.resize(F_ImageRef.GetNRows());
      D_A1_SpecRef = (F_ImageRef.GetPixArray())(Range::all(),0);
    }
    else{
      cout << "MIdentifyList::main: ERROR: Input spectrum not one-dimensional" << endl;
      exit(EXIT_FAILURE);
    }

    /// Cross-correlate D_A1_Spec to reference spectrum
    D_A1_SpecTemp.resize(D_A1_Spec.size());
    D_A1_SpecTemp = D_A1_Spec;
    D_A1_SpecRefTemp.resize(D_A1_SpecRef.size());
    D_A1_SpecRefTemp = D_A1_SpecRef;
    if (D_A1_SpecTemp.size() < D_A1_SpecRefTemp.size())
      D_A1_SpecRefTemp.resizeAndPreserve(D_A1_SpecTemp.size());
    if (D_A1_SpecRefTemp.size() < D_A1_SpecTemp.size())
      D_A1_SpecTemp.resizeAndPreserve(D_A1_SpecRefTemp.size());

    if (!F_ImageRef.CrossCorrelate(D_A1_SpecRefTemp, D_A1_SpecTemp, I_Radius_XCor_In, I_Radius_XCor_In, I_PixShift)){
      cout << "MIdentifyList::main: ERROR: CrossCorrelate returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
//    #ifdef __DEBUG_IDENTIFYLIST__
      cout << "MIdentifyList::main: I_PixShift = " << I_PixShift << endl;
//    #endif

    i_line_temp = 0;
    for (int i_line=0; i_line < D_A2_LineList.rows(); i_line++){
      I_LinePos = D_A2_LineList(i_line,1) - I_PixShift - 1;
      //    #ifdef __DEBUG_IDENTIFYLIST__
        cout << "MIdentifyList::main: i_file=" << i_file << ": I_LinePos = " << I_LinePos << endl;
        cout << "MIdentifyList::main: i_file=" << i_file << ": D_A1_Spec.size() = " << D_A1_Spec.size() << endl;
        cout << "MIdentifyList::main: i_file=" << i_file << ": D_A1_SpecRef.size() = " << D_A1_SpecRef.size() << endl;
        //    #endif

      if (I_LinePos < 0 || I_LinePos >= D_A1_Spec.size()){
//        #ifdef __DEBUG_IDENTIFYLIST__
          cout << "MIdentifyList::main: i_file=" << i_file << ": WARNING: calibration line at " << D_A2_LineList(i_line,0) << " at position " << D_A2_LineList(i_line,1) <<  " outside range" << endl;
          cout << "MIdentifyList::main: i_file=" << i_file << ": D_A2_LineList = " << D_A2_LineList << endl;
 //       #endif
//        D_A2_LineList_Temp.resizeAndPreserve(D_A2_LineList_Temp.rows()-1, 2);
//        cout << "MIdentifyList::main: i_file=" << i_file << ": D_A2_LineList_Temp resized to " << D_A2_LineList_Temp.rows() << endl;
      }
      else{
        D_A2_LineList_Temp(i_line_temp, 0) = D_A2_LineList(i_line, 0);
        D_A2_LineList_Temp(i_line_temp, 1) = I_LinePos;
        i_line_temp++;
      }
    }
    D_A2_LineList_Ident.resize(i_line_temp,2);
    D_A2_LineList_Ident = D_A2_LineList_Temp(Range(0,i_line_temp-1),Range::all());
//    #ifdef __DEBUG_IDENTIFYLIST__
      cout << "MIdentifyList::main: i_file=" << i_file << ": D_A2_LineList_Ident = " << D_A2_LineList_Ident << endl;
//    #endif

    /// Identify lines
    #ifdef __DEBUG_IDENTIFYLIST__
      cout << "MIdentifyList::main: i_file=" << i_file << ": Starting F_Image.Identify: D_A1_Spec = " << D_A1_Spec << endl;
    #endif
    if (D_A1_Spec.size() > D_A1_SpecRef.size()/2.){
      if (!F_Image.Identify(D_A1_Spec,
                            D_A2_LineList_Ident,
                            I_Radius_GaussFit_In,
                            D_FWHM_In,
                            I_Order_In,
                            D_A2_SpecCalib_Out,
                            D_A1_PolyFitCoeffs_Out,
                            D_RMS_Out)){
        cout << "MIdentifyList::main: i_file=" << i_file << ": WARNING: Identify returned FALSE" << endl;
//      exit(EXIT_FAILURE);
      }
      else{
        #ifdef __DEBUG_IDENTIFYLIST__
          cout << "MIdentifyList::main: i_file=" << i_file << ": D_A2_SpecCalib_Out = " << D_A2_SpecCalib_Out << endl;
        #endif
        ///write calibrated spectrum as text file
        if (!F_Image.WriteArrayToFile(D_A2_SpecCalib_Out, CS_A1_TextFiles_Ec_Out(i_file), CString("ascii"))){
          cout << "MIdentifyList::main: i_file=" << i_file << ": ERROR: WriteArrayToFile(spec) returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }

        ///write polynomial fitting coefficients to file
        if (!F_Image.WriteArrayToFile(D_A1_PolyFitCoeffs_Out, CS_A1_Coeffs_List_Out(i_file), CString("ascii"))){
          cout << "MIdentifyList::main: i_file=" << i_file << ": ERROR: WriteArrayToFile(coeffs) returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }

        /// Append RMS to coeffs file
        FILE *p_file;
        p_file = fopen((CS_A1_Coeffs_List_Out(i_file)).Get(), "a");
        fprintf(p_file, "RMS %.7f\n", D_RMS_Out);
        fclose(p_file);
      }
    }
  }

  /// clean up

  return EXIT_SUCCESS;
}

