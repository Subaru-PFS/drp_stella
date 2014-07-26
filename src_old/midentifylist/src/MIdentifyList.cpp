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
  /// stretches and cross-correlates input spectra to match FitsFileName_Ref_In, then tries to identify the emission lines
  #ifdef __DEBUG_IDENTIFYLIST__
    cout << "MIdentifyList::main: argc = " << argc << endl;
  #endif

//  Py_Initialize();
//  PyRun_SimpleString("import pylab");
//  PyRun_SimpleString("pylab.plot(range(5))");
//  PyRun_SimpleString("pylab.show()");
//  Py_Exit(0);
//  exit(EXIT_FAILURE);

/**
 * bool CFits::StretchAndCrossCorrelateSpec(const Array<double, 1> &D_A1_Spec_In,
 const Array<double, 1> &D_A1_SpecRef_In,
 const Array<double, 2> &D_A2_LineList_WLenPix_In,
 const int I_Radius_XCor_In,
 const int I_Stretch_Min_Length_In,
 const int I_Stretch_Max_Length_In,
 const int I_NStretches_In,
 const int I_LengthPieces_In,
 const int I_NCalcs_In,
 const int I_PolyFitOrder_Stretch_In,
 const int I_PolyFitOrder_Shift_In,
 Array<double, 2> &D_A2_LineList_WLenPix_Out) const{
   **/

  if (argc < 16)
  {
    cout << "MIdentifyList::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: identifylist <char[] FitsFileName_Ec_List_In> <char[] FitsFileName_Ref_In> <char[] TextFileName_EcD_List_Out> <char[] TextFileName_Coeffs_List_Out> <char[] LineList_In> <int Radius_XCor_In> <int Stretch_Min_Length_In> <int Stretch_Max_Length_In> <int N_Stretches_In> <int I_LengthPieces_In> <int I_NCalcs_In> <int I_PolyFitOrder_Stretch_In> <int I_PolyFitOrder_Shift_In> <int Radius_GaussFit_In> <double FWHM_In> <int Order_In>" << endl;
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

//  char *P_CharArr_Coeffs_Ref_In = (char*)argv[4];
//  #ifdef __DEBUG_IDENTIFYLIST__
//    cout << "MIdentifyList::main: P_CharArr_Coeffs_Ref_In set to " << P_CharArr_Coeffs_Ref_In << endl;
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

  char *P_CharArr_Stretch_Min_Length_In = (char*)argv[7];
//  #ifdef __DEBUG_IDENTIFYLIST__
    cout << "MIdentifyList::main: P_CharArr_Stretch_Min_Length_In set to " << P_CharArr_Stretch_Min_Length_In << endl;
//  #endif

  char *P_CharArr_Stretch_Max_Length_In = (char*)argv[8];
//  #ifdef __DEBUG_IDENTIFYLIST__
    cout << "MIdentifyList::main: P_CharArr_Stretch_Max_Length_In set to " << P_CharArr_Stretch_Max_Length_In << endl;
//  #endif

  char *P_CharArr_N_Stretches_In = (char*)argv[9];
//  #ifdef __DEBUG_IDENTIFYLIST__
    cout << "MIdentifyList::main: P_CharArr_N_Stretches_In set to " << P_CharArr_N_Stretches_In << endl;
//  #endif

  char *P_CharArr_I_LengthPieces_In = (char*)argv[10];
  //  #ifdef __DEBUG_IDENTIFYLIST__
  cout << "MIdentifyList::main: P_CharArr_I_LengthPieces_In set to " << P_CharArr_I_LengthPieces_In << endl;
  //  #endif

  char *P_CharArr_I_NCalcs_In = (char*)argv[11];
  //  #ifdef __DEBUG_IDENTIFYLIST__
  cout << "MIdentifyList::main: P_CharArr_I_NCalcs_In set to " << P_CharArr_I_NCalcs_In<< endl;
  //  #endif

  char *P_CharArr_I_PolyFitOrder_Stretches_In = (char*)argv[12];
  //  #ifdef __DEBUG_IDENTIFYLIST__
  cout << "MIdentifyList::main: P_CharArr_I_PolyFitOrder_Stretches_In set to " << P_CharArr_I_PolyFitOrder_Stretches_In << endl;
  //  #endif

  char *P_CharArr_I_PolyFitOrder_Shifts_In = (char*)argv[13];
  //  #ifdef __DEBUG_IDENTIFYLIST__
  cout << "MIdentifyList::main: P_CharArr_I_PolyFitOrder_Shifts_In set to " << P_CharArr_I_PolyFitOrder_Shifts_In << endl;
  //  #endif

  char *P_CharArr_Radius_GaussFit_In = (char*)argv[14];
//  #ifdef __DEBUG_IDENTIFYLIST__
    cout << "MIdentifyList::main: P_CharArr_Radius_GaussFit_In set to " << P_CharArr_Radius_GaussFit_In << endl;
//  #endif

  char *P_CharArr_FWHM_In = (char*)argv[15];
//  #ifdef __DEBUG_IDENTIFYLIST__
    cout << "MIdentifyList::main: P_CharArr_FWHM_In set to " << P_CharArr_FWHM_In << endl;
//  #endif

  char *P_CharArr_Order_In = (char*)argv[16];
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

//  CString CS_TextFileName_Coeffs_Ref_In;
//  CS_TextFileName_Coeffs_Ref_In.Set(P_CharArr_Coeffs_Ref_In);
//  #ifdef __DEBUG_IDENTIFYLIST__
//    cout << "MIdentifyList::main: CS_TextFileName_Coeffs_Ref_In set to " << CS_TextFileName_Coeffs_Ref_In << endl;
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

  int I_Stretch_Min_Length_In = atoi(P_CharArr_Stretch_Min_Length_In);
//  #ifdef __DEBUG_IDENTIFYLIST__
    cout << "MIdentifyList::main: I_Stretch_Min_Length_In set to " << I_Stretch_Min_Length_In << endl;
//  #endif

  int I_Stretch_Max_Length_In = atoi(P_CharArr_Stretch_Max_Length_In);
//  #ifdef __DEBUG_IDENTIFYLIST__
    cout << "MIdentifyList::main: I_Stretch_Max_Length_In set to " << I_Stretch_Max_Length_In << endl;
//  #endif

  int I_N_Stretches_In = atoi(P_CharArr_N_Stretches_In);
//  #ifdef __DEBUG_IDENTIFYLIST__
    cout << "MIdentifyList::main: I_N_Stretches_In set to " << I_N_Stretches_In << endl;
//  #endif
/**
 *
 *  char *P_CharArr_I_LengthPieces_In = (char*)argv[10];
 *  //  #ifdef __DEBUG_IDENTIFYLIST__
 *  cout << "MIdentifyList::main: P_CharArr_I_LengthPieces_In set to " << P_CharArr_I_LengthPieces_In << endl;
 *  //  #endif
 *
 *  char *P_CharArr_I_NCalcs_In = (char*)argv[11];
 *  //  #ifdef __DEBUG_IDENTIFYLIST__
 *  cout << "MIdentifyList::main: P_CharArr_I_NCalcs_In set to " << P_CharArr_I_NCalcs_In<< endl;
 *  //  #endif
 *
 *  char *P_CharArr_I_PolyFitOrder_Stretches_In = (char*)argv[12];
 *  //  #ifdef __DEBUG_IDENTIFYLIST__
 *  cout << "MIdentifyList::main: P_CharArr_I_PolyFitOrder_Stretches_In set to " << P_CharArr_I_PolyFitOrder_Stretches_In << endl;
 *  //  #endif
 *
 *  char *P_CharArr_I_PolyFitOrder_Shifts_In = (char*)argv[13];
 *  //  #ifdef __DEBUG_IDENTIFYLIST__
 *  cout << "MIdentifyList::main: P_CharArr_I_PolyFitOrder_Shifts_In set to " << P_CharArr_I_PolyFitOrder_Shifts_In << endl;
 *  //  #endif
 **/
  int I_LengthPieces_In = atoi(P_CharArr_I_LengthPieces_In);
  //  #ifdef __DEBUG_IDENTIFYLIST__
  cout << "MIdentifyList::main: I_LengthPieces_In set to " << I_LengthPieces_In << endl;
  //  #endif

  int I_NCalcs_In = atoi(P_CharArr_I_NCalcs_In);
  //  #ifdef __DEBUG_IDENTIFYLIST__
  cout << "MIdentifyList::main: I_NCalcs_In set to " << I_NCalcs_In << endl;
  //  #endif

  int I_PolyFitOrder_Stretch_In = atoi(P_CharArr_I_PolyFitOrder_Stretches_In);
  //  #ifdef __DEBUG_IDENTIFYLIST__
  cout << "MIdentifyList::main: I_PolyFitOrder_Stretch_In set to " << I_PolyFitOrder_Stretch_In << endl;
  //  #endif

  int I_PolyFitOrder_Shift_In = atoi(P_CharArr_I_PolyFitOrder_Shifts_In);
  //  #ifdef __DEBUG_IDENTIFYLIST__
  cout << "MIdentifyList::main: I_PolyFitOrder_Shift_In set to " << I_PolyFitOrder_Shift_In << endl;
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

  /// Read Coeffs_Ref_In
//  Array<CString, 1> CS_A1_Coeffs_Ref_In(1);
//  if (!F_Image.ReadFileLinesToStrArr(CS_TextFileName_Coeffs_Ref_In, CS_A1_Coeffs_Ref_In)){
//    cout << "MIdentifyList::main: ERROR: F_Image.ReadFileLinesToStrArr(" << CS_TextFileName_Coeffs_Ref_In << ", CS_A1_Coeffs_Ref_In) returned FALSE!" << endl;
//    exit(EXIT_FAILURE);
//  }
//  Array<double, 1> D_A1_Coeffs_Ref_In(CS_A1_Coeffs_Ref_In.size()-1);
//  Array<CString, 1> CS_A1_Coeffs(CS_A1_Coeffs_Ref_In.size()-1);
//  CS_A1_Coeffs = CS_A1_Coeffs_Ref_In(Range(0,CS_A1_Coeffs_Ref_In.size()-2));
//  cout << "CS_A1_Coeffs = " << CS_A1_Coeffs << endl;
//  if (!CS_FitsFileName_Ec_List_In.AToD(CS_A1_Coeffs, D_A1_Coeffs_Ref_In)){
//    cout << "MIdentifyList::main: ERROR: AToD(CS_A1_Coeffs_Ref_In=" << CS_A1_Coeffs_Ref_In(Range(0,CS_A1_Coeffs_Ref_In.size()-2)) << ", D_A1_Coeffs_Ref_In) returned FALSE" << endl;
//    exit(EXIT_FAILURE);
//  }

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
  Array<double, 2> D_A2_LineList(CS_A2_LineList.rows(), 3);
  for (int i_row=0; i_row<CS_A2_LineList.rows(); i_row++){
    for (int i_col=0; i_col<2; i_col++){
      D_A2_LineList(i_row, i_col) = double((CS_A2_LineList(i_row, i_col)).AToD());
    }
  }
  Array<double, 2> D_A2_LineList_Stretched(D_A2_LineList.rows(), D_A2_LineList.cols());
  D_A2_LineList_Stretched = 0.;
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
  D_A1_Spec = 0.;
  Array<double, 1> D_A1_SpecRef(1);
  D_A1_SpecRef = 0.;
  Array<double, 1> D_A1_SpecTemp(1);
  D_A1_SpecTemp = 0.;
  Array<double, 1> D_A1_SpecRefTemp(1);
  D_A1_SpecRefTemp = 0.;
  Array<double, 2> D_A2_SpecCalib_Out(2,2);
  D_A2_SpecCalib_Out = 0.;
  Array<double, 1> D_A1_PolyFitCoeffs_Out(I_Order_In+1);
  D_A1_PolyFitCoeffs_Out = 0.;
  Array<double, 2> D_A2_LineList_Temp(D_A2_LineList.rows(),D_A2_LineList.cols());
  D_A2_LineList_Temp = 0.;
  Array<double, 2> D_A2_LineList_Ident(D_A2_LineList.rows(),D_A2_LineList.cols());
  D_A2_LineList_Ident = 0.;
  double D_RMS_Out = 0.;
//  int I_PixShift = 0;
  int I_LinePos = 0;
  int i_line_temp = 0;
  int i_nlines = 0;
  for (int i_file=0; i_file < CS_A1_FitsFiles_Ec.size(); i_file++){
    CString *P_CS_Sub = CS_A1_FitsFiles_Ec(i_file).SubString(0,CS_A1_FitsFiles_Ec(i_file).LastCharPos('/'));
    cout << "MIdentifyList::main: P_CS_Sub = " << *P_CS_Sub << endl;
    CString CS_Path(*P_CS_Sub);
    if (!CS_Path.MkDir(CS_Path)){
        cout << "MIdentifyList::main: ERROR: MkDir(" << CS_Path << ") returned FALSE" << endl;
        exit(EXIT_FAILURE);
    }
    delete(P_CS_Sub);
    P_CS_Sub = CS_A1_FitsFiles_Ec(i_file).SubString(0,CS_A1_FitsFiles_Ec(i_file).LastCharPos('.')-1);
    CString CS_FNamePlotRoot(*P_CS_Sub);
    cout << "MIdentifyList::main: i_file=" << i_file << ": CS_FNamePlotRoot = " << CS_FNamePlotRoot << endl;
    delete(P_CS_Sub);
//    exit(EXIT_FAILURE);
    i_line_temp = 0;
    cout << "MIdentifyList::main: Starting F_Image.SetFileName(" << (CS_A1_FitsFiles_Ec(i_file)).Get() << ")" << endl;
    if (!F_Image.SetFileName(CS_A1_FitsFiles_Ec(i_file)))
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
    
    /// set line intensities
    for (int i_row=0; i_row < D_A2_LineList.rows(); i_row++){
      D_A2_LineList(i_row, 2) = D_A1_SpecRef(D_A2_LineList(i_row, 1));
    }

    /// Stretch Reference Spectrum
    Array<double, 2> D_A2_LineList_WLenPix_Out(2,2);
    D_A2_LineList_WLenPix_Out = 0.;
    D_A1_Spec = D_A1_Spec * mean(D_A1_SpecRef) / mean(D_A1_Spec);
    if (!F_Image.StretchAndCrossCorrelateSpec(D_A1_Spec,
                                              D_A1_SpecRef,
                                              D_A2_LineList,
                                              I_Radius_XCor_In,
                                              I_Stretch_Min_Length_In,
                                              I_Stretch_Max_Length_In,
                                              I_N_Stretches_In,
                                              I_LengthPieces_In,
                                              I_NCalcs_In,
                                              I_PolyFitOrder_Stretch_In,
                                              I_PolyFitOrder_Shift_In,
                                              CS_FNamePlotRoot,
                                              D_A2_LineList_WLenPix_Out)){
      cout << "MIdentifyList::main: WARNING: StretchAndCrossCorrelateSpec() returned FALSE" << endl;
//      exit(EXIT_FAILURE);
    }
    else{

      cout << "MIdentifyList::main: StretchAndCrossCorrelateSpec returned TRUE" << endl;
      cout << "MIdentifyList::main: D_A2_LineList_WLenPix_Out = " << D_A2_LineList_WLenPix_Out << endl;

      #ifdef __DEBUG_IDENTIFYLIST__
        cout << "MIdentifyList::main: i_file=" << i_file << ": Starting F_Image.Identify: D_A1_Spec = " << D_A1_Spec << endl;
      #endif
      Array<double, 2> D_A2_PixWLen(1,1);
      if (D_A1_Spec.size() > D_A1_SpecRef.size()/2.){
        if (!F_Image.Identify(D_A1_Spec,
                              D_A2_LineList_WLenPix_Out,
                              I_Radius_GaussFit_In,
                              D_FWHM_In,
                              I_Order_In,
                              CS_FNamePlotRoot,
                              D_A2_SpecCalib_Out,
                              D_A1_PolyFitCoeffs_Out,
                              D_RMS_Out,
			      D_A2_PixWLen)){
          cout << "MIdentifyList::main: i_file=" << i_file << ": WARNING: Identify returned FALSE" << endl;
//      exit(EXIT_FAILURE);
        }
        else{
//          cout << "MIdentifyList::main: i_file=" << i_file << ": D_A2_LineList = " << D_A2_LineList << endl;
//          cout << "MIdentifyList::main: i_file=" << i_file << ": D_A2_PixWLen = " << D_A2_PixWLen << endl;
//          exit(EXIT_FAILURE);
          if (D_A2_PixWLen.rows() < D_A2_LineList.rows()){
            cout << "MIdentifyList::main: i_file=" << i_file << ": Not all lines identified" << endl;
            cout << "MIdentifyList::main: i_file=" << i_file << ": D_A2_LineList = " << D_A2_LineList << endl;
            cout << "MIdentifyList::main: i_file=" << i_file << ": D_A2_PixWLen = " << D_A2_PixWLen << endl;
//            exit(EXIT_FAILURE);
          }
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

          CString *P_CS_PixWLenFit = CS_A1_Coeffs_List_Out(i_file).SubString(0,CS_A1_Coeffs_List_Out(i_file).LastCharPos('.')-1);
          P_CS_PixWLenFit->Add(CString("_PixWLenFit.dat"));
          if (!F_Image.WriteArrayToFile(D_A2_PixWLen, *P_CS_PixWLenFit, CString("ascii"))){
            cout << "MIdentifyList::main: i_file=" << i_file << ": ERROR: WriteArrayToFile(PixWLenFit) returned FALSE" << endl;
            exit(EXIT_FAILURE);
          }
          delete(P_CS_PixWLenFit);

          /// Append RMS to coeffs file
          CString *P_CS_ApNum = CS_A1_Coeffs_List_Out(i_file).SubString(CS_A1_Coeffs_List_Out(i_file).LastStrPos(CString("_ap"))+3,CS_A1_Coeffs_List_Out(i_file).LastStrPos(CString("_x"))-1);
          FILE *p_file;
          p_file = fopen((CS_A1_Coeffs_List_Out(i_file)).Get(), "a");
          fprintf(p_file, "RMS %.7f Ap %s\n", D_RMS_Out, P_CS_ApNum->Get());
          fclose(p_file);
	  delete(P_CS_ApNum);
        }
        if (D_RMS_Out > 500.){
          cout << "MIdentifyList::main: Warning: RMS(=" << D_RMS_Out << " > 500" << endl;
        }
      }
    }
//    else{
//    }
//    delete(P_D_A1_Ref_Y);
//    exit(EXIT_FAILURE);
  }

  /// clean up

  //Py_Exit(0);
  return EXIT_SUCCESS;
}

