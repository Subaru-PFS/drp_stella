/*
author: Andreas Ritter
created: 05/08/2012
last edited: 05/08/2012
compiler: g++ 4.4
basis machine: Arch Linux
*/

#include "MIdentify.h"

using namespace std;

int main(int argc, char *argv[])
{
  cout << "MIdentify::main: argc = " << argc << endl;
  if (argc < 8)
  {
    cout << "MIdentify::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: identify char[] FitsFileName_Ec_In, char[] TextFileName_EcD_Out, char[] TextFileName_Coeffs_Out, char[] LineList_In, int Radius_In, double FWHM_In, int Order_In" << endl;
    exit(EXIT_FAILURE);
  }

  char *P_CharArr_Ec_In = (char*)argv[1];
  cout << "MIdentify::main: P_CharArr_Ec_In set to " << P_CharArr_Ec_In << endl;

  char *P_CharArr_Ec_Out = (char*)argv[2];
  cout << "MIdentify::main: P_CharArr_Ec_Out set to " << P_CharArr_Ec_Out << endl;

  char *P_CharArr_Coeffs_Out = (char*)argv[3];
  cout << "MIdentify::main: P_CharArr_Coeff_Out set to " << P_CharArr_Coeffs_Out << endl;

  char *P_CharArr_LineList_In = (char*)argv[4];
  cout << "MIdentify::main: P_CharArr_LineList_In set to " << P_CharArr_LineList_In << endl;

  char *P_CharArr_Radius_In = (char*)argv[5];
  cout << "MIdentify::main: P_CharArr_Radius_In set to " << P_CharArr_Radius_In << endl;

  char *P_CharArr_FWHM_In = (char*)argv[6];
  cout << "MIdentify::main: P_CharArr_FWHM_In set to " << P_CharArr_FWHM_In << endl;

  char *P_CharArr_Order_In = (char*)argv[7];
  cout << "MIdentify::main: P_CharArr_Order_In set to " << P_CharArr_Order_In << endl;

  /// read parameters
  CString CS_FitsFileName_Ec_In;
  CS_FitsFileName_Ec_In.Set(P_CharArr_Ec_In);
  cout << "MIdentify::main: CS_FitsFileName_Ec_In set to " << CS_FitsFileName_Ec_In << endl;

  CString CS_TextFileName_Ec_Out;
  CS_TextFileName_Ec_Out.Set(P_CharArr_Ec_Out);
  cout << "MIdentify::main: CS_TextFileName_Ec_Out set to " << CS_TextFileName_Ec_Out << endl;

  CString CS_TextFileName_Coeffs_Out;
  CS_TextFileName_Coeffs_Out.Set(P_CharArr_Coeffs_Out);
  cout << "MIdentify::main: CS_TextFileName_Coeffs_Out set to " << CS_TextFileName_Coeffs_Out << endl;

  CString CS_LineList_In;
  CS_LineList_In.Set(P_CharArr_LineList_In);
  cout << "MIdentify::main: CS_LineList_In set to " << CS_LineList_In << endl;

  int I_Radius_In = atoi(P_CharArr_Radius_In);
  cout << "MIdentify::main: I_Radius_In set to " << I_Radius_In << endl;

  CString CS_FWHM_In;
  CS_FWHM_In.Set(P_CharArr_FWHM_In);
  double D_FWHM_In = CS_FWHM_In.AToD();
  cout << "MIdentify::main: D_FWHM_In set to " << D_FWHM_In << endl;

  int I_Order_In = atoi(P_CharArr_Order_In);
  cout << "MIdentify::main: I_Order_In set to " << I_Order_In << endl;

  CFits F_Image;
  cout << "MIdentify::main: Starting F_Image.SetFileName(" << CS_FitsFileName_Ec_In.Get() << ")" << endl;
  if (!F_Image.SetFileName(CS_FitsFileName_Ec_In))
  {
    cout << "MIdentify::main: ERROR: F_Image.SetFileName(" << CS_FitsFileName_Ec_In.Get() << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Read FitsFile
  cout << "MIdentify::main: Starting F_Image.ReadArray()" << endl;
  if (!F_Image.ReadArray())
  {
    cout << "MIdentify::main: ERROR: F_Image.ReadArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Write aperture header information
  F_Image.WriteApHead(CString("aphead_")+CS_FitsFileName_Ec_In+CString(".head"));

  /// Read LineList_In
  Array<CString, 2> CS_A2_LineList(2,2);
  if (!F_Image.ReadFileToStrArr(CS_LineList_In, CS_A2_LineList, CString(" "))){
    cout << "MIdentify::main: ERROR: F_Image.ReadFileToStrArr(" << CS_LineList_In << ", CS_A2_LineList, CString(' ')) returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Convert LineList_In to Array<double, 2>
  Array<double, 2> D_A2_LineList(CS_A2_LineList.rows(), CS_A2_LineList.cols());
  for (int i_row=0; i_row<CS_A2_LineList.rows(); i_row++){
    for (int i_col=0; i_col<CS_A2_LineList.cols(); i_col++){
      D_A2_LineList(i_row, i_col) = double((CS_A2_LineList(i_row, i_col)).AToD());
    }
  }
  cout << "MIdentify::main: D_A2_LineList = " << D_A2_LineList << endl;

  cout << "MIdentify::main: F_Image.GetNRows() = " << F_Image.GetNRows() << endl;
  cout << "MIdentify::main: F_Image.GetNCols() = " << F_Image.GetNCols() << endl;

  /// Read F_Image.PixArray to Array<double, 1>
  Array<double, 1> D_A1_Spec(1);
  if (F_Image.GetNRows() == 1){
    D_A1_Spec.resize(F_Image.GetNCols());
    D_A1_Spec = (F_Image.GetPixArray())(0,Range::all());
  }
  else if (F_Image.GetNCols() == 1){
    D_A1_Spec.resize(F_Image.GetNRows());
    D_A1_Spec = (F_Image.GetPixArray())(Range::all(),0);
  }
  else{
    cout << "MIdentify::main: ERROR: Input spectrum not one-dimensional" << endl;
    exit(EXIT_FAILURE);
  }

  /// Identify lines
  Array<double, 2> D_A2_SpecCalib_Out(2,2);
  Array<double, 1> D_A1_PolyFitCoeffs_Out(I_Order_In+1);
  double D_RMS_Out;
  Array<double, 2> D_A2_PixWLenOut(2,2);
  D_A2_PixWLenOut = 0.;
  CString *P_CS_Temp = CS_FitsFileName_Ec_In.SubString(0,CS_FitsFileName_Ec_In.LastStrPos(CString("."))-1);
  if (!F_Image.Identify(D_A1_Spec,
                        D_A2_LineList,
                        I_Radius_In,
                        D_FWHM_In,
                        I_Order_In,
                        *P_CS_Temp,
                        D_A2_SpecCalib_Out,
                        D_A1_PolyFitCoeffs_Out,
                        D_RMS_Out,
			D_A2_PixWLenOut)){
    cout << "MIdentify::main: ERROR: Identify returned FALSE" << endl;
    delete(P_CS_Temp);
    exit(EXIT_FAILURE);
  }
  delete(P_CS_Temp);
  
  ///write calibrated spectrum as text file
  if (!F_Image.WriteArrayToFile(D_A2_SpecCalib_Out, CS_TextFileName_Ec_Out, CString("ascii"))){
    cout << "MIdentify::main: ERROR: WriteArrayToFile(spec) returned FALSE" << endl;
    exit(EXIT_FAILURE);
  }

  ///write polynomial fitting coefficients to file
  if (!F_Image.WriteArrayToFile(D_A1_PolyFitCoeffs_Out, CS_TextFileName_Coeffs_Out, CString("ascii"))){
    cout << "MIdentify::main: ERROR: WriteArrayToFile(coeffs) returned FALSE" << endl;
    exit(EXIT_FAILURE);
  }

  /// clean up

  return EXIT_SUCCESS;
}

