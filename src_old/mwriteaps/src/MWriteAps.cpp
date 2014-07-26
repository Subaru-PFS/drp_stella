/*
author: Andreas Ritter
created: 05/08/2012
last edited: 05/08/2012
compiler: g++ 4.4
basis machine: Arch Linux
*/

#include "MWriteAps.h"

using namespace std;

int main(int argc, char *argv[])
{
  cout << "MWriteAps::main: argc = " << argc << endl;
  if (argc < 5)
  {
    cout << "MWriteAps::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: writeaps <char[] FitsFileName_In> <char[] FitsFileName_In_Ec> <char[] DatabaseFileName_In> <char[] FitsFileName_Out_Root> [[char[] ApertureList_In]]" << endl;
    exit(EXIT_FAILURE);
  }

  char *P_CharArr_In = (char*)argv[1];
  cout << "MWriteAps::main: P_CharArr_In set to " << P_CharArr_In << endl;

  char *P_CharArr_Ec_In = (char*)argv[2];
  cout << "MWriteAps::main: P_CharArr_Ec_In set to " << P_CharArr_Ec_In << endl;

  char *P_CharArr_DB_In = (char*)argv[3];
  cout << "MWriteAps::main: P_CharArr_DB_In set to " << P_CharArr_DB_In << endl;

  char *P_CharArr_Out = (char*)argv[4];
  cout << "MWriteAps::main: P_CharArr_Out set to " << P_CharArr_Out << endl;

  bool B_ApsGiven = false;
  Array<CString, 1> CS_A1_Apertures(1);
  if (argc == 6){
    B_ApsGiven = true;
    CString CS_ApertureList_In((char*)argv[5]);
    if (!CS_ApertureList_In.ReadFileLinesToStrArr(CS_ApertureList_In, CS_A1_Apertures)){
      cout << "MWriteAps::main: ERROR: ReadFileLinesToStrArr(" << CS_ApertureList_In << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
  }
//  cout << "CS_A1_Apertures

  /// read parameters
  CString CS_FitsFileName_In;
  CS_FitsFileName_In.Set(P_CharArr_In);
  cout << "MWriteAps::main: CS_FitsFileName_In set to " << CS_FitsFileName_In << endl;

  CString CS_FitsFileName_Ec_In;
  CS_FitsFileName_Ec_In.Set(P_CharArr_Ec_In);
  cout << "MWriteAps::main: CS_FitsFileName_Ec_In set to " << CS_FitsFileName_Ec_In << endl;

  CString CS_DBFileName_In;
  CS_DBFileName_In.Set(P_CharArr_DB_In);
  cout << "MWriteAps::main: CS_DBFileName_In set to " << CS_DBFileName_In << endl;

  CString CS_FitsFileName_Out;
  CS_FitsFileName_Out.Set(P_CharArr_Out);
  cout << "MWriteAps::main: CS_FitsFileName_Out set to " << CS_FitsFileName_Out << endl;

  CString *P_CS_DB_Path = CS_FitsFileName_Out.SubString(0, CS_FitsFileName_Out.LastStrPos(CString("/"))-1);
  cout << "MWriteAps::main: *P_CS_DB_Path = " << *P_CS_DB_Path << endl;
  if (!CS_FitsFileName_Out.MkDir(*P_CS_DB_Path)){
    cout << "MWriteAps::main: ERROR: MkDir(" << *P_CS_DB_Path << ") returned FALSE" << endl;
    exit(EXIT_FAILURE);
  }
  delete(P_CS_DB_Path);

  CFits F_Image;
  cout << "MWriteAps::main: Starting F_Image.SetFileName(" << CS_FitsFileName_In.Get() << ")" << endl;
  if (!F_Image.SetFileName(CS_FitsFileName_In))
  {
    cout << "MWriteAps::main: ERROR: F_Image.SetFileName(" << CS_FitsFileName_In.Get() << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Read FitsFile
  cout << "MWriteAps::main: Starting F_Image.ReadArray()" << endl;
  if (!F_Image.ReadArray())
  {
    cout << "MWriteAps::main: ERROR: F_Image.ReadArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Read DatabaseFileName_In
  if (!F_Image.SetDatabaseFileName(CS_DBFileName_In)){
    cout << "MWriteAps::main: ERROR: F_Image.SetDatabaseFileName(" << CS_DBFileName_In << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  if (!F_Image.ReadDatabaseEntry()){
    cout << "MWriteAps::main: ERROR: F_Image.ReadDatabaseEntry() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  if (!F_Image.CalcTraceFunctions()){
    cout << "MWriteAps::main: ERROR: F_Image.CalcTraceFunctions() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Write aperture header information
  F_Image.WriteApHead(CString("aphead_")+CS_FitsFileName_In+CString(".head"));

  CFits F_EcImage;
  /// Set CS_FitsFileName_Ec_In
  cout << "MWriteAps::main: Starting F_EcImage.SetFileName(" << CS_FitsFileName_Ec_In << ")" << endl;
  if (!F_EcImage.SetFileName(CS_FitsFileName_Ec_In))
  {
    cout << "MWriteAps::main: ERROR: F_EcImage.SetFileName(" << CS_FitsFileName_Ec_In << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  ///Read FitsFile
  cout << "MWriteAps::main: Starting F_EcImage.ReadArray()" << endl;
  if (!F_EcImage.ReadArray())
  {
    cout << "MWriteAps::main: ERROR: F_EcImage.ReadArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Write Apertures
  Array<double, 1> D_A1_Ap(F_EcImage.GetNCols());
  Array<double, 2> D_A2_Spec(1,1);
  int I_FirstSignal, I_LastSignal;
  CString CS_FitsFileName_Out_Root;
  CS_FitsFileName_Out_Root = CS_FitsFileName_Out;
  CString* P_CS = new CString(" ");
  Array<double, 1> *P_D_A1_YCenters = F_Image.Get_YCenter();
  Array<double, 1> *P_D_A1_YLow = F_Image.Get_YLow();
  Array<double, 1> *P_D_A1_YHigh = F_Image.Get_YHigh();
  Array<double, 1> *P_D_A1_XCenter = F_Image.Get_XCenter();
  int i_ap = 0;
  int i_end = F_EcImage.GetNRows();
  if (B_ApsGiven)
    i_end = CS_A1_Apertures.size();
  for (int iap=0; iap<i_end; iap++){
    if (B_ApsGiven){
      i_ap = CS_A1_Apertures(iap).AToI();
      cout << "B_ApsGiven = true" << endl;
    }
    else{
      i_ap = iap;
    }
    D_A1_Ap = (F_EcImage.GetPixArray())(iap,Range::all());
//    cout << "iap = " << iap << ", i_ap = " << i_ap << endl;
    if (max(D_A1_Ap) > 0.0000000000000000001){
      I_FirstSignal = (*P_D_A1_YCenters)(i_ap) + (*P_D_A1_YLow)(i_ap);
      ///I_FirstSignal = F_EcImage.FirstIndexWithValueGE(D_A1_Ap, 0.00000001);
//      cout << "MWriteAps::main: i_ap = " << i_ap << ": I_FirstSignal = " << I_FirstSignal << endl;
      if (I_FirstSignal >= 0){
        I_LastSignal = (*P_D_A1_YCenters)(i_ap) + (*P_D_A1_YHigh)(i_ap);
        ///I_LastSignal = F_EcImage.LastIndexWithNonZeroValueBefore(D_A1_Ap, D_A1_Ap.size()-1);
//        cout << "MWriteAps::main: i_ap = " << i_ap << ": I_LastSignal = " << I_LastSignal << endl;
        if (I_LastSignal >= 0){
          D_A2_Spec.resize(I_LastSignal - I_FirstSignal + 1,1);
          D_A2_Spec(Range::all(),0) = D_A1_Ap(Range(I_FirstSignal, I_LastSignal));

//          cout << "MWriteAps::main: i_ap = " << i_ap << ": D_A2_Spec = " << D_A2_Spec << endl;

          CS_FitsFileName_Out.Set(CS_FitsFileName_Out_Root.GetPChar());
          CS_FitsFileName_Out += CString("_ap");
          delete(P_CS);
          P_CS = CS_FitsFileName_Out.IToA(i_ap);
          CS_FitsFileName_Out += *P_CS;
          CS_FitsFileName_Out += CString("_x");
          delete(P_CS);
          P_CS = CS_FitsFileName_Out.IToA(int((*P_D_A1_XCenter)(i_ap)));
          CS_FitsFileName_Out += *P_CS;
          CS_FitsFileName_Out += CString("_y");
          delete(P_CS);
          P_CS = CS_FitsFileName_Out.IToA(int((*P_D_A1_YCenters)(i_ap)));
          CS_FitsFileName_Out += *P_CS;
          CS_FitsFileName_Out += CString(".fits");
          F_Image.WriteFits(&D_A2_Spec,CS_FitsFileName_Out);
        }
      }
    }
  }

  /// clean up
  delete(P_D_A1_YCenters);
  delete(P_D_A1_YHigh);
  delete(P_D_A1_YLow);

  return EXIT_SUCCESS;
}
