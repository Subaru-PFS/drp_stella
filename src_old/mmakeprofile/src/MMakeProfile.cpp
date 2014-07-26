/*
author: Andreas Ritter
created: 01/12/2007
last edited: 01/12/2007
compiler: g++ 4.0
basis machine: Ubuntu Linux 6.06
*/

#include "MMakeProfile.h"

int main(int argc, char *argv[])
{
  cout << "MMakeProfile::main: argc = " << argc << endl;
  if (argc < 7)
  {
    cout << "MMakeProfile::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: makeprofile(char[] FitsFileName_In, char[] DatabaseFileName_In, char[] FitsFileName_Out, double Gain, double ReadOutNoise, int[0,1] Telluric[, int I_SwathWidth)" << endl;
    exit(EXIT_FAILURE);
  }
  
  Array<CString, 1> CS_A1_Args(10);
  CS_A1_Args = CString(" ");
  void **PP_Args;
  PP_Args = (void**)malloc(sizeof(void*) * 10);
  
  int I_SwathWidth = 0;
  char *P_CharArr_In = (char*)argv[1];
  char *P_CharArr_DB = (char*)argv[2];
  char *P_CharArr_Out = (char*)argv[3];
  double D_Gain = (double)(atof((char*)argv[4]));
  cout << "MMakeProfile::main: D_Gain set to " << D_Gain << endl;
  double D_ReadOutNoise = (double)(atof((char*)argv[5]));
  cout << "MMakeProfile::main: D_ReadOutNoise set to " << D_ReadOutNoise << endl;
  int I_Telluric = (double)(atof((char*)argv[6]));
  cout << "MMakeProfile::main: I_Telluric set to " << I_Telluric << endl;
  CS_A1_Args(1) = CString("TELLURIC");
  PP_Args[1] = &I_Telluric;
  if (argc == 8)
  {
    I_SwathWidth = (int)(atoi((char*)argv[7]));
    cout << "MMakeProfile::main: I_SwathWidth set to " << I_SwathWidth << endl;
    CS_A1_Args(0) = CString("SWATH_WIDTH");
    PP_Args[0] = &I_SwathWidth;
  }
  else
  {
    CS_A1_Args(0) = CString("");
    PP_Args[0] = &I_SwathWidth;
  }
  
  CString CS_FitsFileName_In;
  CS_FitsFileName_In.Set(P_CharArr_In);
  CString CS_FitsFileName_Out;
  CS_FitsFileName_Out.Set(P_CharArr_Out);
  CString CS_DatabaseFileName_In;
  CS_DatabaseFileName_In.Set(P_CharArr_DB);

  CFits F_Image;
  cout << "MMakeProfile::main: Starting F_Image.SetFileName(" << CS_FitsFileName_In.Get() << ")" << endl;
  if (!F_Image.SetFileName(CS_FitsFileName_In))
  {
    cout << "MMakeProfile::main: ERROR: F_Image.SetFileName(" << CS_FitsFileName_In.Get() << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Set ReadOutNoise
  cout << "MMakeProfile::main: Starting F_Image.Set_ReadOutNoise(" << D_ReadOutNoise << ")" << endl;
  if (!F_Image.Set_ReadOutNoise( D_ReadOutNoise ))
  {
    cout << "MMakeProfile::main: ERROR: F_Image.Set_ReadOutNoise(" << D_ReadOutNoise << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Set Gain
  cout << "MMakeProfile::main: Starting F_Image.Set_Gain(" << D_Gain << ")" << endl;
  if (!F_Image.Set_Gain( D_Gain ))
  {
    cout << "MMakeProfile::main: ERROR: F_Image.Set_Gain(" << D_Gain << ") returned FALSE" << endl;
    exit(EXIT_FAILURE);
  }
  ///
  cout << "MMakeProfile::main: Starting F_Image.Set_OverSample()" << endl;
  if (!F_Image.Set_OverSample( 10 ))
  {
    cout << "MMakeProfile::main: ERROR: F_Image.Set_OverSample() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Read FitsFile
  cout << "MMakeProfile::main: Starting F_Image.ReadArray()" << endl;
  if (!F_Image.ReadArray())
  {
    cout << "MMakeProfile::main: ERROR: F_Image.ReadArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Set DatabaseFileName_In
  cout << "MMakeProfile::main: Starting F_Image.SetDatabaseFileName(" << CS_DatabaseFileName_In << ")" << endl;
  if (!F_Image.SetDatabaseFileName(CS_DatabaseFileName_In))
  {
    cout << "MMakeProfile::main: ERROR: F_Image.SetDatabaseFileName(" << CS_DatabaseFileName_In << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Read DatabaseFileName_In
  cout << "MMakeProfile::main: Starting F_Image.ReadDatabaseEntry()" << endl;
  if (!F_Image.ReadDatabaseEntry())
  {
    cout << "MMakeProfile::main: ERROR: F_Image.ReadDatabaseEntry() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Calculate Trace Functions
  cout << "MMakeProfile::main: Starting F_Image.CalcTraceFunctions()" << endl;
  if (!F_Image.CalcTraceFunctions())
  {
    cout << "MMakeProfile::main: ERROR: F_Image.CalcTraceFunctions() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Calculate Profile Image
  cout << "MMakeProfile::main: Starting F_Image.MkProfIm()" << endl;
  if (!F_Image.MkProfIm(CS_A1_Args, PP_Args))
  {
    cout << "MMakeProfile::main: ERROR: F_Image.MkProfIm() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Set CS_FitsFileName_Out
  cout << "MMakeProfile::main: Starting F_Image.SetFileName(" << CS_FitsFileName_Out << ")" << endl;
  if (!F_Image.SetFileName(CS_FitsFileName_Out))
  {
    cout << "MMakeProfile::main: ERROR: F_Image.SetFileName(" << CS_FitsFileName_Out << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  F_Image.GetPixArray() = F_Image.GetProfArray();
  
  /// Write Profile Image
  cout << "MMakeProfile::main: Starting F_Image.WriteArray()" << endl;
  if (!F_Image.WriteArray())
  {
    cout << "MMakeProfile::main: ERROR: F_Image.WriteArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  
  return EXIT_SUCCESS;
}
