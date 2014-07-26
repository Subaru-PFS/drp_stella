/*
author: Andreas Ritter
created: 03/20/2007
last edited: 03/20/2007
compiler: g++ 4.0
basis machine: Ubuntu Linux 6.06
*/

#include "MExtendAps.h"

int main(int argc, char *argv[])
{
  cout << "MExtendAps::main: argc = " << argc << endl;
  if (argc < 6)
  {
    cout << "MExtendAps::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: extendaps char[] FitsFileName_In, char[] DatabaseFileName_In, int<0 ExtendNPixDown, int>0 ExtendNPixUp char[] DatabaseFileName_Out" << endl;
    exit(EXIT_FAILURE);
  }
  char *P_CharArr_In = (char*)argv[1];
  char *P_CharArr_DB = (char*)argv[2];
  int I_NPixDown = (int)(atoi((char*)argv[3]));
  int I_NPixup = (int)(atoi((char*)argv[4]));
  char *P_CharArr_DBOut = (char*)argv[5];
  
  CString CS_FitsFileName_In;
  CS_FitsFileName_In.Set(P_CharArr_In);
  CString CS_DBFileName_Out;
  CS_DBFileName_Out.Set(P_CharArr_DBOut);
  CString CS_DatabaseFileName_In;
  CS_DatabaseFileName_In.Set(P_CharArr_DB);

  CFits F_Image;
  if (!F_Image.SetFileName(CS_FitsFileName_In))
  {
    cout << "MExtendAps::main: ERROR: F_Image.SetFileName(" << CS_FitsFileName_In.Get() << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MExtendAps::main: FileName <" << CS_FitsFileName_In.Get() << "> set" << endl;

  /// Read FitsFile
  if (!F_Image.ReadArray())
  {
    cout << "MExtendAps::main: ERROR: F_Image.ReadArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MExtendAps::main: F_Image: Array read" << endl;

  /// Set DatabaseFileName_In
  if (!F_Image.SetDatabaseFileName(CS_DatabaseFileName_In))
  {
    cout << "MExtendAps::main: ERROR: F_Image.SetDatabaseFileName(" << CS_DatabaseFileName_In << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MExtendAps::main: F_Image: DatabaseFileName <" << CS_DatabaseFileName_In << "> set" << endl;

  /// Read DatabaseFileName_In
  if (!F_Image.ReadDatabaseEntry())
  {
    cout << "MExtendAps::main: ERROR: F_Image.ReadDatabaseEntry() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MExtendAps::main: F_Image: DatabaseEntry read" << endl;

  /// Calculate Trace Functions
  if (!F_Image.CalcTraceFunctions())
  {
    cout << "MExtendAps::main: ERROR: F_Image.CalcTraceFunctions() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MExtendAps::main: F_Image: TraceFunctions calculated" << endl;

  Array<double, 1> *P_D_A1_YLow = F_Image.Get_YLow();
  cout << "MExtendAps: Original YLow = " << *P_D_A1_YLow << endl;
  (*P_D_A1_YLow) = (*P_D_A1_YLow) + I_NPixDown;
  if (!F_Image.Set_YLow(*P_D_A1_YLow)){
    cout << "MExtendAps: ERROR: F_Image.Set_YLow(*P_D_A1_YLow) returned FALSE" << endl;
    exit(EXIT_FAILURE);
  }
  delete(P_D_A1_YLow);
  
  P_D_A1_YLow = F_Image.Get_YHigh();
  (*P_D_A1_YLow) = (*P_D_A1_YLow) + I_NPixup;
  if (!F_Image.Set_YHigh(*P_D_A1_YLow)){
    cout << "MExtendAps: ERROR: F_Image.Set_YHigh(*P_D_A1_YHigh) returned FALSE" << endl;
    exit(EXIT_FAILURE);
  }
  delete(P_D_A1_YLow);
  
  /// Set CS_FitsFileName_Out
  if (!F_Image.SetDatabaseFileName(CS_DBFileName_Out))
  {
    cout << "MExtendAps::main: ERROR: F_Image.SetDatabaseFileName(" << CS_DBFileName_Out << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MExtendAps::main: F_Image: DatabaseFileName <" << CS_DBFileName_Out << "> set" << endl;
  
  /// Write Center Image
  if (!F_Image.WriteDatabaseEntry())
  {
    cout << "MExtendAps::main: ERROR: F_Image.WriteDatabaseEntry() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MExtendAps::main: F_Image: DatabaseFileName_Out written" << endl;
  
  return EXIT_SUCCESS;
}
