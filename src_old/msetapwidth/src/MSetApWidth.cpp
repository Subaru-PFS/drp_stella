/*
author: Andreas Ritter
created: 03/20/2007
last edited: 03/20/2007
compiler: g++ 4.0
basis machine: Ubuntu Linux 6.06
*/

#include "MSetApWidth.h"

int main(int argc, char *argv[])
{
  cout << "MSetApWidth::main: argc = " << argc << endl;
  if (argc < 6)
  {
    cout << "MSetApWidth::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: extendaps char[] FitsFileName_In, char[] DatabaseFileName_In, double<0 XMin, double>0 XMax char[] DatabaseFileName_Out" << endl;
    exit(EXIT_FAILURE);
  }
  char *P_CharArr_In = (char*)argv[1];
  char *P_CharArr_DB = (char*)argv[2];
  double D_XMin = (double)(atof((char*)argv[3]));
  double D_XMax = (double)(atof((char*)argv[4]));
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
    cout << "MSetApWidth::main: ERROR: F_Image.SetFileName(" << CS_FitsFileName_In.Get() << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MSetApWidth::main: FileName <" << CS_FitsFileName_In.Get() << "> set" << endl;

  /// Read FitsFile
  if (!F_Image.ReadArray())
  {
    cout << "MSetApWidth::main: ERROR: F_Image.ReadArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MSetApWidth::main: F_Image: Array read" << endl;

  /// Set DatabaseFileName_In
  if (!F_Image.SetDatabaseFileName(CS_DatabaseFileName_In))
  {
    cout << "MSetApWidth::main: ERROR: F_Image.SetDatabaseFileName(" << CS_DatabaseFileName_In << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MSetApWidth::main: F_Image: DatabaseFileName <" << CS_DatabaseFileName_In << "> set" << endl;

  /// Read DatabaseFileName_In
  if (!F_Image.ReadDatabaseEntry())
  {
    cout << "MSetApWidth::main: ERROR: F_Image.ReadDatabaseEntry() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MSetApWidth::main: F_Image: DatabaseEntry read" << endl;

  /// Calculate Trace Functions
  if (!F_Image.CalcTraceFunctions())
  {
    cout << "MSetApWidth::main: ERROR: F_Image.CalcTraceFunctions() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MSetApWidth::main: F_Image: TraceFunctions calculated" << endl;

  Array<double, 1> *P_D_A1_X = F_Image.Get_XLow();
  cout << "MSetApWidth: Original XLow = " << *P_D_A1_X << endl;
  
  (*P_D_A1_X) = D_XMin;
  if (!F_Image.Set_XLow(*P_D_A1_X)){
    cout << "MSetApWidth: ERROR: F_Image.Set_XLow(*P_D_A1_XLow) returned FALSE" << endl;
    exit(EXIT_FAILURE);
  }
  if (!F_Image.Set_XMin(*P_D_A1_X)){
    cout << "MSetApWidth: ERROR: F_Image.Set_XLow(*P_D_A1_XMin) returned FALSE" << endl;
    exit(EXIT_FAILURE);
  }
  delete(P_D_A1_X);
  
  P_D_A1_X = F_Image.Get_XHigh();
  (*P_D_A1_X) = D_XMax;
  if (!F_Image.Set_XHigh(*P_D_A1_X)){
    cout << "MSetApWidth: ERROR: F_Image.Set_XHigh(*P_D_A1_XHigh) returned FALSE" << endl;
    exit(EXIT_FAILURE);
  }
  if (!F_Image.Set_XMax(*P_D_A1_X)){
    cout << "MSetApWidth: ERROR: F_Image.Set_XHigh(*P_D_A1_XMax) returned FALSE" << endl;
    exit(EXIT_FAILURE);
  }
  delete(P_D_A1_X);
  
  /// Set CS_FitsFileName_Out
  if (!F_Image.SetDatabaseFileName(CS_DBFileName_Out))
  {
    cout << "MSetApWidth::main: ERROR: F_Image.SetDatabaseFileName(" << CS_DBFileName_Out << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MSetApWidth::main: F_Image: DatabaseFileName <" << CS_DBFileName_Out << "> set" << endl;
  
  /// Write Center Image
  if (!F_Image.WriteDatabaseEntry())
  {
    cout << "MSetApWidth::main: ERROR: F_Image.WriteDatabaseEntry() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MSetApWidth::main: F_Image: DatabaseFileName_Out written" << endl;
  
  return EXIT_SUCCESS;
}
