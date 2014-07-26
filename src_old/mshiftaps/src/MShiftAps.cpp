/*
author: Andreas Ritter
created: 03/20/2007
last edited: 03/20/2007
compiler: g++ 4.0
basis machine: Ubuntu Linux 6.06
*/

#include "MShiftAps.h"

int main(int argc, char *argv[])
{
  cout << "MShiftAps::main: argc = " << argc << endl;
  if (argc < 5)
  {
    cout << "MShiftAps::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: shiftaps <char[] FitsFileName_In> <char[] DatabaseFileName_In> <double<>0 ApertureShift> char[] DatabaseFileName_Out" << endl;
    cout << "MShiftAps: ApertureLength < 0: shift apertures to the left" << endl;
    cout << "MShiftAps: ApertureLength > 0: shift apertures to the right" << endl;
    cout << "shiftaps task: reads FitsFileName_In and DatabaseFileName_In, shifts the apertures by ApertureShift pixels, and writes new aperture definitions to DatabaseFileName_Out" << endl;
    exit(EXIT_FAILURE);
  }
  char *P_CharArr_In = (char*)argv[1];
  char *P_CharArr_DB = (char*)argv[2];
  double D_Shift_In = (double)(atof((char*)argv[3]));
  char *P_CharArr_DBOut = (char*)argv[4];

  CString CS_FitsFileName_In;
  CS_FitsFileName_In.Set(P_CharArr_In);
  CString CS_DBFileName_Out;
  CS_DBFileName_Out.Set(P_CharArr_DBOut);
  CString CS_DatabaseFileName_In;
  CS_DatabaseFileName_In.Set(P_CharArr_DB);

  CFits F_Image;
  if (!F_Image.SetFileName(CS_FitsFileName_In))
  {
    cout << "MShiftAps::main: ERROR: F_Image.SetFileName(" << CS_FitsFileName_In.Get() << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MShiftAps::main: FileName <" << CS_FitsFileName_In.Get() << "> set" << endl;

  /// Read FitsFile
  if (!F_Image.ReadArray())
  {
    cout << "MShiftAps::main: ERROR: F_Image.ReadArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MShiftAps::main: F_Image: Array read" << endl;

  /// Set DatabaseFileName_In
  if (!F_Image.SetDatabaseFileName(CS_DatabaseFileName_In))
  {
    cout << "MShiftAps::main: ERROR: F_Image.SetDatabaseFileName(" << CS_DatabaseFileName_In << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MShiftAps::main: F_Image: DatabaseFileName <" << CS_DatabaseFileName_In << "> set" << endl;

  /// Read DatabaseFileName_In
  if (!F_Image.ReadDatabaseEntry())
  {
    cout << "MShiftAps::main: ERROR: F_Image.ReadDatabaseEntry() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MShiftAps::main: F_Image: DatabaseEntry read" << endl;

  /// Calculate Trace Functions
  if (!F_Image.CalcTraceFunctions())
  {
    cout << "MShiftAps::main: ERROR: F_Image.CalcTraceFunctions() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MShiftAps::main: F_Image: TraceFunctions calculated" << endl;

  F_Image.ShiftApertures(D_Shift_In);

  /// Set CS_FitsFileName_Out
  if (!F_Image.SetDatabaseFileName(CS_DBFileName_Out))
  {
    cout << "MShiftAps::main: ERROR: F_Image.SetDatabaseFileName(" << CS_DBFileName_Out << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MShiftAps::main: F_Image: DatabaseFileName <" << CS_DBFileName_Out << "> set" << endl;

  /// Write Center Image
  if (!F_Image.WriteDatabaseEntry())
  {
    cout << "MShiftAps::main: ERROR: F_Image.WriteDatabaseEntry() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MShiftAps::main: F_Image: DatabaseFileName_Out written" << endl;

  return EXIT_SUCCESS;
}
