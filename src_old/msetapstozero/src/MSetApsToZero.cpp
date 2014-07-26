/*
author: Andreas Ritter
created: 03/20/2007
last edited: 03/20/2007
compiler: g++ 4.0
basis machine: Ubuntu Linux 6.06
*/

#include "MSetApsToZero.h"

int main(int argc, char *argv[])
{
  cout << "MSetApsToZero::main: argc = " << argc << endl;
  if (argc < 4)
  {
    cout << "MSetApsToZero::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: setapstozero <char[] FitsFileName_In> <char[] DatabaseFileName_In> <char[] FitsFileName_Out>" << endl;
    exit(EXIT_FAILURE);
  }
  char *P_CharArr_In = (char*)argv[1];
  char *P_CharArr_DB = (char*)argv[2];
  char *P_CharArr_Out = (char*)argv[3];
  
  CString CS_FitsFileName_In;
  CS_FitsFileName_In.Set(P_CharArr_In);
  CString CS_FitsFileName_Out;
  CS_FitsFileName_Out.Set(P_CharArr_Out);
  CString CS_DatabaseFileName_In;
  CS_DatabaseFileName_In.Set(P_CharArr_DB);

  CFits F_Image;
  if (!F_Image.SetFileName(CS_FitsFileName_In))
  {
    cout << "MSetApsToZero::main: ERROR: F_Image.SetFileName(" << CS_FitsFileName_In.Get() << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MSetApsToZero::main: FileName <" << CS_FitsFileName_In.Get() << "> set" << endl;

  /// Read FitsFile
  if (!F_Image.ReadArray())
  {
    cout << "MSetApsToZero::main: ERROR: F_Image.ReadArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MSetApsToZero::main: F_Image: Array read" << endl;

  /// Set DatabaseFileName_In
  if (!F_Image.SetDatabaseFileName(CS_DatabaseFileName_In))
  {
    cout << "MSetApsToZero::main: ERROR: F_Image.SetDatabaseFileName(" << CS_DatabaseFileName_In << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MSetApsToZero::main: F_Image: DatabaseFileName <" << CS_DatabaseFileName_In << "> set" << endl;
  
//  CString CS_Path(" ");
//  F_Image.Get_Path(CS_Path);
//  cout << "MSetApsToZero: CS_Path = <" << CS_Path << ">" << endl;
//  exit(EXIT_FAILURE);

  /// Read DatabaseFileName_In
  if (!F_Image.ReadDatabaseEntry())
  {
    cout << "MSetApsToZero::main: ERROR: F_Image.ReadDatabaseEntry() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MSetApsToZero::main: F_Image: DatabaseEntry read" << endl;

  /// Calculate Trace Functions
  if (!F_Image.CalcTraceFunctions())
  {
    cout << "MSetApsToZero::main: ERROR: F_Image.CalcTraceFunctions() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MSetApsToZero::main: F_Image: TraceFunctions calculated" << endl;

  
///  Array<double, 1> D_A1_YLow(1);
///  D_A1_YLow(0) = 1;
///  F_Image.Set_YLow(D_A1_YLow);
  
  /// Mark Centers
  if (!F_Image.Set_ApertureDataToZero(0,0))
  {
    cout << "MSetApsToZero::main: ERROR: F_Image.Set_ApertureDataToZero() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MSetApsToZero::main: F_Image: Apertures set to zero" << endl;

  /// Set CS_FitsFileName_Out
  if (!F_Image.SetFileName(CS_FitsFileName_Out))
  {
    cout << "MSetApsToZero::main: ERROR: F_Image.SetFileName(" << CS_FitsFileName_Out << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MSetApsToZero::main: F_Image: FileName <" << CS_FitsFileName_Out << "> set" << endl;
  
  /// Write Center Image
  if (!F_Image.WriteArray())
  {
    cout << "MSetApsToZero::main: ERROR: F_Image.WriteArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MSetApsToZero::main: F_Image: Array written" << endl;
  
  return EXIT_SUCCESS;
}
