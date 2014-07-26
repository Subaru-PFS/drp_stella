/*
author: Andreas Ritter
created: 03/20/2007
last edited: 03/20/2007
compiler: g++ 4.0
basis machine: Ubuntu Linux 6.06
*/

#include "MSetApLength.h"

int main(int argc, char *argv[])
{
  cout << "MSetApLength::main: argc = " << argc << endl;
  if (argc < 5)
  {
    cout << "MSetApLength::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: setaplength <char[] FitsFileName_In> <char[] DatabaseFileName_In> <int<>0 ApertureLength> char[] DatabaseFileName_Out" << endl;
    cout << "MSetApLength: ApertureLength < 0: take y_high and set y_low" << endl;
    cout << "MSetApLength: ApertureLength > 0: take y_low and set y_high" << endl;
    exit(EXIT_FAILURE);
  }
  char *P_CharArr_In = (char*)argv[1];
  char *P_CharArr_DB = (char*)argv[2];
  int I_Length_In = (int)(atoi((char*)argv[3]));
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
    cout << "MSetApLength::main: ERROR: F_Image.SetFileName(" << CS_FitsFileName_In.Get() << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MSetApLength::main: FileName <" << CS_FitsFileName_In.Get() << "> set" << endl;

  /// Read FitsFile
  if (!F_Image.ReadArray())
  {
    cout << "MSetApLength::main: ERROR: F_Image.ReadArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MSetApLength::main: F_Image: Array read" << endl;

  /// Set DatabaseFileName_In
  if (!F_Image.SetDatabaseFileName(CS_DatabaseFileName_In))
  {
    cout << "MSetApLength::main: ERROR: F_Image.SetDatabaseFileName(" << CS_DatabaseFileName_In << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MSetApLength::main: F_Image: DatabaseFileName <" << CS_DatabaseFileName_In << "> set" << endl;

  /// Read DatabaseFileName_In
  if (!F_Image.ReadDatabaseEntry())
  {
    cout << "MSetApLength::main: ERROR: F_Image.ReadDatabaseEntry() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MSetApLength::main: F_Image: DatabaseEntry read" << endl;

  /// Calculate Trace Functions
  if (!F_Image.CalcTraceFunctions())
  {
    cout << "MSetApLength::main: ERROR: F_Image.CalcTraceFunctions() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MSetApLength::main: F_Image: TraceFunctions calculated" << endl;

  Array<double, 1> *P_D_A1_YLow = F_Image.Get_YLow();
  cout << "MSetApLength: Original YLow = " << *P_D_A1_YLow << endl;
  Array<double, 1> *P_D_A1_YHigh = F_Image.Get_YHigh();
  Array<double, 1> *P_D_A1_YCenter = F_Image.Get_YCenter();
  
  if (I_Length_In < 0){
    (*P_D_A1_YLow) = (*P_D_A1_YHigh) + I_Length_In;
    for (int i=0; i<P_D_A1_YLow->size(); i++){
      if ((*P_D_A1_YCenter)(i) + (*P_D_A1_YLow)(i) < 0)
	(*P_D_A1_YLow)(i) = 0 - (*P_D_A1_YCenter)(i);
    }
    if (!F_Image.Set_YLow(*P_D_A1_YLow)){
      cout << "MSetApLength: ERROR: F_Image.Set_YLow(*P_D_A1_YLow) returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
  }
  else{
    (*P_D_A1_YHigh) = (*P_D_A1_YLow) + I_Length_In;
    for (int i=0; i<P_D_A1_YLow->size(); i++){
      if ((*P_D_A1_YCenter)(i) + (*P_D_A1_YHigh)(i) >= F_Image.GetNRows())
	(*P_D_A1_YLow)(i) = F_Image.GetNRows() - (*P_D_A1_YCenter)(i) - 1;
    }
    if (!F_Image.Set_YHigh(*P_D_A1_YHigh)){
      cout << "MSetApLength: ERROR: F_Image.Set_YHigh(*P_D_A1_YHigh) returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
  }
  delete(P_D_A1_YLow);
  delete(P_D_A1_YHigh);
  
  /// Set CS_FitsFileName_Out
  if (!F_Image.SetDatabaseFileName(CS_DBFileName_Out))
  {
    cout << "MSetApLength::main: ERROR: F_Image.SetDatabaseFileName(" << CS_DBFileName_Out << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MSetApLength::main: F_Image: DatabaseFileName <" << CS_DBFileName_Out << "> set" << endl;
  
  /// Write Center Image
  if (!F_Image.WriteDatabaseEntry())
  {
    cout << "MSetApLength::main: ERROR: F_Image.WriteDatabaseEntry() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MSetApLength::main: F_Image: DatabaseFileName_Out written" << endl;
  
  return EXIT_SUCCESS;
}
