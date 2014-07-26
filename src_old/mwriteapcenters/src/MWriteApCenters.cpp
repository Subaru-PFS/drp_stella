/*
author: Andreas Ritter
created: 04/12/2007
last edited: 05/05/2007
compiler: g++ 4.0
basis machine: Ubuntu Linux 6.06
*/

#include "MWriteApCenters.h"

using namespace std;

int main(int argc, char *argv[])
{
  cout << "MWriteApCenters::main: argc = " << argc << endl;
  if (argc < 4)
  {
    cout << "MWriteApCenters::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: writeapcenters <char[] FitsFileName_In> <char[] DatabaseFileName_In> <char[] TextFileName_Out>" << endl;//"[, ERR_FROM_PROFILE_OUT=char[]])" << endl;
    exit(EXIT_FAILURE);
  }

  char *P_CharArr_FitsFileName = (char*)argv[1];
  char *P_CharArr_DB = (char*)argv[2];
  char *P_CharArr_Out = (char*)argv[3];
  CString CS_FitsFileName_In;
  CS_FitsFileName_In.Set(P_CharArr_FitsFileName);
  CString CS_TextFileName_Out;
  CS_TextFileName_Out.Set(P_CharArr_Out);
  CString CS_DatabaseFileName_In;
  CS_DatabaseFileName_In.Set(P_CharArr_DB);

  CFits F_Image;
  /// Set DatabaseFileName_In
  cout << "MWriteApCenters::main: Starting F_Image.SetFileName(" << CS_FitsFileName_In << ")" << endl;
  if (!F_Image.SetFileName(CS_FitsFileName_In))
  {
    cout << "MWriteApCenters::main: ERROR: F_Image.SetFileName(" << CS_FitsFileName_In << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  
  if (!F_Image.ReadArray()){
    cout << "MWriteApCenters::main: ERROR: F_Image.ReadArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  
  cout << "MWriteApCenters::main: Starting F_Image.SetDatabaseFileName(" << CS_DatabaseFileName_In << ")" << endl;
  if (!F_Image.SetDatabaseFileName(CS_DatabaseFileName_In))
  {
    cout << "MWriteApCenters::main: ERROR: F_Image.SetDatabaseFileName(" << CS_DatabaseFileName_In << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Read DatabaseFileName_In
  cout << "MWriteApCenters::main: Starting F_Image.ReadDatabaseEntry()" << endl;
  if (!F_Image.ReadDatabaseEntry())
  {
    cout << "MWriteApCenters::main: ERROR: F_Image.ReadDatabaseEntry() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  
  /// Calculate Trace Functions
  if (!F_Image.CalcTraceFunctions()){
    cout << "MWriteApCenters::main: ERROR: CalcTraceFunctions() returned FALSE" << endl;
    exit(EXIT_FAILURE);
  }

  Array<double, 2> D_A2_ApCenters_Out(F_Image.Get_NApertures(),2);
  Array<double, 2> *P_D_A2_XCenters = F_Image.Get_XCenters();
  Array<double, 1> *P_D_A1_YCenter = F_Image.Get_YCenter();
  Array<double, 1> *P_D_A1_YLow = F_Image.Get_YLow();
  Array<double, 1> *P_D_A1_YHigh = F_Image.Get_YHigh();
  Array<double, 1> D_A1_YCenter(F_Image.Get_NApertures());
  D_A1_YCenter = (*P_D_A1_YCenter) + (*P_D_A1_YLow) + (((*P_D_A1_YHigh) - (*P_D_A1_YLow)) / 2.);
  Array<double, 1> D_A1_XCenter(F_Image.Get_NApertures());
  for (int i=0; i<D_A1_XCenter.size(); i++)
    D_A1_XCenter(i) = (*P_D_A2_XCenters)(i, (int)(D_A1_YCenter)(i));
  D_A2_ApCenters_Out(Range::all(),0) = D_A1_XCenter;
  D_A2_ApCenters_Out(Range::all(),1) = D_A1_YCenter;

  if (!F_Image.WriteArrayToFile(D_A2_ApCenters_Out, CS_TextFileName_Out, CString("ascii"))){
    cout << "MWriteApCenters::main: ERROR: WriteArrayToFile returned false" << endl;
    exit(EXIT_FAILURE);
  }
  delete(P_D_A2_XCenters);
  delete(P_D_A1_YCenter);
  delete(P_D_A1_YLow);
  delete(P_D_A1_YHigh);
  return EXIT_SUCCESS;
}
