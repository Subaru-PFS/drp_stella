/*
author: Andreas Ritter
created: 03/20/2007
last edited: 03/20/2007
compiler: g++ 4.0
basis machine: Ubuntu Linux 6.06
*/

#include "MMarkCenters.h"

int main(int argc, char *argv[])
{
  cout << "MMarkCenters::main: argc = " << argc << endl;
  if (argc < 4)
  {
    cout << "MMarkCenters::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: markcenters <char[] [@]FitsFileName_In> <char[] [@]DatabaseFileName_In> <char[] [@]FitsFileName_Out>" << endl;
    exit(EXIT_FAILURE);
  }
  char *P_CharArr_In = (char*)argv[1];
  char *P_CharArr_DB = (char*)argv[2];
  char *P_CharArr_Out = (char*)argv[3];

  CString CS_FitsFileNameList_In;
  CS_FitsFileNameList_In.Set(P_CharArr_In);
  CString CS_FitsFileName_In;
  CString CS_FitsFileName_Out;
  CString CS_FitsFileNameList_Out;
  CS_FitsFileNameList_Out.Set(P_CharArr_Out);
  CString CS_DatabaseFileName_In;
  CS_DatabaseFileName_In.Set(P_CharArr_DB);
  Array<CString, 1> CS_A1_FitsFileNames_In(1);
  CS_A1_FitsFileNames_In(0).Set(CS_FitsFileNameList_In);
  Array<CString, 1> CS_A1_FitsFileNames_Out(1);
  CS_A1_FitsFileNames_Out(0).Set(CS_FitsFileNameList_Out);
  if (CS_FitsFileNameList_In.IsList()){
    if (!CS_FitsFileName_In.ReadFileLinesToStrArr(CS_FitsFileNameList_In, CS_A1_FitsFileNames_In)){
      cout << "MMarkCenters::main: ERROR: ReadFileLinesToStrArr(" << CS_FitsFileNameList_In << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    if (!CS_FitsFileNameList_Out.IsList()){
      cout << "MMarkCenters::main: ERROR: " << CS_FitsFileNameList_In << " is list, but " << CS_FitsFileNameList_Out << " isn't" << endl;
      exit(EXIT_FAILURE);
    }
    if (!CS_FitsFileName_In.ReadFileLinesToStrArr(CS_FitsFileNameList_Out, CS_A1_FitsFileNames_Out)){
      cout << "MMarkCenters::main: ERROR: ReadFileLinesToStrArr(" << CS_FitsFileNameList_Out << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    if (CS_A1_FitsFileNames_In.size() != CS_A1_FitsFileNames_Out.size()){
      cout << "MMarkCenters::main: ERROR: " << CS_FitsFileNameList_In << " contains " << CS_A1_FitsFileNames_In.size() << " files, but " << CS_FitsFileNameList_Out << " contains " << CS_A1_FitsFileNames_Out.size() << endl;
      exit(EXIT_FAILURE);
    }
  }
  Array<CString, 1> CS_A1_DBFileNames_In(CS_A1_FitsFileNames_In.size());
  CS_A1_DBFileNames_In = CS_DatabaseFileName_In;
  if (CS_DatabaseFileName_In.IsList()){
    if (!CS_DatabaseFileName_In.ReadFileLinesToStrArr(CS_DatabaseFileName_In, CS_A1_DBFileNames_In)){
      cout << "MMarkCenters::main: ERROR: ReadFileLinesToStrArr(" << CS_DatabaseFileName_In << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    if (CS_A1_DBFileNames_In.size() != CS_A1_FitsFileNames_In.size()){
      cout << "MMarkCenters::main: ERROR: CS_A1_DBFileNames_In.size(=" << CS_A1_DBFileNames_In.size() <<") != CS_A1_FitsFileNames_In.size(=" << CS_A1_FitsFileNames_In.size() << ") => Returning FALSE" << endl;
      exit(EXIT_FAILURE);
    }
  }

  CFits F_Image;
  for (int i_file=0; i_file < CS_A1_FitsFileNames_In.size(); i_file++){
    CS_FitsFileName_In.Set(CS_A1_FitsFileNames_In(i_file));
    CS_FitsFileName_Out.Set(CS_A1_FitsFileNames_Out(i_file));
    if (!F_Image.SetFileName(CS_FitsFileName_In))
    {
      cout << "MMarkCenters::main: ERROR: F_Image.SetFileName(" << CS_FitsFileName_In.Get() << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
    cout << "MMarkCenters::main: FileName <" << CS_FitsFileName_In.Get() << "> set" << endl;

    /// Read FitsFile
    if (!F_Image.ReadArray())
    {
      cout << "MMarkCenters::main: ERROR: F_Image.ReadArray() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
    cout << "MMarkCenters::main: F_Image: Array read" << endl;

    /// Set DatabaseFileName_In
    if (!F_Image.SetDatabaseFileName(CS_A1_DBFileNames_In(i_file)))
    {
      cout << "MMarkCenters::main: ERROR: F_Image.SetDatabaseFileName(" << CS_A1_DBFileNames_In(i_file) << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
    cout << "MMarkCenters::main: F_Image: DatabaseFileName <" << CS_A1_DBFileNames_In(i_file) << "> set" << endl;

    /// Read DatabaseFileName_In
    if (!F_Image.ReadDatabaseEntry())
    {
      cout << "MMarkCenters::main: ERROR: F_Image.ReadDatabaseEntry() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
    cout << "MMarkCenters::main: F_Image: DatabaseEntry read" << endl;

    /// Calculate Trace Functions
    if (!F_Image.CalcTraceFunctions())
    {
      cout << "MMarkCenters::main: ERROR: F_Image.CalcTraceFunctions() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
    cout << "MMarkCenters::main: F_Image: TraceFunctions calculated" << endl;


///  Array<double, 1> D_A1_YLow(1);
///  D_A1_YLow(0) = 1;
///  F_Image.Set_YLow(D_A1_YLow);

    /// Mark Centers
    if (!F_Image.MarkCenters())
    {
      cout << "MMarkCenters::main: ERROR: F_Image.MarkCenters() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
    cout << "MMarkCenters::main: F_Image: Centers marked" << endl;

    /// Set CS_FitsFileName_Out
    if (!F_Image.SetFileName(CS_FitsFileName_Out))
    {
      cout << "MMarkCenters::main: ERROR: F_Image.SetFileName(" << CS_FitsFileName_Out << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
    cout << "MMarkCenters::main: F_Image: FileName <" << CS_FitsFileName_Out << "> set" << endl;

    /// Write Center Image
    if (!F_Image.WriteArray())
    {
      cout << "MMarkCenters::main: ERROR: F_Image.WriteArray() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
  cout << "MMarkCenters::main: F_Image: Array written" << endl;
  }

  return EXIT_SUCCESS;
}
