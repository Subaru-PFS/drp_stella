/*
author: Andreas Ritter
created: 10/01/2014
last edited: 10/01/2014
compiler: g++ 4.4
basis machine: Arch Linux
*/

#include "MTransposeArray.h"

using namespace std;

int main(int argc, char *argv[])
{
  cout << "MTransposeArray::main: argc = " << argc << endl;
  if (argc < 3)
  {
    cout << "MTransposeArray::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: transposearray <char[] [@]FileName_In> <char[] [@]FileName_Out>" << endl;
    exit(EXIT_FAILURE);
  }

  char *P_CharArr_In = (char*)argv[1];
  cout << "MTransposeArray::main: P_CharArr_Op1_In set to " << P_CharArr_In << endl;

  char *P_CharArr_Out = (char*)argv[2];
  cout << "MTransposeArray::main: P_CharArr_Out set to " << P_CharArr_Out << endl;

  /// read parameters
  CString CS_FileName_In;
  CS_FileName_In.Set(P_CharArr_In);
  Array<CString, 1> CS_A1_FileNames_In(1);
  CS_A1_FileNames_In(0) = CS_FileName_In;
  cout << "MTransposeArray::main: CS_FileName_Op1_In set to " << CS_FileName_In << endl;

  CString CS_FileName_Out;
  CS_FileName_Out.Set(P_CharArr_Out);
  cout << "MTransposeArray::main: CS_FileName_Out set to " << CS_FileName_Out << endl;
  Array<CString, 1> CS_A1_FileNames_Out(1);
  CS_A1_FileNames_Out(0) = CS_FileName_Out;
  CString *P_CS;
  if (CS_FileName_In.IsList()){
    P_CS = CS_FileName_In.SubString(1);
    if (!P_CS->ReadFileLinesToStrArr(*P_CS, CS_A1_FileNames_In)){
      cout << "MTransposeArray::main: ERROR: ReadFileLinesToStrArr(" << *P_CS << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    if (!CS_FileName_Out.IsList()){
      cout << "MTransposeArray::main: ERROR: FileName_In is list but FileName_Out is not" << endl;
      exit(EXIT_FAILURE);
    }
    delete(P_CS);
    P_CS = CS_FileName_Out.SubString(1);
    if (!P_CS->ReadFileLinesToStrArr(*P_CS, CS_A1_FileNames_Out)){
      cout << "MTransposeArray::main: ERROR: ReadFileLinesToStrArr(" << *P_CS << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    delete(P_CS);
  }

//  CFits F_Image;
  Array<CString, 2> CS_A2_Array_In(2,2);
  CS_A2_Array_In = CString(" ");
  Array<CString, 2> CS_A2_Array_Out(2,2);
  CS_A2_Array_Out = CString(" ");
  CString CS(" ");

  for (int i_file = 0; i_file < CS_A1_FileNames_In.size(); i_file++){
    /// Read File
    cout << "MTransposeArray::main: Starting ReadFileToStrArr(" << CS_A1_FileNames_In(i_file) << ")" << endl;
    if (!CS.ReadFileToStrArr(CS_A1_FileNames_In(i_file), CS_A2_Array_In, CString(" ")))
    {
      cout << "MTransposeArray::main: ERROR: ReadFileToStrArr() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
    cout << "MTransposeArray::main: CS_A2_Array_In = " << CS_A2_Array_In << endl;

    CS_A2_Array_Out.resize(CS_A2_Array_In.cols(), CS_A2_Array_In.rows());
    
    for (int i_row=0; i_row<CS_A2_Array_In.rows(); i_row++){
      for (int i_col=0; i_col<CS_A2_Array_In.cols(); i_col++){
        CS_A2_Array_Out(i_col, i_row) = CS_A2_Array_In(i_row, i_col);
      }
    }

    ///write output file
    if (!CS.WriteStrListToFile(CS_A2_Array_Out, CString(" "), CS_A1_FileNames_Out(i_file))){
      cout << "MTransposeArray::main: ERROR: WriteStrListToFile() returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
  }
  /// clean up

  return EXIT_SUCCESS;
}

