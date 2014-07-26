/*
author: Andreas Ritter
created: 05/08/2012
last edited: 05/08/2012
compiler: g++ 4.4
basis machine: Arch Linux
*/

#include "MReverse.h"

using namespace std;

int main(int argc, char *argv[])
{
  cout << "MReverse::main: argc = " << argc << endl;
  if (argc < 3)
  {
    cout << "MReverse::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: imdivide <char[] [@]FitsFileName[List]_In> <char[] [@]FitsFileName[List]_Out>" << endl;
    exit(EXIT_FAILURE);
  }

  CString CS_In((char*)argv[1]);
  cout << "MReverse::main: CS_In set to " << CS_In << endl;
  
  CString CS_Out = (char*)argv[2];
  cout << "MReverse::main: CS_Out set to " << CS_Out << endl;

  /// read parameters
  Array<CString, 1> CS_A1_In(1);
  CS_A1_In(0) = CS_In;

  Array<CString, 1> CS_A1_Out(1);
  CS_A1_Out(0) = CS_Out;
  
  CString CS_Comp("@");
  CString *P_CS_Temp = CS_In.SubString(0,0);
  CString *P_CS_FN;
  if (P_CS_Temp->EqualValue(CS_Comp)){
    P_CS_FN = CS_In.SubString(1);
    CS_In.Set(*P_CS_FN);
    if (!CS_Comp.ReadFileLinesToStrArr(CS_In, CS_A1_In)){
      cout << "MReverse::main: ERROR: ReadFileLinesToStrArr(" << CS_In << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    delete(P_CS_FN);
    P_CS_FN = CS_Out.SubString(1);
    CS_Out.Set(*P_CS_FN);
    if (!CS_Comp.ReadFileLinesToStrArr(CS_Out, CS_A1_Out)){
      cout << "MReverse::main: ERROR: ReadFileLinesToStrArr(" << CS_Out << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    delete(P_CS_FN);
    if (CS_A1_In.size() != CS_A1_Out.size()){
      cout << "MReverse::main: ERROR: FileList must have same number of elements" << endl;
      exit(EXIT_FAILURE);
    }
  }
  
  CFits F_Image;
  for (int i_file=0; i_file<CS_A1_In.size(); i_file++){
    cout << "MReverse::main: Starting F_Image.SetFileName(" << CS_A1_In(i_file).Get() << ")" << endl;
    if (!F_Image.SetFileName(CS_A1_In(i_file)))
    {
      cout << "MReverse::main: ERROR: F_Image.SetFileName(" << CS_A1_In(i_file).Get() << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
  
    /// Read FitsFile
    cout << "MReverse::main: Starting F_Image.ReadArray()" << endl;
    if (!F_Image.ReadArray())
    {
      cout << "MReverse::main: ERROR: F_Image.ReadArray() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    Array<double, 2> D_A2_Arr(F_Image.GetNRows(), F_Image.GetNCols());
    D_A2_Arr = F_Image.GetPixArray();
    if (D_A2_Arr.rows() > 1)
      cout << "MReverse::main: Before: D_A2_Arr(0:10, *) = " << D_A2_Arr(Range(0,10), Range::all()) << endl;
    else
      cout << "MReverse::main: Before: D_A2_Arr(*,0:10) = " << D_A2_Arr(Range::all(), Range(0,10)) << endl;
    F_Image.Reverse(D_A2_Arr);
    if (D_A2_Arr.rows() > 1)
      cout << "MReverse::main: After: D_A2_Arr(0:10, *) = " << D_A2_Arr(Range(0,10), Range::all()) << endl;
    else
      cout << "MReverse::main: Before: D_A2_Arr(*,0:10) = " << D_A2_Arr(Range::all(), Range(0,10)) << endl;
    F_Image.GetPixArray() = D_A2_Arr;
  
    cout << "MReverse::main: Starting F_Image.SetFileName(" << CS_A1_Out(i_file).Get() << ")" << endl;
    if (!F_Image.SetFileName(CS_A1_Out(i_file)))
    {
      cout << "MReverse::main: ERROR: F_Image.SetFileName(" << CS_A1_Out(i_file).Get() << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
  
    ///write output file
    if (!F_Image.WriteArray()){
      cout << "MReverse::main: ERROR: WriteArray() returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
  }
  /// clean up

  return EXIT_SUCCESS;
}

