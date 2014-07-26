/*
author: Andreas Ritter
created: 05/08/2012
last edited: 05/08/2012
compiler: g++ 4.4
basis machine: Arch Linux
*/

#include "MImArith.h"

using namespace std;

int main(int argc, char *argv[])
{
  cout << "MImArith::main: argc = " << argc << endl;
  if (argc < 5)
  {
    cout << "MImArith::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: imdivide <char[] [@]FitsFileName_Op1> <char Op> <char[] [@]FitsFileName_ArithBy_Op2> <char[] [@]FitsFileName_Out>" << endl;
    cout << "MImArith::main: possible operators: '+' '-' '*' '/' 'sqrt'" << endl;
    cout << "MImArith::main: if operator == 'sqrt' then <char[] [@]FitsFileName_ArithBy_Op2> is ignored, but must be present" << endl;
    exit(EXIT_FAILURE);
  }

  CString CS_Op1_In((char*)argv[1]);
  cout << "MImArith::main: CS_Op1_In set to " << CS_Op1_In << endl;

  CString CS_Op((char*)argv[2]);

  CString CS_Op2_In = (char*)argv[3];
  cout << "MImArith::main: CS_Op2_In set to " << CS_Op2_In << endl;

  CString CS_Out = (char*)argv[4];
  cout << "MImArith::main: CS_Out set to " << CS_Out << endl;

  /// read parameters
  Array<CString, 1> CS_A1_Op1_In(1);
  CS_A1_Op1_In(0) = CS_Op1_In;

  Array<CString, 1> CS_A1_Op2_In(1);
  CS_A1_Op2_In(0) = CS_Op2_In;
  CString CS_Op2_In_Temp(" ");

  Array<CString, 1> CS_A1_Out(1);
  CS_A1_Out(0) = CS_Out;

  CString CS_Comp("@");
  CString *P_CS_Temp = CS_Op1_In.SubString(0,0);
  CString *P_CS_FN;
  double D_Op2 = 0.;
  bool B_Op2_IsNumber = false;
  if (P_CS_Temp->EqualValue(CS_Comp)){
    P_CS_FN = CS_Op1_In.SubString(1);
    CS_Op1_In.Set(*P_CS_FN);
    if (!CS_Comp.ReadFileLinesToStrArr(CS_Op1_In, CS_A1_Op1_In)){
      cout << "MImArith::main: ERROR: ReadFileLinesToStrArr(" << CS_Op1_In << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    delete(P_CS_FN);
    P_CS_FN = CS_Op2_In.SubString(1);
    CS_Op2_In_Temp.Set(*P_CS_FN);
    if (!CS_Comp.ReadFileLinesToStrArr(CS_Op2_In_Temp, CS_A1_Op2_In)){
      cout << "MImArith::main: Warning: ReadFileLinesToStrArr(" << CS_Op2_In_Temp << ") returned FALSE" << endl;
      B_Op2_IsNumber = CS_Op2_In.AToD(D_Op2);
      if (!B_Op2_IsNumber){
	CS_A1_Op2_In.resize(1);
	CS_A1_Op2_In(0) = CS_Op2_In;
        cout << "MImArith::main: B_Op2_IsNumber == FALSE => CS_A1_Op2_In set to " << CS_A1_Op2_In << endl;
      }
      else{
        cout << "MImArith::main: B_Op2_IsNumber == TRUE => D_Op2 set to " << D_Op2 << endl;
      }
    }
    else{
      if (CS_A1_Op1_In.size() != CS_A1_Op2_In.size()){
        cout << "MImArith::main: ERROR: FileList must have same number of elements" << endl;
        exit(EXIT_FAILURE);
      }
    }
    delete(P_CS_FN);
    P_CS_FN = CS_Out.SubString(1);
    CS_Out.Set(*P_CS_FN);
    if (!CS_Comp.ReadFileLinesToStrArr(CS_Out, CS_A1_Out)){
      cout << "MImArith::main: ERROR: ReadFileLinesToStrArr(" << CS_Out << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    delete(P_CS_FN);
    if (CS_A1_Op1_In.size() != CS_A1_Out.size()){
      cout << "MImArith::main: ERROR: FileList must have same number of elements" << endl;
      exit(EXIT_FAILURE);
    }
  }
  else{
    B_Op2_IsNumber = CS_Op2_In.AToD(D_Op2);
    if (B_Op2_IsNumber){
      cout << "MImArith::main: B_Op2_IsNumber == TRUE => D_Op2 set to " << D_Op2 << endl;
    }
  }

  Array<CString, 1> CS_A1_Op(6);
  CS_A1_Op(0) = CString("+");
  CS_A1_Op(1) = CString("-");
  CS_A1_Op(2) = CString("*");
  CS_A1_Op(3) = CString("/");
  CS_A1_Op(4) = CString("sqrt");

  CFits F_Image;
  CFits F_Image2;
  for (int i_file=0; i_file<CS_A1_Op1_In.size(); i_file++){
    cout << "MImArith::main: Starting F_Image.SetFileName(" << CS_A1_Op1_In(i_file).Get() << ")" << endl;
    if (!F_Image.SetFileName(CS_A1_Op1_In(i_file)))
    {
      cout << "MImArith::main: ERROR: F_Image.SetFileName(" << CS_A1_Op1_In(i_file).Get() << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    if (!B_Op2_IsNumber){
      if (CS_A1_Op2_In.size() == 1){
        cout << "MImArith::main: Starting F_Image2.SetFileName(" << CS_A1_Op2_In(0).Get() << ")" << endl;
        if (!F_Image2.SetFileName(CS_A1_Op2_In(0)))
        {
          cout << "MImArith::main: ERROR: F_Image2.SetFileName(" << CS_A1_Op2_In(0).Get() << ") returned FALSE!" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MImArith::main: Starting F_Image2.SetFileName(" << CS_A1_Op2_In(i_file).Get() << ")" << endl;
        if (!F_Image2.SetFileName(CS_A1_Op2_In(i_file)))
        {
          cout << "MImArith::main: ERROR: F_Image2.SetFileName(" << CS_A1_Op2_In(i_file).Get() << ") returned FALSE!" << endl;
          exit(EXIT_FAILURE);
        }
      }
    }

    /// Read FitsFile
    cout << "MImArith::main: Starting F_Image.ReadArray()" << endl;
    if (!F_Image.ReadArray())
    {
      cout << "MImArith::main: ERROR: F_Image.ReadArray() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    if (!B_Op2_IsNumber){
      cout << "MImArith::main: Starting F_Image2.ReadArray()" << endl;
      if (!F_Image2.ReadArray())
      {
        cout << "MImArith::main: ERROR: F_Image2.ReadArray() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }

      /// Check both files for same size
      if ((F_Image.GetNRows() != F_Image2.GetNRows()) || (F_Image.GetNCols() != F_Image2.GetNCols())){
        cout << "MImArith::main: ERROR: files do not have the same size!" << endl;
        exit(EXIT_FAILURE);
      }
    }

    /// Write aperture header information
    F_Image.WriteApHead(CString("aphead_")+CS_A1_Op1_In(i_file)+CString(".head"));

    if (CS_Op.EqualValue(CS_A1_Op(0))){/// "+"
      if (B_Op2_IsNumber)
	F_Image.GetPixArray() = F_Image.GetPixArray() + D_Op2;
      else
        F_Image.GetPixArray() = F_Image.GetPixArray() + F_Image2.GetPixArray();
    }
    else if (CS_Op.EqualValue(CS_A1_Op(1))){/// "-"
      if (B_Op2_IsNumber)
	F_Image.GetPixArray() = F_Image.GetPixArray() - D_Op2;
      else
        F_Image.GetPixArray() = F_Image.GetPixArray() - F_Image2.GetPixArray();
    }
    else if (CS_Op.EqualValue(CS_A1_Op(2))){/// "*"
      if (B_Op2_IsNumber)
	F_Image.GetPixArray() = F_Image.GetPixArray() * D_Op2;
      else
        F_Image.GetPixArray() = F_Image.GetPixArray() * F_Image2.GetPixArray();
    }
    else if (CS_Op.EqualValue(CS_A1_Op(3))){/// "/"
      if (B_Op2_IsNumber){
	if (fabs(D_Op2) < 0.000000001){
	  cout << "MImArith::main: ERROR: D_Op2 == 0 => Returning FALSE" << endl;
	  exit(EXIT_FAILURE);
	}
	else{
          F_Image.GetPixArray() = F_Image.GetPixArray() / D_Op2;
	}
      }
      else{
	Array<double, 2> D_A2_Op1(F_Image.GetNRows(), F_Image.GetNCols());
	D_A2_Op1 = F_Image.GetPixArray();
	Array<double, 2> D_A2_Op2(F_Image.GetNRows(), F_Image.GetNCols());
	D_A2_Op2 = F_Image2.GetPixArray();
	for (int i_row = 0; i_row < F_Image.GetNRows(); i_row++){
	  for (int i_col = 0; i_col < F_Image.GetNCols(); i_col++){
	    if (fabs(D_A2_Op2(i_row, i_col)) < 0.0000001)
	      D_A2_Op1(i_row, i_col) = 0.;
	    else
	      D_A2_Op1(i_row, i_col) = D_A2_Op1(i_row, i_col) / D_A2_Op2(i_row, i_col);
	  }
	}
        F_Image.GetPixArray() = D_A2_Op1;
      }
    }
    else{/// "sqrt'
      Array<double, 2> D_A2_Op1(F_Image.GetNRows(), F_Image.GetNCols());
      D_A2_Op1 = F_Image.GetPixArray();
      F_Image.GetPixArray() = sqrt(D_A2_Op1);
    }

    cout << "MImArith::main: Starting F_Image.SetFileName(" << CS_A1_Out(i_file).Get() << ")" << endl;
    if (!F_Image.SetFileName(CS_A1_Out(i_file)))
    {
      cout << "MImArith::main: ERROR: F_Image.SetFileName(" << CS_A1_Out(i_file).Get() << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    ///write output file
    if (!F_Image.WriteArray()){
      cout << "MImArith::main: ERROR: WriteArray() returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
  }
  /// clean up

  return EXIT_SUCCESS;
}

