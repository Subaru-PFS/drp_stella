/*
author: Andreas Ritter
created: 05/08/2012
last edited: 05/08/2012
compiler: g++ 4.4
basis machine: Arch Linux
*/

#include "MSArith.h"

using namespace std;

int main(int argc, char *argv[])
{
  cout << "MSArith::main: argc = " << argc << endl;
  if (argc < 5)
  {
    cout << "MSArith::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: sarith <char[] [@]FitsFileName_Op1> <char Op> <char[] [@]FitsFileName_ArithBy_Op2> <char[] [@]FitsFileName_Out> [COLUMN=<int column_to_apply_operation>]" << endl;
    cout << "MSArith::main: Note that the Op1 will get re-binned to the same wavelength as Op2" << endl;
    cout << "MSArith::main: possible operators: '+' '-' '*' '/' 'sqrt' 'sum'" << endl;
    cout << "MSArith::main: if operator == 'sqrt' or 'sum' then <char[] [@]FitsFileName_ArithBy_Op2> is ignored, but must be present" << endl;
    cout << "MSArith::main: if operator == 'sum' then all files in @FitsFileName_Op1 will be summed up. In this case all input spectra will be re-binned to the same wavelength as the first spectrum" << endl;
    cout << "MSArith::main: if keyword COLUMN is set then the mathematical operation will only be applied to this column" << endl;
    exit(EXIT_FAILURE);
  }
  
  CFits F_Image;
  bool B_Op2_IsNumber = false;
  
  /// read input parameters to CStrings
  CString CS_Op1_In((char*)argv[1]);
  cout << "MSArith::main: CS_Op1_In set to " << CS_Op1_In << endl;
  Array<CString, 1> CS_A1_Op1(1);
  CS_A1_Op1 = CS_Op1_In;
  if (CS_Op1_In.IsList()){
    if (!CS_Op1_In.ReadFileLinesToStrArr(CS_Op1_In, CS_A1_Op1)){
      cout << "MSArith::main: ERROR: ReadFileLinesToStrArr(" << CS_Op1_In << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
  }

  CString CS_Op((char*)argv[2]);

  CString CS_Op2_In = (char*)argv[3];
  cout << "MSArith::main: CS_Op2_In set to " << CS_Op2_In << endl;
  Array<CString, 1> CS_A1_Op2(CS_A1_Op1.size());
  CS_A1_Op2 = CS_Op2_In;
  if (CS_Op2_In.IsList()){
    if (!CS_Op1_In.ReadFileLinesToStrArr(CS_Op2_In, CS_A1_Op2)){
      cout << "MSArith::main: ERROR: ReadFileLinesToStrArr(" << CS_Op2_In << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    if (CS_A1_Op1.size() != CS_A1_Op2.size()){
      cout << "MSArith::main: ERROR: CS_A1_Op1.size() != CS_A1_Op2.size()" << endl;
      exit(EXIT_FAILURE);
    }
  }

  CString CS_Out = (char*)argv[4];
  cout << "MSArith::main: CS_Out set to " << CS_Out << endl;
  Array<CString, 1> CS_A1_Out(CS_A1_Op1.size());
  CS_A1_Out = CS_Out;
  if (CS_Out.IsList()){
    if (!CS_Out.ReadFileLinesToStrArr(CS_Out, CS_A1_Out)){
      cout << "MSArith::main: ERROR: ReadFileLinesToStrArr(" << CS_Out << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    if (CS_A1_Op1.size() != CS_A1_Out.size()){
      cout << "MSArith::main: ERROR: CS_A1_Op1.size() != CS_A1_Out.size()" << endl;
      exit(EXIT_FAILURE);
    }
  }
  
  int I_Col = 1;
  CString CS(" ");
  CString CS_Comp(" ");
  CString *P_CS;
  bool B_ColumnSet = false;
  for (int i_par=5; i_par <= argc; i_par++){
    CS.Set((char*)(argv[i_par]));
    CS_Comp.Set("COLUMN");
    P_CS = CS.SubString(0,CS_Comp.GetLength()-1);
    if (P_CS->EqualValue(CS_Comp)){
      delete(P_CS);
      P_CS = CS.SubString(CS_Comp.GetLength());
      if(!P_CS->AToI(I_Col)){
        cout << "MSArith::main: ERROR: AToI(" << *P_CS << ") returned FALSE" << endl;
        delete(P_CS);
        exit(EXIT_FAILURE);
      }
      delete(P_CS);
      cout << "MSArith::main: KeyWord 'COLUMN' set: I_Col set to " << I_Col << endl;
      B_ColumnSet = true;
    }
  }

  Array<CString, 1> CS_A1_Op(6);
  CS_A1_Op(0) = CString("+");
  CS_A1_Op(1) = CString("-");
  CS_A1_Op(2) = CString("*");
  CS_A1_Op(3) = CString("/");
  CS_A1_Op(4) = CString("sqrt");
  CS_A1_Op(5) = CString("sum");
  
  Array<double, 2> D_A2_Spec1(2,2);
  Array<double, 2> D_A2_Spec2(2,2);
  Array<double, 1> D_A1_X1(D_A2_Spec1.rows());
  Array<double, 1> D_A1_Y1(D_A2_Spec1.rows());
  Array<double, 1> D_A1_X2(D_A2_Spec2.rows());
  Array<double, 1> D_A1_Y2(D_A2_Spec2.rows());
  Array<double, 1> *P_D_A1_Out = new Array<double, 1>(2);
  Array<double, 2> D_A2_SpecOut(2, 2);
  double D_Op2 = 0.;
  for (int i_file=0; i_file<CS_A1_Op1.size(); i_file++){
    if (!CS_Out.ReadFileToDblArr(CS_A1_Op1(i_file),
                                 D_A2_Spec1,
                                 CString(" "))){
      cout << "MSArith::main: ERROR: Op1: ReadFileToDblArr(" << CS_A1_Op1(i_file) << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }

    if (!(CS_Op.EqualValue(CS_A1_Op(4))) && !(CS_Op.EqualValue(CS_A1_Op(5)))){
      if (!CS_Out.ReadFileToDblArr(CS_A1_Op2(i_file),
                                   D_A2_Spec2,
                                   CString(" "))){
        cout << "MSArith::main: ERROR: ReadFileToDblArr(" << CS_A1_Op2(i_file) << ") returned FALSE" << endl;
        B_Op2_IsNumber = CS_A1_Op2(i_file).AToD(D_Op2);
        if (!B_Op2_IsNumber){
          cout << "MSArith::main: ERROR: B_Op2_IsNumber == FALSE" << endl;
          exit(EXIT_FAILURE);
        }
        else{
          cout << "MSArith::main: B_Op2_IsNumber == TRUE => D_Op2 set to " << D_Op2 << endl;
        }
      }

      D_A1_X1.resize(D_A2_Spec1.rows());
      D_A1_X1 = D_A2_Spec1(Range::all(), 0);
      D_A1_Y1.resize(D_A2_Spec1.rows());
      D_A1_Y1 = D_A2_Spec1(Range::all(), I_Col);

      if (!B_Op2_IsNumber){
        D_A1_X2.resize(D_A2_Spec2.rows());
        D_A1_X2 = D_A2_Spec2(Range::all(), 0);

        if (!F_Image.InterPol(D_A1_Y1,
                              D_A1_X1,
                              D_A1_X2,
                              P_D_A1_Out)){
          cout << "MSArith::main: ERROR: InterPol returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }

        D_A2_Spec1.resize(D_A1_X2.size(),2);
        D_A2_Spec1(Range::all(), 0) = D_A1_X2;
        D_A2_Spec1(Range::all(), 1) = *P_D_A1_Out;

        cout << "D_A2_Spec1 = " << D_A2_Spec1 << endl;
        cout << "D_A2_Spec2 = " << D_A2_Spec2 << endl;
        
        D_A2_SpecOut.resize(D_A2_Spec1.rows(), D_A2_Spec1.cols());
        D_A2_SpecOut = D_A2_Spec1;

  /**
   *   CS_A1_Op(0) = CString("+");
       CS_A1_Op(1) = CString("-");
       CS_A1_Op(2) = CString("*");
       CS_A1_Op(3) = CString("/");
       CS_A1_Op(4) = CString("sqrt");
       CS_A1_Op(5) = CString("sum");
       **/

        if (CS_Op.EqualValue(CS_A1_Op(0))){/// +
          D_A2_SpecOut(Range::all(), I_Col) = D_A2_Spec1(Range::all(), I_Col) + D_A2_Spec2(Range::all(), I_Col);
        } else if (CS_Op.EqualValue(CS_A1_Op(1))){/// -
          D_A2_SpecOut(Range::all(), I_Col) = D_A2_Spec1(Range::all(), I_Col) - D_A2_Spec2(Range::all(), I_Col);
        } else if (CS_Op.EqualValue(CS_A1_Op(2))){/// *
          D_A2_SpecOut(Range::all(), I_Col) = D_A2_Spec1(Range::all(), I_Col) * D_A2_Spec2(Range::all(), I_Col);
        } else if (CS_Op.EqualValue(CS_A1_Op(3))){/// /
          D_A2_SpecOut(Range::all(), I_Col) = D_A2_Spec1(Range::all(), I_Col) / D_A2_Spec2(Range::all(), I_Col);
        }
      }
      else{
        D_A2_SpecOut.resize(D_A2_Spec1.rows(), D_A2_Spec1.cols());
        D_A2_SpecOut = D_A2_Spec1;
        if (CS_Op.EqualValue(CS_A1_Op(0))){/// +
          D_A2_SpecOut(Range::all(), I_Col) = D_A2_Spec1(Range::all(), I_Col) + D_Op2;
        } else if (CS_Op.EqualValue(CS_A1_Op(1))){/// -
          D_A2_SpecOut(Range::all(), I_Col) = D_A2_Spec1(Range::all(), I_Col) - D_Op2;
        } else if (CS_Op.EqualValue(CS_A1_Op(2))){/// *
          D_A2_SpecOut(Range::all(), I_Col) = D_A2_Spec1(Range::all(), I_Col) * D_Op2;
        } else if (CS_Op.EqualValue(CS_A1_Op(3))){/// /
          D_A2_SpecOut(Range::all(), I_Col) = D_A2_Spec1(Range::all(), I_Col) / D_Op2;
        }
      }
      if (!F_Image.WriteArrayToFile(D_A2_SpecOut, CS_A1_Out(i_file), CString("ascii"))){
        cout << "MSArith::main: ERROR: WriteArrayToFile returned FALSE" << endl;
        exit(EXIT_FAILURE);
      }
    }
    else{  
      if (CS_Op.EqualValue(CS_A1_Op(4))){/// sqrt
        D_A2_SpecOut.resize(D_A2_Spec1.rows(), D_A2_Spec1.cols());
        D_A2_SpecOut(Range::all(), 0) = D_A2_Spec1(Range::all(), 0);
        D_A2_SpecOut(Range::all(), I_Col) = sqrt(D_A2_Spec1(Range::all(), I_Col));
        if (!F_Image.WriteArrayToFile(D_A2_SpecOut, CS_A1_Out(i_file), CString("ascii"))){
          cout << "MSArith::main: ERROR: WriteArrayToFile returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      } 
      else if (CS_Op.EqualValue(CS_A1_Op(5))){/// sum
        if (i_file == 0){
          D_A2_SpecOut.resize(D_A2_Spec1.rows(), D_A2_Spec1.cols());
          D_A2_SpecOut = D_A2_Spec1;
          if (B_ColumnSet)
            D_A2_SpecOut(Range::all(), I_Col) = 0.;
          else
            D_A2_SpecOut(Range::all(), Range(1, D_A2_SpecOut.cols()-1)) = 0.;
          
          D_A1_X1.resize(D_A2_Spec1.rows());
          D_A1_X1 = D_A2_Spec1(Range::all(), 0);
        }
        else{
          D_A1_X2.resize(D_A2_Spec1.rows());
          D_A1_X2 = D_A2_Spec1(Range::all(), 0);
          D_A1_Y2.resize(D_A2_Spec1.rows());
          D_A1_Y2 = D_A2_Spec1(Range::all(), I_Col);
          
          if (!F_Image.InterPol(D_A1_Y2,
                                D_A1_X2,
                                D_A1_X1,
                                P_D_A1_Out)){
            cout << "MSArith::main: ERROR: InterPol returned FALSE" << endl;
            exit(EXIT_FAILURE);
          }
              
          D_A2_Spec1.resize(D_A1_X1.size(),2);
          D_A2_Spec1(Range::all(), 0) = D_A1_X1;
          D_A2_Spec1(Range::all(), 1) = *P_D_A1_Out;
          cout << "D_A2_Spec1(0,*) = " << D_A2_Spec1(0,Range::all()) << endl;
          
          if (D_A2_Spec1.rows() != D_A2_SpecOut.rows()){
            cout << "MSArith::main: ERROR: i_file = " << i_file << ": D_A2_Spec1.rows() != D_A2_SpecOut.rows()" << endl;
            exit(EXIT_FAILURE);
          }
        }
        if (D_A2_Spec1.cols() != D_A2_SpecOut.cols()){
          cout << "MSArith::main: ERROR: i_file = " << i_file << ": D_A2_Spec1.cols() != D_A2_SpecOut.cols()" << endl;
          exit(EXIT_FAILURE);
        }
        if (B_ColumnSet)
          D_A2_SpecOut(Range::all(), I_Col) = D_A2_SpecOut(Range::all(), I_Col) + D_A2_Spec1(Range::all(), I_Col);
        else{
          D_A2_SpecOut(Range::all(), Range(1,D_A2_SpecOut.cols()-1)) = D_A2_SpecOut(Range::all(), Range(1,D_A2_SpecOut.cols()-1)) + D_A2_Spec1(Range::all(), Range(1,D_A2_SpecOut.cols()-1));
          cout << "i_file = " << i_file << ": D_A2_SpecOut(0,*) = " << D_A2_SpecOut(0,Range::all()) << endl;
        }
      }
    }
  }
  if (CS_Op.EqualValue(CS_A1_Op(5))){/// sum
    if (!F_Image.WriteArrayToFile(D_A2_SpecOut, CS_Out, CString("ascii"))){
      cout << "MSArith::main: ERROR: WriteArrayToFile returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
  }
  /// clean up
  delete(P_D_A1_Out);

  return EXIT_SUCCESS;
}

