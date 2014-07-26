/*
author: Andreas Ritter
created: 05/12/2013
last edited: 05/12/2013
compiler: g++ 4.8
basis machine: Arch Linux
*/

#include "MXCor2D.h"

using namespace std;

int main(int argc, char *argv[])
{
  cout << "MXCor2D::main: argc = " << argc << endl;
  if (argc < 5)
  {
    cout << "MXCor2D::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: xcor2d <char[] FitsFileNameA_In> <char[] FitsFileNameB_In> <int NPixShiftMax_X_In> <int NPixShiftMax_Y_In> [FILE_A_OUT=<char FitsFileNameA_Out>] [FILE_B_OUT=<char FitsFileNameB_Out>] [A_MINUS_B_OUT=<char FitsFileNameDiff_Out>] [AREA=[int(xmin),int(xmax),int(ymin),int(ymax)]]" << endl;
    exit(EXIT_FAILURE);
  }
  char *P_CharArrA_In = (char*)argv[1];
  char *P_CharArrB_In = (char*)argv[2];
  int I_NPixShiftMax_X_In = (int)(atoi((char*)argv[3]));
  int I_NPixShiftMax_Y_In = (int)(atoi((char*)argv[4]));
  Array<int, 1> I_A1_Area(4);
  I_A1_Area = 0;

  /// read optional parameters
  CString CS(" ");
  CString CS_Comp(" ");
  CString *P_CS_Temp;
  CString *P_CS_TempA;
  CString CS_FitsFileNameA_Out(" ");
  CString CS_FitsFileNameB_Out(" ");
  CString CS_FitsFileNameDiff_Out(" ");
  for (int i = 5; i <= argc; i++){
    CS.Set((char*)argv[i]);
    cout << "MPrepareCollapsedImageNoScatter: Reading Parameter " << CS << endl;

    CS_Comp.Set("FILE_A_OUT");
    if (CS.GetLength() > CS_Comp.GetLength()){
      P_CS_Temp = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MPrepareCollapsedImageNoScatter::main: *P_CS_Temp set to " << *P_CS_Temp << endl;
      if (P_CS_Temp->EqualValue(CS_Comp)){
        P_CS_TempA = CS.SubString(CS_Comp.GetLength()+1);
        CS_FitsFileNameA_Out.Set(*P_CS_TempA);
        delete(P_CS_TempA);
        cout << "MPrepareCollapsedImageNoScatter::main: CS_FitsFileNameA_Out set to " << CS_FitsFileNameA_Out << endl;
      }
      delete(P_CS_Temp);
    }

    CS_Comp.Set("FILE_B_OUT");
    if (CS.GetLength() > CS_Comp.GetLength()){
      P_CS_Temp = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MPrepareCollapsedImageNoScatter::main: *P_CS_Temp set to " << *P_CS_Temp << endl;
      if (P_CS_Temp->EqualValue(CS_Comp)){
        P_CS_TempA = CS.SubString(CS_Comp.GetLength()+1);
        CS_FitsFileNameB_Out.Set(*P_CS_TempA);
        delete(P_CS_TempA);
        cout << "MPrepareCollapsedImageNoScatter::main: CS_FitsFileNameB_Out set to " << CS_FitsFileNameB_Out << endl;
      }
      delete(P_CS_Temp);
    }

    CS_Comp.Set("A_MINUS_B_OUT");
    if (CS.GetLength() > CS_Comp.GetLength()){
      P_CS_Temp = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MPrepareCollapsedImageNoScatter::main: *P_CS_Temp set to " << *P_CS_Temp << endl;
      if (P_CS_Temp->EqualValue(CS_Comp)){
        P_CS_TempA = CS.SubString(CS_Comp.GetLength()+1);
        CS_FitsFileNameDiff_Out.Set(*P_CS_TempA);
        delete(P_CS_TempA);
        cout << "MPrepareCollapsedImageNoScatter::main: CS_FitsFileNameDiff_Out set to " << CS_FitsFileNameDiff_Out << endl;
      }
      delete(P_CS_Temp);
    }

    /// AREA
    CS_Comp.Set("AREA");
    if (CS.GetLength() > CS_Comp.GetLength()){
      P_CS_Temp = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MGaussExtract_Obs_Sky::main: *P_CS_Temp set to " << *P_CS_Temp << endl;
      if (P_CS_Temp->EqualValue(CS_Comp)){
        CString cs_temp;
        cs_temp.Set(",");
        int i_pos_a = CS_Comp.GetLength()+2;
        int i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        cout << "MGaussExtract_Obs_Sky: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
        delete(P_CS_Temp);
        P_CS_Temp = CS.SubString(i_pos_a,i_pos_b-1);
        cout << "MGaussExtract_Obs_Sky: P_CS_Temp set to " << *P_CS_Temp << endl;
        I_A1_Area(0) = (int)(atoi(P_CS_Temp->Get()));
        cout << "MGaussExtract_Obs_Sky: I_A1_Area(0) set to " << I_A1_Area(0) << endl;

        i_pos_a = i_pos_b+1;
        i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        cout << "MGaussExtract_Obs_Sky: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
        delete(P_CS_Temp);
        P_CS_Temp = CS.SubString(i_pos_a,i_pos_b-1);
        cout << "MGaussExtract_Obs_Sky: P_CS_Temp set to " << *P_CS_Temp << endl;
        I_A1_Area(1) = (int)(atoi(P_CS_Temp->Get()));
        cout << "MGaussExtract_Obs_Sky: I_A1_Area(1) set to " << I_A1_Area(1) << endl;

        i_pos_a = i_pos_b+1;
        i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        cout << "MGaussExtract_Obs_Sky: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
        delete(P_CS_Temp);
        P_CS_Temp = CS.SubString(i_pos_a,i_pos_b-1);
        cout << "MGaussExtract_Obs_Sky: P_CS_Temp set to " << *P_CS_Temp << endl;
        I_A1_Area(2) = (int)(atoi(P_CS_Temp->Get()));
        cout << "MGaussExtract_Obs_Sky: I_A1_Area(2) set to " << I_A1_Area(2) << endl;

        i_pos_a = i_pos_b+1;
        cs_temp.Set("]");
        i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        if (i_pos_b < 0){
          cs_temp.Set(")");
          i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        }
        cout << "MGaussExtract_Obs_Sky: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
        delete(P_CS_Temp);
        P_CS_Temp = CS.SubString(i_pos_a,i_pos_b-1);
        cout << "MGaussExtract_Obs_Sky: P_CS_Temp set to " << *P_CS_Temp << endl;
        I_A1_Area(3) = (int)(atoi(P_CS_Temp->Get()));
        cout << "MGaussExtract_Obs_Sky: I_A1_Area(3) set to " << I_A1_Area(3) << endl;
        delete(P_CS_Temp);
      }
    }
  }

  time_t seconds;
  CString CS_FitsFileNameA_In;
  CS_FitsFileNameA_In.Set(P_CharArrA_In);
  CString CS_FitsFileNameB_In;
  CS_FitsFileNameB_In.Set(P_CharArrB_In);

  CFits F_ImageA;
  cout << "MXCor2D::main: Starting F_ImageA.SetFileName(" << CS_FitsFileNameA_In.Get() << ")" << endl;
  if (!F_ImageA.SetFileName(CS_FitsFileNameA_In))
  {
    cout << "MXCor2D::main: ERROR: F_ImageA.SetFileName(" << CS_FitsFileNameA_In.Get() << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Read FitsFile
  cout << "MXCor2D::main: Starting F_ImageA.ReadArray()" << endl;
  if (!F_ImageA.ReadArray())
  {
    cout << "MXCor2D::main: ERROR: F_ImageA.ReadArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  Array<double, 2> D_A2_PixArrA(F_ImageA.GetNRows(), F_ImageA.GetNCols());
  if (I_A1_Area(1) > 0){
    D_A2_PixArrA.resize(I_A1_Area(3) - I_A1_Area(2) + 1, I_A1_Area(1) - I_A1_Area(0) + 1);
    D_A2_PixArrA = F_ImageA.GetPixArray()(Range(I_A1_Area(2), I_A1_Area(3)), Range(I_A1_Area(0), I_A1_Area(1)));
  }
  else{
    D_A2_PixArrA = F_ImageA.GetPixArray();
  }


  CFits F_ImageB;
  cout << "MXCor2D::main: Starting F_ImageB.SetFileName(" << CS_FitsFileNameB_In.Get() << ")" << endl;
  if (!F_ImageB.SetFileName(CS_FitsFileNameB_In))
  {
    cout << "MXCor2D::main: ERROR: F_ImageB.SetFileName(" << CS_FitsFileNameB_In.Get() << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Read FitsFile
  cout << "MXCor2D::main: Starting F_ImageB.ReadArray()" << endl;
  if (!F_ImageB.ReadArray())
  {
    cout << "MXCor2D::main: ERROR: F_ImageB.ReadArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  Array<double, 2> D_A2_PixArrB(F_ImageB.GetNRows(), F_ImageB.GetNCols());
  if (I_A1_Area(1) > 0){
    D_A2_PixArrB.resize(I_A1_Area(3) - I_A1_Area(2) + 1, I_A1_Area(1) - I_A1_Area(0) + 1);
    D_A2_PixArrB = F_ImageB.GetPixArray()(Range(I_A1_Area(2), I_A1_Area(3)), Range(I_A1_Area(0), I_A1_Area(1)));
  }
  else{
    D_A2_PixArrB = F_ImageB.GetPixArray();
  }

  if ((D_A2_PixArrA.rows() != D_A2_PixArrB.rows()) || (D_A2_PixArrA.cols() != D_A2_PixArrB.cols())){
    cout << "MXCor2D::main: ERROR: Images have different dimensions" << endl;
    exit(EXIT_FAILURE);
  }

  Array<double, 2> D_A2_Moving(D_A2_PixArrA.rows(), D_A2_PixArrA.cols());
  D_A2_Moving = D_A2_PixArrA;

  Array<double, 2> D_A2_Static(D_A2_PixArrB.rows()-(2*I_NPixShiftMax_Y_In), D_A2_PixArrB.cols()-(2*I_NPixShiftMax_X_In));
  if ((D_A2_PixArrB.rows() - 2*I_NPixShiftMax_Y_In) != D_A2_Static.rows()){
    cout << "MXCor2D::main: ERROR: Something went wrong" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MXCor2D::main: D_A2_Static.rows() = " << D_A2_Static.rows() << ", D_A2_Static.cols() = " << D_A2_Static.cols() << endl;
  cout << "MXCor2D::main: row Range(" << I_NPixShiftMax_Y_In << ", " << D_A2_PixArrB.rows()-I_NPixShiftMax_Y_In-1 << ")" << endl;
  cout << "MXCor2D::main: col Range(" << I_NPixShiftMax_X_In << ", " << D_A2_PixArrB.cols()-I_NPixShiftMax_X_In-1 << ")" << endl;
//  exit(EXIT_FAILURE);
  D_A2_Static = D_A2_PixArrB(Range(0, D_A2_PixArrB.rows()-2*I_NPixShiftMax_Y_In-1), Range(0, D_A2_PixArrB.cols()-2*I_NPixShiftMax_X_In-1));

  int I_Shift_X = 0;
  int I_Shift_Y = 0;

  if (!F_ImageA.CrossCorrelate2D(D_A2_Static, D_A2_Moving, I_Shift_X, I_Shift_Y)){
    cout << "MXCor2D::main: ERROR: CrossCorrelate2D returned FALSE" << endl;
    exit(EXIT_FAILURE);
  }

  cout << "MXCor2D::main: shift in x = " << I_Shift_X << endl;
  cout << "MXCor2D::main: shift in y = " << I_Shift_Y << endl;

  cout << "MXCor2D::main: Image 1 needs to be shifted by " << I_Shift_X << " pixels to the left and " << I_Shift_Y << " pixels down" << endl;

  if (CS_FitsFileNameA_Out.GetLength() > 2){
    if (I_A1_Area(1) > 0){
      D_A2_PixArrA.resize(F_ImageA.GetNRows(), F_ImageA.GetNCols());
      D_A2_PixArrA = F_ImageA.GetPixArray();
    }
    if (!F_ImageA.SetNCols(F_ImageA.GetNCols()-2*I_Shift_X)){
      cout << "MXCor2D::main: ERROR: F_ImageA.SetNCols(" << F_ImageA.GetNCols() - 2*I_Shift_X << " returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    if (!F_ImageA.SetNRows(F_ImageA.GetNRows()-2*I_Shift_Y)){
      cout << "MXCor2D::main: ERROR: F_ImageA.SetNRows(" << F_ImageA.GetNRows() - 2*I_Shift_Y << " returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    F_ImageA.GetPixArray() = D_A2_PixArrA(Range(I_Shift_Y, D_A2_PixArrA.rows()-I_Shift_Y-1), Range(I_Shift_X, D_A2_PixArrA.cols()-I_Shift_X-1));
    if (!F_ImageA.SetFileName(CS_FitsFileNameA_Out)){
      cout << "MXCor2D::main: ERROR: F_ImageA.SetFileName(" << CS_FitsFileNameA_Out << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    if (!F_ImageA.WriteArray()){
      cout << "MXCor2D::main: ERROR: F_ImageA.WriteArray() returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
  }

  if (CS_FitsFileNameB_Out.GetLength() > 2){
    if (I_A1_Area(1) > 0){
      D_A2_PixArrB.resize(F_ImageB.GetNRows(), F_ImageB.GetNCols());
      D_A2_PixArrB = F_ImageB.GetPixArray();
    }
    if (!F_ImageB.SetNCols(F_ImageB.GetNCols()-2*I_Shift_X)){
      cout << "MXCor2D::main: ERROR: F_ImageB.SetNCols(" << F_ImageB.GetNCols() - 2*I_Shift_X << " returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    if (!F_ImageB.SetNRows(F_ImageB.GetNRows()-2*I_Shift_Y)){
      cout << "MXCor2D::main: ERROR: F_ImageB.SetNRows(" << F_ImageB.GetNRows() - 2*I_Shift_Y << " returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    F_ImageB.GetPixArray() = D_A2_PixArrB(Range(0, D_A2_PixArrB.rows()-2*I_Shift_Y-1), Range(0, D_A2_PixArrB.cols()-2*I_Shift_X-1));
    if (!F_ImageB.SetFileName(CS_FitsFileNameB_Out)){
      cout << "MXCor2D::main: ERROR: F_ImageB.SetFileName(" << CS_FitsFileNameB_Out << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    if (!F_ImageB.WriteArray()){
      cout << "MXCor2D::main: ERROR: F_ImageB.WriteArray() returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
  }

  if (CS_FitsFileNameDiff_Out.GetLength() > 2){
    Array<double, 2> D_A2_Diff(F_ImageA.GetNRows(), F_ImageA.GetNCols());
    D_A2_Diff = F_ImageA.GetPixArray() - F_ImageB.GetPixArray();
    F_ImageB.GetPixArray() = D_A2_Diff;
    if (!F_ImageB.SetFileName(CS_FitsFileNameDiff_Out)){
      cout << "MXCor2D::main: ERROR: F_ImageB.SetFileName(" << CS_FitsFileNameDiff_Out << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    if (!F_ImageB.WriteArray()){
      cout << "MXCor2D::main: ERROR: 2. F_ImageB.WriteArray() returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
  }

  return EXIT_SUCCESS;
}
