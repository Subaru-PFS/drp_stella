/*
author: Andreas Ritter
created: 05/12/2013
last edited: 05/12/2013
compiler: g++ 4.8
basis machine: Arch Linux
*/

#include "MXCor.h"

using namespace std;

int main(int argc, char *argv[])
{
  cout << "MXCor::main: argc = " << argc << endl;
  if (argc < 5)
  {
    cout << "MXCor::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: xcor <char[] [@]FitsFileNameA_In> <char[] [@]FitsFileNameB_In> <double NPixShiftMax_In> <char[] PixelShiftList_Out> [A_MINUS_B_OUT=<char [@]FitsFileNameDiff_Out>]" << endl;
    cout << "NOTE: Possible: FitsFileNameA_In is not a list and FitsFileNameB_In is also not a list" << endl;
    cout << "NOTE: Possible: FitsFileNameA_In is list and FitsFileNameB_In is not" << endl;
    cout << "NOTE: Possible: FitsFileNameA_In is not a list and FitsFileNameB_In is a list" << endl;
    cout << "NOTE: Possible: FitsFileNameA_In is list and FitsFileNameB_In is list, then both lists must have the same number of elements" << endl;
    cout << "NOTE: Input files can be fits files or ascii files" << endl;
    exit(EXIT_FAILURE);
  }
  char *P_CharArrA_In = (char*)argv[1];
  char *P_CharArrB_In = (char*)argv[2];
  int I_NPixShiftMax_In = (int)(atoi((char*)argv[3]));
  char *P_CharArr_Out = (char*)argv[4];
//  int I_NPixShiftMax_Y_In = (int)(atoi((char*)argv[4]));
//  Array<int, 1> I_A1_Area(4);
//  I_A1_Area = 0;

  /// read optional parameters
  CString CS(" ");
  CString CS_Comp(" ");
  CString *P_CS_Temp;
  CString *P_CS_TempA;
  CString CS_FitsFileNameA_Out(" ");
  CString CS_FitsFileNameB_Out(" ");
  CString CS_FitsFileNameDiff_Out(" ");
  CString CS_PixelShiftList_Out(P_CharArr_Out);
  for (int i = 5; i <= argc; i++){
    CS.Set((char*)argv[i]);
    cout << "MPrepareCollapsedImageNoScatter: Reading Parameter " << CS << endl;

  /*  CS_Comp.Set("FILE_A_OUT");
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
*/
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
/*    CS_Comp.Set("AREA");
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
    }*/
  }
  
  Array<CString, 1> CS_A1_FNameDiff_Out(1);
  CS_A1_FNameDiff_Out = CS_FitsFileNameDiff_Out;
  if (CS_FitsFileNameDiff_Out.GetLength() > 2){
    if (CS_FitsFileNameDiff_Out.IsList()){
      P_CS_Temp = CS_FitsFileNameDiff_Out.SubString(1);
      if (!CS_FitsFileNameDiff_Out.ReadFileLinesToStrArr(*P_CS_Temp, CS_A1_FNameDiff_Out)){
        cout << "MXCor::main: ERROR: ReadFileLinesToStrArr(" << *P_CS_Temp << ") returned FALSE" << endl;
        delete(P_CS_Temp);
        exit(EXIT_FAILURE);
      }
      delete(P_CS_Temp);
    }
  }

  time_t seconds;
  CString CS_FitsFileNameA_In;
  CS_FitsFileNameA_In.Set(P_CharArrA_In);
  Array<CString, 1> CS_A1_FitsFileNamesA_In(1);
  CS_A1_FitsFileNamesA_In = CS_FitsFileNameA_In;
  if (CS_FitsFileNameA_In.IsList()){
    CString *P_CS_FN = CS_FitsFileNameA_In.SubString(1);
    if (!CS_FitsFileNameA_In.ReadFileLinesToStrArr(*P_CS_FN, CS_A1_FitsFileNamesA_In)){
      cout << "MXCor::main: ERROR: ReadFileLinesToStrArr(" << *P_CS_FN << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    delete(P_CS_FN);
//    if (!CS_FitsFileNameB_In.IsList()){
  }
  CString CS_FitsFileNameB_In;
  CS_FitsFileNameB_In.Set(P_CharArrB_In);
  Array<CString, 1> CS_A1_FitsFileNamesB_In(1);
  CS_A1_FitsFileNamesB_In = CS_FitsFileNameB_In;
  if (CS_FitsFileNameB_In.IsList()){
    CString *P_CS_FN = CS_FitsFileNameB_In.SubString(1);
    if (!CS_FitsFileNameA_In.ReadFileLinesToStrArr(*P_CS_FN, CS_A1_FitsFileNamesB_In)){
      cout << "MXCor::main: ERROR: ReadFileLinesToStrArr(" << *P_CS_FN << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    delete(P_CS_FN);
  }
  if (CS_FitsFileNameA_In.IsList() && CS_FitsFileNameB_In.IsList()){
    if (CS_A1_FitsFileNamesA_In.size() != CS_A1_FitsFileNamesB_In.size()){
      cout << "MXCor::main: ERROR: Both input files are lists, but have different numbers of elements" << endl;
      exit(EXIT_FAILURE);
    }
  }

  Array<double, 1> D_A1_PixShifts_Out(1);
  int I_NRuns = 1;
  if (CS_FitsFileNameA_In.IsList()){
    D_A1_PixShifts_Out.resize(CS_A1_FitsFileNamesA_In.size());
    I_NRuns = CS_A1_FitsFileNamesA_In.size();
  }
  if (CS_FitsFileNameB_In.IsList()){
    D_A1_PixShifts_Out.resize(CS_A1_FitsFileNamesB_In.size());
    I_NRuns = CS_A1_FitsFileNamesB_In.size();
  }
  D_A1_PixShifts_Out = 0.;
  if (CS_FitsFileNameDiff_Out.GetLength() > 2){
    if (CS_A1_FNameDiff_Out.size() != I_NRuns){
      cout << "MXCor::main: ERROR: CS_A1_FNameDiff_Out.size(=" << CS_A1_FNameDiff_Out.size() << ") != I_NRuns(=" << I_NRuns << ")" << endl;
      exit(EXIT_FAILURE);
    }
  }
  
  CFits F_ImageA;
  CFits F_ImageB;
  CString CS_FName_A(" ");
  CString CS_FName_B(" ");
  CString *P_CS_FNEnd;
  CString CS_Fits("fits");
  Array<double, 1> D_A1_Moving(2);
  Array<double, 1> D_A1_Static(2);
  D_A1_Moving = 0.;
  D_A1_Static = 0.;
  
  for (int i_run=0; i_run < I_NRuns; i_run++){
    if (CS_FitsFileNameA_In.IsList()){
      CS_FName_A.Set(CS_A1_FitsFileNamesA_In(i_run));
    }
    else{
      CS_FName_A.Set(CS_FitsFileNameA_In);
    }
    if (CS_FitsFileNameB_In.IsList()){
      CS_FName_B.Set(CS_A1_FitsFileNamesB_In(i_run));
    }
    else{
      CS_FName_B.Set(CS_FitsFileNameB_In);
    }

    P_CS_FNEnd = CS_FName_A.SubString(CS_FName_A.LastStrPos(CString("."))+1);
    if (P_CS_FNEnd->EqualValue(CS_Fits)){
      
      cout << "MXCor::main: Starting F_ImageA.SetFileName(" << CS_FName_A << ")" << endl;
      if (!F_ImageA.SetFileName(CS_FName_A)){
        cout << "MXCor::main: ERROR: F_ImageA.SetFileName(" << CS_FName_A << ") returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }

      /// Read FitsFile
      cout << "MXCor::main: Starting F_ImageA.ReadArray()" << endl;
      if (!F_ImageA.ReadArray()){
        cout << "MXCor::main: ERROR: F_ImageA.ReadArray() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }

      Array<double, 2> D_A2_PixArrA(F_ImageA.GetNRows(), F_ImageA.GetNCols());
//  if (I_A1_Area(1) > 0){
//    D_A2_PixArrA.resize(I_A1_Area(3) - I_A1_Area(2) + 1, I_A1_Area(1) - I_A1_Area(0) + 1);
//    D_A2_PixArrA = F_ImageA.GetPixArray()(Range(I_A1_Area(2), I_A1_Area(3)), Range(I_A1_Area(0), I_A1_Area(1)));
//  }
//  else{
      D_A2_PixArrA = F_ImageA.GetPixArray();
//  }


      cout << "MXCor::main: Starting F_ImageB.SetFileName(" << CS_FName_B << ")" << endl;
      if (!F_ImageB.SetFileName(CS_FName_B)){
        cout << "MXCor::main: ERROR: F_ImageB.SetFileName(" << CS_FName_B << ") returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }

      /// Read FitsFile
      cout << "MXCor::main: Starting F_ImageB.ReadArray()" << endl;
      if (!F_ImageB.ReadArray()){
        cout << "MXCor::main: ERROR: F_ImageB.ReadArray() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }

      Array<double, 2> D_A2_PixArrB(F_ImageB.GetNRows(), F_ImageB.GetNCols());
//    if (I_A1_Area(1) > 0){
//      D_A2_PixArrB.resize(I_A1_Area(3) - I_A1_Area(2) + 1, I_A1_Area(1) - I_A1_Area(0) + 1);
//    D_A2_PixArrB = F_ImageB.GetPixArray()(Range(I_A1_Area(2), I_A1_Area(3)), Range(I_A1_Area(0), I_A1_Area(1)));
//  }
//  else{
      D_A2_PixArrB = F_ImageB.GetPixArray();
//  }

      if ((D_A2_PixArrA.rows() != D_A2_PixArrB.rows()) || (D_A2_PixArrA.cols() != D_A2_PixArrB.cols())){
        cout << "MXCor::main: ERROR: Images have different dimensions" << endl;
        exit(EXIT_FAILURE);
      }

      D_A1_Moving.resize(D_A2_PixArrA.rows());
      D_A1_Moving = D_A2_PixArrA(Range::all(), 0);

      D_A1_Static.resize(D_A2_PixArrB.rows());
      cout << "MXCor::main: D_A1_Static.size() = " << D_A1_Static.size() << endl;
      D_A1_Static = D_A2_PixArrB(Range::all(), 0);
    }
    else{
      Array<double, 2> D_A2_PixArr(2,2);
      D_A2_PixArr = 0.;
      if (!CS_FName_A.ReadFileToDblArr(CS_FName_A, D_A2_PixArr, CString(" "))){
        cout << "MXCor::main: ERROR: ReadFileToDblArr(" << CS_FName_A << ") returned FALSE" << endl;
        exit(EXIT_FAILURE);
      }
      D_A1_Moving.resize(D_A2_PixArr.rows());
      D_A1_Moving = D_A2_PixArr(Range::all(), 0);
      
      if (!CS_FName_A.ReadFileToDblArr(CS_FName_B, D_A2_PixArr, CString(" "))){
        cout << "MXCor::main: ERROR: ReadFileToDblArr(" << CS_FName_B << ") returned FALSE" << endl;
        exit(EXIT_FAILURE);
      }
      D_A1_Static.resize(D_A2_PixArr.rows());
      D_A1_Static = D_A2_PixArr(Range::all(), 0);
    }
    cout << "MXCor::main: D_A1_Moving = " << D_A1_Moving << endl;
    cout << "MXCor::main: D_A1_Static = " << D_A1_Static << endl;
//    exit(EXIT_FAILURE);
    if (D_A1_Moving.size() != D_A1_Static.size()){
      cout << "MXCor::main: ERROR: Something went wrong" << endl;
      exit(EXIT_FAILURE);
    }
    double D_Shift_Out = 0.;
    double D_ChiSqu_Out = 0.;

    D_A1_Moving = D_A1_Moving * mean(D_A1_Static) / mean(D_A1_Moving);
    if (!F_ImageA.CrossCorrelate(D_A1_Static, 
                                 D_A1_Moving, 
                                 I_NPixShiftMax_In, 
                                 I_NPixShiftMax_In,
                                 D_Shift_Out,
                                 D_ChiSqu_Out)){
      cout << "MXCor::main: ERROR: CrossCorrelate returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    D_A1_PixShifts_Out(i_run) = D_Shift_Out;
    cout << "MXCor::main: shift out = " << D_Shift_Out << endl;

    cout << "MXCor::main: Image 2 needs to be shifted by " << D_Shift_Out << " pixels (<0: to the left, >0: to the right)" << endl;

/*  if (CS_FitsFileNameA_Out.GetLength() > 2){
    if (!F_ImageA.SetNCols(F_ImageA.GetNCols()-2*I_Shift_X)){
      cout << "MXCor::main: ERROR: F_ImageA.SetNCols(" << F_ImageA.GetNCols() - 2*I_Shift_X << " returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    if (!F_ImageA.SetNRows(F_ImageA.GetNRows()-2*I_Shift_Y)){
      cout << "MXCor::main: ERROR: F_ImageA.SetNRows(" << F_ImageA.GetNRows() - 2*I_Shift_Y << " returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    F_ImageA.GetPixArray() = D_A2_PixArrA(Range(I_Shift_Y, D_A2_PixArrA.rows()-I_Shift_Y-1), Range(I_Shift_X, D_A2_PixArrA.cols()-I_Shift_X-1));
    if (!F_ImageA.SetFileName(CS_FitsFileNameA_Out)){
      cout << "MXCor::main: ERROR: F_ImageA.SetFileName(" << CS_FitsFileNameA_Out << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    if (!F_ImageA.WriteArray()){
      cout << "MXCor::main: ERROR: F_ImageA.WriteArray() returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
  }

  if (CS_FitsFileNameB_Out.GetLength() > 2){
    if (I_A1_Area(1) > 0){
      D_A2_PixArrB.resize(F_ImageB.GetNRows(), F_ImageB.GetNCols());
      D_A2_PixArrB = F_ImageB.GetPixArray();
    }
    if (!F_ImageB.SetNCols(F_ImageB.GetNCols()-2*I_Shift_X)){
      cout << "MXCor::main: ERROR: F_ImageB.SetNCols(" << F_ImageB.GetNCols() - 2*I_Shift_X << " returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    if (!F_ImageB.SetNRows(F_ImageB.GetNRows()-2*I_Shift_Y)){
      cout << "MXCor::main: ERROR: F_ImageB.SetNRows(" << F_ImageB.GetNRows() - 2*I_Shift_Y << " returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    F_ImageB.GetPixArray() = D_A2_PixArrB(Range(0, D_A2_PixArrB.rows()-2*I_Shift_Y-1), Range(0, D_A2_PixArrB.cols()-2*I_Shift_X-1));
    if (!F_ImageB.SetFileName(CS_FitsFileNameB_Out)){
      cout << "MXCor::main: ERROR: F_ImageB.SetFileName(" << CS_FitsFileNameB_Out << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    if (!F_ImageB.WriteArray()){
      cout << "MXCor::main: ERROR: F_ImageB.WriteArray() returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
  }*/

    if (CS_FitsFileNameDiff_Out.GetLength() > 2){
      Array<double, 1> *P_D_A1_PixNumber = F_ImageA.DIndGenArr(D_A1_Moving.size());
      Array<double, 1> D_A1_PixNumberShift(P_D_A1_PixNumber->size());
      D_A1_PixNumberShift = (*P_D_A1_PixNumber) + D_A1_PixShifts_Out(i_run);
      Array<double, 1> D_A1_ValuesShift(P_D_A1_PixNumber->size());
      D_A1_ValuesShift = 0.;
      if (!F_ImageA.InterPol(D_A1_Moving, D_A1_PixNumberShift, *P_D_A1_PixNumber, &D_A1_ValuesShift)){
        cout << "MXCor::main: ERROR: InterPol returned FALSE" << endl;
        exit(EXIT_FAILURE);
      }
      Array<double, 1> D_A1_Diff(D_A1_Moving.size());
      D_A1_Diff = D_A1_ValuesShift - D_A1_Static;
      if (P_CS_FNEnd->EqualValue(CS_Fits)){
        F_ImageB.GetPixArray()(Range::all(), 0) = D_A1_Diff;
        if (!F_ImageB.SetFileName(CS_A1_FNameDiff_Out(i_run))){
          cout << "MXCor::main: ERROR: F_ImageB.SetFileName(" << CS_A1_FNameDiff_Out(i_run) << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
        if (!F_ImageB.WriteArray()){
          cout << "MXCor::main: ERROR: 2. F_ImageB.WriteArray() returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        if (!F_ImageA.WriteArrayToFile(D_A1_Diff, CS_A1_FNameDiff_Out(i_run), CString("ascii"))){
          cout << "MXCor::main: ERROR: WriteArrayToFile(D_A1_Diff, " << CS_A1_FNameDiff_Out(i_run) << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
    }
    delete(P_CS_FNEnd);
  }
  if (!F_ImageA.WriteArrayToFile(D_A1_PixShifts_Out, CS_PixelShiftList_Out, CString("ascii"))){
    cout << "MXCor::main: ERROR: WriteArrayToFile(D_A1_PixShifts_Out, " << CS_PixelShiftList_Out << ") returned FALSE" << endl;
    exit(EXIT_FAILURE);
  }
  return EXIT_SUCCESS;
}
