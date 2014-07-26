/*
author: Andreas Ritter
created: 03/20/2007
last edited: 03/20/2007
compiler: g++ 4.0
basis machine: Ubuntu Linux 6.06
*/

#include "MEstimateScatter.h"

int main(int argc, char *argv[])
{
  cout << "MEstimateScatter::main: argc = " << argc << endl;
  if (argc < 4)
  {
    cout << "MEstimateScatter::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: estimatescatter <char[] FitsFileName_In> <char[] FitsFileName_Out> <int ClusterSizeX> <int ClusterSizeY> [AREA=[<int I_XMin>,<int I_XMax>,<int I_YMin>,<int I_YMax>]]" << endl;
    exit(EXIT_FAILURE);
  }
  char *P_CharArr_In = (char*)argv[1];
  char *P_CharArr_Out = (char*)argv[2];
  int I_ClusterSizeX = atoi((char*)argv[3]);
  cout << "MEstimateScatter: I_ClusterSizeX = " << I_ClusterSizeX << endl;
  int I_ClusterSizeY = atoi((char*)argv[4]);
  cout << "MEstimateScatter: I_ClusterSizeY = " << I_ClusterSizeY << endl;

  CString CS_FitsFileName_In;
  CS_FitsFileName_In.Set(P_CharArr_In);
  CString CS_FitsFileName_Out;
  CS_FitsFileName_Out.Set(P_CharArr_Out);

  Array<CString, 1> CS_A1_Args(1);
  CS_A1_Args(0) = CString(" ");
  void **PP_Args = (void**)malloc(sizeof(void*) * 1);

  CString CS(" ");
  CString CS_comp(" ");
  CString *P_CS;
  CS_comp.Set("AREA");
  Array<int, 1> I_A1_Area(4);
  bool B_Area_Set = false;
  if (argc == 6){
    CS.Set((char*)argv[5]);
    if (CS.GetLength() > CS_comp.GetLength()){
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MOptExtract_Obs_Sky::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        B_Area_Set = true;
        CString cs_temp;
        cs_temp.Set(",");
        int i_pos_a = CS_comp.GetLength()+2;
        int i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        cout << "MOptExtract_Obs_Sky: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
        delete(P_CS);
        P_CS = CS.SubString(i_pos_a,i_pos_b-1);
        cout << "MOptExtract_Obs_Sky: P_CS set to " << *P_CS << endl;
        I_A1_Area(0) = (int)(atoi(P_CS->Get()));
        cout << "MOptExtract_Obs_Sky: I_A1_Area(0) set to " << I_A1_Area(0) << endl;

        i_pos_a = i_pos_b+1;
        i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        cout << "MOptExtract_Obs_Sky: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
        delete(P_CS);
        P_CS = CS.SubString(i_pos_a,i_pos_b-1);
        cout << "MOptExtract_Obs_Sky: P_CS set to " << *P_CS << endl;
        I_A1_Area(1) = (int)(atoi(P_CS->Get()));
        cout << "MOptExtract_Obs_Sky: I_A1_Area(1) set to " << I_A1_Area(1) << endl;

        i_pos_a = i_pos_b+1;
        i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        cout << "MOptExtract_Obs_Sky: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
        delete(P_CS);
        P_CS = CS.SubString(i_pos_a,i_pos_b-1);
        cout << "MOptExtract_Obs_Sky: P_CS set to " << *P_CS << endl;
        I_A1_Area(2) = (int)(atoi(P_CS->Get()));
        cout << "MOptExtract_Obs_Sky: I_A1_Area(2) set to " << I_A1_Area(2) << endl;

        i_pos_a = i_pos_b+1;
        cs_temp.Set("]");
        i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        if (i_pos_b < 0){
          cs_temp.Set(")");
          i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        }
        cout << "MOptExtract_Obs_Sky: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
        delete(P_CS);
        P_CS = CS.SubString(i_pos_a,i_pos_b-1);
        cout << "MOptExtract_Obs_Sky: P_CS set to " << *P_CS << endl;
        I_A1_Area(3) = (int)(atoi(P_CS->Get()));
        delete(P_CS);
        cout << "MOptExtract_Obs_Sky: I_A1_Area(3) set to " << I_A1_Area(3) << endl;

        CS_A1_Args(0) = CString("AREA");
        PP_Args[0] = &I_A1_Area;
        cout << "MOptExtract_Obs_Sky::main: I_A1_Area set to " << I_A1_Area << endl;
      }
    }
  }

  CFits F_Image;
  if (!F_Image.SetFileName(CS_FitsFileName_In))
  {
    cout << "MEstimateScatter::main: ERROR: F_Image.SetFileName(" << CS_FitsFileName_In.Get() << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MEstimateScatter::main: FileName <" << CS_FitsFileName_In.Get() << "> set" << endl;

  /// Read FitsFile
  if (!F_Image.ReadArray())
  {
    cout << "MEstimateScatter::main: ERROR: F_Image.ReadArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MEstimateScatter::main: F_Image: Array read" << endl;

  if (!B_Area_Set){
    I_A1_Area(0) = 0;
    I_A1_Area(1) = F_Image.GetNCols()-1;
    I_A1_Area(2) = 0;
    I_A1_Area(3) = F_Image.GetNRows()-1;
  }

  int I_NBoxesX = (I_A1_Area(1) - I_A1_Area(0) + 1) / (I_ClusterSizeX);
  int I_NBoxesY = (I_A1_Area(3) - I_A1_Area(2) + 1) / (I_ClusterSizeY);

  Array<double, 2> D_A2_ScatteredLight_Out(2,2);
  D_A2_ScatteredLight_Out = 0.;
  if (!F_Image.EstScatterKriging(I_NBoxesX, I_NBoxesY, D_A2_ScatteredLight_Out, CS_A1_Args, PP_Args)){
    cout << "MEstimateScatter::main: ERROR: F_Image.EstScatterKriging(" << I_NBoxesX << ", " << I_NBoxesY << ",...) returned FALSE" << endl;
    exit(EXIT_FAILURE);
  }

  F_Image.WriteFits(&D_A2_ScatteredLight_Out, CS_FitsFileName_Out);

  return EXIT_SUCCESS;
}
