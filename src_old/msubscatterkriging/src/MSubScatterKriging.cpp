/*
author: Andreas Ritter
created: 22/06/2012
last edited: 22/06/2012
compiler: g++ 4.4
basis machine: Ubuntu Linux 10.04
*/

#include "MSubScatterKriging.h"

int main(int argc, char *argv[])
{
  cout << "MSubScatterKriging::main: argc = " << argc << endl;
  if (argc < 10)
  {
    cout << "MSubScatterKriging::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: subscatterkriging char[] FitsFileName_In, char[] DatabaseFileName_In, char[] FitsFileName_Out, int[] ClusterSizes_In, int AddNPixToAp_X, int AddNPixToAp_Y, int NRectangles_X, int NRectangles_Y[, FileName_ApZero_Out=FileName_ApZero_Out][, FileName_Scatter_Out=FileName_Scatter_Out][,FileName_Clustered_Out=FileName_Clustered_Out]" << endl;
    exit(EXIT_FAILURE);
  }

  Array<CString, 1> CS_A1_Args(3);
  CS_A1_Args = " ";

  void **PP_Args;
  PP_Args = (void**)malloc(sizeof(void*)*3);

  char *P_CharArr_In = (char*)argv[1];
  char *P_CharArr_DB = (char*)argv[2];
  char *P_CharArr_Out = (char*)argv[3];
  char *P_CharArr_ClusterSizes = (char*)argv[4];
  Array<int, 2> I_A2_ClusterSizes_X_Y_In(1,2);
  int I_AddNPixToAp_X = atoi((char*)argv[5]);
  int I_AddNPixToAp_Y = atoi((char*)argv[6]);
  int I_NRectangles_X = atoi((char*)argv[7]);
  int I_NRectangles_Y = atoi((char*)argv[8]);
  Array<double, 2> D_A2_ScatteredLight_Out(2,2);
  CString CS(" ");
  CString CS_comp(" ");
  CString *P_CS = new CString(" ");
  CString CS_FileName_ApZero_Out(" ");
  CString CS_FileName_Scatter_Out(" ");
  CString CS_FileName_Clustered_Out(" ");
  for (int i=9; i <= argc; i++)
  {
    CS.Set((char*)argv[i]);
    cout << "MSubScatterKriging::main: Reading Parameter " << CS << endl;
    CS_comp.Set("FileName_ApZero_Out");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MSubScatterKriging::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        CS_A1_Args(0).Set("FileName_ApZero_Out");
        delete(P_CS);
        P_CS = CS.SubString(CS_comp.GetLength()+1);
        CS_FileName_ApZero_Out = P_CS->Get();
        cout << "MSubScatterKriging::main: CS_FileName_ApZero_Out set to " << CS_FileName_ApZero_Out << endl;
        PP_Args[0] = &CS_FileName_ApZero_Out;
      }
    }

    CS_comp.Set("FileName_Scatter_Out");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MSubScatterKriging::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        CS_A1_Args(1).Set("FileName_ScatterFit_Out");
        delete(P_CS);
        P_CS = CS.SubString(CS_comp.GetLength()+1);
        CS_FileName_Scatter_Out = P_CS->Get();
        cout << "MSubScatterKriging::main: CS_FileName_Scatter_Out set to " << CS_FileName_Scatter_Out << endl;
        PP_Args[1] = &CS_FileName_Scatter_Out;
      }
    }

    CS_comp.Set("FileName_Clustered_Out");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MSubScatterKriging::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        CS_A1_Args(2).Set("FileName_Clustered_Out");
        delete(P_CS);
        P_CS = CS.SubString(CS_comp.GetLength()+1);
        CS_FileName_Clustered_Out = P_CS->Get();
        cout << "MSubScatterKriging::main: CS_FileName_Clustered_Out set to " << CS_FileName_Clustered_Out << endl;
        PP_Args[2] = &CS_FileName_Clustered_Out;
      }
    }
  }
  delete(P_CS);

  CString CS_FitsFileName_In;
  CS_FitsFileName_In.Set(P_CharArr_In);
  cout << "MSubScatterKriging::main: CS_FitsFileName_In set to " << CS_FitsFileName_In << endl;

  CString CS_FitsFileName_Out;
  CS_FitsFileName_Out.Set(P_CharArr_Out);
  cout << "MSubScatterKriging::main: CS_FitsFileName_Out set to " << CS_FitsFileName_Out << endl;
  CString CS_DatabaseFileName_In;
  CS_DatabaseFileName_In.Set(P_CharArr_DB);
  cout << "MSubScatterKriging::main: CS_DatabaseFileName_In set to " << CS_DatabaseFileName_In << endl;
  CString CS_ClusterSizes;
  CS_ClusterSizes.Set(P_CharArr_ClusterSizes);
  cout << "MSubScatterKriging::main: CS_ClusterSizes set to " << CS_ClusterSizes << endl;
  CString *P_CS_ClusterSizes = CS_ClusterSizes.SubString(1);
  cout << "MSubScatterKriging::main: P_CS_ClusterSizes set to " << *P_CS_ClusterSizes << endl;
  CString *P_CS_ClusterSizesTemp;
  int I_NClusterSizes = 0;
  int I_Cluster = 0;
  CString *P_CS_Number;
  CString CS_Komma(",");
  CString CS_Bracket("]");
  cout << "MSubScatterKriging::main: Starting do loop: CS_Komma = " << CS_Komma << endl;
  do{
    I_Cluster++;
    if (I_Cluster > I_A2_ClusterSizes_X_Y_In.rows())
      I_A2_ClusterSizes_X_Y_In.resizeAndPreserve(I_Cluster, 2);
    cout << "MSubScatterKriging::main: do loop:  I_Cluster = " << I_Cluster << ": P_CS_ClusterSizes->StrPos(CS_Komma) = " << P_CS_ClusterSizes->StrPos(CS_Komma) << endl;
    P_CS_Number = P_CS_ClusterSizes->SubString(0,P_CS_ClusterSizes->StrPos(CS_Komma)-1);
    cout << "MSubScatterKriging::main: do loop:  I_Cluster = " << I_Cluster << ": P_CS_Number set to " << *P_CS_Number << endl;
    I_A2_ClusterSizes_X_Y_In(I_Cluster-1, 0) = atoi(P_CS_Number->Get());
    delete(P_CS_Number);
    P_CS_ClusterSizesTemp = P_CS_ClusterSizes->SubString(P_CS_ClusterSizes->StrPos(CS_Komma)+1);
    cout << "MSubScatterKriging::main: do loop: I_Cluster = " << I_Cluster << ": P_CS_ClusterSizesTemp set to " << *P_CS_ClusterSizesTemp << endl;
    delete(P_CS_ClusterSizes);
    if (P_CS_ClusterSizesTemp->StrPos(CS_Komma) > 0){
      P_CS_Number = P_CS_ClusterSizesTemp->SubString(0,P_CS_ClusterSizesTemp->StrPos(CS_Komma)-1);
      cout << "MSubScatterKriging::main: do loop: I_Cluster = " << I_Cluster << ": P_CS_Number set to " << *P_CS_Number << endl;
      P_CS_ClusterSizes = P_CS_ClusterSizesTemp->SubString(P_CS_ClusterSizesTemp->StrPos(CS_Komma)+1);
      cout << "MSubScatterKriging::main: do loop: I_Cluster = " << I_Cluster << ": P_CS_ClusterSizes set to " << *P_CS_ClusterSizes << endl;
    }
    else{
      cout << "MSubScatterKriging::main: do loop:  I_Cluster = " << I_Cluster << ": P_CS_ClusterSizesTemp->StrPos(CS_Bracket) = " << P_CS_ClusterSizesTemp->StrPos(CS_Bracket) << endl;
      P_CS_Number = P_CS_ClusterSizesTemp->SubString(0,P_CS_ClusterSizesTemp->StrPos(CS_Bracket)-1);
      cout << "MSubScatterKriging::main: do loop: I_Cluster = " << I_Cluster << ": P_CS_Number set to " << *P_CS_Number << endl;
      P_CS_ClusterSizes = new CString(" ");
      cout << "MSubScatterKriging::main: do loop: I_Cluster = " << I_Cluster << ": P_CS_ClusterSizes set to " << *P_CS_ClusterSizes << endl;
    }
    I_A2_ClusterSizes_X_Y_In(I_Cluster-1, 1) = atoi(P_CS_Number->Get());
    cout << "MSubScatterKriging::main: do loop: I_Cluster = " << I_Cluster << ": I_A2_ClusterSizes_X_Y_In = " << I_A2_ClusterSizes_X_Y_In << endl;
    delete(P_CS_ClusterSizesTemp);
  } while (P_CS_ClusterSizes->StrPos(CS_Komma) >= 0);
  cout << "MSubScatterKriging::main: do loop finished" << endl;
//  delete(P_CharArr_ClusterSizes);
  cout << "MSubScatterKriging::main: I_A2_ClusterSizes_X_Y_In = " << I_A2_ClusterSizes_X_Y_In << endl;

  CFits F_Image;
  cout << "MSubScatterKriging::main: Starting F_Image.SetFileName(" << CS_FitsFileName_In.Get() << ")" << endl;
  if (!F_Image.SetFileName(CS_FitsFileName_In))
  {
    cout << "MSubScatterKriging::main: ERROR: F_Image.SetFileName(" << CS_FitsFileName_In.Get() << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Read FitsFile
  cout << "MSubScatterKriging::main: Starting F_Image.ReadArray()" << endl;
  if (!F_Image.ReadArray())
  {
    cout << "MSubScatterKriging::main: ERROR: F_Image.ReadArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Set DatabaseFileName_In
  cout << "MSubScatterKriging::main: Starting F_Image.SetDatabaseFileName(" << CS_DatabaseFileName_In << ")" << endl;
  if (!F_Image.SetDatabaseFileName(CS_DatabaseFileName_In))
  {
    cout << "MSubScatterKriging::main: ERROR: F_Image.SetDatabaseFileName(" << CS_DatabaseFileName_In << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Read DatabaseFileName_In
  cout << "MSubScatterKriging::main: Starting F_Image.ReadDatabaseEntry()" << endl;
  if (!F_Image.ReadDatabaseEntry())
  {
    cout << "MSubScatterKriging::main: ERROR: F_Image.ReadDatabaseEntry() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Calculate Trace Functions
  cout << "MSubScatterKriging::main: Starting F_Image.CalcTraceFunctions()" << endl;
  if (!F_Image.CalcTraceFunctions())
  {
    cout << "MSubScatterKriging::main: ERROR: F_Image.CalcTraceFunctions() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Calculate and subtract scattered light
  cout << "MSubScatterKriging::main: Starting F_Image.CalcScatterKriging()" << endl;

  if (!F_Image.CalcScatterKriging(I_A2_ClusterSizes_X_Y_In,
                                  I_AddNPixToAp_X,
                                  I_AddNPixToAp_Y,
                                  I_NRectangles_X,
                                  I_NRectangles_Y,
                                  D_A2_ScatteredLight_Out,
                                  CS_A1_Args,
                                  PP_Args))
  {
    cout << "MSubScatterKriging::main: ERROR: F_Image.CalcScatterKriging() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

//  if (!F_Image.WriteFits(&D_A2_ScatteredLight_Out, CS_FileName_Scatter_Out)){
//    cout << "MSubScatterKriging::main: ERROR: WriteFits(D_A2_ScatteredLight_Out) returned FALSE => Returning FALSE" << endl;
//    exit(EXIT_FAILURE);
//  }

  CFits *P_CFits_Out = new CFits(F_Image);

  *P_CFits_Out -= D_A2_ScatteredLight_Out;

  /// Set CS_FitsFileName_Out
  cout << "MSubScatterKriging::main: Starting P_CFits_Out->SetFileName(" << CS_FitsFileName_Out << ")" << endl;
  if (!P_CFits_Out->SetFileName(CS_FitsFileName_Out))
  {
    cout << "MSubScatterKriging::main: ERROR: P_CFits_Out->SetFileName(" << CS_FitsFileName_Out << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

//  F_Image.GetPixArray() = F_Image.GetProfArray();

  /// Write Image
  cout << "MSubScatterKriging::main: Starting P_CFits_Out->WriteArray()" << endl;
  if (!P_CFits_Out->WriteArray())
  {
    cout << "MSubScatterKriging::main: ERROR: P_CFits_Out->WriteArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  delete(P_CFits_Out);

  return EXIT_SUCCESS;
}
