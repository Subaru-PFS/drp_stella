/*
author: Andreas Ritter
created: 23/12/2011
last edited: 23/12/2011
compiler: g++ 4.0
basis machine: Ubuntu Linux 9.04
*/

#include "MFindCurvature.h"

using namespace std;

int main(int argc, char *argv[])
{
  cout << "MFindCurvature::main: argc = " << argc << endl;
  if (argc < 8)
  {
    cout << "MFindCurvature::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: findcurvature(char[] FitsFileName_In, char[] DatabaseFileName_In, char[] FileList, int I_RefCol, int I_NPixMaxUp, int I_NPixMaxDown, int I_FitOrder)" << endl;
    exit(EXIT_FAILURE);
  }

  char *P_CharArr_In = (char*)argv[1];
  char *P_CharArr_DB = (char*)argv[2];
  char *P_FileList = (char*)argv[3];
  int I_RefCol_In = (int)(atoi((char*)argv[4]));
  int I_NPixMaxUp = (int)(atoi((char*)argv[5]));
  int I_NPixMaxDown = (int)(atoi((char*)argv[6]));
  int I_FitOrder = (int)(atoi((char*)argv[7]));
///double D_Gain = (double)(atof((char*)argv[4]));

  time_t seconds;
  CString CS_FitsFileName_In;
  CS_FitsFileName_In.Set(P_CharArr_In);
//  CString CS_FileName_Out;
//  CS_FileName_Out.Set(P_CharArr_Out);
  CString CS_DatabaseFileName_In;
  CS_DatabaseFileName_In.Set(P_CharArr_DB);
  CString CS_FileList;
  CS_FileList.Set(P_FileList);

  CFits F_Image;
  cout << "MFindCurvature::main: Starting F_Image.SetFileName(" << CS_FitsFileName_In.Get() << ")" << endl;
  if (!F_Image.SetFileName(CS_FitsFileName_In))
  {
    cout << "MFindCurvature::main: ERROR: F_Image.SetFileName(" << CS_FitsFileName_In.Get() << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Read FitsFile
  cout << "MFindCurvature::main: Starting F_Image.ReadArray()" << endl;
  if (!F_Image.ReadArray())
  {
    cout << "MFindCurvature::main: ERROR: F_Image.ReadArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Set DatabaseFileName_In
  cout << "MFindCurvature::main: Starting F_Image.SetDatabaseFileName(" << CS_DatabaseFileName_In << ")" << endl;
  if (!F_Image.SetDatabaseFileName(CS_DatabaseFileName_In))
  {
    cout << "MFindCurvature::main: ERROR: F_Image.SetDatabaseFileName(" << CS_DatabaseFileName_In << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Read DatabaseFileName_In
  cout << "MFindCurvature::main: Starting F_Image.ReadDatabaseEntry()" << endl;
  if (!F_Image.ReadDatabaseEntry())
  {
    cout << "MFindCurvature::main: ERROR: F_Image.ReadDatabaseEntry() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Calculate Trace Functions
  cout << "MFindCurvature::main: Starting F_Image.CalcTraceFunctions()" << endl;
  if (!F_Image.CalcTraceFunctions())
  {
    cout << "MFindCurvature::main: ERROR: F_Image.CalcTraceFunctions() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Calculate Feature Offsets and correct images
  cout << "MFindCurvature::main: Starting F_Image.CalculateFeatureOffsetAndShiftAllImages()" << endl;
  if (!F_Image.CalculateFeatureOffsetAndShiftAllImages(CS_FitsFileName_In,
                                                        CS_DatabaseFileName_In,
                                                        CS_FileList,
                                                        CS_FileList,
                                                        I_RefCol_In,
                                                        I_NPixMaxUp,
                                                        I_NPixMaxDown,
                                                        I_FitOrder,
                                                        CString("Poly"))){
    cout << "MFindCurvature::main: ERROR: F_Image.CalcFeatureOffsetAndShiftAllImages() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  return EXIT_SUCCESS;

  Array<int, 1> I_A1_X_Out(1);
  Array<int, 2> I_A2_ColsMinMax_Out(1,1);
  Array<double, 1> D_A1_Out(1);

  /// Cross-correlate all columns in apertures to column specified in input parameters
  cout << "MFindCurvature::main: Starting F_Image.CalcFeatureOffsets()" << endl;
  if (!F_Image.CalcFeatureOffsets(I_RefCol_In,
                                  I_NPixMaxUp,
                                  I_NPixMaxDown,
                                  I_FitOrder,
                                  I_A1_X_Out,
                                  I_A2_ColsMinMax_Out,
                                  D_A1_Out,
                                  CString("Poly"))){
    cout << "MFindCurvature::main: ERROR: F_Image.CalcFeatureOffsets() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MFindCurvature::main: F_Image.CalcFeatureOffsets() ready" << endl;
//  cout << "MFindCurvature::main: I_A1_X_Out = " << I_A1_X_Out << endl;
//  cout << "MFindCurvature::main: I_A2_ColsMinMax_Out = " << I_A2_ColsMinMax_Out << endl;
//  cout << "MFindCurvature::main: D_A1_Out = " << D_A1_Out << endl;

/**  Array<double, 2> D_A2_Out(I_A1_X_Out.size(), 3);
  bool B_Found = false;
  int I_Ap;
  for (int i=0; i < I_A1_X_Out.size(); i++){
    I_Ap = 0;
    B_Found = false;
    while (!B_Found){
      if ((I_A1_X_Out(i) >= I_A2_ColsMinMax_Out(I_Ap,0)) && (I_A1_X_Out(i) <= I_A2_ColsMinMax_Out(I_Ap,1))){
        B_Found = true;
      }
      else{
        I_Ap++;
      }
    }
    D_A2_Out(i,0) = I_Ap;
    D_A2_Out(i,1) = I_A1_X_Out(i);
    D_A2_Out(i,2) = D_A1_Out(i);
  }

  /// write outfile
  CString tempstr = CString(" ");
  tempstr.Set(P_CharArr_Out);
  if (!F_Image.WriteArrayToFile(D_A2_Out,
                                tempstr,
                                CString("ascii"))){
    cout << "MFindCurvature::main: ERROR: F_Image.WriteArrayToFile() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }**/

  Array<CString, 1> CS_A1_FileNames(1);
  if (!F_Image.ReadFileLinesToStrArr(CS_FileList,
                                     CS_A1_FileNames)){
    cout << "MFindCurvature::main: ERROR: F_Image.ReadFileLinesToStrArr() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MFindCurvature::main: CS_A1_FileNames = " << CS_A1_FileNames << endl;

//  delete(P_CS);
  return EXIT_SUCCESS;
}
