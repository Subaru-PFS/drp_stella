/*
author: Andreas Ritter
created: 03/20/2007
last edited: 03/20/2007
compiler: g++ 4.0
basis machine: Ubuntu Linux 6.06
*/

#include "MFindApsInCirc.h"

int main(int argc, char *argv[])
{
  cout << "MFindApsInCirc::main: argc = " << argc << endl;
  if (argc < 7)
  {
    cout << "MFindApsInCirc::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: findapsincirc <char[] FitsFileName_In> <char[] DatabaseFileName_In> <int x_center> <int y_center> <int radius> <char[] FileList_Out>" << endl;
    exit(EXIT_FAILURE);
  }
  CString CS_FitsFileName_In((char*)argv[1]);
  char *P_CharArr_DB = (char*)argv[2];
  int I_XCenter = (int)(atoi((char*)argv[3]));
  int I_YCenter = (int)(atoi((char*)argv[4]));
  int I_Radius = (int)(atoi((char*)argv[5]));
  char *P_CharArr_FileList_Out = (char*)argv[6];
  
  CString CS_DBFileName_In;
  CS_DBFileName_In.Set(P_CharArr_DB);
  CString CS_FileListName_Out;
  CS_FileListName_Out.Set(P_CharArr_FileList_Out);

  CFits F_Image;

  /// Set FitsFileName_In
  if (!F_Image.SetFileName(CS_FitsFileName_In))
  {
    cout << "MFindApsInCirc::main: ERROR: F_Image.SetFileName(" << CS_FitsFileName_In << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MFindApsInCirc::main: F_Image: FileName <" << CS_FitsFileName_In << "> set" << endl;
  
  if (!F_Image.ReadArray()){
    cout << "MFindApsInCirc::main: ERROR: ReadArray() returned FALSE" << endl;
    exit(EXIT_FAILURE);
  }

  /// Set DatabaseFileName_In
  if (!F_Image.SetDatabaseFileName(CS_DBFileName_In))
  {
    cout << "MFindApsInCirc::main: ERROR: F_Image.SetDatabaseFileName(" << CS_DBFileName_In << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MFindApsInCirc::main: F_Image: DatabaseFileName <" << CS_DBFileName_In << "> set" << endl;

  /// Read DatabaseFileName_In
  if (!F_Image.ReadDatabaseEntry())
  {
    cout << "MFindApsInCirc::main: ERROR: F_Image.ReadDatabaseEntry() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MFindApsInCirc::main: F_Image: DatabaseEntry read" << endl;
  
  if (!F_Image.CalcTraceFunctions()){
    cout << "MFindApsInCirc::main: ERROR: CalcTraceFunctions returned FALSE" << endl;
  }

  Array<double, 2> *P_D_A2_XCenters = F_Image.Get_XCenters();
  
  Array<double, 1> *P_D_A1_YCenter = F_Image.Get_YCenter();
  Array<double, 1> *P_D_A1_YLow = F_Image.Get_YLow();
  Array<double, 1> *P_D_A1_YHigh = F_Image.Get_YHigh();
  Array<double, 1> D_A1_YCenter(P_D_A1_YCenter->size());
  D_A1_YCenter = (*P_D_A1_YCenter) + (*P_D_A1_YLow) + ((*P_D_A1_YHigh) + (*P_D_A1_YLow)) / 2.;
  Array<double, 1> D_A1_XCenter(P_D_A1_YHigh->size());
  for (int i_ap=0; i_ap<P_D_A1_YCenter->size(); i_ap++)
  D_A1_XCenter(i_ap) = (*P_D_A2_XCenters)(i_ap,(int)(D_A1_YCenter(i_ap)));
  
  Array<double, 1> D_A1_Radius(P_D_A1_YCenter->size());
  D_A1_Radius = sqrt(pow2(D_A1_XCenter - double(I_XCenter)) + pow2((*P_D_A1_YCenter) - double(I_YCenter)));
  
  delete(P_D_A2_XCenters);
  delete(P_D_A1_YCenter);
  delete(P_D_A1_YLow);
  delete(P_D_A1_YHigh);

  int I_NInd;
  Array<int, 1> I_A1_Where(D_A1_Radius.size());
  I_A1_Where = where(D_A1_Radius < double(I_Radius), 1, 0);
  Array<int, 1> *P_I_A1_Index = F_Image.GetIndex(I_A1_Where, I_NInd);
  if (I_NInd < 0){
    cout << "MFindApsInCirc::main: ERROR: I_NInd = " << I_NInd << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MFindApsInCirc::main: P_I_A1_Index = " << *P_I_A1_Index << endl;

  if (!F_Image.WriteArrayToFile(*P_I_A1_Index, CS_FileListName_Out, CString("ascii"))){
    cout << "MFindApsInCirc::main: ERROR: WriteArrayToFile(P_I_A1_Index, " << CS_FileListName_Out << ") returned FALSE" << endl;
    exit(EXIT_FAILURE);
  }
  
  return EXIT_SUCCESS;
}
