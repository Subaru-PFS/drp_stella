/*
author: Andreas Ritter
created: 04/12/2007
last edited: 05/05/2007
compiler: g++ 4.0
basis machine: Ubuntu Linux 6.06
*/

#include "MSubtractOverscanAndTrim.h"

using namespace std;

int main(int argc, char *argv[])
{
  cout << "MSubtractOverscanAndTrim::main: argc = " << argc << endl;
  if (argc < 5)
  {
    cout << "MSubtractOverscanAndTrim::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: subtractoverscanandtrim <char[] [@]FitsFileName_In> <char[] [@]FitsFileName_Out> <[int(xmin),int(xmax),int(ymin),int(ymax)] OverscanArea> <[int(xmin),int(xmax),int(ymin),int(ymax)] TrimArea>" << endl;
    exit(EXIT_FAILURE);
  }
  Array<int, 1> I_A1_OverscanArea(4);
  Array<int, 1> I_A1_TrimArea(4);

  Array<CString, 1> CS_A1_Args(2);
  CS_A1_Args = CString("\0");
  void **PP_Args;
  PP_Args = (void**)malloc(sizeof(void*) * 2);
  CString CS(" ");
  CString CS_comp(" ");
  CString *P_CS = new CString(" ");

  CString CS_FitsFileList_In((char*)argv[1]);
  CString CS_FitsFileList_Out((char*)argv[2]);
  CString CS_OverscanArea((char*)(argv[3]));
  CString CS_TrimArea((char*)(argv[4]));

  Array<CString, 1> CS_A1_FitsFileList_In(1);
  CS_A1_FitsFileList_In(0) = CS_FitsFileList_In;
  if (CS_FitsFileList_In.IsList()){
    if (!CS_FitsFileList_In.ReadFileLinesToStrArr(CS_FitsFileList_In, CS_A1_FitsFileList_In)){
      cout << "MSubtractOverscanAndTrim::main: ERROR: ReadFileLinesToStrArr(CS_FitsFileList_In=" << CS_FitsFileList_In << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
  }

  Array<CString, 1> CS_A1_FitsFileList_Out(1);
  CS_A1_FitsFileList_Out(0) = CS_FitsFileList_Out;
  if (CS_FitsFileList_Out.IsList()){
    if (!CS_FitsFileList_Out.ReadFileLinesToStrArr(CS_FitsFileList_Out, CS_A1_FitsFileList_Out)){
      cout << "MSubtractOverscanAndTrim::main: ERROR: ReadFileLinesToStrArr(CS_FitsFileList_Out=" << CS_FitsFileList_Out << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
  }

  if (CS_A1_FitsFileList_In.size() != CS_A1_FitsFileList_Out.size()){
    cout << "MSubtractOverscanAndTrim::main: ERROR: CS_A1_FitsFileList_In.size(=" << CS_A1_FitsFileList_In.size() << ") != CS_A1_FitsFileList_Out.size(=" << CS_A1_FitsFileList_Out.size() << ") => Returning FALSE" << endl;
    exit(EXIT_FAILURE);
  }

  CString cs_temp(",");
  int i_pos_a = 1;
  int i_pos_b = CS_OverscanArea.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
  cout << "MSubtractOverscanAndTrim: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
  delete(P_CS);
  P_CS = CS_OverscanArea.SubString(i_pos_a,i_pos_b-1);
  cout << "MSubtractOverscanAndTrim: P_CS set to " << *P_CS << endl;
    I_A1_OverscanArea(2) = (int)(atoi(P_CS->Get()));
  cout << "MSubtractOverscanAndTrim: I_A1_Area(0) set to " << I_A1_OverscanArea(0) << endl;

  i_pos_a = i_pos_b+1;
  i_pos_b = CS_OverscanArea.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
  cout << "MSubtractOverscanAndTrim: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
  delete(P_CS);
  P_CS = CS_OverscanArea.SubString(i_pos_a,i_pos_b-1);
  cout << "MSubtractOverscanAndTrim: P_CS set to " << *P_CS << endl;
    I_A1_OverscanArea(3) = (int)(atoi(P_CS->Get()));
  cout << "MSubtractOverscanAndTrim: I_A1_Area(1) set to " << I_A1_OverscanArea(1) << endl;

  i_pos_a = i_pos_b+1;
  i_pos_b = CS_OverscanArea.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
  cout << "MSubtractOverscanAndTrim: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
  delete(P_CS);
  P_CS = CS_OverscanArea.SubString(i_pos_a,i_pos_b-1);
  cout << "MSubtractOverscanAndTrim: P_CS set to " << *P_CS << endl;
    I_A1_OverscanArea(0) = (int)(atoi(P_CS->Get()));
  cout << "MSubtractOverscanAndTrim: I_A1_Area(2) set to " << I_A1_OverscanArea(2) << endl;

  i_pos_a = i_pos_b+1;
  cs_temp.Set("]");
  i_pos_b = CS_OverscanArea.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
  if (i_pos_b < 0){
    cs_temp.Set(")");
    i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
  }
  cout << "MSubtractOverscanAndTrim: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
  delete(P_CS);
  P_CS = CS_OverscanArea.SubString(i_pos_a,i_pos_b-1);
  cout << "MSubtractOverscanAndTrim: P_CS set to " << *P_CS << endl;
    I_A1_OverscanArea(1) = (int)(atoi(P_CS->Get()));
  cout << "MSubtractOverscanAndTrim: I_A1_Area(3) set to " << I_A1_OverscanArea(3) << endl;

  /// Read TrimArea
  cs_temp.Set(",");
  i_pos_a = 1;
  i_pos_b = CS_TrimArea.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
  cout << "MSubtractOverscanAndTrim: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
  delete(P_CS);
  P_CS = CS_TrimArea.SubString(i_pos_a,i_pos_b-1);
  cout << "MSubtractOverscanAndTrim: P_CS set to " << *P_CS << endl;
  I_A1_TrimArea(2) = (int)(atoi(P_CS->Get()));
  cout << "MSubtractOverscanAndTrim: I_A1_Area(0) set to " << I_A1_TrimArea(0) << endl;

  i_pos_a = i_pos_b+1;
  i_pos_b = CS_TrimArea.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
  cout << "MSubtractOverscanAndTrim: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
  delete(P_CS);
  P_CS = CS_TrimArea.SubString(i_pos_a,i_pos_b-1);
  cout << "MSubtractOverscanAndTrim: P_CS set to " << *P_CS << endl;
  I_A1_TrimArea(3) = (int)(atoi(P_CS->Get()));
  cout << "MSubtractOverscanAndTrim: I_A1_Area(1) set to " << I_A1_TrimArea(1) << endl;

  i_pos_a = i_pos_b+1;
  i_pos_b = CS_TrimArea.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
  cout << "MSubtractOverscanAndTrim: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
  delete(P_CS);
  P_CS = CS_TrimArea.SubString(i_pos_a,i_pos_b-1);
  cout << "MSubtractOverscanAndTrim: P_CS set to " << *P_CS << endl;
  I_A1_TrimArea(0) = (int)(atoi(P_CS->Get()));
  cout << "MSubtractOverscanAndTrim: I_A1_Area(2) set to " << I_A1_TrimArea(2) << endl;

  i_pos_a = i_pos_b+1;
  cs_temp.Set("]");
  i_pos_b = CS_TrimArea.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
  if (i_pos_b < 0){
    cs_temp.Set(")");
    i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
  }
  cout << "MSubtractOverscanAndTrim: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
  delete(P_CS);
  P_CS = CS_TrimArea.SubString(i_pos_a,i_pos_b-1);
  cout << "MSubtractOverscanAndTrim: P_CS set to " << *P_CS << endl;
  I_A1_TrimArea(1) = (int)(atoi(P_CS->Get()));
  cout << "MSubtractOverscanAndTrim: I_A1_Area(3) set to " << I_A1_TrimArea(3) << endl;
  //  CS_A1_Args(0) = CString("AREA");
//  PP_Args[0] = &I_A1_Area;
//  cout << "MSubtractOverscanAndTrim::main: AREA set to " << I_A1_Area << endl;
  time_t seconds;
//  if (argc == 8)
//  {
//    I_SwathWidth = (int)(atoi((char*)argv[7]));
//    cout << "MSubtractOverscanAndTrim::main: I_SwathWidth set to " << I_SwathWidth << endl;
  CString CS_FitsFileName_In;
  CString CS_FitsFileName_Out;
  CFits F_Image;
  for (int i_file=0; i_file < CS_A1_FitsFileList_In.size(); i_file++){
    CS_FitsFileName_In.Set(CS_A1_FitsFileList_In(i_file));
    CS_FitsFileName_Out.Set(CS_A1_FitsFileList_Out(i_file));

    cout << "MSubtractOverscanAndTrim::main: Starting F_Image.SetFileName(" << CS_FitsFileName_In.Get() << ")" << endl;
    if (!F_Image.SetFileName(CS_FitsFileName_In))
    {
      cout << "MSubtractOverscanAndTrim::main: ERROR: F_Image.SetFileName(" << CS_FitsFileName_In.Get() << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    /// Read FitsFile
    cout << "MSubtractOverscanAndTrim::main: Starting F_Image.ReadArray()" << endl;
    if (!F_Image.ReadArray())
    {
      cout << "MSubtractOverscanAndTrim::main: ERROR: F_Image.ReadArray() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    /// Write aperture header information
    F_Image.WriteApHead(CString("aphead_")+CS_FitsFileName_In+CString(".head"));

    CFits F_OutImage;
    /// Set CS_FitsFileName_In
    cout << "MSubtractOverscanAndTrim::main: Starting F_OutImage.SetFileName(" << CS_FitsFileName_In << ")" << endl;
    if (!F_OutImage.SetFileName(CS_FitsFileName_In))
    {
      cout << "MSubtractOverscanAndTrim::main: ERROR: F_OutImage.SetFileName(" << CS_FitsFileName_In << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    ///Read FitsFile
    cout << "MSubtractOverscanAndTrim::main: Starting F_OutImage.ReadArray()" << endl;
    if (!F_OutImage.ReadArray())
    {
      cout << "MSubtractOverscanAndTrim::main: ERROR: F_OutImage.ReadArray() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    /// Set CS_FitsFileName_Out
    cout << "MSubtractOverscanAndTrim::main: Starting F_OutImage.SetFileName(" << CS_FitsFileName_Out << ")" << endl;
    if (!F_OutImage.SetFileName(CS_FitsFileName_Out))
    {
      cout << "MSubtractOverscanAndTrim::main: ERROR: F_OutImage.SetFileName(" << CS_FitsFileName_Out << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    /// change size of F_OutImage to (NApertures x NRows)
    if (!F_OutImage.SetNCols(I_A1_TrimArea(1) - I_A1_TrimArea(0) + 1))
    {
      cout << "MSubtractOverscanAndTrim::main: ERROR: F_OutImage.SetNCols(" << F_OutImage.GetNCols() << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
    if (!F_OutImage.SetNRows(I_A1_TrimArea(3) - I_A1_TrimArea(2) + 1))
    {
      cout << "MSubtractOverscanAndTrim::main: ERROR: F_OutImage.SetNRows(" << F_OutImage.GetNRows() << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    Array<double, 2> D_A2_Overscan(I_A1_OverscanArea(1) - I_A1_OverscanArea(0) + 1, I_A1_OverscanArea(3) - I_A1_OverscanArea(2) + 1);
    D_A2_Overscan = F_Image.GetPixArray()(Range(I_A1_OverscanArea(0), I_A1_OverscanArea(1)), Range(I_A1_OverscanArea(2), I_A1_OverscanArea(3)));
    double D_Overscan = F_Image.Median(D_A2_Overscan, false);
    cout << "MSubtractOverscanAndTrim::main: Overscan level = " << D_Overscan << endl;

    Array<double, 2> D_A2_Trim(I_A1_TrimArea(1) - I_A1_TrimArea(0) + 1, I_A1_TrimArea(3) - I_A1_TrimArea(2) + 1);
    D_A2_Trim = F_Image.GetPixArray()(Range(I_A1_TrimArea(0), I_A1_TrimArea(1)), Range(I_A1_TrimArea(2), I_A1_TrimArea(3)));
    if (!F_OutImage.SetNCols(D_A2_Trim.cols())){
      cout << "MSubtractOverscanAndTrim::main: ERROR: F_OutImage.SetNCols(" << D_A2_Trim.cols() << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    if (!F_OutImage.SetNRows(D_A2_Trim.rows())){
      cout << "MSubtractOverscanAndTrim::main: ERROR: F_OutImage.SetNRows(" << D_A2_Trim.rows() << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }

    D_A2_Trim = D_A2_Trim - D_Overscan;
    F_OutImage.GetPixArray() = D_A2_Trim;

    cout << "MSubtractOverscanAndTrim::main: Starting F_OutImage.WriteArray()" << endl;
    if (!F_OutImage.WriteArray())
    {
      cout << "MSubtractOverscanAndTrim::main: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
  }
  delete(P_CS);
  return EXIT_SUCCESS;
}
