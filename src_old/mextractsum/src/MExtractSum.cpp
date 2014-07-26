/*
author: Andreas Ritter
created: 04/12/2007
last edited: 05/05/2007
compiler: g++ 4.0
basis machine: Ubuntu Linux 6.06
*/

#include "MExtractSum.h"

using namespace std;

int main(int argc, char *argv[])
{
  cout << "MExtractSum::main: argc = " << argc << endl;
  if (argc < 5)
  {
    cout << "MExtractSum::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: extractsum <char[] FitsFileName_In> <char[] DatabaseFileName_In> <char[] FitsFileName_Out> <double Gain> [ERR_IN=char[]] [ERR_OUT_EC=char[]] [AREA=[int(xmin),int(xmax),int(ymin),int(ymax)]] [APERTURES=<char[] ApertureFile_In]" << endl;
    exit(EXIT_FAILURE);
  }
  Array<int, 1> I_A1_Area(4);

  Array<CString, 1> CS_A1_Args(2);
  CS_A1_Args = CString("\0");
  void **PP_Args;
  PP_Args = (void**)malloc(sizeof(void*) * 2);
  CString CS(" ");
  CString CS_comp(" ");
  CString *P_CS = new CString(" ");
  CString *P_CS_ErrIn = new CString(" ");
  CString *P_CS_EcOut = new CString(" ");
  CString *P_CS_ErrOut = new CString(" ");
  CString *P_CS_ApertureList_In = new CString(" ");

  char *P_CharArr_In = (char*)argv[1];
  char *P_CharArr_DB = (char*)argv[2];
  char *P_CharArr_Out = (char*)argv[3];
  CString CS_Gain((char*)(argv[4]));
  double D_Gain = 0.;
  if (!CS_Gain.AToD(D_Gain)){
    cout << "MExtractSum::main: ERROR: CS_Gain(=" << CS_Gain << ").AToD(D_Gain) returning FALSE" << endl;
    exit(EXIT_FAILURE);
  }
  Array<int, 1> I_A1_Apertures(1);
  I_A1_Apertures = 0;
  Array<CString, 1> CS_A1_ApNum(1);
  CS_A1_ApNum(0).Set(" ");
  bool B_Apertures_Set = false;

  /// read optional parameters
  for (int i = 5; i <= argc; i++){
    CS.Set((char*)argv[i]);
    cout << "MExtractSum: Reading Parameter " << CS << endl;

    CS_comp.Set("ERR_IN");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtractSum::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS_ErrIn);
        P_CS_ErrIn = CS.SubString(CS_comp.GetLength()+1);
        cout << "MExtractSum::main: P_CS_ERRIN set to " << *P_CS_ErrIn << endl;
      }
    }

    CS_comp.Set("ERR_OUT_EC");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtractSum::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS_ErrOut);
        P_CS_ErrOut = CS.SubString(CS_comp.GetLength()+1);
        cout << "MExtractSum::main: P_CS_ErrOut set to " << *P_CS_ErrOut << endl;
      }
    }

    CS_comp.Set("APERTURES");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtractSum::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS_ApertureList_In);
        P_CS_ApertureList_In = CS.SubString(CS_comp.GetLength()+1);
        cout << "MExtractSum::main: P_CS_ApertureList_In set to " << *P_CS_ApertureList_In << endl;
	if (!CS_comp.ReadFileLinesToStrArr(*P_CS_ApertureList_In, CS_A1_ApNum)){
	  cout << "MExtractSum::main: ERROR: ReadFileLinesToStrArr(" << *P_CS_ApertureList_In << ") returned FALSE" << endl;
	  exit(EXIT_FAILURE);
	}
	B_Apertures_Set = true;

        CS_A1_Args(1) = CString("APERTURES");
        PP_Args[1] = &I_A1_Apertures;
      }
    }

    /// AREA
    CS_comp.Set("AREA");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtractSum::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        CString cs_temp;
        cs_temp.Set(",");
        int i_pos_a = CS_comp.GetLength()+2;
        int i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        cout << "MExtractSum: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
        delete(P_CS);
        P_CS = CS.SubString(i_pos_a,i_pos_b-1);
        cout << "MExtractSum: P_CS set to " << *P_CS << endl;
        I_A1_Area(0) = (int)(atoi(P_CS->Get()));
        cout << "MExtractSum: I_A1_Area(0) set to " << I_A1_Area(0) << endl;

        i_pos_a = i_pos_b+1;
        i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        cout << "MExtractSum: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
        delete(P_CS);
        P_CS = CS.SubString(i_pos_a,i_pos_b-1);
        cout << "MExtractSum: P_CS set to " << *P_CS << endl;
        I_A1_Area(1) = (int)(atoi(P_CS->Get()));
        cout << "MExtractSum: I_A1_Area(1) set to " << I_A1_Area(1) << endl;

        i_pos_a = i_pos_b+1;
        i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        cout << "MExtractSum: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
        delete(P_CS);
        P_CS = CS.SubString(i_pos_a,i_pos_b-1);
        cout << "MExtractSum: P_CS set to " << *P_CS << endl;
        I_A1_Area(2) = (int)(atoi(P_CS->Get()));
        cout << "MExtractSum: I_A1_Area(2) set to " << I_A1_Area(2) << endl;

        i_pos_a = i_pos_b+1;
        cs_temp.Set("]");
        i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        if (i_pos_b < 0){
          cs_temp.Set(")");
          i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        }
        cout << "MExtractSum: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
        delete(P_CS);
        P_CS = CS.SubString(i_pos_a,i_pos_b-1);
        cout << "MExtractSum: P_CS set to " << *P_CS << endl;
        I_A1_Area(3) = (int)(atoi(P_CS->Get()));
        cout << "MExtractSum: I_A1_Area(3) set to " << I_A1_Area(3) << endl;

        CS_A1_Args(0) = CString("AREA");
        PP_Args[0] = &I_A1_Area;
        cout << "MExtractSum::main: AREA set to " << I_A1_Area << endl;
      }
    }
  }
  time_t seconds;
//  if (argc == 8)
//  {
//    I_SwathWidth = (int)(atoi((char*)argv[7]));
//    cout << "MExtractSum::main: I_SwathWidth set to " << I_SwathWidth << endl;
  CString CS_FitsFileName_In;
  CS_FitsFileName_In.Set(P_CharArr_In);
  CString CS_FitsFileName_Out;
  CS_FitsFileName_Out.Set(P_CharArr_Out);
  CString CS_DatabaseFileName_In;
  CS_DatabaseFileName_In.Set(P_CharArr_DB);

  CFits F_Image;
  cout << "MExtractSum::main: Starting F_Image.SetFileName(" << CS_FitsFileName_In.Get() << ")" << endl;
  if (!F_Image.SetFileName(CS_FitsFileName_In))
  {
    cout << "MExtractSum::main: ERROR: F_Image.SetFileName(" << CS_FitsFileName_In.Get() << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Read FitsFile
  cout << "MExtractSum::main: Starting F_Image.ReadArray()" << endl;
  if (!F_Image.ReadArray())
  {
    cout << "MExtractSum::main: ERROR: F_Image.ReadArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// convert ADUs to photons
  cout << "MExtractSum::main: (919,817) = " << F_Image.GetPixArray()(919,817) << endl;
  F_Image.GetPixArray() *= D_Gain;
  cout << "MExtractSum::main: after gain: (919,817) = " << F_Image.GetPixArray()(919,817) << endl;

  /// Set DatabaseFileName_In
  cout << "MExtractSum::main: Starting F_Image.SetDatabaseFileName(" << CS_DatabaseFileName_In << ")" << endl;
  if (!F_Image.SetDatabaseFileName(CS_DatabaseFileName_In))
  {
    cout << "MExtractSum::main: ERROR: F_Image.SetDatabaseFileName(" << CS_DatabaseFileName_In << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Read DatabaseFileName_In
  cout << "MExtractSum::main: Starting F_Image.ReadDatabaseEntry()" << endl;
  if (!F_Image.ReadDatabaseEntry())
  {
    cout << "MExtractSum::main: ERROR: F_Image.ReadDatabaseEntry() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Calculate Trace Functions
  cout << "MExtractSum::main: Starting F_Image.CalcTraceFunctions()" << endl;
  if (!F_Image.CalcTraceFunctions())
  {
    cout << "MExtractSum::main: ERROR: F_Image.CalcTraceFunctions() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MExtractSum::main: after trace function (919,817) = " << F_Image.GetPixArray()(919,817) << endl;
//exit(EXIT_FAILURE);
  cout << "MExtractSum::main: P_CS_ErrIn = <" << *P_CS_ErrIn << ">" << endl;
  if (P_CS_ErrIn->GetLength() > 1){
    /// Set ErrFileName_In
    cout << "MExtractSum::main: Starting F_Image.SetErrFileName(" << *P_CS_ErrIn << ")" << endl;
    if (!F_Image.SetErrFileName(*P_CS_ErrIn))
    {
      cout << "MExtractSum::main: ERROR: F_Image.SetErrFileName(" << *P_CS_ErrIn << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    /// Read Error image
    cout << "MExtractSum::main: Starting F_Image.ReadErrArray()" << endl;
    if (!F_Image.ReadErrArray())
    {
      cout << "MExtractSum::main: ERROR: F_Image.ReadErrArray() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
  }

  /// Write aperture header information
  F_Image.WriteApHead(CString("aphead_")+CS_FitsFileName_In+CString(".head"));

  CFits F_OutImage;
  /// Set CS_FitsFileName_In
  cout << "MExtractSum::main: Starting F_OutImage.SetFileName(" << CS_FitsFileName_In << ")" << endl;
  if (!F_OutImage.SetFileName(CS_FitsFileName_In))
  {
    cout << "MExtractSum::main: ERROR: F_OutImage.SetFileName(" << CS_FitsFileName_In << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  ///Read FitsFile
  cout << "MExtractSum::main: Starting F_OutImage.ReadArray()" << endl;
  if (!F_OutImage.ReadArray())
  {
    cout << "MExtractSum::main: ERROR: F_OutImage.ReadArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Set CS_FitsFileName_Out
  cout << "MExtractSum::main: Starting F_OutImage.SetFileName(" << CS_FitsFileName_Out << ")" << endl;
  if (!F_OutImage.SetFileName(CS_FitsFileName_Out))
  {
    cout << "MExtractSum::main: ERROR: F_OutImage.SetFileName(" << CS_FitsFileName_Out << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// change size of F_OutImage to (NApertures x NRows)
  if (!F_OutImage.SetNCols(F_Image.GetNRows()))
  {
    cout << "MExtractSum::main: ERROR: F_OutImage.SetNCols(" << F_Image.GetNRows() << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  int I_NAps_Out = F_Image.Get_NApertures();
  if (B_Apertures_Set)
    I_NAps_Out = I_A1_Apertures.size();
  if (!F_OutImage.SetNRows(F_Image.Get_NApertures()))
  {
    cout << "MExtractSum::main: ERROR: F_OutImage.SetNRows(" << F_Image.Get_NApertures() << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Write EcFromProfileOut 1D
  cout << "MExtractSum::main: Before extract: (919,817) = " << F_Image.GetPixArray()(919,817) << endl;
  cout << "MExtractSum::main: Starting ExtractSimpleSum: F_Image.Get_NApertures() = " << F_Image.Get_NApertures() << endl;
  if (!F_Image.ExtractSimpleSum(CS_A1_Args, PP_Args))
  {
    cout << "MExtractSum::main: ERROR: F_Image.ExtractSimpleSum() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  F_OutImage.GetPixArray() = F_Image.GetLastExtracted();//.transpose(secondDim, firstDim);
  if (!F_OutImage.WriteArray())
  {
    cout << "MExtractSum::main: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Write aperture header information
  F_Image.WriteApHead(CString("aphead_")+CS_FitsFileName_In+CString(".head"));

  /// Create ErrOutEc
  if (P_CS_ErrOut->GetLength() > 1){// || P_CS_ErrOutEc->GetLength() > 1){
    cout << "MExtractSum::main: Starting F_Image.ExtractErrors()" << endl;
    if (!F_Image.ExtractSimpleSum(F_Image.GetErrArray(),CS_A1_Args, PP_Args))
    {
      cout << "MExtractSum::main: ERROR: F_Image.ExtractErrors() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
  }

  cout << "MExtractSum::main: Starting F_OutImage.WriteArray()" << endl;
  if (!F_OutImage.WriteArray())
  {
    cout << "MExtractSum::main: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Write ErrFromProfile 1D
  if (P_CS_ErrOut->GetLength() > 1)
  {
    if (!F_OutImage.SetFileName(*P_CS_ErrOut))
    {
      cout << "MExtractSum::main: ERROR: F_OutImage.SetFileName() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
    cout << "MExtractSum::main: Starting to write ErrFromProfile" << endl;
    F_OutImage.GetPixArray() = F_Image.GetLastExtracted();//.transpose(secondDim, firstDim);
    if (!F_OutImage.WriteArray())
    {
      cout << "MExtractSum::main: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
  }

  delete(P_CS);
  delete(P_CS_ErrIn);
  delete(P_CS_EcOut);
  delete(P_CS_ErrOut);
  return EXIT_SUCCESS;
}
