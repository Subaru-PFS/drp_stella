/*
author: Andreas Ritter
created: 04/12/2007
last edited: 05/05/2007
compiler: g++ 4.0
basis machine: Ubuntu Linux 6.06
*/

#include "MExtractSpecFromProfile.h"

using namespace std;

int main(int argc, char *argv[])
{
  cout << "MExtractSpecFromProfile::main: argc = " << argc << endl;
  if (argc < 4)
  {
    cout << "MExtractSpecFromProfile::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: extractfromprofile char[] FitsFileName_In, char[] DatabaseFileName_In, char[] FitsFileName_Profile_In, char[] FitsFileName_Out[, ERR_IN=char[]][, ERR_OUT_EC=char[]][, SKY_OUT_EC=char[]][, SKY_OUT_2D=char[]][, SKY_ERR_OUT_EC=char[]][, IM_REC_OUT=char[]][, MASK_OUT=char[]][,AREA=[int(xmin),int(xmax),int(ymin),int(ymax)]][,WITH_SKY=bool]" << endl;
    exit(EXIT_FAILURE);
  }

  Array<int, 1> I_A1_Area(4);
  bool B_WithSky = false;

  Array<CString, 1> CS_A1_Args(10);
  CS_A1_Args = CString("\0");
  void **PP_Args;
  PP_Args = (void**)malloc(sizeof(void*) * 10);
//  int I_NArgs = 0;
  CString CS(" ");
  CString CS_comp(" ");
  CString *P_CS = new CString(" ");
  CString *P_CS_ErrIn = new CString(" ");
  CString *P_CS_ErrEcOut = new CString(" ");
  CString *P_CS_SkyOutEc = new CString(" ");
  CString *P_CS_SkyRec = new CString(" ");
  CString *P_CS_SkyErrOutEc = new CString(" ");
  CString *P_CS_ImRecOut = new CString(" ");
  CString *P_CS_MaskOut = new CString(" ");

  char *P_CharArr_In = (char*)argv[1];
  char *P_CharArr_DB = (char*)argv[2];
  char *P_CharArr_Prof_In = (char*)argv[3];
  char *P_CharArr_Out = (char*)argv[4];

  /// read optional parameters
  for (int i = 5; i <= argc; i++){
    CS.Set((char*)argv[i]);
    cout << "MExtractSpecFromProfile Reading Parameter " << CS << endl;

    CS_comp.Set("ERR_OUT_EC");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtractSpecFromProfile::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS_ErrEcOut);
        P_CS_ErrEcOut = CS.SubString(CS_comp.GetLength()+1);
        cout << "MExtractSpecFromProfile::main: P_CS_ErrEcOut set to " << *P_CS_ErrEcOut << endl;
      }
    }

    CS_comp.Set("ERR_IN");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtractSpecFromProfile::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS_ErrIn);
        P_CS_ErrIn = CS.SubString(CS_comp.GetLength()+1);
        cout << "MExtractSpecFromProfile::main: ERR_IN set to " << *P_CS_ErrIn << endl;
      }
    }

    CS_comp.Set("SKY_OUT_EC");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtractSpecFromProfile::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS_SkyOutEc);
        P_CS_SkyOutEc = CS.SubString(CS_comp.GetLength()+1);
        cout << "MExtractSpecFromProfile::main: SKY_OUT_EC set to " << *P_CS_SkyOutEc << endl;
      }
    }

    CS_comp.Set("SKY_OUT_2D");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtractSpecFromProfile::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS_SkyRec);
        P_CS_SkyRec = CS.SubString(CS_comp.GetLength()+1);
        cout << "MExtractSpecFromProfile::main: SKY_OUT_2D set to " << *P_CS_SkyRec << endl;
      }
    }

    CS_comp.Set("SKY_ERR_OUT_EC");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtractSpecFromProfile::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS_SkyErrOutEc);
        P_CS_SkyErrOutEc = CS.SubString(CS_comp.GetLength()+1);
        cout << "MExtractSpecFromProfile::main: SKY_ERR_OUT_EC set to " << *P_CS_SkyErrOutEc << endl;
      }
    }

    CS_comp.Set("IM_REC_OUT");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtractSpecFromProfile::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS_ImRecOut);
        P_CS_ImRecOut = CS.SubString(CS_comp.GetLength()+1);
        cout << "MExtractSpecFromProfile::main: IM_REC_OUT set to " << *P_CS_ImRecOut << endl;
      }
    }

    CS_comp.Set("MASK_OUT");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtractSpecFromProfile::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS_MaskOut);
        P_CS_MaskOut = CS.SubString(CS_comp.GetLength()+1);
        cout << "MExtractSpecFromProfile::main: MASK_OUT set to " << *P_CS_MaskOut << endl;
      }
    }

    /// AREA
    CS_comp.Set("AREA");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtractSpecFromProfile::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        CString cs_temp;
        cs_temp.Set(",");
        int i_pos_a = CS_comp.GetLength()+2;
        int i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        cout << "MExtractSpecFromProfile: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
        delete(P_CS);
        P_CS = CS.SubString(i_pos_a,i_pos_b-1);
        cout << "MExtractSpecFromProfile: P_CS set to " << *P_CS << endl;
        I_A1_Area(0) = (int)(atoi(P_CS->Get()));
        cout << "MExtractSpecFromProfile: I_A1_Area(0) set to " << I_A1_Area(0) << endl;

        i_pos_a = i_pos_b+1;
        i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        cout << "MExtractSpecFromProfile: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
        delete(P_CS);
        P_CS = CS.SubString(i_pos_a,i_pos_b-1);
        cout << "MExtractSpecFromProfile: P_CS set to " << *P_CS << endl;
        I_A1_Area(1) = (int)(atoi(P_CS->Get()));
        cout << "MExtractSpecFromProfile: I_A1_Area(1) set to " << I_A1_Area(1) << endl;

        i_pos_a = i_pos_b+1;
        i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        cout << "MExtractSpecFromProfile: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
        delete(P_CS);
        P_CS = CS.SubString(i_pos_a,i_pos_b-1);
        cout << "MExtractSpecFromProfile: P_CS set to " << *P_CS << endl;
        I_A1_Area(2) = (int)(atoi(P_CS->Get()));
        cout << "MExtractSpecFromProfile: I_A1_Area(2) set to " << I_A1_Area(2) << endl;

        i_pos_a = i_pos_b+1;
        cs_temp.Set("]");
        i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        if (i_pos_b < 0){
          cs_temp.Set(")");
          i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        }
        cout << "MExtractSpecFromProfile: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
        delete(P_CS);
        P_CS = CS.SubString(i_pos_a,i_pos_b-1);
        cout << "MExtractSpecFromProfile: P_CS set to " << *P_CS << endl;
        I_A1_Area(3) = (int)(atoi(P_CS->Get()));
        cout << "MExtractSpecFromProfile: I_A1_Area(3) set to " << I_A1_Area(3) << endl;

        CS_A1_Args(0) = CString("AREA");
        PP_Args[0] = &I_A1_Area;
        cout << "MExtractSpecFromProfile::main: AREA set to " << I_A1_Area << endl;
      }
    }

    /// WITH_SKY
    CS_comp.Set("WITH_SKY");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtractSpecFromProfile::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS);
        P_CS = CS.SubString(CS.CharPos('=')+1);
        if (P_CS->EqualValue(CString("true"))){
          B_WithSky = true;
          CS_A1_Args(1).Set("WITH_SKY");
          PP_Args[1] = &B_WithSky;
        }
      }
    }
  }
  time_t seconds;
//  if (argc == 8)
//  {
//    I_SwathWidth = (int)(atoi((char*)argv[7]));
//    cout << "MExtractSpecFromProfile::main: I_SwathWidth set to " << I_SwathWidth << endl;
  CString CS_FitsFileName_In;
  CS_FitsFileName_In.Set(P_CharArr_In);
  CString CS_FitsFileName_Profile_In;
  CS_FitsFileName_Profile_In.Set(P_CharArr_Prof_In);
  CString CS_FitsFileName_Out;
  CS_FitsFileName_Out.Set(P_CharArr_Out);
  CString CS_DatabaseFileName_In;
  CS_DatabaseFileName_In.Set(P_CharArr_DB);

  CFits F_Image;
  cout << "MExtractSpecFromProfile::main: Starting F_Image.SetFileName(" << CS_FitsFileName_In.Get() << ")" << endl;
  if (!F_Image.SetFileName(CS_FitsFileName_In))
  {
    cout << "MExtractSpecFromProfile::main: ERROR: F_Image.SetFileName(" << CS_FitsFileName_In.Get() << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Read FitsFile
  cout << "MExtractSpecFromProfile::main: Starting F_Image.ReadArray()" << endl;
  if (!F_Image.ReadArray())
  {
    cout << "MExtractSpecFromProfile::main: ERROR: F_Image.ReadArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Set DatabaseFileName_In
  cout << "MExtractSpecFromProfile::main: Starting F_Image.SetDatabaseFileName(" << CS_DatabaseFileName_In << ")" << endl;
  if (!F_Image.SetDatabaseFileName(CS_DatabaseFileName_In))
  {
    cout << "MExtractSpecFromProfile::main: ERROR: F_Image.SetDatabaseFileName(" << CS_DatabaseFileName_In << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Read DatabaseFileName_In
  cout << "MExtractSpecFromProfile::main: Starting F_Image.ReadDatabaseEntry()" << endl;
  if (!F_Image.ReadDatabaseEntry())
  {
    cout << "MExtractSpecFromProfile::main: ERROR: F_Image.ReadDatabaseEntry() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Calculate Trace Functions
  cout << "MExtractSpecFromProfile::main: Starting F_Image.CalcTraceFunctions()" << endl;
  if (!F_Image.CalcTraceFunctions())
  {
    cout << "MExtractSpecFromProfile::main: ERROR: F_Image.CalcTraceFunctions() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  cout << "MExtractSpecFromProfile::main: P_CS_ErrIn = " << *P_CS_ErrIn << ")" << endl;
  if (P_CS_ErrIn->GetLength() > 1){
    /// Set ErrFileName_In
    cout << "MExtractSpecFromProfile::main: Starting F_Image.SetErrFileName(" << *P_CS_ErrIn << ")" << endl;
    if (!F_Image.SetErrFileName(*P_CS_ErrIn))
    {
      cout << "MExtractSpecFromProfile::main: ERROR: F_Image.SetErrFileName(" << *P_CS_ErrIn << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    /// Read Error image
    cout << "MExtractSpecFromProfile::main: Starting F_Image.ReadErrArray()" << endl;
    if (!F_Image.ReadErrArray())
    {
      cout << "MExtractSpecFromProfile::main: ERROR: F_Image.ReadErrArray() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
  }

  ///Read Profile Image
  if (!F_Image.Read_ProfArray(CS_FitsFileName_Profile_In)){
    cout << "MExtractSpecFromProfile::main: ERROR: F_Image.Read_ProfArray(" << CS_FitsFileName_Profile_In << ") returned FALSE => Returning FALSE" << endl;
    exit(EXIT_FAILURE);
  }

  /// Write aperture header information
  F_Image.WriteApHead(CString("aphead_")+CS_FitsFileName_In+CString(".head"));

  /// ExtractSpecFromProfile
  cout << "MExtractSpecFromProfile::main: Starting ExtractSpecFromProfile" << endl;
  if (!F_Image.ExtractSpecFromProfile(F_Image.GetPixArray(), CS_A1_Args, PP_Args))
  {
    cout << "MExtractSpecFromProfile::main: ERROR: F_Image.ExtractSpecFromProfile() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  CFits F_OutImage;
  /// Set CS_FitsFileName_In
  cout << "MExtractSpecFromProfile::main: Starting F_OutImage.SetFileName(" << CS_FitsFileName_In << ")" << endl;
  if (!F_OutImage.SetFileName(CS_FitsFileName_In))
  {
    cout << "MExtractSpecFromProfile::main: ERROR: F_OutImage.SetFileName(" << CS_FitsFileName_In << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  ///Read FitsFile
  cout << "MExtractSpecFromProfile::main: Starting F_OutImage.ReadArray()" << endl;
  if (!F_OutImage.ReadArray())
  {
    cout << "MExtractSpecFromProfile::main: ERROR: F_OutImage.ReadArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// 2D
  /// Reconstructed object image
  if (P_CS_ImRecOut->GetLength() > 1){
    cout << "MExtractSpecFromProfile::main: Writing Reconstructed Object image" << endl;
    if (!F_OutImage.SetFileName(*P_CS_ImRecOut))
    {
      cout << "MExtractSpecFromProfile::main: ERROR: F_OutImage.SetFileName(" << *P_CS_ImRecOut << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
    F_OutImage.GetPixArray() = F_Image.GetRecFitArray();
    if (!F_OutImage.WriteArray())
    {
      cout << "MExtractSpecFromProfile::main: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
  }

  /// Reconstructed sky image
  if (B_WithSky){
    if (P_CS_SkyRec->GetLength() > 1){
      cout << "MExtractSpecFromProfile::main: Writing Reconstructed Sky image" << endl;
      if (!F_OutImage.SetFileName(*P_CS_SkyRec))
      {
        cout << "MExtractSpecFromProfile::main: ERROR: F_OutImage.SetFileName(" << *P_CS_SkyRec << ") returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
      F_OutImage.GetPixArray() = F_Image.GetRecSkyArray();
      if (!F_OutImage.WriteArray())
      {
        cout << "MExtractSpecFromProfile::main: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
    }
  }

  /// Mask image
  if (P_CS_MaskOut->GetLength() > 1){
    cout << "MExtractSpecFromProfile::main: Writing Mask image" << endl;
    if (!F_OutImage.SetFileName(*P_CS_MaskOut))
    {
      cout << "MExtractSpecFromProfile::main: ERROR: F_OutImage.SetFileName(" << *P_CS_MaskOut << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
    Array<int, 2> D_A2_Mask(F_Image.GetNRows(), F_Image.GetNCols());
    D_A2_Mask = F_Image.GetMaskArray();
    for (int i_row = 0; i_row < F_Image.GetNRows(); i_row++){
      for (int i_col = 0; i_col < F_Image.GetNRows(); i_col++){
        (F_OutImage.GetPixArray())(i_row, i_col) = double(D_A2_Mask(i_row, i_col));
      }
    }
    if (!F_OutImage.WriteArray())
    {
      cout << "MExtractSpecFromProfile::main: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
  }

  /// change size of F_OutImage to (NApertures x NRows)
  if (!F_OutImage.SetNCols(F_Image.GetNRows()))
  {
    cout << "MExtractSpecFromProfile::main: ERROR: F_OutImage.SetNCols(" << F_Image.GetNRows() << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  if (!F_OutImage.SetNRows(F_Image.Get_NApertures()))
  {
    cout << "MExtractSpecFromProfile::main: ERROR: F_OutImage.SetNRows(" << F_Image.Get_NApertures() << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Set CS_FitsFileName_Out
  cout << "MExtractSpecFromProfile::main: Starting F_OutImage.SetFileName(" << CS_FitsFileName_Out << ")" << endl;
  if (!F_OutImage.SetFileName(CS_FitsFileName_Out))
  {
    cout << "MExtractSpecFromProfile::main: ERROR: F_OutImage.SetFileName(" << CS_FitsFileName_Out << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Write EcOut 1D
  cout << "MExtractSpecFromProfile::main: Starting to write EcFromProfileOut" << endl;
  F_OutImage.GetPixArray() = F_Image.GetLastExtracted();//.transpose(secondDim, firstDim);
  if (!F_OutImage.WriteArray())
  {
    cout << "MExtractSpecFromProfile::main: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Write aperture header information
  F_Image.WriteApHead(CString("aphead_")+CS_FitsFileName_In+CString(".head"));

  /// Create ErrOutEc
///  if (P_CS_ErrEcOut->GetLength() > 1){// || P_CS_ErrOutEc->GetLength() > 1){
///    cout << "MExtractSpecFromProfile::main: Starting F_Image.ExtractErrors()" << endl;
///    if (!F_Image.ExtractErrors())
///    {
///      cout << "MExtractSpecFromProfile::main: ERROR: F_Image.ExtractErrors() returned FALSE!" << endl;
///      exit(EXIT_FAILURE);
///    }
///  }

  /// Write ErrEcOut
  if (P_CS_ErrEcOut->GetLength() > 1)
  {
    if (!F_OutImage.SetFileName(*P_CS_ErrEcOut))
    {
      cout << "MExtractSpecFromProfile::main: ERROR: F_OutImage.SetFileName() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
    cout << "MExtractSpecFromProfile::main: Starting to write ErrFromProfile" << endl;
    F_OutImage.GetPixArray() = F_Image.GetErrorsEc();//.transpose(secondDim, firstDim);
    if (!F_OutImage.WriteArray())
    {
      cout << "MExtractSpecFromProfile::main: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
  }

  delete(P_CS);
  delete(P_CS_ErrIn);
  delete(P_CS_ErrEcOut);
  delete(P_CS_SkyOutEc);
  delete(P_CS_SkyRec);
  delete(P_CS_SkyErrOutEc);
  delete(P_CS_ImRecOut);
  delete(P_CS_MaskOut);
  return EXIT_SUCCESS;
}
