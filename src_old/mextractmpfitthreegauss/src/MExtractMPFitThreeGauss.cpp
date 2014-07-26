/*
 * author: Andreas Ritter
 * created: 07/10/2013
 * last edited: 07/10/2013
 * compiler: g++ 4.4
 * basis machine: Arch Linux
 */

#include "MExtractMPFitThreeGauss.h"

using namespace std;

int main(int argc, char *argv[])
{
  cout << "MExtractMPFitThreeGauss::main: argc = " << argc << endl;
  if (argc < 8)
  {
    cout << "MExtractMPFitThreeGauss::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: extractmpfitthreegauss <char[] [@]FitsFileName[List]_In> <char[] [@]DatabaseFileName[List]_In> <char[] [@]FitsFileName[List]_Out> <double Gain> <int B_WithBackground[0,1]> <[double(min_sdev),double(max_sdev)]> <[double(max_mean_offset_left_of_aperture_trace),double(max_mean_offset_right_of_aperture_trace)]> [ERR_IN=char[@]] [ERR_OUT_EC=char[@]] [AREA=[int(xmin),int(xmax),int(ymin),int(ymax)]] [APERTURES=<char[] ApertureFile_In>]" << endl;
    exit(EXIT_FAILURE);
  }
  Array<int, 1> I_A1_Area(4);

  Array<CString, 1> CS_A1_Args(2);
  CS_A1_Args = CString("\0");
  void **PP_Args;
  PP_Args = (void**)malloc(sizeof(void*) * 2);
  CString CS(" ");
  CString CS_comp(" ");
  CString CS_List("@");
  CString *P_CS = new CString(" ");
  CString *P_CS_ErrIn = new CString(" ");
  CString *P_CS_EcOut = new CString(" ");
  CString *P_CS_ErrOut = new CString(" ");
  CString *P_CS_ApertureList_In = new CString(" ");

  char *P_CharArr_In = (char*)argv[1];
  char *P_CharArr_DB = (char*)argv[2];
  char *P_CharArr_Out = (char*)argv[3];
  CString CS_FitsFileName_In;
  CS_FitsFileName_In.Set(P_CharArr_In);
  bool B_FitsFileName_In_IsList = CS_FitsFileName_In.IsList();
  CString CS_FitsFileName_Out;
  CS_FitsFileName_Out.Set(P_CharArr_Out);
  bool B_FitsFileName_Out_IsList = CS_FitsFileName_Out.IsList();
  CString CS_DatabaseFileName_In;
  CS_DatabaseFileName_In.Set(P_CharArr_DB);
  bool B_DatabaseFileName_In_IsList = CS_DatabaseFileName_In.IsList();
  bool B_ErrIn_IsList = false;
  bool B_ErrOutEc_IsList = false;
  CString CS_Gain((char*)(argv[4]));
  double D_Gain = 0.;
  if (!CS_Gain.AToD(D_Gain)){
    cout << "MExtractMPFitThreeGauss::main: ERROR: CS_Gain(=" << CS_Gain << ").AToD(D_Gain) returning FALSE" << endl;
    exit(EXIT_FAILURE);
  }
  CString CS_WithBackground((char*)(argv[5]));
  int I_WithBackground = 0.;
  if (!CS_WithBackground.AToI(I_WithBackground)){
    cout << "MExtractMPFitThreeGauss::main: ERROR: CS_WithBackground(=" << CS_WithBackground << ").AToI(I_WithBackground) returning FALSE" << endl;
    exit(EXIT_FAILURE);
  }
  bool B_WithBackground = false;
  if (I_WithBackground == 1)
    B_WithBackground = true;
  CString CS_SDevLimits((char*)(argv[6]));
  CString CS_MeanLimits((char*)(argv[7]));
  Array<double, 1> D_A1_SDevLimits(2);
  Array<double, 1> D_A1_MeanLimits(2);
  CString *P_CS_Temp = CS_SDevLimits.SubString(1,CS_SDevLimits.CharPos(',')-1);
  double D_Temp = 0.;
  if (!P_CS_Temp->AToD(D_Temp)){
    cout << "MExtractMPFitThreeGauss::main: ERROR: P_CS_Temp(=" << *P_CS_Temp << ")->AToD(D_Temp) returnedFALSE" << endl;
    exit(EXIT_FAILURE);
  }
  D_A1_SDevLimits(0) = D_Temp;
  delete(P_CS_Temp);
  P_CS_Temp = CS_SDevLimits.SubString(CS_SDevLimits.CharPos(',')+1,CS_SDevLimits.GetLength()-2);
  if (!P_CS_Temp->AToD(D_Temp)){
    cout << "MExtractMPFitThreeGauss::main: ERROR: P_CS_Temp(=" << *P_CS_Temp << ")->AToD(D_Temp) returnedFALSE" << endl;
    exit(EXIT_FAILURE);
  }
  D_A1_SDevLimits(1) = D_Temp;
  delete(P_CS_Temp);
  cout << "MExtractMPFitThreeGauss::main: D_A1_SDevLimits set to " << D_A1_SDevLimits << endl;

  P_CS_Temp = CS_MeanLimits.SubString(1,CS_MeanLimits.CharPos(',')-1);
  if (!P_CS_Temp->AToD(D_Temp)){
    cout << "MExtractMPFitThreeGauss::main: ERROR: P_CS_Temp(=" << *P_CS_Temp << ")->AToD(D_Temp) returnedFALSE" << endl;
    exit(EXIT_FAILURE);
  }
  D_A1_MeanLimits(0) = D_Temp;
  delete(P_CS_Temp);
  P_CS_Temp = CS_MeanLimits.SubString(CS_MeanLimits.CharPos(',')+1,CS_MeanLimits.GetLength()-2);
  if (!P_CS_Temp->AToD(D_Temp)){
    cout << "MExtractMPFitThreeGauss::main: ERROR: P_CS_Temp(=" << *P_CS_Temp << ")->AToD(D_Temp) returnedFALSE" << endl;
    exit(EXIT_FAILURE);
  }
  D_A1_MeanLimits(1) = D_Temp;
  delete(P_CS_Temp);
  cout << "MExtractMPFitThreeGauss::main: D_A1_MeanLimits set to " << D_A1_MeanLimits << endl;

  Array<int, 1> I_A1_Apertures(1);
  I_A1_Apertures = 0;
  Array<CString, 1> CS_A1_ApNum(1);
  CS_A1_ApNum(0).Set(" ");
  bool B_Apertures_Set = false;

  /// read optional parameters
  for (int i = 8; i <= argc; i++){
    CS.Set((char*)argv[i]);
    cout << "MExtractMPFitThreeGauss: Reading Parameter " << CS << endl;

    CS_comp.Set("ERR_IN");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtractMPFitThreeGauss::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS_ErrIn);
        P_CS_ErrIn = CS.SubString(CS_comp.GetLength()+1);
        cout << "MExtractMPFitThreeGauss::main: P_CS_ERRIN set to " << *P_CS_ErrIn << endl;
      }
      B_ErrIn_IsList = P_CS_ErrIn->IsList();
    }

    CS_comp.Set("ERR_OUT_EC");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtractMPFitThreeGauss::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS_ErrOut);
        P_CS_ErrOut = CS.SubString(CS_comp.GetLength()+1);
        cout << "MExtractMPFitThreeGauss::main: P_CS_ErrOut set to " << *P_CS_ErrOut << endl;
      }
      B_ErrOutEc_IsList = P_CS_ErrOut->IsList();
    }

    CS_comp.Set("APERTURES");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtractMPFitThreeGauss::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS_ApertureList_In);
        P_CS_ApertureList_In = CS.SubString(CS_comp.GetLength()+1);
        cout << "MExtractMPFitThreeGauss::main: P_CS_ApertureList_In set to " << *P_CS_ApertureList_In << endl;
	if (!CS_comp.ReadFileLinesToStrArr(*P_CS_ApertureList_In, CS_A1_ApNum)){
	  cout << "MExtractMPFitThreeGauss::main: ERROR: ReadFileLinesToStrArr(" << *P_CS_ApertureList_In << ") returned FALSE" << endl;
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
      cout << "MExtractMPFitThreeGauss::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        CString cs_temp;
        cs_temp.Set(",");
        int i_pos_a = CS_comp.GetLength()+2;
        int i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        cout << "MExtractMPFitThreeGauss: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
        delete(P_CS);
        P_CS = CS.SubString(i_pos_a,i_pos_b-1);
        cout << "MExtractMPFitThreeGauss: P_CS set to " << *P_CS << endl;
        I_A1_Area(0) = (int)(atoi(P_CS->Get()));
        cout << "MExtractMPFitThreeGauss: I_A1_Area(0) set to " << I_A1_Area(0) << endl;

        i_pos_a = i_pos_b+1;
        i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        cout << "MExtractMPFitThreeGauss: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
        delete(P_CS);
        P_CS = CS.SubString(i_pos_a,i_pos_b-1);
        cout << "MExtractMPFitThreeGauss: P_CS set to " << *P_CS << endl;
        I_A1_Area(1) = (int)(atoi(P_CS->Get()));
        cout << "MExtractMPFitThreeGauss: I_A1_Area(1) set to " << I_A1_Area(1) << endl;

        i_pos_a = i_pos_b+1;
        i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        cout << "MExtractMPFitThreeGauss: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
        delete(P_CS);
        P_CS = CS.SubString(i_pos_a,i_pos_b-1);
        cout << "MExtractMPFitThreeGauss: P_CS set to " << *P_CS << endl;
        I_A1_Area(2) = (int)(atoi(P_CS->Get()));
        cout << "MExtractMPFitThreeGauss: I_A1_Area(2) set to " << I_A1_Area(2) << endl;

        i_pos_a = i_pos_b+1;
        cs_temp.Set("]");
        i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        if (i_pos_b < 0){
          cs_temp.Set(")");
          i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        }
        cout << "MExtractMPFitThreeGauss: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
        delete(P_CS);
        P_CS = CS.SubString(i_pos_a,i_pos_b-1);
        cout << "MExtractMPFitThreeGauss: P_CS set to " << *P_CS << endl;
        I_A1_Area(3) = (int)(atoi(P_CS->Get()));
        cout << "MExtractMPFitThreeGauss: I_A1_Area(3) set to " << I_A1_Area(3) << endl;

        CS_A1_Args(0) = CString("AREA");
        PP_Args[0] = &I_A1_Area;
        cout << "MExtractMPFitThreeGauss::main: AREA set to " << I_A1_Area << endl;
      }
    }
  }
  time_t seconds;
//  if (argc == 8)
//  {
//    I_SwathWidth = (int)(atoi((char*)argv[7]));
//    cout << "MExtractMPFitThreeGauss::main: I_SwathWidth set to " << I_SwathWidth << endl;

  Array<CString, 1> CS_A1_FitsFileNames_In(1);
  CS_A1_FitsFileNames_In(0) = CS_FitsFileName_In;
  if (B_FitsFileName_In_IsList){
    if (!CS_comp.ReadFileLinesToStrArr(CS_FitsFileName_In, CS_A1_FitsFileNames_In)){
      cout << "MExtractMPFitThreeGauss::main: ERROR: ReadFileLinesToStrArr(CS_FitsFileName_In=<" << CS_FitsFileName_In << ">) returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
  }

  Array<CString, 1> CS_A1_FitsFileNames_Out(1);
  CS_A1_FitsFileNames_Out(0) = CS_FitsFileName_Out;
  if (B_FitsFileName_Out_IsList){
    if (!CS_comp.ReadFileLinesToStrArr(CS_FitsFileName_Out, CS_A1_FitsFileNames_Out)){
      cout << "MExtractMPFitThreeGauss::main: ERROR: ReadFileLinesToStrArr(CS_FitsFileName_Out=<" << CS_FitsFileName_Out << ">) returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    if (CS_A1_FitsFileNames_In.size() != CS_A1_FitsFileNames_Out.size()){
      cout << "MExtractMPFitThreeGauss::main: ERROR: CS_A1_FitsFileNames_In.size(=" << CS_A1_FitsFileNames_In.size() << ") != CS_A1_FitsFileNames_Out.size(=" << CS_A1_FitsFileNames_Out.size() << ") => Returning FALSE" << endl;
      exit(EXIT_FAILURE);
    }
  }
  if (B_FitsFileName_In_IsList && !B_FitsFileName_Out_IsList){
    cout << "MExtractMPFitThreeGauss::main: ERROR: B_FitsFileName_In_IsList = TRUE but B_FitsFileName_Out_IsList = FALSE" << endl;
    exit(EXIT_FAILURE);
  }

  Array<CString, 1> CS_A1_DatabaseFileNames_In(CS_A1_FitsFileNames_In.size());
  if (B_DatabaseFileName_In_IsList){
    if (!CS_comp.ReadFileLinesToStrArr(CS_DatabaseFileName_In, CS_A1_DatabaseFileNames_In)){
      cout << "MExtractMPFitThreeGauss::main: ERROR: ReadFileLinesToStrArr(CS_DatabaseFileName_In=<" << CS_DatabaseFileName_In << ">) returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    if (CS_A1_FitsFileNames_In.size() != CS_A1_DatabaseFileNames_In.size()){
      cout << "MExtractMPFitThreeGauss::main: ERROR: CS_A1_FitsFileNames_In.size(=" << CS_A1_FitsFileNames_In.size() << ") != CS_A1_DatabaseFileNames_In.size(=" << CS_A1_DatabaseFileNames_In.size() << ") => Returning FALSE" << endl;
      exit(EXIT_FAILURE);
    }
  }
  else{
    CS_A1_DatabaseFileNames_In = CS_DatabaseFileName_In;
  }

  Array<CString, 1> CS_A1_ErrIn(1);
  if (P_CS_ErrIn->GetLength() > 1){
    if (B_ErrIn_IsList){
      if (!CS_comp.ReadFileLinesToStrArr(*P_CS_ErrIn, CS_A1_ErrIn)){
        cout << "MExtractMPFitThreeGauss::main: ERROR: ReadFileLinesToStrArr(*P_CS_ErrIn=<" << *P_CS_ErrIn << ">) returned FALSE" << endl;
        exit(EXIT_FAILURE);
      }
      if (CS_A1_FitsFileNames_In.size() != CS_A1_ErrIn.size()){
        cout << "MExtractMPFitThreeGauss::main: ERROR: CS_A1_FitsFileNames_In.size(=" << CS_A1_FitsFileNames_In.size() << ") != CS_A1_ErrIn.size(=" << CS_A1_ErrIn.size() << ") => Returning FALSE" << endl;
        exit(EXIT_FAILURE);
      }
    }
    else{
      CS_A1_ErrIn(0) = *P_CS_ErrIn;
    }
  }

  Array<CString, 1> CS_A1_ErrOutEc(1);
  if (P_CS_ErrOut->GetLength() > 1){
    if (B_ErrOutEc_IsList){
      if (!CS_comp.ReadFileLinesToStrArr(*P_CS_ErrOut, CS_A1_ErrOutEc)){
        cout << "MExtractMPFitThreeGauss::main: ERROR: ReadFileLinesToStrArr(*P_CS_ErrOut=<" << *P_CS_ErrOut << ">) returned FALSE" << endl;
        exit(EXIT_FAILURE);
      }
      if (CS_A1_FitsFileNames_In.size() != CS_A1_ErrOutEc.size()){
        cout << "MExtractMPFitThreeGauss::main: ERROR: CS_A1_FitsFileNames_In.size(=" << CS_A1_FitsFileNames_In.size() << ") != CS_A1_ErrOutEc.size(=" << CS_A1_ErrOutEc.size() << ") => Returning FALSE" << endl;
        exit(EXIT_FAILURE);
      }
    }
    else{
      CS_A1_ErrOutEc(0) = *P_CS_ErrOut;
    }
  }

  CFits F_Image;
  CFits F_OutImage;
  for (int i_file=0; i_file<CS_A1_FitsFileNames_In.size(); i_file++){
    cout << "MExtractMPFitThreeGauss::main: Starting F_Image.SetFileName(" << CS_A1_FitsFileNames_In(i_file) << ")" << endl;
    if (!F_Image.SetFileName(CS_A1_FitsFileNames_In(i_file))){
      cout << "MExtractMPFitThreeGauss::main: ERROR: F_Image.SetFileName(" << CS_A1_FitsFileNames_In(i_file) << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    /// Read FitsFile
    cout << "MExtractMPFitThreeGauss::main: Starting F_Image.ReadArray()" << endl;
    if (!F_Image.ReadArray())
    {
      cout << "MExtractMPFitThreeGauss::main: ERROR: F_Image.ReadArray() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    /// convert ADUs to photons
    F_Image.GetPixArray() /= D_Gain;

    /// Set DatabaseFileName_In
    cout << "MExtractMPFitThreeGauss::main: Starting F_Image.SetDatabaseFileName(" << CS_A1_DatabaseFileNames_In(i_file) << ")" << endl;
    if (!F_Image.SetDatabaseFileName(CS_A1_DatabaseFileNames_In(i_file)))
    {
      cout << "MExtractMPFitThreeGauss::main: ERROR: F_Image.SetDatabaseFileName(" << CS_A1_DatabaseFileNames_In(i_file) << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    /// Read DatabaseFileName_In
    cout << "MExtractMPFitThreeGauss::main: Starting F_Image.ReadDatabaseEntry()" << endl;
    if (!F_Image.ReadDatabaseEntry())
    {
      cout << "MExtractMPFitThreeGauss::main: ERROR: F_Image.ReadDatabaseEntry() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    /// Calculate Trace Functions
    cout << "MExtractMPFitThreeGauss::main: Starting F_Image.CalcTraceFunctions()" << endl;
    if (!F_Image.CalcTraceFunctions())
    {
      cout << "MExtractMPFitThreeGauss::main: ERROR: F_Image.CalcTraceFunctions() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    cout << "MExtractMPFitThreeGauss::main: P_CS_ErrIn = <" << *P_CS_ErrIn << ">" << endl;
    if (P_CS_ErrIn->GetLength() > 1){
      /// Set ErrFileName_In
      cout << "MExtractMPFitThreeGauss::main: Starting F_Image.SetErrFileName(" << CS_A1_ErrIn(i_file) << ")" << endl;
      if (!F_Image.SetErrFileName(CS_A1_ErrIn(i_file)))
      {
        cout << "MExtractMPFitThreeGauss::main: ERROR: F_Image.SetErrFileName(" << CS_A1_ErrIn(i_file) << ") returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }

      /// Read Error image
      cout << "MExtractMPFitThreeGauss::main: Starting F_Image.ReadErrArray()" << endl;
      if (!F_Image.ReadErrArray())
      {
        cout << "MExtractMPFitThreeGauss::main: ERROR: F_Image.ReadErrArray() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
    }

    /// Write aperture header information
    F_Image.WriteApHead(CString("aphead_")+CS_A1_FitsFileNames_In(i_file)+CString(".head"));

    /// Set CS_FitsFileName_In
    cout << "MExtractMPFitThreeGauss::main: Starting F_OutImage.SetFileName(" << CS_FitsFileName_In << ")" << endl;
    if (!F_OutImage.SetFileName(CS_A1_FitsFileNames_In(i_file)))
    {
      cout << "MExtractMPFitThreeGauss::main: ERROR: F_OutImage.SetFileName(" << CS_FitsFileName_In << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    ///Read FitsFile
    cout << "MExtractMPFitThreeGauss::main: Starting F_OutImage.ReadArray()" << endl;
    if (!F_OutImage.ReadArray())
    {
      cout << "MExtractMPFitThreeGauss::main: ERROR: F_OutImage.ReadArray() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    /// Set CS_FitsFileName_Out
    cout << "MExtractMPFitThreeGauss::main: Starting F_OutImage.SetFileName(" << CS_FitsFileName_Out << ")" << endl;
    if (!F_OutImage.SetFileName(CS_A1_FitsFileNames_Out(i_file)))
    {
      cout << "MExtractMPFitThreeGauss::main: ERROR: F_OutImage.SetFileName(" << CS_FitsFileName_Out << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    /// change size of F_OutImage to (NApertures x NRows)
    if (!F_OutImage.SetNCols(F_Image.GetNRows()))
    {
      cout << "MExtractMPFitThreeGauss::main: ERROR: F_OutImage.SetNCols(" << F_Image.GetNRows() << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    int I_NAps_Out = F_Image.Get_NApertures();
    if (B_Apertures_Set)
      I_NAps_Out = I_A1_Apertures.size();
    if (!F_OutImage.SetNRows(F_Image.Get_NApertures()))
    {
      cout << "MExtractMPFitThreeGauss::main: ERROR: F_OutImage.SetNRows(" << F_Image.Get_NApertures() << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    /// Write EcFromProfileOut 1D
    cout << "MExtractMPFitThreeGauss::main: Starting MPFitGaussExtract: F_Image.Get_NApertures() = " << F_Image.Get_NApertures() << endl;
    if (!F_Image.MPFitThreeGaussExtract(F_Image.GetPixArray(),
                                        D_A1_SDevLimits,
                                        D_A1_MeanLimits,
                                        B_WithBackground,
                                        CS_A1_Args,
                                        PP_Args)){
      cout << "MExtractMPFitThreeGauss::main: ERROR: F_Image.ExtractSimpleSum() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
    F_OutImage.GetPixArray() = F_Image.GetLastExtracted();//.transpose(secondDim, firstDim);
    if (!F_OutImage.WriteArray())
    {
      cout << "MExtractMPFitThreeGauss::main: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    /// Write aperture header information
    F_Image.WriteApHead(CString("aphead_")+CS_A1_FitsFileNames_In(i_file)+CString(".head"));

    /// Create ErrOutEc
    if (P_CS_ErrOut->GetLength() > 1){// || P_CS_ErrOutEc->GetLength() > 1){
      cout << "MExtractMPFitThreeGauss::main: Starting F_Image.ExtractErrors()" << endl;
      if (!F_Image.ExtractSimpleSum(F_Image.GetErrArray(),CS_A1_Args, PP_Args))
      {
        cout << "MExtractMPFitThreeGauss::main: ERROR: F_Image.ExtractErrors() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
    }

    cout << "MExtractMPFitThreeGauss::main: Starting F_OutImage.WriteArray()" << endl;
    if (!F_OutImage.WriteArray())
    {
      cout << "MExtractMPFitThreeGauss::main: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    /// Write ErrFromProfile 1D
    if (P_CS_ErrOut->GetLength() > 1)
    {
      if (!F_OutImage.SetFileName(CS_A1_ErrOutEc(i_file)))
      {
        cout << "MExtractMPFitThreeGauss::main: ERROR: F_OutImage.SetFileName() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
      cout << "MExtractMPFitThreeGauss::main: Starting to write ErrFromProfile" << endl;
      F_OutImage.GetPixArray() = F_Image.GetLastExtracted();//.transpose(secondDim, firstDim);
      if (!F_OutImage.WriteArray())
      {
        cout << "MExtractMPFitThreeGauss::main: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
    }
  }
  delete(P_CS);
  delete(P_CS_ErrIn);
  delete(P_CS_EcOut);
  delete(P_CS_ErrOut);
  return EXIT_SUCCESS;
}
