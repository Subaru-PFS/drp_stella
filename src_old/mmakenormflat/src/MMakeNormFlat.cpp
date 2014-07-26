/*
author: Andreas Ritter
created: 03/20/2007
last edited: 03/20/2007
compiler: g++ 4.0
basis machine: Ubuntu Linux 6.06
*/
#include "MMakeNormFlat.h"

int main(int argc, char *argv[])
    /// string input_fits_file_name: in, string input_database_file_name: in, string output_fits_file_name: out, string output_blaze_file_name: out, double Gain: in, double ReadOutNoise: in, int LambdaSP: in, double MinSNR: in[, int SwathWidth: in]
{
  cout << "MMakeNormFlat::main: argc = " << argc << endl;
  if (argc < 10)
  {
    cout << "MMakeNormFlat::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: makenormflat char[] FitsFileName_In, char[] DatabaseFileName_In, char[] NormalisedFlat_Out, char[] ProfileFits_Out, char[] BlazeFits_Out, double Gain, double ReadOutNoise, int SmoothSP, double MinSNR[, int SwathWidth][,SMOOTH_SF=double][,AREA=[int(xmin),int(xmax),int(ymin),int(ymax)]][,char[] MASK_OUT=Mask_Out][,char[] RECONSTRUCTED_IMAGE_OUT=ReconstructedImage_Out]" << endl;
    exit(EXIT_FAILURE);
  }
  Array<int, 1> I_A1_Area(4);
  Array<CString, 1> CS_A1_Args(3);
  CS_A1_Args = CString("\0");

  void **PP_Args;
  PP_Args = (void**)malloc(sizeof(void*) * 3);

  char *P_CharArr_In = (char*)argv[1];
  char *P_CharArr_DB = (char*)argv[2];
  char *P_CharArr_Out = (char*)argv[3];
  char *P_CharArr_ProfileOut = (char*)argv[4];
  char *P_CharArr_BlazeOut = (char*)argv[5];
  double D_Gain = (double)(atof((char*)argv[6]));
  cout << "MMakeNormFlat::main: D_Gain set to " << D_Gain << endl;
  double D_ReadOutNoise = (double)(atof((char*)argv[7]));
  cout << "MMakeNormFlat::main: D_ReadOutNoise set to " << D_ReadOutNoise << endl;
  int I_Lambda_SP_In = (int)(atoi((char*)argv[8]));
  cout << "MMakeNormFlat::main: I_Lambda_SP_In set to " << I_Lambda_SP_In << endl;
  double D_MinSNR = (double)(atof((char*)argv[9]));
  cout << "MMakeNormFlat::main: D_MinSNR set to " << D_MinSNR << endl;

  CString CS, CS_comp, CS_Mask_Out, CS_Rec_Out;
  CString* P_CS = new CString(" ");

  int I_SwathWidth = 0;
  double D_LambdaSF = 0.;
  /// read optional parameters
  for (int i = 10; i <= argc; i++){
    CS.Set((char*)argv[i]);
    cout << "MMakeNormFlat: Reading Parameter " << CS << endl;

    /// SWATH_WIDTH
    CS_comp.Set("SWATH_WIDTH");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MMakeNormFlat::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS);
        P_CS = CS.SubString(CS_comp.GetLength()+1);
        I_SwathWidth = (int)(atoi(P_CS->GetPChar()));
        cout << "MMakeNormFlat::main: I_SwathWidth set to " << I_SwathWidth << endl;
        CS_A1_Args(0) = CString("SWATH_WIDTH");
        PP_Args[0] = &I_SwathWidth;
      }
    }

    /// SMOOTH_SF
    CS_comp.Set("SMOOTH_SF");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MMakeNormFlat::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS);
        P_CS = CS.SubString(CS_comp.GetLength()+1);
        D_LambdaSF = (double)(atof(P_CS->GetPChar()));
        cout << "MMakeNormFlat::main: D_LambdaSF set to " << D_LambdaSF << endl;
        CS_A1_Args(2) = CString("LAMBDA_SF");
        PP_Args[2] = &D_LambdaSF;
      }
    }

    /// AREA
    CS_comp.Set("AREA");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MMakeNormFlat::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        CString cs_temp;
        cs_temp.Set(",");
        int i_pos_a = CS_comp.GetLength()+2;
        int i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        cout << "MMakeNormFlat: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
        delete(P_CS);
        P_CS = CS.SubString(i_pos_a,i_pos_b-1);
        cout << "MMakeNormFlat: P_CS set to " << *P_CS << endl;
        I_A1_Area(0) = (int)(atoi(P_CS->Get()));
        cout << "MMakeNormFlat: I_A1_Area(0) set to " << I_A1_Area(0) << endl;

        i_pos_a = i_pos_b+1;
        i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        cout << "MMakeNormFlat: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
        delete(P_CS);
        P_CS = CS.SubString(i_pos_a,i_pos_b-1);
        cout << "MMakeNormFlat: P_CS set to " << *P_CS << endl;
        I_A1_Area(1) = (int)(atoi(P_CS->Get()));
        cout << "MMakeNormFlat: I_A1_Area(1) set to " << I_A1_Area(1) << endl;

        i_pos_a = i_pos_b+1;
        i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        cout << "MMakeNormFlat: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
        delete(P_CS);
        P_CS = CS.SubString(i_pos_a,i_pos_b-1);
        cout << "MMakeNormFlat: P_CS set to " << *P_CS << endl;
        I_A1_Area(2) = (int)(atoi(P_CS->Get()));
        cout << "MMakeNormFlat: I_A1_Area(2) set to " << I_A1_Area(2) << endl;

        i_pos_a = i_pos_b+1;
        cs_temp.Set("]");
        i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        if (i_pos_b < 0){
          cs_temp.Set(")");
          i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        }
        cout << "MMakeNormFlat: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
        delete(P_CS);
        P_CS = CS.SubString(i_pos_a,i_pos_b-1);
        cout << "MMakeNormFlat: P_CS set to " << *P_CS << endl;
        I_A1_Area(3) = (int)(atoi(P_CS->Get()));
        cout << "MMakeNormFlat: I_A1_Area(3) set to " << I_A1_Area(3) << endl;

        CS_A1_Args(1) = CString("AREA");
        PP_Args[1] = &I_A1_Area;
        cout << "MMakeNormFlat::main: AREA set to " << I_A1_Area << endl;
      }
    }

    ///MASK_OUT
    CS_comp.Set("MASK_OUT");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MMakeNormFlat::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS);
        P_CS = CS.SubString(CS_comp.GetLength()+1);
        CS_Mask_Out.Set(P_CS->GetPChar());
        cout << "MMakeNormFlat::main: CS_Mask_Out set to " << CS_Mask_Out << endl;
      }
    }

    ///RECONSTRUCTED_IMAGE_OUT
    CS_comp.Set("RECONSTRUCTED_IMAGE_OUT");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MMakeNormFlat::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS);
        P_CS = CS.SubString(CS_comp.GetLength()+1);
        CS_Rec_Out.Set(P_CS->GetPChar());
        cout << "MMakeNormFlat::main: CS_Rec_Out set to " << CS_Rec_Out << endl;
      }
    }
  }

  CString CS_FitsFileName_In;
  CS_FitsFileName_In.Set(P_CharArr_In);
  CString CS_FitsFileName_Out;
  CS_FitsFileName_Out.Set(P_CharArr_Out);
  CString CS_DatabaseFileName_In;
  CS_DatabaseFileName_In.Set(P_CharArr_DB);
  CString CS_FitsFileName_Profile_Out;
  CS_FitsFileName_Profile_Out.Set(P_CharArr_ProfileOut);
  CString CS_FitsFileName_Blaze_Out;
  CS_FitsFileName_Blaze_Out.Set(P_CharArr_BlazeOut);

  CFits F_Image;
  if (!F_Image.SetFileName(CS_FitsFileName_In))
  {
    cout << "MMakeNormFlat::main: ERROR: F_Image.SetFileName(" << CS_FitsFileName_In.Get() << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Set ReadOutNoise
  if (!F_Image.Set_ReadOutNoise( D_ReadOutNoise ))
  {
    cout << "MMakeNormFlat::main: ERROR: F_Image.Set_ReadOutNoise(" << D_ReadOutNoise << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Set Gain
  if (!F_Image.Set_Gain( D_Gain ))
  {
    cout << "MMakeNormFlat::main: ERROR: F_Image.Set_Gain(" << D_Gain << ") returned FALSE" << endl;
    exit(EXIT_FAILURE);
  }

  /// Set OverSample
  if (!F_Image.Set_OverSample( 10 ))
  {
    cout << "MMakeNormFlat::main: ERROR: F_Image.Set_OverSample() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Read FitsFile
  if (!F_Image.ReadArray())
  {
    cout << "MMakeNormFlat::main: ERROR: F_Image.ReadArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Set DatabaseFileName_In
  if (!F_Image.SetDatabaseFileName(CS_DatabaseFileName_In))
  {
    cout << "MMakeNormFlat::main: ERROR: F_Image.SetDatabaseFileName(" << CS_DatabaseFileName_In << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  cout << "MMakeNormFlat::main: Reading DatabaseEntry" << endl;
  /// Read DatabaseFileName_In
  if (!F_Image.ReadDatabaseEntry())
  {
    cout << "MMakeNormFlat::main: ERROR: F_Image.ReadDatabaseEntry() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  cout << "MMakeNormFlat::main: Calculating Trace Functions" << endl;
  /// Calculate Trace Functions
  if (!F_Image.CalcTraceFunctions())
  {
    cout << "MMakeNormFlat::main: ERROR: F_Image.CalcTraceFunctions() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Calculate NormFlat Image
  cout << "MMakeNormFlat::main: Starting MkNormFlatProf" << endl;
  if (!F_Image.MkNormFlatProf(I_Lambda_SP_In, CS_A1_Args, PP_Args))
  {
    cout << "MMakeNormFlat::main: ERROR: F_Image.MkNormFlat() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Write profile image
  if (!F_Image.SetFileName(CS_FitsFileName_Profile_Out))
  {
    cout << "MMakeNormFlat::main: ERROR: F_Image.SetFileName(" << CS_FitsFileName_Out << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  Array<double, 2> D_A2_InputImage(F_Image.GetNRows(), F_Image.GetNCols());
  D_A2_InputImage = F_Image.GetPixArray();
  F_Image.GetPixArray() = F_Image.GetProfArray();
  if (!F_Image.WriteArray())
  {
    cout << "MMakeNormFlat::main: ERROR: F_Image.WriteArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  F_Image.GetPixArray() = D_A2_InputImage;

  /// Set CS_FitsFileName_Out
  if (!F_Image.SetFileName(CS_FitsFileName_Out))
  {
    cout << "MMakeNormFlat::main: ERROR: F_Image.SetFileName(" << CS_FitsFileName_Out << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Set pixels with SNR < D_MinSNR to 1.
  Array<int, 2> I_A2_ToLow(F_Image.GetNRows(), F_Image.GetNCols());
  Array<int, 2> I_A2_ProfZero(F_Image.GetNRows(), F_Image.GetNCols());
  I_A2_ToLow = where((D_A2_InputImage / D_Gain) < (D_MinSNR * D_MinSNR), 1, 0);
  I_A2_ProfZero = where(fabs(F_Image.GetProfArray()) < 0.000000001, 1, 0);
//  for (int irow = 0; irow < F_Image.GetNRows(); irow++)
//  {
//    for (int icol = 0; icol < F_Image.GetNCols(); icol++)
//    {
//      if (I_A2_ProfZero(irow, icol) == 1)
//        (F_Image.GetProfArray())(irow, icol) = 1000.;
//      if (I_A2_ToLow(irow, icol) == 1)
//        (F_Image.GetProfArray())(irow, icol) = 1.;
//    }
//  }

  /// Divide combined Flat by Reconstructed Image
  Array<double, 2> D_A2_RecImage(F_Image.GetNRows(), F_Image.GetNCols());
  D_A2_RecImage = F_Image.GetRecArray();
  D_A2_InputImage = D_A2_InputImage / D_A2_RecImage;
  //F_Image.GetPixArray() = F_Image.GetPixArray() / F_Image.GetProfArray();

  /// Set pixels with SNR < D_MinSNR to 1.
  for (int irow = 0; irow < F_Image.GetNRows(); irow++)
  {
    for (int icol = 0; icol < F_Image.GetNCols(); icol++)
    {
      if ((I_A2_ProfZero(irow, icol) == 1) || (I_A2_ToLow(irow, icol) == 1) || (D_A2_RecImage(irow, icol) < 0.001))
        D_A2_InputImage(irow, icol) = 1.;
    }
  }
  F_Image.GetPixArray() = D_A2_InputImage;

  /// Write NormFlat Image
  if (!F_Image.WriteArray())
  {
    cout << "MMakeNormFlat::main: ERROR: F_Image.WriteArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  CFits F_Blaze_Out;
  /// Set CS_FitsFileName_Out
  cout << "MMakeNormFlat::main: Starting F_OutImage.SetFileName(" << CS_FitsFileName_Blaze_Out << ")" << endl;
  if (!F_Blaze_Out.SetFileName(CS_FitsFileName_Blaze_Out))
  {
    cout << "MMakeNormFlat::main: ERROR: F_Blaze_Out.SetFileName(" << CS_FitsFileName_Blaze_Out << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  if (!F_Blaze_Out.SetNCols(F_Image.GetNRows()))
  {
    cout << "MMakeNormFlat::main: ERROR: F_Blaze_Out.SetNCols(" << F_Image.GetNRows() << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  if (!F_Blaze_Out.SetNRows(F_Image.Get_NApertures()))
  {
    cout << "MMakeNormFlat::main: ERROR: F_Blaze_Out.SetNRows(" << F_Image.Get_NApertures() << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  F_Blaze_Out.GetPixArray() = F_Image.GetSpec();

  /// Write Blaze Functions
  cout << "MMakeNormFlat::main: Starting F_Blaze_Out.WriteArray()" << endl;
  if (!F_Blaze_Out.WriteArray())
  {
    cout << "MMakeNormFlat::main: ERROR: F_Blaze_Out.WriteArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Write Mask
  if (CS_Mask_Out.GetLength()>1){
    F_Image.SetFileName(CS_Mask_Out);
    F_Image.GetPixArray() = 1.*F_Image.GetMaskArray();
    F_Image.WriteArray();
  }

  /// Write Reconstructed Image
  if (CS_Rec_Out.GetLength()>1){
    F_Image.SetFileName(CS_Rec_Out);
    F_Image.GetPixArray() = F_Image.GetRecArray();
    F_Image.WriteArray();
  }

  return EXIT_SUCCESS;
}
