/*
author: Andreas Ritter
created: 20/06/2012
last edited: 20/06/2012
compiler: g++ 4.4
basis machine: Ubuntu Linux 10.04
*/

///TODO: parameter I_MininumApertureLength

#include "MFindAndTraceAps.h"

using namespace std;

int main(int argc, char *argv[])
{
  cout << "MFindAndTraceAps::main: argc = " << argc << endl;
  if (argc < 19)
  {
    cout << "MFindAndTraceAps::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: findandtraceaps <char[] FitsFileName_In> <int CentralSize_X_In> <int CentralSize_Y_In> <double SaturationLevel_In> <double SignalThreshold_In> <double ApertureFWHM_In> <int NTermsGaussFit_In> <int PolyFitOrder_In> <int NLost_In> <double XLow_In> <double XMin_In> <double XHigh_In> <double XMax_In> <int MinLength_In> <int MaxLength_In> <int ExtendAps_Up> <int ApertureLength_Down> <char[] DataBaseFileName_Out> [FitsFileName_RefScatter_In=<char[] ReferenceScatteredLightImage_In>] [FitsFileName_MinusScatter_Out=<char[] FitsFileName_MinusScatter_Out>] [FitsFileName_CentersMarked_Out=<char[] FitsFileName_CentersMarked_Out>]" << endl;
    cout << "Parameter 1: char[] FitsFileName_In," << endl;
    cout << "Parameter 2: int CentralSize_X_In to fit reference scattered light (Set to any number if scattered-light subtraction is not desired)," << endl;
    cout << "Parameter 3: int CentralSize_Y_In to fit reference scattered light (Set to any number if scattered-light subtraction is not desired)," << endl;
    cout << "Parameter 4: double SaturationLevel_In," << endl;
    cout << "Parameter 5: double SignalThreshold_In - pixel values below this value will be set to zero," << endl;
    cout << "Parameter 6: double ApertureFWHM_In - FWHM of the assumed Gaussian spatial profile," << endl;
    cout << "Parameter 7: int NTermsGaussFit_In (3-6) - 3: fit mean, sigma, center; 4: 3+constant background; 5: 4 plus background slope," << endl;
    cout << "Parameter 8: int PolyFitOrder_In - Polynomial fitting order for the trace functions," << endl;
    cout << "Parameter 9: int NLost_In - number of consecutive times the profile is lost before breaking up the trace," << endl;
    cout << "Parameter 10: double XLow_In - IRAF database parameter - object width left of center," << endl;
    cout << "Parameter 11: double XMin_In - IRAF database parameter - background width left of center," << endl;
    cout << "Parameter 12: double XHigh_In - IRAF database parameter - object width right of center," << endl;
    cout << "Parameter 13: double XMax_In - IRAF database parameter - background width right of center," << endl;
    cout << "Parameter 14: int MinLength_In - minimum length of a trace to count as aperture," << endl;
    cout << "Parameter 15: int MaxLength_In - maximum length of a trace before breaking up the trace," << endl;
    cout << "Parameter 16: int ExtendAps_Up - number of pixels to extend a trace upwards," << endl;
    cout << "Parameter 17: int ApertureLength_Down - aperture length downwards after applying ExtendAps_Up," << endl;
    cout << "Parameter 18: char[] DataBaseFileName_Out - name of database file (database/ap...)" << endl;
    cout << "Parameter 19: [FitsFileName_RefScatter_In=(char[] ReferenceScatteredLightImage_In)]" << endl;
    cout << "Parameter 20: [FitsFileName_MinusScatter_Out=(char[] FitsFileName_MinusScatter_Out)]" << endl;
    cout << "Parameter 21: [FitsFileName_Out=(char[] FitsFileName_Out) - Quality control file with all found apertures set to zero]" << endl;
    cout << "Parameter 22: [FitsFileName_CentersMarked_Out=(char[] FitsFileName_CentersMarked_Out) - Quality control file with all traces set to zero]" << endl;
    exit(EXIT_FAILURE);
  }

  CString CS(" ");
  CString CS_comp(" ");
  CString *P_CS = new CString(" ");
  CString CS_FitsFileName_MinusScatter_Out(" ");
  CString CS_FitsFileName_Out(" ");
  CString CS_FitsFileName_CentersMarked_Out(" ");

  char *P_CharArr_In = (char*)argv[1];
//  char *P_CharArr_ScatterIn = (char*)argv[2];
  int I_CentralSize_X_In = (int)(atoi((char*)argv[2]));
  int I_CentralSize_Y_In = (int)(atoi((char*)argv[3]));
  double D_SaturationLevel_In = (double)(atof((char*)argv[4]));
  double D_SignalThreshold_In = (double)(atof((char*)argv[5]));
  double D_ApertureFWHM_In = (double)(atof((char*)argv[6]));
  int I_NTermsGaussFit_In = (int)(atoi((char*)argv[7]));
  int I_PolyFitOrder_In = (int)(atoi((char*)argv[8]));
  int I_NLost_In = (int)(atoi((char*)argv[9]));
  double D_XLow_In = (double)(atof((char*)argv[10]));
  double D_XMin_In = (double)(atof((char*)argv[11]));
  double D_XHigh_In = (double)(atof((char*)argv[12]));
  double D_XMax_In = (double)(atof((char*)argv[13]));
  int I_MinLength_In = (int)(atoi((char*)argv[14]));
  int I_MaxLength_In = (int)(atoi((char*)argv[15]));
  int I_ExtendAps_Up = (int)(atoi((char*)argv[16]));
  int I_ApertureLength_Down = (int)(atoi((char*)argv[17]));
  char *P_CharArr_DBOut = (char*)argv[18];
//  char *P_CharArr_Out = (char*)argv[16];

  bool B_Subtract_RefScatter = false;
  CString CS_RefScatter_In(" ");
  /// read optional parameters
  for (int i = 19; i <= argc; i++){
    CS.Set((char*)argv[i]);
    cout << "MFindAndTraceAps: Reading Parameter " << CS << endl;

    CS_comp.Set("FitsFileName_RefScatter_In");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MFindAndTraceAps::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS);
        P_CS = CS.SubString(CS_comp.GetLength()+1);
        CS_RefScatter_In = P_CS->Get();
        cout << "MFindAndTraceAps::main: CS_FitsFileName_RefScatter_In set to " << CS_RefScatter_In << endl;
	B_Subtract_RefScatter = true;
      }
    }

    CS_comp.Set("FitsFileName_MinusScatter_Out");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MFindAndTraceAps::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS);
        P_CS = CS.SubString(CS_comp.GetLength()+1);
        CS_FitsFileName_MinusScatter_Out = P_CS->Get();
        cout << "MFindAndTraceAps::main: CS_FitsFileName_MinusScatter_Out set to " << CS_FitsFileName_MinusScatter_Out << endl;
      }
    }

    CS_comp.Set("FitsFileName_Out");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MFindAndTraceAps::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS);
        P_CS = CS.SubString(CS_comp.GetLength()+1);
        CS_FitsFileName_Out = P_CS->Get();
        cout << "MFindAndTraceAps::main: CS_FitsFileName_Out set to " << CS_FitsFileName_Out << endl;
      }
    }

    CS_comp.Set("FitsFileName_CentersMarked_Out");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MFindAndTraceAps::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS);
        P_CS = CS.SubString(CS_comp.GetLength()+1);
        CS_FitsFileName_CentersMarked_Out = P_CS->Get();
        cout << "MFindAndTraceAps::main: CS_FitsFileName_CentersMarked_Out set to " << CS_FitsFileName_CentersMarked_Out << endl;
      }
    }
  }

//  return false;

  time_t seconds;
  CString CS_FitsFileName_In;
  CS_FitsFileName_In.Set(P_CharArr_In);
//  CString CS_FitsFileName_Out;
//  CS_FitsFileName_Out.Set(P_CharArr_Out);
  CString CS_DatabaseFileName_Out;
  CS_DatabaseFileName_Out.Set(P_CharArr_DBOut);

  cout << "MFindAndTraceAps::main: CS_DatabaseFileName_Out = " << CS_DatabaseFileName_Out << endl;
  int IPos = CS_DatabaseFileName_Out.LastStrPos(CString("/"));
  cout << "MFindAndTraceAps::main: CS_DatabaseFileName_Out.LastStrPos(CString(/)) = " << IPos << endl;
  CString *P_CS_DB_Path = CS_DatabaseFileName_Out.SubString(0, CS_DatabaseFileName_Out.LastStrPos(CString("/"))-1);
  cout << "MFindAndTraceAps::main: *P_CS_DB_Path = " << *P_CS_DB_Path << endl;
  if (!CS_DatabaseFileName_Out.MkDir(*P_CS_DB_Path)){
    cout << "MFindAndTraceAps::main: ERROR: MkDir(" << *P_CS_DB_Path << ") returned FALSE" << endl;
    exit(EXIT_FAILURE);
  }
  delete(P_CS_DB_Path);

  CFits F_Image;
  cout << "MFindAndTraceAps::main: Starting F_Image.SetFileName(" << CS_FitsFileName_In.Get() << ")" << endl;
  if (!F_Image.SetFileName(CS_FitsFileName_In))
  {
    cout << "MFindAndTraceAps::main: ERROR: F_Image.SetFileName(" << CS_FitsFileName_In.Get() << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Read FitsFile
  cout << "MFindAndTraceAps::main: Starting F_Image.ReadArray()" << endl;
  if (!F_Image.ReadArray())
  {
    cout << "MFindAndTraceAps::main: ERROR: F_Image.ReadArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  if (B_Subtract_RefScatter){
    /// Scale Reference Scattered-Light image to fit background in the central region
    seconds = time(NULL);
    cout << "MFindAndTraceAps::main: Starting to fit scattered light: time = " << seconds << endl;

    CFits *P_CF_Scatter = new CFits();
    P_CF_Scatter->SetFileName(CS_RefScatter_In);
    P_CF_Scatter->ReadArray();

    Array<double, 2> D_A2_Flat = F_Image.GetPixArray();

    int I_X_Start = (D_A2_Flat.rows() / 2) - (I_CentralSize_X_In / 2);
    int I_X_End = I_X_Start + I_CentralSize_X_In - 1;

    int I_Y_Start = (D_A2_Flat.cols() / 2) - (I_CentralSize_Y_In / 2);
    int I_Y_End = I_Y_Start + I_CentralSize_Y_In - 1;

    Array<double, 2> D_A2_FlatCrop(I_CentralSize_X_In, I_CentralSize_Y_In);
    D_A2_FlatCrop = D_A2_Flat(Range(I_X_Start, I_X_End), Range(I_Y_Start, I_Y_End));

    Array<double, 2> D_A2_Scatter = P_CF_Scatter->GetPixArray();
    Array<double, 2> D_A2_ScatterCrop(I_CentralSize_X_In, I_CentralSize_Y_In);
    D_A2_ScatterCrop = D_A2_Scatter(Range(I_X_Start, I_X_End), Range(I_Y_Start, I_Y_End));

    double D_Fac;
    if (!P_CF_Scatter->ScaleImageToFitBackground(D_A2_FlatCrop, D_A2_ScatterCrop, D_Fac))
      return false;
    cout << "MFindAndTraceAps: D_Fac = " << D_Fac << endl;

    cout << "MFindAndTraceAps: D_A2_Scatter.rows() = " << D_A2_Scatter.rows() << endl;
    cout << "MFindAndTraceAps: D_A2_Scatter.cols() = " << D_A2_Scatter.cols() << endl;

    Array<double, 2> D_A2_Temp(D_A2_Scatter.rows(), D_A2_Scatter.cols());
    D_A2_Temp = D_A2_Scatter * D_Fac;
//  D_A2_Temp.resizeAndPreserve(D_A2_Flat.rows(), D_A2_Flat.cols());
    F_Image -= D_A2_Temp;

    /// Write ImOut
    cout << "MFindAndTraceAps: CS_FitsFileName_MinusScatter_Out = <" << CS_FitsFileName_MinusScatter_Out << ">" << endl;
    if (CS_FitsFileName_MinusScatter_Out.GetLength() > 1){
      if (!F_Image.SetFileName(CS_FitsFileName_MinusScatter_Out))
      {
        cout << "MFindAndTraceAps::main: ERROR: F_Image.SetFileName() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
      if (!F_Image.WriteArray())
      {
        cout << "MFindAndTraceAps::main: ERROR: F_Image.WriteArray() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
    }
    cout << "MFindAndTraceAps: deleting P_CF_Scatter" << endl;
    delete(P_CF_Scatter);
  }/// end if (B_Subtract_RefScatter)

  /// Find and trace apertures
  F_Image.Set_SaturationLevel(D_SaturationLevel_In);
  F_Image.Set_SignalThreshold(D_SignalThreshold_In);
  F_Image.Set_ApertureFWHM(D_ApertureFWHM_In);
  cout << "MFindAndTraceAps: Starting FindAndTraceApertures" << endl;
  if (!F_Image.FindAndTraceApertures(I_NTermsGaussFit_In,
                                     I_PolyFitOrder_In,
				     I_MinLength_In,
				     I_MaxLength_In,
                                     I_NLost_In)){
    cout << "MFindAndTraceAps: ERROR: F_Image.FindAndTraceApertures(...) returned FALSE = Exiting program" << endl;
    exit(EXIT_FAILURE);
  }
//  F_Image.SetDatabaseFileName(CS_DatabaseFileName_Out);
  Array<double, 1> D_A1_XLow(F_Image.Get_NApertures());
  D_A1_XLow = D_XLow_In;
  F_Image.Set_XLow(D_A1_XLow);
  D_A1_XLow = D_XMin_In;
  F_Image.Set_XMin(D_A1_XLow);
  D_A1_XLow = D_XHigh_In;
  F_Image.Set_XHigh(D_A1_XLow);
  D_A1_XLow = D_XMax_In;
  F_Image.Set_XMax(D_A1_XLow);






  /// Extend Aperture lengths to specified values
  Array<double, 1> *P_D_A1_YCenter = F_Image.Get_YCenter();
  Array<double, 1> D_A1_YLow(F_Image.Get_NApertures());
  Array<double, 1> *P_D_A1_YHigh = F_Image.Get_YHigh();
  Array<double, 1> D_A1_YCenter_New(F_Image.Get_NApertures());
  D_A1_YCenter_New = (*P_D_A1_YCenter);
  Array<double, 1> D_A1_YLow_New(F_Image.Get_NApertures());
  Array<double, 1> D_A1_YHigh_New(F_Image.Get_NApertures());
  for (int i_ap=0; i_ap < F_Image.Get_NApertures(); i_ap++){
    (*P_D_A1_YHigh)(i_ap) = (*P_D_A1_YHigh)(i_ap) + I_ExtendAps_Up;
    if ((*P_D_A1_YCenter)(i_ap) + (*P_D_A1_YHigh)(i_ap) >= F_Image.GetNRows())
      (*P_D_A1_YHigh)(i_ap) = F_Image.GetNRows() - (*P_D_A1_YCenter)(i_ap) - 1;
  }
  F_Image.Set_YHigh(*P_D_A1_YHigh);
  for (int i_ap=0; i_ap < F_Image.Get_NApertures(); i_ap++){
    D_A1_YLow(i_ap) = (*P_D_A1_YHigh)(i_ap) - I_ApertureLength_Down;
    if ((*P_D_A1_YCenter)(i_ap) + D_A1_YLow(i_ap) < 0)
      D_A1_YLow(i_ap) = 0. - (*P_D_A1_YCenter)(i_ap);
  }
  cout << "MFindAndTraceAps::main: P_D_A1_YHigh = " << *P_D_A1_YHigh << endl;
  cout << "MFindAndTraceAps::main: D_A1_YLow = " << D_A1_YLow << endl;
//  exit(EXIT_FAILURE);
  F_Image.Set_YLow(D_A1_YLow);
  D_A1_YHigh_New = (*P_D_A1_YHigh);
  D_A1_YLow_New = D_A1_YLow;
  /// Set DatabaseFileName_Out
  cout << "MFindAndTraceAps::main: Starting F_Image.SetDatabaseFileName(" << CS_DatabaseFileName_Out << ")" << endl;
  if (!F_Image.SetDatabaseFileName(CS_DatabaseFileName_Out))
  {
    cout << "MFindAndTraceAps::main: ERROR: F_Image.SetDatabaseFileName(" << CS_DatabaseFileName_Out << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Write DatabaseFileName_Out
  cout << "MFindAndTraceAps::main: Starting F_Image.WriteDatabaseEntry()" << endl;
  if (!F_Image.WriteDatabaseEntry())
  {
    cout << "MFindAndTraceAps::main: ERROR: F_Image.WriteDatabaseEntry() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

/*  /// CrossCorrelate spectra
  F_Image.ReadArray();

  /// Find an aperture close to the CCD center
  double D_XCenter = F_Image.GetNCols() / 2.;
  double D_YCenter = F_Image.GetNRows() / 2.;
  Array<double, 1> D_A1_Reference(2);
  D_A1_Reference(0) = D_XCenter;
  D_A1_Reference(1) = D_YCenter;
  Array<double, 1> D_A1_NearestNeighbour_Out(2);
  int I_NearestNeighbourPos = 0;
  Array<double, 2> D_A2_ApertureCenters(F_Image.Get_NApertures(), 2);
  cout << "MFindAndTraceAps::main: Starting F_Image.CalcTraceFunctions()" << endl;
  if (!F_Image.CalcTraceFunctions()){
    cout << "MFindAndTraceAps::main: ERROR: F_Image.CalcTraceFunctions() returned FALSE" << endl;
    exit(EXIT_FAILURE);
  }
  if (!F_Image.GetApertureCenters(D_A2_ApertureCenters)){
    cout << "MFindAndTraceAps::main: ERROR: GetApertureCenters(D_A2_ApertureCenters) returned FALSE" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MFindAndTraceAps::main: D_A2_ApertureCenters = " << D_A2_ApertureCenters << endl;
  CFits F_Image_Aps;
  F_Image_Aps.SetFileName(CS_FitsFileName_In);
  F_Image_Aps.ReadArray();

  /// Set DatabaseFileName_Out
  cout << "MFindAndTraceAps::main: Starting F_Image.SetDatabaseFileName(" << CS_DatabaseFileName_Out << ")" << endl;
  if (!F_Image_Aps.SetDatabaseFileName(CS_DatabaseFileName_Out))
  {
    cout << "MFindAndTraceAps::main: ERROR: F_Image_Aps.SetDatabaseFileName(" << CS_DatabaseFileName_Out << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Write DatabaseFileName_Out
  cout << "MFindAndTraceAps::main: Starting F_Image.WriteDatabaseEntry()" << endl;
  if (!F_Image_Aps.ReadDatabaseEntry())
  {
    cout << "MFindAndTraceAps::main: ERROR: F_Image_Aps.ReadDatabaseEntry() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  if (!F_Image_Aps.CalcTraceFunctions()){
    cout << "MFindAndTraceAps::main: ERROR: F_Image_Aps.CalcTraceFunctions() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  Array<double, 2> D_A2_Static(2,2);
  Array<double, 2> D_A2_Moving(2,2);
  int I_Row, I_XMin, I_XMax_Temp, I_XMax, I_XMin_Temp, I_YMin, I_YMax, I_XMin_Static, I_XMin_Moving, I_XMax_Static, I_XMax_Moving, I_YMin_Static, I_YMin_Moving, I_YMax_Static, I_YMax_Moving;
  int I_DX = 2;
  int I_DY = 10;
  Array<int, 1> *P_I_A1_ApertureNumbers = F_Image.IndGenArr(F_Image.Get_NApertures());
  if (!F_Image.FindNearestNeighbour(D_A1_Reference,
                                    D_A2_ApertureCenters,
                                    D_A1_NearestNeighbour_Out,
                                    I_NearestNeighbourPos)){
    cout << "MFindAndTraceAps::main: ERROR: FindNearestNeighbour(D_A1_Reference=" << D_A1_Reference << ",...) returned FALSE" << endl;
    exit(EXIT_FAILURE);
  }
//  int I_CenterAperture = I_NearestNeighbourPos;
  cout << "MFindAndTraceAps::main: I_CenterAperture = " << I_NearestNeighbourPos << endl;
  D_A1_Reference = D_A2_ApertureCenters(I_NearestNeighbourPos, Range::all());

  Array<double, 2> *P_D_A2_XCenters = F_Image.Get_XCenters();

  for (int i_ap=0; i_ap<F_Image.Get_NApertures()-1; i_ap++){
    // create D_A2_Static for CrossCorrelate2D
    I_Row = (*P_D_A1_YCenter)(I_NearestNeighbourPos) + D_A1_YLow(I_NearestNeighbourPos);
    cout << "MFindAndTraceAps::main: I_Row(low) = " << I_Row << endl;
    I_XMin = int((*P_D_A2_XCenters)((*(P_I_A1_ApertureNumbers))(I_NearestNeighbourPos), I_Row) + D_XMin_In);
    I_XMax_Temp = int(double(I_XMin) - D_XMin_In + D_XMax_In);
    cout << "MFindAndTraceAps::main: I_XMin = " << I_XMin << ", I_XMax_Temp = " << I_XMax_Temp << endl;
    I_Row = (*P_D_A1_YCenter)(I_NearestNeighbourPos) + (*P_D_A1_YHigh)(I_NearestNeighbourPos);
    cout << "MFindAndTraceAps::main: I_Row(high) = " << I_Row << endl;
    I_XMax = int((*P_D_A2_XCenters)((*(P_I_A1_ApertureNumbers))(I_NearestNeighbourPos), I_Row) + D_XMax_In);
    I_XMin_Temp = int(double(I_XMax) + D_XMin_In - D_XMax_In);
    cout << "MFindAndTraceAps::main: I_XMax = " << I_XMax << ", I_XMin_Temp = " << I_XMin_Temp << endl;
    if (I_XMax < I_XMin){
      swap(I_XMax, I_XMin);
      swap(I_XMax_Temp, I_XMin_Temp);
    }
    if (I_XMin < 0)
      I_XMin = 0;
    if (I_XMin_Temp < 0)
      I_XMin_Temp = 0;
    if (I_XMax >= F_Image.GetNCols())
      I_XMax = F_Image.GetNCols()-1;
    if (I_XMax_Temp >= F_Image.GetNCols())
      I_XMax_Temp = F_Image.GetNCols()-1;
    if (I_XMin > I_XMin_Temp)
      I_XMin = I_XMin_Temp;
    if (I_XMax < I_XMax_Temp)
      I_XMax = I_XMax_Temp;
    cout << "MFindAndTraceAps::main: I_XMin = " << I_XMin << ", I_XMax = " << I_XMax << endl;
    I_YMin = (*P_D_A1_YCenter)(I_NearestNeighbourPos) + D_A1_YLow(I_NearestNeighbourPos);
    I_YMax = (*P_D_A1_YCenter)(I_NearestNeighbourPos) + (*P_D_A1_YHigh)(I_NearestNeighbourPos);
    if (I_YMin < 0)
      I_YMin = 0;
    if (I_YMax >= F_Image.GetNRows())
      I_YMax = F_Image.GetNRows()-1;
    cout << "MFindAndTraceAps::main: I_YMin = " << I_YMin << ", I_YMax = " << I_YMax << endl;
    I_XMin_Static = I_XMin;
    I_XMax_Static = I_XMax;
    I_YMin_Static = I_YMin;
    I_YMax_Static = I_YMax;
    D_A2_Static.resize(I_YMax - I_YMin + 1, I_XMax - I_XMin + 1);
    D_A2_Static = F_Image.GetPixArray()(Range(I_YMin, I_YMax), Range(I_XMin, I_XMax));
    cout << "MFindAndTraceAps::main: D_A2_Static = " << D_A2_Static << endl;

    if (!F_Image.Remove_Ith_ElementFromArray((*P_I_A1_ApertureNumbers), I_NearestNeighbourPos)){
      cout << "MFindAndTraceAps::main: ERROR: F_Image.Remove_Ith_ElementFromArray((*P_I_A1_ApertureNumbers)=" << (*(P_I_A1_ApertureNumbers)) << ", I_NearestNeighbourPos=" << I_NearestNeighbourPos << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    if (!F_Image.Remove_Ith_ElementFromArray(*P_D_A1_YCenter, I_NearestNeighbourPos)){
      cout << "MFindAndTraceAps::main: ERROR: F_Image.Remove_Ith_ElementFromArray(D_A1_YCenter=" << *P_D_A1_YCenter << ", I_NearestNeighbourPos=" << I_NearestNeighbourPos << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    if (!F_Image.Remove_Ith_ElementFromArray(D_A1_YLow, I_NearestNeighbourPos)){
      cout << "MFindAndTraceAps::main: ERROR: F_Image.Remove_Ith_ElementFromArray(D_A1_YLow=" << D_A1_YLow << ", I_NearestNeighbourPos=" << I_NearestNeighbourPos << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    if (!F_Image.Remove_Ith_ElementFromArray(*P_D_A1_YHigh, I_NearestNeighbourPos)){
      cout << "MFindAndTraceAps::main: ERROR: F_Image.Remove_Ith_ElementFromArray(D_A1_YHigh=" << *P_D_A1_YHigh << ", I_NearestNeighbourPos=" << I_NearestNeighbourPos << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    if (!F_Image.Remove_Ith_RowFromArray(D_A2_ApertureCenters, I_NearestNeighbourPos)){
      cout << "MFindAndTraceAps::main: ERROR: F_Image.Remove_Ith_ElementFromArray(D_A2_ApertureCenters=" << D_A2_ApertureCenters << ", I_NearestNeighbourPos=" << I_NearestNeighbourPos << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }

    /// Find Nearest Neighbour
    if (!F_Image.FindNearestNeighbour(D_A1_Reference,
                                      D_A2_ApertureCenters,
                                      D_A1_NearestNeighbour_Out,
                                      I_NearestNeighbourPos)){
      cout << "MFindAndTraceAps::main: ERROR: FindNearestNeighbour(D_A2_ApertureCenters) returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    cout << "MFindAndTraceAps::main: I_NearestNeighbourPos = " << I_NearestNeighbourPos << endl;
    cout << "MFindAndTraceAps::main: D_A1_NearestNeighbour_Out = " << D_A1_NearestNeighbour_Out << endl;

    // create D_A2_Moving for CrossCorrelate2D
    I_Row = (*P_D_A1_YCenter)(I_NearestNeighbourPos) + D_A1_YLow(I_NearestNeighbourPos);
    cout << "MFindAndTraceAps::main: I_Row(low) = " << I_Row << endl;
    I_XMin = int((*P_D_A2_XCenters)((*(P_I_A1_ApertureNumbers))(I_NearestNeighbourPos), I_Row) + D_XMin_In);
    I_XMax_Temp = int(double(I_XMin) - D_XMin_In + D_XMax_In);
    cout << "MFindAndTraceAps::main: I_XMin = " << I_XMin << ", I_XMax_Temp = " << I_XMax_Temp << endl;
    I_Row = (*P_D_A1_YCenter)(I_NearestNeighbourPos) + (*P_D_A1_YHigh)(I_NearestNeighbourPos);
    cout << "MFindAndTraceAps::main: I_Row(high) = " << I_Row << endl;
    I_XMax = int((*P_D_A2_XCenters)((*(P_I_A1_ApertureNumbers))(I_NearestNeighbourPos), I_Row) + D_XMax_In);
    I_XMin_Temp = int(double(I_XMax) + D_XMin_In - D_XMax_In);
    cout << "MFindAndTraceAps::main: I_XMax = " << I_XMax << ", I_XMin_Temp = " << I_XMin_Temp << endl;
    if (I_XMax < I_XMin){
      swap(I_XMax, I_XMin);
      swap(I_XMax_Temp, I_XMin_Temp);
    }
    if (I_XMin < 0)
      I_XMin = 0;
    if (I_XMin_Temp < 0)
      I_XMin_Temp = 0;
    if (I_XMax >= F_Image.GetNCols())
      I_XMax = F_Image.GetNCols()-1;
    if (I_XMax_Temp >= F_Image.GetNCols())
      I_XMax_Temp = F_Image.GetNCols()-1;
    if (I_XMin > I_XMin_Temp)
      I_XMin = I_XMin_Temp;
    if (I_XMax < I_XMax_Temp)
      I_XMax = I_XMax_Temp;
    I_XMin = I_XMin - I_DX;
    if (I_XMin < 0)
      I_XMin = 0;
    I_XMax = I_XMax + I_DX;
    if (I_XMax >= F_Image.GetNCols())
      I_XMax = F_Image.GetNCols()-1;
    cout << "MFindAndTraceAps::main: I_XMin = " << I_XMin << ", I_XMax = " << I_XMax << endl;
    I_YMin = int((*P_D_A1_YCenter)(I_NearestNeighbourPos) + D_A1_YLow(I_NearestNeighbourPos)) - I_DY;
    I_YMax = int((*P_D_A1_YCenter)(I_NearestNeighbourPos) + (*P_D_A1_YHigh)(I_NearestNeighbourPos)) + I_DY;


  /// TODO: Check if length in Y smaller than length of D_A2_Static



    if (I_YMin < 0)
      I_YMin = 0;
    if (I_YMax >= F_Image.GetNRows())
      I_YMax = F_Image.GetNRows()-1;
    cout << "MFindAndTraceAps::main: I_YMin = " << I_YMin << ", I_YMax = " << I_YMax << endl;
//  if (I_YMin < 0)
    D_A2_Moving.resize(I_YMax - I_YMin + 1, I_XMax - I_XMin + 1);
    D_A2_Moving = F_Image.GetPixArray()(Range(I_YMin, I_YMax), Range(I_XMin, I_XMax));
    I_XMin_Moving = I_XMin;
    I_XMax_Moving = I_XMax;
    I_YMin_Moving = I_YMin;
    I_YMax_Moving = I_YMax;
    cout << "MFindAndTraceAps::main: D_A2_Moving = " << D_A2_Moving << endl;
    F_Image.WriteFits(&D_A2_Static, CString("D_A2_Static.fits"));
    F_Image.WriteFits(&D_A2_Moving, CString("D_A2_Moving.fits"));

    int I_XCor_X_Out = 0;
    int I_XCor_Y_Out = 0;
    if (!F_Image.CrossCorrelate2D(D_A2_Static,
                                  D_A2_Moving,
//                                4,//I_NPixMaxShift_X_Left,
//                                4,//I_NPixMaxShift_X_Right,
//                                20,//I_NPixMaxShift_Y_Down,
//                                20,//I_NPixMaxShift_Y_Up,
//                                20,//I_NSteps_X,
//                                100,//I_NSteps_Y,
                                  I_XCor_X_Out,
                                  I_XCor_Y_Out)){
      exit(EXIT_FAILURE);
      cout << "MFindAndTraceAps::main: ERROR: CrossCorrelate2D(D_A2_Static = " << D_A2_Static << ", D_A2_Moving = " << D_A2_Moving << ") returned FALSE" << endl;
    }
    cout << "MFindAndTraceAps::main: CrossCorrelate2D returned TRUE: I_XCor_X_Out = " << I_XCor_X_Out << ", I_XCor_Y_Out = " << I_XCor_Y_Out << endl;

    D_A1_YLow_New((*(P_I_A1_ApertureNumbers))(I_NearestNeighbourPos)) = double(I_YMin_Moving) + double(I_XCor_Y_Out) - (*P_D_A1_YCenter)(I_NearestNeighbourPos);// - double(I_XCor_Y_Out) + double(I_DY);
    D_A1_YHigh_New((*(P_I_A1_ApertureNumbers))(I_NearestNeighbourPos)) = D_A1_YLow_New((*(P_I_A1_ApertureNumbers))(I_NearestNeighbourPos)) + I_ApertureLength_Down;

    cout << "MFindAndTraceAps::main: D_A1_Reference = " << D_A1_Reference << ": Nearest Neighbour = " << D_A2_ApertureCenters(I_NearestNeighbourPos, Range::all()) << endl;
    cout << "MFindAndTraceAps::main: I_YMin_Moving = " << I_YMin_Moving << ", I_YMax_Moving = " << I_YMax_Moving << endl;
    cout << "MFindAndTraceAps::main: I_YMin_Static = " << I_YMin_Static << ", I_YMax_Static = " << I_YMax_Static << endl;
    cout << "MFindAndTraceAps::main: I_DX = " << I_DX << ", I_DY = " << I_DY << endl;
    cout << "MFindAndTraceAps::main: D_A1_YHigh_New((*(P_I_A1_ApertureNumbers))(I_NearestNeighbourPos)=" << (*(P_I_A1_ApertureNumbers))(I_NearestNeighbourPos) << ") set to " << D_A1_YHigh_New((*(P_I_A1_ApertureNumbers))(I_NearestNeighbourPos)) << endl;
    cout << "MFindAndTraceAps::main: D_A1_YLow_New((*(P_I_A1_ApertureNumbers))(I_NearestNeighbourPos)=" << (*(P_I_A1_ApertureNumbers))(I_NearestNeighbourPos) << ") set to " << D_A1_YLow_New((*(P_I_A1_ApertureNumbers))(I_NearestNeighbourPos)) << endl;
    if ((*P_D_A1_YCenter)(I_NearestNeighbourPos) + D_A1_YHigh_New((*(P_I_A1_ApertureNumbers))(I_NearestNeighbourPos)) >= F_Image.GetNRows())
      D_A1_YLow_New((*(P_I_A1_ApertureNumbers))(I_NearestNeighbourPos)) = F_Image.GetNRows() - (*P_D_A1_YCenter)(I_NearestNeighbourPos)-1;
    if ((*P_D_A1_YCenter)(I_NearestNeighbourPos) + D_A1_YLow_New((*(P_I_A1_ApertureNumbers))(I_NearestNeighbourPos)) < 0)
      D_A1_YLow_New((*(P_I_A1_ApertureNumbers))(I_NearestNeighbourPos)) = 0. - (*P_D_A1_YCenter)(I_NearestNeighbourPos);
    //D_A1_YLow(I_NearestNeighbourPos) - double(I_XCor_Y_Out) + double(I_DY);

    int I_Y_Up = (*P_D_A1_YCenter)(I_NearestNeighbourPos) + D_A1_YHigh_New((*(P_I_A1_ApertureNumbers))(I_NearestNeighbourPos));
    int I_Y_Down = (*P_D_A1_YCenter)(I_NearestNeighbourPos) + D_A1_YLow_New((*(P_I_A1_ApertureNumbers))(I_NearestNeighbourPos));

    D_A1_YLow(I_NearestNeighbourPos) = D_A1_YLow_New((*(P_I_A1_ApertureNumbers))(I_NearestNeighbourPos));
    (*P_D_A1_YHigh)(I_NearestNeighbourPos) = D_A1_YHigh_New((*(P_I_A1_ApertureNumbers))(I_NearestNeighbourPos));

    cout << "MFindAndTraceAps::main: I_Y_Up = " << I_Y_Up << ", I_Y_Down = " << I_Y_Down << endl;

    D_A1_Reference = D_A2_ApertureCenters(I_NearestNeighbourPos, Range::all());
//    if (i_ap == 10){
//      cout << "MFindAndTraceAps::main: i_ap = 1 -> Returning FALSE" << endl;
//      i_ap = F_Image.Get_NApertures()-1;
//      exit(EXIT_FAILURE);
//    }
  }
  F_Image.Set_YHigh(D_A1_YHigh_New);
  F_Image.Set_YLow(D_A1_YLow_New);
  delete(P_D_A1_YHigh);
  delete(P_D_A1_YCenter);
  delete(P_D_A2_XCenters);
  delete(P_I_A1_ApertureNumbers);

  F_Image.WriteDatabaseEntry();

  /// Write aperture header information
  F_Image.WriteApHead(CString("aphead_")+CS_FitsFileName_In+CString(".head"));
*/
  seconds = time(NULL);
  cout << "MFindAndTraceAps::main: Task finished in " << seconds << " seconds" << endl;

  /// Write ImOut
  if (CS_FitsFileName_Out.GetLength() > 1){
    if (!F_Image.SetFileName(CS_FitsFileName_Out))
    {
      cout << "MFindAndTraceAps::main: ERROR: F_Image.SetFileName() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
    if (!F_Image.WriteArray())
    {
      cout << "MFindAndTraceAps::main: ERROR: F_Image.WriteArray() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
  }


  /// Write ImOut
  cout << "MFindAndTraceAps::main: CS_FitsFileName_CentersMarked_Out = " << CS_FitsFileName_CentersMarked_Out << endl;
  if (CS_FitsFileName_CentersMarked_Out.GetLength() > 1){
    cout << "MFindAndTraceAps::main: Starting F_Image.SetFileName(" << CS_FitsFileName_In << ")" << endl;
    if (!F_Image.SetFileName(CS_FitsFileName_In))
    {
      cout << "MFindAndTraceAps::main: ERROR: F_Image.SetFileName() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
    cout << "MFindAndTraceAps::main: Starting F_Image.ReadArray()" << endl;
    if (!F_Image.ReadArray()){
      cout << "MFindAndTraceAps::main: ERROR: F_Image.ReadArray() returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    cout << "MFindAndTraceAps::main: Starting F_Image.ReadDatabaseEntry()" << endl;
    if (!F_Image.ReadDatabaseEntry()){
      cout << "MFindAndTraceAps::main: ERROR: F_Image.ReadDatabaseEntry() returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    cout << "MFindAndTraceAps::main: Starting F_Image.CalcTraceFunctions()" << endl;
    if (!F_Image.CalcTraceFunctions()){
      cout << "MFindAndTraceAps::main: ERROR: F_Image.CalcTraceFunctions() returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    cout << "MFindAndTraceAps::main: Starting F_Image.MarkCenters()" << endl;
    if (!F_Image.MarkCenters())
    {
      cout << "MFindAndTraceAps::main: ERROR: F_Image.WriteArray() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
    cout << "MFindAndTraceAps::main: Starting F_Image.SetFileName(" << CS_FitsFileName_CentersMarked_Out << ")" << endl;
    if (!F_Image.SetFileName(CS_FitsFileName_CentersMarked_Out))
    {
      cout << "MFindAndTraceAps::main: ERROR: F_Image.SetFileName() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
    cout << "MFindAndTraceAps::main: Starting F_Image.WriteArray()" << endl;
    if (!F_Image.WriteArray())
    {
      cout << "MFindAndTraceAps::main: ERROR: F_Image.WriteArray() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
  }

  cout << "MFindAndTraceAps: deleting P_CS" << endl;
  delete(P_CS);
  cout << "MFindAndTraceAps: P_CF_Scatter deleted" << endl;
  return EXIT_SUCCESS;
}
