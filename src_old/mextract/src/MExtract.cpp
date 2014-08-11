/*
author: Andreas Ritter
created: 04/12/2007
last edited: 05/05/2007
compiler: g++ 4.0
basis machine: Ubuntu Linux 6.06
*/

#include "MExtract.h"

using namespace std;

int main(int argc, char *argv[])
{
  cout << "MExtract::main: argc = " << argc << endl;
/*  
  string str("1234567890");
  cout << "str = <" << str << ">, str.length() = <" << str.length() << ">" << endl;
  char dbfn[str.length()+1];
  for (int i=0; i<=str.length(); i++)
    dbfn[i] = '0';
  cout << "dbfn = <" << dbfn << ">" << endl;
  size_t len = str.copy(dbfn, str.length(), 0);
  dbfn[len] = '\0';
  cout << "dbfn = <" << dbfn << ">" << endl;
  dbfn[len-1] = '\0';
  cout << "dbfn = <" << dbfn << ">" << endl;
  
  CFits cf;
  int i_ncols = cf.countCols("/home/azuri/temp/temptab", " ");
  cout << "i_ncols = <" << i_ncols << ">" << endl;
  string st = "01" + str + to_string(i_ncols);
  cout << "st = <" << st << ">" << endl;
  Array<string, 2> S_A2_Data(2,2);
  if (!cf.readFileToStrArr(string("/home/azuri/temp/temptab"), S_A2_Data, string(" "))){
    cout << "Reading file failed" << endl;
  }
  cout << "S_A2_Data = <" << S_A2_Data << ">" << endl;
  
  Array<double, 2> D_A2_Data(2,2);
  if (!cf.readFileToDblArr(string("/home/azuri/temp/temptab"), D_A2_Data, string(" "))){
    cout << "Reading double file failed" << endl;
  }
  cout << "D_A2_Data = <" << D_A2_Data << ">" << endl;
  
  Array<string, 1> S_A1_Data(2);
  if (!cf.readFileLinesToStrArr(string("/home/azuri/temp/temptab"), S_A1_Data)){
    cout << "Reading file failed" << endl;
  }
  cout << "S_A1_Data = <" << S_A1_Data << ">" << endl;
  
  string sTemp = "0123456789";
  string sst = sTemp.substr(0, 7);
  cout << "sTemp = <" << sTemp << ">, sst = <" << sst << ">" << endl;
  if (strcmp(sst.c_str(), "0123456789") == 0)
    cout << "strcmp(sst.c_str(), \"0123456789\") returned '0'" << endl;
  char c[255];
  strcpy(c, sTemp.c_str());
  cout << "c = <" << c << ">" << endl;
  
  const char *pc = sTemp.c_str();
  cout << "pc = <" << pc << ">" << endl;

  if (sTemp.compare("0123456789") == 0)
    cout << "sTemp == <0123456789>" << endl;
  
  exit(EXIT_FAILURE);  
  */
  if (argc < 6)
  {
    cout << "MExtract::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: optextract <char[] (@)FitsFileName_In> <char[] (@)DatabaseFileName_In> <char[] (@)FitsFileName_Out> <double Gain> <double ReadOutNoise> [TELLURIC=int[0 - none, 1 - Piskunov, 2 - LinFit]] [MAX_ITER_SF=int] [MAX_ITER_SKY=int] [MAX_ITER_SIG=int] [SWATH_WIDTH=int] [SF_SMOOTH=double] [SP_SMOOTH=int] [WING_SMOOTH_FACTOR=double] [ERR_IN=char[](@)] [ERR_OUT_2D=char[](@)] [ERR_OUT_EC=char[](@)] [ERRFIT_OUT_EC=char[](@)] [SKY_OUT_EC=char[](@)][SKYFIT_OUT_EC=char[](@)] [SKY_OUT_2D=char[](@)] [SKYFIT_OUT_2D=char[](@)] [SKY_ERR_OUT_EC=char[](@)] [SKYFIT_ERR_OUT_EC=char[](@)] [PROFILE_OUT=char[](@)] [IM_REC_OUT=char[](@)] [REC_FIT_OUT=char[](@)] [MASK_OUT=char[](@)] [SPFIT_OUT_EC=char[](@)] [EC_FROM_PROFILE_OUT=char[](@)] [AREA=[int(xmin),int(xmax),int(ymin),int(ymax)]] [XCOR_PROF=int] [APERTURES=char[](@)]" << endl;//"[, ERR_FROM_PROFILE_OUT=char[]])" << endl;
    cout << "FitsFileName_In: image to extract" << endl;
    cout << "DatabaseFileName_In: aperture-definition file to use for extraction" << endl;
    cout << "FitsFileName_Out: output filename containing extracted spectra" << endl;
    cout << "Gain: CCD gain" << endl;
    cout << "ReadOutNoise: CCD readout noise" << endl;
    cout << "TELLURIC: 0 - none, 1 - Piskunov, 2 - LinFit" << endl;
    cout << "MAX_ITER_SF: maximum number of iterations calculating the slit function (spatial profile)" << endl;
    cout << "MAX_ITER_SKY: maximum number of iterations calculating the sky (TELLURIC = 2 only)" << endl;
    cout << "MAX_ITER_SIG: maximum number of iterations rejecting cosmic-ray hits" << endl;
    cout << "SWATH_WIDTH: width of swath (bin) for which an individual profile shall be calculated" << endl;
    cout << "SMOOTH_SF: Width of median SlitFunc smoothing" << endl;
    cout << "SMOOTH_SP: Width of median Spectrum/Blaze smoothing" << endl;
    cout << "WING_SMOOTH_FACTOR: Width of median SlitFunc-Wing smoothing" << endl;
    cout << "ERR_IN: input image containing the uncertainties in the pixel values of FitsFileName_In" << endl;
    cout << "ERR_OUT_2D: output uncertainty image - same as ERR_IN, but with detected cosmic-ray hits set to 10,000" << endl;
    cout << "ERR_OUT_EC: output file containing the uncertainties in the extracted spectra's pixel values from ExtractErrors" << endl;
    cout << "ERRFIT_OUT_EC: output file containing the uncertainties in the extracted spectra's pixel values from Fit" << endl;
    cout << "SKY_OUT_EC: output sky-only spectra (TELLURIC > 0 only)" << endl;
    cout << "SKYFIT_OUT_EC: output sky-only spectra (TELLURIC > 0 only)" << endl;
    cout << "SKY_OUT_2D: reconstructed sky-only image" << endl;
    cout << "SKYFIT_OUT_2D: reconstructed sky-only image" << endl;
    cout << "SKY_ERR_OUT_EC: uncertainties in the calculated sky-only values (only differs from SKYFIT_ERR_OUT_EC if TELLURIC==1" << endl;
    cout << "SKYFIT_ERR_OUT_EC: uncertainties in the calculated sky-only values from Fit" << endl;
    cout << "PROFILE_OUT: reconstructed image of the spatial profiles" << endl;
    cout << "IM_REC_OUT: reconstructed input image from the profile-fitting/extraction" << endl;
    cout << "SPFIT_OUT_EC: extracted spectra from linear fit of spatial profiles to input spectra with 3-sigma rejection (ignoring mask), with sky if TELLURIC>0, without sky if TELLURIC=0" << endl;
    cout << "REC_FIT_OUT: reconstructed input image for SPFIT_OUT_EC" << endl;
    cout << "MASK_OUT: output mask with detected cosmic-ray hits set to 0, good pixels set to 1" << endl;
    cout << "EC_FROM_PROFILE_OUT: extracted spectra from simply multiplying the input image with the profile image as weight and summing up" << endl;///SHALL I REJECT COSMIC-RAY HITS???????????????
    cout << "AREA: Area from which to extract spectra if center of aperture is in specified area" << endl;
    cout << "XCOR_PROF: How many cross-correlation runs from -1 pixel to +1 pixel compared to XCenter?" << endl;
    cout << "APERTURES: input filename containing a list of apertures to extract" << endl;
//    cout << "ERR_FROM_PROFILE_OUT: uncertainties of extracted spectra from EC_FROM_PROFILE_OUT (EC_FROM_PROFILE_OUT must be set as well)" << endl;
    exit(EXIT_FAILURE);
  }

/*  double D_D1 = 34.;
  double D_D2 = 34.;
  cout << "D_D1 - D_D2 = " << D_D1 - D_D2 << endl;
  if (D_D1 == D_D2)
    cout << "D_D1 == D_D2" << endl;
  cout << "abs(D_D1 - D_D2) = " << abs(D_D1 - D_D2) << endl;
  cout << "fabs(D_D1 - D_D2) = " << fabs(D_D1 - D_D2) << endl;
  exit(EXIT_FAILURE);
*/
  CFits cf;
  blitz::Array<int, 1> I_A1_Area(4);
  blitz::Array<string, 1> S_A1_Args(8);
  S_A1_Args = "";
  void **PP_Args;
  PP_Args = (void**)malloc(sizeof(void*) * 8);
  string s = " ";
  string sComp = " ";
  string sSubStr = " ";
  string errorImageIn = " ";
  string errorImageOut = " ";
  string errorOutEc = " ";
  string errorFitOutEc = " ";
  string skyOutEc = " ";
  string skyFitOutEc = " ";
  string skyImageOut = " ";
  string skyFitImageOut = " ";
  string skyErrorOutEc = " ";
  string skyFitErrorOutEc = " ";
  string imageOut = " ";
  string imageFitOut = " ";
  string profileImageOut = " ";
  string maskOut = " ";
  string specFitOut = " ";
  string specFromProfileOut = " ";
  string errorFromProfileOut = " ";
  string apertureListIn = " ";

  int I_SwathWidth = 0;
  int I_MaxIterSF = 8;
  int I_MaxIterSky = 12;
  int I_MaxIterSig = 2;
  int I_SmoothSP = 1;
  double D_SmoothSF = 1.;
  double D_WingSmoothFactor = 1.;
  char *P_CharArr_In = (char*)argv[1];
  char *P_CharArr_DB = (char*)argv[2];
  char *P_CharArr_Out = (char*)argv[3];
  double D_Gain = (double)(atof((char*)argv[4]));
  cout << "MExtract::main: D_Gain set to " << D_Gain << endl;
  double D_ReadOutNoise = (double)(atof((char*)argv[5]));
  cout << "MExtract::main: D_ReadOutNoise set to " << D_ReadOutNoise << endl;
  int I_Telluric=0;
  int I_XCorProf = 0;
  blitz::Array<int, 1> I_A1_Apertures(1);
  I_A1_Apertures = 0;
  bool B_AperturesSet = false;

  /// read optional parameters
  for (int i = 6; i < argc; i++){
    cout << "MExtract: Reading Parameter " << i << endl;
    s = ((char*)argv[i]);
    cout << "MExtract: Reading Parameter " << s << endl;

    sSubStr = s.substr(0,s.find('='));
    cout << "MExtract::main: sSubStr set to " << sSubStr << endl;

    sComp = "TELLURIC";
    if (sSubStr.compare(sComp) == 0){
      sSubStr = s.substr(sComp.length()+1);
      I_Telluric = stoi(sSubStr);
      cout << "MExtract::main: I_Telluric set to " << I_Telluric << endl;
    }

    sComp = "SWATH_WIDTH";
    if (sSubStr.compare(sComp) == 0){
      sSubStr = s.substr(sComp.length()+1);
      I_SwathWidth = stoi(sSubStr);
      cout << "MExtract::main: I_SwathWidth set to " << I_SwathWidth << endl;
    }

    sComp = "MAX_ITER_SF";
    if (sSubStr.compare(sComp) == 0){
      sSubStr = s.substr(sComp.length()+1);
      I_MaxIterSF = stoi(sSubStr);
      cout << "MExtract::main: I_MaxIterSF set to " << I_MaxIterSF << endl;
    }

    sComp = "MAX_ITER_SKY";
    if (sSubStr.compare(sComp) == 0){
      sSubStr = s.substr(sComp.length()+1);
      I_MaxIterSky = stoi(sSubStr);
      cout << "MExtract::main: I_MaxIterSky set to " << I_MaxIterSky << endl;
    }

    sComp = "MAX_ITER_SIG";
    if (sSubStr.compare(sComp) == 0){
      sSubStr = s.substr(sComp.length()+1);
      I_MaxIterSig = stoi(sSubStr);
      cout << "MExtract::main: I_MaxIterSig set to " << I_MaxIterSig << endl;
    }

    sComp = "SMOOTH_SF";
    if (sSubStr.compare(sComp) == 0){
      sSubStr = s.substr(sComp.length()+1);
      D_SmoothSF = stod(sSubStr);
      cout << "MExtract::main: D_SmoothSF set to " << D_SmoothSF << endl;
      S_A1_Args(3) = "LAMBDA_SF";
      PP_Args[3] = &D_SmoothSF;
    }

    sComp = "SMOOTH_SP";
    if (sSubStr.compare(sComp) == 0){
      sSubStr = s.substr(sComp.length()+1);
      I_SmoothSP = stoi(sSubStr);
      cout << "MExtract::main: I_SmoothSP set to " << I_SmoothSP << endl;
      S_A1_Args(4) = "LAMBDA_SP";
      PP_Args[4] = &I_SmoothSP;
    }

    sComp = "WING_SMOOTH_FACTOR";
    if (sSubStr.compare(sComp) == 0){
      sSubStr = s.substr(sComp.length()+1);
      D_WingSmoothFactor = stod(sSubStr);
      cout << "MExtract::main: D_WingSmoothFactor set to " << D_WingSmoothFactor << endl;
      S_A1_Args(5) = "WING_SMOOTH_FACTOR";
      PP_Args[5] = &D_WingSmoothFactor;
    }

    /// AREA
    sComp = "AREA";
    if (sSubStr.compare(sComp) == 0){
      string sTemp = ",";
      int i_pos_a = sComp.length()+2;
      int i_pos_b = s.find(sTemp,i_pos_a+1);
      cout << "MExtract: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
      sSubStr = s.substr(i_pos_a,i_pos_b-i_pos_a);
      cout << "MExtract: sSubStr set to " << sSubStr << endl;
      I_A1_Area(0) = stoi(sSubStr);
      cout << "MExtract: I_A1_Area(0) set to " << I_A1_Area(0) << endl;
      i_pos_a = i_pos_b+1;
      i_pos_b = s.find(sTemp,i_pos_a+1);
      cout << "MExtract: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
      sSubStr = s.substr(i_pos_a,i_pos_b-i_pos_a);
      cout << "MExtract: sSubStr set to " << sSubStr << endl;
      I_A1_Area(1) = stoi(sSubStr);
      cout << "MExtract: I_A1_Area(1) set to " << I_A1_Area(1) << endl;
      i_pos_a = i_pos_b+1;
      i_pos_b = s.find(sTemp,i_pos_a+1);
      cout << "MExtract: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
      sSubStr = s.substr(i_pos_a,i_pos_b-i_pos_a);
      cout << "MExtract: sSubStr set to " << sSubStr << endl;
      I_A1_Area(2) = stoi(sSubStr);
      cout << "MExtract: I_A1_Area(2) set to " << I_A1_Area(2) << endl;
      i_pos_a = i_pos_b+1;
      sTemp = "]";
      i_pos_b = s.find(sTemp,i_pos_a+1);
      if (i_pos_b < 0){
        sTemp = ")";
        i_pos_b = s.find(sTemp,i_pos_a+1);
      }
      cout << "MExtract: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
      sSubStr = s.substr(i_pos_a,i_pos_b-i_pos_a);
      cout << "MExtract: sSubStr set to " << sSubStr << endl;
      I_A1_Area(3) = stoi(sSubStr);
      cout << "MExtract: I_A1_Area(3) set to " << I_A1_Area(3) << endl;
      S_A1_Args(2) = "AREA";
      PP_Args[2] = &I_A1_Area;
      cout << "MExtract::main: I_A1_Area set to " << I_A1_Area << endl;
    }

    /// 2D
    sComp = "ERR_IN";
    if (sSubStr.compare(sComp) == 0){
      errorImageIn = s.substr(sComp.length()+1);
      cout << "MExtract::main: ERR_IN set to " << errorImageIn << endl;
    }

    /// 2D
    sComp = "ERR_OUT_2D";
    if (sSubStr.compare(sComp) == 0){
      errorImageOut = s.substr(sComp.length()+1);
      cout << "MExtract::main: ERR_OUT_2D set to " << errorImageOut << endl;
    }

    /// 1D
    sComp = "ERR_OUT_EC";
    if (sSubStr.compare(sComp) == 0){
      errorOutEc = s.substr(sComp.length()+1);
      cout << "MExtract::main: ERR_OUT_EC set to " << errorOutEc << endl;
    }

    sComp = "ERRFIT_OUT_EC";
    if (sSubStr.compare(sComp) == 0){
      errorFitOutEc = s.substr(sComp.length()+1);
      cout << "MExtract::main: ERR_OUT_EC set to " << errorFitOutEc << endl;
    }

    /// 1D
    sComp = "SKY_OUT_EC";
    if (sSubStr.compare(sComp) == 0){
      skyOutEc = s.substr(sComp.length()+1);
      cout << "MExtract::main: SKY_OUT_EC set to " << skyOutEc << endl;
    }

    sComp = "SKYFIT_OUT_EC";
    if (sSubStr.compare(sComp) == 0){
      skyFitOutEc = s.substr(sComp.length()+1);
      cout << "MExtract::main: SKYFIT_OUT_EC set to " << skyFitOutEc << endl;
    }

    /// 2D
    sComp = "SKY_OUT_2D";
    if (sSubStr.compare(sComp) == 0){
      skyImageOut = s.substr(sComp.length()+1);
      cout << "MExtract::main: SKY_OUT_2D set to " << skyImageOut << endl;
    }
    
    sComp = "SKYFIT_OUT_2D";
    if (sSubStr.compare(sComp) == 0){
      skyFitImageOut = s.substr(sComp.length()+1);
      cout << "MExtract::main: SKYFIT_OUT_2D set to " << skyFitImageOut << endl;
    }

    /// 1D
    sComp = "SKY_ERR_OUT_EC";
    if (sSubStr.compare(sComp) == 0){
      skyErrorOutEc = s.substr(sComp.length()+1);
      cout << "MExtract::main: SKY_ERR_OUT_EC set to " << skyErrorOutEc << endl;
    }
    
    sComp = "SKYFIT_ERR_OUT_EC";
    if (sSubStr.compare(sComp) == 0){
      skyFitErrorOutEc = s.substr(sComp.length()+1);
      cout << "MExtract::main: SKYFIT_ERR_OUT_EC set to " << skyFitErrorOutEc << endl;
    }

    /// 2D
    sComp = "IM_REC_OUT";
    if (sSubStr.compare(sComp) == 0){
      imageOut = s.substr(sComp.length()+1);
      cout << "MExtract::main: IM_REC_OUT set to " << imageOut << endl;
    }

    /// 2D
    sComp = "REC_FIT_OUT";
    if (sSubStr.compare(sComp) == 0){
      imageFitOut = s.substr(sComp.length()+1);
      cout << "MExtract::main: REC_FIT_OUT set to " << imageFitOut << endl;
    }

    /// 2D
    sComp = "PROFILE_OUT";
    if (sSubStr.compare(sComp) == 0){
      profileImageOut = s.substr(sComp.length()+1);
      cout << "MExtract::main: PROFILE_OUT set to " << profileImageOut << endl;
    }

    /// 2D
    sComp = "MASK_OUT";
    if (sSubStr.compare(sComp) == 0){
      maskOut = s.substr(sComp.length()+1);
      cout << "MExtract::main: MASK_OUT set to " << maskOut << endl;
    }

    /// 1D
    sComp = "SPFIT_OUT_EC";
    if (sSubStr.compare(sComp) == 0){
      specFitOut = s.substr(sComp.length()+1);
      cout << "MExtract::main: SPFIT_OUT_EC set to " << specFitOut << endl;
    }

    /// 1D
    sComp = "EC_FROM_PROFILE_OUT";
    if (sSubStr.compare(sComp) == 0){
      specFromProfileOut = s.substr(sComp.length()+1);
      cout << "MExtract::main: EC_FROM_PROFILE_OUT set to " << specFromProfileOut << endl;
    }

    /// 1D
    sComp = "ERR_FROM_PROFILE_OUT";
    if (sSubStr.compare(sComp) == 0){
      errorFromProfileOut = s.substr(sComp.length()+1);
      cout << "MExtract::main: ERR_FROM_PROFILE_OUT set to " << errorFromProfileOut << endl;
    }

    sComp = "APERTURES";
    if (sSubStr.compare(sComp) == 0){
      apertureListIn = s.substr(sComp.length()+1);
      cout << "MExtract::main: apertureListIn set to " << apertureListIn << endl;
      B_AperturesSet = true;
      blitz::Array<string, 1> S_A1_AperturesToExtract(1);
      S_A1_AperturesToExtract = " ";
      if (!cf.readFileLinesToStrArr(apertureListIn, S_A1_AperturesToExtract)){
        cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << apertureListIn << ") returned FALSE" << endl;
        exit(EXIT_FAILURE);
      }
      I_A1_Apertures.resize(S_A1_AperturesToExtract.size());
      for (int i_ap=0; i_ap < S_A1_AperturesToExtract.size(); i_ap++)
        I_A1_Apertures(i_ap) = stoi(S_A1_AperturesToExtract(i_ap));
      S_A1_Args(7) = "APERTURES";
      PP_Args[7] = &I_A1_Apertures;
    }

    sComp = "XCOR_PROF";
    if (sSubStr.compare(sComp) == 0){
      sSubStr = s.substr(sComp.length()+1);
      I_XCorProf = stoi(sSubStr);
      cout << "MExtract::main: I_XCorProf set to " << I_XCorProf << endl;
      S_A1_Args(6) = "XCOR_PROF";
      PP_Args[6] = &I_XCorProf;
    }
  }

//  return false;

  if (I_Telluric > 0){
    S_A1_Args(1) = "TELLURIC";
    PP_Args[1] = &I_Telluric;
  }
  time_t seconds;
//  if (argc == 8)
//  {
//    I_SwathWidth = (int)(atoi((char*)argv[7]));
//    cout << "MExtract::main: I_SwathWidth set to " << I_SwathWidth << endl;
  if (I_SwathWidth > 0.){
    S_A1_Args(0) = "SWATH_WIDTH";
    PP_Args[0] = &I_SwathWidth;
  }
  else
  {
    S_A1_Args(0) = "";
    PP_Args[0] = &I_SwathWidth;
  }

  bool B_Lists = false;
  string fitsFileName_In;
  fitsFileName_In = P_CharArr_In;
  blitz::Array<string, 1> S_A1_FitsFileNames_In(1);
  S_A1_FitsFileNames_In(0) = fitsFileName_In;
  if (isList(fitsFileName_In)){
    B_Lists = true;
    if (!cf.readFileLinesToStrArr(fitsFileName_In, S_A1_FitsFileNames_In)){
      cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << fitsFileName_In << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
  }

  string fitsFileName_Out;
  fitsFileName_Out = P_CharArr_Out;
  blitz::Array<string, 1> S_A1_FitsFileNames_Out(1);
  S_A1_FitsFileNames_Out(0) = fitsFileName_Out;
  if (B_Lists){
    if (isList(fitsFileName_Out)){
      if (!cf.readFileLinesToStrArr(fitsFileName_Out, S_A1_FitsFileNames_Out)){
        cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << fitsFileName_Out << ") returned FALSE" << endl;
        exit(EXIT_FAILURE);
      }
    }
    else{
      cout << "MExtract::main: ERROR: InputFileNames is list but OutputFileNames is not" << endl;
      exit(EXIT_FAILURE);
    }
  }

  string databaseFileName_In;
  databaseFileName_In = P_CharArr_DB;
  blitz::Array<string, 1> S_A1_DatabaseFileNames_In(1);
  S_A1_DatabaseFileNames_In(0) = databaseFileName_In;
  if (B_Lists){
    if (isList(databaseFileName_In)){
      if (!cf.readFileLinesToStrArr(databaseFileName_In, S_A1_DatabaseFileNames_In)){
        cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << databaseFileName_In << ") returned FALSE" << endl;
        exit(EXIT_FAILURE);
      }
    }
    else{
      cout << "MExtract::main: ERROR: InputFileNames is list but DatabaseFileNames is not" << endl;
      exit(EXIT_FAILURE);
    }
  }

  blitz::Array<string, 1> S_A1_ErrIn(1);
  if (errorImageIn.length() > 1){
    if (B_Lists){
      if (isList(errorImageIn)){
        if (!cf.readFileLinesToStrArr(errorImageIn, S_A1_ErrIn)){
          cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << errorImageIn << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MExtract::main: ERROR: InputFileNames is list but ErrIn is not" << endl;
      }
    }
    else{
      S_A1_ErrIn(0) = errorImageIn;
    }
  }

  blitz::Array<string, 1> S_A1_ErrOut(1);
  if (errorImageOut.length() > 1){
    if (B_Lists){
      if (isList(errorImageOut)){
        if (!cf.readFileLinesToStrArr(errorImageOut, S_A1_ErrOut)){
          cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << errorImageOut << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MExtract::main: ERROR: InputFileNames is list but ErrOut is not" << endl;
      }
    }
    else{
      S_A1_ErrOut(0) = errorImageOut;
    }
  }

  blitz::Array<string, 1> S_A1_ErrOutEc(1);
  if (errorOutEc.length() > 1){
    if (B_Lists){
      if (isList(errorOutEc)){
        if (!cf.readFileLinesToStrArr(errorOutEc, S_A1_ErrOutEc)){
          cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << errorOutEc << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MExtract::main: ERROR: InputFileNames is list but ErrOutEc is not" << endl;
      }
    }
    else{
      S_A1_ErrOutEc(0) = errorOutEc;
    }
  }

  blitz::Array<string, 1> S_A1_ErrFitOutEc(1);
  if (errorFitOutEc.length() > 1){
    if (B_Lists){
      if (isList(errorFitOutEc)){
        if (!cf.readFileLinesToStrArr(errorFitOutEc, S_A1_ErrFitOutEc)){
          cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << errorFitOutEc << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MExtract::main: ERROR: InputFileNames is list but ErrFitOutEc is not" << endl;
      }
    }
    else{
      S_A1_ErrFitOutEc(0) = errorFitOutEc;
    }
  }

  blitz::Array<string, 1> S_A1_SkyOut(1);
  if (skyOutEc.length() > 1){
    if (B_Lists){
      if (isList(skyOutEc)){
        if (!cf.readFileLinesToStrArr(skyOutEc, S_A1_SkyOut)){
          cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << skyOutEc << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MExtract::main: ERROR: InputFileNames is list but SkyOut is not" << endl;
      }
    }
    else{
      S_A1_SkyOut(0) = skyOutEc;
    }
  }

  blitz::Array<string, 1> S_A1_SkyFitOut(1);
  if (skyFitOutEc.length() > 1){
    if (B_Lists){
      if (isList(skyFitOutEc)){
        if (!cf.readFileLinesToStrArr(skyFitOutEc, S_A1_SkyFitOut)){
          cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << skyFitOutEc << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MExtract::main: ERROR: InputFileNames is list but SkyFitOut is not" << endl;
      }
    }
    else{
      S_A1_SkyFitOut(0) = skyFitOutEc;
    }
  }

  blitz::Array<string, 1> S_A1_SkyImagesOut(1);
  if (skyImageOut.length() > 1){
    if (B_Lists){
      if (isList(skyImageOut)){
        if (!cf.readFileLinesToStrArr(skyImageOut, S_A1_SkyImagesOut)){
          cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << skyImageOut << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MExtract::main: ERROR: InputFileNames is list but SkyArrOut is not" << endl;
      }
    }
    else{
      S_A1_SkyImagesOut(0) = skyImageOut;
    }
  }

  blitz::Array<string, 1> S_A1_SkyFitImagesOut(1);
  if (skyFitImageOut.length() > 1){
    if (B_Lists){
      if (isList(skyFitImageOut)){
        if (!cf.readFileLinesToStrArr(skyFitImageOut, S_A1_SkyFitImagesOut)){
          cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << skyFitImageOut << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MExtract::main: ERROR: InputFileNames is list but SkyFitArrOut is not" << endl;
      }
    }
    else{
      S_A1_SkyFitImagesOut(0) = skyFitImageOut;
    }
  }

  blitz::Array<string, 1> S_A1_SkyErrorOutEc(1);
  if (skyErrorOutEc.length() > 1){
    if (B_Lists){
      if (isList(skyErrorOutEc)){
        if (!cf.readFileLinesToStrArr(skyErrorOutEc, S_A1_SkyErrorOutEc)){
          cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << skyErrorOutEc << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MExtract::main: ERROR: InputFileNames is list but SkyErrOut is not" << endl;
      }
    }
    else{
      S_A1_SkyErrorOutEc(0) = skyErrorOutEc;
    }
  }

  blitz::Array<string, 1> S_A1_SkyFitErrorOutEc(1);
  if (skyFitErrorOutEc.length() > 1){
    if (B_Lists){
      if (isList(skyFitErrorOutEc)){
        if (!cf.readFileLinesToStrArr(skyFitErrorOutEc, S_A1_SkyFitErrorOutEc)){
          cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << skyFitErrorOutEc << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MExtract::main: ERROR: InputFileNames is list but skyFitErrorOutEc is not" << endl;
      }
    }
    else{
      S_A1_SkyFitErrorOutEc(0) = skyFitErrorOutEc;
    }
  }

  blitz::Array<string, 1> S_A1_ImageOut(1);
  if (imageOut.length() > 1){
    if (B_Lists){
      if (isList(imageOut)){
        if (!cf.readFileLinesToStrArr(imageOut, S_A1_ImageOut)){
          cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << imageOut << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MExtract::main: ERROR: InputFileNames is list but imageOut is not" << endl;
      }
    }
    else{
      S_A1_ImageOut(0) = imageOut;
    }
  }

  blitz::Array<string, 1> S_A1_ImageFitOut(1);
  if (imageFitOut.length() > 1){
    if (B_Lists){
      if (isList(imageFitOut)){
        if (!cf.readFileLinesToStrArr(imageFitOut, S_A1_ImageFitOut)){
          cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << imageFitOut << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MExtract::main: ERROR: InputFileNames is list but imageFitOut is not" << endl;
      }
    }
    else{
      S_A1_ImageFitOut(0) = imageFitOut;
    }
  }

  blitz::Array<string, 1> S_A1_ProfileImagesOut(1);
  if (profileImageOut.length() > 1){
    if (B_Lists){
      if (isList(profileImageOut)){
        if (!cf.readFileLinesToStrArr(profileImageOut, S_A1_ProfileImagesOut)){
          cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << profileImageOut << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MExtract::main: ERROR: InputFileNames is list but profileImageOut is not" << endl;
      }
    }
    else{
      S_A1_ProfileImagesOut(0) = profileImageOut;
    }
  }

  //  P_CS_MaskOut = CS.SubString(CS_comp.GetLength()+1);
  blitz::Array<string, 1> S_A1_MaskOut(1);
  if (maskOut.length() > 1){
    if (B_Lists){
      if (isList(maskOut)){
        if (!cf.readFileLinesToStrArr(maskOut, S_A1_MaskOut)){
          cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << maskOut << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MExtract::main: ERROR: InputFileNames is list but maskOut is not" << endl;
      }
    }
    else{
      S_A1_MaskOut(0) = maskOut;
    }
  }

  blitz::Array<string, 1> S_A1_SpecFitOut(1);
  if (specFitOut.length() > 1){
    if (B_Lists){
      if (isList(specFitOut)){
        if (!cf.readFileLinesToStrArr(specFitOut, S_A1_SpecFitOut)){
          cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << specFitOut << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MExtract::main: ERROR: InputFileNames is list but specFitOut is not" << endl;
      }
    }
    else{
      S_A1_SpecFitOut(0) = specFitOut;
    }
  }

  blitz::Array<string, 1> S_A1_SpecFromProfileOut(1);
  if (specFromProfileOut.length() > 1){
    if (B_Lists){
      if (isList(specFromProfileOut)){
        if (!cf.readFileLinesToStrArr(specFromProfileOut, S_A1_SpecFromProfileOut)){
          cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << specFromProfileOut << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MExtract::main: ERROR: InputFileNames is list but specFromProfileOut is not" << endl;
      }
    }
    else{
      S_A1_SpecFromProfileOut(0) = specFromProfileOut;
    }
  }

  blitz::Array<string, 1> S_A1_ErrorFromProfileOut(1);
  if (errorFromProfileOut.length() > 1){
    if (B_Lists){
      if (isList(errorFromProfileOut)){
        if (!cf.readFileLinesToStrArr(errorFromProfileOut, S_A1_ErrorFromProfileOut)){
          cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << errorFromProfileOut << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MExtract::main: ERROR: InputFileNames is list but errorFromProfileOut is not" << endl;
      }
    }
    else{
      S_A1_ErrorFromProfileOut(0) = errorFromProfileOut;
    }
  }

  CFits F_Image;
  CFits F_OutImage;
  for (int i_file = 0; i_file < S_A1_FitsFileNames_In.size(); i_file++){
    cout << "MExtract::main: Starting F_Image.setFileName(" << fitsFileName_In << ")" << endl;
    if (!F_Image.setFileName(S_A1_FitsFileNames_In(i_file)))
    {
      cout << "MExtract::main: ERROR: F_Image.setFileName(" << fitsFileName_In << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    /// Set ReadOutNoise
    cout << "MExtract::main: Starting F_Image.Set_ReadOutNoise(" << D_ReadOutNoise << ")" << endl;
    if (!F_Image.Set_ReadOutNoise( D_ReadOutNoise ))
    {
      cout << "MExtract::main: ERROR: F_Image.Set_ReadOutNoise(" << D_ReadOutNoise << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    /// Set Gain
    cout << "MExtract::main: Starting F_Image.Set_Gain(" << D_Gain << ")" << endl;
    if (!F_Image.Set_Gain( D_Gain ))
    {
      cout << "MExtract::main: ERROR: F_Image.Set_Gain(" << D_Gain << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }

    /// Set Oversample to 10
    cout << "MExtract::main: Starting F_Image.Set_OverSample(10)" << endl;
    if (!F_Image.Set_OverSample( 10 ))
    {
      cout << "MExtract::main: ERROR: F_Image.Set_OverSample() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    /// Set I_MaxIterSF
    cout << "MExtract::main: Starting F_Image.Set_MaxIterSF(" << I_MaxIterSF << ")" << endl;
    if (!F_Image.Set_MaxIterSF( I_MaxIterSF ))
    {
      cout << "MExtract::main: ERROR: F_Image.Set_MaxIterSF(" << I_MaxIterSF << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    /// Set I_MaxIterSky
    cout << "MExtract::main: Starting F_Image.Set_MaxIterSky(" << I_MaxIterSky << ")" << endl;
    if (!F_Image.Set_MaxIterSky( I_MaxIterSky ))
    {
      cout << "MExtract::main: ERROR: F_Image.Set_MaxIterSky(" << I_MaxIterSky << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    /// Set I_MaxIterSig
    cout << "MExtract::main: Starting F_Image.Set_MaxIterSig(" << I_MaxIterSig << ")" << endl;
    if (!F_Image.Set_MaxIterSig( I_MaxIterSig ))
    {
      cout << "MExtract::main: ERROR: F_Image.Set_MaxIterSig(" << I_MaxIterSig << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    /// Read FitsFile
    cout << "MExtract::main: Starting F_Image.ReadArray()" << endl;
    if (!F_Image.ReadArray())
    {
      cout << "MExtract::main: ERROR: F_Image.ReadArray() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
    F_Image.GetPixArray() = F_Image.GetPixArray() * F_Image.Get_Gain();

    /// Set DatabaseFileName_In
    cout << "MExtract::main: Starting F_Image.setDatabaseFileName(" << S_A1_DatabaseFileNames_In(i_file) << ")" << endl;
    if (!F_Image.setDatabaseFileName(S_A1_DatabaseFileNames_In(i_file)))
    {
      cout << "MExtract::main: ERROR: F_Image.setDatabaseFileName(" << S_A1_DatabaseFileNames_In(i_file) << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    /// Read DatabaseFileName_In
    cout << "MExtract::main: Starting F_Image.ReadDatabaseEntry()" << endl;
    if (!F_Image.ReadDatabaseEntry())
    {
      cout << "MExtract::main: ERROR: F_Image.ReadDatabaseEntry() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    /// Calculate Trace Functions
    cout << "MExtract::main: Starting F_Image.CalcTraceFunctions()" << endl;
    if (!F_Image.CalcTraceFunctions())
    {
      cout << "MExtract::main: ERROR: F_Image.CalcTraceFunctions() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

//  blitz::Array<double, 1> D_A1_YLow(1);
//  D_A1_YLow(0) = -401;
//  F_Image.Set_YLow(D_A1_YLow);
//  D_A1_YLow(0) = 401;
//  F_Image.Set_YHigh(D_A1_YLow);

    cout << "MExtract::main: errorImageIn = " << errorImageIn << ")" << endl;
    if (errorImageIn.length() > 1){
    /// Set ErrFileName_In
      cout << "MExtract::main: Starting F_Image.setErrFileName(" << S_A1_ErrIn(i_file) << ")" << endl;
      if (!F_Image.setErrFileName(S_A1_ErrIn(i_file)))
      {
        cout << "MExtract::main: ERROR: F_Image.setErrFileName(" << S_A1_ErrIn(i_file) << ") returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }

      /// Read Error image
      cout << "MExtract::main: Starting F_Image.ReadErrArray()" << endl;
      if (!F_Image.ReadErrArray())
      {
        cout << "MExtract::main: ERROR: F_Image.ReadErrArray() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
    }


    /// Write aperture header information
    F_Image.WriteApHead(string("aphead_")+fitsFileName_In+string(".head"));

    if (!B_AperturesSet){
      Array<int, 1> *P_I_A1_Apertures = F_Image.IndGenArr(F_Image.Get_NApertures());
      I_A1_Apertures.resize(P_I_A1_Apertures->size());
      I_A1_Apertures = (*P_I_A1_Apertures);
      delete(P_I_A1_Apertures);
    }

    /// Calculate Profile Image
    seconds = time(NULL);
    cout << "MExtract::main: Starting F_Image.MkProfIm(): time = " << seconds << endl;

    if (!F_Image.MkProfIm(S_A1_Args, PP_Args))
    {
      cout << "MExtract::main: ERROR: F_Image.MkProfIm() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
    seconds = time(NULL);
    cout << "MExtract::main: MkProfIm returned true at " << seconds << endl;

    /// Set CS_FitsFileName_In
    cout << "MExtract::main: Starting F_OutImage.setFileName(" << S_A1_FitsFileNames_In(i_file) << ")" << endl;
    if (!F_OutImage.setFileName(S_A1_FitsFileNames_In(i_file)))
    {
      cout << "MExtract::main: ERROR: F_OutImage.setFileName(" << S_A1_FitsFileNames_In(i_file) << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    ///Read FitsFile
    cout << "MExtract::main: Starting F_OutImage.ReadArray()" << endl;
    if (!F_OutImage.ReadArray())
    {
      cout << "MExtract::main: ERROR: F_OutImage.ReadArray() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }


    /// Write MaskOut 2D
    if (maskOut.length() > 1){
      if (!F_OutImage.setFileName(S_A1_MaskOut(i_file)))
      {
        cout << "MExtract::main: ERROR: F_OutImage.setFileName() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
      cout << "MExtract::main: Starting to write MaskOut" << endl;
      blitz::Array<int, 2> I_A2_MaskArray(F_Image.GetNRows(), F_Image.GetNCols());
      I_A2_MaskArray = F_Image.GetMaskArray();
      blitz::Array<double, 2> D_A2_MaskArray(F_Image.GetNRows(), F_Image.GetNCols());
      D_A2_MaskArray = 1. * I_A2_MaskArray;
      F_OutImage.GetPixArray() = D_A2_MaskArray;
      if (!F_OutImage.WriteArray())
      {
        cout << "MExtract::main: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
    }


    /// Set CS_FitsFileName_Out
    cout << "MExtract::main: Starting F_OutImage.setFileName(" << S_A1_FitsFileNames_Out(i_file) << ")" << endl;
    if (!F_OutImage.setFileName(S_A1_FitsFileNames_Out(i_file)))
    {
      cout << "MExtract::main: ERROR: F_OutImage.setFileName(" << S_A1_FitsFileNames_Out(i_file) << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    /// change size of F_OutImage to (NApertures x NRows)
    if (!F_OutImage.SetNCols(F_Image.GetNRows()))
    {
      cout << "MExtract::main: ERROR: F_OutImage.SetNCols(" << F_Image.GetNRows() << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    if (!F_OutImage.SetNRows(I_A1_Apertures.size()))
    {
      cout << "MExtract::main: ERROR: F_OutImage.SetNRows(" << F_Image.Get_NApertures() << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    for (int i_ap=0; i_ap<I_A1_Apertures.size(); i_ap++)
      F_OutImage.GetPixArray()(i_ap, Range::all()) = F_Image.GetSpec()(I_A1_Apertures(i_ap), Range::all());
    F_OutImage.WriteArray();

    blitz::Array<string, 1> S_A1_Args_ExtractFromProfile(2);
    S_A1_Args_ExtractFromProfile = "";
    S_A1_Args_ExtractFromProfile(0) = "APERTURES";
    void **PP_Args_ExtractFromProfile = (void**)malloc(sizeof(void*) * 2);
    PP_Args_ExtractFromProfile[0] = &I_A1_Apertures;

    /// Write EcFromProfileOut 1D
    if (specFromProfileOut.length() > 1)
    {
      cout << "MExtract::main: Starting to write EcFromProfileOut" << endl;
      if (!F_Image.ExtractFromProfile(F_Image.GetPixArray(), S_A1_Args_ExtractFromProfile, PP_Args_ExtractFromProfile))
      {
        cout << "MExtract::main: ERROR: F_Image.ExtractFromProfile() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
      if (!F_OutImage.setFileName(S_A1_SpecFromProfileOut(i_file)))
      {
        cout << "MExtract::main: ERROR: F_OutImage.setFileName() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
      for (int i_ap=0; i_ap<I_A1_Apertures.size(); i_ap++)
        F_OutImage.GetPixArray()(i_ap, Range::all()) = F_Image.GetLastExtracted()(I_A1_Apertures(i_ap), Range::all());//.transpose(secondDim, firstDim);
      if (!F_OutImage.WriteArray())
      {
        cout << "MExtract::main: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
    }

    /// Write SP_Fit 1D
    bool B_WithSky = false;
    if (I_Telluric > 0)
      B_WithSky = true;
    S_A1_Args_ExtractFromProfile(0) = "WITH_SKY";
    PP_Args_ExtractFromProfile[1] = &B_WithSky;
    if (specFitOut.length() > 1)
    {
      if (!F_OutImage.setFileName(S_A1_SpecFitOut(i_file)))
      {
        cout << "MExtract::main: ERROR: F_OutImage.setFileName() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
      cout << "MExtract::main: calculuating SPFit" << endl;
/**      if (I_Telluric < 2){
        if (!F_Image.ExtractSpecFromProfile(F_Image.GetPixArray(), CS_A1_Args_ExtractFromProfile, PP_Args_ExtractFromProfile)){
          cout << "MExtract::main: ERROR: F_Image.ExtractSpecFromProfile(false) returned FALSE!" << endl;
          exit(EXIT_FAILURE);
        }
        for (int i_ap=0; i_ap < P_I_A1_Apertures->size(); i_ap++)
          F_OutImage.GetPixArray()(i_ap, Range::all()) = F_Image.GetLastExtracted()((*P_I_A1_Apertures)(i_ap), Range::all());
      }
      else{*/
        for (int i_ap=0; i_ap < I_A1_Apertures.size(); i_ap++)
          F_OutImage.GetPixArray()(i_ap, Range::all()) = F_Image.GetSpecFit()(I_A1_Apertures(i_ap), Range::all());//.transpose(secondDim, firstDim);
//      }
      cout << "MExtract::main: Starting to write SPFit" << endl;
      if (!F_OutImage.WriteArray())
      {
        cout << "MExtract::main: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
    }

    /// Write ImOut 2D
    if (imageOut.length() > 1){
      if (!F_Image.setFileName(S_A1_ImageOut(i_file)))
      {
        cout << "MExtract::main: ERROR: F_Image.setFileName() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
      cout << "MExtract::main: Starting to write ImOut" << endl;
      F_Image.GetPixArray() = F_Image.GetRecArray();
      if (!F_Image.WriteArray())
      {
        cout << "MExtract::main: ERROR: F_Image.WriteArray() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
    }

    /// Write RecFitOut 2D
    if (imageFitOut.length() > 1){
      if (!F_Image.setFileName(S_A1_ImageFitOut(i_file)))
      {
        cout << "MExtract::main: ERROR: F_Image.setFileName() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
      cout << "MExtract::main: Starting to write RecFitOut" << endl;
      F_Image.GetPixArray() = F_Image.GetRecFitArray();
      if (!F_Image.WriteArray())
      {
        cout << "MExtract::main: ERROR: F_Image.WriteArray() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
    }

    /// Write ProfileOut 2D
    if (profileImageOut.length() > 1){
      if (!F_Image.setFileName(S_A1_ProfileImagesOut(i_file)))
      {
        cout << "MExtract::main: ERROR: F_Image.setFileName() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
      cout << "MExtract::main: Starting to write ProfileOut" << endl;
      F_Image.GetPixArray() = F_Image.GetProfArray();
      if (!F_Image.WriteArray())
      {
        cout << "MExtract::main: ERROR: F_Image.WriteArray() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
    }

    /** Write MaskOut 2D
    if (P_CS_MaskOut->GetLength() > 1){
      if (!F_Image.SetFileName(*P_CS_MaskOut))
    {
      cout << "MExtract::main: ERROR: F_Image.SetFileName() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
    cout << "MExtract::main: Starting to write MaskOut" << endl;
    blitz::Array<int, 2> I_A2_MaskArray(F_Image.GetNRows(), F_Image.GetNCols());
    I_A2_MaskArray = F_Image.GetMaskArray();
    blitz::Array<double, 2> D_A2_MaskArray(F_Image.GetNRows(), F_Image.GetNCols());
    D_A2_MaskArray = 1. * I_A2_MaskArray;
    F_Image.GetPixArray() = D_A2_MaskArray;
    if (!F_Image.WriteArray())
    {
      cout << "MExtract::main: ERROR: F_Image.WriteArray() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
  }**/

  /// Write SkyArrOut 2D
    if (skyImageOut.length() > 1){
      if (!F_Image.setFileName(S_A1_SkyImagesOut(i_file)))
      {
        cout << "MExtract::main: ERROR: F_Image.setFileName() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
      cout << "MExtract::main: Starting to write SkyArrOut" << endl;
      F_Image.GetPixArray() = F_Image.GetRecSkyArray();
      cout << "MExtract::main: F_Image.GetRecSkyArray().rows() = " << F_Image.GetRecSkyArray().rows() << endl;
      cout << "MExtract::main: F_Image.GetRecSkyArray().cols() = " << F_Image.GetRecSkyArray().cols() << endl;
      if (!F_Image.WriteArray())
      {
        cout << "MExtract::main: ERROR: F_Image.WriteArray() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
    }
    
    if (skyFitImageOut.length() > 1){
      if (!F_Image.setFileName(S_A1_SkyFitImagesOut(i_file)))
      {
        cout << "MExtract::main: ERROR: F_Image.setFileName() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
      cout << "MExtract::main: Starting to write SkyArrOut" << endl;
      F_Image.GetPixArray() = F_Image.GetRecSkyFitArray();
      cout << "MExtract::main: F_Image.GetRecSkyFitArray().rows() = " << F_Image.GetRecSkyFitArray().rows() << endl;
      cout << "MExtract::main: F_Image.GetRecSkyFitArray().cols() = " << F_Image.GetRecSkyFitArray().cols() << endl;
      if (!F_Image.WriteArray())
      {
        cout << "MExtract::main: ERROR: F_Image.WriteArray() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
    }

    /// Write ErrOut 2D
    if (errorImageOut.length() > 1){
      cout << "MExtract::main: Writing F_Image.GetErrArray()" << endl;
      if (!F_Image.setFileName(S_A1_ErrOut(i_file)))
      {
        cout << "MExtract::main: ERROR: F_Image.setFileName() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
      cout << "MExtract::main: Starting to write ErrOut" << endl;
      cout << "MExtract::main: F_Image.GetErrArray().rows() = " << F_Image.GetErrArray().rows() << endl;
      cout << "MExtract::main: F_Image.GetErrArray().cols() = " << F_Image.GetErrArray().cols() << endl;
      F_Image.GetPixArray() = F_Image.GetErrArray();
      if (!F_Image.WriteArray())
      {
        cout << "MExtract::main: ERROR: F_Image.WriteArray() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
    }

    /// Write aperture header information
    F_Image.WriteApHead(string("aphead_")+fitsFileName_In+string(".head"));

    // Create ErrOutEc
//  if (P_CS_ErrFromProfileOut->GetLength() > 1){// || P_CS_ErrOutEc->GetLength() > 1){
//    cout << "MExtract::main: Starting F_Image.ExtractErrors()" << endl;
//    if (!F_Image.ExtractErrors())
//    {
//      cout << "MExtract::main: ERROR: F_Image.ExtractErrors() returned FALSE!" << endl;
//      exit(EXIT_FAILURE);
//    }
//    F_OutImage.GetPixArray() = F_Image.GetErrorsEc();
//    F_OutImage.setFileName(*P_CS_ErrOutEc);
//    F_OutImage.WriteArray();
//  }

    /// output extracted spectrum 1D
    cout << "MExtract::main: Starting to write EcOut" << endl;
    for (int i_ap=0; i_ap<I_A1_Apertures.size(); i_ap++)
      F_OutImage.GetPixArray()(i_ap, Range::all()) = F_Image.GetSpec()(I_A1_Apertures(i_ap), Range::all());//.transpose(secondDim, firstDim);

  //cout << "MExctract: F_Image.GetSpec = " << F_Image.GetSpec() << endl;

  // Write Profile Image
/*  if (!F_OutImage.setFileName(CS_FitsFileName_Out))
  {
    cout << "MExtract::main: ERROR: F_OutImage.setFileName(" << CS_FitsFileName_Out << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
}*/
    cout << "MExtract::main: Starting F_OutImage.WriteArray()" << endl;
    if (!F_OutImage.WriteArray())
    {
      cout << "MExtract::main: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    /// Write ErrOutEc 1D
    cout << "MExtract::main: Writing F_Image.GetErrorsEc()" << endl;
    if (errorOutEc.length() > 1)
    {
/*      if (I_Telluric == 1){
        CS_A1_Args_ExtractFromProfile(1) = CString("");
        if (!F_Image.ExtractErrors(CS_A1_Args_ExtractFromProfile, PP_Args_ExtractFromProfile))
        {
          cout << "MExtract::main: ERROR: F_Image.ExtractErrors() returned FALSE!" << endl;
          exit(EXIT_FAILURE);
        }
      }*/
      if (!F_OutImage.setFileName(S_A1_ErrOutEc(i_file)))
      {
        cout << "MExtract::main: ERROR: F_OutImage.setFileName() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
      cout << "MExtract::main: Starting to write ErrOutEc" << endl;
      for (int i_ap=0; i_ap<I_A1_Apertures.size(); i_ap++)
        F_OutImage.GetPixArray()(i_ap, Range::all()) = F_Image.GetErrorsEc()(I_A1_Apertures(i_ap), Range::all());//.transpose(secondDim, firstDim);
      if (!F_OutImage.WriteArray())
      {
        cout << "MExtract::main: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
    }

    /// Write ErrFitOutEc 1D
    cout << "MExtract::main: Writing F_Image.GetErrorsFitEc()" << endl;
    if (errorFitOutEc.length() > 1)
    {
      if (!F_OutImage.setFileName(S_A1_ErrFitOutEc(i_file)))
      {
        cout << "MExtract::main: ERROR: F_OutImage.setFileName() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
      cout << "MExtract::main: Starting to write ErrOutEc" << endl;
      for (int i_ap=0; i_ap<I_A1_Apertures.size(); i_ap++)
        F_OutImage.GetPixArray()(i_ap, Range::all()) = F_Image.GetErrorsEcFit()(I_A1_Apertures(i_ap), Range::all());//.transpose(secondDim, firstDim);
      if (!F_OutImage.WriteArray())
      {
        cout << "MExtract::main: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
    }

    /// Write ErrFromProfile 1D
//  if (P_CS_ErrFromProfileOut->GetLength() > 1)
//  {
//    if (!F_OutImage.setFileName(*P_CS_ErrFromProfileOut))
//    {
//      cout << "MExtract::main: ERROR: F_OutImage.SetFileName() returned FALSE!" << endl;
//      exit(EXIT_FAILURE);
//    }
//    cout << "MExtract::main: Starting to write ErrFromProfile" << endl;
//    F_OutImage.GetPixArray() = F_Image.GetLastExtracted();//.transpose(secondDim, firstDim);
//    if (!F_OutImage.WriteArray())
//    {
//      cout << "MExtract::main: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
//      exit(EXIT_FAILURE);
//    }
//  }

    /// Write SkyOut 1D
    if (I_Telluric > 0 && skyOutEc.length() > 1)
    {
      if (!F_OutImage.setFileName(S_A1_SkyOut(i_file)))
      {
        cout << "MExtract::main: ERROR: F_OutImage.setFileName() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
      cout << "MExtract::main: Starting to write SkyOut" << endl;
      for (int i_ap=0; i_ap < I_A1_Apertures.size(); i_ap++)
        F_OutImage.GetPixArray()(i_ap, Range::all()) = F_Image.GetSky()(I_A1_Apertures(i_ap), Range::all());//.transpose(secondDim, firstDim);
      if (!F_OutImage.WriteArray())
      {
        cout << "MExtract::main: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
    }

    /// Write SkyOut 1D
    if (I_Telluric > 0 && skyFitOutEc.length() > 1)
    {
      if (!F_OutImage.setFileName(S_A1_SkyFitOut(i_file)))
      {
        cout << "MExtract::main: ERROR: F_OutImage.setFileName() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
      cout << "MExtract::main: Starting to write SkyFitOut" << endl;
      for (int i_ap=0; i_ap < I_A1_Apertures.size(); i_ap++)
        F_OutImage.GetPixArray()(i_ap, Range::all()) = F_Image.GetSkyFit()(I_A1_Apertures(i_ap), Range::all());//.transpose(secondDim, firstDim);
      if (!F_OutImage.WriteArray())
      {
        cout << "MExtract::main: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
    }

    /// Write SkyErrOut 1D
    if (I_Telluric > 0 && skyErrorOutEc.length() > 1)
    {
      if (!F_OutImage.setFileName(S_A1_SkyErrorOutEc(i_file)))
      {
        cout << "MExtract::main: ERROR: F_OutImage.setFileName() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
      cout << "MExtract::main: Starting to write SkyErrOut" << endl;
      for (int i_ap=0; i_ap < I_A1_Apertures.size(); i_ap++)
        F_OutImage.GetPixArray()(i_ap, Range::all()) = F_Image.GetSkyError()(I_A1_Apertures(i_ap), Range::all());//.transpose(secondDim, firstDim);
      if (!F_OutImage.WriteArray())
      {
        cout << "MExtract::main: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
    }

    /// Write SkyFitErrOut 1D
    if (I_Telluric > 0 && skyFitErrorOutEc.length() > 1)
    {
      if (!F_OutImage.setFileName(S_A1_SkyFitErrorOutEc(i_file)))
      {
        cout << "MExtract::main: ERROR: F_OutImage.setFileName() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
      cout << "MExtract::main: Starting to write SkyFitErrOut" << endl;
      for (int i_ap=0; i_ap < I_A1_Apertures.size(); i_ap++)
        F_OutImage.GetPixArray()(i_ap, Range::all()) = F_Image.GetSkyFitError()(I_A1_Apertures(i_ap), Range::all());//.transpose(secondDim, firstDim);
      if (!F_OutImage.WriteArray())
      {
        cout << "MExtract::main: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
    }
  }
/*  fitsfile *P_FitsFileIn;
  fitsfile *P_FitsFileOut;
  int bitpixA, bitpixB, hdunumA, hdunumB, nhdusA, nhdusB, hdutypeA, hdutypeB;
  int anynulA, anynulB, extendA, extendB, simpleA, simpleB;
  int naxisA, naxisB;
  long pcountA, pcountB, gcountA, gcountB;
  long naxesA[2], naxesB[2];
  long fpixelA, fpixelB, nelementsA, nelementsB;
  int      countA, countB, Status;
  double *p_ArrayA, *p_ArrayB;
  float nullvalA, nullvalB;
  char strbufA[256], strbufB[256];

  Status=0;
  fits_open_file(&P_FitsFileIn, CS_FitsFileName_In.Get(), READONLY, &Status);
  fits_read_imghdr(P_FitsFileIn, 2, &simpleA , &bitpixA, &naxisA, naxesA,
                   &pcountA, &gcountA, &extendA, &Status);
  if (Status !=0)
  {
    printf("CFits::ReadArray: Error %d opening file %s\n", Status, CS_FitsFileName_In.Get());
    char* P_ErrMsg = new char[255];
    ffgerr(Status, P_ErrMsg);
    cout << "CFits::ReadArray: <" << P_ErrMsg << "> => Returning FALSE" << endl;
    delete[] P_ErrMsg;
    return false;
  }
  cout << "CFits::ReadArray: FitsFileName <" << CS_FitsFileName_In.Get() << "> opened" << endl;
  cout << "CFits::ReadArray: FitsFileName contains <" << naxesA[1]  << "> rows!!!!!!! and <" << naxesA[0] << "> columns!!!!!!!!! naxes = <" << naxesA << ">" << endl;

  fits_open_file(&P_FitsFileOut, CS_FitsFileName_Out.Get(), READWRITE, &Status);
//  fits_read_imghdr(P_FitsFileOut, 2, &simpleB , &bitpixB, &naxisB, naxesB,
//                   &pcountB, &gcountB, &extendB, &Status);
  if (Status !=0)
  {
    printf("CFits::ReadArray: Error %d opening file %s\n", Status, CS_FitsFileName_Out.Get());
    char* P_ErrMsg = new char[255];
    ffgerr(Status, P_ErrMsg);
    cout << "CFits::ReadArray: <" << P_ErrMsg << "> => Returning FALSE" << endl;
    delete[] P_ErrMsg;
    return false;
  }
  cout << "CFits::ReadArray: FitsFileName <" << CS_FitsFileName_Out.Get() << "> opened" << endl;
//  cout << "CFits::ReadArray: FitsFileName contains <" << naxesB[1]  << "> rows!!!!!!! and <" << naxesB[0] << "> columns!!!!!!!!! naxes = <" << naxesB << ">" << endl;
  nhdusB = fits_get_num_hdus(P_FitsFileOut, &hdunumB, &Status);
  if (Status !=0)
  {
    cout << "CFits::ReadArray: Error " << Status << " reading hdunum" << endl;
    char* P_ErrMsg = new char[255];
    ffgerr(Status, P_ErrMsg);
    cout << "CFits::ReadArray: <" << P_ErrMsg << "> => Returning FALSE" << endl;
    delete[] P_ErrMsg;
    return false;
  }
  cout << "MExctract::main: nhdusB = " << nhdusB << ", hdunumB = " << hdunumB << endl;

  fits_get_hdu_type(P_FitsFileOut, &hdutypeB, &Status);
  if (Status !=0)
  {
    cout << "CFits::ReadArray: Error " << Status << " reading hdutype" << endl;
    char* P_ErrMsg = new char[255];
    ffgerr(Status, P_ErrMsg);
    cout << "CFits::ReadArray: <" << P_ErrMsg << "> => Returning FALSE" << endl;
    delete[] P_ErrMsg;
    return false;
  }
  cout << "MExctract::main: hdutypeB = " << hdutypeB << endl;

  fits_movabs_hdu(P_FitsFileOut, 1, &hdutypeB, &Status);
  if (Status !=0)
  {
    cout << "CFits::ReadArray: Error " << Status << " moving to hdunum 1 " << endl;
    char* P_ErrMsg = new char[255];
    ffgerr(Status, P_ErrMsg);
    cout << "CFits::ReadArray: <" << P_ErrMsg << "> => Returning FALSE" << endl;
    delete[] P_ErrMsg;
    return false;
  }
  cout << "MExctract::main: moved to hdunum 1" << endl;

  char card[FLEN_CARD];
  int  nkeysA, nkeysB, ii;
  fits_get_hdrspace(P_FitsFileIn, &nkeysA, NULL, &Status);
  cout << "MExctract::main: nkeysA = " << nkeysA << endl;
  fits_get_hdrspace(P_FitsFileOut, &nkeysB, NULL, &Status);
  cout << "MExctract::main: nkeysB = " << nkeysB << endl;

  for (ii = 1; ii <= nkeysA; ii++)  {
    fits_read_record(P_FitsFileIn, ii, card, &Status);
    cout << card << endl;
    fits_write_record(P_FitsFileOut, card, &Status);
    if (Status !=0)
    {
      cout << "CFits::ReadArray: Error " << Status << " writing header" << endl;
      char* P_ErrMsg = new char[255];
      ffgerr(Status, P_ErrMsg);
      cout << "CFits::ReadArray: <" << P_ErrMsg << "> => Returning FALSE" << endl;
      delete[] P_ErrMsg;
      return false;
    }
  }


/*  fits_copy_hdu(P_FitsFileIn, P_FitsFileOut, &Status);
  if (Status !=0)
  {
    cout << "CFits::ReadArray: Error " << Status << " copying header" << endl;
    char* P_ErrMsg = new char[255];
    ffgerr(Status, P_ErrMsg);
    cout << "CFits::ReadArray: <" << P_ErrMsg << "> => Returning FALSE" << endl;
    delete[] P_ErrMsg;
    return false;
  }
  cout << "MExctract::main: Header copied" << endl;
  /*

  nhdusA = fits_get_num_hdus(P_FitsFileIn, &hdunumA, &Status);
  if (Status !=0)
  {
    cout << "CFits::ReadArray: Error " << Status << " reading hdunum" << endl;
    char* P_ErrMsg = new char[255];
    ffgerr(Status, P_ErrMsg);
    cout << "CFits::ReadArray: <" << P_ErrMsg << "> => Returning FALSE" << endl;
    delete[] P_ErrMsg;
    return false;
  }
  cout << "MExctract::main: nhdusA = " << nhdusA << ", hdunumA = " << hdunumA << endl;

  fits_get_hdu_type(P_FitsFileIn, &hdutypeA, &Status);
  if (Status !=0)
  {
    cout << "CFits::ReadArray: Error " << Status << " reading hdutype" << endl;
    char* P_ErrMsg = new char[255];
    ffgerr(Status, P_ErrMsg);
    cout << "CFits::ReadArray: <" << P_ErrMsg << "> => Returning FALSE" << endl;
    delete[] P_ErrMsg;
    return false;
  }
  cout << "MExctract::main: hdutypeA = " << hdutypeA << endl;

  fits_movabs_hdu(P_FitsFileIn, 1, &hdutypeA, &Status);
  if (Status !=0)
  {
    cout << "CFits::ReadArray: Error " << Status << " moving to hdunum 1 " << endl;
    char* P_ErrMsg = new char[255];
    ffgerr(Status, P_ErrMsg);
    cout << "CFits::ReadArray: <" << P_ErrMsg << "> => Returning FALSE" << endl;
    delete[] P_ErrMsg;
    return false;
  }
  cout << "MExctract::main: moved to hdunum 1" << endl;

  fits_copy_file(P_FitsFileIn, P_FitsFileOut, 0, 1, 1, &Status);
  if (Status !=0)
  {
    cout << "CFits::ReadArray: Error " << Status << " copying header" << endl;
    char* P_ErrMsg = new char[255];
    ffgerr(Status, P_ErrMsg);
    cout << "CFits::ReadArray: <" << P_ErrMsg << "> => Returning FALSE" << endl;
    delete[] P_ErrMsg;
    return false;
  }
  cout << "MExctract::main: Header copied" << endl;
  */
/*  fits_close_file(P_FitsFileIn, &Status);
  if (Status !=0)
  {
    cout << "CFits::ReadArray: Error " << Status << " closing file " << CS_FitsFileName_In.Get() << endl;
    char* P_ErrMsg = new char[255];
    ffgerr(Status, P_ErrMsg);
    cout << "CFits::ReadArray: <" << P_ErrMsg << "> => Returning FALSE" << endl;
    delete[] P_ErrMsg;
    return false;
  }

  fits_close_file(P_FitsFileOut, &Status);
  if (Status !=0)
  {
    cout << "CFits::ReadArray: Error " << Status << " closing file " << CS_FitsFileName_Out.Get() << endl;
    char* P_ErrMsg = new char[255];
    ffgerr(Status, P_ErrMsg);
    cout << "CFits::ReadArray: <" << P_ErrMsg << "> => Returning FALSE" << endl;
    delete[] P_ErrMsg;
    return false;
  }
  */
  return EXIT_SUCCESS;
}
