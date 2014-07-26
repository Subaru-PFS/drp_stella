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
  Array<int, 1> I_A1_Area(4);
  Array<CString, 1> CS_A1_Args(8);
  CS_A1_Args = CString("\0");
  void **PP_Args;
  PP_Args = (void**)malloc(sizeof(void*) * 8);
  CString CS(" ");
  CString CS_comp(" ");
  CString *P_CS = new CString(" ");
  CString *P_CS_ErrIn = new CString(" ");
  CString *P_CS_ErrOut = new CString(" ");
  CString *P_CS_ErrOutEc = new CString(" ");
  CString *P_CS_ErrFitOutEc = new CString(" ");
  CString *P_CS_SkyOut = new CString(" ");
  CString *P_CS_SkyFitOut = new CString(" ");
  CString *P_CS_SkyArrOut = new CString(" ");
  CString *P_CS_SkyFitArrOut = new CString(" ");
  CString *P_CS_SkyErrOut = new CString(" ");
  CString *P_CS_SkyFitErrOut = new CString(" ");
  CString *P_CS_ImOut = new CString(" ");
  CString *P_CS_RecFitOut = new CString(" ");
  CString *P_CS_ProfileOut = new CString(" ");
  CString *P_CS_MaskOut = new CString(" ");
  CString *P_CS_SPFitOut = new CString(" ");
  CString *P_CS_EcFromProfileOut = new CString(" ");
  CString *P_CS_ErrFromProfileOut = new CString(" ");
  CString *P_CS_ApertureListIn = new CString(" ");

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
  Array<int, 1> *P_I_A1_Apertures = new Array<int, 1>(1);
  (*P_I_A1_Apertures) = 0;
  bool B_AperturesSet = false;

  /// read optional parameters
  for (int i = 6; i <= argc; i++){
    CS.Set((char*)argv[i]);
    cout << "MExtract: Reading Parameter " << CS << endl;
    CS_comp.Set("TELLURIC");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtract::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS);
        P_CS = CS.SubString(CS_comp.GetLength()+1);
        I_Telluric = (int)(atoi(P_CS->Get()));
        cout << "MExtract::main: I_Telluric set to " << I_Telluric << endl;
      }
    }

    CS_comp.Set("SWATH_WIDTH");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtract::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS);
        P_CS = CS.SubString(CS_comp.GetLength()+1);
        I_SwathWidth = (int)(atoi(P_CS->Get()));
        cout << "MExtract::main: I_SwathWidth set to " << I_SwathWidth << endl;
      }
    }

    CS_comp.Set("MAX_ITER_SF");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtract::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS);
        P_CS = CS.SubString(CS_comp.GetLength()+1);
        I_MaxIterSF = (int)(atoi(P_CS->Get()));
        cout << "MExtract::main: I_MaxIterSF set to " << I_MaxIterSF << endl;
      }
    }

    CS_comp.Set("MAX_ITER_SKY");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtract::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS);
        P_CS = CS.SubString(CS_comp.GetLength()+1);
        I_MaxIterSky = (int)(atoi(P_CS->Get()));
        cout << "MExtract::main: I_MaxIterSky set to " << I_MaxIterSky << endl;
      }
    }

    CS_comp.Set("MAX_ITER_SIG");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtract::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS);
        P_CS = CS.SubString(CS_comp.GetLength()+1);
        I_MaxIterSig = (int)(atoi(P_CS->Get()));
        cout << "MExtract::main: I_MaxIterSig set to " << I_MaxIterSig << endl;
      }
    }

    CS_comp.Set("SMOOTH_SF");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtract::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS);
        P_CS = CS.SubString(CS_comp.GetLength()+1);
        D_SmoothSF = (double)(atof(P_CS->Get()));
        cout << "MExtract::main: D_SmoothSF set to " << D_SmoothSF << endl;
        CS_A1_Args(3) = CString("LAMBDA_SF");
        PP_Args[3] = &D_SmoothSF;
      }
    }

    CS_comp.Set("SMOOTH_SP");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtract::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS);
        P_CS = CS.SubString(CS_comp.GetLength()+1);
        I_SmoothSP = (int)(atoi(P_CS->Get()));
        cout << "MExtract::main: I_SmoothSP set to " << I_SmoothSP << endl;
        CS_A1_Args(4) = CString("LAMBDA_SP");
        PP_Args[4] = &I_SmoothSP;
      }
    }

    CS_comp.Set("WING_SMOOTH_FACTOR");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtract::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS);
        P_CS = CS.SubString(CS_comp.GetLength()+1);
        D_WingSmoothFactor = (double)(atof(P_CS->Get()));
        cout << "MExtract::main: D_WingSmoothFactor set to " << D_WingSmoothFactor << endl;
        CS_A1_Args(5) = CString("WING_SMOOTH_FACTOR");
        PP_Args[5] = &D_WingSmoothFactor;
      }
    }

    /// AREA
    CS_comp.Set("AREA");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtract::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        CString cs_temp;
        cs_temp.Set(",");
        int i_pos_a = CS_comp.GetLength()+2;
        int i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        cout << "MExtract: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
        delete(P_CS);
        P_CS = CS.SubString(i_pos_a,i_pos_b-1);
        cout << "MExtract: P_CS set to " << *P_CS << endl;
        I_A1_Area(0) = (int)(atoi(P_CS->Get()));
        cout << "MExtract: I_A1_Area(0) set to " << I_A1_Area(0) << endl;

        i_pos_a = i_pos_b+1;
        i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        cout << "MExtract: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
        delete(P_CS);
        P_CS = CS.SubString(i_pos_a,i_pos_b-1);
        cout << "MExtract: P_CS set to " << *P_CS << endl;
        I_A1_Area(1) = (int)(atoi(P_CS->Get()));
        cout << "MExtract: I_A1_Area(1) set to " << I_A1_Area(1) << endl;

        i_pos_a = i_pos_b+1;
        i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        cout << "MExtract: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
        delete(P_CS);
        P_CS = CS.SubString(i_pos_a,i_pos_b-1);
        cout << "MExtract: P_CS set to " << *P_CS << endl;
        I_A1_Area(2) = (int)(atoi(P_CS->Get()));
        cout << "MExtract: I_A1_Area(2) set to " << I_A1_Area(2) << endl;

        i_pos_a = i_pos_b+1;
        cs_temp.Set("]");
        i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        if (i_pos_b < 0){
          cs_temp.Set(")");
          i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        }
        cout << "MExtract: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
        delete(P_CS);
        P_CS = CS.SubString(i_pos_a,i_pos_b-1);
        cout << "MExtract: P_CS set to " << *P_CS << endl;
        I_A1_Area(3) = (int)(atoi(P_CS->Get()));
        cout << "MExtract: I_A1_Area(3) set to " << I_A1_Area(3) << endl;

        CS_A1_Args(2) = CString("AREA");
        PP_Args[2] = &I_A1_Area;
        cout << "MExtract::main: I_A1_Area set to " << I_A1_Area << endl;
      }
    }

    /// 2D
    CS_comp.Set("ERR_IN");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtract::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS_ErrIn);
        P_CS_ErrIn = CS.SubString(CS_comp.GetLength()+1);
        cout << "MExtract::main: ERR_IN set to " << *P_CS_ErrIn << endl;
      }
    }

    /// 2D
    CS_comp.Set("ERR_OUT_2D");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtract::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS_ErrOut);
        P_CS_ErrOut = CS.SubString(CS_comp.GetLength()+1);
        cout << "MExtract::main: ERR_OUT_2D set to " << *P_CS_ErrOut << endl;
      }
    }

    /// 1D
    CS_comp.Set("ERR_OUT_EC");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtract::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS_ErrOutEc);
        P_CS_ErrOutEc = CS.SubString(CS_comp.GetLength()+1);
        cout << "MExtract::main: ERR_OUT_EC set to " << *P_CS_ErrOut << endl;
      }
    }

    CS_comp.Set("ERRFIT_OUT_EC");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtract::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS_ErrFitOutEc);
        P_CS_ErrFitOutEc = CS.SubString(CS_comp.GetLength()+1);
        cout << "MExtract::main: ERR_OUT_EC set to " << *P_CS_ErrOut << endl;
      }
    }

    /// 1D
    CS_comp.Set("SKY_OUT_EC");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtract::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS_SkyOut);
        P_CS_SkyOut = CS.SubString(CS_comp.GetLength()+1);
        cout << "MExtract::main: SKY_OUT_EC set to " << *P_CS_SkyOut << endl;
      }
    }

    CS_comp.Set("SKYFIT_OUT_EC");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtract::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS_SkyFitOut);
        P_CS_SkyFitOut = CS.SubString(CS_comp.GetLength()+1);
        cout << "MExtract::main: SKYFIT_OUT_EC set to " << *P_CS_SkyFitOut << endl;
      }
    }

    /// 2D
    CS_comp.Set("SKY_OUT_2D");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtract::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS_SkyArrOut);
        P_CS_SkyArrOut = CS.SubString(CS_comp.GetLength()+1);
        cout << "MExtract::main: SKY_OUT_2D set to " << *P_CS_SkyArrOut << endl;
      }
    }
    CS_comp.Set("SKYFIT_OUT_2D");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtract::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS_SkyFitArrOut);
        P_CS_SkyFitArrOut = CS.SubString(CS_comp.GetLength()+1);
        cout << "MExtract::main: SKYFIT_OUT_2D set to " << *P_CS_SkyFitArrOut << endl;
      }
    }

    /// 1D
    CS_comp.Set("SKY_ERR_OUT_EC");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtract::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS_SkyErrOut);
        P_CS_SkyErrOut = CS.SubString(CS_comp.GetLength()+1);
        cout << "MExtract::main: SKY_ERR_OUT_EC set to " << *P_CS_SkyErrOut << endl;
      }
    }
    CS_comp.Set("SKYFIT_ERR_OUT_EC");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtract::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS_SkyFitErrOut);
        P_CS_SkyFitErrOut = CS.SubString(CS_comp.GetLength()+1);
        cout << "MExtract::main: SKYFIT_ERR_OUT_EC set to " << *P_CS_SkyFitErrOut << endl;
      }
    }

    /// 2D
    CS_comp.Set("IM_REC_OUT");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtract::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS_ImOut);
        P_CS_ImOut = CS.SubString(CS_comp.GetLength()+1);
        cout << "MExtract::main: IM_REC_OUT set to " << *P_CS_ImOut << endl;
      }
    }

    /// 2D
    CS_comp.Set("REC_FIT_OUT");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtract::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS_RecFitOut);
        P_CS_RecFitOut = CS.SubString(CS_comp.GetLength()+1);
        cout << "MExtract::main: REC_FIT_OUT set to " << *P_CS_RecFitOut << endl;
      }
    }

    /// 2D
    CS_comp.Set("PROFILE_OUT");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtract::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS_ProfileOut);
        P_CS_ProfileOut = CS.SubString(CS_comp.GetLength()+1);
        cout << "MExtract::main: PROFILE_OUT set to " << *P_CS_ProfileOut << endl;
      }
    }

    /// 2D
    CS_comp.Set("MASK_OUT");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtract::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS_MaskOut);
        P_CS_MaskOut = CS.SubString(CS_comp.GetLength()+1);
        cout << "MExtract::main: MASK_OUT set to " << *P_CS_MaskOut << endl;
      }
    }

    /// 1D
    CS_comp.Set("SPFIT_OUT_EC");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtract::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS_SPFitOut);
        P_CS_SPFitOut = CS.SubString(CS_comp.GetLength()+1);
        cout << "MExtract::main: SPFIT_OUT_EC set to " << *P_CS_SPFitOut << endl;
      }
    }

    /// 1D
    CS_comp.Set("EC_FROM_PROFILE_OUT");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtract::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS_EcFromProfileOut);
        P_CS_EcFromProfileOut = CS.SubString(CS_comp.GetLength()+1);
        cout << "MExtract::main: EC_FROM_PROFILE_OUT set to " << *P_CS_EcFromProfileOut << endl;
      }
    }

    /// 1D
    CS_comp.Set("ERR_FROM_PROFILE_OUT");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtract::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS_ErrFromProfileOut);
        P_CS_ErrFromProfileOut = CS.SubString(CS_comp.GetLength()+1);
        cout << "MExtract::main: ERR_FROM_PROFILE_OUT set to " << *P_CS_ErrFromProfileOut << endl;
      }
    }

    CS_comp.Set("APERTURES");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtract::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS_ApertureListIn);
        P_CS_ApertureListIn = CS.SubString(CS_comp.GetLength()+1);
        cout << "MExtract::main: P_CS_ApertureListIn set to " << *P_CS_ApertureListIn << endl;
	B_AperturesSet = true;
	Array<CString, 1> CS_A1_AperturesToExtract(1);
	CS_A1_AperturesToExtract = CString(" ");
	if (!CS.ReadFileLinesToStrArr(*P_CS_ApertureListIn, CS_A1_AperturesToExtract)){
	  cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << *P_CS_ApertureListIn << ") returned FALSE" << endl;
	  exit(EXIT_FAILURE);
	}
	P_I_A1_Apertures->resize(CS_A1_AperturesToExtract.size());
	for (int i_ap=0; i_ap < CS_A1_AperturesToExtract.size(); i_ap++)
	  (*P_I_A1_Apertures)(i_ap) = CS_A1_AperturesToExtract(i_ap).AToI();
        CS_A1_Args(7) = CString("APERTURES");
        PP_Args[7] = P_I_A1_Apertures;
      }
    }

    CS_comp.Set("XCOR_PROF");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MExtract::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        delete(P_CS);
        P_CS = CS.SubString(CS_comp.GetLength()+1);
        I_XCorProf = (int)(atoi(P_CS->Get()));
        cout << "MExtract::main: I_XCorProf set to " << I_XCorProf << endl;
        CS_A1_Args(6) = CString("XCOR_PROF");
        PP_Args[6] = &I_XCorProf;
      }
    }
  }

//  return false;

  if (I_Telluric > 0){
    CS_A1_Args(1).Set("TELLURIC");
    PP_Args[1] = &I_Telluric;
  }
  time_t seconds;
//  if (argc == 8)
//  {
//    I_SwathWidth = (int)(atoi((char*)argv[7]));
//    cout << "MExtract::main: I_SwathWidth set to " << I_SwathWidth << endl;
  if (I_SwathWidth > 0.){
    CS_A1_Args(0).Set("SWATH_WIDTH");
    PP_Args[0] = &I_SwathWidth;
  }
  else
  {
    CS_A1_Args(0) = CString("");
    PP_Args[0] = &I_SwathWidth;
  }

  bool B_Lists = false;
  CString CS_FitsFileName_In;
  CS_FitsFileName_In.Set(P_CharArr_In);
  Array<CString, 1> CS_A1_FitsFileNames_In(1);
  CS_A1_FitsFileNames_In(0) = CS_FitsFileName_In;
  if (CS_FitsFileName_In.IsList()){
    B_Lists = true;
    if (!CS_FitsFileName_In.ReadFileLinesToStrArr(CS_FitsFileName_In, CS_A1_FitsFileNames_In)){
      cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << CS_FitsFileName_In << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
  }

  CString CS_FitsFileName_Out;
  CS_FitsFileName_Out.Set(P_CharArr_Out);
  Array<CString, 1> CS_A1_FitsFileNames_Out(1);
  CS_A1_FitsFileNames_Out(0) = CS_FitsFileName_Out;
  if (B_Lists){
    if (CS_FitsFileName_Out.IsList()){
      if (!CS_FitsFileName_In.ReadFileLinesToStrArr(CS_FitsFileName_Out, CS_A1_FitsFileNames_Out)){
        cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << CS_FitsFileName_Out << ") returned FALSE" << endl;
        exit(EXIT_FAILURE);
      }
    }
    else{
      cout << "MExtract::main: ERROR: InputFileNames is list but OutputFileNames is not" << endl;
      exit(EXIT_FAILURE);
    }
  }

  CString CS_DatabaseFileName_In;
  CS_DatabaseFileName_In.Set(P_CharArr_DB);
  Array<CString, 1> CS_A1_DatabaseFileNames_In(1);
  CS_A1_DatabaseFileNames_In(0) = CS_DatabaseFileName_In;
  if (B_Lists){
    if (CS_DatabaseFileName_In.IsList()){
      if (!CS_FitsFileName_In.ReadFileLinesToStrArr(CS_DatabaseFileName_In, CS_A1_DatabaseFileNames_In)){
        cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << CS_DatabaseFileName_In << ") returned FALSE" << endl;
        exit(EXIT_FAILURE);
      }
    }
    else{
      cout << "MExtract::main: ERROR: InputFileNames is list but DatabaseFileNames is not" << endl;
      exit(EXIT_FAILURE);
    }
  }

  Array<CString, 1> CS_A1_ErrIn(1);
  if (P_CS_ErrIn->GetLength() > 1){
    if (B_Lists){
      if (P_CS_ErrIn->IsList()){
        if (!CS_FitsFileName_In.ReadFileLinesToStrArr(*P_CS_ErrIn, CS_A1_ErrIn)){
          cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << *P_CS_ErrIn << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MExtract::main: ERROR: InputFileNames is list but ErrIn is not" << endl;
      }
    }
    else{
      CS_A1_ErrIn(0).Set(*P_CS_ErrIn);
    }
  }

  Array<CString, 1> CS_A1_ErrOut(1);
  if (P_CS_ErrOut->GetLength() > 1){
    if (B_Lists){
      if (P_CS_ErrOut->IsList()){
        if (!CS_FitsFileName_In.ReadFileLinesToStrArr(*P_CS_ErrOut, CS_A1_ErrOut)){
          cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << *P_CS_ErrOut << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MExtract::main: ERROR: InputFileNames is list but ErrOut is not" << endl;
      }
    }
    else{
      CS_A1_ErrOut(0).Set(*P_CS_ErrOut);
    }
  }

  Array<CString, 1> CS_A1_ErrOutEc(1);
  if (P_CS_ErrOutEc->GetLength() > 1){
    if (B_Lists){
      if (P_CS_ErrOutEc->IsList()){
        if (!CS_FitsFileName_In.ReadFileLinesToStrArr(*P_CS_ErrOutEc, CS_A1_ErrOutEc)){
          cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << *P_CS_ErrOutEc << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MExtract::main: ERROR: InputFileNames is list but ErrOutEc is not" << endl;
      }
    }
    else{
      CS_A1_ErrOutEc(0).Set(*P_CS_ErrOutEc);
    }
  }

  Array<CString, 1> CS_A1_ErrFitOutEc(1);
  if (P_CS_ErrFitOutEc->GetLength() > 1){
    if (B_Lists){
      if (P_CS_ErrFitOutEc->IsList()){
        if (!CS_FitsFileName_In.ReadFileLinesToStrArr(*P_CS_ErrFitOutEc, CS_A1_ErrFitOutEc)){
          cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << *P_CS_ErrFitOutEc << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MExtract::main: ERROR: InputFileNames is list but ErrFitOutEc is not" << endl;
      }
    }
    else{
      CS_A1_ErrFitOutEc(0).Set(*P_CS_ErrFitOutEc);
    }
  }

  Array<CString, 1> CS_A1_SkyOut(1);
  if (P_CS_SkyOut->GetLength() > 1){
    if (B_Lists){
      if (P_CS_SkyOut->IsList()){
        if (!CS_FitsFileName_In.ReadFileLinesToStrArr(*P_CS_SkyOut, CS_A1_SkyOut)){
          cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << *P_CS_SkyOut << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MExtract::main: ERROR: InputFileNames is list but SkyOut is not" << endl;
      }
    }
    else{
      CS_A1_SkyOut(0).Set(*P_CS_SkyOut);
    }
  }

  Array<CString, 1> CS_A1_SkyFitOut(1);
  if (P_CS_SkyFitOut->GetLength() > 1){
    if (B_Lists){
      if (P_CS_SkyFitOut->IsList()){
        if (!CS_FitsFileName_In.ReadFileLinesToStrArr(*P_CS_SkyFitOut, CS_A1_SkyFitOut)){
          cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << *P_CS_SkyFitOut << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MExtract::main: ERROR: InputFileNames is list but SkyFitOut is not" << endl;
      }
    }
    else{
      CS_A1_SkyFitOut(0).Set(*P_CS_SkyFitOut);
    }
  }

  Array<CString, 1> CS_A1_SkyArrOut(1);
  if (P_CS_SkyArrOut->GetLength() > 1){
    if (B_Lists){
      if (P_CS_SkyArrOut->IsList()){
        if (!CS_FitsFileName_In.ReadFileLinesToStrArr(*P_CS_SkyArrOut, CS_A1_SkyArrOut)){
          cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << *P_CS_SkyArrOut << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MExtract::main: ERROR: InputFileNames is list but SkyArrOut is not" << endl;
      }
    }
    else{
      CS_A1_SkyArrOut(0).Set(*P_CS_SkyArrOut);
    }
  }

  Array<CString, 1> CS_A1_SkyFitArrOut(1);
  if (P_CS_SkyFitArrOut->GetLength() > 1){
    if (B_Lists){
      if (P_CS_SkyFitArrOut->IsList()){
        if (!CS_FitsFileName_In.ReadFileLinesToStrArr(*P_CS_SkyFitArrOut, CS_A1_SkyFitArrOut)){
          cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << *P_CS_SkyFitArrOut << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MExtract::main: ERROR: InputFileNames is list but SkyFitArrOut is not" << endl;
      }
    }
    else{
      CS_A1_SkyFitArrOut(0).Set(*P_CS_SkyFitArrOut);
    }
  }

  Array<CString, 1> CS_A1_SkyErrOut(1);
  if (P_CS_SkyErrOut->GetLength() > 1){
    if (B_Lists){
      if (P_CS_SkyErrOut->IsList()){
        if (!CS_FitsFileName_In.ReadFileLinesToStrArr(*P_CS_SkyErrOut, CS_A1_SkyErrOut)){
          cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << *P_CS_SkyErrOut << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MExtract::main: ERROR: InputFileNames is list but SkyErrOut is not" << endl;
      }
    }
    else{
      CS_A1_SkyErrOut(0).Set(*P_CS_SkyErrOut);
    }
  }

  Array<CString, 1> CS_A1_SkyFitErrOut(1);
  if (P_CS_SkyFitErrOut->GetLength() > 1){
    if (B_Lists){
      if (P_CS_SkyFitErrOut->IsList()){
        if (!CS_FitsFileName_In.ReadFileLinesToStrArr(*P_CS_SkyFitErrOut, CS_A1_SkyFitErrOut)){
          cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << *P_CS_SkyFitErrOut << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MExtract::main: ERROR: InputFileNames is list but SkyFitErrOut is not" << endl;
      }
    }
    else{
      CS_A1_SkyFitErrOut(0).Set(*P_CS_SkyFitErrOut);
    }
  }

  Array<CString, 1> CS_A1_ImOut(1);
  if (P_CS_ImOut->GetLength() > 1){
    if (B_Lists){
      if (P_CS_ImOut->IsList()){
        if (!CS_FitsFileName_In.ReadFileLinesToStrArr(*P_CS_ImOut, CS_A1_ImOut)){
          cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << *P_CS_ImOut << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MExtract::main: ERROR: InputFileNames is list but P_CS_ImOut is not" << endl;
      }
    }
    else{
      CS_A1_ImOut(0).Set(*P_CS_ImOut);
    }
  }

  Array<CString, 1> CS_A1_RecFitOut(1);
  if (P_CS_RecFitOut->GetLength() > 1){
    if (B_Lists){
      if (P_CS_RecFitOut->IsList()){
        if (!CS_FitsFileName_In.ReadFileLinesToStrArr(*P_CS_RecFitOut, CS_A1_RecFitOut)){
          cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << *P_CS_RecFitOut << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MExtract::main: ERROR: InputFileNames is list but P_CS_RecFitOut is not" << endl;
      }
    }
    else{
      CS_A1_RecFitOut(0).Set(*P_CS_RecFitOut);
    }
  }

  Array<CString, 1> CS_A1_ProfileOut(1);
  if (P_CS_ProfileOut->GetLength() > 1){
    if (B_Lists){
      if (P_CS_ProfileOut->IsList()){
        if (!CS_FitsFileName_In.ReadFileLinesToStrArr(*P_CS_ProfileOut, CS_A1_ProfileOut)){
          cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << *P_CS_ProfileOut << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MExtract::main: ERROR: InputFileNames is list but P_CS_ProfileOut is not" << endl;
      }
    }
    else{
      CS_A1_ProfileOut(0).Set(*P_CS_ProfileOut);
    }
  }

//  P_CS_MaskOut = CS.SubString(CS_comp.GetLength()+1);
  Array<CString, 1> CS_A1_MaskOut(1);
  if (P_CS_MaskOut->GetLength() > 1){
    if (B_Lists){
      if (P_CS_MaskOut->IsList()){
        if (!CS_FitsFileName_In.ReadFileLinesToStrArr(*P_CS_MaskOut, CS_A1_MaskOut)){
          cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << *P_CS_MaskOut << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MExtract::main: ERROR: InputFileNames is list but P_CS_MaskOut is not" << endl;
      }
    }
    else{
      CS_A1_MaskOut(0).Set(*P_CS_MaskOut);
    }
  }

  Array<CString, 1> CS_A1_SPFitOut(1);
  if (P_CS_SPFitOut->GetLength() > 1){
    if (B_Lists){
      if (P_CS_SPFitOut->IsList()){
        if (!CS_FitsFileName_In.ReadFileLinesToStrArr(*P_CS_SPFitOut, CS_A1_SPFitOut)){
          cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << *P_CS_SPFitOut << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MExtract::main: ERROR: InputFileNames is list but P_CS_SPFitOut is not" << endl;
      }
    }
    else{
      CS_A1_SPFitOut(0).Set(*P_CS_SPFitOut);
    }
  }

  Array<CString, 1> CS_A1_EcFromProfileOut(1);
  if (P_CS_EcFromProfileOut->GetLength() > 1){
    if (B_Lists){
      if (P_CS_EcFromProfileOut->IsList()){
        if (!CS_FitsFileName_In.ReadFileLinesToStrArr(*P_CS_EcFromProfileOut, CS_A1_EcFromProfileOut)){
          cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << *P_CS_EcFromProfileOut << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MExtract::main: ERROR: InputFileNames is list but P_CS_EcFromProfileOut is not" << endl;
      }
    }
    else{
      CS_A1_EcFromProfileOut(0).Set(*P_CS_EcFromProfileOut);
    }
  }

  Array<CString, 1> CS_A1_ErrFromProfileOut(1);
  if (P_CS_ErrFromProfileOut->GetLength() > 1){
    if (B_Lists){
      if (P_CS_ErrFromProfileOut->IsList()){
        if (!CS_FitsFileName_In.ReadFileLinesToStrArr(*P_CS_ErrFromProfileOut, CS_A1_ErrFromProfileOut)){
          cout << "MExtract::main: ERROR: ReadFileLinesToStrArr(" << *P_CS_ErrFromProfileOut << ") returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
      }
      else{
        cout << "MExtract::main: ERROR: InputFileNames is list but P_CS_ErrFromProfileOut is not" << endl;
      }
    }
    else{
      CS_A1_ErrFromProfileOut(0).Set(*P_CS_ErrFromProfileOut);
    }
  }

  CFits F_Image;
  CFits F_OutImage;
  for (int i_file = 0; i_file < CS_A1_FitsFileNames_In.size(); i_file++){
    cout << "MExtract::main: Starting F_Image.SetFileName(" << CS_FitsFileName_In.Get() << ")" << endl;
    if (!F_Image.SetFileName(CS_A1_FitsFileNames_In(i_file)))
    {
      cout << "MExtract::main: ERROR: F_Image.SetFileName(" << CS_FitsFileName_In.Get() << ") returned FALSE!" << endl;
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
    cout << "MExtract::main: Starting F_Image.SetDatabaseFileName(" << CS_A1_DatabaseFileNames_In(i_file) << ")" << endl;
    if (!F_Image.SetDatabaseFileName(CS_A1_DatabaseFileNames_In(i_file)))
    {
      cout << "MExtract::main: ERROR: F_Image.SetDatabaseFileName(" << CS_DatabaseFileName_In << ") returned FALSE!" << endl;
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

//  Array<double, 1> D_A1_YLow(1);
//  D_A1_YLow(0) = -401;
//  F_Image.Set_YLow(D_A1_YLow);
//  D_A1_YLow(0) = 401;
//  F_Image.Set_YHigh(D_A1_YLow);

    cout << "MExtract::main: P_CS_ErrIn = " << *P_CS_ErrIn << ")" << endl;
    if (P_CS_ErrIn->GetLength() > 1){
    /// Set ErrFileName_In
      cout << "MExtract::main: Starting F_Image.SetErrFileName(" << CS_A1_ErrIn(i_file) << ")" << endl;
      if (!F_Image.SetErrFileName(CS_A1_ErrIn(i_file)))
      {
        cout << "MExtract::main: ERROR: F_Image.SetErrFileName(" << CS_A1_ErrIn(i_file) << ") returned FALSE!" << endl;
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
    F_Image.WriteApHead(CString("aphead_")+CS_FitsFileName_In+CString(".head"));

    if (!B_AperturesSet){
      delete(P_I_A1_Apertures);
      P_I_A1_Apertures = F_Image.IndGenArr(F_Image.Get_NApertures());
    }

    /// Calculate Profile Image
    seconds = time(NULL);
    cout << "MExtract::main: Starting F_Image.MkProfIm(): time = " << seconds << endl;

    if (!F_Image.MkProfIm(CS_A1_Args, PP_Args))
    {
      cout << "MExtract::main: ERROR: F_Image.MkProfIm() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
    seconds = time(NULL);
    cout << "MExtract::main: MkProfIm returned true at " << seconds << endl;

    /// Set CS_FitsFileName_In
    cout << "MExtract::main: Starting F_OutImage.SetFileName(" << CS_A1_FitsFileNames_In(i_file) << ")" << endl;
    if (!F_OutImage.SetFileName(CS_A1_FitsFileNames_In(i_file)))
    {
      cout << "MExtract::main: ERROR: F_OutImage.SetFileName(" << CS_A1_FitsFileNames_In(i_file) << ") returned FALSE!" << endl;
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
    if (P_CS_MaskOut->GetLength() > 1){
      if (!F_OutImage.SetFileName(CS_A1_MaskOut(i_file)))
      {
        cout << "MExtract::main: ERROR: F_OutImage.SetFileName() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
      cout << "MExtract::main: Starting to write MaskOut" << endl;
      Array<int, 2> I_A2_MaskArray(F_Image.GetNRows(), F_Image.GetNCols());
      I_A2_MaskArray = F_Image.GetMaskArray();
      Array<double, 2> D_A2_MaskArray(F_Image.GetNRows(), F_Image.GetNCols());
      D_A2_MaskArray = 1. * I_A2_MaskArray;
      F_OutImage.GetPixArray() = D_A2_MaskArray;
      if (!F_OutImage.WriteArray())
      {
        cout << "MExtract::main: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
    }


    /// Set CS_FitsFileName_Out
    cout << "MExtract::main: Starting F_OutImage.SetFileName(" << CS_A1_FitsFileNames_Out(i_file) << ")" << endl;
    if (!F_OutImage.SetFileName(CS_A1_FitsFileNames_Out(i_file)))
    {
      cout << "MExtract::main: ERROR: F_OutImage.SetFileName(" << CS_A1_FitsFileNames_Out(i_file) << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    /// change size of F_OutImage to (NApertures x NRows)
    if (!F_OutImage.SetNCols(F_Image.GetNRows()))
    {
      cout << "MExtract::main: ERROR: F_OutImage.SetNCols(" << F_Image.GetNRows() << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    if (!F_OutImage.SetNRows(P_I_A1_Apertures->size()))
    {
      cout << "MExtract::main: ERROR: F_OutImage.SetNRows(" << F_Image.Get_NApertures() << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    for (int i_ap=0; i_ap<P_I_A1_Apertures->size(); i_ap++)
      F_OutImage.GetPixArray()(i_ap, Range::all()) = F_Image.GetSpec()((*P_I_A1_Apertures)(i_ap), Range::all());
    F_OutImage.WriteArray();

    Array<CString, 1> CS_A1_Args_ExtractFromProfile(2);
    CS_A1_Args_ExtractFromProfile = CString("");
    CS_A1_Args_ExtractFromProfile(0) = CString("APERTURES");
    void **PP_Args_ExtractFromProfile = (void**)malloc(sizeof(void*) * 2);
    PP_Args_ExtractFromProfile[0] = P_I_A1_Apertures;

    /// Write EcFromProfileOut 1D
    if (P_CS_EcFromProfileOut->GetLength() > 1)
    {
      cout << "MExtract::main: Starting to write EcFromProfileOut" << endl;
      if (!F_Image.ExtractFromProfile(F_Image.GetPixArray(), CS_A1_Args_ExtractFromProfile, PP_Args_ExtractFromProfile))
      {
        cout << "MExtract::main: ERROR: F_Image.ExtractFromProfile() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
      if (!F_OutImage.SetFileName(CS_A1_EcFromProfileOut(i_file)))
      {
        cout << "MExtract::main: ERROR: F_OutImage.SetFileName() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
      for (int i_ap=0; i_ap<P_I_A1_Apertures->size(); i_ap++)
        F_OutImage.GetPixArray()(i_ap, Range::all()) = F_Image.GetLastExtracted()((*P_I_A1_Apertures)(i_ap), Range::all());//.transpose(secondDim, firstDim);
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
    CS_A1_Args_ExtractFromProfile(0) = CString("WITH_SKY");
    PP_Args_ExtractFromProfile[1] = &B_WithSky;
    if (P_CS_SPFitOut->GetLength() > 1)
    {
      if (!F_OutImage.SetFileName(CS_A1_SPFitOut(i_file)))
      {
        cout << "MExtract::main: ERROR: F_OutImage.SetFileName() returned FALSE!" << endl;
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
        for (int i_ap=0; i_ap < P_I_A1_Apertures->size(); i_ap++)
          F_OutImage.GetPixArray()(i_ap, Range::all()) = F_Image.GetSpecFit()((*P_I_A1_Apertures)(i_ap), Range::all());//.transpose(secondDim, firstDim);
//      }
      cout << "MExtract::main: Starting to write SPFit" << endl;
      if (!F_OutImage.WriteArray())
      {
        cout << "MExtract::main: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
    }

    /// Write ImOut 2D
    if (P_CS_ImOut->GetLength() > 1){
      if (!F_Image.SetFileName(CS_A1_ImOut(i_file)))
      {
        cout << "MExtract::main: ERROR: F_Image.SetFileName() returned FALSE!" << endl;
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
    if (P_CS_RecFitOut->GetLength() > 1){
      if (!F_Image.SetFileName(CS_A1_RecFitOut(i_file)))
      {
        cout << "MExtract::main: ERROR: F_Image.SetFileName() returned FALSE!" << endl;
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
    if (P_CS_ProfileOut->GetLength() > 1){
      if (!F_Image.SetFileName(CS_A1_ProfileOut(i_file)))
      {
        cout << "MExtract::main: ERROR: F_Image.SetFileName() returned FALSE!" << endl;
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
    Array<int, 2> I_A2_MaskArray(F_Image.GetNRows(), F_Image.GetNCols());
    I_A2_MaskArray = F_Image.GetMaskArray();
    Array<double, 2> D_A2_MaskArray(F_Image.GetNRows(), F_Image.GetNCols());
    D_A2_MaskArray = 1. * I_A2_MaskArray;
    F_Image.GetPixArray() = D_A2_MaskArray;
    if (!F_Image.WriteArray())
    {
      cout << "MExtract::main: ERROR: F_Image.WriteArray() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
  }**/

  /// Write SkyArrOut 2D
    if (P_CS_SkyArrOut->GetLength() > 1){
      if (!F_Image.SetFileName(CS_A1_SkyArrOut(i_file)))
      {
        cout << "MExtract::main: ERROR: F_Image.SetFileName() returned FALSE!" << endl;
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
    if (P_CS_SkyFitArrOut->GetLength() > 1){
      if (!F_Image.SetFileName(CS_A1_SkyFitArrOut(i_file)))
      {
        cout << "MExtract::main: ERROR: F_Image.SetFileName() returned FALSE!" << endl;
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
    if (P_CS_ErrOut->GetLength() > 1){
      cout << "MExtract::main: Writing F_Image.GetErrArray()" << endl;
      if (!F_Image.SetFileName(CS_A1_ErrOut(i_file)))
      {
        cout << "MExtract::main: ERROR: F_Image.SetFileName() returned FALSE!" << endl;
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
    F_Image.WriteApHead(CString("aphead_")+CS_FitsFileName_In+CString(".head"));

    // Create ErrOutEc
//  if (P_CS_ErrFromProfileOut->GetLength() > 1){// || P_CS_ErrOutEc->GetLength() > 1){
//    cout << "MExtract::main: Starting F_Image.ExtractErrors()" << endl;
//    if (!F_Image.ExtractErrors())
//    {
//      cout << "MExtract::main: ERROR: F_Image.ExtractErrors() returned FALSE!" << endl;
//      exit(EXIT_FAILURE);
//    }
//    F_OutImage.GetPixArray() = F_Image.GetErrorsEc();
//    F_OutImage.SetFileName(*P_CS_ErrOutEc);
//    F_OutImage.WriteArray();
//  }

    /// output extracted spectrum 1D
    cout << "MExtract::main: Starting to write EcOut" << endl;
    for (int i_ap=0; i_ap<P_I_A1_Apertures->size(); i_ap++)
      F_OutImage.GetPixArray()(i_ap, Range::all()) = F_Image.GetSpec()((*P_I_A1_Apertures)(i_ap), Range::all());//.transpose(secondDim, firstDim);

  //cout << "MExctract: F_Image.GetSpec = " << F_Image.GetSpec() << endl;

  // Write Profile Image
/*  if (!F_OutImage.SetFileName(CS_FitsFileName_Out))
  {
    cout << "MExtract::main: ERROR: F_OutImage.SetFileName(" << CS_FitsFileName_Out << ") returned FALSE!" << endl;
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
    if (P_CS_ErrOutEc->GetLength() > 1)
    {
/*      if (I_Telluric == 1){
        CS_A1_Args_ExtractFromProfile(1) = CString("");
        if (!F_Image.ExtractErrors(CS_A1_Args_ExtractFromProfile, PP_Args_ExtractFromProfile))
        {
          cout << "MExtract::main: ERROR: F_Image.ExtractErrors() returned FALSE!" << endl;
          exit(EXIT_FAILURE);
        }
      }*/
      if (!F_OutImage.SetFileName(CS_A1_ErrOutEc(i_file)))
      {
        cout << "MExtract::main: ERROR: F_OutImage.SetFileName() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
      cout << "MExtract::main: Starting to write ErrOutEc" << endl;
      for (int i_ap=0; i_ap<P_I_A1_Apertures->size(); i_ap++)
        F_OutImage.GetPixArray()(i_ap, Range::all()) = F_Image.GetErrorsEc()((*P_I_A1_Apertures)(i_ap), Range::all());//.transpose(secondDim, firstDim);
      if (!F_OutImage.WriteArray())
      {
        cout << "MExtract::main: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
    }

    /// Write ErrFitOutEc 1D
    cout << "MExtract::main: Writing F_Image.GetErrorsEc()" << endl;
    if (P_CS_ErrFitOutEc->GetLength() > 1)
    {
      if (!F_OutImage.SetFileName(CS_A1_ErrFitOutEc(i_file)))
      {
        cout << "MExtract::main: ERROR: F_OutImage.SetFileName() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
      cout << "MExtract::main: Starting to write ErrOutEc" << endl;
      for (int i_ap=0; i_ap<P_I_A1_Apertures->size(); i_ap++)
        F_OutImage.GetPixArray()(i_ap, Range::all()) = F_Image.GetErrorsEcFit()((*P_I_A1_Apertures)(i_ap), Range::all());//.transpose(secondDim, firstDim);
      if (!F_OutImage.WriteArray())
      {
        cout << "MExtract::main: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
    }

    /// Write ErrFromProfile 1D
//  if (P_CS_ErrFromProfileOut->GetLength() > 1)
//  {
//    if (!F_OutImage.SetFileName(*P_CS_ErrFromProfileOut))
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
    if (I_Telluric > 0 && P_CS_SkyOut->GetLength() > 1)
    {
      if (!F_OutImage.SetFileName(CS_A1_SkyOut(i_file)))
      {
        cout << "MExtract::main: ERROR: F_OutImage.SetFileName() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
      cout << "MExtract::main: Starting to write SkyOut" << endl;
      for (int i_ap=0; i_ap<P_I_A1_Apertures->size(); i_ap++)
        F_OutImage.GetPixArray()(i_ap, Range::all()) = F_Image.GetSky()((*P_I_A1_Apertures)(i_ap), Range::all());//.transpose(secondDim, firstDim);
      if (!F_OutImage.WriteArray())
      {
        cout << "MExtract::main: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
    }

    /// Write SkyOut 1D
    if (I_Telluric > 0 && P_CS_SkyFitOut->GetLength() > 1)
    {
      if (!F_OutImage.SetFileName(CS_A1_SkyFitOut(i_file)))
      {
        cout << "MExtract::main: ERROR: F_OutImage.SetFileName() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
      cout << "MExtract::main: Starting to write SkyFitOut" << endl;
      for (int i_ap=0; i_ap<P_I_A1_Apertures->size(); i_ap++)
        F_OutImage.GetPixArray()(i_ap, Range::all()) = F_Image.GetSkyFit()((*P_I_A1_Apertures)(i_ap), Range::all());//.transpose(secondDim, firstDim);
      if (!F_OutImage.WriteArray())
      {
        cout << "MExtract::main: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
    }

    /// Write SkyErrOut 1D
    if (I_Telluric > 0 && P_CS_SkyErrOut->GetLength() > 1)
    {
      if (!F_OutImage.SetFileName(CS_A1_SkyErrOut(i_file)))
      {
        cout << "MExtract::main: ERROR: F_OutImage.SetFileName() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
      cout << "MExtract::main: Starting to write SkyErrOut" << endl;
      for (int i_ap=0; i_ap<P_I_A1_Apertures->size(); i_ap++)
        F_OutImage.GetPixArray()(i_ap, Range::all()) = F_Image.GetSkyError()((*P_I_A1_Apertures)(i_ap), Range::all());//.transpose(secondDim, firstDim);
      if (!F_OutImage.WriteArray())
      {
        cout << "MExtract::main: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
    }

    /// Write SkyFitErrOut 1D
    if (I_Telluric > 0 && P_CS_SkyFitErrOut->GetLength() > 1)
    {
      if (!F_OutImage.SetFileName(CS_A1_SkyFitErrOut(i_file)))
      {
        cout << "MExtract::main: ERROR: F_OutImage.SetFileName() returned FALSE!" << endl;
        exit(EXIT_FAILURE);
      }
      cout << "MExtract::main: Starting to write SkyErrOut" << endl;
      for (int i_ap=0; i_ap<P_I_A1_Apertures->size(); i_ap++)
        F_OutImage.GetPixArray()(i_ap, Range::all()) = F_Image.GetSkyFitError()((*P_I_A1_Apertures)(i_ap), Range::all());//.transpose(secondDim, firstDim);
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
  delete(P_CS);
  delete(P_CS_ErrIn);
  delete(P_CS_ErrOut);
  delete(P_CS_SkyOut);
  delete(P_CS_SkyErrOut);
  delete(P_CS_ImOut);
  delete(P_CS_ProfileOut);
  delete(P_CS_ErrOutEc);
  delete(P_CS_SkyArrOut);
  delete(P_CS_MaskOut);
  delete(P_CS_SPFitOut);
  delete(P_CS_EcFromProfileOut);
  delete(P_CS_ErrFromProfileOut);
  delete(P_I_A1_Apertures);
  return EXIT_SUCCESS;
}
