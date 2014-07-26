/*
author: Andreas Ritter
created: 01/12/2007
last edited: 01/12/2007
compiler: g++ 4.0
basis machine: Ubuntu Linux 6.06
*/

///TODO: LogLevel 1-3
///TODO: Change functions which return object constructed with new to return pointer, not object
///TODO: Remove const classifier for functions that return original object
///TODO: Set dimensions of all output arrays before starting the function
///TODO: Add Smooth wings of spatial profile to MkNormFlatProf
///TODO: FIT function to X-correlates in MkSlitFunc, and replace outliers with fitted value
///TODO: FIT function to X-correlates in MkSlitFunc, add to XCenters, re-run without X-correlation
///TODO: Add remaining (new) arrays to equal value and copy and stuff [P_D_A2_WLen, ...]
///TODO: make Gain consistent (* or /)
///TODO: InterPol with B_PreserveFlux can return huge numbers if calculated value outside x-range of input data
#ifndef __CFITS_H__
  #define __CFITS_H__

  #ifdef HAVE_CONFIG_H
    #include <config.h>
  #endif

//  #define __WITH_PLOTS__

  #include <stdio.h>
  #include <iostream>
  #include <cstdlib>
//  #include <c++/4/bits/stdc++.h>
  #include <blitz/array.h>
  #include <blitz/vector.h>
  #include <blitz/vector-et.h>
  #include <blitz/vecwhere.h>
  #include <blitz/tinyvec.h>
  #include <blitz/tinyvec-et.h>
  #include "./cfitsio3-3.230/fitsio.h"
  #include "./cfitsio3-3.230/fitsio2.h"
//  #include "/home/azuri/setup/cfitsio/include/fitsio.h"
  #include <fstream>
//  #include <numerical_recipes/nr.h>
//  #include <numerical_recipes/nrutil.h>
  #include <time.h>
  #include <sys/time.h>

  #ifdef __WITH_PLOTS__
    #include <mgl2/mgl.h>
  #endif

  #include "CAny.h"
  #include "CString.h"
  #include "kriging/geostat.h"
//  #include "../../fit/MyFitUtil.h"
  #include "../../cmpfit-1.2/MyFit.h"

  #ifdef __DEBUG__
    #define __DEBUG_FITS__
  #endif
//  #ifdef __DEBUG_FITS__
//    #define __DEBUG_FITS_APPLYDIMENSION__
//    #define __DEBUG_FITS_BANDSOL__
//    #define __DEBUG_FITS_BOTTOM__
//    #define __DEBUG_FITS_BOXCARFILTER__
//    #define __DEBUG_FITS_CALCCROSSPOINT__
//    #define __DEBUG_FITS_CALCOVERLAPFIG__
//    #define __DEBUG_FITS_CALCSCATTER__
//    #define __DEBUG_FITS_CALCULATELENSLETCORNERS__
//    #define __DEBUG_FITS_CLASSINVARIANT__
//    #define __DEBUG_FITS_CONSTRUCTORS_
//    #define __DEBUG_FITS_COPY__
//    #define __DEBUG_FITS_COUNTCOLS__
//    #define __DEBUG_FITS_COUNTPIXGTZERO__
//    #define __DEBUG_FITS_CROSSCORRELATE__
//    #define __DEBUG_FITS_CURVEFIT__
//    #define __DEBUG_FITS_DISPCOR__
//    #define __DEBUG_FITS_EQUALVALUE__
//    #define __DEBUG_EXTRACT_ERRORS__
//    #define __DEBUG_EXTRACTSPECFROMPROFILE__
//    #define __DEBUG_FITS_EXTRACTFROMPROFILE__
//    #define __DEBUG_FITS_FINDANDTRACE__
//    #define __DEBUG_FITS_FINDNEARESTNEIGHBOUR__
//    #define __DEBUG_FITS_FIT__
//    #define __DEBUG_FITS_GAUSSEXTRACT__
//    #define __DEBUG_FITS_GAUSSFIT__
//    #define __DEBUG_FITS_GET__
//    #define __DEBUG_FITS_GETSUBARRCOPY__
//    #define __DEBUG_FITS_HISTOGRAM__
//    #define __DEBUG_FITS_IDENTIFY__
//    #define __DEBUG_FITS_INTEGRAL__
//    #define __DEBUG_FITS_INTERPOL__
//    #define __DEBUG_FITS_INVERT__
//    #define __DEBUG_FITS_LEGENDRE__
//    #define __DEBUG_FITS_LINFIT__
//    #define __DEBUG_FITS_LSTOFIT__
//    #define __DEBUG_FITS_MARK__
//    #define __DEBUG_FITS_MEDIAN__
//    #define __DEBUG_FITS_MIDDLE__
//    #define __DEBUG_FITS_MINCENMAX__
//    #define __DEBUG_FITS_MKPROFIM__
//    #define __DEBUG_FITS_MKSCATTER__
//    #define __DEBUG_FITS_MKSLITFUNC__
//    #define __DEBUG_FITS_MULT__
//    #define __DEBUG_FITS_PISKUNOV__
//    #define __DEBUG_FITS_PIXELISINTRIANGLE__
//    #define __DEBUG_FITS_POLY__
//    #define __DEBUG_FITS_POLYFIT__
//    #define __DEBUG_FITS_READ__
//    #define __DEBUG_FITS_READAPERTURES__
//    #define __DEBUG_FITS_READARRAY__
//    #define __DEBUG_FITS_READFILELINESTOSTRARR__
//    #define __DEBUG_FITS_REBIN__
//    #define __DEBUG_FITS_REFORM__
//    #define __DEBUG_SEDM__
//    #define __DEBUG_FITS_SELECT__
//    #define __DEBUG_FITS_SET__
//    #define __DEBUG_FITS_SET_NAPERTURES__
//    #define __DEBUG_FITS_SHOW__
//    #define __DEBUG_FITS_SLITFUNC__
//    #define __DEBUG_FITS_SLITFUNC_N__
//    #define __DEBUG_FITS_SLITFUNC_M__
//    #define __DEBUG_FITS_SLITFUNC_SF__
//    #define __DEBUG_FITS_SORT__
//    #define __DEBUG_FITS_SORTCOORDSCCW__
//    #define __DEBUG_FITS_STRETCHANDCROSSCORRELATE__
//    #define __DEBUG_FITS_STRETCHANDCROSSCORRELATESPEC__
//    #define __DEBUG_FITS_STRINGMETHODS__
//    #define __DEBUG_FITS_TELLURIC__
//    #define __DEBUG_FITS_TRACEFUNC__
//    #define __DEBUG_FITS_WRITEAPERTURES__

//  #define __PISKUNOV_ORIG__

  #define DEBUGDIR ""// /home/azuri/entwicklung/idl/REDUCE/16_03_2013/"//stella/ses-pipeline/c/msimulateskysubtraction/data/"//spectra/elaina/eso_archive/red_564/red_r/"
//#endif
  #define MIN(a,b) ((a<b)?a:b)
  #define MAX(a,b) ((a>b)?a:b)

  static double dmaxarg1,dmaxarg2;
  #define DMAX(a,b) (dmaxarg1=(a),dmaxarg2=(b),(dmaxarg1) > (dmaxarg2) ?\
          (dmaxarg1) : (dmaxarg2))

  static double dminarg1,dminarg2;
  #define DMIN(a,b) (dminarg1=(a),dminarg2=(b),(dminarg1) < (dminarg2) ?\
          (dminarg1) : (dminarg2))

  static float maxarg1,maxarg2;
  #define FMAX(a,b) (maxarg1=(a),maxarg2=(b),(maxarg1) > (maxarg2) ?\
          (maxarg1) : (maxarg2))

  static float minarg1,minarg2;
  #define FMIN(a,b) (minarg1=(a),minarg2=(b),(minarg1) < (minarg2) ?\
          (minarg1) : (minarg2))

  static long lmaxarg1,lmaxarg2;
  #define LMAX(a,b) (lmaxarg1=(a),lmaxarg2=(b),(lmaxarg1) > (lmaxarg2) ?\
          (lmaxarg1) : (lmaxarg2))

  static long lminarg1,lminarg2;
  #define LMIN(a,b) (lminarg1=(a),lminarg2=(b),(lminarg1) < (lminarg2) ?\
          (lminarg1) : (lminarg2))

  static int imaxarg1,imaxarg2;
  #define IMAX(a,b) (imaxarg1=(a),imaxarg2=(b),(imaxarg1) > (imaxarg2) ?\
          (imaxarg1) : (imaxarg2))

  static int iminarg1,iminarg2;
  #define IMIN(a,b) (iminarg1=(a),iminarg2=(b),(iminarg1) < (iminarg2) ?\
          (iminarg1) : (iminarg2))

  #define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
  #define SPEED_OF_LIGHT 299792458
  #define D_PI 3.14159265359
  #define D_H_C 1.9865e-8 /// erg Ang
//#define __TELLURIC_ORIG__
//#define __TELLURIC_MINE__
//  #define MAXAPERTURES 1000
//  #define SWAP(a,b) double
  using namespace std;
  using namespace blitz;

  class CFits: public CAny
  {
    private:
      bool DoCopy(const CFits &CF_ToCopy);

    protected:
      /// File name of fits file to reduce
      CString *P_CS_FileName;

      /// File name of database entry
      CString *P_CS_DatabaseFileName;

      /// File name of error fits file to propagate
      CString *P_CS_ErrFileName;

      /// File name of log file to write debug informations into
      CString *P_CS_LogFileName;

      /// OFStream of log file
      ofstream *P_OFS_Log;

      /// Instance of fitsfile
      fitsfile *P_FitsFile;

      /// Pixel array
      Array<double, 2> *P_D_A2_PixArray;/// NRows x NCols

      /// Slit profile array
      Array<double, 2> *P_D_A2_ProfArray;/// NRows x NCols

      /// Uncertainty per pixel array
      Array<double, 2> *P_D_A2_ErrArray;/// NRows x NCols

      /// Reconstructed Sky array
      Array<double, 2> *P_D_A2_RecSkyArray;/// NRows x NCols
      Array<double, 2> *P_D_A2_RecSkyFitArray;/// NRows x NCols

      /// Reconstructed array
      Array<double, 2> *P_D_A2_RecArray;/// NRows x NCols

      /// Reconstructed array from Fit
      Array<double, 2> *P_D_A2_RecFitArray;/// NRows x NCols

      /// Mask array
      Array<int, 2> *P_I_A2_MaskArray;/// NRows x NCols

      /// Coefficients for the polynomial trace functions
      Array<double, 2> *P_D_A2_Coeffs;  /// NApertures x max(P_I_A1_NCoeffs)

      /// Centres of the apertures in x(cols)
      Array<double, 2> *P_D_A2_XCenters;/// NApertures x NRows

      /// Blaze functions for every aperture
      Array<double, 2> *P_D_A2_Blaze;   /// NApertures x NRows

      /// extracted spectrum from fit
      Array<double, 2> *P_D_A2_SP_Fit;   /// NApertures x NRows

      /// errors for every aperture calculated from profile
      Array<double, 2> *P_D_A2_Errors_Ec;   /// NApertures x NRows

      /// errors for every aperture calculated from fit
      Array<double, 2> *P_D_A2_Errors_EcFit;   /// NApertures x NRows

      /// errors for every aperture
      Array<double, 2> *P_D_A2_LastExtracted;   /// NApertures x NRows

      /// sky for every aperture
      Array<double, 2> *P_D_A2_Sky;   /// NApertures x NRows
      Array<double, 2> *P_D_A2_SkyFit;   /// NApertures x NRows

      /// sky error for every aperture
      Array<double, 2> *P_D_A2_SkyError;   /// NApertures x NRows
      Array<double, 2> *P_D_A2_SkyFitError;   /// NApertures x NRows

      /// Wavelength for every aperture NOTE: use only when P_D_A2_PixArray is in the format (aperture, row)
      Array<double, 2> *P_D_A2_WLen; /// NRows x NCol

      /// lower aperture limit for every aperture relative to center including sky (< 0.), cols
      Array<double, 1> *P_D_A1_XLow;     /// NApertures

      /// higher aperture limit for every aperture relative to center including sky (> 0.), cols
      Array<double, 1> *P_D_A1_XHigh;    /// NApertures

      /// lower aperture limit for every aperture (< 0.), rows
      Array<double, 1> *P_D_A1_YLow;     /// NApertures

      /// higher aperture limit for every aperture (> 0.), rows
      Array<double, 1> *P_D_A1_YHigh;    /// NApertures

      /// center position in x (column) for every aperture
      Array<double, 1> *P_D_A1_XCenter; /// NApertures

      /// center position in y (row) for every aperture
      Array<double, 1> *P_D_A1_YCenter; /// NApertures

      /// aperture lower limit relative to center for determining the background
      Array<double, 1> *P_D_A1_XMin;    /// NApertures

      /// aperture upper limit relative to center for determining the background
      Array<double, 1> *P_D_A1_XMax;    /// NApertures

      /// polynomial order of aperture trace function
      Array<int, 1>    *P_I_A1_Orders;  /// NApertures

      /// number of coefficients for polynomial trace function
      Array<int, 1>    *P_I_A1_NCoeffs; /// NApertures

      /// Name of function used to trace the apertures
      ///  0: chebyshev
      ///  1: legendre
      ///  2: cubic
      ///  3: linear
      ///  4: polynomial
      Array<CString, 1> *P_CS_A1_Functions; /// NApertures
      int NRows;
      int NCols;
      int I_DispAxis;
      int I_NApertures;
      int I_OverSample;
      int I_MaxIterSF;
      int I_MaxIterSky;
      int I_MaxIterSig;

      /// CCD parameters
      double D_Gain;
      double D_ReadOutNoise;

      /// Parameters for finding and tracing apertures
      double D_SaturationLevel;
      double D_ApertureFWHM;
      double D_SignalThreshold;
      int I_MaxNumberOfAperturesToBeFound;

      bool ArrayInitialized;
      bool TraceFunctionsCalculated;
      bool ProfileCalculated;
      bool ErrorsRead;

      bool ApplyDimension();

    public:
     /** Constructors:
         =============

     Standard constructor
     Constructs an object of 'CFits' and sets the values of the attributes
     to current date.**/
      CFits();

      /** Constructs an object of type 'CFits' and sets this objects
      <FileName> to <fn>.**/
      CFits(const CString &fn, int nrows, int ncols);

      /** Copy constructor
          Constructs an object of 'CFits' with the values of 'fit', if 'fit' is
         'ClassInvariant()', else to "".**/
      CFits(const CFits &fit);

      /** Destructor:
          ===========
          Destructs the object.**/
      virtual ~CFits();

      /** Methods:
          ========

         task   : Proves the invariance.
         require: none
         ensure : Returns "TRUE", if the invariance is valid (1900<'Year'<2200,
                  'Month', 'Day', 'Hour', 'Minute' and 'Second' valid),
                  else returns "FALSE".**/
      virtual bool ClassInvariant() const;

      /** task   : Compares the object's attributes.
          require: Both objects are 'ClassInvariant()'.
          ensure : Returns "TRUE", if the objects' attributes are equal,
                   else returns "FALSE".**/
      virtual bool EqualValue(const CAny &any) const;

      /** task   : Copies the object 'any', if it is possible.
          require: Objects are not 'Equal()' and 'any' is a 'ClassInvariant()'
                   instance of 'CFits'.
          ensure : Returns "FALSE", if 'any' is not 'ClassInvariant()' or
                   the objects are 'Equal()', else returns "TRUE".**/
      virtual bool Copy(const CAny &any);

      bool ForceCopy(const CFits &CF_ToCopy);

      /** Operators:
          ==========

          task   : Like 'Copy' but 'dat' doesn't have to be 'ClassInvariant()'
          require: none
          ensure : works**/
      CFits& operator=(CFits &fit);

      /**
         subtract PixArray from one CFits instance from this one
       **/
      CFits& operator-=(CFits &fit);

      /**
      subtract D_Sub from this->P_D_A2_PixArray
      **/
      CFits& operator-=(double D_Sub);

      /**
      subtract D_A2_Sub from this->P_D_A2_PixArray
      **/
      CFits& operator-=(const Array<double, 2> &D_A2_Sub);

      /**
        add PixArray from one CFits instance to this one
      **/
      CFits& operator+=(CFits &fit);

      /**
        multiply PixArray from one CFits instance to this one
      **/
      CFits& operator*=(CFits &fit);

      /**
        divide this->P_D_A2_PixArray by PixArray from another CFits instance
      **/
      CFits& operator/=(CFits &fit);


      /** Set methods
          ===========
          task   : Set the specified value.
          require: Values are valid.
          ensure : Return "TRUE", if the specified values are valid, else set
          the value of the specified attribute to the nearest valid one and
          return "FALSE".**/
      bool SetFileName(const CString &fn);
      bool SetErrFileName(const CString &fn);
      bool SetDatabaseFileName(const CString &fn);
      bool SetNRows(int nro);
      bool SetNCols(int nco);
      bool Set_DispAxis(int I_DispAxis_In);
      bool Set_NApertures(int I_NApertures_In);
      bool Set_XHigh(const Array<double, 1>& D_A1_In);
      bool Set_XLow(const Array<double, 1>& D_A1_In);
      bool Set_YHigh(const Array<double, 1>& D_A1_In);
      bool Set_YLow(const Array<double, 1>& D_A1_In);
      bool Set_XMax(const Array<double, 1>& D_A1_In);
      bool Set_XMin(const Array<double, 1>& D_A1_In);
      bool Set_Coeffs(const Array<double, 2>& D_A2_In);
      bool Set_XCenters(const Array<double, 2>& D_A2_In);
      bool Set_XCenter(const Array<double, 1>& D_A1_In);
      bool Set_YCenter(const Array<double, 1>& D_A1_In);
      bool Set_Orders(const Array<int, 1>& I_A1_In);
      bool Set_NCoeffs(const Array<int, 1>& I_A1_In);
      bool Set_Functions(const Array<CString, 1> CS_A1_Functions_In);
      bool Set_SubArray(Array<double, 1> &D_A1_InOut, Array<int, 1> &I_A1_Indices_In, Array<double, 1> &D_A1_In) const;
      bool Set_SubArray(Array<int, 1> &I_A1_InOut, Array<int, 1> &I_A1_Indices_In, Array<int, 1> &I_A1_In) const;
      bool Set_ProfArray(const Array<double, 2> &D_A2_ProfArray_In);
      bool Read_ProfArray(const CString &CS_FitsFileName_In);

      /// I_A3_Indices_In: (*,0): row_InOut
      /// I_A3_Indices_In: (*,1): col_InOut
      /// I_A3_Indices_In: (*,2): row_In
      /// I_A3_Indices_In: (*,3): col_In
      bool Set_SubArray(Array<double, 2> &D_A2_InOut, Array<int, 2> &I_A2_Indices_In, Array<double, 2> &D_A2_In) const;
      bool Set_SubArray(Array<int, 2> &I_A2_InOut, Array<int, 2> &I_A2_Indices_In, Array<int, 2> &I_A2_In) const;

      /// I_A3_Indices_In: (*,0): row_InOut
      /// I_A3_Indices_In: (*,1): col_InOut
      bool Set_SubArray(Array<double, 2> &D_A2_InOut, Array<int, 2> &I_A2_Indices_In, Array<double, 1> &D_A1_In) const;
      bool Set_SubArray(Array<int, 2> &I_A2_InOut, Array<int, 2> &I_A2_Indices_In, Array<int, 1> &I_A1_In) const;

      bool Set_Gain(double D_Gain_In);
      bool Set_ReadOutNoise(double D_ReadOutNoise_In);
      bool Set_SaturationLevel(double D_SaturationLevel_In);
      bool Set_ApertureFWHM(double D_ApertureFWHM_In);
      bool Set_SignalThreshold(double D_SignalThreshold_In);
      bool Set_MaxNumberOfAperturesToBeFound(int I_MaxNumberOfAperturesToBeFound_In);
      bool Set_OverSample(int I_OverSample_In);
      bool Set_MaxIterSF(int I_MaxIterSF_In);
      bool Set_MaxIterSky(int I_MaxIterSky_In);
      bool Set_MaxIterSig(int I_MaxIterSig_In);
      /**
      SetZerosToUnity()
       **/
      bool SetZerosToUnity();

      /// <0: move aperture centers to the left
      /// >0: move aperture centers to the right
      bool Set_ApCenterOffset(double D_Offset);

      /** Get methods
          ===========
          task   : Return the specified value.
          require: none
          ensure : work**/
      CString& GetFileName() const;
      CString& GetErrFileName() const;
      CString& GetDatabaseFileName() const;
      int GetNCols() const;
      int GetNRows() const;
      Array<double, 2>& GetPixArray(); /// NRows x NCols
      Array<double, 2>& GetRecArray(); /// NRows x NCols
      Array<double, 2>& GetRecFitArray(); /// NRows x NCols
      Array<double, 2>& GetProfArray();/// NRows x NCols
      Array<double, 2>& GetErrArray(); /// NRows x NCols
      Array<double, 2>& GetRecSkyArray(); /// NRows x NCols
      Array<int, 2>& GetMaskArray(); /// NRows x NCols
      Array<double, 2>& GetSpec();     /// NApertures x NRows
      Array<double, 2>& GetSpecFit();  /// NApertures x NRows
      Array<double, 2>& GetSky();      /// NApertures x NRows
      Array<double, 2>& GetSkyError(); /// NApertures x NRows
      Array<double, 2>& GetSkyFitError();
      Array<double, 2>& GetSkyFit();
      Array<double, 2>& GetErrorsEcFit();
      Array<double, 2>& GetRecSkyFitArray();
      Array<double, 2>& GetLastExtracted(); /// NApertures x NRows
      Array<double, 2>& GetErrorsEc(); /// NApertures x NRows
      Array<double, 2>& GetWavelengths(); /// NApertures x NRows
      int Get_DispAxis() const;
      int Get_NApertures() const;
      Array<double, 1>* Get_XHigh() const;    /// NApertures (Upper limit of aperture relative to center including sky)
      Array<double, 1>* Get_XLow() const;     /// NApertures (Upper limit of aperture relative to including sky)
      Array<double, 1>* Get_YHigh() const;    /// NApertures
      Array<double, 1>* Get_YLow() const;     /// NApertures
      Array<double, 1>* Get_XMax() const;    /// NApertures (Upper limit of aperture relative to center excluding sky)
      Array<double, 1>* Get_XMin() const;    /// NApertures (Upper limit of aperture relative to center excluding sky)
      Array<double, 2>* Get_Coeffs() const;  /// NApertures x max(NCoeffs)
      Array<double, 2>* Get_XCenters() const;/// NApertures x NRows
      Array<double, 1>* Get_XCenter() const; /// NApertures
      Array<double, 1>* Get_YCenter() const; /// NApertures
      Array<int, 1>* Get_Orders() const;     /// NApertures
      Array<int, 1>* Get_NCoeffs() const;    /// NApertures
      Array<CString, 1>* Get_Functions() const;         /// NApertures
      double Get_Gain() const;
      double Get_ReadOutNoise() const;
      double Get_SaturationLevel() const;
      double Get_ApertureFWHM() const;
      double Get_SignalThreshold() const;
      int Get_MaxNumberOfAperturesToBeFound() const;
      int Get_OverSample() const;
      int Get_MaxIterSF() const;
      int Get_MaxIterSky() const;
      int Get_MaxIterSig() const;
//      bool Get_Path(CString &CS_Path_Out) const;

      /**
       function int CountLines(const CString &fnc: inout)
       Returns number of lines of file <fnc>.
      **/
      long CountLines(const CString &fnc) const;

      /**
       function int CountDataLines(const CString &fnc: inout)
       Returns number of lines which do not start with '#' of file <fnc>.
       **/
      long CountDataLines(const CString &fnc) const;

      /**
      function int CountCols(const CString &fnc: inout)
      Returns number of columns of file <fnc>.
       **/
      long CountCols(const CString &CS_FileName_In, const CString &CS_Delimiter) const;

      double GetNormalized(const double XVal,
                           const double XMin,
                           const double XMax) const;
      double GetA(const double XVal,
                  const double XMin,
                  const double XMax,
                  const int Order) const;
      double GetB(const double XVal,
                  const double XMin,
                  const double XMax,
                  const int Order) const;
      long GetJ(const double XVal,
                const double XMin,
                const double XMax,
                const int Order) const;
      double GetS(const double XVal,
                  const double XMin,
                  const double XMax,
                  const int Order) const;

      bool LinearSpline(Array<double, 1> &D_A1_XCenters_Out,
                        const Array<double, 1> &D_A1_Coeffs_In,
                        double D_XCenter_In,
                        double D_YCenter_In,
                        double D_YMin_In,
                        double D_YMax_In,
                        double D_Low_In,
                        double D_High_In,
                        int I_Order_In,
                        int I_NCols_In);

      bool CubicSpline(Array<double, 1> &D_A1_XCenters_Out,
                       const Array<double, 1> &D_A1_Coeffs_In,
                       double D_XCenter_In,
                       double D_YCenter_In,
                       double D_YMin_In,
                       double D_YMax_In,
                       double D_XLow_In,
                       double D_XHigh_In,
                       int I_Order_In,
                       int I_NCols_In);

//      bool ChebyLegend(double D_Normalised_In, double )

      bool ChebyLegend(Array<double, 1> &D_A1_XCenters_Out,
                       double *P_D_YMin_Out,
                       double *P_D_YMax_Out,
                       const Array<double, 1> &D_A1_Coeffs_In,
                       double D_XCenter_In,
                       double D_YCenter_In,
                       double D_YMin_In,
                       double D_YMax_In,
                       double D_Low_In,
                       double D_High_In,
                       int I_Order_In,
                       int I_NCols_In,
                       const CString &CS_Function_In);

      bool Legendre(Array<double, 1> &D_A1_XCenters_Out,
                    double *P_D_YMin_Out,
                    double *P_D_YMax_Out,
                    const Array<double, 1> &D_A1_Coeffs_In,
                    double D_XCenter_In,
                    double D_YCenter_In,
                    double D_YMin_In,
                    double D_YMax_In,
                    double D_XLow_In,
                    double D_XHigh_In,
                    int I_Order_In,
                    int I_NCols_In);

      bool Chebyshev(Array<double, 1> &D_A1_XCenters_Out,
                     double *P_D_YMin_Out,
                     double *P_D_YMax_Out,
                     const Array<double, 1> &D_A1_Coeffs_In,
                     double D_XCenter_In,
                     double D_YCenter_In,
                     double D_YMin_In,
                     double D_YMax_In,
                     double D_XLow_In,
                     double D_XHigh_In,
                     int I_Order_In,
                     int I_NCols_In);

      bool ShiftApertures(double D_Shift_In);

//      bool FitGaussianArea(const Array<double, 1> &D_A1_X_In,
//                           const Array<double, 1> &D_A1_Y_In,
//                           const Array<double, 1> &D_A1_SigY_In,
//                           const Array<double, 1> &D_A1_Guess_In,//with constant background: (background, mean, sigma, area)
//                                                                 //without background: (mean, sigma, area)
//                           bool B_WithBackground,
//                           Array<double, 1> &D_A1_Coeffs_Out) const;

      bool Fit2DGaussianCB(const Array<double, 1> &D_A1_X_In,
                           const Array<double, 1> &D_A1_Y_In,
                           const Array<double, 1> &D_A1_SigY_In,
                           const Array<double, 1> &D_A1_Guess_In,//(background, peak, mean_x, mean_y, sigma)
                           const Array<double, 2> &D_A2_Limits_In,//(background, peak, mean_x, mean_y, sigma)
                           Array<double, 1> &D_A1_Coeffs_Out) const;

      /**
       Fit Gaussian to spatial profile (NTerms=3)
       Spatial profile must be at least 4 pixels wide
       Pre: Set_SaturationLevel(double)
            Set_ApertureFWHM(double)
            Set_SignalThreshold(double)
            ReadArray()
            Apertures more or less parallel to columns
       **
      bool FindAndTraceApertures();

      /** Set I_NTermsGaussFit to
       1 to look for maximum only without GaussFit
       3 to fit Gaussian
       4 to fit Gaussian plus constant (sky)
         Spatial profile must be at least 5 pixels wide
       5 to fit Gaussian plus linear term (sloped sky)
         Spatial profile must be at least 6 pixels wide
       **/
      bool FindAndTraceApertures(int I_NTermsGaussFit,
                                 int I_In_PolyFitOrder,
                                 int I_In_MinLength,
                                 int I_In_MaxLength,
                                 int I_In_NLost);

      bool CalcTraceFunction(int I_Aperture_In);

      bool CalcTraceFunctions();

      bool ExtractErrors();
      bool ExtractErrors(const Array<CString, 1> &CS_A1_Args_In,
                         void* PP_Args_In[]);
      /// Args:
      /// APERTURES: Array<int, 1>

      bool ExtractSimpleSum();
      bool ExtractSimpleSum(const Array<CString, 1> &CS_A1_Args_In,       ///: in
                            void *ArgV_In[]);
      /// Args:
      /// AREA: Array<int, 1>(4)
      ///    I_XMin = I_A1_Area(0);
      ///    I_XMax = I_A1_Area(1);
      ///    I_YMin = I_A1_Area(2);
      ///    I_YMax = I_A1_Area(3);
      ///             All apertures with ((*(this->P_D_A1_XCenter))(iap) >= I_XMin)
      ///                                && ((*(this->P_D_A1_XCenter))(iap) <= I_XMax)
      ///                                && ((((*(this->P_D_A1_YCenter))(iap)+(*(this->P_D_A1_YLow))(iap) >= I_YMin)
      ///                                    && ((*(this->P_D_A1_YCenter))(iap)+(*(this->P_D_A1_YLow))(iap) <= I_YMax))
      ///                                   || (((*(this->P_D_A1_YCenter))(iap)+(*(this->P_D_A1_YHigh))(iap) >= I_YMin)
      ///                                    && ((*(this->P_D_A1_YCenter))(iap)+(*(this->P_D_A1_YHigh))(iap) <= I_YMax))) get extracted
      /// APERTURES: Array<int, 1>

      bool ExtractSimpleSum(const Array<double, 2> &D_A2_ArrayToExtract_In,
                            const Array<CString, 1> &CS_A1_Args_In,       ///: in
                            void *ArgV_In[]);
      /// Args:
      /// AREA: Array<int, 1>(4)
      ///    I_XMin = I_A1_Area(0);
      ///    I_XMax = I_A1_Area(1);
      ///    I_YMin = I_A1_Area(2);
      ///    I_YMax = I_A1_Area(3);
      /// APERTURES: Array<int, 1>

      bool GaussExtract(const Array<double, 2> &D_A2_ArrayToExtract_In,
                        bool B_WithBackground,
                        const double D_ApertureOffset);

      /// D_A1_SDevLimits_In(0) = minimum standard deviation of Gauss curve
      /// D_A1_SDevLimits_In(1) = maximum standard deviation of Gauss curve
      /// D_A1_MeanLimits_In(0) = maximum difference to the left of aperture center
      /// D_A1_MeanLimits_In(1) = maximum difference to the right of aperture center
      bool MPFitGaussExtract(const Array<double, 2> &D_A2_ArrayToExtract_In,
                             const Array<double, 1> &D_A1_SDevLimits_In,
                             const Array<double, 1> &D_A1_MeanLimits_In,
                             const bool B_WithBackground_In);

      /// D_A1_SDevLimits_In(0) = minimum standard deviation of Gauss curve
      /// D_A1_SDevLimits_In(1) = maximum standard deviation of Gauss curve
      /// D_A1_MeanLimits_In(0) = maximum difference to the left of aperture center
      /// D_A1_MeanLimits_In(1) = maximum difference to the right of aperture center
      bool MPFitTwoGaussExtract(const Array<double, 2> &D_A2_ArrayToExtract_In,
                                const Array<double, 1> &D_A1_SDevLimits_In,
                                const Array<double, 1> &D_A1_MeanLimits_In,
                                const bool B_WithBackground_In);

      /// D_A1_SDevLimits_In(0) = minimum standard deviation of Gauss curve
      /// D_A1_SDevLimits_In(1) = maximum standard deviation of Gauss curve
      /// D_A1_MeanLimits_In(0) = maximum difference to the left of aperture center
      /// D_A1_MeanLimits_In(1) = maximum difference to the right of aperture center
      bool MPFitTwoGaussExtract(const Array<double, 2> &D_A2_ArrayToExtract_In,
                                const Array<double, 1> &D_A1_SDevLimits_In,
                                const Array<double, 1> &D_A1_MeanLimits_In,
                                const bool B_WithBackground_In,
                                const Array<CString, 1> &CS_A1_Args_In,
                                void *ArgV_In[]);

      /// D_A1_SDevLimits_In(0) = minimum standard deviation of Gauss curve
      /// D_A1_SDevLimits_In(1) = maximum standard deviation of Gauss curve
      /// D_A1_MeanLimits_In(0) = maximum difference to the left of aperture center
      /// D_A1_MeanLimits_In(1) = maximum difference to the right of aperture center
      bool MPFitThreeGaussExtract(const Array<double, 2> &D_A2_ArrayToExtract_In,
                                  const Array<double, 1> &D_A1_SDevLimits_In,
                                  const Array<double, 1> &D_A1_MeanLimits_In,
                                  const bool B_WithBackground_In);

      /// D_A1_SDevLimits_In(0) = minimum standard deviation of Gauss curve
      /// D_A1_SDevLimits_In(1) = maximum standard deviation of Gauss curve
      /// D_A1_MeanLimits_In(0) = maximum difference to the left of aperture center
      /// D_A1_MeanLimits_In(1) = maximum difference to the right of aperture center
      bool MPFitThreeGaussExtract(const Array<double, 2> &D_A2_ArrayToExtract_In,
                                  const Array<double, 1> &D_A1_SDevLimits_In,
                                  const Array<double, 1> &D_A1_MeanLimits_In,
                                  const bool B_WithBackground_In,
                                  const Array<CString, 1> &CS_A1_Args_In,
                                  void *ArgV_In[]);

      /// No fitting, simply apply weights from Profile to spectrum and sum up (no sky!!!)
      bool ExtractFromProfile(const Array<double, 2> &D_A2_ArrayToExtract);
      bool ExtractFromProfile(const Array<double, 2> &D_A2_ArrayToExtract, const Array<CString, 1> &CS_A1_Args_In, void *ArgV_In[]);
      /// Args:
      /// AREA: Array<int, 1>(4)
      ///    I_XMin = I_A1_Area(0);
      ///    I_XMax = I_A1_Area(1);
      ///    I_YMin = I_A1_Area(2);
      ///    I_YMax = I_A1_Area(3);
      /// APERTURES: Array<int, 1>

      /// performs linear fitting to profile, with sky
      bool ExtractSpecFromProfile();

      /// performs linear fitting to profile, with or without sky
      bool ExtractSpecFromProfile(bool B_WithSky);

      /// performs linear fitting to profile, with sky
      bool ExtractSpecFromProfile(const Array<double, 2> &D_A2_ArrayToExtract);

      /// performs linear fitting to profile, with or without sky
      bool ExtractSpecFromProfile(const Array<double, 2> &D_A2_ArrayToExtract,
                                  const Array<CString, 1> &CS_A1_Args_In,
                                  void* PP_Args_In[]);
      /// Args:
      /// AREA: Array<int, 1>(4)
      ///    I_XMin = I_A1_Area(0);
      ///    I_XMax = I_A1_Area(1);
      ///    I_YMin = I_A1_Area(2);
      ///    I_YMax = I_A1_Area(3);
      /// APERTURES: Array<int, 1>

      bool ReadApertureData(double **PP_XArr,
                            double **PP_PixelValue,
                            const int Axis);

      bool ReadApertureDefinition(int I_Aperture_In);

      bool ReadDatabaseEntry();

      bool WriteDatabaseEntry() const;

      bool ReadArray();

      bool ReadErrArray();

      bool WriteArray() const;

      bool WriteErrArray() const;

      bool WriteFits(const Array<double,2>* P_D_A2_Image_In, const CString CS_FileName_In) const;

      bool WriteFits(const Array<double,1>* P_D_A1_Image_In, const CString CS_FileName_In) const;

      bool WriteFits(const Array<int,2>* P_I_A2_Image_In, const CString CS_FileName_In) const;

      bool WriteFits(const Array<int,1>* P_I_A1_Image_In, const CString CS_FileName_In) const;

      bool WriteApHead(CString CS_FileName_Out) const;

      bool WriteApertures(const CString &CS_FitsFileName_Out_Root, Array<CString, 1> &CS_A1_FileNames_Out) const;

      bool WriteApertures(const CString &CS_FitsFileName_Out_Root, Array<CString, 1> &CS_A1_FileNames_Out, const Array<int, 1> &I_A1_Apertures) const;

      bool WriteApCenters(const CString &CS_TextFileName_Out) const;

      bool WriteApCenters(const CString &CS_TextFileName_Out,
                          const Array<int, 1> &I_A1_Apertures) const;

      bool WriteApCenters(const CString &CS_FitsFileName_In,
                          const CString &CS_DatabaseFileName_In,
                          const CString &CS_TextFileName_Out) const;

      bool WriteApCenters(const CString &CS_FitsFileName_In,
                          const CString &CS_DatabaseFileName_In,
                          const CString &CS_TextFileName_Out,
                          const Array<int, 1> &I_A1_Apertures) const;

      bool Access() const;

      bool FitsAccess(const CString &fn) const;

      bool FileAccess(const CString &fn) const;

      bool AccessDatabaseFile() const;

      CString** CharArrayToCStringArray(char* p_charr[], int len) const;

      bool CalculateScatteredLight(int I_FittingOrder_In, int I_BoxCarWidth_In);

      /**
        Methods from Piskunov and Valenti
       **/
      /**
       MkSlitFunc
       Make Slit Function
       Parameter(s) to be tuned:
         swath_width - swath width in columns
         SF_SMOOTH - smoothing accross dispersion
         SP_SMOOTH - smoothing in dispersion direction
         OSAMPLE   - slit function is reconstructed on
                     subpixel grid with stepsize
                     OSAMPLE times smaller than CCD
                     pixels. Larger OSAMPLE require
                     more computing time and larger
                     SF_SMOOTH. Plots of residuals
                     allow to verify if the OSAMPLE
                     is good enough.
       **/
      bool MkSlitFunc(const Array<double, 1> &D_A1_ScatterBelow,  //: in
                      const Array<double, 1> &D_A1_XScatterBelow, //: in
                      const Array<double, 1> &D_A1_ScatterAbove,  //: in
                      const Array<double, 1> &D_A1_XScatterAbove, //: in
                      Array<double, 1> &D_A1_XSlitF,        //: out
                      Array<double, 2> &D_A2_SlitF,         //: out
                      Array<double, 1> &D_A1_BinCen,        //: out
                      const int I_IAperture_In,                    //: in
                      const Array<CString, 1> &CS_A1_Args,     //: in
                      void *ArgV[]);                        //: in
/* KeyWords and Values:      LAMBDA_SF   : double          : in
                             LAMBDA_SP   : int             : out
                             WING_SMOOTH_FACTOR = double  : in
                             SWATH_WIDTH : int             : in
                             BLZ         : Array<double, 1>: out
                             MASK        : Array<double, 2>: in
                             CCD_GAIN    : double          : in
                             CCD_READN   : double          : in
                             NO_SCATTER  : void
                             TELLURIC    : int (0-none, 1-Piskunov, 2-mine, 3-mine with sky background)
                             FILENAME    : CString         : in
                             XCOR_PROF   : int             : in (Number of Cross-correlations of profile and spectrum from one pixel to the left to one pixel to the right)
      */

      /**
      MkSlitFunc old version
      Make Slit Function
      **/
      bool MkSlitFunc_Old(Array<double, 1> &D_A1_ScatterBelow,  //: in
                           Array<double, 1> &D_A1_XScatterBelow, //: in
                           Array<double, 1> &D_A1_ScatterAbove,  //: in
                           Array<double, 1> &D_A1_XScatterAbove, //: in
                           //                       Array<double, 1> &D_A1_XCenters,      ///: in //
                           //                       double &D_XLeftLim,                      ///: in //
                           //                       double &D_XRightLim,                     ///: in //
                           Array<double, 1> &D_A1_XSlitF,        //: out
                           Array<double, 2> &D_A2_SlitF,         //: out
                           Array<double, 1> &D_A1_BinCen,        //: out
                           int I_IAperture_In,                   //: in
                           //                       int I_NArgs,                          //: in
                           const Array<CString, 1> &CS_A1_Args,           //: in
                           void *ArgV[]);                  //: in


      /**
      SlitFunc old version
      SlitFunc
      **/
      bool SlitFunc_Old(Array<double, 2> &D_A2_ImM,
                     int I_IAperture_In,
                     Array<double, 1> &D_A1_XCenters_In, //: in
                     Array<double, 1> &SPVecArr,   //: out
                     Array<double, 1> &SFVecArr,   //: out
                     //                     int NArgs,                //: in
                     const Array<CString, 1> &CS_A1_Args,   //: in
                     void *ArgV[]);     //: in


      /**
      MkProfIm Old version
      Make Profile Image
      **/
      bool MkProfIm_Old();
      bool MkProfIm_Old(const Array<CString, 1> &CS_A1_Args_In,       ///: in
                               void *ArgV[]);                                  ///: in

      /**
      SlitFunc
      Calculates slit function for one swath
       **/
      bool SlitFunc(const Array<double, 2> &D_A2_ImM,         ///: in
                    int I_IAperture_In,                       ///: in
                    const Array<double, 1> &D_A1_XCenOffset,  ///: in
                    Array<double, 1> &D_A1_SP_Out,                     ///: out
                    Array<double, 2> &D_A2_SF_Out,                     ///: out
                    const Array<CString, 1> &Args,            ///: in
                    void *ArgV[]);                            ///: in
/** KeyWords and Values:  NOISE      = double          : in
                          OVERSAMPLE = int             : in
                          IM_OUT     = Array<double, 2>: out
                          PROF_OUT   = Array<double, 2>: out
                          LAMBDA_SF  = double          : in
                          LAMBDA_SP  = int             : in
                          WING_SMOOTH_FACTOR = double  : in
                          USE_ROW    = int             : in
                          BAD        = Array<int, 1>   : out
                          MASK       = Array<double, 2>: in/out
                          TELLURIC   = int [0-none,1-Piskunov,2-mine]     : in
                          STOP       = int [0,1]                          : in
                          SKY        = Array<double, 1>(D_A2_ImM.rows())  : out
                          ERRORS     = Array<double, 2>(D_A2_ImM.rows(), D_A2_ImM.cols()): in/out
                          ERRORS_OUT = Array<double, 1>(D_A2_ImM.rows())  : out
                          ERR_SKY    = Array<double, 1>(D_A2_ImM.rows())  : out
                          SP_FIT     = Array<double, 1>(D_A2_ImM.rows())  : out
                          I_BIN      = int
      **/

  bool SlitFunc_2D(//const Array<double, 2> &D_A2_Im_In,
			//const Array<double, 1> &D_A1_XCenters_In,
			//Array<double, 1> &D_A1_SP_Out,
			//Array<double, 2> &D_A2_SF_Out,
			//int I_Delta_X_In,
			//double D_Shear_X_In,
			const Array<CString, 1> &CS_A1_Args_In,            ///: in
                        void *PP_ArgsV_In[]);

      /**
      MkProfIm
      Calculate Profile Image and optimally extract orders
       **/
      bool MkProfIm();

      /**
      MkProfIm
      Calculate Profile Image and optimally extract orders
       **/
      bool MkProfIm(const Array<CString, 1> &CS_A1_Args,  ///: in
                    void *ArgV[]);                        ///: in
/** KeyWords and Values:   SWATH_WIDTH:              int: in
                           LAMBDA_SP  :              int: in (Smoothing along dispersion)
                           BLZ        : Array<double, 1>: out
                           FLAT       :             bool: in
                           TELLURIC   :             bool: in
                           WING_SMOOTH_FACTOR :   double: in
                           XCOR_PROF  : int: in
                           APERTURES  : Array<int, 1>: in (Apertures to extract)
 **/


      /**
      MkNormFlatProf
      Make Profile for Normalized Flat
       **/
      bool MkNormFlatProf(int I_LambdaSP_In,
                          Array<CString, 1> CS_A1_Args,
                          void *ArgV[]);
      /**
       * Keywords and Values: SWATH_WIDTH:                 int: in
       *                      AREA:        Array<double, 1>(4): in
       *                      LAMBDA_SF:                double: in
       * **/

      /**
      MkScatter
      Make Scattered Light / Sky image
       **
      bool MkScatter(//back,                                ///out
                     //yback,                               ///out
                     const Array<CString, 1> &CS_A1_Args, ///: in
                     void *ArgV[]);                       ///in
      /** KeyWords and Values: COLRANGE=colrange
                               LAMBDA_SF=lam_sf...double
                               LAMBDA_SP=lam_sp...double
                               SWATH_WIDTH=swath_width...int
                               OSAMPLE=osample...int
                               MASK=mask
                               SUBTRACT=subtract...bool
                               POLY=pol
                               DATA=back_data
                               ORDER_WIDTH=order_width
      **/

      /**
       KeyWord_Set(const CString *p_cstr, const CString &cstr) const
       Returns Position of <cstr> in Array of CStrings <p_cstr>, if <p_cstr> contains CString <cstr>, else returns -1.
       **/
      int KeyWord_Set(const Array<CString, 1> &CS_A1_In, const CString &cstr) const;

      /**
      BandSol(Array<double, 2> &D_A2_A_In, Array<double, 1> &D_A1_R_In, int N, int I_ND) const

      bandsol solve a sparse system of linear equations with
      band-diagonal matrix.
      Band is assumed to be symmetrix relative to the main diaginal.
      Usage:
      bandsol(D_A2_A_In, D_A1_R_In, I_N, I_ND)
      where D_A2_A_In is 2D array [I_N,m] where I_N - is the number
      of equations and I_ND is the width of the band (3 for
      tri-diagonal system), I_ND is always an odd number. The main
      diagonal should be in D_A2_A_In(*,I_ND/2)
      The first lower subdiagonal should be in D_A2_A_In(1:I_N-1,I_ND-2-1),
      the first upper subdiagonal is in D_A2_A_In(0:I_N-2,I_ND/2+1)
      etc. For example:
                    / 0 0 X X X \
                    | 0 X X X X |
                    | X X X X X |
                    | X X X X X |
      D_A2_A_In =   | X X X X X |
                    | X X X X X |
                    | X X X X X |
                    | X X X X 0 |
                    \ X X X 0 0 /
      D_A1_R_In is the array of RHS of size I_N.
      Argv: Array<double, 2> &D_A2_A_InOut, Array<double, 1> &D_A1_R_InOut, int N, int I_ND
  **/
      int BandSol(int Argc, void *Argv[]) const;

      /**
      FIndGen(int len) const
       **/
      Vector<float>* FIndGen(int len) const;

      /**
      FIndGenArr(int len) const
       **/
      Array<float, 1>* FIndGenArr(int len) const;

      /**
      DIndGen(int len) const
      **/
      Vector<double>* DIndGen(int len) const;

      /**
      DIndGenArr(int len) const
      **/
      Array<double, 1>* DIndGenArr(int len) const;

      /**
      LIndGen(int len) const
       **/
      Vector<long>* LIndGen(int len) const;

      /**
      LIndGenArr(int len) const
       **/
      Array<long, 1>* LIndGenArr(int len) const;

      /**
      IndGen(int len) const
       **/
      Vector<int>* IndGen(int len) const;

      /**
      IndGenArr(int len) const
       **/
      Array<int, 1>* IndGenArr(int len) const;

      /**
; NAME:
      ;       MOMENT
      ;
; PURPOSE:
      ;       This function computes the mean, variance, skewness and kurtosis
      ;       of an N-element vector. IF the vector contains N identical elements,
      ;       the mean and variance are computed; the skewness and kurtosis are
      ;       not defined and are returned as IEEE NaN (Not a Number). Optionally,
      ;       the mean absolute deviation and standard deviation may also be
      ;       computed. The returned result is a 4-element vector containing the
      ;       mean, variance, skewness and kurtosis of the input vector X.
      ;
; CATEGORY:
      ;       Statistics.
      ;
; CALLING SEQUENCE:
      ;       Result = Moment(X)
      ;
; INPUTS:
      ;       X:      An N-element vector of type integer, float or double.
      ;
; KEYWORD PARAMETERS:
      ;       DOUBLE: IF set to a non-zero value, computations are done in
      ;               double precision arithmetic.
      ;
      ;       MDEV:   Use this keyword to specify a named variable which returns
      ;               the mean absolute deviation of X.
      ;
      ;       SDEV:   Use this keyword to specify a named variable which returns
      ;               the standard deviation of X.
      ;
;       MAXMOMENT:
;               Use this keyword to limit the number of moments:
      ;               Maxmoment = 1  Calculate only the mean.
      ;               Maxmoment = 2  Calculate the mean and variance.
      ;               Maxmoment = 3  Calculate the mean, variance, and skewness.
      ;               Maxmoment = 4  Calculate the mean, variance, skewness,
      ;                              and kurtosis (the default).
      ;
      ;       NAN:    Treat NaN elements as missing data.
      ;               (Due to problems with IEEE support on some platforms,
      ;                infinite data will be treated as missing as well. )
      ;
; EXAMPLE:
      ;       Define the N-element vector of sample data.
      ;         x = [65, 63, 67, 64, 68, 62, 70, 66, 68, 67, 69, 71, 66, 65, 70]
      ;       Compute the mean, variance, skewness and kurtosis.
      ;         result = moment(x)
;       The result should be the 4-element vector:
      double Moment(Array<double, 1>) const;
      ;       [66.7333, 7.06667, -0.0942851, -1.18258]
      ;
      ;
; PROCEDURE:
      ;       MOMENT computes the first four "moments" about the mean of an N-element
      ;       vector of sample data. The computational formulas are given in the IDL
      ;       Reference Guide.
      ;
; REFERENCE:
      ;       APPLIED STATISTICS (third edition)
      ;       J. Neter, W. Wasserman, G.A. Whitmore
      ;       ISBN 0-205-10328-6
      ;
      FUNCTION Moment, X, Double = Double, Mdev = Mdev, Sdev = Sdev, $
      Maxmoment = Maxmoment, NaN = nan
       **/
      Array<double,1>* Moment(const Array<double, 1> &D_A1_Arr_In, int I_MaxMoment_In) const;
      Array<double,1>* Moment(const Array<double, 2> &D_A2_Arr_In, int I_MaxMoment_In) const;

      /**
      double StdDev(Array<double, 1>) const;
       **/
      double StdDev(const Array<double, 1> &Arr) const;
      double StdDev(const Array<double, 2> &Arr) const;
      double StdDev(const Array<double, 2> &Arr, bool B_IgnoreZeros) const;

      /**
      double Median(Array<double, 1>) const;
      **/
      int Median(const Array<int, 1> &Arr) const;
      double Median(const Array<double, 1> &Arr) const;

      /**
      double Median(Array<double, 2>) const;
      **/
      double Median(const Array<double, 2> &Arr, bool B_IgnoreZeros) const;

      /**
      double Median(Array<double, 1>, CString &Mode) const;
      Args: MODE(CString)
            ERRORS_IN(Array<double, 1>)
            ERR_OUT(double)
       **/
      int Median(const Array<int, 1> &Arr, const Array<CString, 1> &CS_A1_Args_In, void *PP_Args[]) const;
      double Median(const Array<double, 1> &Arr, const Array<CString, 1> &CS_A1_Args_In, void *PP_Args[]) const;

      /**
      MedianVec(Array<double, 1>, int Width, CString "Even");
       **/
      Array<double, 1>* MedianVec(const Array<double, 1> &arr, int Width, const CString &Mode) const;

      /**
      MedianVec(Array<double, 1>, int Width");
       **/
      Array<double, 1>* MedianVec(const Array<double, 1> &arr, int Width) const;

      /**
      LinearRegression(Array<double, 1> y, const Array<double, 1> x, a1, a0);
      calculates a0 and a1 for the system of equations yvec = a0 + a1 * xvec
       **/
      bool LinearRegression(const Array<double, 1> &D_A1_CCD_In,
                            const Array<double, 1> &D_A1_SF_In,
                            double &D_SP_FIT,
                            double &D_Sky_Out,
                            double &D_STDDEV_Out,/// double &D_CorrelationCoefficient_Out,
                            double &D_Covariance_Out) const;

      /**
      LinearRegression(Array<double, 1> y, const Array<double, 1> x, a1, a0);
      calculates a0 and a1 for the system of equations yvec = a0 + a1 * xvec
       **/
      bool LinearRegression(const Array<double, 2> &D_A2_CCD_In,
                            const Array<double, 2> &D_A2_SF_In,
                            Array<double,1> &D_A1_SP_Out,
                            Array<double,1> &D_A1_Sky_Out,
                            Array<double,1> &D_A1_STDDEV_Out,/// double &D_CorrelationCoefficient_Out
                            Array<double,1> &D_A1_Covariance_Out) const;

      /**
      ; PURPOSE:
      ;       This function fits the paired data {X(i), Y(i)} to the linear model,
      ;       y = A + Bx, by minimizing the chi-square error statistic. The result
      ;       is a two-element vector containing the model parameters,[A,B].
      ;
; CATEGORY:
      ;       Statistics.
      ;
; CALLING SEQUENCE:
      ;       Result = LINFIT(X, Y)
      ;
; INPUTS:
      ;       X:    An n-element vector of type integer, float or double.
      ;
      ;       Y:    An n-element vector of type integer, float or double.
      ;
; KEYWORD PARAMETERS:
      ;   CHISQ:    Use this keyword to specify a named variable which returns the
      ;             chi-square error statistic as the sum of squared errors between
      ;             Y(i) and A + BX(i). If individual standard deviations are
      ;             supplied, then the chi-square error statistic is computed as
      ;             the sum of squared errors divided by the standard deviations.
      ;
      ;  COVAR:   Set this keyword to a named variable that will contain the
      ;           covariance matrix of the fitted coefficients.
      ;
      ;  DOUBLE:    If set to a non-zero value, computations are done in double
      ;             precision arithmetic.
      ;
      ;   MEASURE_ERRORS: Set this keyword to a vector containing standard
      ;       measurement errors for each point Y[i].  This vector must be the same
      ;       length as X and Y.
      ;
      ;     Note - For Gaussian errors (e.g. instrumental uncertainties),
      ;        MEASURE_ERRORS should be set to the standard
      ; 	     deviations of each point in Y. For Poisson or statistical weighting
      ; 	     MEASURE_ERRORS should be set to sqrt(Y).
      ;
      ;    PROB:    Use this keyword to specify a named variable which returns the
      ;             probability that the computed fit would have a value of CHISQR
      ;             or greater. If PROB is greater than 0.1, the model parameters
      ;             are "believable". If PROB is less than 0.1, the accuracy of the
      ;             model parameters is questionable.
      ;
      ;   SIGMA:    Use this keyword to specify a named variable which returns a
      ;             two-element vector of probable uncertainties for the model par-
      ;             ameters, [SIG_A,SIG_B].
      ;
      ;        Note: if MEASURE_ERRORS is omitted, then you are assuming that the
      ;              linear fit is the correct model. In this case,
      ;              SIGMA is multiplied by SQRT(CHISQ/(N-M)), where N is the
      ;              number of points in X. See section 15.2 of Numerical Recipes
      ;              in C (Second Edition) for details.
      ;
      ;    YFIT:    Set this keyword to a named variable that will contain the
      ;             vector of calculated Y values.

           REJECT:  Set this keyword if you want pixels which deviate from YFIT
                    by more than REJECT * stddev rejected from the fitting
                    procedure

           MASK:    Set this keyword to a named variable that contains
                      1 for pixels which shall be taken into account during the fit, and
                      0 for pixels which shall not be taken into account
                    If REJECT is set this MASK might get changed during the calculation.
      ;
; EXAMPLE:
      ;       Define two n-element vectors of paired data.
      ;         x = [-3.20, 4.49, -1.66, 0.64, -2.43, -0.89, -0.12, 1.41, $
      ;               2.95, 2.18,  3.72, 5.26]
      ;         y = [-7.14, -1.30, -4.26, -1.90, -6.19, -3.98, -2.87, -1.66, $
      ;              -0.78, -2.61,  0.31,  1.74]
      ;       Define a vector of standard deviations with a constant value of 0.85
      ;         sdev = replicate(0.85, n_elements(x))
      ;       Compute the model parameters, A and B.
      ;         result = linfit(x, y, chisq = chisq, prob = prob, sdev = sdev)
;       The result should be the two-element vector:
      ;         [-3.44596, 0.867329]
;       The keyword parameters should be returned as:
      ;         chisq = 11.4998, prob = 0.319925
      ;
; REFERENCE:
      ;       Numerical Recipes, The Art of Scientific Computing (Second Edition)
      ;       Cambridge University Press
      ;       ISBN 0-521-43108-5
      ;

      LinFit(Array<double, 1> y, const Array<double, 1> x, a1, a0);
      calculates a0 and a1 for the system of equations yvec = a0 + a1 * xvec

      based on IDL
       **/
      bool LinFit(const Array<double, 1> &D_A1_CCD_In,      /// y: in
                  const Array<double, 1> &D_A1_SF_In,       /// x: in
                  double &D_SP_Out,                         /// a1: out
                  double &D_Sky_Out,                        /// a0: out
                  const Array<CString, 1> &CS_A1_Args_In,   ///: in
                  void *ArgV_In[]) const;                   ///: in/out
      ///            SDEV_IN = Array<double,1>(D_A1_CCD_In.size)        : in
      ///            MEASURE_ERRORS = Array<double,1>(D_A1_CCD_In.size) : in
      ///            MASK = Array<int,1>(D_A1_CCD_In.size)              : in/out
      ///            REJECT = double                                    : in
      ///            CHISQ = double                                     : out
      ///            PROB = double                                      : out
      ///            SIGMA = Array<double,1>(2)                         : out
      ///            COVAR = Array<double,2>(2,2)                       : out
      ///            YFIT = Array<double,1>(D_A1_CCD_In.size)           : out

      /**
      LinFit(Array<double, 1> y, const Array<double, 1> x, a1, a0);
      calculates a0 and a1 for the system of equations yvec = a0 + a1 * xvec

      based on IDL
       **/
      bool LinFit(const Array<double, 2> &D_A2_CCD_In,      /// y: in
                  const Array<double, 2> &D_A2_SF_In,       /// x: in
                  Array<double,1> &D_A1_SP_Out,             /// a1: out
                  Array<double,1> &D_A1_Sky_Out,            /// a0: out
                  const Array<CString, 1> &CS_A1_Args_In,   ///: in
                  void *ArgV_In[]) const;                   ///: in/out
  /// SDEV_IN = Array<double,2>(D_A2_CCD_In.rows, D_A2_CCD_In.cols)        : in
  /// MEASURE_ERRORS = Array<double,2>(D_A2_CCD_In.rows, D_A2_CCD_In.cols) : in
  /// REJECT = double                                                      : in
  /// MASK = Array<int,2>(D_A2_CCD_In.rows, D_A2_CCD_In.cols)              : in/out
  /// CHISQ = Array<double,1>(D_A2_CCD_In.rows)                            : out
  /// PROB = Array<double,1>(D_A2_CCD_In.rows)                             : out
  /// SIGMA = Array<double,2>(D_A2_CCD_In.rows, 2)                         : out
  /// COVAR = Array<double,3>(D_A2_CCD_In.rows, 2, 2)                      : out
  /// YFIT = Array<double,2>(D_A2_CCD_In.rows, D_A2_CCD_In.cols)           : out

      /**;  Fit(Array<double, 1> y, const Array<double, 1> x, a1, a0);
            calculates a0 and a1 for the system of equations yvec = a0 + a1 * xvec

            Given a set of data points x[0..ndata-1], y[0..ndata-1] with individual standard deviations sig[0..ndata-1], fit them to a straight line y = a + bx by minimizing Chi^2. Returned are a, b, and their respective probable uncertainties siga and sigb, the chi-square chi2, and the goodness-of-fit probability q (that the fit would have Chi^2 this large or larger). if mwt=0 on input, then the standard deviations are assumed to be unavailable: q is returned as 1.0 and the normalization of chi2 is to unit standard deviation on all points.

            CHANGES to original function:
              * D_Sky_Out must be >= 0.
              * D_SP_Out must be >= 0.
              * if D_Sky_Out is set to be < -1.e-10 in input it is set to 0. and D_SP_Out is calculated as if there was no sky at all
              * weights set to Slit Function (didn't work => changed back to original weights 1 / sig^2)
              * added REJECT_In as optinal parameter to reject cosmic rays from fit
              * added MASK_INOUT as optional parameter
      **/
      bool Fit(const Array<double, 1> &D_A1_CCD_In,      /// y: in
               const Array<double, 1> &D_A1_SF_In,       /// x: in
               double &D_SP_Out,                         /// a1: out
               double &D_Sky_Out,                        /// a0: in/out
               const Array<CString, 1> &CS_A1_Args_In,   ///: in
               void *ArgV_In[]) const;                   ///: in/out
    /// MEASURE_ERRORS_IN = Array<double,1>(D_A1_CCD_In.size)             : in
    /// REJECT_IN = double                                                : in
    /// MASK_INOUT = Array<int,1>(D_A1_CCD_In.size)                    : in/out
    /// CHISQ_OUT = double                                                : out
    /// Q_OUT = double                                                    : out
    /// SIGMA_OUT = Array<double,1>(2): [*,0]: sigma_sp, [*,1]: sigma_sky : out
    /// YFIT_OUT = Array<double, 1>(D_A1_CCD_In.size)                     : out

      bool Fit(const Array<double, 1> &D_A1_CCD_In,      /// y: in
               const Array<double, 1> &D_A1_SF_In,       /// x: in
               double &D_SP_Out,                         /// a1: out
               double &D_Sky_Out,                        /// a0: in/out
               bool B_WithSky,                           /// with sky: in
               const Array<CString, 1> &CS_A1_Args_In,   ///: in
               void *ArgV_In[]) const;                   ///: in/out

      /**
      Fit(Array<double, 2> y, const Array<double, 2> x, const Array<double, 1>a1, const Array<double, 1>a0);
            calculates a0 and a1 for the system of equations yvec = a0 + a1 * xvec
       **/
      bool Fit(const Array<double, 2> &D_A2_CCD_In,      ///: in
               const Array<double, 2> &D_A2_SF_In,       ///: in
               Array<double,1> &D_A1_SP_Out,             ///: out
               Array<double,1> &D_A1_Sky_Out,            ///: in/out
               const Array<CString, 1> &CS_A1_Args_In,   ///: in
               void *ArgV_In[]) const;                   ///: in/out

  /// MEASURE_ERRORS_IN = Array<double,2>(D_A2_CCD_In.rows, D_A2_CCD_In.cols) : in
  /// REJECT_IN = double                                                      : in
  /// MASK_INOUT = Array<double,2>(D_A1_CCD_In.rows,D_A1_CCD_In.cols)         : in/out
  /// CHISQ_OUT = Array<double,1>(D_A2_CCD_In.rows)                           : out
  /// Q_OUT = Array<double,1>(D_A2_CCD_In.rows)                               : out
  /// SIGMA_OUT = Array<double,2>(D_A2_CCD_In.rows, 2): [*,0]: sigma_sp, [*,1]: sigma_sky : out
      bool Fit(const Array<double, 2> &D_A2_CCD_In,      ///: in
               const Array<double, 2> &D_A2_SF_In,       ///: in
               Array<double,1> &D_A1_SP_Out,             ///: out
               Array<double,1> &D_A1_Sky_Out,            ///: in/out
               bool B_WithSky,                           ///: with sky: in
               const Array<CString, 1> &CS_A1_Args_In,   ///: in
               void *ArgV_In[]) const;                   ///: in/out
    /// MEASURE_ERRORS_IN = Array<double,1>(D_A1_CCD_In.size)             : in
    /// REJECT_IN = double                                                : in
    /// MASK_INOUT = Array<int,1>(D_A1_CCD_In.size)                    : in/out
    /// CHISQ_OUT = double                                                : out
    /// Q_OUT = double                                                    : out
    /// SIGMA_OUT = Array<double,1>(2): [*,0]: sigma_sp, [*,1]: sigma_sky : out
    /// YFIT_OUT = Array<double, 1>(D_A1_CCD_In.size)                     : out

    /**
     * D_A1_CCD_In(N): N pixels
     * D_A2_SF_In(N,2): N pixels x 2 profiles
     * D_A1_SP_Out(2): spectrum values for profiles in D_A2_SF_In
     * D_Sky_Out: constant sky under both profiles
     **/
      bool LinFitBevingtonTwoProfiles(const Array<double, 1> &D_A1_CCD_In,      /// y: in
                                      const Array<double, 2> &D_A2_SF_In,       /// x: in
                                      Array<double, 1> &D_A1_SP_Out,                         /// a1: out
                                      double &D_Sky_Out,                        /// a0: in/out
                                      bool B_WithSky,                           /// with sky: in
                                      const Array<CString, 1> &CS_A1_Args_In,   ///: in
                                      void *ArgV_In[]) const;                   ///: in/out
    /// MEASURE_ERRORS_IN = Array<double,1>(D_A1_CCD_In.size)             : in
    /// REJECT_IN = double                                                : in
    /// MASK_INOUT = Array<int,1>(D_A1_CCD_In.size)                    : in/out
    /// CHISQ_OUT = double                                                : out
    /// Q_OUT = double                                                    : out
    /// SIGMA_OUT = Array<double,2>(2,2): [*,0]: sigma_sp, [*,1]: sigma_sky : out
    /// YFIT_OUT = Array<double, 1>(D_A1_CCD_In.size)                     : out
    /// ALLOW_SKY_LT_ZERO = 1
    /// ALLOW_SPEC_LT_ZERO = 1
    
     /**
            CHANGES to original function:
              * D_Sky_Out must be >= 0. unless stated otherwise by the ALLOW_SKY_LT_ZERO parameter
              * D_SP_Out must be >= 0. unless stated otherwise by the ALLOW_SPEC_LT_ZERO parameter
              * added REJECT_IN as optinal parameter to reject cosmic rays from fit (times sigma)
              * added MASK_INOUT as optional parameter
      **/
      bool LinFitBevington(const Array<double, 1> &D_A1_CCD_In,      /// y: in
                           const Array<double, 1> &D_A1_SF_In,       /// x: in
                           double &D_SP_Out,                         /// a1: out
                           double &D_Sky_Out,                        /// a0: in/out
                           bool B_WithSky,                           /// with sky: in
                           const Array<CString, 1> &CS_A1_Args_In,   ///: in
                           void *ArgV_In[]) const;                   ///: in/out
    /// MEASURE_ERRORS_IN = Array<double,1>(D_A1_CCD_In.size)             : in
    /// REJECT_IN = double                                                : in
    /// MASK_INOUT = Array<int,1>(D_A1_CCD_In.size)                    : in/out
    /// CHISQ_OUT = double                                                : out
    /// Q_OUT = double                                                    : out
    /// SIGMA_OUT = Array<double,1>(2): [*,0]: sigma_sp, [*,1]: sigma_sky : out
    /// YFIT_OUT = Array<double, 1>(D_A1_CCD_In.size)                     : out
    /// ALLOW_SKY_LT_ZERO = 1
    /// ALLOW_SPEC_LT_ZERO = 1

     /**
            CHANGES to original function:
              * D_Sky_Out must be >= 0. unless stated otherwise by the ALLOW_SKY_LT_ZERO parameter
              * D_SP_Out must be >= 0. unless stated otherwise by the ALLOW_SPEC_LT_ZERO parameter
              * if D_Sky_Out is set to be < -1.e-10 in input it is set to 0. and D_SP_Out is calculated as if there was no sky at all
              * added REJECT_IN as optinal parameter to reject cosmic rays from fit (times sigma)
              * added MASK_INOUT as optional parameter
      **/
      bool LinFitBevington(const Array<double, 1> &D_A1_CCD_In,      /// y: in
                           const Array<double, 1> &D_A1_SF_In,       /// x: in
                           double &D_SP_Out,                         /// a1: out
                           double &D_Sky_Out,                        /// a0: in/out
                           const Array<CString, 1> &CS_A1_Args_In,   ///: in
                           void *ArgV_In[]) const;                   ///: in/out
    /// MEASURE_ERRORS_IN = Array<double,1>(D_A1_CCD_In.size)             : in
    /// REJECT_IN = double                                                : in
    /// MASK_INOUT = Array<int,1>(D_A1_CCD_In.size)                    : in/out
    /// CHISQ_OUT = double                                                : out
    /// Q_OUT = double                                                    : out
    /// SIGMA_OUT = Array<double,1>(2): [*,0]: sigma_sp, [*,1]: sigma_sky : out
    /// YFIT_OUT = Array<double, 1>(D_A1_CCD_In.size)                     : out
    /// ALLOW_SKY_LT_ZERO = 1
    /// ALLOW_SPEC_LT_ZERO = 1

      /**
      Fit(Array<double, 2> y, const Array<double, 2> x, const Array<double, 1>a1, const Array<double, 1>a0);
            calculates a0 and a1 for the system of equations yvec = a0 + a1 * xvec
       **/
     /**
            CHANGES to original function:
              * D_Sky_Out must be >= 0.
              * D_SP_Out must be >= 0.
              * if D_Sky_Out is set to be < -1.e-10 in input it is set to 0. and D_SP_Out is calculated as if there was no sky at all
              * added REJECT_IN as optinal parameter to reject cosmic rays from fit (times sigma)
              * added MASK_INOUT as optional parameter
      **/
      bool LinFitBevington(const Array<double, 2> &D_A2_CCD_In,      ///: in
                           const Array<double, 2> &D_A2_SF_In,       ///: in
                           Array<double,1> &D_A1_SP_Out,             ///: out
                           Array<double,1> &D_A1_Sky_Out,            ///: in/out
                           const Array<CString, 1> &CS_A1_Args_In,   ///: in
                           void *ArgV_In[]) const;                   ///: in/out
  /// MEASURE_ERRORS_IN = Array<double,2>(D_A2_CCD_In.rows, D_A2_CCD_In.cols) : in
  /// REJECT_IN = double                                                      : in
  /// MASK_INOUT = Array<double,2>(D_A1_CCD_In.rows,D_A1_CCD_In.cols)         : in/out
  /// CHISQ_OUT = Array<double,1>(D_A2_CCD_In.rows)                           : out
  /// Q_OUT = Array<double,1>(D_A2_CCD_In.rows)                               : out
  /// SIGMA_OUT = Array<double,2>(D_A2_CCD_In.rows, 2): [*,0]: sigma_sp, [*,1]: sigma_sky : out

      bool LinFitBevington(const Array<double, 2> &D_A2_CCD_In,      ///: in
                           const Array<double, 2> &D_A2_SF_In,       ///: in
                           Array<double,1> &D_A1_SP_Out,             ///: out
                           Array<double,1> &D_A1_Sky_Out,            ///: in/out
                           bool B_WithSky,                           ///: with sky: in
                           const Array<CString, 1> &CS_A1_Args_In,   ///: in
                           void *ArgV_In[]) const;                   ///: in/out
    /// MEASURE_ERRORS_IN = Array<double,1>(D_A1_CCD_In.size)             : in
    /// REJECT_IN = double                                                : in
    /// MASK_INOUT = Array<int,1>(D_A1_CCD_In.size)                    : in/out
    /// CHISQ_OUT = double                                                : out
    /// Q_OUT = double                                                    : out
    /// SIGMA_OUT = Array<double,1>(2): [*,0]: sigma_sp, [*,1]: sigma_sky : out
    /// YFIT_OUT = Array<double, 1>(D_A1_CCD_In.size)                     : out
    
     /**
      print,'syntax: bottom,f,{filter/order}[,ITER=iter[,EPS=eps $'
      print,'               [,MIN=mn[,[MAX=mx[,/POLY]]]]]]'
      print,'where f      is the function to fit,'
      print,'      filter is the smoothing parameter for the optimal filter.'
      print,'             If POLY is set, it is interpreted as the order'
      print,'             of the smoothing polynomial,'
      print,'      iter   is the maximum number of iterations [def: 40]'
      print,'      eps    is convergence level [def: 0.001]'
      print,'      mn     minimum function values to be considered [def: min(f)]'
      print,'      mx     maximum function values to be considered [def: max(f)]'
      double Bottom(Array<double, 1>) const;
       **/
      bool Bottom(const Array<double,1> D_A1_Arr_In, int I_Filter_In, const Array<CString, 1> &CS_A1_Args, ///: in
                  void *ArgV[], Array<double, 1>* P_D_A1_Out) const;

/**Function middle,f,filter,ITER=iter,EPS=eps,POLY=pol,MIN=mn1,MAX=mx1
      ; middle tries to fit a smooth curve that is located
      ; along the "middle" of 1D data array f. Filter size "filter"
      ; together with the total number of iterations determine
      ; the smoothness and the quality of the fit. The total
      ; number of iterations can be controlled by limiting the
      ; maximum number of iterations (iter) and/or by setting
      ; the convergence criterion for the fit (eps)
      ; 4-Nov-200 N.Piskunov wrote.
      ;
      if(n_params() lt 2) then begin
      print,'syntax: middle,f,{filter/order}[,ITER=iter[,EPS=eps $'
      print,'               [,MIN=mn[,[MAX=mx[,/POLY]]]]]]'
      print,'where f      is the function to fit,'
      print,'      filter is the smoothing parameter for the optimal filter.'
      print,'             If POLY is set, it is interpreted as the order'
      print,'             of the smoothing polynomial,'
      print,'      iter   is the maximum number of iterations [def: 40]'
      print,'      eps    is convergence level [def: 0.001]'
      print,'      mn     minimum function values to be considered [def: min(f)]'
      print,'      mx     maximum function values to be considered [def: max(f)]'
      return,0
      endif**/
      bool Middle(const Array<double,1> D_A1_Arr_In, int I_Filter_In, const Array<CString, 1> &CS_A1_Args, ///: in
                  void *ArgV[], Array<double,1>* P_D_A1_Out) const;

      /**
      Function Opt_Filter,y,par,par1,DOUBLE=dbl,MAXIT=maxiter
      ;
      ; Optimal filtering of 1D and 2D arrays.
      ; Uses tridiag in 1D case and sprsin and linbcg in 2D case.
      ; Written by N.Piskunov 8-May-2000
      ;
      if(n_params() lt 2) then begin
      print,'Optimal filtering routine:'
      print,'Syntax: r=Opt_Filter(f,xwidth[,ywidth[,/DOUBLE[,MAXIT=maxiter]]])'
      print,'where:  f      - 1D or 2D array of type I,F or D'
      print,'        xwidth - filter width (for 2D array width in X direction (1st index)'
      print,'        ywidth - (for 2D array only) filter width in Y direction (2nd index)'
      print,'                 if ywidth is missing for 2D array, it set equal to xwidth'
      print,'        DOUBLE - perform calculations in double precision'
      print,'        maxiter- maximum number of iteration for filtering of 2D array'
      print,'Opt_Filter solves the optimization problem for r:'
      print,'        total((f - r)^2) + width*total((r(i) - r(i-1))^2) = min'
       **/
      Array<double,1>* OptFilter(const Array<double,1> &D_A1_In, int I_Width) const;
      Array<double,2>* OptFilter(const Array<double,2> &D_A2_In, int I_XWidth, int I_YWidth, int I_MaxIter) const;

      /**
      Select(Array<double, int> &arr, int KThSmallest) const;
      Returns the <KThSmallest> value of <Arr>.
       **/
      int Select(const Array<int, 1> &arr, int KThSmallest) const;
      double Select(const Array<double, 1> &arr, int KThSmallest) const;

      /**
        bool IsOddNumber(long No) const
        Returns TRUE, if <No> is an Odd Number, FALSE if <No> is an Even Number.
       **/
      bool IsOddNumber(long No) const;

      /**
        Swap values of both doubles
        **/
      void Swap(double &A,double &B) const;

      /**
      Swap values of both integers
       **/
      void Swap(int &A,int &B) const;

      /**
      Replicate(double val, int Len);
      Out: Array<double, 1>(Len)
       **/
      Array<double, 1>* Replicate(double val, int Len) const;

      /**
      Replicate(double val, int Len);
      Out: Array<double, 1>(Len)
       **/
      Array<int, 1>* Replicate(int val, int Len) const;

      /**
      MatrixATimesB(Array<double, 2> &Arr, Array<double, 2> &B);
      Out: Array<double, 2>(A.rows(), B.cols())
       **/
      Array<double, 2>* MatrixATimesB(Array<double, 2> &A, Array<double, 2> &B) const;

      /**
      MatrixBTimesA(Array<double, 2> &Arr, Array<double, 2> &B);
      Out: Array<double, 2>(B.rows(), A.cols())
       **/
      Array<double, 2>* MatrixBTimesA(Array<double, 2> &A, Array<double, 2> &B) const;

      /**
      MatrixTimesVecArr(Array<double, 2> &Arr, Array<double, 1> &B);
      Out: Array<double, 1>(A.rows())
       **/
      Array<double, 1>* MatrixTimesVecArr(Array<double, 2> &A, Array<double, 1> &B) const;

      /**
      VecArrTimesMatrix(Array<double, 1> &Arr, Array<double, 2> &B);
      Out: Array<double, 1>(B.cols())
      equivalent to IDL::operator #
      computes array elements by multiplying the rows of the first array by the columns of the second array
       **/
      Array<double, 1>* VecArrTimesMatrix(Array<double, 1> &A, Array<double, 2> &B) const;

      /**
      VecArrACrossB(Array<double, 1> &Arr, Array<double, 1> &B);
      Out: Array<double, 2>(A.size(), B.size())
       **/
      Array<double, 2>* VecArrACrossB(Array<double, 1> &A, Array<double, 1> &B) const;

      /**
      VecArrACrossB(Array<int, 1> &Arr, Array<int, 1> &B);
      Out: Array<int, 2>(A.size(), B.size())
       **/
      Array<int, 2>* VecArrACrossB(Array<int, 1> &A, Array<int, 1> &B) const;

      /**
      VecArrAScalarB(Array<double, 1> &Arr, Array<double, 1> &B);
      Out: double
       **/
      double VecArrAScalarB(Array<double, 1> &A, Array<double, 1> &B) const;

      /**
      Reform(Array<double, 1> &Arr, int DimA, int DimB);
      Reformats Vector to Array of given size
       **/
      Array<double, 2>* Reform(Array<double, 1> &Arr, int DimA, int DimB) const;

      /**
      Reform(Array<double, 2> &Arr);
      Reformates an Array to a Vector
       **/
      Array<double, 1>* Reform(Array<double, 2> &Arr) const;

      /**
      Reform(Array<int, 2> &Arr);
      Reformates an Array to a Vector
       **/
      Array<int, 1>* Reform(Array<int, 2> &Arr) const;

      /**
      Reform(Array<double, 2> &Arr, int DimA, int DimB);
      Reformates an Array
       **
      Array<double, 2>* Reform(Array<double, 2> &Arr, int DimA, int DimB) const;*/

//      Array<double, 2>* Transpose(Array<double, 2> &Arr) const;

      /**
       bool TriDag
       Solves for a vector Array<double, N> UVecArr the tridiagonal
       linear set given by equation
[ b_1  c_1  0  ...                       ]   [  u_1  ]   [  r_1  ]
[ a_2  b_2  c_2 ...                      ]   [  u_2  ]   [  r_2  ]
[            ...                         ] * [  ...  ] = [  ...  ]
[            ...  a_(N-1) b_(N-1) c_(N-1)]   [u_(N-1)]   [r_(N-1)]
[            ...     0     a_N      b_N  ]   [  u_N  ]   [  r_N  ]
      BVecArr, CVecArr, and RVecArr are input vectors and are not
      modified.
      **/
      bool TriDag(Array<double, 1> &AVecArr, Array<double, 1> &BVecArr, Array<double, 1> &CVecArr, Array<double, 1> &RVecArr, Array<double, 1> &UVecArr) const;

      /**
        LsToFit
       **/
      bool LsToFit(const Array<double, 1> &XXVecArr, const Array<double, 1> &YVecArr, const double &XM, double &D_Out) const;

      /**
        InvertGaussJ(AArray, BArray)
        FROM: Numerical Recipes
        Linear equation solution by Gauss-Jordan elimination
        AArray(0:N-1, 0:N-1) is the input matrix. BArray(0:N-1,
        0:M-1) is input containing the m right-hand side vectors.
        On output, AArray is replaced by its matrix inverse, and
        BArray is replaced by the corresponding set of solution
        vectors.
       **/
      bool InvertGaussJ(Array<double, 2> &AArray, Array<double, 2> &BArray) const;

      /**
        InvertGaussJ(AArray)
        FROM: Numerical Recipes
        Linear equation solution by Gauss-Jordan elimination with B == Unity
        AArray(0:N-1, 0:N-1) is the input matrix.
        On output, AArray is replaced by its matrix inverse.
       **/
      bool InvertGaussJ(Array<double, 2> &AArray) const;

      /**
      Spline
      Given Arrays XVecArr(0:N-1) and YVecArr(0:N-1) containing a
      tabulated function, i.e., y_i = f(x_i), with x_1 < x_2 < ... <
      x_N, and given values YP1 and YPN for the first derivative of
      the interpolating function at points 1 and N, respectively,
      this routine returns an Array y2(0:N-1) that contains the
      second derivatives of the interpolating function at the
      tabulated points x_i. If YP1 and/or YPN are equal to 1x10^30 or
      larger, the routine is signaled to set the corresponding
      boundary condition for a natural spline, with zero second
      derivative on that boundary.
       **/
      bool Spline(Array<double, 1> &XVecArr,
                  Array<double, 1> &YVecArr,
                  double YP1,
                  double YPN,
                  Array<double, 1> &Y2VecArr) const;

      /**
      Spline
      Given Arrays XVecArr(0:N-1) and YVecArr(0:N-1) containing a
      tabulated function, i.e., y_i = f(x_i), with x_1 < x_2 < ... <
      x_N, this routine returns an Array y2(0:N-1) that contains the
      second derivatives of the interpolating function at the
      tabulated points x_i. The routine is signaled to set the
      corresponding boundary condition for a natural spline, with
      zero second derivative on that boundary.
      **/
      bool Spline(Array<double, 1> &XVecArr,
                  Array<double, 1> &YVecArr,
                  Array<double, 1> &Y2VecArr) const;

      /**
      SplInt
      Given the Arrays XAVecArr(0:N-1) and YAVecArr(0:N-1), which
      tabulate a function (whith the XAVecArr(i)'s in order), and
      given the array Y2AVecArr(0:N-1), which is the output from
      Spline above, and given a value of X, this routine returns a
      cubic-spline interpolated value Y;
      **/
      bool SplInt(Array<double, 1> &XAVecArr,
                  Array<double, 1> &YAVecArr,
                  Array<double, 1> &Y2AVecArr,
                  double X,
                  double *Y) const;

      /**
      FUNCTION InterPol(V, X, U, SPLINE=spline, LSQUADRATIC=ls2, QUADRATIC=quad)
      NAME:
               INTERPOL
      PURPOSE:
               Linearly interpolate vectors with a regular or irregular grid.
               Quadratic or a 4 point least-square fit to a quadratic
               interpolation may be used as an option.

      CATEGORY:
               E1 - Interpolation

      CALLING SEQUENCE:
             Result = INTERPOL(V, N)         For regular grids.

             Result = INTERPOL(V, X, U)      For irregular grids.

      INPUTS:
             V:      The input vector can be any type except string.

         For regular grids:
             N:      The number of points in the result when both
                     input and output grids are regular.

         Irregular grids:
             X:      The absicissae values for V.  This vector must
                     have same # of elements as V.  The values MUST be
                     monotonically ascending or descending.

             U:      The absicissae values for the result.  The result
                     will have the same number of elements as U.  U
                     does not need to be monotonic.  If U is outside
                     the range of X, then the closest two endpoints of
                     (X,V) are linearly extrapolated.

      Keyword Input Parameters:
             LSQUADRATIC = if set, interpolate using a least squares
               quadratic fit to the equation y = a + bx + cx^2, for
               each 4 point neighborhood (x[i-1], x[i], x[i+1], x[i+2])
               surrounding the interval, x[i] <= u < x[i+1].

             QUADRATIC = if set, interpolate by fitting a quadratic
                         y = a + bx + cx^2, to the three point neighborhood
                         (x[i-1], x[i], x[i+1]) surrounding the interval
                         x[i] <= u < x[i+1].

             SPLINE = if set, interpolate by fitting a cubic spline to
                      the 4 point neighborhood (x[i-1], x[i], x[i+1],
                      x[i+2]) surrounding the interval, x[i] <= u <
                      x[i+1].

             Note: if LSQUADRATIC or QUADRATIC or SPLINE is not set,
                   the default linear interpolation is used.

      OUTPUTS:
             INTERPOL returns a floating-point vector of N points
             determined by interpolating the input vector by the
             specified method.

   PROCEDURE:
             For linear interpolation,
               Result(i) = V(x) + (x - FIX(x)) * (V(x+1) - V(x))
               where   x = i*(m-1)/(N-1) for regular grids.
                       m = # of elements in V, i=0 to N-1.

             For irregular grids, x = U(i).
                       m = number of points of input vector.

             For QUADRATIC interpolation, the equation y=a+bx+cx^2 is
               solved explicitly for each three point interval, and is
               then evaluated at the interpolate.

             For LSQUADRATIC interpolation, the coefficients a, b,
               and c, from the above equation are found, for the four
               point interval surrounding the interpolate using a
               least square fit.  Then the equation is evaluated at
               the interpolate.
               For SPLINE interpolation, a cubic spline is fit over
               the 4 point interval surrounding each interpolate,
               using the routine SPL_INTERP().
       **/
      /**
       InterPol linear, not regular
      **/
      bool InterPol(const Array<double, 1> &VVecArr,
                    const Array<double, 1> &XVecArr,
                    const Array<double, 1> &UVecArr,
                    Array<double, 1> *P_D_A1_Out) const;

      bool InterPol(const Array<double, 1> &VVecArr,
                    const Array<double, 1> &XVecArr,
                    const Array<double, 1> &UVecArr,
                    Array<double, 1> *P_D_A1_Out,
                    bool B_PreserveFlux) const;

      /**
        InterPol
         The InterPol function performs linear, quadratic, or spline interpolation on vectors with an irregular grid.
       **/
      bool InterPol(const Array<double, 1> &VVecArr,
                    const Array<double, 1> &XVecArr,
                    const Array<double, 1> &UVecArr,
                    const Array<CString, 1> &CS_A1_In,
                    Array<double,1> *P_D_A1_Out) const;

      /**
        InterPol
         This function performs linear, quadratic, or spline interpolation on vectors with a regular grid.
       **/
      bool InterPol(Array<double, 1> &VVecArr,
                    long N,
                    const Array<CString, 1> &CS_A1_In,
                    Array<double,1> *P_D_A1_Out) const;

      /**
        HInterPol
        Help function for InterPol methods
       **/
      bool HInterPol(const Array<double, 1> &VVecArr,
                     const Array<double, 1> &XVecArr,
                     Array<int, 1> &SVecArr,
                     const Array<double, 1> &UVecArr,
                     const Array<CString, 1> &CS_A1_In,
                     Array<double,1> *P_D_A1_Out) const;

      /**
       InterPolate
       This Method returns an Array of linear interpolates.
       The returned Array has the same type of D_A1_P and its dimension depends on those of the location parameter D_A1_X.
       INPUTS: Array<double, 1> D_A1_P(N)
                 The array of data values.
               Array<double, 1> D_A1_X(M)
                 The array of interpolation locations.
        NOTE: InterPolate considers location points with values
              between 0 and N, where N is the number of values in the
              input array D_A1_P, to be valid. Location points
              outside this range are considered missing data.
              Location points in the Range N-1 <= x < N return the
              last data value in the array D_A1_P.

        OUTPUTS: Array<double, 1> (M)
                   The array of interpolates.
       **/
      Array<double, 1>* InterPolate(const Array<double, 1> &D_A1_V,
                                    const Array<double, 1> &D_A1_X) const;

/**      ; NAME:
;       POLY
;
; PURPOSE:
;       Evaluate a polynomial function of a variable.
;
; CATEGORY:
;       C1 - Operations on polynomials.
;
; CALLING SEQUENCE:
;       Result = POLY(X,C)
;
; INPUTS:
;       X:      The variable.  This value can be a scalar, vector or array.
;
;       C:      The vector of polynomial coefficients.  The degree of
;               of the polynomial is N_ELEMENTS(C) - 1.
;
; OUTPUTS:
;       POLY returns a result equal to:
;                C[0] + c[1] * X + c[2]*x^2 + ...
;
; COMMON BLOCKS:
;       None.
;
; SIDE EFFECTS:
;       None.
;
; RESTRICTIONS:
;       None.
;
; PROCEDURE:
;       Straightforward.
**/
      Array<double, 1>* Poly(const Array<double, 1> &VecArr,
			     const Array<double, 1> &VecCoeffs) const;
      double Poly(const double D_X_In,
                  const Array<double, 1> &D_A1_Coeffs) const;
                             
                             
      /**
       * Calculates positive roots for a given polynomial
       * D_A1_PolyCoeffs_In: Coefficients for the polynomial
       * D_Y_In:             y value to find x
       * D_XStart_In:        x value to start search
       * D_XStep_In:         initial step for search (<0: go left, >0: go right)
       * CS_Function_In:     [Poly,Chebyshev,Legendre]
       * **/
      bool PolyRoot(const Array<double, 1> &D_A1_PolyCoeffs_In,
                    const double &D_Y_In,
                    const double &D_XStart_In,
                    const double &D_XStep_In,
                    const double &D_MaxDist_In,
                    const CString &CS_Function_In,
                    double &D_X_Out) const;

/**; NAME:
;   POLY_FIT
;
; PURPOSE:
;   Perform a least-square polynomial fit with optional error estimates.
;
;   This routine uses matrix inversion.  A newer version of this routine,
;   SVDFIT, uses Singular Value Decomposition.  The SVD technique is more
;   flexible, but slower.
;
; CATEGORY:
;   Curve fitting.
;
; CALLING SEQUENCE:
;   Result = POLY_FIT(X, Y, Degree)
;
; INPUTS:
;   X:  The independent variable vector.
;
;   Y:  The dependent variable vector, should be same length as x.
;
;   Degree: The degree of the polynomial to fit.
;
; OUTPUTS:
;   POLY_FIT returns a vector of coefficients with a length of NDegree+1.
;
; KEYWORDS:
;   CHISQ:   Sum of squared errors divided by MEASURE_ERRORS if specified.
;
;   COVAR:   Covariance matrix of the coefficients.
;
;   DOUBLE:  if set, force computations to be in double precision.
;
;   MEASURE_ERRORS: Set this keyword to a vector containing standard
;       measurement errors for each point Y[i].  This vector must be the same
;       length as X and Y.
;
;     Note - For Gaussian errors (e.g. instrumental uncertainties),
;        MEASURE_ERRORS should be set to the standard
;        deviations of each point in Y. For Poisson or statistical weighting
;        MEASURE_ERRORS should be set to sqrt(Y).
;
;   SIGMA:   The 1-sigma error estimates of the returned parameters.
;
;     Note: if MEASURE_ERRORS is omitted, then you are assuming that
;           your model is correct. In this case, SIGMA is multiplied
;           by SQRT(CHISQ/(N-M)), where N is the number of points
;           in X and M is the number of terms in the fitting function.
;           See section 15.2 of Numerical Recipes in C (2nd ed) for details.
    ;
    ;   STATUS = Set this keyword to a named variable to receive the status
;          of the operation. Possible status values are:
    ;          0 for successful completion, 1 for a singular array (which
    ;          indicates that the inversion is invalid), and 2 which is a
    ;          warning that a small pivot element was used and that significant
    ;          accuracy was probably lost.
    ;
    ;    Note: if STATUS is not specified then any error messages will be output
    ;          to the screen.
    ;
    ;   YBAND:  1 standard deviation error estimate for each point.
    ;
    ;   YERROR: The standard error between YFIT and Y.
    ;
    ;   YFIT:   Vector of calculated Y's. These values have an error
    ;           of + or - YBAND.
    ;
; COMMON BLOCKS:
    ;   None.
    ;
; SIDE EFFECTS:
    ;   None.
    FUNCTION POLY_FIT, x, y, ndegree, $
    CHISQ=chisq: double: out
    COVAR=covar: Array<double, 2>(I_Degree+1, I_Degree+1): out
    MEASURE_ERRORS=measure_errors: Array<double, 1>(D_A1_X_In.size()): in
    SIGMA=sigma: Array<double, 1>(I_Degree+1): out
    STATUS=status: int: out
    YBAND=yband: Array<double, 1>(D_A1_X.size()): in
    YERROR=yerror: in
    YFIT=yfit: Array<double, 1>(D_A1_X_In.size()): out
    ;**/
    bool PolyFit(const Array<double, 1> &D_A1_X_In,
		 const Array<double, 1> &D_A1_Y_In,
		 int I_Degree,
		 const Array<CString, 1> &CS_A1_Args,
		 void *ArgV[],
		 Array<double, 1>* P_D_A1_Out) const;

    bool PolyFit(const Array<double, 1> &D_A1_X_In,
		 const Array<double, 1> &D_A1_Y_In,
		 int I_Degree,
		 Array<double, 1>* P_D_A1_Out) const;
/** Additional Keywords:
    REJECTED=Array<int, 1>
    NOT_REJECTED=Array<int, 1>
    N_REJECTED=int
    **/
    bool PolyFit(const Array<double, 1> &D_A1_X_In,
                 const Array<double, 1> &D_A1_Y_In,
                 const int I_Degree,
                 const double D_Reject,
                 const Array<CString, 1> &CS_A1_Args,
                 void *ArgV[],
                 Array<double, 1>* P_D_A1_Out) const;
    bool PolyFit(const Array<double, 1> &D_A1_X_In,
                 const Array<double, 1> &D_A1_Y_In,
                 const int I_Degree,
                 const double D_LReject,
                 const double D_HReject,
                 const int I_NIter,
                 const Array<CString, 1> &CS_A1_Args,
                 void *ArgV[],
                 Array<double, 1>* P_D_A1_Out) const;

    bool PolyFit(const Array<double, 1> &D_A1_X_In,
                 const Array<double, 1> &D_A1_Y_In,
                 const int I_Degree,
                 const double D_Reject,
                 Array<double, 1>* P_D_A1_Out) const;
    bool PolyFit(const Array<double, 1> &D_A1_X_In,
                 const Array<double, 1> &D_A1_Y_In,
                 const int I_Degree,
                 const double D_LReject,
                 const double D_HReject,
                 const int I_NIter,
                 Array<double, 1>* P_D_A1_Out) const;

/**
; PURPOSE:
;   EVALUATE THE SUM OF A GAUSSIAN AND A 2ND ORDER POLYNOMIAL
;   AND OPTIONALLY RETURN THE VALUE OF IT'S PARTIAL DERIVATIVES.
;   NORMALLY, THIS FUNCTION IS USED BY CURVEFIT TO FIT THE
;   SUM OF A LINE AND A VARYING BACKGROUND TO ACTUAL DATA.
;
; CATEGORY:
;   E2 - CURVE AND SURFACE FITTING.
; CALLING SEQUENCE:
;   FUNCT,X,A,F,PDER
; INPUTS:
;   X = VALUES OF INDEPENDENT VARIABLE.
;   A = PARAMETERS OF EQUATION DESCRIBED BELOW.
; OUTPUTS:
;   F = VALUE OF FUNCTION AT EACH X(I).
;
; OPTIONAL OUTPUT PARAMETERS:
;   PDER = (N_ELEMENTS(X),6) ARRAY CONTAINING THE
;       PARTIAL DERIVATIVES.  P(I,J) = DERIVATIVE
;       AT ITH POINT W/RESPECT TO JTH PARAMETER.
; COMMON BLOCKS:
;   NONE.
; SIDE EFFECTS:
;   NONE.
; RESTRICTIONS:
;   NONE.
; PROCEDURE:
;   F = A(0)*EXP(-Z^2/2) + A(3) + A(4)*(X-A(1)) + A(5)*(X-A(1))^2
;   Z = (X-A(1))/A(2)
;   Elements beyond A(2) are optional.
; MODIFICATION HISTORY:
;   WRITTEN, DMS, RSI, SEPT, 1982.
;   Modified, DMS, Oct 1990.  Avoids divide by 0 if A(2) is 0.
;   Added to Gauss_fit, when the variable function name to
;       Curve_fit was implemented.  DMS, Nov, 1990.
;   CT, RSI, Dec 2003: Return correct array size if A[2] is 0.
;




; NAME:
;   GAUSSFIT
;
; PURPOSE:
;   Fit the equation y=f(x) where:
;
;       F(x) = A0*EXP(-z^2/2) + A3 + A4*(x-A1) + A5*(x-A1)^2
;           and
;       z=(x-A1)/A2
;
;   A0 = height of exp, A1 = center of exp, A2 = sigma (the width).
;   A3 = constant term, A4 = linear term, A5 = quadratic term.
;   Terms A3, A4, and A5 are optional.
;   The parameters A0, A1, A2, A3 are estimated and then CURVEFIT is
;   called.
;
; CATEGORY:
;   ?? - fitting
;
; CALLING SEQUENCE:
;   Result = GAUSSFIT(X, Y [, A])
;
; INPUTS:
;   X:  The independent variable.  X must be a vector.
;   Y:  The dependent variable.  Y must have the same number of points
;       as X. Y must not contain zero values
;
; KEYWORD INPUTS:
;
;   CHISQ: Set this keyword to a named variable that will contain
;      the value of the chi-square goodness-of-fit.
;
;   ESTIMATES = optional starting estimates for the parameters of the
;       equation.  Should contain NTERMS (6 if NTERMS is not
;       provided) elements.
;
;   MEASURE_ERRORS: Set this keyword to a vector containing standard
;       measurement errors for each point Y[i].  This vector must be the same
;       length as X and Y. It must not contain zero values.
;
;     Note - For Gaussian errors (e.g. instrumental uncertainties),
;        MEASURE_ERRORS should be set to the standard
;        deviations of each point in Y. For Poisson or statistical weighting
;        MEASURE_ERRORS should be set to sqrt(Y).
;
;   NTERMS = Set NTERMS to 3 to compute the fit: F(x) = A0*EXP(-z^2/2).
;      Set it to 4 to fit:  F(x) = A0*EXP(-z^2/2) + A3
;      Set it to 5 to fit:  F(x) = A0*EXP(-z^2/2) + A3 + A4*(x*A1)
;
;   SIGMA: Set this keyword to a named variable that will contain
;      the 1-sigma error estimates of the returned parameters.
;
;     Note: if MEASURE_ERRORS is omitted, then you are assuming that
;           your model is correct. In this case, SIGMA is multiplied
;           by SQRT(CHISQ/(N-M)), where N is the number of points
;           in X and M is the number of terms in the fitting function.
;           See section 15.2 of Numerical Recipes in C (2nd ed) for details.
;
;   YERROR: The standard error between YFIT and Y.
;
; OUTPUTS:
;   The fitted function is returned.
;
; OPTIONAL OUTPUT PARAMETERS:
;   A:  The coefficients of the fit.  A is a three to six
;       element vector as described under PURPOSE.
;
; COMMON BLOCKS:
;   None.
;
; SIDE EFFECTS:
;   None.
;
; RESTRICTIONS:
;   The peak or minimum of the Gaussian must be the largest
;   or smallest point in the Y vector.
;
; PROCEDURE:
;   The initial estimates are either calculated by the below procedure
;   or passed in by the caller.  Then the function CURVEFIT is called
;   to find the least-square fit of the gaussian to the data.
;
;  Initial estimate calculation:
;   If NTERMS>=4 then a constant term is subtracted first.
;   If NTERMS>=5 then a linear term is subtracted first.
;   If the (MAX-AVG) of Y is larger than (AVG-MIN) then it is assumed
;   that the line is an emission line, otherwise it is assumed there
;   is an absorbtion line.  The estimated center is the MAX or MIN
;   element.  The height is (MAX-AVG) or (AVG-MIN) respectively.
;   The width is found by searching out from the extrema until
;   a point is found less than the 1/e value.
**/
    bool GaussFit(const Array<double, 1> &D_A1_X,
		  const Array<double, 1> &D_A1_Y,
		  Array<double,1> &D_A1_A,
		  const Array<CString, 1> &CS_A1_Args_In,
		  void *ArgV[]) const;
///                  CHISQ=variable,
///                  ESTIMATES=array,
///                  MEASURE_ERRORS=vector,
///                  NTERMS=integer{3 to 6},
///                  SIGMA=double,
///                  YERROR=double,
///                  YFIT=Array<double, 1>*)

    bool GaussFit(const Array<double, 1> &D_A1_X,
		  const Array<double, 1> &D_A1_Y,
		  Array<double,1> &D_A1_A) const;

    /**
    ; NAME:
;       CURVEFIT
;
; PURPOSE:
;       Non-linear least squares fit to a function of an arbitrary
;       number of parameters.  The function may be any non-linear
;       function.  If available, partial derivatives can be calculated by
;       the user function, else this routine will estimate partial derivatives
;       with a forward difference approximation.
;
; CATEGORY:
;       E2 - Curve and Surface Fitting.
;
; CALLING SEQUENCE:
;       Result = CURVEFIT(X, Y, Weights, A, SIGMA, FUNCTION_NAME = name, $
;                         ITMAX=ITMAX, ITER=ITER, TOL=TOL, /NODERIVATIVE)
;
; INPUTS:
;       X:  A row vector of independent variables.  This routine does
;           not manipulate or use values in X, it simply passes X
;           to the user-written function.
;
;       Y:  A row vector containing the dependent variable.
;
;  Weights:  A row vector of weights, the same length as Y.
;            For no weighting,
;                 Weights(i) = 1.0.
;            For instrumental (Gaussian) weighting,
;                 Weights(i)=1.0/sigma(i)^2
;            For statistical (Poisson)  weighting,
;                 Weights(i) = 1.0/y(i), etc.
;
;            For no weighting, set Weights to an undefined variable.
;
;       A:  A vector, with as many elements as the number of terms, that
;           contains the initial estimate for each parameter.  IF A is double-
;           precision, calculations are performed in double precision,
;           otherwise they are performed in single precision. Fitted parameters
;           are returned in A.
;
; KEYWORDS:
;	FITA:   A vector, with as many elements as A, which contains a zero for
;   		each fixed parameter, and a non-zero value for elements of A to
;   		fit. If not supplied, all parameters are taken to be non-fixed.
;
;       FUNCTION_NAME:  The name of the function (actually, a procedure) to
;       fit.  IF omitted, "FUNCT" is used. The procedure must be written as
;       described under RESTRICTIONS, below.
;
;       ITMAX:  Maximum number of iterations. Default = 20.
;       ITER:   The actual number of iterations which were performed
;       TOL:    The convergence tolerance. The routine returns when the
;               relative decrease in chi-squared is less than TOL in an
;               interation. Default = 1.e-3.
;       CHI2:   The value of chi-squared on exit (obselete)
;
;       CHISQ:   The value of reduced chi-squared on exit
;       NODERIVATIVE:   IF this keyword is set THEN the user procedure will not
;               be requested to provide partial derivatives. The partial
;               derivatives will be estimated in CURVEFIT using forward
;               differences. IF analytical derivatives are available they
;               should always be used.
;
;       DOUBLE = Set this keyword to force the calculation to be done in
;                double-precision arithmetic.
;
;   STATUS: Set this keyword to a named variable in which to return
;           the status of the computation. Possible values are:
;           STATUS = 0: The computation was successful.
;           STATUS = 1: The computation failed. Chi-square was
;                       increasing without bounds.
;           STATUS = 2: The computation failed to converge in ITMAX
;                       iterations.
;
;   YERROR: The standard error between YFIT and Y.
;
; OUTPUTS:
;       Returns a vector of calculated values.
;       A:  A vector of parameters containing fit.
;
; OPTIONAL OUTPUT PARAMETERS:
;       Sigma:  A vector of standard deviations for the parameters in A.
;
;     Note: if Weights is undefined, then you are assuming that
;           your model is correct. In this case, SIGMA is multiplied
;           by SQRT(CHISQ/(N-M)), where N is the number of points
;           in X and M is the number of terms in the fitting function.
;           See section 15.2 of Numerical Recipes in C (2nd ed) for details.
;
; COMMON BLOCKS:
;       NONE.
;
; SIDE EFFECTS:
;       None.
;
; RESTRICTIONS:
;       The function to be fit must be defined and called FUNCT,
;       unless the FUNCTION_NAME keyword is supplied.  This function,
;       (actually written as a procedure) must accept values of
;       X (the independent variable), and A (the fitted function's
;       parameter values), and return F (the function's value at
;       X), and PDER (a 2D array of partial derivatives).
;       For an example, see FUNCT in the IDL User's Libaray.
;       A call to FUNCT is entered as:
;       FUNCT, X, A, F, PDER
; where:
;       X = Variable passed into CURVEFIT.  It is the job of the user-written
;           function to interpret this variable.
;       A = Vector of NTERMS function parameters, input.
;       F = Vector of NPOINT values of function, y(i) = funct(x), output.
;       PDER = Array, (NPOINT, NTERMS), of partial derivatives of funct.
;               PDER(I,J) = DErivative of function at ith point with
;               respect to jth parameter.  Optional output parameter.
;               PDER should not be calculated IF the parameter is not
;               supplied in call. IF the /NODERIVATIVE keyword is set in the
;               call to CURVEFIT THEN the user routine will never need to
;               calculate PDER.
;
; PROCEDURE:
;       Copied from "CURFIT", least squares fit to a non-linear
;       function, pages 237-239, Bevington, Data Reduction and Error
;       Analysis for the Physical Sciences.  This is adapted from:
;       Marquardt, "An Algorithm for Least-Squares Estimation of Nonlinear
;       Parameters", J. Soc. Ind. Appl. Math., Vol 11, no. 2, pp. 431-441,
;       June, 1963.
;
;       "This method is the Gradient-expansion algorithm which
;       combines the best features of the gradient search with
;       the method of linearizing the fitting function."
;
;       Iterations are performed until the chi square changes by
;       only TOL or until ITMAX iterations have been performed.
;
;       The initial guess of the parameter values should be
;       as close to the actual values as possible or the solution
;       may not converge.
;
; EXAMPLE:  Fit a function of the form f(x) = a * exp(b*x) + c to
;           sample pairs contained in x and y.
;           In this example, a=a(0), b=a(1) and c=a(2).
;           The partials are easily computed symbolicaly:
;           df/da = exp(b*x), df/db = a * x * exp(b*x), and df/dc = 1.0
;
;           Here is the user-written procedure to return F(x) and
;           the partials, given x:
;
;       pro gfunct, x, a, f, pder      ; Function + partials
;         bx = exp(a(1) * x)
;         f= a(0) * bx + a(2)         ;Evaluate the function
;         IF N_PARAMS() ge 4 THEN $   ;Return partials?
;         pder= [[bx], [a(0) * x * bx], [replicate(1.0, N_ELEMENTS(f))]]
;       end
;
;         x=findgen(10)                  ;Define indep & dep variables.
;         y=[12.0, 11.0,10.2,9.4,8.7,8.1,7.5,6.9,6.5,6.1]
;         Weights=1.0/y            ;Weights
;         a=[10.0,-0.1,2.0]        ;Initial guess
;         yfit=curvefit(x,y,Weights,a,sigma,function_name='gfunct')
;         print, 'Function parameters: ', a
;         print, yfit
;       end
**/
    bool CurveFit(const Array<double, 1> &D_A1_X,
		  const Array<double, 1> &D_A1_Y,
		  const Array<double, 1> &D_A1_WeightsIn,
		  Array<double, 1> &D_A1_A,
		  Array<double, 1> &D_A1_Sigma,
		  const Array<CString, 1> &CS_A1_ArgsIn,
		  void* PP_ArgsIn[]) const;
		  /// FUNCTION_NAME = *CString,
		  /// STATUS = *integer
		  /// TOL = *double
		  /// ITMAX = *integer
		  /// CHISQ = *double
		  /// FITA = Array<int, 1>*
		  /// NODERIVATIVE

/**
; NAME:
;	FUNCT
;
; PURPOSE:
;	Evaluate the sum of a Gaussian and a 2nd-order polynomial
;	and optionally return the value of its partial derivatives.
;	Normally, this function is used by CURVEFIT to fit the
;	sum of a line and a varying background to actual data.
;
; CATEGORY:
;	E2 - Curve and surface fitting.
;
; CALLING SEQUENCE:
;	FUNCT, X, A, F [, Pder]
;
; INPUTS:
;	X:	The values of the independent variable.
;	A:	The parameters of the equation described in PROCEDURE below.
;
; OUTPUTS:
;	F:	The value of the function at each X(i).
;
; OPTIONAL OUTPUT PARAMETERS:
;	Pder:	An array of the size (N_ELEMENTS(X),6) that contains the
;		partial derivatives.  Pder(i,j) represents the derivative
;		at the i'th point with respect to j'th parameter.
;
; COMMON BLOCKS:
;	None.
;
; SIDE EFFECTS:
;	None.
;
; RESTRICTIONS:
;	None.
;
; PROCEDURE:
;	F = A(0)*EXP(-Z^2/2) + A(3) + A(4)*X + A(5)*X^2
;	Z = (X-A(1))/A(2)
**/
    bool Funct(const Array<double, 1> &D_A1_X,
	       const Array<double, 1> &D_A1_A,
	       Array<double, 1> &D_A1_YFit) const;

    bool Funct(const Array<double, 1> &D_A1_X,
	       const Array<double, 1> &D_A1_A,
	       Array<double, 1> &D_A1_YFit,
	       Array<double, 1> &D_A1_EZ,
	       Array<double, 1> &D_A1_Z) const;

    bool Funct(const Array<double, 1> &D_A1_X,
	       const Array<double, 1> &D_A1_A,
	       Array<double, 1> &D_A1_YFit,
	       Array<double, 2> &D_A2_PDer) const;

/**; NAME:
;   GAUSS_FUNCT
;
; PURPOSE:
;   EVALUATE THE SUM OF A GAUSSIAN AND A 2ND ORDER POLYNOMIAL
;   AND OPTIONALLY RETURN THE VALUE OF IT'S PARTIAL DERIVATIVES.
;   NORMALLY, THIS FUNCTION IS USED BY CURVEFIT TO FIT THE
;   SUM OF A LINE AND A VARYING BACKGROUND TO ACTUAL DATA.
;
; CATEGORY:
;   E2 - CURVE AND SURFACE FITTING.
; CALLING SEQUENCE:
;   FUNCT,X,A,F,PDER
; INPUTS:
;   X = VALUES OF INDEPENDENT VARIABLE.
;   A = PARAMETERS OF EQUATION DESCRIBED BELOW.
; OUTPUTS:
;   F = VALUE OF FUNCTION AT EACH X(I).
;
; OPTIONAL OUTPUT PARAMETERS:
;   PDER = (N_ELEMENTS(X),6) ARRAY CONTAINING THE
;       PARTIAL DERIVATIVES.  P(I,J) = DERIVATIVE
;       AT ITH POINT W/RESPECT TO JTH PARAMETER.
; COMMON BLOCKS:
;   NONE.
; SIDE EFFECTS:
;   NONE.
; RESTRICTIONS:
;   NONE.
; PROCEDURE:
;   F = A(0)*EXP(-Z^2/2) + A(3) + A(4)*(X-A(1)) + A(5)*(X-A(1))^2
;   Z = (X-A(1))/A(2)
;   Elements beyond A(2) are optional.
;
PRO GAUSS_FUNCT,X,A,F,PDER
**/
    bool GaussFunct(const Array<double, 1> &D_A1_X,
		    const Array<double, 1> &D_A1_A,
		    Array<double, 1> &D_A1_F,
		    Array<double, 2> &D_A2_PDer) const;

    bool GaussFunct(const Array<double, 1> &D_A1_X,
		    const Array<double, 1> &D_A1_A,
		    Array<double, 1> &D_A1_F) const;

    bool GaussFunct(const Array<double, 1> &D_A1_X,
		    const Array<double, 1> &D_A1_A,
		    Array<double, 1> &D_A1_F,
		    Array<double, 1> &D_A1_Z,
		    Array<double, 1> &D_A1_EZ) const;

    void GaussFunct(double D_X,
		    const Array<double, 1> &D_A1_A,
		    double D_Y) const;

    /**
     * Calculate D_A1_YFit_Out
     * D_A1_Coeffs_In.size() == 3: one Gauss, no background
     * D_A1_Coeffs_In.size() == 4: one Gauss, constant background
     * D_A1_Coeffs_In.size() == 6: two Gauss, no background (NOTE: Not yet implemented)
     * D_A1_Coeffs_In.size() == 7: two Gauss, constant background (NOTE: Not yet implemented)
     * D_A1_Coeffs_In.size() == 9: three Gauss, no background (NOTE: Not yet implemented)
     * D_A1_Coeffs_In.size() == 10: three Gauss, constant background (NOTE: Not yet implemented)
     **/
     bool GaussArea(const Array<double, 1> &D_A1_X_In,
                  const Array<double, 1> &D_A1_Coeffs_In,
                  Array<double, 1> &D_A1_YFit_Out) const;
    /**
      ValueLocate
      Returns the Start Index of the Range of the two indixes of the monotonically increasing or decreasing Vector VecArr, in which Val falls.
      If Vector is monotonically increasing, the result is
      if j = -1       Value(i) < VecArr(0)
      if 0 <= j < N-1 VecArr(j) <= Value(i) < VecArr(j+1)
      if j = N-1      VecArr(N-1) <= Value(i)

      If Vector is monotonically decreasing, the result is
      if j = -1       VecArr(0) <= Value(i)
      if 0 <= j < N-1 VecArr(j+1) <= Value(i) < VecArr(j)
      if j = N-1      Value(i) < Vector(N-1)
    **/
    Array<int, 1>* ValueLocate(const Array<double, 1> &VecArr,
			       const Array<double, 1> &ValArr) const;

    /**
     * Fix(double)
     * Returns integer value cut at decimal point
     **/
    int Fix(double D_In) const;

    /**
      Fix(Array<double, 1> &VecArr, CString Mode)
      Returns an Array of the same size containing the integer values of VecArr. If <Mode> is set to "ROUND" the double values are rounded, else cut.
    **/
    Array<int, 1>* Fix(const Array<double, 1> &VecArr,
		       const CString &Mode) const;

    /**
      Fix(Array<double, 1> &VecArr, CString Mode)
      Returns an Array of the same size containing the not rounded integer values of VecArr.
    **/
    Array<int, 1>* Fix(const Array<double, 1> &VecArr) const;

      /**
      FixDF(Array<double, 1> &VecArr, CString Mode)
      Returns an Array of the same size containing the float values of VecArr.
       **/
      Array<float, 1>* FixDF(const Array<double, 1> &VecArr) const;

      /**
       Fix(Array<double, 2> &Arr, CString Mode)
       Returns an Array of the same size containing the integer values of Arr. If <Mode> is set to "ROUND" the double values are rounded, else cut.
       **/
      Array<int, 2>* Fix(const Array<double, 2> &Arr,
                         const CString &Mode) const;

      /**
       Fix(Array<double, 2> &Arr)
       Returns an Array of the same size containing the not rounded integer values of Arr.
       **/
      Array<int, 2>* Fix(const Array<double, 2> &Arr) const;

      /**
      FixL(Array<double, 1> &VecArr, CString Mode)
      Returns an Array of the same size containing the long integer values of VecArr. If <Mode> is set to "ROUND" the double values are rounded, else cut.
       **/
      Array<long, 1>* FixL(const Array<double, 1> &VecArr,
                           const CString &Mode) const;

      /**
      FixL(Array<double, 1> &VecArr, CString Mode)
      Returns an Array of the same size containing the not rounded long integer values of VecArr.
       **/
      Array<long, 1>* FixL(const Array<double, 1> &VecArr) const;

      /**
      FixL(Array<double, 2> &Arr, CString Mode)
      Returns an Array of the same size containing the long integer values of Arr. If <Mode> is set to "ROUND" the double values are rounded, else cut.
       **/
      Array<long, 2>* FixL(const Array<double, 2> &Arr,
                           const CString &Mode) const;

      /**
      FixL(Array<double, 2> &Arr)
      Returns an Array of the same size containing the not rounded long integer values of Arr.
       **/
      Array<long, 2>* FixL(const Array<double, 2> &Arr) const;

      /**
      FixD(Array<long, 1> &VecArr)
      Returns an Array of the same size containing the double values of Array<long, 1> Arr.
       **/
      Array<double, 1>* FixD(const Array<long, 1> &VecArr) const;

      /**
      FixD(Array<int, 1> &VecArr)
      Returns an Array of the same size containing the double values of Array<int, 1> Arr.
       **/
      Array<double, 1>* FixD(const Array<int, 1> &VecArr) const;

      /**
      FixIL(Array<int, 1> &VecArr)
      Returns an Array of the same size containing the long integer values of Array<int, 1> Arr.
       **/
      Array<long, 1>* FixIL(const Array<int, 1> &VecArr) const;

      /**
      FixLI(Array<long, 1> &VecArr)
      Returns an Array of the same size containing the integer values of Array<long, 1> Arr.
       **/
      Array<int, 1>* FixLI(const Array<long, 1> &VecArr) const;

      double Round(const double ToRound, int DigitsBehindDot) const;

      int Round(const double ToRound) const;

//      Array<int, 1>* Round(const Array<double, 1> D_A1_ToRound) const;

      long RoundToLong(const double ToRound) const;
      Array<int, 1>* Round(const Array<double, 1> &D_A1_ToRound_In) const;

      /**
        NAME:
            UNIQ

        PURPOSE:
            Return the subscripts of the unique elements in an array.

            This command is inspired by the Unix uniq(1) command.

        CATEGORY:
            Array manipulation.

        CALLING SEQUENCE:
            Uniq(Array<int, 1> IA1_In, Array<int, 1> IA1_Out)

        INPUTS:
            Array<int, 1> IA1_In:  The array to be scanned. The number of dimensions of the array is not important.  The array must be sorted into monotonic order.

        OUTPUTS:
            An array of indicies into ARRAY (Array<int, 1> IA1_Out) is returned.  The expression:

            Uniq(IA1_In, IA1_Out);
            Array<int, 1> SubArr(this->GetSubArr(IA1_In, I_A1_Out))

        will be a copy of the sorted Array with duplicate adjacent elements removed.

       **/
      bool Uniq(const Array<int, 1> &IA1_In,
                Array<int, 1> &IA1_Out) const;

      /**
      CrossCorrelate
        IN: DA1_Static: Vector held in place which the other vector is compared to
            DA1_Moving: Vector moving relative to DA1_Static, must have same length
                        as DA1_Static
            I_NPixMaxUp: Maximum number of elements DA1_Moving is shifted upwards compared to DA1_Static (>=0)
            I_NPixMaxDown: Maximum number of elements DA1_Moving is shifted downwards compared to DA1_Static (>=0)
        OUT: I_Out: Number of elements DA1_Moving was shifted where ChiSq is minimal (neg: up, pos: down)
      **/
      bool CrossCorrelate(const Array<double, 1> &DA1_Static, 
                          const Array<double, 1> &DA1_Moving, 
                          int I_NPixMaxUp, 
                          int I_NPixMaxDown, 
                          int &I_Out, 
                          double &D_ChiSquareMin_Out) const;

      bool CrossCorrelate(const Array<double, 1> &DA1_Static, 
                          const Array<double, 1> &DA1_Moving, 
                          int I_NPixMaxUp, 
                          int I_NPixMaxDown, 
                          double &D_Out, 
                          double &D_ChiSquareMin_Out) const;
                          
      /**
      CrossCorrelateAllColsToColNo
        IA2_Out(this->I_NApertures, 2) 0... smallest column # of aperture
                                       1... largest column # of aperture
       **/
      bool CrossCorrelateAllApertureColsToColNo(int I_Col,
                                                int I_NPixMaxUp,
                                                int I_NPixMaxDown,
                                                Array<int, 1> &IA1_Out,
                                                Array<int, 2> &IA2_Out) const;

      /**
      CalcFeatureOffsets
        Cross-correlates all columns to the column I_Col and calculates the offset of the spectral features with respect to column I_Col and fits the offsets
        I_Col          : IN:  Number of reference column
        I_NPixMaxUp    : IN:  Maximum shift upwards during cross-correlation
        I_NPixMaxDown  : IN:  Maximum shift downwards during cross-correlation
        I_Order        : IN:  Order for polynomial fit
        I_A1_X_Out     : OUT: Column numbers for which cross-correlation was performed (size: aperture_width x NAps)
        I_A2_ColsMinMax: OUT: Minimum and maximum column number per aperture (size: NAps, 2)
        D_A1_Out       : OUT: Fitted offset for each column the cross-correlation was performed (size: aperture_width x NAps) - see I_A1_X_Out
        CS_Mode        : IN: Poly/Chebyshev/Legendre - NOTE: So far only Poly is implemented
      **/
      bool CalcFeatureOffsets(int I_Col,
                              int I_NPixMaxUp,
                              int I_NPixMaxDown,
                              int I_Order,
                              Array<int, 1> &I_A1_X_Out,
                              Array<int, 2> &I_A2_ColsMinMax,
                              Array<double, 1> &D_A1_Out,
                              const CString &CS_Mode) const;

      /**
      Shift
        Shifts column D_A1_In by D_Offset
        KeyWords: none... linear interpolation
                  QUADRATIC... quadratic interpolation (See InterPol)
                  LSQUADRATIC... least squares quadratic interpolation
                  SPLINE... spline interpolation
        D_Offset < 0: shift up
        D_Offset > 0: shift down
       **/
      bool Shift(Array<double, 1> &D_A1_InOut,
                 double D_Offset,
                 const Array<CString, 1> &CS_A1_In) const;

      /**
      ShiftColumns
      Shifts columns I_A1_Cols of D_A2_In by D_A1_Offset
      I_A1_Cols and D_A1_Offset must have same length
      D_Offset < 0: shift up
      D_Offset > 0: shift down
       **/
      bool ShiftColumns(Array<double, 2> &D_A2_InOut,
                        const Array<int, 1> &I_A1_Cols,
                        const Array<double, 1> &D_A1_Offset,
                        const Array<CString, 1> &CS_A1_In) const;

      /**
      ShiftColumns
      Shifts columns I_A1_Cols of this->P_D_A2_PixArray by D_A1_Offset
      I_A1_Cols and D_A1_Offset must have same length
      D_Offset < 0: shift up
      D_Offset > 0: shift down
       **/
      bool ShiftColumns(const Array<int, 1> &I_A1_Cols,
                        const Array<double, 1> &D_A1_Offset,
                        const Array<CString, 1> &CS_A1_In);

      /**
       CalculateFeatureOffsetAndShiftAllImages
        - calculates Feature Offsets for spectral features with respect to colum I_ColWithRespectTo of image CS_ArcFile, straightens the spectral features in images in CS_FileListToShift, and writes them to images in CS_FileListOut (See CalcFeatureOffsets)
        - NOTE: only works if spectral features in one arperture all have the same curvature
       **/
      bool CalculateFeatureOffsetAndShiftAllImages(const CString &CS_ArcFile,
                                                   const CString &CS_ArcDatabaseFile,
                                                   const CString &CS_FileListToShift,
                                                   const CString &CS_FileListOut,
                                                   int I_ColWithRespectTo,
                                                   int I_NPixMaxUp,
                                                   int I_NPixMaxDown,
                                                   int I_FitOrder,
//                                                   Array<int, 1> &I_A1_X_Out,
//                                                   Array<int, 2> &I_A2_ColsMinMax,
//                                                   Array<double, 1> &D_A1_Out,
                                                   const CString &CS_Mode) const;

      bool CastIntArrToDblArr(const Array<int, 1> &I_A1_In, Array<double, 1> &D_A1_Out) const;
      bool CastIntArrToDblArr(const Array<int, 2> &I_A2_In, Array<double, 2> &D_A2_Out) const;

      /**
        GetSubArrCopy(Array<double, 1> &DA1_In, Array<int, 1> &IA1_Indices, Array<double, 1> &DA1_Out) const
        Copies the values of DA1_In(IA1_Indices) to DA1_Out
      **/
      bool GetSubArrCopy(const Array<double, 1> &DA1_In,
                         const Array<int, 1> &IA1_Indices,
                         Array<double, 1> &DA1_Out) const;

      /**
        GetSubArr(Array<double, 1> &DA1_In, Array<double, 1> &IA1_Indices) const
        Copies the adresses of DA1_In(IA1_Indices) to DA1_Out
      **/
      Array<double, 1>& GetSubArr(const Array<double, 1> &D_A1_In,
                                  const Array<int, 1> &I_A1_Indices) const;

      /**
      GetSubArrCopy(Array<int, 1> &IA1_In, Array<int, 1> &IA1_Indices, Array<int, 1> &IA1_Out) const
      Copies the values of IA1_In(IA1_Indices) to IA1_Out
       **/

      bool GetSubArrCopy(const Array<int, 1> &IA1_In,
                         const Array<int, 1> &IA1_Indices,
                         Array<int, 1> &IA1_Out) const;

      /**
        GetSubArr(Array<int, 1> &IA1_In, Array<int, 1> &IA1_Indices,
        Array<int, 1> &IA1_Out) const
        Copies the adresses of IA1_In(IA1_Indices) to IA1_Out
       **/
      Array<int, 1>& GetSubArr(const Array<int, 1> &IA1_In,
                               const Array<int, 1> &IA1_Indices) const;

      /**
        GetSubArrCopy(Array<double, 2> &DA2_In, Array<int, 1> &IA1_Indices, int I_Mode_In, Array<double, 2> &DA2_Out) const
        Copies the values of DA1_In(IA1_Indices) to DA2_Out
        I_Mode_In: 0: IA1_Indices are row numbers
                   1: IA1_Indices are column numbers
       **/
      bool GetSubArrCopy(const Array<double, 2> &DA2_In,
                         const Array<int, 1> &IA1_Indices,
                         int I_Mode_In,
                         Array<double, 2> &DA2_Out) const;

      /**
        GetSubArrCopy(Array<double, 2> &DA2_In, Array<int, 1> &IA1_Indices, int I_Mode_In, Array<double, 1> &DA1_Out) const
        Copies the values of DA1_In(IA1_Indices) to DA1_Out
        I_Mode_In: 0: IA1_Indices are running numbers going through rows, then cols: rows=5, cols=7, Ind=10 => (0,2)
                   1: IA1_Indices are running numbers going through cols, then rows: rows=5, cols=7, Ind=10 => (1,3)
       **/
      bool GetSubArrCopy(const Array<double, 2> &DA2_In,
                         const Array<int, 1> &IA1_Indices,
                         int I_Mode_In,
                         Array<double, 1> &DA1_Out) const;

      /**
        GetSubArr(Array<double, 2> &DA2_In, Array<double, 1> &IA1_Indices, int I_Mode_In) const
        Copies the adresses of DA1_In(IA1_Indices) to DA1_Out
        I_Mode_In: 0: IA1_Indices are row numbers
                   1: IA1_Indices are column numbers
                   2: IA1_Indices are running numbers going through rows, then cols: rows=5, cols=7, Ind=10 => (0,2)
                   3: IA1_Indices are running numbers going through cols, then rows: rows=5, cols=7, Ind=10 => (1,3)
      **/
      Array<double, 2>& GetSubArr(const Array<double, 2> &D_A2_In,
                                  const Array<int, 1> &I_A1_Indices,
                                  int I_Mode_In) const;

      /**
      GetSubArrCopy(Array<int, 2> &IA2_In, Array<int, 1> &IA1_Indices, int I_Mode_In, Array<int, 2> &IA2_Out) const
      Copies the values of IA1_In(IA1_Indices) to IA2_Out
        I_Mode_In: 0: IA1_Indices are row numbers
                   1: IA1_Indices are column numbers
                   **/
      bool GetSubArrCopy(const Array<int, 2> &I_A2_In,
                         const Array<int, 1> &I_A1_Indices,
                         int I_Mode_In,
                         Array<int, 2> &I_A2_Out) const;

      /**
      GetSubArrCopy(Array<int, 2> &IA2_In, Array<int, 1> &IA1_Indices, int I_Mode_In, Array<int, 1> &IA1_Out) const
      Copies the values of IA1_In(IA1_Indices) to IA2_Out
        I_Mode_In: 0: IA1_Indices are running numbers going through rows, then cols: rows=5, cols=7, Ind=10 => (0,2)
                   1: IA1_Indices are running numbers going through cols, then rows: rows=5, cols=7, Ind=10 => (1,3)
       **/
      bool GetSubArrCopy(const Array<int, 2> &I_A2_In,
                         const Array<int, 1> &I_A1_Indices,
                         int I_Mode_In,
                         Array<int, 1> &I_A1_Out) const;

      /**
        GetSubArr(Array<int, 2> &IA2_In, Array<int, 1> &IA1_Indices,
        int I_Mode_In) const
        Copies the adresses of IA2_In(IA1_Indices) to IA1_Out
        I_Mode_In: 0: IA1_Indices are row numbers
                   1: IA1_Indices are column numbers
                   2: IA1_Indices are running numbers going through rows, then cols: rows=5, cols=7, Ind=10 => (0,2)
                   3: IA1_Indices are running numbers going through cols, then rows: rows=5, cols=7, Ind=10 => (1,3)
       **/
      Array<int, 2>& GetSubArr(const Array<int, 2> &I_A2_In,
                               const Array<int, 1> &IA1_Indices,
                               int I_Mode_In) const;

      /**
        GetSubArr(Array<double, 2> &DA1_In, Array<int, 3> &I_A3_Indices) const
        Copies the adresses of D_A2_In(I_A3_Indices(row,col,0), I_A3_Indices(row,col,1)) to D_A2_Out
      **/
      Array<double, 2>& GetSubArr(const Array<double, 2> &D_A2_In,
                                  const Array<int, 3> &I_A3_Indices) const;

      /**
        GetSubArr(Array<int, 2> &I_A2_In, Array<int, 1> &I_A3_Indices) const
        Copies the adresses of I_A1_In(I_A3_Indices(row,col,0), I_A3_Indices(row,col,1)) to I_A2_Out
      **/
      Array<int, 2>& GetSubArr(const Array<int, 2> &I_A2_In,
                               const Array<int, 3> &I_A3_Indices) const;

      bool GetSubArrCopy(const Array<double, 2> &D_A2_In,
                         const Array<int, 2> &I_A2_Indices,
                         Array<double, 1> &D_A1_Out) const;

      /**
        Find
        Returns and Array<int, 1> containing the
      **
      Array<int,1>* Find(const TinyVector<int,1> &I_TV_Index) const;**/

      /**
       Ceil(double D_In)
       Returns the closest long integer greater than or equal to its
       argument.
      **/
      long Ceil(double D_In) const;

      /**
      Ceil(Array<double, 1> D_A1_In)
      Returns the closest long integer greater than or equal to its
      argument.
       **/
      Array<long, 1>* Ceil(Array<double, 1> &D_A1_In) const;

      /**
      Floor(double D_In)
      Returns the closest long integer lower than or equal to its
      argument.
       **/
      long Floor(double D_In) const;

      /**
      Floor(Array<double, 1> D_A1_In)
      Returns the closest long integer lower than or equal to its
      argument.
       **/
      Array<long, 1>* Floor(Array<double, 1> &D_A1_In) const;

      /**
      Signum(double D_In)
      Returns +1 if D_In >= 0., else returns -1
      **/
      int Signum(double D_In) const;

      /**
      SortIndices(Array<double, 1> D_A1_In)
      Returns an integer array of the same size like <D_A1_In>,
      containing the indixes of <D_A1_In> in sorted order.
       **/
      Array<int, 1>* SortIndices(const Array<double, 1> &D_A1_In) const;

      /** Show methods:
          =============

          task   : Shows the object's attributes at 'os'.
          require: none
          ensure : works**/
      virtual void Show(std::ostream &os) const;

      /** Test methods:
          =============

      task: Sets the aperture centers to 0
      **/
      bool MarkCenters();

      /**
      task: Writes the aperture centers of aperture no <iap> to file <fn>
      **/
      bool WriteCenters(int iap, CString &fn);

      /**
       *      task: Writes Array <I_A1_In> to file <CS_FileName_In>
       *      CS_Mode: [binary, ascii]
       **/
      bool WriteArrayToFile(const Array<int, 1> &I_A1_In,
                            const CString &CS_FileName_In,
                            const CString &CS_Mode) const;

      /**
      task: Writes Array <D_A1_In> to file <CS_FileName_In>
      CS_Mode: [binary, ascii]
       **/
      bool WriteArrayToFile(const Array<double, 1> &D_A1_In,
                            const CString &CS_FileName_In,
                            const CString &CS_Mode) const;

      /**
      task: Writes Array <D_A2_In> to file <CS_FileName_In>
      CS_Mode: [binary, ascii]
       **/
      bool WriteArrayToFile(const Array<double, 2> &D_A2_In,
                            const CString &CS_FileName_In,
                            const CString &CS_Mode) const;

      /**
      task: Writes Array <I_A2_In> to file <CS_FileName_In>
      CS_Mode: [binary, ascii]
       **/
      bool WriteArrayToFile(const Array<int, 2> &D_A2_In,
                            const CString &CS_FileName_In,
                            const CString &CS_Mode) const;

      /**
        function Bubble Sort
       **/
      Array<int, 1>* BubbleSort(const Array<int, 1> &I_A1_ArrIn) const;
      Array<double, 1>* BubbleSort(const Array<double, 1> &D_A1_ArrIn) const;

      /**
        function Sort - seems faster than Bubble Sort
        **/
      Array<double, 1>* Sort(const Array<double, 1> &D_A1_ArrIn) const;

      /**
        function CountPixGTZero
        replaces input vector with vector of the same size where values are zero where the input vector is zero and all other values represent the number of non-zero values since the last zero value
       **/
      bool CountPixGTZero(Array<int, 1> &I_A1_VecInOut) const;

      /**
        function FirstIndexWithValueGE
        returns first index of integer input vector where value is greater than or equal to I_MinValue
        returns -1 if values are always smaller than I_MinValue
       **/
      int FirstIndexWithValueGE(const Array<int, 1> &I_A1_VecIn, const int I_MinValue) const;

      /**
       *        function FirstIndexWithValueGE
       *        returns first index of double input vector where value is greater than or equal to D_MinValue_In
       *        returns -1 if values are always smaller than D_MinValue_In
       **/
      int FirstIndexWithValueGE(const Array<double, 1> &D_A1_VecIn, const double D_MinValue_In) const;

      /**
        function FirstIndexWithValueGEFrom
        returns first index of integer input vector where value is greater than or equal to I_MinValue, starting at index I_FromIndex
        returns -1 if values are always smaller than I_MinValue
       **/
      int FirstIndexWithValueGEFrom(const Array<int, 1> &I_A1_VecIn, const int I_MinValue, const int I_FromIndex) const;

      /**
       *        function FirstIndexWithValueGEFrom
       *        returns first index of double input vector where value is greater than or equal to D_MinValue_In, starting at index I_FromIndex
       *        returns -1 if values are always smaller than D_MinValue_In
       **/
      int FirstIndexWithValueGEFrom(const Array<double, 1> &D_A1_VecIn, const double D_MinValue_In, const int I_FromIndex) const;

      /**
        function FirstIndexWithValueGT
        returns first index of integer input vector where value is greater than I_MinValue
        returns -1 if values are always smaller than or equal to I_MinValue
       **/
      int FirstIndexWithValueGT(const Array<int, 1> &I_A1_VecIn, const int I_MinValue) const;

      /**
       *        function FirstIndexWithValueGT
       *        returns first index of double input vector where value is greater than D_MinValue_In
       *        returns -1 if values are always smaller than or equal to D_MinValue_In
       **/
      int FirstIndexWithValueGT(const Array<double, 1> &D_A1_VecIn, const double D_MinValue_In) const;

      /**
        function FirstIndexWithValueGTFrom
        returns first index of integer input vector where value is greater than I_MinValue, starting at index I_FromIndex
        returns -1 if values are always smaller than or equal to I_MinValue
       **/
      int FirstIndexWithValueGTFrom(const Array<int, 1> &I_A1_VecIn, const int I_MinValue, const int I_FromIndex) const;

      /**
       *        function FirstIndexWithValueGTFrom
       *        returns first index of double input vector where value is greater than D_MinValue_In, starting at index I_FromIndex
       *        returns -1 if values are always smaller than or equal to D_MinValue_In
       **/
      int FirstIndexWithValueGTFrom(const Array<double, 1> &D_A1_VecIn, const double D_MinValue_In, const int I_FromIndex) const;

      /**
        function LastIndexWithZeroValueBefore
        returns last index of integer input vector where value is equal to zero, starting at index I_StartPos
        returns -1 if values are always greater than 0 before I_StartPos
       **/
      int LastIndexWithZeroValueBefore(const Array<int, 1> &I_A1_VecIn, const int I_StartPos) const;

      /**
       *        function LastIndexWithZeroValueBefore
       *        returns last index of double input vector where value is equal to zero, starting at index I_StartPos
       *        returns -1 if values are always greater than 0 before I_StartPos
       **/
      int LastIndexWithZeroValueBefore(const Array<double, 1> &D_A1_VecIn, const int I_StartPos) const;

      /**
        function FirstIndexWithZeroValueFrom
        returns first index of integer input vector where value is equal to zero, starting at index I_StartPos
        returns -1 if values are always greater than 0 past I_StartPos
       **/
      int FirstIndexWithZeroValueFrom(const Array<int, 1> &I_A1_VecIn, const int I_StartPos) const;

      /**
       *        function FirstIndexWithZeroValueFrom
       *        returns first index of double input vector where value is equal to zero, starting at index I_StartPos
       *        returns -1 if values are always greater than 0 past I_StartPos
       **/
      int FirstIndexWithZeroValueFrom(const Array<double, 1> &D_A1_VecIn, const int I_StartPos) const;

      /**
       *        function LastIndexWithNonZeroValueBefore
       *        returns last index of integer input vector where value is not equal to zero, starting at index I_StartPos
       *        returns -1 if values are always equal to 0 before I_StartPos
       **/
      int LastIndexWithNonZeroValueBefore(const Array<int, 1> &I_A1_VecIn, const int I_StartPos) const;

      /**
       *        function LastIndexWithNonZeroValueBefore
       *        returns last index of double input vector where value is not equal to zero, starting at index I_StartPos
       *        returns -1 if values are always equal to 0 before I_StartPos
       **/
      int LastIndexWithNonZeroValueBefore(const Array<double, 1> &D_A1_VecIn, const int I_StartPos) const;

      /**
       function GetRowFromIndex(int I_Index_In, int I_NRows_In) const
       task: Returns Row specified by I_Index_In from the formula
             Col = (int)(I_Index_In / I_NRows_In)
             Row = I_Index_In - Col * I_NRows_In
       **/
      int GetRowFromIndex(int I_Index_In, int I_NRows_In) const;

      /**
       function GetColFromIndex(int I_Index_In, int I_NRows_In) const
       task: Returns Col specified by I_Index_In from the formula
             Col = (int)(I_Index_In / I_NRows_In)
             Row = I_Index_In - Col * I_NRows_In
       **/
      int GetColFromIndex(int I_Index_In, int I_NRows_In) const;

      bool InsertAt(Array<double,1> *P_D_A1_In, double D_ToInsert, int I_Pos) const;
      bool InsertAt(Array<int,1> *P_D_A1_In, int D_ToInsert, int I_Pos) const;

      /**
        Returns Indexes of I_A1_Where where I_A1_Where equals 1 and writes sum(I_A1_Where) to I_NInd_Out
      **/
      Array<int,1>* GetIndex(const Array<int,1> &I_A1_Where, int &I_NInd_Out) const;

      /**
      Returns Indexes of I_A1_Where where I_A1_Where equals 1 and writes sum(I_A1_Where) to I_NInd_Out
       **/
      bool GetIndex(const Array<int,1> &I_A1_Where, int &I_NInd_Out, Array<int, 1> &I_IndArr_Out) const;

      /**
      Returns Indexes of I_A2_Where where I_A2_Where equals 1 and writes sum(I_A2_Where) to I_NInd_Out
      Array<int, 2> *P_I_A2_Out(I_NInd_Out, 2)
      **/
      Array<int,2>* GetIndex(const Array<int,2> &I_A2_Where, int &I_NInd_Out) const;

      /**
      Returns Indexes of I_A2_Where where I_A2_Where equals 1 and writes sum(I_A2_Where) to I_NInd_Out
      Array<int, 2> I_IndArr_Out(I_NInd_Out, 2)
      **/
      bool GetIndex(const Array<int,2> &I_A2_Where, int &I_NInd_Out, Array<int, 2> &I_IndArr_Out) const;

      /**
      Returns I_TV_Index as Array<int,1>
       **/
      Array<int,1>* MaxIndex(TinyVector<int,1> &I_TV_Index) const;

      /**
      Returns I_A1_Index as Array<int,2>
      I_Mode == 0: IA1_Indices are running numbers going through rows, then cols: rows=5, cols=7, Ind=10 => (0,2)
                1: IA1_Indices are running numbers going through cols, then rows: rows=5, cols=7, Ind=10 => (1,3) **/
      bool A1ToA2(Array<int,1> &I_A1_Indices, int I_NRows, int I_NCols, Array<int,2>& I_A2_Out, int I_Mode) const;

      /**
        Returns integer array containing the number of elements in D_A1_In within bin i
      **/
      Array<int, 1>* Histogram(const Array<double,1> &D_A1_In, double D_BinSize_In) const;

      /**
       Helper function to calculate incomplete Gamma Function
       **/
      double GammLn(double D_X_In) const;

      /**
      Helper function to calculate incomplete Gamma Function
       **/
      bool GSER(double *P_D_Gamser_In, double a, double x, double *P_D_GLn) const;

      /**
      Helper function to calculate incomplete Gamma Function
       **/
      bool GCF(double *P_D_Gamser_In, double a, double x, double *P_D_GLn) const;

      /**
      Function to calculate incomplete Gamma Function P
       **/
      bool GammP(double a, double x, double *D_Out) const;

      /**
      Function to calculate incomplete Gamma Function Q = 1 - P
       **/
      bool GammQ(double a, double x, double *D_Out) const;

      /**
      Solves A*X=B for a vector X, where A is specified by the Arrays U[0...M-1][0...N-1], W[0...N-1], V[0...N-1][0...N-1] as returned by SVDCMP. M and N are the dimensions of A, and will be equal for square matrices. B[0...M-1] is the input right-hand side. X[0...N-1] is the output solution vector. No input quantities are destroyed, so the routine may be called sequentially with different B's.
       **/
      bool SVBKSB(const Array<double, 2> &U, const Array<double, 1> &W, const Array<double, 2> &V, int M, int N, const Array<double, 1> &B, Array<double, 1> &X) const;

      /**
      Given a matrix a[0..m-1][0..n-1], this routine computes its singular value
      decomposition, A = U*W*V^T.  The matrix U replaces a on output.  The diagonal
      matrix of singular values W is output as a vector W[0..n-1].  The matrix V (not
      the transpose VT) is output as V[0..N-1][0..N-1].
       **/
      bool SVDCMP(Array<double, 2> &A, int M, int N, Array<double, 1> &W, Array<double, 2> &V) const;

      /**
      Computes sqrt(a^2 + b^2) without distructive underflow or overflow
      **/
      double Pythag(double a, double b) const;

      /**
      Given a set of data points X[0..n-1], Y[0..n-1] with individual standard deviations Sig[0..n-1], use Chi^2 minimization to determine the coefficients a[0..ma-1] of the fitting function y = sum(a_i * aFunc_i(X), i). Here we solve the fitting equations using singular value decomposition of the n by ma matrix, as in $2.6. Arrays U[0..n-1][0..ma-1], V[0..ma-1][0..ma-1], and W[0..ma] provide workspace on input; on output they define the singular value decomposition, and can be used to obtain the covariance matrix. The program returns values for the ma fit parameters A, and Chi^2, ChiSq. The user supplies a routine Funcs(X, AFunc, ma) that returns the ma basis functions evaluated at x = X in the array AFunc[0..ma].
       **/
      bool SVDFit(const Array<double, 1> &X,
		  const Array<double, 1> &Y,
		  const Array<double, 1> &Sig,
		  Array<double, 1> &A,
		  Array<double, 2> &U,
		  Array<double, 2> &V,
		  Array<double, 1> &W,
		  Array<double, 1> &ChiSq,
		  void (*Funcs)(double, Array<double, 1>, int)) const;

      bool ReadFileToStrArr(const CString &CS_FileName_In,
                            Array<CString, 2> &CS_A2_Out,
                            const CString &CS_Delimiter) const;

      bool ReadFileToDblArr(const CString &CS_FileName_In,
                            Array<double, 2> &D_A2_Out,
                            const CString &CS_Delimiter) const;

      bool ReadFileLinesToStrArr(const CString &CS_FileName_In,
                                 Array<CString, 1> &CS_A1_Out) const;

      bool BoxCarFilter(Array<double, 2> D_A2_Data_InOut,
                        int I_BoxCarWidth_In,
                        bool B_IgnoreZeros_In) const;

      bool Set_ApertureDataToZero(const int I_PlusNPixels_X, const int I_PlusNPixels_Y);
      bool Set_ApertureDataToZero(Array<double, 2> &D_A2_InOut, const int I_PlusNPixels_X, const int I_PlusNPixels_Y) const;

      ///Mode: 0: Median
      ///      1: Mean
      bool FindNonZeroClusters(Array<double, 2> &D_A2_InOut_PixArray,
                               int I_In_BoxSize_X,
                               int I_In_BoxSize_Y,
                               int I_In_Mode,
                               Array<double, 2> &D_A2_Out) const;

      /// Sets Pixel values to zero and the centers of the Rectangles to the mean value of the Rectangle
      bool FindMeanValuesOfRectangles(int I_In_WidthX, int I_In_WidthY);

      bool ScaleImageToFitBackground(const Array<double, 2> &D_A2_Scatter_In,
                                     double &D_Fact_Out) const;

      bool ScaleImageToFitBackground(const Array<double, 2> &D_A2_Image,
                                     const Array<double, 2> &D_A2_Scatter_In,
                                     double &D_Fact_Out) const;

      Array<double, 1>* MakeVector(const Array<double, 2> &D_A2_In) const;

      /// Calculate Scattered Light using Kriging Algorithm
      /// IntArr_ClusterSizes_X_Y_In: Array(NClusterRuns, 2)
      /// I_AddNPixToAp_X: number of pixels to add to aperture_width in X
      /// I_AddNPixToAp_X: number of pixels to add to aperture_width in Y
      /// I_NRectangles_X: number of rectangles in X direction
      /// I_NRectangles_Y: number of rectangles in Y direction
      /// FileName_ApZero_Out=<CString>
      /// FileName_Clustered_Out=<CString>
      /// FileName_ScatterFit_Out=<CString>

      bool CalcScatterKriging(const Array<int, 2> &IntArr_ClusterSizes_X_Y_In,
                              const int I_AddNPixToAp_X,
                              const int I_AddNPixToAp_Y,
                              const int I_NRectangles_X,
                              const int I_NRectangles_Y,
                              Array<double, 2> &D_A2_ScatteredLight_Out,
                              Array<CString, 1> &CS_A1_Args_In,
                              void *PP_Args_In[]) const;

      /// Estimate Scattered Light using Kriging Algorithm by taking the minimum value per rectangle
      /// plus 1 ReadOutNoise as input value for the fit
      /// I_NRectangles_X: number of rectangles in X direction
      /// I_NRectangles_Y: number of rectangles in Y direction
      bool EstScatterKriging(const int I_NRectangles_X,
                             const int I_NRectangles_Y,
                             Array<double, 2> &D_A2_ScatteredLight_Out);//,
//                             Array<CString, 1> &CS_A1_Args_In,
//                             void *PP_Args_In[]) const;

      bool EstScatterKriging(const int I_NRectangles_X,
                             const int I_NRectangles_Y,
                             Array<double, 2> &D_A2_ScatteredLight_Out,
                             Array<CString, 1> &CS_A1_Args_In,
                             void *PP_Args_In[]) const;
                              /// AREA = Array<int, 1>(4) 0: I_XMin, 1: I_XMax, 2: I_YMin, 3: I_YMax

      bool EstScatterKriging(const Array<double, 2> &D_A2_PixArray_In,
                             const int I_NRectangles_X,
                             const int I_NRectangles_Y,
                             Array<double, 2> &D_A2_ScatteredLight_Out,
                             Array<CString, 1> &CS_A1_Args_In,
                             void *PP_Args_In[]) const;
                              /// AREA = Array<int, 1>(4) 0: I_XMin, 1: I_XMax, 2: I_YMin, 3: I_YMax

      /**
       * WriteApToFile
       * Writes aperture I_Aperture_In from double array D_A2_Spectra_In(NApertures, NRows), using aperture definitions of this (YCenter, YLow, YHigh), to fits file CS_FitsFileName_Out
       **/
      bool WriteApToFile(const Array<double, 2> &D_A2_Spectra_In,
                         const int I_Aperture_In,
                         const CString &CS_FitsFileName_Out) const;

      /**
       * StretchAndCrossCorrelateSpec
       * + Splits D_A1_Spec_In and D_A1_SpecRef_In into I_NPieces_In and calculates shift and stretch to achieve minimum ChiSq in crosscorrelation
       * + fits polynomials to Stretches and Shifts with 3sigma rejection
       * + applies shifts and stretches to D_A2_LineList_WLenPix_In to calculate D_A2_LineList_WLenPix_Out
       **/
      bool StretchAndCrossCorrelateSpec(const Array<double, 1> &D_A1_Spec_In,
                                        const Array<double, 1> &D_A1_SpecRef_In,
                                        const Array<double, 2> &D_A2_LineList_WLenPix_In,
                                        const int I_Radius_XCor_In,
                                        const int I_Stretch_Min_Length_In,
                                        const int I_Stretch_Max_Length_In,
                                        const int I_NStretches_In,
                                        const int I_LengthPieces_In,
                                        const int I_NCalcs_In,
                                        const int I_PolyFitOrder_Stretch_In,
                                        const int I_PolyFitOrder_Shift_In,
                                        const CString &CS_FName_In,
                                        Array<double, 2> &D_A2_LineList_WLenPix_Out) const;

      bool StretchAndCrossCorrelate(const Array<double, 1> &D_A1_Spec_In,
                                    const Array<double, 1> &D_A1_SpecRef_In,
                                    const int I_Radius_XCor_In,
                                    const int I_Stretch_Min_Length_In,
                                    const int I_Stretch_Max_Length_In,
                                    const int I_N_Stretches_In,
                                    double &D_Stretch_Out,
                                    double &D_Shift_Out,
                                    Array<double, 2> &D_A2_SpecStretched_MinChiSq) const;

      /**
       * Identify
       * Identifies calibration lines, given in D_A2_LineList_In the format [wlen, approx_pixel] in
       * wavelength-calibration spectrum D_A2_Spec_In [pixel_number, flux]
       * within the given position plus/minus I_Radius_In,
       * fits Gaussians to each line, fits Polynomial of order I_PolyFitOrder_In, and
       * returns calibrated spectrum D_A2_CalibratedSpec_Out in the format
       * [WLen, flux] and PolyFit coefficients D_A1_PolyFitCoeffs_Out
       *  
       * If D_A2_LineList_In contains 3 columns, the 3rd column will be used to decide which line
       * to keep in case a weak line close to a strong line gets wrongly identified as the strong
       * line
       **/
      bool Identify(const Array<double, 1> &D_A1_Spec_In,
                    const Array<double, 2> &D_A2_LineList_In,
                    const int I_Radius_In,
                    const double D_FWHM_In,
                    const int I_PolyFitOrder_In,
                    const CString &CS_FName_In,
                    Array<double, 2> &D_A2_CalibratedSpec_Out,
                    Array<double, 1> &D_A1_PolyFitCoeffs_Out,
                    double &D_RMS_Out,
		    Array<double, 2> &D_A2_PixWLen_Out) const;

      /**
       * DispCor
       * Applies dispersion correction to spectrum and returns wavelength-calibrated
       * two-dimensional spectrum (D_A1_Spec_In.size(), 2) [WLen, Flux]
       **/
      bool DispCor(const Array<double, 1> &D_A1_Spec_In,
                   const Array<double, 1> &D_A1_PolyFitCoeffs_In,
                   Array<double, 2> &D_A2_Spec_Out);

      /**
       * apply pixel shift to index positions (<0: shift object spectra down, >0: shift object spectra up)
       **/
      bool DispCor(const Array<double, 1> &D_A1_Spec_In,
                   const Array<double, 1> &D_A1_PolyFitCoeffs_In,
                   const double D_PixShift_In,
                   Array<double, 2> &D_A2_Spec_Out);
      
      /**
       * NOTE: Use only when P_D_A2_PixArray contains the spectra in the format
       * (I_NApertures, I_NRows)
       * Writes wavelengths for each spectrum to P_D_A2_WLen
       */
      bool DispCorList(const Array<CString, 1>& CS_A1_TextFiles_Coeffs_In,
                       const Array<CString, 1>& CS_A1_TextFiles_EcD_Out,
                       const double D_MaxRMS_In);

      bool DispCorList(const Array<CString, 1>& CS_A1_TextFiles_Coeffs_In,
                       const Array<CString, 1>& CS_A1_TextFiles_EcD_Out,
                       const double D_MaxRMS_In,
                       const Array<int, 1> &I_A1_Apertures);

      /**
       * Apply pixel shift to original object spectra
       **/      
      bool DispCorList(const Array<CString, 1>& CS_A1_TextFiles_Coeffs_In,
                       const Array<CString, 1>& CS_A1_TextFiles_EcD_Out,
                       const double D_MaxRMS_In,
                       const double D_PixShift_In,
                       const Array<int, 1> &I_A1_Apertures);
      
      bool RebinTextList(const Array<CString, 1> &CS_A1_TextFileNames_EcD_List_In,
                         const Array<CString, 1> &CS_A1_TextFileNames_EcDR_List_Out,
                         const CString &CS_FitsFileName_EcDR_Out,
                         double D_Lambda_Start,
                         double D_Lambda_End,
                         double D_DLambda) const;

      bool RebinTextList(const Array<CString, 1> &CS_A1_TextFileNames_EcD_List_In,
                         const Array<CString, 1> &CS_A1_TextFileNames_EcDR_List_Out,
                         const CString &CS_FitsFileName_EcDR_Out,
                         double D_Lambda_Start,
                         double D_Lambda_End,
                         double D_DLambda,
                         bool B_PreserveFlux) const;

      bool CollapseSpectra(const Array<CString, 1> &CS_A1_TextFileNameList_ap_x_y_In,
                           const CString &CS_TextFileName_Out) const;

      /**
       * bool WriteCube(Array<double, 3> &D_A3_In, CString &CS_FitsFileName_Out) const
       **/
      bool WriteCube(Array<double, 3> &D_A3_In, CString &CS_FitsFileName_Out) const;

      /** find nearest neighbour to D_ReferencePoint_In in D_A1_ArrayToLookForNeighbour_In, and returns index of nearest neighbour
       * */
      int FindNearestNeighbour(const double &D_ReferencePoint_In,
                                const Array<double, 1> &D_A1_ArrayToLookForNeighbour_In) const;
      
      /**
       * find nearest neighbour to D_A1_ReferencePoint_In(x,y) in D_A2_ArrayToLookForNeighbour_In and write coordinates to D_A1_NearestNeighbour_Out
       **/
      bool FindNearestNeighbour(const Array<double, 1> &D_A1_ReferencePoint_In,
                                const Array<double, 2> &D_A2_ArrayToLookForNeighbour_In,
                                Array<double, 1> &D_A1_NearestNeighbour_Out,
                                int &I_Pos_Out) const;
                                
      /**
       * I_A1_Area = [x_min, x_max, y_min, y_max]
       * **/
      bool FindNearestNeighbour(const Array<double, 1> &D_A1_ReferencePoint_In,
                                const Array<double, 2> &D_A2_ArrayToLookForNeighbour_In,
                                const Array<int, 1> &I_A1_Area,
                                Array<double, 1> &D_A1_NearestNeighbour_Out,
                                int &I_Pos) const;

      /**
       * find I_N nearest neighbours to D_A1_ReferencePoint_In(x,y) in D_A2_ArrayToLookForNeighbour_In and write coordinates to D_A2_NearestNeighbour_Out
       **/
      bool FindNearestNeighbours(const Array<double, 1> &D_A1_ReferencePoint_In,
                                 const Array<double, 2> &D_A2_ArrayToLookForNeighbour_In,
                                 const int I_N,
                                 Array<double, 2> &D_A2_NearestNeighbours_Out,
                                 Array<int, 1> &I_A1_Pos) const;

      /**
       * find I_N nearest neighbours to D_A1_ReferencePoint_In(x,y) in D_A2_ArrayToLookForNeighbour_In and write coordinates to D_A2_NearestNeighbour_Out
       **/
      bool FindNearestNeighbours(const Array<double, 1> &D_A1_ReferencePoint_In,
                                 const Array<double, 2> &D_A2_ArrayToLookForNeighbour_In,
                                 const int I_N,
                                 const Array<int, 1> &D_A1_Area_In,
                                 Array<double, 2> &D_A2_NearestNeighbours_Out,
                                 Array<int, 1> &I_A1_Pos) const;

      /**
       * Calculate Integral under line between two points
       * D_A2_Coords_In(0,0) = x0
       * D_A2_Coords_In(0,1) = y0
       * D_A2_Coords_In(1,0) = x1
       * D_A2_Coords_In(1,1) = y1
       * **/
      bool IntegralUnderLine(const Array<double, 2> &D_A2_Coords_In,
                             double &D_Integral_Out) const;

      /**
       * Calculate Integral in Triangle
       * **/
      bool IntegralInTriangle(const Array<double, 2> &D_A2_Coords_In,
                              double &D_Integral_Out) const;

      /**
       * Calculate Integral within closed structure
       * D_A2_Coords_In(0,0) = x0
       * D_A2_Coords_In(0,1) = y0
       * D_A2_Coords_In(1,0) = x1
       * D_A2_Coords_In(1,1) = y1
       *           *
       *           *
       *           *
       * **/
      bool IntegralInClosedStructure(const Array<double, 2> &D_A2_Coords_In,
                                     double &D_Integral_Out) const;

      /**
       * Calculate Integral under curve from D_A1_XInt(0) to D_A1_XInt(1)
       **/
      bool IntegralUnderCurve(const Array<double, 1> &D_A1_XIn,
	                      const Array<double, 1> &D_A1_YIn,
			      const Array<double, 1> &D_A1_XInt,
			      double &D_Integral_Out) const;

      /**
       * Integral-normalise a function
       **/
      bool IntegralNormalise(const Array<double, 1> &D_A1_XIn,
	                     const Array<double, 1> &D_A1_YIn,
			     Array<double, 1> &D_A1_YOut) const;

      /**
       * Integral-normalise a function
       **/
      bool IntegralNormalise(const Array<double, 1> &D_A1_XIn,
	                     Array<double, 1> &D_A1_YInOut) const;

      /**
       * Sort 6 coordinates to create hexagon
       * **/
      bool SortCoordinatesToCreateHexagon(const Array<double, 2> &D_A2_Coords_In,
                                          Array<double, 2> &D_A2_Coords_Out) const;

      /**
       * Sort Coordinates to create closed structure, starting with lower left corner
       * **/
      bool SortCoordinatesCounterClockWise(const Array<double, 2> &D_A2_Corners_In,
                                           Array<double, 2> &D_A2_Corners_Out) const;

      bool Remove_SubArrayFromArray(Array<int, 1> &A1_Array_InOut, const Array<int, 1> &A1_SubArray) const;
      
      bool Remove_ElementsFromArray(Array<int, 1> &I_A1_Array_InOut, const Array<int, 1> &I_A1_ElementIndicesToRemove_In) const;
      
      bool Remove_ElementsFromArray(Array<double, 1> &D_A1_Array_InOut, const Array<int, 1> &I_A1_ElementIndicesToRemove_In) const;
      
      bool Remove_Ith_ElementFromArray(Array<double, 1> &D_A1_Array_InOut,
                                       int I_Pos_In) const;

      bool Remove_Ith_ElementFromArray(Array<int, 1> &I_A1_Array_InOut,
                                       int I_Pos_In) const;

      bool Remove_Ith_RowFromArray(Array<double, 2> &D_A2_Array_InOut,
                                   int I_Row_In) const;

      bool Remove_Ith_RowFromArray(Array<int, 2> &I_A2_Array_InOut,
                                   int I_Row_In) const;

      /**
       * Returns distance between two points in 2D coordinate system
       * **/
      double Distance(const Array<double, 1> &D_A1_PointA_In,
                      const Array<double, 1> &D_A1_PointB_In) const;

      /**
       * Calculate end coordinate which extends the line from D_A1_StartPos_In through
       * D_A1_ThroughPos_In to length D_Length_In
       * **/
      bool ExtendLineToLength(const Array<double, 1> &D_A1_StartPos_In,
                              const Array<double, 1> &D_A1_ThroughPos_In,
                              const double D_Length_In,
                              Array<double, 1> &D_A1_EndPos_Out) const;

      /**
       * Returns TRUE if point D_A1_PixCoords_In is in triangular area defined by
       * D_A2_Triag_In((x0,y0),(x1,y1),(x2,y2))
       * **/
      bool PixelIsInTriangle(const Array<double, 1> &D_A1_PixCoords_In,
                             const Array<double, 2> &D_A2_Triag_In) const;

      /**
       * Returns TRUE if point D_A1_PixCoords_In is in rectangular area defined by
       * D_A2_Rect_In(pix_lower_left(x0,y0)
       *              pix_lower_right(x1,y0)
       *              pix_upper_right(x1,y1)
       *              pix_upper_left(x0,y1))
       * **/
      bool PixelIsInRectangle(const Array<double, 1> &D_A1_PixCoords_In,
                              const Array<double, 2> &D_A2_Rect_In) const;

      /**
       * Returns TRUE if point D_A1_PixCoords_In is in area defined by
       * D_A2_Rect_In(pix_lower_left(x0,y0)
       *              pix_lower_right(x1,y0)
       *              pix_upper_right(x1,y1)
       *              pix_upper_left(x0,y1))
       * **/
      bool PixelIsInFigure(const Array<double, 1> &D_A1_PixCoords_In,
                           const Array<double, 2> &D_A2_Rect_In) const;

      /**
       * Calculate m and n for line y=mx+n from 2 points D_A1_PointA_In(x,y) and
       * D_A1_PointB_In(x,y)
       * **/
      bool CalculateLine(const Array<double, 1> &D_A1_PointA_In,
                         const Array<double, 1> &D_A1_PointB_In,
                         double &D_M_Out,
                         double &D_N_Out) const;

      /**
       * Looks for crossing point of two given lines, defined by D_A2_LineA_In and
       * D_A2_LineB_In ((x1, y1)(x2,y2))
       * writes point to D_A1_Cross_Out and returns true, if found,
       * else returns false
       * **/
      bool FindCrossPoint(const Array<double, 2> &D_A2_LineA_In,
                          const Array<double, 2> &D_A2_LineB_In,
                          Array<double, 1> &D_A1_Cross_Out) const;

      bool FindApsInCircle(const int I_CenterX_In,
                           const int I_CenterY_In,
                           const int I_Radius_In,
                           Array<int, 1> &I_A1_Apertures_Out) const;

      bool FindApsInCircle(const int I_CenterX_In,
                           const int I_CenterY_In,
                           const int I_Radius_In,
                           const Array<int, 1> &I_A1_Area_In,
                           Array<int, 1> &I_A1_Apertures_Out) const;

      bool FindApsInRing(const int I_CenterX_In,
                         const int I_CenterY_In,
                         const int I_InnerRadius_In,
                         const int I_OuterRadius_In,
                         Array<int, 1> &I_A1_Apertures_Out) const;

      bool FindApsInRing(const int I_CenterX_In,
                         const int I_CenterY_In,
                         const int I_InnerRadius_In,
                         const int I_OuterRadius_In,
                         const Array<int, 1> &I_A1_Area,
                         Array<int, 1> &I_A1_Apertures_Out) const;

      /**
       * CalcOverlapFig
       * Calculates Figure from overlap of two given figures and writes its corner points
       * to D_A2_OverlapFig_Out
       * NOTE::Coordinates of both figures must be ordered counterclockwise
       * Returns TRUE if figures overlap, else returns FALSE
       * **/
      bool CalcOverlapFig(const Array<double, 2> &D_A2_FigA_In,
                          const Array<double, 2> &D_A2_FigB_In,
                          Array<double, 2> &D_A2_OverlapFig_Out) const;

      bool PixIsInArray(const Array<double, 1> &D_A1_Pix, const Array<double, 2> &D_A2_Corners) const;

      void ResizeAndPreserve(Array<double, 1> &D_A1_Arr_InOut, int I_NewSize);
      void ResizeAndPreserve(Array<double, 2> &D_A2_Arr_InOut, int I_NewRows, int I_NewCols);
      void ResizeAndPreserve(Array<int, 1> &I_A1_Arr_InOut, int I_NewSize);
      void ResizeAndPreserve(Array<int, 2> &I_A2_Arr_InOut, int I_NewRows, int I_NewCols);
      void ResizeAndPreserve(Array<CString, 1> &CS_A1_Arr_InOut, int I_NewSize);
      void ResizeAndPreserve(Array<CString, 2> &CS_A2_Arr_InOut, int I_NewRows, int I_NewCols);

//      bool GetBias(&D_BiasValue_Out) const;
      bool CreateErrorImage();

      bool AddHeaderToFitsFile(const CString &CS_FitsFileName_In, const CString &CS_HeaderFile_In) const;

      /// reverses order of elements in D_A1_InOut
      bool Reverse(Array<double, 1> D_A1_InOut) const;

      /// reverses order of rows in D_A1_InOut
      bool Reverse(Array<double, 2> D_A2_InOut) const;

      /// converts Number of Photons vs Wavelength to Flux vs Wavelength
      bool PhotonsToFlux(const Array<double, 2> &D_A2_WLen_NPhotons_In,
			 const double D_ExpTime_In,
			 const double D_ATel_In,
			 Array<double, 2> &D_A2_WLenFlux_Out) const;

      bool PhotonsToFlux(const Array<double, 2> &D_A2_NPhotons_In, /// NAps x NRows
			 const double D_ExpTime_In,                /// Exposure time
			 const double D_ATel_In,                   /// Telescope effective surface
			 const Array<int, 1> &I_A1_Apertures_In,   /// Apertures to convert
			 Array<double, 2> &D_A2_Flux_Out);         /// NAps x NRows

      bool CalculateLensletCorners(Array<double, 3> &D_A3_LensletCorners_Out);

      bool Rebin2D(const Array<double, 2> &D_A2_XYZ_In,///(NPoints, 3)
                   const Array<double, 1> &D_A1_XNew_In,///(NPointsNew)
                   const Array<double, 1> &D_A1_YNew_In,///(NPointsNew)
                   Array<double, 2> &D_A2_XYZ_Out,
                   int I_NNearestNeighbours) const;///(NPointsNew, 3)

      bool InterPol3D(const Array<double, 2> &D_A2_RefPoints,///(NPoints, 3)
                      const Array<double, 1> &D_A1_XY_In,
                      double &D_Z_Out) const;

      /// D_A2_Moving must be bigger than D_A2_Static
      /// D_A2_Static will be moved from the lower left edge around up to the upper right edge
      bool CrossCorrelate2D(const Array<double, 2> &D_A2_Static,
                            const Array<double, 2> &D_A2_Moving,
//                            const int &I_NPixMaxShift_X_Left,
//                            const int &I_NPixMaxShift_X_Right,
//                            const int &I_NPixMaxShift_Y_Down,
//                            const int &I_NPixMaxShift_Y_Up,
//                            const int &I_NSteps_X,
//                            const int &I_NSteps_Y,
                            int &I_XCor_X_Out,
                            int &I_XCor_Y_Out) const;

      bool GetApertureCenters(const Array<int, 1> &I_A1_Apertures_In,
                              Array<double, 2> &D_A2_ApertureCenters_Out) const;

      bool GetApertureCenters(Array<double, 2> &D_A2_ApertureCenters_Out) const;

      /**
       * Returns -1 if D_A1_Array_In is monotonically decreasing
       *         0  if D_A1_Array_In is non monotonic
       *         +1 if D_A1_Array_In is monotonically increasing
       */
      int IsMonotonic(const Array<double, 1> &D_A1_Array_In) const;
      
      /**
       * Calculates aperture minimum pixel, central position, and maximum pixel for aperture number I_Aperture_In,
       * and writes result to I_A2_MinCenMax_Out
       **/
      bool CalcMinCenMax(const int I_Aperture_In,
			 Array<int, 2> &I_A2_MinCenMax_Out) const;

      /**
       * Calculates Slit Function for each pixel in an aperture row from oversampled Slit Function D_A1_OSF_In,
       * and writes result to D_A1_SF_Out
       **/
      bool CalcSF(const int I_Aperture_In,
                  const int I_Row_In,
                  const Array<double, 1> &D_A1_OSF_In,
                  Array<double, 1> &D_A1_SF_Out) const;

      /**
       * Corrects a given input spectrum D_A2_Spec_In(col0: Wavelength, col1: spectrum) observed at 
       * airmass D_AirMass for the atmospheric extinction given in D_A2_Extinction_In(col0: Wavelength, col1: Extinction)
       * */
      bool RemoveAtmosphericExtinction(const Array<double, 2> &D_A2_Spec_In,
                                       const double &D_AirMass,
                                       const Array<double, 2> &D_A2_Extinction_In,
                                       Array<double, 2> &D_A2_Spec_Out) const;
      
      /* Calculate Airmass using Hardie Function
       * D_A1_ZenithDist_Deg_In: input zenith distances in degrees
       * */
      bool AirMass_Hardie_Deg(const Array<double, 1> &D_A1_ZenithDist_Deg_In,
                              Array<double, 1> &D_A1_AirMass_Out) const;
      
      /* Calculate Airmass using Hardie Function
       * D_A1_ZenithDist_Rad_In: input zenith distances in radians
       * */
      bool AirMass_Hardie_Rad(const Array<double, 1> &D_A1_ZenithDist_Rad_In,
                              Array<double, 1> &D_A1_AirMass_Out) const;
                                       
    #ifdef __WITH_PLOTS__
      bool ArrayToMGLArray(const Array<double, 1> &D_A1_Array_In,
                           mglData *P_D_A1_MGLArray_Out) const;
      bool ArrayToMGLArray(const Array<double, 2> &D_A2_Array_In,
                           mglData *P_D_A2_MGLArray_Out) const;
    #endif


  };
#endif
