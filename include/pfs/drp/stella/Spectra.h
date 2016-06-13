#if !defined(PFS_DRP_STELLA_SPECTRA_H)
#define PFS_DRP_STELLA_SPECTRA_H

#include <vector>
#include <iostream>
#include <cmath>
//#pragma clang diagnostic push
//#pragma clang diagnostic ignored "-Wunused-variable"
//#pragma clang diagnostic pop
#include "boost/algorithm/string/trim.hpp"
#include "lsst/base.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/pex/config.h"
#include "lsst/pex/exceptions.h"
#include "lsst/pex/logging.h"
//#include "lsst/daf/base/Citizen.h"
//#include "lsst/daf/base/Persistable.h"
#include <fitsio.h>
#include <fitsio2.h>
#include "math/Math.h"
#include "math/CurveFitting.h"
#include "utils/Utils.h"
#include "Controls.h"
#include "lsst/afw/fits.h"
#include "lsst/afw/image/fits/fits_io.h"
#include "lsst/afw/image/fits/fits_io_mpl.h"

#define stringify( name ) # name

//#define __DEBUG_IDENTIFY__
#define __DEBUG_SETLENGTH__
//#define __DEBUG_STRETCHANDCROSSCORRELATESPEC__
//#define __DEBUG_STRETCHANDCROSSCORRELATESPEC_LINELIST__
//#define __DEBUG_CREATELINELIST__

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
namespace pexExcept = lsst::pex::exceptions;

using namespace std;
namespace pfs { namespace drp { namespace stella {
/**
 * \brief Describe a single fiber trace
 */
template<typename SpectrumT, typename MaskT = afwImage::MaskPixel, typename VarianceT = afwImage::VariancePixel, typename WavelengthT = afwImage::VariancePixel>
class Spectrum {
  public:

    // Class Constructors and Destructor
    explicit Spectrum(size_t length = 0, 
                      size_t iTrace = 0);
    
    /// iTrace is only assigned to _iTrace if != 0, otherwise spectrum._iTrace is copied to this->_iTrace
    Spectrum(Spectrum const& spectrum,
                      size_t iTrace = 0)
        : _length(spectrum.getLength()),
          _iTrace(spectrum.getITrace()),
          _isWavelengthSet(spectrum.isWavelengthSet())
      {
        _spectrum = ndarray::allocate(spectrum.getSpectrum().getShape()[0]);
        _sky = ndarray::allocate(spectrum.getSky().getShape()[0]);
        _mask = ndarray::allocate(spectrum.getMask().getShape()[0]);
        _covar = ndarray::allocate(spectrum.getCovar().getShape());
        _wavelength = ndarray::allocate(spectrum.getWavelength().getShape()[0]);
        _dispersion = ndarray::allocate(spectrum.getDispersion().getShape()[0]);
        _spectrum.deep() = spectrum.getSpectrum();
        _sky.deep() = spectrum.getSky();
        _mask.deep() = spectrum.getMask();
        _covar.deep() = spectrum.getCovar();
        _wavelength.deep() = spectrum.getWavelength();
        _dispersion.deep() = spectrum.getDispersion();
        if (iTrace != 0)
          _iTrace = iTrace;
        _yLow = spectrum.getYLow();
        _yHigh = spectrum.getYHigh();
        _nCCDRows = spectrum.getNCCDRows();
      }
    
    ~Spectrum() {}

    /// Return a shared pointer to the spectrum
    ndarray::Array<SpectrumT, 1, 1> getSpectrum() { return _spectrum; }
    const ndarray::Array<SpectrumT, 1, 1> getSpectrum() const { return _spectrum; }

    /// Return a shared pointer to the sky spectrum
    ndarray::Array<SpectrumT, 1, 1> getSky() { return _sky; }
    const ndarray::Array<SpectrumT, 1, 1> getSky() const { return _sky; }
    
    /// Set the spectrum (deep copy)
    /// sets this->_spectrum to spectrum and returns TRUE if spectrum->size() == this->getLength(), otherwise returns false
    /// pre: set length of this to spectrum.size() to adjust length of all vectors in this
    bool setSpectrum( ndarray::Array< SpectrumT, 1, 1 > const& spectrum )
    {
      /// Check length of input spectrum
      if (spectrum.getShape()[0] != _length){
        string message("pfsDRPStella::Spectrum::setSpectrum: ERROR: spectrum->size()=");
        message += to_string(spectrum.getShape()[0]) + string(" != _length=") + to_string(_length);
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
      }
      _spectrum.deep() = spectrum;
      return true;
    }

    bool setSky( const ndarray::Array< SpectrumT, 1, 1 > & sky );
    
    /// Return the pointer to the variance of this spectrum
    ndarray::Array<VarianceT, 1, 1> getVariance() { return ndarray::Array<VarianceT, 1, 1 >(_covar[ ndarray::view( 3 )( ) ]); }
    const ndarray::Array<VarianceT, 1, 1> getVariance() const { return _covar[ ndarray::view( 3 )( ) ]; }
    
    /// Return the pointer to the covariance of this spectrum
    ndarray::Array<VarianceT, 2, 1> getCovar() { return _covar; }
    const ndarray::Array<VarianceT, 2, 1> getCovar() const { return _covar; }

    /// Set the covariance pointer of this fiber trace to covar (deep copy)
    /// sets this->_covar to covar and returns TRUE if covar->size() == this->getLength(), otherwise returns false
    bool setVariance( ndarray::Array<VarianceT, 1, 1> const& variance )
    {
      /// Check length of input covar
      if (variance.getShape()[ 0 ] != _length){
        string message("pfsDRPStella::Spectrum::setVariance: ERROR: variance->size()=");
        message += to_string( variance.getShape()[ 0 ] ) + string( " != _length=" ) + to_string( _length );
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
      }
      _covar[ ndarray::view( 3 )( ) ] = variance;
      return true;
    }

    /// Set the covariance pointer of this fiber trace to covar (deep copy)
    /// sets this->_covar to covar and returns TRUE if covar->size() == this->getLength(), otherwise returns false
    bool setCovar(const ndarray::Array<VarianceT, 2, 1> & covar );

    /// Return the pointer to the wavelength vector of this spectrum
    ndarray::Array<WavelengthT, 1, 1> getWavelength() { return _wavelength; }
    const ndarray::Array<WavelengthT, 1, 1> getWavelength() const { return _wavelength; }

    /// Return the pointer to the wavelength dispersion vector of this spectrum
    ndarray::Array<WavelengthT, 1, 1> getDispersion() { return _dispersion; }
    const ndarray::Array<WavelengthT, 1, 1> getDispersion() const { return _dispersion; }

    /// Set the wavelength vector of this spectrum (deep copy)
    /// sets this->_wavelength to wavelength and returns TRUE if wavelength->size() == this->getLength(), otherwise returns false
    bool setWavelength(const ndarray::Array<WavelengthT, 1, 1> & wavelength);

    /// Set the dispersion vector of this spectrum (deep copy)
    /// sets this->_dispersion to dispersion and returns TRUE if dispersion->size() == this->getLength(), otherwise returns false
    bool setDispersion(const ndarray::Array<WavelengthT, 1, 1> & dispersion);

    /// Return the pointer to the mask vector of this spectrum
    ndarray::Array<MaskT, 1, 1> getMask() { return _mask; }
    const ndarray::Array<MaskT, 1, 1> getMask() const { return _mask; }

    /// Set the mask vector of this spectrum (deep copy)
    /// sets this->_mask to mask and returns TRUE if mask->size() == this->getLength(), otherwise returns false
    bool setMask(const ndarray::Array<MaskT, 1, 1> & mask)
    {
      /// Check length of input mask
      if (mask.getShape()[0] != _length){
        string message("pfsDRPStella::Spectrum::setMask: ERROR: mask->size()=");
        message += to_string(mask.getShape()[0]) + string(" != _length=") + to_string(_length);
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
      }
      _mask.deep() = mask;
      return true;
    }

    size_t getLength() const {return _length;}
    
    /// Resize all vectors to size length.
    /// If length is smaller than the current container size, the contents of all vectors are reduced to their first length elements, 
    /// removing those beyond (and destroying them).
    /// If length is greater than the current container size, the contents of all vectors are expanded by inserting at the end as 
    /// many elements as needed to reach a size of length. The new elements of all vectors except for _wavelength are initialized 
    /// with 0. The new elements of _wavelength are initialized with _wavelength(_length - 1).
    bool setLength(const size_t length);
    
    size_t getITrace() const {return _iTrace;}
    void setITrace(size_t iTrace){_iTrace = iTrace;}
    
    ndarray::Array< double, 1, 1 > getDispCoeffs( ) const { return _dispCoeffs; };
    double getDispRms( ) const { return _dispRms; };

    /// Return _dispCorControl
    PTR(DispCorControl) getDispCorControl() const { return _dispCorControl; }
  
    /**
      * @brief: Identifies calibration lines, given in D_A2_LineList_In the format [wlen, approx_pixel] in
      * wavelength-calibration spectrum D_A2_Spec_In [pixel_number, flux]
      * within the given position plus/minus I_Radius_In,
      * fits Gaussians to each line, fits Polynomial of order I_PolyFitOrder_In, and
      * writes _wavelength and PolyFit coefficients to _dispCoeffs
      **/
    template< typename T >
    bool identify( ndarray::Array< T, 2, 1 > const& lineList,
                   DispCorControl const& dispCorControl,
                   size_t nLinesCheck = 0 );
  
    /**
      * @brief: Identifies calibration lines, given in D_A2_LineList_In the format [wlen, approx_pixel] in
      * wavelength-calibration spectrum D_A2_Spec_In [pixel_number, flux]
      * within the given position plus/minus I_Radius_In,
      * fits Gaussians to each line, fits Polynomial of order I_PolyFitOrder_In, and
      * writes _wavelength and PolyFit coefficients to _dispCoeffs
      * @param lineList             :: [ nLines, 2 ]: [ wLen, approx_pixel ]
      * @param predicted            :: [ nLines ]: predicted pixel position for each line from the zemax model
      * @param predictedWLenAllPix  :: [ nRows in spectrum (yHigh - yLow): predicted wavelength from the zemax model
      * @param dispCorControl       :: parameters controlling the dispersion fitting
      * @param nLinesCheck          :: hold back this many lines from the fit to check the quality of the wavelength solution
      **/
    template< typename T >
    bool identify( ndarray::Array< T, 2, 1 > const& lineList,
                   ndarray::Array< T, 1, 0 > const& predicted,
                   ndarray::Array< T, 1, 0 > const& predictedWLenAllPix,
                   DispCorControl const& dispCorControl,
                   size_t nLinesCheck = 0 );
    
    bool isWavelengthSet() const {return _isWavelengthSet;}
//    void setIsWavelengthSet(bool isWavelengthSet) {_isWavelengthSet = isWavelengthSet;}
    
    size_t getYLow() const { return _yLow; };
    size_t getYHigh() const { return _yHigh; };
    size_t getNCCDRows() const { return _nCCDRows; };
    
    bool setYLow( const size_t yLow )
    {
      if ( yLow > _nCCDRows ){
        string message("pfsDRPStella::Spectrum::setYLow: ERROR: yLow=");
        message += to_string( yLow ) + string(" > _nCCDRows=") + to_string(_nCCDRows);
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
      }
      _yLow = yLow;
      return true;
    }

    bool setYHigh( const size_t yHigh )
    {
      if ( yHigh > _nCCDRows ){
        _nCCDRows = _yLow + yHigh;
      }
      _yHigh = yHigh;
      return true;
    }
    
    bool setNCCDRows( const size_t nCCDRows ){
      if ( _yLow > nCCDRows ){
        string message("pfsDRPStella::Spectrum::setYLow: ERROR: _yLow=");
        message += to_string( _yLow ) + string(" > nCCDRows=") + to_string(nCCDRows);
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
      }
      _nCCDRows = nCCDRows;
      return true;
    }

    
  private:
    /**
     * @brief: Returns pixel positions of emission lines in lineList fitted in _spectrum
     * @param[in] lineList :: line list  [ nLines, 2 ]: [ wLen, approx_pixel ]
     */
    template< typename T >
    ndarray::Array< double, 1, 1 > hIdentify( ndarray::Array< T, 2, 1 > const& lineList )
    {
      ///for each line in line list, find maximum in spectrum and fit Gaussian
      int I_MaxPos = 0;
      int I_Start = 0;
      int I_End = 0;
      int I_NTerms = 4;
      std::vector< double > V_GaussSpec( 1, 0. );
      ndarray::Array< double, 1, 1 > D_A1_GaussCoeffs = ndarray::allocate( I_NTerms );
      D_A1_GaussCoeffs.deep() = 0.;
      ndarray::Array< double, 1, 1 > D_A1_EGaussCoeffs = ndarray::allocate( I_NTerms );
      D_A1_EGaussCoeffs.deep() = 0.;
      ndarray::Array< int, 2, 1 > I_A2_Limited = ndarray::allocate( I_NTerms, 2 );
      I_A2_Limited.deep() = 1;
      ndarray::Array< double, 2, 1 > D_A2_Limits = ndarray::allocate( I_NTerms, 2 );
      D_A2_Limits.deep() = 0.;
      ndarray::Array< double, 1, 1 > D_A1_Guess = ndarray::allocate( I_NTerms );
      std::vector< double > V_MeasureErrors( 2, 0.);
      ndarray::Array< double, 1, 1 > D_A1_Ind = math::indGenNdArr( double( _spectrum.getShape()[ 0 ] ) );
      std::vector< double > V_X( 1, 0. );
      ndarray::Array< double, 1, 1 > D_A1_GaussPos = ndarray::allocate( lineList.getShape()[0] );
      D_A1_GaussPos.deep() = 0.;
      #ifdef __WITH_PLOTS__
        CString CS_PlotName("");
        CString *P_CS_Num;
      #endif
      for ( int i_line = 0; i_line < lineList.getShape()[ 0 ]; ++i_line ){
        I_Start = int( lineList[ ndarray::makeVector( i_line, 1 ) ] ) - _dispCorControl->searchRadius;
        if ( I_Start < 0 )
          I_Start = 0;
        #ifdef __DEBUG_IDENTIFY__
          cout << "identify: i_line = " << i_line << ": I_Start = " << I_Start << endl;
        #endif
        I_End = int( lineList[ ndarray::makeVector( i_line, 1 ) ] ) + _dispCorControl->searchRadius;
        if ( I_End >= _spectrum.getShape()[ 0 ] )
          I_End = _spectrum.getShape()[ 0 ] - 1;
        if ( ( I_End - I_Start ) > ( 1.5 * _dispCorControl->searchRadius ) ){
          #ifdef __DEBUG_IDENTIFY__
            cout << "identify: i_line = " << i_line << ": I_End = " << I_End << endl;
          #endif
          if ( I_Start >= I_End ){
            cout << "identify: Warning: I_Start(=" << I_Start << ") >= I_End(=" << I_End << ")" << endl;// => Returning FALSE" << endl;
            cout << "identify: _spectrum = " << _spectrum << endl;
            cout << "identify: lineList = " << lineList << endl;
          }
          else{
            auto itMaxElement = std::max_element( _spectrum.begin() + I_Start, _spectrum.begin() + I_End + 1 );
            I_MaxPos = std::distance(_spectrum.begin(), itMaxElement);
      //        #ifdef __DEBUG_IDENTIFY__
      //          cout << "identify: i_line = " << i_line << ": indexPos = " << indexPos << endl;
      //        #endif
      //        I_MaxPos = indexPos;// + I_Start;
            #ifdef __DEBUG_IDENTIFY__
              cout << "identify: I_MaxPos = " << I_MaxPos << endl;
            #endif
            I_Start = std::round( double( I_MaxPos ) - ( 1.5 * _dispCorControl->fwhm ) );
            if (I_Start < 0)
              I_Start = 0;
            #ifdef __DEBUG_IDENTIFY__
              cout << "identify: I_Start = " << I_Start << endl;
            #endif
            I_End = std::round( double( I_MaxPos ) + ( 1.5 * _dispCorControl->fwhm ) );
            if ( I_End >= _spectrum.getShape()[ 0 ] )
              I_End = _spectrum.getShape()[ 0 ] - 1;
            #ifdef __DEBUG_IDENTIFY__
              cout << "identify: I_End = " << I_End << endl;
            #endif
            if ( I_End < I_Start + 4 ){
              cout << "identify: WARNING: Line position outside spectrum" << endl;
            }
            else{
              V_GaussSpec.resize( I_End - I_Start + 1 );
              V_MeasureErrors.resize( I_End - I_Start + 1 );
              V_X.resize( I_End - I_Start + 1 );
              auto itSpec = _spectrum.begin() + I_Start;
              for ( auto itGaussSpec = V_GaussSpec.begin(); itGaussSpec != V_GaussSpec.end(); ++itGaussSpec, ++itSpec )
                *itGaussSpec = *itSpec;
              #ifdef __DEBUG_IDENTIFY__
                cout << "identify: V_GaussSpec = ";
                for ( int iPos = 0; iPos < V_GaussSpec.size(); ++iPos )
                  cout << V_GaussSpec[iPos] << " ";
                cout << endl;
              #endif
              for( auto itMeasErr = V_MeasureErrors.begin(), itGaussSpec = V_GaussSpec.begin(); itMeasErr != V_MeasureErrors.end(); ++itMeasErr, ++itGaussSpec ){
                *itMeasErr = sqrt( std::fabs( *itGaussSpec ) );
                if (*itMeasErr < 0.00001)
                  *itMeasErr = 1.;
              }
              #ifdef __DEBUG_IDENTIFY__
                cout << "identify: V_MeasureErrors = ";
                for (int iPos = 0; iPos < V_MeasureErrors.size(); ++iPos )
                  cout << V_MeasureErrors[iPos] << " ";
                cout << endl;
              #endif
              auto itInd = D_A1_Ind.begin() + I_Start;
              for ( auto itX = V_X.begin(); itX != V_X.end(); ++itX, ++itInd )
                *itX = *itInd;
              #ifdef __DEBUG_IDENTIFY__
                cout << "identify: V_X = ";
                for (int iPos = 0; iPos < V_X.size(); ++iPos )
                  cout << V_X[iPos] << " ";
                cout << endl;
              #endif
      //        if (!this->GaussFit(D_A1_X,
      //                            D_A1_GaussSpec,
      //                            D_A1_GaussCoeffs,
      //                            CS_A1_KeyWords,
      //                            PP_Args)){

            /*     p[3] = constant offset
             *     p[0] = peak y value
             *     p[1] = x centroid position
             *     p[2] = gaussian sigma width
             */
      //          ndarray::Array< double, 2, 1 > toFit = ndarray::allocate( D_A1_X.getShape()[ 0 ], 2 );
      //          toFit[ ndarray::view()(0) ] = D_A1_X;
      //          toFit[ ndarray::view()(1) ] = D_A1_GaussSpec;
      //            ndarray::Array< double, 1, 1 > gaussFitResult = gaussFit()
              D_A1_Guess[ 3 ] = *min_element( V_GaussSpec.begin(), V_GaussSpec.end() );
              D_A1_Guess[ 0 ] = *max_element( V_GaussSpec.begin(), V_GaussSpec.end() ) - D_A1_Guess(3);
              D_A1_Guess[ 1 ] = V_X[ 0 ] + ( V_X[ V_X.size() - 1 ] - V_X[ 0 ] ) / 2.;
              D_A1_Guess[ 2 ] = _dispCorControl->fwhm;
              #ifdef __DEBUG_IDENTIFY__
                cout << "identify: D_A1_Guess = " << D_A1_Guess << endl;
              #endif
              D_A2_Limits[ ndarray::makeVector( 0, 0 ) ] = 0.;
              D_A2_Limits[ ndarray::makeVector( 0, 1 ) ] = std::fabs( 1.5 * D_A1_Guess[ 0 ] );
              D_A2_Limits[ ndarray::makeVector( 1, 0 ) ] = V_X[ 1 ];
              D_A2_Limits[ ndarray::makeVector( 1, 1 ) ] = V_X[ V_X.size() - 2 ];
              D_A2_Limits[ ndarray::makeVector( 2, 0 ) ] = D_A1_Guess[ 2 ] / 3.;
              D_A2_Limits[ ndarray::makeVector( 2, 1 ) ] = 2. * D_A1_Guess[ 2 ];
              D_A2_Limits[ ndarray::makeVector( 3, 1 ) ] = std::fabs( 1.5 * D_A1_Guess[ 3 ] ) + 1;
              #ifdef __DEBUG_IDENTIFY__
                cout << "identify: D_A2_Limits = " << D_A2_Limits << endl;
              #endif
              ndarray::Array< double, 1, 1 > D_A1_X = ndarray::external( V_X.data(), ndarray::makeVector( int( V_X.size() ) ), ndarray::makeVector( 1 ) );
              ndarray::Array< double, 1, 1 > D_A1_GaussSpec = ndarray::external( V_GaussSpec.data(), ndarray::makeVector( int( V_GaussSpec.size() ) ), ndarray::makeVector( 1 ) );
              ndarray::Array< double, 1, 1 > D_A1_MeasureErrors = ndarray::external( V_MeasureErrors.data(), ndarray::makeVector( int( V_MeasureErrors.size() ) ), ndarray::makeVector( 1 ) );
              if (!MPFitGaussLim(D_A1_X,
                                 D_A1_GaussSpec,
                                 D_A1_MeasureErrors,
                                 D_A1_Guess,
                                 I_A2_Limited,
                                 D_A2_Limits,
                                 true,
                                 false,
                                 D_A1_GaussCoeffs,
                                 D_A1_EGaussCoeffs,
                                 true)){
                cout << "identify: WARNING: GaussFit returned FALSE" << endl;
              //        return false;
              }
              else{
                #ifdef __DEBUG_IDENTIFY__
                  cout << "identify: i_line = " << i_line << ": D_A1_GaussCoeffs = " << D_A1_GaussCoeffs << endl;
                #endif
                if ( std::fabs( double( I_MaxPos ) - D_A1_GaussCoeffs[ 1 ] ) < 2.5 ){//D_FWHM_In){
                  D_A1_GaussPos[ i_line ] = D_A1_GaussCoeffs[ 1 ];
                  #ifdef __DEBUG_IDENTIFY__
                    cout << "identify: D_A1_GaussPos[" << i_line << "] = " << D_A1_GaussPos[ i_line ] << endl;
                  #endif
                  if ( i_line > 0 ){
                    if ( std::fabs( D_A1_GaussPos[ i_line ] - D_A1_GaussPos[ i_line - 1 ] ) < 1.5 ){/// wrong line identified!
                      if ( lineList.getShape()[ 1 ] > 2 ){
                        if ( lineList[ ndarray::makeVector( i_line, 2 ) ] < lineList[ ndarray::makeVector( i_line - 1, 2 ) ] ){
                          cout << "identify: WARNING: i_line=" << i_line << ": line " << i_line << " at " << D_A1_GaussPos[ i_line ] << " has probably been misidentified (D_A1_GaussPos(" << i_line-1 << ")=" << D_A1_GaussPos[ i_line - 1 ] << ") => removing line from line list" << endl;
                          D_A1_GaussPos[ i_line ] = 0.;
                        }
                        else{
                          cout << "identify: WARNING: i_line=" << i_line << ": line at D_A1_GaussPos[" << i_line-1 << "] = " << D_A1_GaussPos[ i_line - 1 ] << " has probably been misidentified (D_A1_GaussPos(" << i_line << ")=" << D_A1_GaussPos[ i_line ] << ") => removing line from line list" << endl;
                          D_A1_GaussPos[ i_line - 1 ] = 0.;
                        }
      //                  exit(EXIT_FAILURE);
                      }
                    }
                  }
                }
                else{
                  cout << "identify: WARNING: I_MaxPos=" << I_MaxPos << " - D_A1_GaussCoeffs[ 1 ]=" << D_A1_GaussCoeffs[ 1 ] << " >= 2.5 => Skipping line" << endl;
                }
              }
            }
          }
        }
      }/// end for (int i_line=0; i_line < D_A2_LineList_In.rows(); i_line++){
      return D_A1_GaussPos;
    }

    size_t _yLow;
    size_t _yHigh;
    size_t _length;
    size_t _nCCDRows;
    ndarray::Array<SpectrumT, 1, 1> _spectrum;
    ndarray::Array<SpectrumT, 1, 1> _sky;
    ndarray::Array<MaskT, 1, 1> _mask;/// 0: all pixels of the wavelength element used for extraction were okay
                                  /// 1: at least one pixel was not okay but the extraction algorithm believes it could fix it
                                  /// 2: at least one pixel was problematic
    ndarray::Array<VarianceT, 2, 1> _covar;
    ndarray::Array<WavelengthT, 1, 1> _wavelength;
    ndarray::Array<WavelengthT, 1, 1> _dispersion;
    size_t _iTrace;/// for logging / debugging purposes only
    ndarray::Array< double, 1, 1 > _dispCoeffs;
    double _dispRms;
    bool _isWavelengthSet;
    PTR(DispCorControl) _dispCorControl;

  protected:
};

/************************************************************************************************************/
/**
 * \brief Describe a set of spectra
 *
 */
template<typename SpectrumT, typename MaskT = afwImage::MaskPixel, typename VarianceT = afwImage::VariancePixel, typename WavelengthT = afwImage::VariancePixel>
class SpectrumSet// : public lsst::daf::base::Persistable,
                 //public lsst::daf::base::Citizen 
{
  public:
    /// Class Constructors and Destructor
      
    /// Creates a new SpectrumSet object of size 0
    explicit SpectrumSet(size_t nSpectra=0, size_t length=0);
        
    /// Copy constructor
    /// If spectrumSet is not empty, the object shares ownership of spectrumSet's spectrum vector and increases the use count.
    /// If spectrumSet is empty, an empty object is constructed (as if default-constructed).
    SpectrumSet( SpectrumSet const& spectrumSet)
        ://     lsst::daf::base::Citizen(typeid(this)),
              _spectra(spectrumSet.getSpectra())
        {}

    /// Construct an object with a copy of spectrumVector
    explicit SpectrumSet( std::vector< Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT > > const& spectrumVector);

    /**
     *  @brief Construct a SpectrumSet by reading a regular FITS file.
     *
     *  @param[in]      fileName      File to read.
     *  @param[in,out]  metadata      Metadata read from the primary HDU header.
     *  @param[in,out]  fluxMetadata  Metadata read from the flux HDU header.
     *  @param[in,out]  covarMetadata Metadata read from the covar HDU header.
     *  @param[in,out]  maskMetadata  Metadata read from the mask HDU header.
     *  @param[in,out]  wLenMetadata  Metadata read from the wavelength HDU header.
     *  @param[in,out]  wDispMetadata Metadata read from the dispersion HDU header.
     *  @param[in,out]  skyMetadata   Metadata read from the sky HDU header.
     */
    explicit SpectrumSet(
        std::string const& fileName,
        PTR(lsst::daf::base::PropertySet) metadata=PTR(lsst::daf::base::PropertySet)(),
        PTR(lsst::daf::base::PropertySet) fluxMetadata=PTR(lsst::daf::base::PropertySet)(),
        PTR(lsst::daf::base::PropertySet) covarMetadata=PTR(lsst::daf::base::PropertySet)(),
        PTR(lsst::daf::base::PropertySet) maskMetadata=PTR(lsst::daf::base::PropertySet)(),
        PTR(lsst::daf::base::PropertySet) wLenMetadata=PTR(lsst::daf::base::PropertySet)(),
        PTR(lsst::daf::base::PropertySet) wDispMetadata=PTR(lsst::daf::base::PropertySet)(),
        PTR(lsst::daf::base::PropertySet) skyMetadata=PTR(lsst::daf::base::PropertySet)()
    );

    /**
     *  @brief Construct a SpectrumSet by reading a FITS image in memory.
     *
     *  @param[in]      manager       An object that manages the memory buffer to read.
     *  @param[in,out]  metadata      Metadata read from the primary HDU header.
     *  @param[in,out]  fluxMetadata  Metadata read from the flux HDU header.
     *  @param[in,out]  covarMetadata Metadata read from the covar HDU header.
     *  @param[in,out]  maskMetadata  Metadata read from the mask HDU header.
     *  @param[in,out]  wLenMetadata  Metadata read from the wavelength HDU header.
     *  @param[in,out]  wDispMetadata Metadata read from the dispersion HDU header.
     *  @param[in,out]  skyMetadata   Metadata read from the sky HDU header.
     */
    explicit SpectrumSet(
        lsst::afw::fits::MemFileManager const& manager,
        PTR(lsst::daf::base::PropertySet) metadata=PTR(lsst::daf::base::PropertySet)(),
        PTR(lsst::daf::base::PropertySet) fluxMetadata=PTR(lsst::daf::base::PropertySet)(),
        PTR(lsst::daf::base::PropertySet) covarMetadata=PTR(lsst::daf::base::PropertySet)(),
        PTR(lsst::daf::base::PropertySet) maskMetadata=PTR(lsst::daf::base::PropertySet)(),
        PTR(lsst::daf::base::PropertySet) wLenMetadata=PTR(lsst::daf::base::PropertySet)(),
        PTR(lsst::daf::base::PropertySet) wDispMetadata=PTR(lsst::daf::base::PropertySet)(),
        PTR(lsst::daf::base::PropertySet) skyMetadata=PTR(lsst::daf::base::PropertySet)()
    );

    /**
     *  @brief Construct a SpectrumSet from an already-open FITS object.
     *
     *  @param[in]      fitsfile      A FITS object to read from.  Current HDU is ignored.
     *  @param[in,out]  metadata      Metadata read from the primary HDU header.
     *  @param[in,out]  fluxMetadata  Metadata read from the flux HDU header.
     *  @param[in,out]  covarMetadata Metadata read from the covar HDU header.
     *  @param[in,out]  maskMetadata  Metadata read from the mask HDU header.
     *  @param[in,out]  wLenMetadata  Metadata read from the wavelength HDU header.
     *  @param[in,out]  wDispMetadata Metadata read from the dispersion HDU header.
     *  @param[in,out]  skyMetadata   Metadata read from the sky HDU header.
     */
    explicit SpectrumSet(
        lsst::afw::fits::Fits const& fitsfile,
        PTR(lsst::daf::base::PropertySet) metadata=PTR(lsst::daf::base::PropertySet)(),
        PTR(lsst::daf::base::PropertySet) fluxMetadata=PTR(lsst::daf::base::PropertySet)(),
        PTR(lsst::daf::base::PropertySet) covarMetadata=PTR(lsst::daf::base::PropertySet)(),
        PTR(lsst::daf::base::PropertySet) maskMetadata=PTR(lsst::daf::base::PropertySet)(),
        PTR(lsst::daf::base::PropertySet) wLenMetadata=PTR(lsst::daf::base::PropertySet)(),
        PTR(lsst::daf::base::PropertySet) wDispMetadata=PTR(lsst::daf::base::PropertySet)(),
        PTR(lsst::daf::base::PropertySet) skyMetadata=PTR(lsst::daf::base::PropertySet)()
    );
        
    virtual ~SpectrumSet() {}

    /// Return the number of spectra/apertures
    size_t size() const { return _spectra.size(); }

    /// Return the Spectrum for the ith aperture
    PTR(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>) getSpectrum(const size_t i);

    Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT> getSpectrum(const size_t i) const
    { 
      if (i >= _spectra.size()){
        string message("SpectrumSet::getSpectrum(i=");
        message += to_string(i) + "): ERROR: i >= _spectra.size()=" + to_string(_spectra.size());
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      return _spectra.at( i );
    }

    /**
     * @brief Set the ith Spectrum
     * @param i :: Set which spectrum in set?
     * @param spectrum :: spectrum to copy to this set at position i
     * */
    bool setSpectrum(size_t const i,
                     Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT > const& spectrum);

    /// add one Spectrum to the set
    void addSpectrum( Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT > const& spectrum ) {
        _spectra.push_back( spectrum );
    }
    
    void addSpectrum( PTR( Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT > ) const& spectrum ){
        _spectra.push_back( *spectrum );
    }

    std::vector< Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT > > getSpectra() const { return _spectra; }

    
    /// Removes from the vector either a single element (position) or a range of elements ([first,last)).
    /// This effectively reduces the container size by the number of elements removed, which are destroyed.
    bool erase(const size_t iStart, const size_t iEnd=0);
    
    /**
     *  @brief Write a FITS binary table to an open file object.
     *
     *  @param[in,out] fitsfile Fits file object to write to.
     *  @param[in] flags        Table-subclass-dependent bitflags that control the details of how to
     *                          read the catalogs.  See e.g. SourceFitsFlags.
     */
    void writeFits(lsst::afw::fits::Fits & fitsfile, int flags=0) const;
    
    /**
     *  @brief Write a FITS binary table to an open file object.
     *
     *  @param[in] fileName     Fits file object to write to.
     *  @param[in] flags        Table-subclass-dependent bitflags that control the details of how to
     *                          read the catalogs.  See e.g. SourceFitsFlags.
     */
    void writeFits( std::string const& fileName, int flags=0) const;

    /**
     *  @brief Write a MaskedImage to a FITS RAM file.
     *
     *  @param[in] manager       Manager object for the memory block to write to.
     *  @param[in] flags        Table-subclass-dependent bitflags that control the details of how to
     *                          read the catalogs.  See e.g. SourceFitsFlags.
     *
     *  The FITS file will have four HDUs; the primary HDU will contain only metadata,
     *  while the image, mask, and variance HDU headers will use the "INHERIT='T'" convention
     *  to indicate that the primary metadata applies to those HDUs as well.
     */
    void writeFits( lsst::afw::fits::MemFileManager & manager, int flags=0 ) const;

    /**
     *  @brief Read a SpectrumSet from a regular FITS file.
     *
     *  @param[in] filename    Name of the file to read.
     */
    static SpectrumSet& readFits( std::string const & filename ) {
        static SpectrumSet< SpectrumT, MaskT, VarianceT, WavelengthT > spectrumSet( filename );
        return spectrumSet;
    }

    /**
     *  @brief Read a MaskedImage from a FITS RAM file.
     *
     *  @param[in] manager     Object that manages the memory to be read.
     */
    static SpectrumSet& readFits( lsst::afw::fits::MemFileManager & manager ) {
        static SpectrumSet< SpectrumT, MaskT, VarianceT, WavelengthT > spectrumSet( manager );
        return spectrumSet;
    }
    
  private:
    std::vector< Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT > > _spectra; // spectra for each aperture
};

namespace math{
    
    template< typename T, typename U >
    struct StretchAndCrossCorrelateSpecResult{
        ndarray::Array< U, 2, 1 > lineList;
//        #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC__
            ndarray::Array< T, 3, 1 > specPieces;
//        #endif
    };
    
    template< typename T, typename U >
    StretchAndCrossCorrelateSpecResult< T, U > stretchAndCrossCorrelateSpec( ndarray::Array< T, 1, 1 > const& spec,
                                                                             ndarray::Array< T, 1, 1 > const& specRef,
                                                                             ndarray::Array< U, 2, 1 > const& lineList_WLenPix,
                                                                             DispCorControl const& dispCorControl );
    
    /**
     * @brief: create line list from wavelength array of size nCCDRows and list of wavelengths of emission lines used to calibrate the spectrum
     * @param wLen
     * @param lines
     * @return array(lines.shape[0], 2) col 0: wavelength, col 1: pixel
     */
    template< typename T >
    ndarray::Array< T, 2, 1 > createLineList( ndarray::Array< T, 1, 1 > const& wLen,
                                              ndarray::Array< T, 1, 1 > const& lines );

}

}}}
#endif