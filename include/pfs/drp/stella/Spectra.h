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
                      size_t iTrace = 0);
    
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
    bool setSpectrum( ndarray::Array< SpectrumT, 1, 1 > const& spectrum );

    bool setSky( ndarray::Array< SpectrumT, 1, 1 > const& sky );
    
    /// Return the pointer to the variance of this spectrum
    ndarray::Array<VarianceT, 1, 1> getVariance() { return ndarray::Array<VarianceT, 1, 1 >(_covar[ ndarray::view( 3 )( ) ]); }
    const ndarray::Array<VarianceT, 1, 1> getVariance() const { return _covar[ ndarray::view( 3 )( ) ]; }
    
    /// Return the pointer to the covariance of this spectrum
    ndarray::Array<VarianceT, 2, 1> getCovar() { return _covar; }
    const ndarray::Array<VarianceT, 2, 1> getCovar() const { return _covar; }

    /// Set the covariance pointer of this fiber trace to covar (deep copy)
    /// sets this->_covar to covar and returns TRUE if covar->size() == this->getLength(), otherwise returns false
    bool setVariance( ndarray::Array<VarianceT, 1, 1> const& variance );

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
    bool setWavelength( ndarray::Array<WavelengthT, 1, 1> const& wavelength );

    /// Set the dispersion vector of this spectrum (deep copy)
    /// sets this->_dispersion to dispersion and returns TRUE if dispersion->size() == this->getLength(), otherwise returns false
    bool setDispersion( ndarray::Array<WavelengthT, 1, 1> const& dispersion );

    /// Return the pointer to the mask vector of this spectrum
    ndarray::Array<MaskT, 1, 1> getMask() { return _mask; }
    const ndarray::Array<MaskT, 1, 1> getMask() const { return _mask; }

    /// Set the mask vector of this spectrum (deep copy)
    /// sets this->_mask to mask and returns TRUE if mask->size() == this->getLength(), otherwise returns false
    bool setMask(const ndarray::Array<MaskT, 1, 1> & mask);

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
                   size_t nLinesCheck = 0 ){
        DispCorControl tempDispCorControl( dispCorControl );
        _dispCorControl.reset();
        _dispCorControl = tempDispCorControl.getPointer();

        ///for each line in line list, find maximum in spectrum and fit Gaussian
        ndarray::Array< double, 1, 1 > D_A1_GaussPos = hIdentify( lineList );

        ///remove lines which could not be found from line list
        std::vector< int > V_Index( D_A1_GaussPos.getShape()[ 0 ], 0 );
        size_t pos = 0;
        for (auto it = D_A1_GaussPos.begin(); it != D_A1_GaussPos.end(); ++it, ++pos ){
          if ( *it > 0. )
            V_Index[ pos ] = 1;
        }
        #ifdef __DEBUG_IDENTIFY__
          cout << "identify: D_A1_GaussPos = " << D_A1_GaussPos << endl;
          cout << "identify: V_Index = ";
          for (int iPos = 0; iPos < V_Index.size(); ++iPos)
            cout << V_Index[iPos] << " ";
          cout << endl;
        #endif
        std::vector< size_t > indices = math::getIndices( V_Index );
        size_t nInd = std::accumulate( V_Index.begin(), V_Index.end(), 0 );
        #ifdef __DEBUG_IDENTIFY__
          cout << "Identify: " << nInd << " lines identified" << endl;
          cout << "Identify: indices = ";
          for (int iPos = 0; iPos < indices.size(); ++iPos )
            cout << indices[iPos] << " ";
          cout << endl;
        #endif

        /// separate lines to fit and lines for RMS test
        std::vector< size_t > indCheck;
        for ( size_t i = 0; i < nLinesCheck; ++i ){
          srand( 0 ); //seed initialization
          int randNum = rand() % ( indices.size() - 2 ) + 1; // Generate a random number between 0 and 1
          indCheck.push_back( size_t( randNum ) );
          indices.erase( indices.begin() + randNum );
        }

        if ( nInd < ( std::round( double( lineList.getShape()[ 0 ] ) * 0.66 ) ) ){
          std::string message("pfs::drp::stella::identify: ERROR: ");
          message += "identify: ERROR: less than " + std::to_string( std::round( double( lineList.getShape()[ 0 ] ) * 0.66 ) ) + " lines identified";
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        ndarray::Array< size_t, 1, 1 > I_A1_IndexPos = ndarray::external( indices.data(), ndarray::makeVector( int( indices.size() ) ), ndarray::makeVector( 1 ) );
        ndarray::Array< double, 1, 1 > D_A1_WLen = ndarray::allocate( lineList.getShape()[ 0 ] );
        ndarray::Array< double, 1, 1 > D_A1_FittedPos = math::getSubArray( D_A1_GaussPos, 
                                                                           I_A1_IndexPos );
        ndarray::Array< size_t, 1, 1 > I_A1_IndexCheckPos = ndarray::external( indCheck.data(), ndarray::makeVector( int( indCheck.size() ) ), ndarray::makeVector( 1 ) );
        ndarray::Array< double, 1, 1 > D_A1_FittedCheckPos = math::getSubArray( D_A1_GaussPos, 
                                                                                I_A1_IndexCheckPos );
        #ifdef __DEBUG_IDENTIFY__
          cout << "identify: D_A1_FittedPos = " << D_A1_FittedPos << endl;
        #endif

        D_A1_WLen[ ndarray::view() ] = lineList[ ndarray::view()( 0 ) ];
        ndarray::Array< double, 1, 1 > D_A1_FittedWLen = math::getSubArray( D_A1_WLen, I_A1_IndexPos );
        cout << "Identify: found D_A1_FittedWLen = " << D_A1_FittedWLen << endl;

        ndarray::Array< double, 1, 1 > D_A1_FittedWLenCheck = math::getSubArray( D_A1_WLen, I_A1_IndexCheckPos );

        _dispCoeffs = ndarray::allocate( dispCorControl.order + 1 );
        _dispCoeffs.deep() = math::PolyFit( D_A1_FittedPos,
                                            D_A1_FittedWLen,
                                            dispCorControl.order );
        ndarray::Array< double, 1, 1 > D_A1_WLen_Gauss = math::Poly( D_A1_FittedPos, 
                                                                     _dispCoeffs );
        ndarray::Array< double, 1, 1 > D_A1_WLen_GaussCheck = math::Poly( D_A1_FittedCheckPos, 
                                                                          _dispCoeffs );
        cout << "Identify: D_A1_WLen_PolyFit = " << D_A1_WLen_Gauss << endl;
        cout << "identify: _dispCoeffs = " << _dispCoeffs << endl;

        ///Calculate RMS
        ndarray::Array< double, 1, 1 > D_A1_WLenMinusFit = ndarray::allocate( D_A1_WLen_Gauss.getShape()[ 0 ] );
        D_A1_WLenMinusFit.deep() = D_A1_FittedWLen - D_A1_WLen_Gauss;
        cout << "Identify: D_A1_WLenMinusFit = " << D_A1_WLenMinusFit << endl;
        _dispRms = math::calcRMS( D_A1_WLenMinusFit );
        cout << "Identify: _dispRms = " << _dispRms << endl;
        cout << "======================================" << endl;

        ///Calculate RMS for test lines
        ndarray::Array< double, 1, 1 > D_A1_WLenMinusFitCheck = ndarray::allocate( D_A1_WLen_GaussCheck.getShape()[ 0 ] );
        D_A1_WLenMinusFitCheck.deep() = D_A1_FittedWLenCheck - D_A1_WLen_GaussCheck;
        cout << "Identify: D_A1_WLenMinusFitCheck = " << D_A1_WLenMinusFitCheck << endl;
        double dispRmsCheck = math::calcRMS( D_A1_WLenMinusFitCheck );
        cout << "Identify: dispRmsCheck = " << dispRmsCheck << endl;
        cout << "======================================" << endl;

        ///calibrate spectrum
        ndarray::Array< double, 1, 1 > D_A1_Indices = math::indGenNdArr( double( _spectrum.getShape()[ 0 ] ) );
        _wavelength = ndarray::allocate( _spectrum.getShape()[ 0 ] );
        _wavelength.deep() = math::Poly( D_A1_Indices, _dispCoeffs );
        #ifdef __DEBUG_IDENTIFY__
          cout << "identify: _wavelength = " << _wavelength << endl;
        #endif

        /// Check for monotonic
        if ( math::isMonotonic( _wavelength ) == 0 ){
          cout << "Identify: WARNING: Wavelength solution is not monotonic => Setting identifyResult.rms to 1000" << endl;
          _dispRms = 1000.;
          cout << "Identify: RMS = " << _dispRms << endl;
          cout << "======================================" << endl;
        }

        _isWavelengthSet = true;
        return _isWavelengthSet;
    }

  
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
                   size_t nLinesCheck = 0 ){
        DispCorControl tempDispCorControl( dispCorControl );
        _dispCorControl.reset();
        _dispCorControl = tempDispCorControl.getPointer();

        ///for each line in line list, find maximum in spectrum and fit Gaussian
        ndarray::Array< double, 1, 1 > D_A1_GaussPos = hIdentify( lineList );

        ///remove lines which could not be found from line list
        std::vector< int > V_Index( D_A1_GaussPos.getShape()[ 0 ], 0 );
        size_t pos = 0;
        for (auto it = D_A1_GaussPos.begin(); it != D_A1_GaussPos.end(); ++it, ++pos ){
          if ( *it > 0. )
            V_Index[ pos ] = 1;
        }
        #ifdef __DEBUG_IDENTIFY__
          cout << "identify: D_A1_GaussPos = " << D_A1_GaussPos << endl;
          cout << "identify: V_Index = ";
          for (int iPos = 0; iPos < V_Index.size(); ++iPos)
            cout << V_Index[iPos] << " ";
          cout << endl;
        #endif
        std::vector< size_t > indices = math::getIndices( V_Index );
        size_t nInd = std::accumulate( V_Index.begin(), V_Index.end(), 0 );
        #ifdef __DEBUG_IDENTIFY__
          cout << "Identify: " << nInd << " lines identified" << endl;
          cout << "Identify: indices = ";
          for (int iPos = 0; iPos < indices.size(); ++iPos )
            cout << indices[iPos] << " ";
          cout << endl;
        #endif
        if ( nInd < ( std::round( double( lineList.getShape()[ 0 ] ) * 0.66 ) ) ){
          std::string message("pfs::drp::stella::identify: ERROR: ");
          message += "identify: ERROR: less than " + std::to_string( std::round( double( lineList.getShape()[ 0 ] ) * 0.66 ) ) + " lines identified";
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        ndarray::Array< size_t, 1, 1 > I_A1_IndexPos = ndarray::external( indices.data(), ndarray::makeVector( int( indices.size() ) ), ndarray::makeVector( 1 ) );
        ndarray::Array< double, 1, 1 > fittedPosLinesFound = math::getSubArray( D_A1_GaussPos, 
                                                                                I_A1_IndexPos );
        #ifdef __DEBUG_IDENTIFY__
          cout << "identify: fittedPosLinesFound = " << fittedPosLinesFound << endl;
        #endif

        ndarray::Array< double, 1, 1 > predictedWLen = ndarray::allocate( lineList.getShape()[ 0 ] );
        predictedWLen[ ndarray::view() ] = lineList[ ndarray::view()( 0 ) ];
        ndarray::Array< double, 1, 1 > fittedWLenLinesFound = math::getSubArray( predictedWLen, 
                                                                                 I_A1_IndexPos );
        cout << "Identify: found fittedWLenLinesFound = " << fittedWLenLinesFound << endl;

        ndarray::Array< double, 1, 1 > predictedPos = ndarray::allocate( predicted.getShape()[ 0 ] );
        predictedPos[ ndarray::view() ] = predicted[ ndarray::view() ];

        ndarray::Array< double, 1, 1 > predictedPosFound = math::getSubArray( predictedPos, 
                                                                              I_A1_IndexPos );

        ndarray::Array< double, 1, 1 > pixOffsetToFit = ndarray::allocate( nInd );
        pixOffsetToFit[ ndarray::view() ] = fittedPosLinesFound[ ndarray::view() ] - predictedPosFound[ ndarray::view() ];

        _dispCoeffs = ndarray::allocate( dispCorControl.order + 1 );
        _dispCoeffs.deep() = math::PolyFit( fittedPosLinesFound,
                                            pixOffsetToFit,
                                            dispCorControl.order );
        ndarray::Array< double, 1, 1 > D_A1_PixOffsetFit = math::Poly( fittedPosLinesFound, 
                                                                       _dispCoeffs );

        ///Interpolate wavelength from predicted wavelengths and measured pixel offset
        ndarray::Array< double, 1, 1 > pixIndex = math::indGenNdArr( double( _length ) );
        std::vector< std::string > interpolKeyWords( 1 );
        interpolKeyWords[ 0 ] = std::string( "SPLINE" );
        ndarray::Array< double, 1, 1 > predictedWLenAllPixA = ndarray::allocate( predictedWLenAllPix.getShape()[ 0 ] );
        predictedWLenAllPixA.deep() = predictedWLenAllPix;
        ndarray::Array< double, 1, 1 > wLenLinesFoundCheck = math::interPol( predictedWLenAllPixA, 
                                                                             pixIndex, 
                                                                             fittedPosLinesFound,
                                                                             interpolKeyWords );
        wLenLinesFoundCheck[ ndarray::view() ] = wLenLinesFoundCheck[ ndarray::view() ] + predictedPosFound[ ndarray::view() ];
        cout << "Identify: wLenLinesFoundCheck = " << wLenLinesFoundCheck << endl;
        cout << "identify: _dispCoeffs = " << _dispCoeffs << endl;

        ///Calculate RMS
        ndarray::Array< double, 1, 1 > D_A1_WLenMinusFit = ndarray::allocate( wLenLinesFoundCheck.getShape()[ 0 ] );
        D_A1_WLenMinusFit.deep() = fittedWLenLinesFound - wLenLinesFoundCheck;
        cout << "Identify: D_A1_WLenMinusFit = " << D_A1_WLenMinusFit << endl;
        _dispRms = math::calcRMS( D_A1_WLenMinusFit );
        cout << "Identify: _dispRms = " << _dispRms << endl;
        cout << "======================================" << endl;

        ///calibrate spectrum
        ndarray::Array< double, 1, 1 > D_A1_Indices = math::indGenNdArr( double( _spectrum.getShape()[ 0 ] ) );
        _wavelength = ndarray::allocate( _spectrum.getShape()[ 0 ] );
        _wavelength.deep() = math::Poly( D_A1_Indices, _dispCoeffs );
        #ifdef __DEBUG_IDENTIFY__
          cout << "identify: _wavelength = " << _wavelength << endl;
        #endif

        /// Check for monotonic
        if ( math::isMonotonic( _wavelength ) == 0 ){
          cout << "Identify: WARNING: Wavelength solution is not monotonic => Setting identifyResult.rms to 1000" << endl;
          _dispRms = 1000.;
          cout << "Identify: RMS = " << _dispRms << endl;
          cout << "======================================" << endl;
        }
        _isWavelengthSet = true;
        return _isWavelengthSet;
    }

    
    bool isWavelengthSet() const {return _isWavelengthSet;}
//    void setIsWavelengthSet(bool isWavelengthSet) {_isWavelengthSet = isWavelengthSet;}
    
    size_t getYLow() const { return _yLow; };
    size_t getYHigh() const { return _yHigh; };
    size_t getNCCDRows() const { return _nCCDRows; };
    
    bool setYLow( const size_t yLow );

    bool setYHigh( const size_t yHigh );
    
    bool setNCCDRows( const size_t nCCDRows );
    
  private:
    /**
     * @brief: Returns pixel positions of emission lines in lineList fitted in _spectrum
     * @param[in] lineList :: line list  [ nLines, 2 ]: [ wLen, approx_pixel ]
     */
    template< typename T >
    ndarray::Array< double, 1, 1 > hIdentify( ndarray::Array< T, 2, 1 > const& lineList );

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

    Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT> getSpectrum(const size_t i) const;

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

#include "SpectraTemplates.hpp"

#endif