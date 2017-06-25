#if !defined(PFS_DRP_STELLA_SPECTRA_H)
#define PFS_DRP_STELLA_SPECTRA_H

#include <iostream>
#include <vector>

#include "boost/algorithm/string/trim.hpp"
#include <fitsio.h>
#include <fitsio2.h>

#include "Controls.h"
#include "lsst/afw/geom.h"
#include "lsst/log/Log.h"
#include "lsst/pex/exceptions.h"
#include "math/CurveFitting.h"
#include "math/Math.h"
#include "utils/Utils.h"
#include "lsst/afw/fits.h"
#include "lsst/afw/image/fits/fits_io.h"
#include "lsst/afw/image/fits/fits_io_mpl.h"

#define stringify( name ) # name

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
    Spectrum( Spectrum & spectrum,
              size_t iTrace = 0,
              bool deep = false );

    Spectrum( Spectrum const& spectrum );//,
    
    virtual ~Spectrum() {}

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
    
    /// Return a copy of the variance of this spectrum
    ndarray::Array<VarianceT, 1, 1> getVariance() const;// { return _covar[ ndarray::view( )( 1 ) ]; }
    ndarray::Array<VarianceT, 1, 1> getVariance();// { return _covar[ ndarray::view( )( 1 ) ]; }
    
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
    afwImage::Mask<MaskT> getMask() { return _mask; }
    const afwImage::Mask<MaskT> getMask() const { return _mask; }

    /// Set the mask vector of this spectrum (deep copy)
    /// sets this->_mask to mask and returns TRUE if mask->size() == this->getLength(), otherwise returns false
    bool setMask(const afwImage::Mask<MaskT> & mask);

    size_t getLength() const {return _length;}
    
    /// Resize all vectors to size length.
    /// If length is smaller than the current container size, the contents of all vectors are reduced to
    /// their first length elements, removing those beyond (and destroying them).
    /// If length is greater than the current container size, the contents of all vectors are expanded by
    /// inserting at the end as many elements as needed to reach a size of length. The new elements of all
    /// vectors except for _wavelength are initialized with 0. The new elements of _wavelength are
    /// initialized with _wavelength(_length - 1).
    bool setLength(const size_t length);
    
    size_t getITrace() const {return _iTrace;}
    void setITrace(size_t iTrace){_iTrace = iTrace;}
    
    ndarray::Array< double, 1, 1 > getDispCoeffs( ) const { return _dispCoeffs; };
    
    bool setDispCoeffs( ndarray::Array< double, 1, 1 > const& dispCoeffs );

    /*
     * @brief Return the Root Mean Squared (RMS) of the lines used for the wavelength calibration
     */
    double getDispRms() const {return _dispRms;}

    /*
     * @brief Return the Root Mean Squared (RMS) of the lines held back from the wavelength calibration
     */
    double getDispRmsCheck() const {return _dispRmsCheck;}

    /*
     * @brief Return the number of not rejected lines used for the wavelength calibration
     */
    size_t getNGoodLines() const {return _nGoodLines;}

    /// Return _dispCorControl
    PTR(DispCorControl) getDispCorControl() const { return _dispCorControl; }
  
    /**
      * @brief: Identifies calibration lines, given in <lineList> in the format [wlen, approx_pixel] in
      * wavelength-calibration spectrum
      * fits Gaussians to each line, fits Polynomial of order I_PolyFitOrder_In, and
      * writes _wavelength and PolyFit coefficients to _dispCoeffs
      * @param lineList in: input line list [wLen, approx_pixel]
      * @param dispCorControl in: DispCorControl to use for wavelength calibration
      * @param nLinesCheck in: number of lines to hold back from fitting procedure
      **/
    template< typename T >
    bool identify( ndarray::Array< T, 2, 1 > const& lineList,
                   DispCorControl const& dispCorControl,
                   size_t nLinesCheck = 0 );
    
    bool isWavelengthSet() const {return _isWavelengthSet;}
    
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
    afwImage::Mask<MaskT> _mask;
    ndarray::Array<VarianceT, 2, 1> _covar;
    ndarray::Array<WavelengthT, 1, 1> _wavelength;
    ndarray::Array<WavelengthT, 1, 1> _dispersion;
    size_t _iTrace;/// for logging / debugging purposes only
    ndarray::Array< double, 1, 1 > _dispCoeffs;
    double _dispRms;
    double _dispRmsCheck;
    size_t _nGoodLines;
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
      
    /// Creates a new SpectrumSet object of size 'nSpectra' of length 'length'
    explicit SpectrumSet(size_t nSpectra=0, size_t length=0);
        
    /// Copy constructor
    /// If spectrumSet is not empty, the object shares ownership of spectrumSet's spectrum vector and increases the use count.
    /// If spectrumSet is empty, an empty object is constructed (as if default-constructed).
    SpectrumSet( SpectrumSet const& spectrumSet)
        : _spectra(spectrumSet.getSpectra())
        {}

    /// Construct an object with a copy of spectrumVector
    explicit SpectrumSet(PTR(std::vector<PTR(Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT>)>) const& spectrumVector);
        
    virtual ~SpectrumSet() {}

    /// Return the number of spectra/apertures
    size_t size() const { return _spectra->size(); }

    /** @brief  Return the Spectrum for the ith fiberTrace
     *  @param i :: number of spectrum ( or number of respective FiberTrace ) to return
     * **/
    PTR( Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT > ) getSpectrum( const size_t i );

    /** @brief  Return the Spectrum for the ith fiberTrace
     *  @param i :: number of spectrum ( or number of respective FiberTrace ) to return
     **/
    PTR( Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT > ) getSpectrum( const size_t i ) const;

    /**
     * @brief Set the ith Spectrum
     * @param i :: Set which spectrum in set?
     * @param spectrum :: spectrum to set to this set at position i
     **/
    bool setSpectrum(size_t const i,
                     PTR( Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT > ) const& spectrum);
    
    /**
     * @brief Set the ith Spectrum
     * @param i :: Set which spectrum in set?
     * @param spectrum :: spectrum to copy to this set at position i
     **/
    bool setSpectrum(size_t const i,
                     Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT > const& spectrum);

    /** 
     * @brief Add one Spectrum to the set
     * @param spectrum :: spectrum to add 
     **/
    void addSpectrum(Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT > const& spectrum) {
        _spectra->push_back(PTR(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>)(new Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>(spectrum)));
    }
    
    /** 
     * @brief Add one Spectrum to the set
     * @param spectrum :: spectrum to add 
     **/
    void addSpectrum(PTR(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>) const& spectrum){
        _spectra->push_back(spectrum);
    }

    const PTR( std::vector< PTR( Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT > ) > ) getSpectra() const { return _spectra; }
    PTR( std::vector< PTR( Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT > ) > ) getSpectra() { return _spectra; }

    /// Removes from the vector either a single element (position) or a range of elements ([first,last)).
    /// This effectively reduces the container size by the number of elements removed, which are destroyed.
    bool erase(const size_t iStart, const size_t iEnd=0);
    
    /**
     * @brief Return all fluxes in an array [nCCDRows x nFibers]
     */
    ndarray::Array< float, 2, 1 > getAllFluxes() const;
    
    /**
     * @brief Return all wavelengths in an array [nCCDRows x nFibers]
     */
    ndarray::Array< float, 2, 1 > getAllWavelengths() const;
    
    /**
     * @brief Return all dispersions in an array [nCCDRows x nFibers]
     */
    ndarray::Array< float, 2, 1 > getAllDispersions() const;
    
    /**
     * @brief Return all masks in an array [nCCDRows x nFibers]
     */
    ndarray::Array< int, 2, 1 > getAllMasks() const;
    
    /**
     * @brief Return all skies in an array [nCCDRows x nFibers]
     */
    ndarray::Array< float, 2, 1 > getAllSkies() const;
    
    /**
     * @brief Return all variances in an array [nCCDRows x nFibers]
     */
    ndarray::Array< float, 2, 1 > getAllVariances() const;
    
    /**
     * @brief Return all covariances in an array [nCCDRows x 3 x nFibers]
     */
    ndarray::Array< float, 3, 1 > getAllCovars() const;
    
  private:
    PTR( std::vector< PTR(Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT > ) > ) _spectra; // spectra for each aperture
};

namespace math{
    
    template< typename T, typename U >
    struct StretchAndCrossCorrelateSpecResult{
        ndarray::Array< U, 2, 1 > lineList;
        ndarray::Array< T, 3, 1 > specPieces;
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
    template< typename T, int I >
    ndarray::Array< T, 2, 1 > createLineList( ndarray::Array< T, 1, I > const& wLen,
                                              ndarray::Array< T, 1, I > const& lines );

}

}}}

#include "SpectraTemplates.hpp"

#endif
