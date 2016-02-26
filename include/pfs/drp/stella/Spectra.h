#if !defined(PFS_DRP_STELLA_SPECTRA_H)
#define PFS_DRP_STELLA_SPECTRA_H

#include <vector>
#include <iostream>
#include <cmath>
#include "lsst/base.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/pex/config.h"
#include "lsst/pex/exceptions/Exception.h"
//#include "blitz.h"
#include <fitsio.h>
#include <fitsio2.h>
#include "math/Math.h"
#include "math/CurveFitting.h"
#include "utils/Utils.h"
#include "Controls.h"

#define stringify( name ) # name

//#define __DEBUG_IDENTIFY__
//#define __DEBUG_STRETCHANDCROSSCORRELATESPEC__
//#define __DEBUG_STRETCHANDCROSSCORRELATESPEC_LINELIST__

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
    explicit Spectrum(Spectrum const& spectrum,
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
    bool setSpectrum( const ndarray::Array< SpectrumT, 1, 1 > & spectrum );

    bool setSky( const ndarray::Array< SpectrumT, 1, 1 > & sky );
    
    /// Return the pointer to the variance of this spectrum
    ndarray::Array<VarianceT, 1, 1> getVariance() { return ndarray::Array<VarianceT, 1, 1 >(_covar[ ndarray::view( 3 )( ) ]); }
    const ndarray::Array<VarianceT, 1, 1> getVariance() const { return _covar[ ndarray::view( 3 )( ) ]; }
    
    /// Return the pointer to the covariance of this spectrum
    ndarray::Array<VarianceT, 2, 1> getCovar() { return _covar; }
    const ndarray::Array<VarianceT, 2, 1> getCovar() const { return _covar; }

    /// Set the covariance pointer of this fiber trace to covar (deep copy)
    /// sets this->_covar to covar and returns TRUE if covar->size() == this->getLength(), otherwise returns false
    bool setVariance(const ndarray::Array<VarianceT, 1, 1> & variance );

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
    template< typename T >
    bool identify( ndarray::Array< T, 2, 1 > const& lineList,
                   DispCorControl const& dispCorControl );
    
    bool isWavelengthSet() const {return _isWavelengthSet;}
//    void setIsWavelengthSet(bool isWavelengthSet) {_isWavelengthSet = isWavelengthSet;}
    
    size_t getYLow() const { return _yLow; };
    size_t getYHigh() const { return _yHigh; };
    size_t getNCCDRows() const { return _nCCDRows; };
    
    bool setYLow( size_t yLow );
    bool setYHigh( size_t yHigh );
    bool setNCCDRows( size_t nCCDRows );
    
  private:
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
class SpectrumSet {
  public:
    /// Class Constructors and Destructor
      
    /// Creates a new SpectrumSet object of size 0
    explicit SpectrumSet(size_t nSpectra=0, size_t length=0);
        
    /// Copy constructor
    /// If spectrumSet is not empty, the object shares ownership of spectrumSet's spectrum vector and increases the use count.
    /// If spectrumSet is empty, an empty object is constructed (as if default-constructed).
    explicit SpectrumSet(const SpectrumSet &spectrumSet)
        : _spectra(spectrumSet.getSpectra())
        {}

    /// Construct an object with a copy of spectrumVector
    explicit SpectrumSet(const PTR(std::vector<PTR(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>)>) &spectrumVector);
        
    virtual ~SpectrumSet() {}

    /// Return the number of spectra/apertures
    size_t size() const { return _spectra->size(); }

    /// Return the Spectrum for the ith aperture
    PTR(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>) &getSpectrum(const size_t i);

    PTR(const Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>) const& getSpectrum(const size_t i) const;

    /// Set the ith Spectrum
    bool setSpectrum(const size_t i,     /// which spectrum?
                     const PTR(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>) & spectrum);

    /// add one Spectrum to the set
    void addSpectrum(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT> const& spectrum);
    void addSpectrum(PTR(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>) const& spectrum);

    PTR(std::vector<PTR(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>)>) getSpectra() const { return _spectra; }

    
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
    
  private:
    PTR(std::vector<PTR(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>)>) _spectra; // spectra for each aperture
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

}

namespace utils{
  
  template<typename T>
  PTR(T) getPointer(T &);

}

}}}
#endif