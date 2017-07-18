#if !defined(PFS_DRP_STELLA_SPECTRA_H)
#define PFS_DRP_STELLA_SPECTRA_H

#include <iostream>
#include <vector>

#include "boost/algorithm/string/trim.hpp"
#include <fitsio.h>
#include <fitsio2.h>

#include "Controls.h"
#include "lsst/log/Log.h"
#include "lsst/pex/exceptions.h"
#include "math/CurveFitting.h"
#include "math/Math.h"
#include "utils/Utils.h"

#define stringify( name ) # name

namespace afwImage = lsst::afw::image;
namespace pexExcept = lsst::pex::exceptions;

namespace pfs { namespace drp { namespace stella {
/**
 * \brief Describe a single fiber trace
 */
template<typename PixelT,
         typename MaskT=afwImage::MaskPixel,
         typename VarianceT=afwImage::VariancePixel,
         typename WavelengthT=afwImage::VariancePixel>
class Spectrum {
  public:
    typedef ndarray::Array<PixelT, 1, 1> SpectrumVector;
    typedef ndarray::Array<VarianceT, 1, 1> VarianceVector;
    typedef ndarray::Array<WavelengthT, 1, 1> WavelengthVector;
    typedef ndarray::Array<VarianceT, 2, 1> CovarianceMatrix;
    typedef lsst::afw::image::Mask<MaskT> Mask;
    typedef ndarray::Array<double, 1, 1> Coefficients;

    // Class Constructors and Destructor
    explicit Spectrum(std::size_t length=0,
                      std::size_t iTrace=0);
    
    /// iTrace is only assigned to _iTrace if != 0, otherwise spectrum._iTrace is copied to this->_iTrace
    Spectrum(Spectrum & spectrum,
             std::size_t iTrace=0,
             bool deep=false);

    Spectrum(Spectrum const& spectrum);
    
    virtual ~Spectrum() {}

    /// Return a shared pointer to the spectrum
    SpectrumVector getSpectrum() { return _spectrum; }
    SpectrumVector const getSpectrum() const { return _spectrum; }

    /// Return a shared pointer to the sky spectrum
    SpectrumVector getSky() { return _sky; }
    SpectrumVector const getSky() const { return _sky; }
    
    /// Set the spectrum (deep copy)
    void setSpectrum(SpectrumVector const& spectrum);

    void setSky(SpectrumVector const& sky );
    
    /// Return a copy of the variance of this spectrum
    VarianceVector getVariance() const;
    VarianceVector getVariance();
    
    /// Return the pointer to the covariance of this spectrum
    CovarianceMatrix getCovar() { return _covar; }
    CovarianceMatrix getCovar() const { return _covar; }

    /// Set the covariance pointer of this fiber trace to covar (deep copy)
    void setVariance(VarianceVector const& variance);

    /// Set the covariance pointer of this fiber trace to covar (deep copy)
    void setCovar(CovarianceMatrix const& covar);

    /// Return the pointer to the wavelength vector of this spectrum
    WavelengthVector getWavelength() { return _wavelength; }
    WavelengthVector const getWavelength() const { return _wavelength; }

    /// Return the pointer to the wavelength dispersion vector of this spectrum
    WavelengthVector getDispersion() { return _dispersion; }
    WavelengthVector const getDispersion() const { return _dispersion; }

    /// Set the wavelength vector of this spectrum (deep copy)
    void setWavelength(WavelengthVector const& wavelength);

    /// Set the dispersion vector of this spectrum (deep copy)
    void setDispersion(WavelengthVector const& dispersion);

    /// Return the pointer to the mask vector of this spectrum
    Mask getMask() { return _mask; }
    Mask const getMask() const { return _mask; }

    /// Set the mask vector of this spectrum (deep copy)
    void setMask(Mask const& mask);

    std::size_t getLength() const { return _length; }
    
    /// Resize all vectors to size length.
    /// If length is smaller than the current container size, the contents of all vectors are reduced to
    /// their first length elements, removing those beyond (and destroying them).
    /// If length is greater than the current container size, the contents of all vectors are expanded by
    /// inserting at the end as many elements as needed to reach a size of length. The new elements of all
    /// vectors except for _wavelength are initialized with 0. The new elements of _wavelength are
    /// initialized with _wavelength(_length - 1).
    void setLength(const std::size_t length);
    
    std::size_t getITrace() const { return _iTrace; }
    void setITrace(std::size_t iTrace) { _iTrace = iTrace; }
    
    Coefficients const getDispCoeffs() const { return _dispCoeffs; };

    void setDispCoeffs(Coefficients const& dispCoeffs);

    /*
     * @brief Return the Root Mean Squared (RMS) of the lines used for the wavelength calibration
     */
    double getDispRms() const { return _dispRms; }

    /*
     * @brief Return the Root Mean Squared (RMS) of the lines held back from the wavelength calibration
     */
    double getDispRmsCheck() const { return _dispRmsCheck; }

    /*
     * @brief Return the number of not rejected lines used for the wavelength calibration
     */
    std::size_t getNGoodLines() const { return _nGoodLines; }

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
    template <typename T>
    void identify(ndarray::Array<T, 2, 1> const& lineList,
                  DispCorControl const& dispCorControl,
                  std::size_t nLinesCheck=0);
    
    bool isWavelengthSet() const { return _isWavelengthSet; }
    
    std::size_t getYLow() const { return _yLow; };
    std::size_t getYHigh() const { return _yHigh; };
    std::size_t getNCCDRows() const { return _nCCDRows; };
    
    void setYLow(std::size_t yLow);
    void setYHigh(std::size_t yHigh);
    void setNCCDRows(std::size_t nCCDRows);
    
  private:
    /**
     * @brief: Returns pixel positions of emission lines in lineList fitted in _spectrum
     * @param[in] lineList :: line list  [ nLines, 2 ]: [ wLen, approx_pixel ]
     */
    template< typename T >
    ndarray::Array< double, 1, 1 > hIdentify( ndarray::Array< T, 2, 1 > const& lineList );

    std::size_t _yLow;
    std::size_t _yHigh;
    std::size_t _length;
    std::size_t _nCCDRows;
    SpectrumVector _spectrum;
    SpectrumVector _sky;
    Mask _mask;
    CovarianceMatrix _covar;
    WavelengthVector _wavelength;
    WavelengthVector _dispersion;
    std::size_t _iTrace;/// for logging / debugging purposes only
    Coefficients _dispCoeffs;
    double _dispRms;
    double _dispRmsCheck;
    std::size_t _nGoodLines;
    bool _isWavelengthSet;
    PTR(DispCorControl) _dispCorControl;

  protected:
};

/************************************************************************************************************/
/**
 * \brief Describe a set of spectra
 *
 */
template <typename ImageT,
          typename MaskT=afwImage::MaskPixel,
          typename VarianceT=afwImage::VariancePixel,
          typename WavelengthT=afwImage::VariancePixel>
class SpectrumSet
{
  public:
    typedef Spectrum<ImageT, MaskT, VarianceT, WavelengthT> SpectrumT;
    typedef std::vector<PTR(SpectrumT)> Spectra;

    /// Class Constructors and Destructor
      
    /// Creates a new SpectrumSet object of size 'nSpectra' of length 'length'
    explicit SpectrumSet(std::size_t nSpectra=0, std::size_t length=0);
        
    /// Copy constructor
    /// If spectrumSet is not empty, the object shares ownership of spectrumSet's spectrum vector and increases the use count.
    /// If spectrumSet is empty, an empty object is constructed (as if default-constructed).
    SpectrumSet(SpectrumSet const& spectrumSet)
        : _spectra(spectrumSet.getSpectra())
        {}

    /// Construct an object with a copy of spectrumVector
    explicit SpectrumSet(Spectra const& spectrumVector);
        
    virtual ~SpectrumSet() {}

    /// Return the number of spectra/apertures
    std::size_t size() const { return _spectra->size(); }

    /** @brief  Return the Spectrum for the ith fiberTrace
     *  @param i :: number of spectrum ( or number of respective FiberTrace ) to return
     * **/
    PTR(SpectrumT) getSpectrum( const std::size_t i );

    /** @brief  Return the Spectrum for the ith fiberTrace
     *  @param i :: number of spectrum ( or number of respective FiberTrace ) to return
     **/
    PTR(SpectrumT) getSpectrum( const std::size_t i ) const;

    /**
     * @brief Set the ith Spectrum
     * @param i :: Set which spectrum in set?
     * @param spectrum :: spectrum to set to this set at position i
     **/
    void setSpectrum(std::size_t const i,
                     PTR(SpectrumT) const& spectrum);
    
    /**
     * @brief Set the ith Spectrum
     * @param i :: Set which spectrum in set?
     * @param spectrum :: spectrum to copy to this set at position i
     **/
    void setSpectrum(std::size_t const i,
                     SpectrumT const& spectrum);

    /** 
     * @brief Add one Spectrum to the set
     * @param spectrum :: spectrum to add 
     **/
    void addSpectrum(SpectrumT const& spectrum) {
        _spectra->push_back(PTR(SpectrumT)(new SpectrumT(spectrum)));
    }
    
    /** 
     * @brief Add one Spectrum to the set
     * @param spectrum :: spectrum to add 
     **/
    void addSpectrum(PTR(SpectrumT) const& spectrum){
        _spectra->push_back(spectrum);
    }

    const PTR(Spectra) getSpectra() const { return _spectra; }
    PTR(Spectra) getSpectra() { return _spectra; }

    /// Removes from the vector either a single element (position) or a range of elements ([first,last)).
    /// This effectively reduces the container size by the number of elements removed, which are destroyed.
    void erase(std::size_t iStart, std::size_t iEnd=0);
    
    /**
     * @brief Return all fluxes in an array [nCCDRows x nFibers]
     */
    ndarray::Array<float, 2, 1> getAllFluxes() const;
    
    /**
     * @brief Return all wavelengths in an array [nCCDRows x nFibers]
     */
    ndarray::Array<float, 2, 1> getAllWavelengths() const;
    
    /**
     * @brief Return all dispersions in an array [nCCDRows x nFibers]
     */
    ndarray::Array<float, 2, 1> getAllDispersions() const;
    
    /**
     * @brief Return all masks in an array [nCCDRows x nFibers]
     */
    ndarray::Array< int, 2, 1> getAllMasks() const;
    
    /**
     * @brief Return all skies in an array [nCCDRows x nFibers]
     */
    ndarray::Array<float, 2, 1> getAllSkies() const;
    
    /**
     * @brief Return all variances in an array [nCCDRows x nFibers]
     */
    ndarray::Array<float, 2, 1> getAllVariances() const;
    
    /**
     * @brief Return all covariances in an array [nCCDRows x 3 x nFibers]
     */
    ndarray::Array<float, 3, 1> getAllCovars() const;
    
  private:
    PTR(Spectra) _spectra; // spectra for each aperture
};

namespace math{
    
    template <typename T, typename U>
    struct StretchAndCrossCorrelateSpecResult{
        ndarray::Array<U, 2, 1> lineList;
        ndarray::Array<T, 3, 1> specPieces;
    };
    
    template <typename T, typename U>
    StretchAndCrossCorrelateSpecResult<T, U> stretchAndCrossCorrelateSpec(
        ndarray::Array< T, 1, 1 > const& spec,
        ndarray::Array< T, 1, 1 > const& specRef,
        ndarray::Array< U, 2, 1 > const& lineList_WLenPix,
        DispCorControl const& dispCorControl
    );
    
    /**
     * @brief: create line list from wavelength array of size nCCDRows and list of wavelengths of emission lines used to calibrate the spectrum
     * @param wLen
     * @param lines
     * @return array(lines.shape[0], 2) col 0: wavelength, col 1: pixel
     */
    template <typename T, int I>
    ndarray::Array<T, 2, 1> createLineList(ndarray::Array<T, 1, I> const& wLen,
                                           ndarray::Array<T, 1, I> const& lines);

}

}}}

#include "SpectraTemplates.hpp"

#endif
