#if !defined(PFS_DRP_STELLA_SPECTRA_H)
#define PFS_DRP_STELLA_SPECTRA_H

#include <vector>

#include "lsst/afw/image/MaskedImage.h"
#include "pfs/drp/stella/Controls.h"

namespace pfs { namespace drp { namespace stella {
/**
 * \brief Describe a calibration line
 */
class ReferenceLine {
public:
    enum Status {                       // Line's status
        NOWT=0,
        FIT=1,                          // line was found and fit; n.b. powers of 2
        KEPT_BACK=2,                    // line was not used
        MISIDENTIFIED=4                 // line was misidentified
    };

    ReferenceLine(Status _status=NOWT, float _wavelength=0, float _guessedPixelPos=0,
                  float _fitPixelPos=0, float _fitPixelPosErr=0
                 )
        : status(_status),
          wavelength(_wavelength),
          guessedPixelPos(_guessedPixelPos),
          fitPixelPos(_fitPixelPos),
          fitPixelPosErr(_fitPixelPosErr)
        { }

    int status;                         // status of line
    float wavelength;                   // vacuum wavelength, nm
    int guessedPixelPos;                // input guess on pixel position
    float fitPixelPos;                  // fit line position
    float fitPixelPosErr;               // estimated standard deviation of fitPixelPos
};

/**
 * \brief Describe a single fiber trace
 */
class Spectrum {
  public:
    typedef float ImageT;
    typedef float VarianceT;

    typedef lsst::afw::image::Mask<lsst::afw::image::MaskPixel> Mask;

    // Class Constructors and Destructor
    explicit Spectrum(std::size_t length=0,
                      std::size_t iTrace=0);

    Spectrum(Spectrum const& spectrum) = delete;
    
    virtual ~Spectrum() {}

    /// Return the number of pixels in the spectrum
    std::size_t getNpix() const { return _length; }

    /// Return a shared pointer to the spectrum
    ndarray::Array<ImageT, 1, 1> getSpectrum() { return _spectrum; }
    ndarray::Array<ImageT, 1, 1> const getSpectrum() const { return _spectrum; }

    /// Set the spectrum (deep copy)
    void setSpectrum(ndarray::Array<ImageT, 1, 1>  const& spectrum);

    ndarray::Array<ImageT, 1, 1> getBackground() { return _background; }
    ndarray::Array<const ImageT, 1, 1> getBackground() const { return _background; }

    /// Return a copy of the variance of this spectrum
    ndarray::Array<VarianceT, 1, 1> getVariance() const;
    ndarray::Array<VarianceT, 1, 1> getVariance();
    
    /// Return the pointer to the covariance of this spectrum
    ndarray::Array<VarianceT, 2, 1> getCovar() { return _covar; }
    ndarray::Array<VarianceT, 2, 1> getCovar() const { return _covar; }

    /// Set the covariance pointer of this fiber trace to covar (deep copy)
    void setVariance(ndarray::Array<ImageT, 1, 1> const& variance);

    /// Set the covariance pointer of this fiber trace to covar (deep copy)
    void setCovar(ndarray::Array<VarianceT, 2, 1> const& covar);

    /// Return the pointer to the wavelength vector of this spectrum
    ndarray::Array<ImageT, 1, 1> getWavelength() { return _wavelength; }
    ndarray::Array<ImageT, 1, 1> const getWavelength() const { return _wavelength; }

    /// Set the wavelength vector of this spectrum (deep copy)
    void setWavelength(ndarray::Array<ImageT, 1, 1> const& wavelength);

    /// Return the pointer to the mask vector of this spectrum
    Mask getMask() { return _mask; }
    Mask const getMask() const { return _mask; }

    /// Set the mask vector of this spectrum (deep copy)
    void setMask(Mask const& mask);
    
    std::size_t getITrace() const { return _iTrace; }
    void setITrace(std::size_t iTrace) { _iTrace = iTrace; }

    /*
     * @brief Return the Root Mean Squared (RMS) of the lines used for the wavelength calibration
     */
    float getDispRms() const { return _dispRms; }

    /*
     * @brief Return the Root Mean Squared (RMS) of the lines held back from the wavelength calibration
     */
    float getDispRmsCheck() const { return _dispRmsCheck; }

    /*
     * @brief Return the number of not rejected lines used for the wavelength calibration
     */
    std::size_t getNGoodLines() const { return _nGoodLines; }

    /**
      * @brief: Identifies calibration lines, given in <lineList> in the format [wlen, approx_pixel] in
      * wavelength-calibration spectrum
      * fits Gaussians to each line, fits Polynomial of order I_PolyFitOrder_In, and
      * writes _wavelength and PolyFit coefficients to _dispCoeffs
      * @param lineList in: input line list [wLen, approx_pixel]
      * @param dispCorControl in: DispCorControl to use for wavelength calibration
      * @param nLinesCheck in: number of lines to hold back from fitting procedure
      **/
    void identify(ndarray::Array<float, 2, 1> const& lineList,
                  DispCorControl const& dispCorControl,
                  int nLinesCheck=0);

    std::vector<std::shared_ptr<ReferenceLine>>& getReferenceLines() { return _referenceLines; }
    
    bool isWavelengthSet() const { return _isWavelengthSet; }
    
  private:
    /**
     * @brief: Returns pixel positions of emission lines in lineList fitted in _spectrum
     * @param[in] lineList :: line list  [ nLines, 2 ]: [ wLen, approx_pixel ]
     */
    ndarray::Array< float, 1, 1 > hIdentify(ndarray::Array<float, 1, 1> const& lineListLambda,
                                            ndarray::Array<float, 1, 1> const& lineListPixel,
                                            DispCorControl const& dispCorControl
                                           );

    std::size_t _length;
    ndarray::Array<ImageT, 1, 1> _spectrum;
    Mask _mask;
    ndarray::Array<ImageT, 1, 1> _background;
    ndarray::Array<VarianceT, 2, 1> _covar;
    ndarray::Array<ImageT, 1, 1> _wavelength;
    ndarray::Array<ImageT, 1, 1> _dispersion;
    std::size_t _iTrace;/// for logging / debugging purposes only
    ndarray::Array<float, 1, 1> _dispCoeffs;
    std::vector<std::shared_ptr<ReferenceLine>> _referenceLines;
    float _dispRms;
    float _dispRmsCheck;
    std::size_t _nGoodLines;
    bool _isWavelengthSet;

  protected:
};

/************************************************************************************************************/
/**
 * \brief Describe a set of spectra
 *
 */
class SpectrumSet
{
  public:
    typedef std::vector<PTR(Spectrum)> Spectra;

    /// Class Constructors and Destructor
      
    /// Creates a new SpectrumSet object of size 'nSpectra' of length 'length'
    explicit SpectrumSet(std::size_t nSpectra=0, std::size_t length=0);
        
    /// Copy constructor
    SpectrumSet(SpectrumSet const& spectrumSet) = delete;
        
    virtual ~SpectrumSet() {}

    /// Return the number of spectra/apertures
    std::size_t getNtrace() const { return _spectra->size(); }

    /** @brief  Return the Spectrum for the ith fiberTrace
     *  @param i :: number of spectrum ( or number of respective FiberTrace ) to return
     * **/
    PTR(Spectrum) getSpectrum( const std::size_t i );

    /** @brief  Return the Spectrum for the ith fiberTrace
     *  @param i :: number of spectrum ( or number of respective FiberTrace ) to return
     **/
    PTR(const Spectrum) getSpectrum( const std::size_t i ) const;

    /**
     * @brief Set the ith Spectrum
     * @param i :: Set which spectrum in set?
     * @param spectrum :: spectrum to set to this set at position i
     **/
    void setSpectrum(std::size_t const i, PTR(Spectrum) spectrum);
    
    /** 
     * @brief Add one Spectrum to the set
     * @param spectrum :: spectrum to add 
     **/
    void addSpectrum(PTR(Spectrum) spectrum) {
        _spectra->push_back(spectrum);
    }
    
    /**
     * @brief Return all fluxes in an array [length x nFibers]
     */
    ndarray::Array<float, 2, 1> getAllFluxes() const;
    
    /**
     * @brief Return all wavelengths in an array [length x nFibers]
     */
    ndarray::Array<float, 2, 1> getAllWavelengths() const;
        
    /**
     * @brief Return all masks in an array [length x nFibers]
     */
    ndarray::Array< int, 2, 1> getAllMasks() const;
    
    /**
     * @brief Return all covariances in an array [length x 3 x nFibers]
     */
    ndarray::Array<float, 3, 1> getAllCovars() const;
    
  private:
    PTR(Spectra) _spectra; // spectra for each aperture
};

namespace math{
    /**
     * @brief: create line list from wavelength array of size length and list of wavelengths of emission lines used to calibrate the spectrum
     * @param wLen
     * @param lines
     * @return array(lines.shape[0], 2) col 0: wavelength, col 1: pixel
     */
    template <typename T, int I>
    ndarray::Array<T, 2, 1> createLineList(ndarray::Array<T, 1, I> const& wLen,
                                           ndarray::Array<T, 1, I> const& lines);

}

}}}

#endif
