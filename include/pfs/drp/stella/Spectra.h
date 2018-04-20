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
        RESERVED=2,                     // line was not used in estimating rms
        MISIDENTIFIED=4,                // line was misidentified
        CLIPPED=8,                      // line was clipped in fitting distortion
        SATURATED=16,                   // (centre of) line was saturated
        INTERPOLATED=32,                // (centre of) line was interpolated
        CR=64,                          // line is contaminated by a cosmic ray
    };

    ReferenceLine(std::string const& _description, Status _status=NOWT, float _wavelength=0,
                  float _guessedIntensity=0, float _guessedPixelPos=0,
                  float _fitIntensity=0, float _fitPixelPos=0, float _fitPixelPosErr=0
                 )
        : description(_description),
          status(_status),
          wavelength(_wavelength),
          guessedIntensity(_guessedIntensity),
          guessedPixelPos(_guessedPixelPos),
          fitIntensity(_fitIntensity),
          fitPixelPos(_fitPixelPos),
          fitPixelPosErr(_fitPixelPosErr)
        { }

    std::string description;            // description of line (e.g. Hg[II])
    int status;                         // status of line
    float wavelength;                   // vacuum wavelength, nm
    float guessedIntensity;             // input guess for intensity (amplitude of peak)
    float guessedPixelPos;              // input guess for pixel position
    float fitIntensity;                 // estimated intensity (amplitude of peak)
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
    using ImageArray = ndarray::Array<ImageT, 1, 1>;
    using ConstImageArray = ndarray::Array<const ImageT, 1, 1>;
    using VarianceArray = ndarray::Array<VarianceT, 1, 1>;
    using CovarianceMatrix = ndarray::Array<VarianceT, 2, 1>;
    using ReferenceLineList = std::vector<std::shared_ptr<ReferenceLine>>;
    using ConstReferenceLineList = std::vector<std::shared_ptr<const ReferenceLine>>;

    // Class Constructors and Destructor
    explicit Spectrum(std::size_t length=0,
                      std::size_t fiberId=0);

    Spectrum(Spectrum const& spectrum) = delete;
    
    virtual ~Spectrum() {}

    /// Return the number of pixels in the spectrum
    std::size_t getNpix() const { return _length; }

    /// Return a shared pointer to the spectrum
    ImageArray getSpectrum() { return _spectrum; }
    ImageArray const getSpectrum() const { return _spectrum; }

    /// Set the spectrum (deep copy)
    void setSpectrum(ndarray::Array<ImageT, 1, 1>  const& spectrum);

    ImageArray getBackground() { return _background; }
    ConstImageArray getBackground() const { return _background; }

    /// Return a copy of the variance of this spectrum
    VarianceArray getVariance() const;
    VarianceArray getVariance();
    
    /// Return the pointer to the covariance of this spectrum
    CovarianceMatrix getCovar() { return _covar; }
    CovarianceMatrix getCovar() const { return _covar; }

    /// Set the background pointer of this fiber trace to covar (deep copy)
    void setBackground(ImageArray const& background);

    /// Set the covariance pointer of this fiber trace to covar (deep copy)
    void setVariance(VarianceArray const& variance);

    /// Set the covariance pointer of this fiber trace to covar (deep copy)
    void setCovar(CovarianceMatrix const& covar);

    /// Return the pointer to the wavelength vector of this spectrum
    ImageArray getWavelength() { return _wavelength; }
    ImageArray const getWavelength() const { return _wavelength; }

    /// Set the wavelength vector of this spectrum (deep copy)
    void setWavelength(ImageArray const& wavelength);

    /// Return the pointer to the mask vector of this spectrum
    Mask & getMask() { return _mask; }
    Mask const& getMask() const { return _mask; }

    /// Set the mask vector of this spectrum (deep copy)
    void setMask(Mask const& mask);
    
    std::size_t getFiberId() const { return _fiberId; }
    void setFiberId(std::size_t fiberId) { _fiberId = fiberId; }

    /**
      * @brief: Identifies calibration lines, given input linelist for the wavelength-calibration spectrum
      * and fits Gaussians to each line
      *
      * Saves copy of lineList with as-observed values in _referenceLines
      **/
    void identify(ConstReferenceLineList const& lineList, ///< List of arc lines
                  DispCorControl const& dispCorControl, ///< configuration params for wavelength calibration
                  int nLinesCheck=0                     ///< number of lines to hold back from fitting procedure
                 );

    ReferenceLineList & getReferenceLines() { return _referenceLines; }
    ReferenceLineList const& getReferenceLines() const { return _referenceLines; }
    void setReferenceLines(ReferenceLineList const& lines) {
        _referenceLines = lines;
    }
    
    bool isWavelengthSet() const { return _isWavelengthSet; }
    
  private:
    std::size_t _length;
    ImageArray _spectrum;
    Mask _mask;
    ImageArray _background;
    CovarianceMatrix _covar;
    ImageArray _wavelength;
    std::size_t _fiberId;               // for logging / debugging purposes only
    ReferenceLineList _referenceLines;
    bool _isWavelengthSet;
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

}}}

#endif
