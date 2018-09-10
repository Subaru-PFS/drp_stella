#if !defined(PFS_DRP_STELLA_SPECTRA_H)
#define PFS_DRP_STELLA_SPECTRA_H

#include <vector>

#include "lsst/afw/image/MaskedImage.h"
#include "pfs/drp/stella/Controls.h"

namespace pfs { namespace drp { namespace stella {
/**
 * \brief Describe a calibration line
 */
struct ReferenceLine {
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

    ReferenceLine(std::string const& _description, Status _status=NOWT, double _wavelength=0,
                  double _guessedIntensity=0, double _guessedPosition=0,
                  double _fitIntensity=0, double _fitPosition=0, double _fitPositionErr=0
                 ) :
        description(_description),
        status(_status),
        wavelength(_wavelength),
        guessedIntensity(_guessedIntensity),
        guessedPosition(_guessedPosition),
        fitIntensity(_fitIntensity),
        fitPosition(_fitPosition),
        fitPositionErr(_fitPositionErr)
    {}

    std::string description;            // description of line (e.g. Hg[II])
    int status;                         // status of line
    double wavelength;                   // vacuum wavelength, nm
    double guessedIntensity;             // input guess for intensity (amplitude of peak)
    double guessedPosition;              // input guess for pixel position
    double fitIntensity;                 // estimated intensity (amplitude of peak)
    double fitPosition;                  // fit line position
    double fitPositionErr;               // estimated standard deviation of fitPosition
};

/**
 * \brief Describe a single fiber trace
 */
class Spectrum {
  public:
    using ImageT = float;
    using VarianceT = float;
    using Mask = lsst::afw::image::Mask<lsst::afw::image::MaskPixel>;
    using ImageArray = ndarray::Array<ImageT, 1, 1>;
    using ConstImageArray = ndarray::Array<const ImageT, 1, 1>;
    using VarianceArray = ndarray::Array<VarianceT, 1, 1>;
    using ConstVarianceArray = ndarray::Array<VarianceT const, 1, 1>;
    using CovarianceMatrix = ndarray::Array<VarianceT, 2, 1>;
    using ConstCovarianceMatrix = ndarray::Array<VarianceT const, 2, 1>;
    using ReferenceLineList = std::vector<std::shared_ptr<ReferenceLine>>;
    using ConstReferenceLineList = std::vector<std::shared_ptr<const ReferenceLine>>;

    /// Construct an empty Spectrum
    ///
    /// @param length  Number of elements in spectrum
    /// @param fiberId  Fiber identifier
    explicit Spectrum(std::size_t length,
                      std::size_t fiberId=0);

    /// Construct Spectrum from elements
    ///
    /// @param spectrum  Spectrum values
    /// @param mask  Mask values
    /// @param background  Background values
    /// @param covariance  Covariance matrix
    /// @param wavelength  Wavelength values
    /// @param lines  Line list
    /// @param fiberId  Fiber identifier
    Spectrum(
        ImageArray const& spectrum,
        Mask const& mask,
        ImageArray const& background,
        CovarianceMatrix const& covariance,
        ImageArray const& wavelength,
        ReferenceLineList const& lines=ReferenceLineList(),
        std::size_t fiberId=0
    );

    Spectrum(Spectrum const&) = delete;
    Spectrum(Spectrum &&) = default;
    Spectrum & operator=(Spectrum const&) = delete;
    Spectrum & operator=(Spectrum&&) = default;

    virtual ~Spectrum() {}

    /// Return the number of pixels in the spectrum
    std::size_t getNumPixels() const { return _length; }

    //@{
    /// Return the spectrum
    ImageArray getSpectrum() { return _spectrum; }
    ConstImageArray getSpectrum() const { return _spectrum; }
    //@}

    //@{
    /// Return the background
    ImageArray getBackground() { return _background; }
    ConstImageArray getBackground() const { return _background; }
    //@}

    //@{
    /// Return the variance of this spectrum
    VarianceArray getVariance() { return _covariance[0]; }
    ConstVarianceArray getVariance() const { return _covariance[0]; }
    //@}
    
    //@{
    /// Return the pointer to the covariance of this spectrum
    CovarianceMatrix getCovariance() { return _covariance; }
    ConstCovarianceMatrix getCovariance() const { return _covariance; }
    //@}

    //@{
    /// Return the pointer to the wavelength vector of this spectrum
    ImageArray getWavelength() { return _wavelength; }
    ConstImageArray const getWavelength() const { return _wavelength; }
    //@}

    //@{
    /// Return the pointer to the mask vector of this spectrum
    Mask & getMask() { return _mask; }
    Mask const& getMask() const { return _mask; }
    //@}

    //@{
    /// Return the list of reference lines
    ReferenceLineList & getReferenceLines() { return _referenceLines; }
    ReferenceLineList const& getReferenceLines() const { return _referenceLines; }
    //@}

    /// Return the fiber identifier for this spectrum
    std::size_t getFiberId() const { return _fiberId; }

    /// Set the spectrum (deep copy)
    void setSpectrum(ndarray::Array<ImageT, 1, 1>  const& spectrum);

    /// Set the background pointer of this fiber trace to covar (deep copy)
    void setBackground(ImageArray const& background);

    /// Set the covariance pointer of this fiber trace to covar (deep copy)
    void setVariance(VarianceArray const& variance);

    /// Set the covariance pointer of this fiber trace to covar (deep copy)
    void setCovariance(CovarianceMatrix const& covar);

    /// Set the wavelength vector of this spectrum (deep copy)
    void setWavelength(ImageArray const& wavelength);

    /// Set the mask vector of this spectrum (deep copy)
    void setMask(Mask const& mask);

    /// Set the fiber identifier of this spectrum
    void setFiberId(std::size_t fiberId) { _fiberId = fiberId; }

    /// Set the list of reference lines
    void setReferenceLines(ReferenceLineList const& lines) { _referenceLines = lines; }

    /**
      * @brief: Identifies calibration lines, given input linelist for the wavelength-calibration spectrum
      * and fits Gaussians to each line
      *
      * Saves copy of lineList with as-observed values in _referenceLines
      **/
    void identify(ConstReferenceLineList const& lineList, ///< List of arc lines
                  DispersionCorrectionControl const& dispCorControl, ///< configuration params for wavelength calibration
                  int nLinesCheck=0                     ///< number of lines to hold back from fitting procedure
                  );

    bool isWavelengthSet() const { return _isWavelengthSet; }
    
  private:
    std::size_t _length;
    ImageArray _spectrum;
    Mask _mask;
    ImageArray _background;
    CovarianceMatrix _covariance;
    ImageArray _wavelength;
    std::size_t _fiberId;
    ReferenceLineList _referenceLines;
    bool _isWavelengthSet;
};

/************************************************************************************************************/
/**
 * @brief A set of spectra with a consistent length
 *
 * Spectra are stored by index, which is unrelated to the fiber identifier.
 *
 * All spectra must be the same length (otherwise, you'd just use a std::vector).
 */
class SpectrumSet {
  public:
    using SpectrumPtr = std::shared_ptr<Spectrum>;
    using ConstSpectrumPtr = std::shared_ptr<Spectrum const>;
    using Collection = std::vector<SpectrumPtr>;
    using ImageArray = ndarray::Array<float, 2, 1>;
    using CovarianceArray = ndarray::Array<float, 3, 1>;
    using MaskArray = ndarray::Array<lsst::afw::image::MaskPixel, 2, 1>;
    using iterator = Collection::iterator;
    using const_iterator = Collection::const_iterator;

    /// Construct an empty set of spectra
    explicit SpectrumSet(std::size_t length) : _length(length) {}

    /// Construct a set of spectra
    ///
    /// The spectra have fiber identifiers increasing from zero.
    explicit SpectrumSet(std::size_t numSpectra, std::size_t length);

    /// Construct from internal representation
    explicit SpectrumSet(Collection const& spectra);

    SpectrumSet(SpectrumSet const&) = delete;
    SpectrumSet(SpectrumSet &&) = default;
    SpectrumSet & operator=(SpectrumSet const&) = delete;
    SpectrumSet & operator=(SpectrumSet &&) = default;

    virtual ~SpectrumSet() {}

    /// Return the number of spectra/apertures
    std::size_t size() const { return _spectra.size(); }

    /// Reserve space for spectra
    void reserve(std::size_t num) { _spectra.reserve(num); }

    /// Return standard length
    std::size_t getLength() const { return _length; }

    //@{
    /// Get i-th spectrum
    ///
    /// No bounds checking.
    SpectrumPtr operator[](std::size_t i) { return _spectra[i]; }
    ConstSpectrumPtr operator[](std::size_t i) const { return _spectra[i]; }
    //@}

    //@{
    /// Get i-th spectrum
    ///
    /// Includes bounds checking.
    SpectrumPtr get(std::size_t i) { return _spectra.at(i); }
    ConstSpectrumPtr get(std::size_t i) const { return _spectra.at(i); }
    //@}

    //@{
    /// Iterator
    iterator begin() { return _spectra.begin(); }
    const_iterator begin() const { return _spectra.begin(); }
    iterator end() { return _spectra.end(); }
    const_iterator end() const { return _spectra.end(); }
    //@}

    /// Set the i-th spectrum
    void set(std::size_t i, SpectrumPtr spectrum);

    /// Add a spectrum
    void add(SpectrumPtr spectrum);

    /**
     * @brief Return all fluxes in an array [length x nFibers]
     */
    ImageArray getAllFluxes() const;
    
    /**
     * @brief Return all wavelengths in an array [length x nFibers]
     */
    ImageArray getAllWavelengths() const;
        
    /**
     * @brief Return all masks in an array [length x nFibers]
     */
    MaskArray getAllMasks() const;
    
    /**
     * @brief Return all covariances in an array [length x 3 x nFibers]
     */
    CovarianceArray getAllCovariances() const;

    /// Return the backgrounds of all spectra in an array [length x nFibers]
    ImageArray getAllBackgrounds() const;

    /// Return the internal representation
    Collection const& getInternal() const { return _spectra; }

  private:
    std::size_t const _length;  // length of each spectrum
    Collection _spectra; // spectra for each aperture
};

}}}

#endif
