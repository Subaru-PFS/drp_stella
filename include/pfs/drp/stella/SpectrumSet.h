#if !defined(PFS_DRP_STELLA_SPECTRUMSET_H)
#define PFS_DRP_STELLA_SPECTRUMSET_H

#include <memory>

#include "lsst/afw/image/MaskedImage.h"
#include "pfs/drp/stella/Spectrum.h"

namespace pfs { namespace drp { namespace stella {

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
    using ImageArray = ndarray::Array<double, 2, 1>;
    using CovarianceArray = ndarray::Array<double, 3, 1>;
    using MaskArray = ndarray::Array<lsst::afw::image::MaskPixel, 2, 1>;
    using WavelengthArray = ndarray::Array<double, 2, 1>;
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
     * @brief Return all fiberIds
     */
    ndarray::Array<int, 1, 1> getAllFiberIds() const;

    /**
     * @brief Return all fluxes in an array [length x nFibers]
     */
    ImageArray getAllFluxes() const;

    /**
     * @brief Return all wavelengths in an array [length x nFibers]
     */
    WavelengthArray getAllWavelengths() const;

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
    std::size_t _length;  // length of each spectrum
    Collection _spectra; // spectra for each aperture
};

}}}

#endif
