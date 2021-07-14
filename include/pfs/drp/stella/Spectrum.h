#if !defined(PFS_DRP_STELLA_SPECTRUM_H)
#define PFS_DRP_STELLA_SPECTRUM_H

#include <memory>
#include <vector>

#include "lsst/afw/image/MaskedImage.h"

namespace pfs { namespace drp { namespace stella {

/**
 * \brief Describe a single spectrum
 */
class Spectrum {
  public:
    using ImageT = double;
    using VarianceT = double;
    using Mask = lsst::afw::image::Mask<lsst::afw::image::MaskPixel>;
    using ImageArray = ndarray::Array<ImageT, 1, 1>;
    using ConstImageArray = ndarray::Array<const ImageT, 1, 1>;
    using VarianceArray = ndarray::Array<VarianceT, 1, 1>;
    using ConstVarianceArray = ndarray::Array<VarianceT const, 1, 1>;
    using CovarianceMatrix = ndarray::Array<VarianceT, 2, 1>;
    using ConstCovarianceMatrix = ndarray::Array<VarianceT const, 2, 1>;
    using WavelengthArray = ndarray::Array<double, 1, 1>;
    using ConstWavelengthArray = ndarray::Array<double const, 1, 1>;

    /// Construct an empty Spectrum
    ///
    /// @param length  Number of elements in spectrum
    /// @param fiberId  Fiber identifier
    explicit Spectrum(std::size_t length,
                      int fiberId=0);

    /// Construct Spectrum from elements
    ///
    /// @param flux  Spectrum values
    /// @param mask  Mask values
    /// @param background  Background values
    /// @param norm  Normalisation
    /// @param covariance  Covariance matrix
    /// @param wavelength  Wavelength values
    /// @param lines  Line list
    /// @param fiberId  Fiber identifier
    Spectrum(
        ImageArray const& flux,
        Mask const& mask,
        ImageArray const& background,
        ImageArray const& norm,
        CovarianceMatrix const& covariance,
        WavelengthArray const& wavelength,
        int fiberId=0
    );

    Spectrum(Spectrum const&) = delete;
    Spectrum(Spectrum &&) = default;
    Spectrum & operator=(Spectrum const&) = delete;
    Spectrum & operator=(Spectrum&&) = default;

    virtual ~Spectrum() {}

    /// Return the number of pixels in the spectrum
    std::size_t getNumPixels() const { return _length; }

    //@{
    /// Return the flux array
    ImageArray getFlux() { return _flux; }
    ConstImageArray getFlux() const { return _flux; }
    //@}

    //@{
    /// Return the spectrum
    ///
    /// Synonym for getFlux.
    ImageArray getSpectrum() { return _flux; }
    ConstImageArray getSpectrum() const { return _flux; }
    //@}

    //@{
    /// Return the background
    ImageArray getBackground() { return _background; }
    ConstImageArray getBackground() const { return _background; }
    //@}

    //@{
    /// Return the normalisation
    ImageArray getNorm() { return _norm; }
    ConstImageArray getNorm() const { return _norm; }
    //@}

    /// Return the normalised flux
    ImageArray getNormFlux() const;

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
    WavelengthArray getWavelength() { return _wavelength; }
    ConstWavelengthArray const getWavelength() const { return _wavelength; }
    //@}

    //@{
    /// Return the pointer to the mask vector of this spectrum
    Mask & getMask() { return _mask; }
    Mask const& getMask() const { return _mask; }
    //@}

    /// Return the fiber identifier for this spectrum
    int getFiberId() const { return _fiberId; }

    /// Set the flux (deep copy)
    void setFlux(ndarray::Array<ImageT, 1, 1>  const& flux);

    /// Set the spectrum (deep copy)
    ///
    /// A synonym for setFlux.
    void setSpectrum(ndarray::Array<ImageT, 1, 1>  const& spectrum) {
        return setFlux(spectrum);
    }

    /// Set the background pointer of this fiber trace to covar (deep copy)
    void setBackground(ImageArray const& background);

    /// Set the normalisation (deep copy)
    void setNorm(ImageArray const& norm);

    /// Set the covariance pointer of this fiber trace to covar (deep copy)
    void setVariance(VarianceArray const& variance);

    /// Set the covariance pointer of this fiber trace to covar (deep copy)
    void setCovariance(CovarianceMatrix const& covar);

    /// Set the wavelength vector of this spectrum (deep copy)
    void setWavelength(WavelengthArray const& wavelength);

    /// Set the mask vector of this spectrum (deep copy)
    void setMask(Mask const& mask);

    /// Set the fiber identifier of this spectrum
    void setFiberId(int fiberId) { _fiberId = fiberId; }

    bool isWavelengthSet() const { return _isWavelengthSet; }

  private:
    std::size_t _length;
    ImageArray _flux;
    Mask _mask;
    ImageArray _background;
    CovarianceMatrix _covariance;
    WavelengthArray _wavelength;
    int _fiberId;
    bool _isWavelengthSet;
};

}}} // namespace pfs::drp::stella

#endif
