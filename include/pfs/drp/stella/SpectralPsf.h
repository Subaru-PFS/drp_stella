#include "lsst/pex/exceptions.h"
#include "lsst/afw/detection/Psf.h"
#include "lsst/afw/image.h"

#include "pfs/drp/stella/DetectorMap.h"

namespace pfs {
namespace drp {
namespace stella {


/// Resample the kernel image
///
/// The kernel image is resampled by the appropriate factor, with the center
/// placed at the nominated position, and normalised to unit sum (as required
/// by LSST afw's Psf class).
///
/// The input image must have xy0 set.
///
/// @param image : kernel image to resample
/// @param binning : binning factor to apply
/// @param bbox : desired bounding box
/// @param center : desired center (in native pixel frame)
/// @return resampled kernel image
template <typename T>
std::shared_ptr<lsst::afw::image::Image<T>> resampleKernelImage(
    lsst::afw::image::Image<T> const& image,
    int binning,
    lsst::geom::Box2I const& bbox,
    lsst::geom::Point2D const& center=lsst::geom::Point2D(0, 0)
);


/// Recenter an oversampled kernel image
///
/// The centroid is measured, and the image is shifted so that the centroid
/// sits at the nominated position (when xy0 is respected and a subsequent
/// resampling is performed).
///
/// This is a version of lsst::afw::detection::Psf::recenterKernelImage that
/// works on oversampled images. In particular, it puts the center in the proper
/// position for a subsequent resampling: when oversampled by an odd factor,
/// we place the centroid at the center of a pixel; when oversampled by an even
/// factor, we place the centroid on the corner of a pixel.
///
/// @param image : image to recenter.
/// @param binning : binning factor to be applied (important for putting the
///     center in the right place
/// @param target : desired center (in native pixel frame)
/// @return recentered image.
template<typename T>
std::shared_ptr<lsst::afw::image::Image<T>> recenterOversampledKernelImage(
    lsst::afw::image::Image<T> const& image,
    int binning=1,
    lsst::geom::Point2D const& target=lsst::geom::Point2D(0, 0)
);


/** A PSF constructed from over-sampled images
 *
 * This is a pure-virtual layer on top of the LSST Psf class to provide a
 * point-spread function implementation when you can construct an oversampled
 * representation of the PSF. Having an oversampled representation is important
 * when the PSF is not fully Nyquist-sampled, since subpixel shifts used to
 * place the PSF centroid at the correct position employ Lanczos resampling,
 * which is only valid for Nyquist-sampled images.
 *
 * This class implements the required computation interfaces for the Psf class
 * (computeImage, computeKernelImage, computeApertureFlux, computeShape,
 * computeBBox), given an oversampled kernel image. The user class needs to
 * supply the following virtual methods:
 * - doComputeOversampledKernelImage: provides an oversampled kernel image of
 *   the PSF for a position on the image. The center of the PSF can be anywhere
 *   in the image: we will recenter as required. This private method is required
 *   by this (OversampledPsf) base class.
 * - clone: provide a copy of the user class. This public method is required by
 *   the Psf base class.
 * - resized: provide a copy of the user class with a different size. This can
 *   be accomplished by calling this (OversampledPsf) base class' constructor
 *   with the appropriate targetSize argument. This public method is required by
 *   the Psf base class.
 */
class OversampledPsf : public virtual lsst::afw::detection::Psf {
  public:
    using Image = lsst::afw::detection::Psf::Image;

    /// Return the size of the PSF
    lsst::geom::Extent2I getTargetSize() const { return _targetSize; }

    /// Return the oversampling factor
    int getOversampleFactor() const { return _oversampleFactor; }

  protected:
    /** Ctor
     *
     * @param oversampleFactor : integer factor by which the PSF is oversampled.
     * @param targetSize : desired size of the PSF kernel image after
     *            resampling.
     * @param isFixed : is the PSF not a function of position?
     * @param capacity : size of the cache.
     */
    OversampledPsf(
        int oversampleFactor,
        lsst::geom::Extent2I const& targetSize,
        bool isFixed=false,
        std::size_t capacity=100
    ) : Psf(isFixed, capacity),
        _oversampleFactor(oversampleFactor),
        _targetSize(targetSize)
        {}

private:
    //@{
    /// Implementations required for lsst::afw::detection::Psf
    virtual std::shared_ptr<Image> doComputeImage(
        lsst::geom::Point2D const& position,
        lsst::afw::image::Color const& color
    ) const override final;
    virtual std::shared_ptr<Image> doComputeKernelImage(
        lsst::geom::Point2D const& position,
        lsst::afw::image::Color const& color
    ) const override final;
    virtual double doComputeApertureFlux(
        double radius,
        lsst::geom::Point2D const& position,
        lsst::afw::image::Color const& color
    ) const override final;
    virtual lsst::afw::geom::ellipses::Quadrupole doComputeShape(
        lsst::geom::Point2D const& position,
        lsst::afw::image::Color const& color
    ) const override final;
    virtual lsst::geom::Box2I doComputeBBox(
        lsst::geom::Point2D const& position,
        lsst::afw::image::Color const& color
    ) const override final;
    //@}

    /** Provide an oversampled representation of the PSF
     *
     * This is the chief method that user classes must implement.
     *
     * The centroid of the PSF should be at the center of the 0,0 pixel (when
     * xy0 is accounted for).
     *
     * @param position : Position on the image at which to realise the
     *            oversampled PSF.
     * @returns oversampled image of the PSF.
     */
    virtual std::shared_ptr<Image> doComputeOversampledKernelImage(
        lsst::geom::Point2D const& position
    ) const = 0;

    int const _oversampleFactor;  ///< Factor by which the PSF is oversampled
    lsst::geom::Extent2I const _targetSize;   ///< desired size of the PSF kernel image after resampling
};


/** A layer on top of Psf to provide public fiberId,wavelength interfaces
 *
 * PSFs need to be realised for a particular fiber at a particular wavelength,
 * but the Psf class operates on x,y position on the detector. This base class
 * provides methods for accessing the outputs of the Psf class via fiberId and
 * wavelength, by folding in a DetectorMap.
 * */
class SpectralPsf : public virtual lsst::afw::detection::Psf {
    using Super = lsst::afw::detection::Psf;

  public:
    /// Return the detectorMap
    std::shared_ptr<DetectorMap> getDetectorMap() const { return _detMap; }

    /** Compute an image of the PSF that can be compared with a spectral line on the image
     *
     * See also lsst::afw::detection::Psf::computeImage.
     *
     * @param fiberId : fiber identifier
     * @param wavelength : wavelength of line (nm)
     * @return image of the PSF
     */
    std::shared_ptr<Image> computeImage(int fiberId, double wavelength) const {
        return Super::computeImage(getPosition(fiberId, wavelength));
    }
    std::shared_ptr<Image> computeImage(lsst::geom::Point2D const& position) const {
        return Super::computeImage(position);
    }

    /** Compute an image of the PSF centered at the appropriate subpixel position
     *
     * See also lsst::afw::detection::Psf::computeImage.
     *
     * @param fiberId : fiber identifier
     * @param wavelength : wavelength of line (nm)
     * @return image of the PSF
     */
    std::shared_ptr<Image> computeKernelImage(int fiberId, double wavelength) const {
        return Super::computeKernelImage(getPosition(fiberId, wavelength));
    }
    std::shared_ptr<Image> computeKernelImage(lsst::geom::Point2D const& position) const {
        return Super::computeKernelImage(position);
    }

    /** Compute the value of the peak of the PSF
     *
     * See also lsst::afw::detection::Psf::computePeak.
     *
     * @param fiberId : fiber identifier
     * @param wavelength : wavelength of line (nm)
     * @return peak value of the PSF
     */
    double computePeak(int fiberId, double wavelength) const {
        return Super::computePeak(getPosition(fiberId, wavelength));
    }
    double computePeak(lsst::geom::Point2D const& position) const {
        return Super::computePeak(position);
    }

    /** Compute the flux of the PSF within a circular aperture
     *
     * See also lsst::afw::detection::Psf::computeApertureFlux.
     *
     * @param radius : circular aperture radius (pixels)
     * @param fiberId : fiber identifier
     * @param wavelength : wavelength of line (nm)
     * @return flux within aperture
     */
    double computeApertureFlux(double radius, int fiberId, double wavelength) const {
        return Super::computeApertureFlux(radius, getPosition(fiberId, wavelength));
    }
    double computeApertureFlux(double radius, lsst::geom::Point2D const& position) const {
        return Super::computeApertureFlux(radius, position);
    }

    /** Compute the shape of the PSF
     *
     * See also lsst::afw::detection::Psf::computeShape.
     *
     * @param fiberId : fiber identifier
     * @param wavelength : wavelength of line (nm)
     * @return shape of the PSF
     */
    lsst::afw::geom::ellipses::Quadrupole computeShape(int fiberId, double wavelength) const {
        return Super::computeShape(getPosition(fiberId, wavelength));
    }
    lsst::afw::geom::ellipses::Quadrupole computeShape(lsst::geom::Point2D const& position) const {
        return Super::computeShape(position);
    }
    lsst::afw::geom::ellipses::Quadrupole computeShape() const {
        return Super::computeShape(getAveragePosition());
    }

    /** Return a kernel corresponding to the PSF
     *
     * See also lsst::afw::detection::Psf::getLocalKernel.
     *
     * @param fiberId : fiber identifier
     * @param wavelength : wavelength of line (nm)
     * @return kernel for the PSF
     */
    std::shared_ptr<lsst::afw::math::Kernel const> getLocalKernel(int fiberId, double wavelength) const {
        return Super::getLocalKernel(getPosition(fiberId, wavelength));
    }
    std::shared_ptr<lsst::afw::math::Kernel const> getLocalKernel(lsst::geom::Point2D const& position) const {
        return Super::getLocalKernel(position);
    }

    /** Compute the bounding box of the PSF
     *
     * See also lsst::afw::detection::Psf::computeBBox.
     *
     * @param fiberId : fiber identifier
     * @param wavelength : wavelength of line (nm)
     * @return bounding box of the PSF image
     */
    lsst::geom::Box2I computeBBox(int fiberId, double wavelength) const {
        return Super::computeBBox(getPosition(fiberId, wavelength));
    }
    lsst::geom::Box2I computeBBox(lsst::geom::Point2D const& position) const {
        return Super::computeBBox(position);
    }

  protected:
    /** Ctor
     *
     * @param detMap : mapping between x,y and fiberId,wavelength
     * @param isFixed : is the PSF not a function of position?
     * @param capacity : size of the cache.
     */
    SpectralPsf(
        std::shared_ptr<DetectorMap> detMap,
        bool isFixed=false,
        std::size_t capacity=100
    ) : Psf(isFixed, capacity),
        _detMap(detMap)
    {
        if (!detMap) {
            throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "detectorMap not set");
        }
    }

    /** Convert fiberId,wavelength to x,y
     *
     * Throws an exception if the point is off the image.
     * @param fiberId : fiber identifier
     * @param wavelength : wavelength of line (nm)
     * @returns position of line.
     */
    lsst::geom::Point2D getPosition(int fiberId, double wavelength) const;

    std::shared_ptr<DetectorMap> _detMap;  ///< mapping between x,y and fiberId,wavelength
};


/** An adaptor to use an LSST Psf class as a SpectralPsf
 *
 * We compose an LSST Psf object with a SpectralPsf interface. This allows us
 * to use an imaging PSF for spectral work.
 */
class ImagingSpectralPsf :
    public lsst::afw::table::io::PersistableFacade<ImagingSpectralPsf>,
    public SpectralPsf {
  public:
    /** Ctor
     *
     * @param base : imaging PSF
     * @param detMap : mapping between x,y and fiberId,wavelength
     */
    ImagingSpectralPsf(
        std::shared_ptr<Psf> const base,
        std::shared_ptr<DetectorMap> detMap
    ) : SpectralPsf(detMap),
        _base(base)
        {}

    /// Return the imaging PSF
    std::shared_ptr<Psf const> getBase() const { return _base; }

    virtual std::shared_ptr<Psf> clone() const override {
        return std::make_shared<ImagingSpectralPsf>(_base, _detMap);
    }

    virtual std::shared_ptr<Psf> resized(int width, int height) const override {
        return std::make_shared<ImagingSpectralPsf>(_base->resized(width, height), _detMap);
    }

    /// Whether the Psf is persistable; always true.
    bool isPersistable() const noexcept override { return true; }

protected:
    std::string getPersistenceName() const override { return "ImagingSpectralPsf"; }
    std::string getPythonModule() const override { return "pfs.drp.stella"; }

    void write(OutputArchiveHandle& handle) const override;

    //@{
    /// Implementations required for lsst::afw::detection::Psf
    virtual std::shared_ptr<Image> doComputeKernelImage(
        lsst::geom::Point2D const& position,
        lsst::afw::image::Color const& color
    ) const override final {
        return _base->computeKernelImage(position, color);
    }

    virtual double doComputeApertureFlux(
        double radius,
        lsst::geom::Point2D const& position,
        lsst::afw::image::Color const& color
    ) const override final {
        return _base->computeApertureFlux(radius, position, color);
    }

    virtual lsst::afw::geom::ellipses::Quadrupole doComputeShape(
        lsst::geom::Point2D const& position,
        lsst::afw::image::Color const& color
    ) const override final {
        return _base->computeShape(position, color);
    }

    virtual lsst::geom::Box2I doComputeBBox(
        lsst::geom::Point2D const& position,
        lsst::afw::image::Color const& color
    ) const override final {
        return _base->computeBBox(position, color);
    }
    //@}

  private:
    std::shared_ptr<Psf> const _base;  ///< imaging PSF
};

}}}  // namespace pfs::drp::stella