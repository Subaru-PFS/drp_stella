#include "ndarray.h"

#include "pfs/drp/stella/SpectralPsf.h"

namespace pfs {
namespace drp {
namespace stella {

/** PSF for PFS spectrograph provided by Neven Caplar
 *
 * Neven provides oversampled PSF images. This class chooses the appropriate
 * images for a position and interpolates between them in order to provide the
 * PSF.
 */
class NevenPsf
  : public lsst::afw::table::io::PersistableFacade<NevenPsf>,
    public SpectralPsf,
    public OversampledPsf
{
  public:
    /** Ctor
     *
     * @param detMap : mapping between x,y and fiberId,wavelength
     * @param xx : x positions of the oversampled images
     * @param yy : y positions of the oversampled images
     * @param images : the oversampled images
     * @param oversampleFactor : integer factor by which the PSF is oversampled
     * @param targetSize : desired size of the PSF kernel image after
     *            resampling
     * @param xMaxDistance : maximum distance in x for selecting images for
     *            interpolation
     */
    NevenPsf(
        DetectorMap const& detMap,
        ndarray::Array<float const, 1, 1> const& xx,
        ndarray::Array<float const, 1, 1> const& yy,
        std::vector<ndarray::Array<double const, 2, 1>> const& images,
        int oversampleFactor,
        lsst::geom::Extent2I const& targetSize,
        float xMaxDistance=20
    );

    virtual ~NevenPsf() = default;
    NevenPsf(NevenPsf const&) = default;
    NevenPsf(NevenPsf&&) = default;
    NevenPsf& operator=(NevenPsf const&) = delete;
    NevenPsf& operator=(NevenPsf&&) = delete;

    ndarray::Array<float const, 1, 1> getX() const { return _xx; }
    ndarray::Array<float const, 1, 1> getY() const { return _yy; }
    std::vector<ndarray::Array<double const, 2, 1>> getImages() const { return _images; }
    float getXMaxDistance() const { return _xMaxDistance; }

    virtual std::shared_ptr<Psf> clone() const override;
    virtual std::shared_ptr<lsst::afw::detection::Psf> resized(int width, int height) const override;

    bool isPersistable() const noexcept override { return true; }

    class Factory;

  protected:
    std::string getPersistenceName() const override { return "NevenPsf"; }

    std::string getPythonModule() const override { return "pfs.drp.stella"; }

    void write(lsst::afw::table::io::OutputArchiveHandle & handle) const override;

    /** Provide an oversampled representation of the PSF
     *
     * This method chooses the appropriate PSF images and interpolates between
     * them.
     *
     * @param position : Position on the image at which to realise the
     *            oversampled PSF.
     * @returns oversampled image of the PSF.
     */
    virtual std::shared_ptr<OversampledPsf::Image> doComputeOversampledKernelImage(
        lsst::geom::Point2D const& position
    ) const override;

  private:
    ndarray::Array<float const, 1, 1> const _xx;  ///< x positions of the oversampled images
    ndarray::Array<float const, 1, 1> const _yy;  ///< y positions of the oversampled images
    std::vector<ndarray::Array<double const, 2, 1>> const _images;  ///< oversampled images
    float const _xMaxDistance;  ///< maximum distance in x for selecting images for interpolation
};

}}}  // namespace pfs::drp::stella
