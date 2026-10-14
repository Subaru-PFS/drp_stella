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
    using OversampledImage = ndarray::Array<float, 2, 1>;

    /** Ctor
     *
     * @param detMap : mapping between x,y and fiberId,wavelength
     * @param fiberId : fiberIds of the oversampled images
     * @param wavelength : wavelength of the oversampled images
     * @param images : the oversampled images
     * @param oversampleFactor : integer factor by which the PSF is oversampled
     * @param targetSize : desired size of the PSF kernel image after
     *            resampling
     */
    NevenPsf(
        std::shared_ptr<DetectorMap> detMap,
        ndarray::Array<int, 1, 1> const& fiberId,
        ndarray::Array<double, 1, 1> const& wavelength,
        std::vector<OversampledImage> const& images,
        int oversampleFactor,
        lsst::geom::Extent2I const& targetSize
    );

    virtual ~NevenPsf() = default;
    NevenPsf(NevenPsf const&) = default;
    NevenPsf(NevenPsf&&) = default;
    NevenPsf& operator=(NevenPsf const&) = delete;
    NevenPsf& operator=(NevenPsf&&) = delete;

    std::size_t size() const;
    ndarray::Array<int, 1, 1> getFiberId() const;
    ndarray::Array<double, 1, 1> getWavelength() const;
    std::vector<OversampledImage> getImages() const;

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
    struct Data {
        double wavelength;
        double y;
        OversampledImage image;
    };
    using DataArray = std::vector<Data>;
    std::map<int, DataArray> _data;  ///< oversampled images for each fiber
};

}}}  // namespace pfs::drp::stella
