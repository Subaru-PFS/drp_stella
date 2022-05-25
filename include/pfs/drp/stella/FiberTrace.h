#if !defined(PFS_DRP_STELLA_FIBERTRACE_H)
#define PFS_DRP_STELLA_FIBERTRACE_H

#include <string>
#include <memory>

#include "lsst/afw/image/MaskedImage.h"

#include "pfs/drp/stella/Spectrum.h"

namespace pfs { namespace drp { namespace stella {

std::string const fiberMaskPlane = "FIBERTRACE";  ///< Mask plane we care about

/**
 * @brief Describe a single fiber trace
 */
template<typename ImageT,
         typename MaskT=lsst::afw::image::MaskPixel,
         typename VarianceT=lsst::afw::image::VariancePixel>
class FiberTrace {
  public:
    typedef lsst::afw::image::MaskedImage<ImageT, MaskT, VarianceT> MaskedImageT;
    typedef lsst::afw::image::Image<ImageT> Image;
    typedef lsst::afw::image::Mask<MaskT> Mask;
    typedef lsst::afw::image::Image<VarianceT> Variance;

    /** @brief Class Constructors and Destructor
     *
     * @param maskedImage : maskedImage to set _trace to
     * @param xCenters : position of center for each row
     * @param fiberTraceId : FiberTrace ID
     * */
    explicit FiberTrace(MaskedImageT const& trace,
                        std::size_t fiberTraceId=0);

    /**
     * @brief Copy constructor (deep if required)
     *
     * @param fiberTrace : FiberTrace to copy
     * @param deep : Deep copy if true, shallow copy if false
     */
    FiberTrace(FiberTrace<ImageT, MaskT, VarianceT> const& fiberTrace,
               bool deep=false);

    FiberTrace(FiberTrace const&) = default;
    FiberTrace(FiberTrace &&) = default;
    FiberTrace & operator=(FiberTrace const&) = default;
    FiberTrace & operator=(FiberTrace &&) = default;

    /**
     * @brief Destructor
     */
    virtual ~FiberTrace() {}

    //@{
    /**
     * @brief Return the 2D MaskedImage of this fiber trace
     */
    MaskedImageT & getTrace() { return _trace; }
    MaskedImageT const& getTrace() const { return _trace; }
    //@}

    /**
     * @brief Return an image containing the reconstructed 2D spectrum of the FiberTrace
     *
     * @param spectrum : 1D spectrum to reconstruct the 2D image from
     * @param bbox : bounding box of image
     */
    std::shared_ptr<Image> constructImage(
        Spectrum const& spectrum,
        lsst::geom::Box2I const & bbox
    ) const;

    /**
     * @brief Return an image containing the reconstructed 2D spectrum of the FiberTrace
     *
     * @param spectrum : 1D spectrum to reconstruct the 2D image from
     */
    std::shared_ptr<Image> constructImage(Spectrum const& spectrum) const {
        return constructImage(spectrum, getTrace().getBBox());
    }

    /**
     * @brief Create an image containing the reconstructed 2D spectrum of the FiberTrace
     *
     * @param image : image into which to reconstruct trace
     * @param spectrum : 1D spectrum to reconstruct the 2D image from
     */
    void constructImage(
        lsst::afw::image::Image<ImageT> & image,
        Spectrum const& spectrum
    ) const;

    /**
     * @brief set the ID number of this trace (_fiberId) to this number
     * @param fiberId : ID to be assigned to this FiberTrace
     */
    void setFiberId(std::size_t fiberId) { _fiberId = fiberId; }

    /**
     * @brief Return ID of this FiberTrace
     */
    std::size_t getFiberId() const { return _fiberId; }

    /**
     * @brief Construct from a fiber profile
     *
     * @param fiberId : fiber identifier.
     * @param dims : dimensions of image (not just this fiber trace).
     * @param radius : distance either side (i.e., a half-width) of the center
     *     the profile is measured for.
     * @param oversample : oversample factor for the profile.
     * @param rows : average row value for each swath, of length Nswath.
     * @param profiles : profiles for each swath, each of length Nswath and
     *     width = 2*(radius + 1)*oversample + 1.
     * @param good : indicates which values in the profiles may be used.
     * @param centers : center of profile for each row in the image.
     * @param norm : normalisation to apply
     */
    static FiberTrace fromProfile(
        int fiberId,
        lsst::geom::Extent2I const& dims,
        int radius,
        double oversample,
        ndarray::Array<double, 1, 1> const& rows,
        ndarray::Array<double, 2, 1> const& profiles,
        ndarray::Array<bool, 2, 1> const& good,
        ndarray::Array<double, 1, 1> const& centers,
        ndarray::Array<Spectrum::ImageT, 1, 1> const& norm=ndarray::Array<Spectrum::ImageT, 1, 1>()
    );

  private:
    lsst::afw::image::MaskedImage<ImageT, MaskT, VarianceT> _trace;
    std::size_t _fiberId;
};

/************************************************************************************************************/

}}}

#endif
