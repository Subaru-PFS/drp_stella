#include "lsst/afw/table.h"
#include "lsst/afw/image/Exposure.h"
#include "lsst/afw/detection/GaussianPsf.h"
#include "lsst/meas/base/SdssCentroid.h"

#include "pfs/drp/stella/centroidImage.h"

namespace pfs {
namespace drp {
namespace stella {

namespace {

/// Construct a GaussianPsf
///
/// Takes some care to get a PSF kernel that will fit in the image.
///
/// @param psfSigma : Gaussian sigma of PSF.
/// @param dims : Dimensions of the image.
/// @param nSigma : Number of sigma (either side) for size of PSF kernel.
/// @return Point-spread function
std::shared_ptr<lsst::afw::detection::Psf> makeGaussianPsf(
    float psfSigma,
    lsst::geom::Extent2I const& dims,
    float nSigma=3
) {
    int const psfSize = 2*int(nSigma*psfSigma) + 1;  // nSigma either side, and odd.
    // SdssCentroid requires an additional 2 pixels on either side, and size must be odd
    int const xSize = std::min(dims.getX() - (dims.getX() % 2 ? 4 : 5), psfSize);
    int const ySize = std::min(dims.getY() - (dims.getY() % 2 ? 4 : 5), psfSize);
    return std::make_shared<lsst::afw::detection::GaussianPsf>(xSize, ySize, psfSigma);
}

}  // anonymous namespace


lsst::geom::Point2D centroidExposure(
    lsst::afw::image::Exposure<float> const& exposure,
    lsst::geom::Point2D const& point
) {
    // Generate measurement framework infrastructure
    std::string const guessName = "guess";
    std::string const resultName = "centroid";
    auto schema = lsst::afw::table::SourceTable::makeMinimalSchema();

    lsst::meas::base::SdssCentroidControl ctrl;
    ctrl.binmax = 1;  // No binning should be fine if the PSF is reasonable.
    lsst::meas::base::SdssCentroidAlgorithm algorithm(ctrl, resultName, schema);
    lsst::afw::table::SourceCatalog catalog(schema);
    auto & source = *catalog.addNew();
    auto fp = std::make_shared<lsst::afw::detection::Footprint>();
    fp->addPeak(point.getX(), point.getY(), std::numeric_limits<float>::quiet_NaN());
    source.setFootprint(fp);

    // Run the centroider
    algorithm.measure(source, exposure);
    return lsst::afw::table::PointKey<double>(schema[resultName]).get(source);
}


template <typename T>
lsst::geom::Point2D centroidImage(
    lsst::afw::image::Image<T> const& image,
    std::shared_ptr<lsst::afw::detection::Psf> psf
) {
    // Determine starting centroid guess
    std::size_t const width = image.getWidth();
    std::size_t const height = image.getHeight();
    auto const& array = image.getArray();
    lsst::geom::Extent2D guess{0, 0};
    double sum = 0;
    for (std::size_t yy = 0; yy < height; ++yy) {
        for (std::size_t xx = 0; xx < width; ++xx) {
            double const value = array[yy][xx];
            guess += value*lsst::geom::Extent2D(xx, yy);
            sum += value;
        }
    }

    auto imgPtr = std::make_shared<lsst::afw::image::Image<float>>(image, !std::is_same<T, float>::value);
    lsst::afw::image::MaskedImage<float, lsst::afw::image::MaskPixel, float> mi{imgPtr};
    auto exposure = lsst::afw::image::makeExposure(mi);
    exposure->setPsf(psf);
    return centroidExposure(*exposure, lsst::geom::Point2D(image.getXY0()) + guess/sum);
}


template <typename T>
lsst::geom::Point2D centroidImage(
    lsst::afw::image::Image<T> const& image,
    float psfSigma,
    float nSigma
) {
    return centroidImage(image, makeGaussianPsf(psfSigma, image.getDimensions(), nSigma));
}


template <typename T>
lsst::geom::Point2D centroidImage(
    lsst::afw::image::Image<T> const& image,
    float nSigma
) {
    // Determine starting centroid guess and PSF size
    std::size_t const width = image.getWidth();
    std::size_t const height = image.getHeight();
    auto const& array = image.getArray();
    lsst::geom::Extent2D guess{0, 0};
    double sum = 0;
    for (std::size_t yy = 0; yy < height; ++yy) {
        for (std::size_t xx = 0; xx < width; ++xx) {
            double const value = array[yy][xx];
            guess += value*lsst::geom::Extent2D(xx, yy);
            sum += value;
        }
    }
    guess /= sum;

    double x2Sum = 0;
    double y2Sum = 0;
    for (std::size_t yy = 0; yy < height; ++yy) {
        double const dy = yy - guess.getY();
        double const y2 = std::pow(dy, 2);
        for (std::size_t xx = 0; xx < width; ++xx) {
            double const dx = xx - guess.getX();
            double const value = array[yy][xx];
            x2Sum += std::pow(dx, 2)*value;
            y2Sum += y2*value;
        }
    }
    double const xSigma = std::sqrt(x2Sum/sum);
    double const ySigma = std::sqrt(y2Sum/sum);
    double const psfSigma = 0.5*(xSigma + ySigma);

    auto imgPtr = std::make_shared<lsst::afw::image::Image<float>>(image, !std::is_same<T, float>::value);
    lsst::afw::image::MaskedImage<float, lsst::afw::image::MaskPixel, float> mi{imgPtr};
    auto exposure = lsst::afw::image::makeExposure(mi);
    exposure->setPsf(makeGaussianPsf(psfSigma, image.getDimensions(), nSigma));
    return centroidExposure(*exposure, lsst::geom::Point2D(image.getXY0()) + guess);
}


// Explicit instantiation
#define INSTANTIATE(TYPE) \
    template lsst::geom::Point2D centroidImage<TYPE>( \
        lsst::afw::image::Image<TYPE> const& image, \
        std::shared_ptr<lsst::afw::detection::Psf> psf \
    ); \
    template lsst::geom::Point2D centroidImage<TYPE>( \
        lsst::afw::image::Image<TYPE> const& image, \
        float psfSigma, \
        float nSigma \
    ); \
    template lsst::geom::Point2D centroidImage<TYPE>( \
        lsst::afw::image::Image<TYPE> const& image, \
        float nSigma \
    );


INSTANTIATE(double);


}}}  // namespace pfs::drp::stella
