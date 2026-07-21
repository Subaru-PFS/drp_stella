#include "lsst/afw/math/warpExposure.h"
#include "lsst/afw/math/detail/WarpAtOnePoint.h"

#include "pfs/drp/stella/utils/checkSize.h"
#include "pfs/drp/stella/GridTransform.h"
#include "pfs/drp/stella/warp.h"



namespace pfs {
namespace drp {
namespace stella {


namespace {


std::pair<lsst::afw::math::WarpingControl, lsst::afw::image::MaskedImage<float>::SinglePixel>
makeWarpingControl(std::string const &warpingKernelName) {
    lsst::afw::math::WarpingControl ctrl(warpingKernelName, "bilinear");
    lsst::afw::image::MaskedImage<float>::SinglePixel pad = {
        std::numeric_limits<float>::quiet_NaN(),
        lsst::afw::image::Mask<lsst::afw::image::MaskPixel>::getMaskPlane("NO_DATA"),
        std::numeric_limits<float>::quiet_NaN()
    };
    return {ctrl, pad};
}


}  // anonymous namespace


lsst::afw::image::MaskedImage<float> warpFiber(
    lsst::afw::image::MaskedImage<float> const& image,
    pfs::drp::stella::DetectorMap const& detectorMap,
    int fiberId,
    int halfWidth,
    std::string const &warpingKernelName
) {
    auto ctrl = makeWarpingControl(warpingKernelName);
    return warpFiber(image, detectorMap, fiberId, halfWidth, ctrl.first, ctrl.second);
}


lsst::afw::image::MaskedImage<float> warpFiber(
    lsst::afw::image::MaskedImage<float> const& image,
    pfs::drp::stella::DetectorMap const& detectorMap,
    int fiberId,
    int halfWidth,
    lsst::afw::math::WarpingControl const& ctrl,
    lsst::afw::image::MaskedImage<float>::SinglePixel const& pad
) {
    using ImageT = lsst::afw::image::MaskedImage<float>;

    ImageT out(2*halfWidth + 1, image.getHeight());
    lsst::afw::math::detail::WarpAtOnePoint<ImageT, ImageT> warp(image, ctrl, pad);

    for (int yy = 0; yy < out.getHeight(); ++yy) {
        auto iter = out.row_begin(yy);
        double const xCenter = detectorMap.getXCenter(fiberId, yy);
        double position = xCenter - halfWidth;
        for (int xx = 0; xx < out.getWidth(); ++xx, position += 1.0, ++iter) {
            warp(iter, lsst::geom::Point2D(position, yy), 1.0, lsst::afw::image::detail::MaskedImage_tag());
        }
    }

    out.setXY0(-halfWidth, 0);
    return out;
}


lsst::afw::image::MaskedImage<float> warpImage(
    lsst::afw::image::MaskedImage<float> const& fromImage,
    pfs::drp::stella::DetectorMap const& fromDetectorMap,
    pfs::drp::stella::DetectorMap const& toDetectorMap,
    std::string const &warpingKernelName,
    int numWavelengthKnots
) {
    auto ctrl = makeWarpingControl(warpingKernelName);
    return warpImage(fromImage, fromDetectorMap, toDetectorMap, ctrl.first, ctrl.second, numWavelengthKnots);
}


lsst::afw::image::MaskedImage<float> warpImage(
    lsst::afw::image::MaskedImage<float> const& fromImage,
    pfs::drp::stella::DetectorMap const& fromDetectorMap,
    pfs::drp::stella::DetectorMap const& toDetectorMap,
    lsst::afw::math::WarpingControl const& ctrl,
    lsst::afw::image::MaskedImage<float>::SinglePixel const& pad,
    int numWavelengthKnots
) {
    utils::checkSize(fromDetectorMap.getNumFibers(), toDetectorMap.getNumFibers(), "numFibers");

    using ImageT = lsst::afw::image::MaskedImage<float>;

    ImageT out(toDetectorMap.getBBox().getDimensions());
    out.setXY0(toDetectorMap.getBBox().getMin());

    lsst::afw::math::detail::WarpAtOnePoint<ImageT, ImageT> warp(fromImage, ctrl, pad);

    // Calculate the transformation grid
    std::size_t const numFibers = fromDetectorMap.getNumFibers();
    ndarray::Array<double, 2, 2> xFrom = ndarray::allocate(numFibers, numWavelengthKnots);
    ndarray::Array<double, 2, 2> yFrom = ndarray::allocate(numFibers, numWavelengthKnots);
    ndarray::Array<double, 2, 2> xTo = ndarray::allocate(numFibers, numWavelengthKnots);
    ndarray::Array<double, 2, 2> yTo = ndarray::allocate(numFibers, numWavelengthKnots);
    for (std::size_t ii = 0; ii < numFibers; ++ii) {
        int const fiberId = fromDetectorMap.getFiberId()[ii];
        double const spacing = fromDetectorMap.getBBox().getHeight() / (numWavelengthKnots - 1);
        for (int jj = 0; jj < numWavelengthKnots; ++jj) {
            double const row = fromDetectorMap.getBBox().getMinY() + jj * spacing;
            xFrom[ii][jj] = fromDetectorMap.getXCenter(fiberId, row);
            yFrom[ii][jj] = row;
            double const wavelength = fromDetectorMap.getWavelength(fiberId, row);
            auto const point = toDetectorMap.findPoint(fiberId, wavelength);
            xTo[ii][jj] = point.getX();
            yTo[ii][jj] = point.getY();
        }
    }
    GridTransform transform(xTo, yTo, xFrom, yFrom);

    for (int yy = 0; yy < out.getHeight(); ++yy) {
        auto iter = out.row_begin(yy);
        for (int xx = 0; xx < out.getWidth(); ++xx, ++iter) {
            warp(
                iter,
                transform(xx, yy),
                transform.calculateRelativeArea(xx, yy),  // from/to
                lsst::afw::image::detail::MaskedImage_tag()
            );
        }
    }
    return out;
}


}}}  // namespace pfs::drp::stella
