#include <cmath>
#include <vector>

#include "lsst/pex/exceptions.h"
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
        lsst::afw::image::Mask<lsst::afw::image::MaskPixel>::getPlaneBitMask("NO_DATA"),
        std::numeric_limits<float>::quiet_NaN()
    };
    return {ctrl, pad};
}


/// Fill in the non-finite points of the grid for the warp
///
/// The positions in the "to" detectorMap can be non-finite where the
/// wavelength is off the end of the fiber's trace (e.g., when the shift
/// between the detectorMaps is large). Non-finite points would break the
/// GridTransform, so we replace them using the shift (to minus from) of the
/// nearest good point on the same fiber or, if the fiber has no good points,
/// of the same knot on the nearest fiber with good points. The shift varies
/// slowly, and such points are near the edge of the detector.
///
/// @param xFrom, yFrom : positions in the "from" detectorMap (all finite)
/// @param xTo, yTo : positions in the "to" detectorMap; modified
void fillGrid(
    ndarray::Array<double, 2, 2> const& xFrom,
    ndarray::Array<double, 2, 2> const& yFrom,
    ndarray::Array<double, 2, 2> & xTo,
    ndarray::Array<double, 2, 2> & yTo
) {
    std::size_t const numFibers = xFrom.getShape()[0];
    std::size_t const numKnots = xFrom.getShape()[1];
    auto const isGood = [&](std::size_t ii, std::size_t jj) {
        return std::isfinite(xTo[ii][jj]) && std::isfinite(yTo[ii][jj]);
    };

    std::vector<bool> fiberGood(numFibers, false);
    std::vector<std::size_t> good;
    good.reserve(numKnots);
    for (std::size_t ii = 0; ii < numFibers; ++ii) {
        good.clear();
        for (std::size_t jj = 0; jj < numKnots; ++jj) {
            if (isGood(ii, jj)) {
                good.push_back(jj);
            }
        }
        if (good.empty()) {
            continue;
        }
        fiberGood[ii] = true;
        if (good.size() == numKnots) {
            continue;
        }
        for (std::size_t jj = 0; jj < numKnots; ++jj) {
            if (isGood(ii, jj)) {
                continue;
            }
            std::size_t nearest = good.front();
            for (std::size_t kk : good) {
                if (std::abs(double(kk) - double(jj)) < std::abs(double(nearest) - double(jj))) {
                    nearest = kk;
                }
            }
            xTo[ii][jj] = xFrom[ii][jj] + xTo[ii][nearest] - xFrom[ii][nearest];
            yTo[ii][jj] = yFrom[ii][jj] + yTo[ii][nearest] - yFrom[ii][nearest];
        }
    }

    std::vector<std::size_t> goodFibers;
    for (std::size_t ii = 0; ii < numFibers; ++ii) {
        if (fiberGood[ii]) {
            goodFibers.push_back(ii);
        }
    }
    if (goodFibers.empty()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::RuntimeError,
            "No fiber has positions in both detectorMaps for the warp"
        );
    }
    for (std::size_t ii = 0; ii < numFibers; ++ii) {
        if (fiberGood[ii]) {
            continue;
        }
        std::size_t nearest = goodFibers.front();
        for (std::size_t kk : goodFibers) {
            if (std::abs(double(kk) - double(ii)) < std::abs(double(nearest) - double(ii))) {
                nearest = kk;
            }
        }
        for (std::size_t jj = 0; jj < numKnots; ++jj) {
            xTo[ii][jj] = xFrom[ii][jj] + xTo[nearest][jj] - xFrom[nearest][jj];
            yTo[ii][jj] = yFrom[ii][jj] + yTo[nearest][jj] - yFrom[nearest][jj];
        }
    }
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

    // Calculate the transformation grid, using only fibers with a trace across the "from" detectorMap
    double const spacing = fromDetectorMap.getBBox().getHeight() / (numWavelengthKnots - 1);
    std::vector<int> fiberIds;
    fiberIds.reserve(fromDetectorMap.getNumFibers());
    for (int fiberId : fromDetectorMap.getFiberId()) {
        bool good = true;
        for (int jj = 0; jj < numWavelengthKnots && good; ++jj) {
            double const row = fromDetectorMap.getBBox().getMinY() + jj * spacing;
            good = std::isfinite(fromDetectorMap.getXCenter(fiberId, row)) &&
                std::isfinite(fromDetectorMap.getWavelength(fiberId, row));
        }
        if (good) {
            fiberIds.push_back(fiberId);
        }
    }
    std::size_t const numFibers = fiberIds.size();
    if (numFibers < 2) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::RuntimeError,
            "Insufficient fibers with traces across the detectorMap for the warp"
        );
    }
    ndarray::Array<double, 2, 2> xFrom = ndarray::allocate(numFibers, numWavelengthKnots);
    ndarray::Array<double, 2, 2> yFrom = ndarray::allocate(numFibers, numWavelengthKnots);
    ndarray::Array<double, 2, 2> xTo = ndarray::allocate(numFibers, numWavelengthKnots);
    ndarray::Array<double, 2, 2> yTo = ndarray::allocate(numFibers, numWavelengthKnots);
    for (std::size_t ii = 0; ii < numFibers; ++ii) {
        int const fiberId = fiberIds[ii];
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
    fillGrid(xFrom, yFrom, xTo, yTo);
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
