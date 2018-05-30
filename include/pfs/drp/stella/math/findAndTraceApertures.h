#ifndef PFS_DRP_STELLA_MATH_FINDANDTRACEAPERTURES_H
#define PFS_DRP_STELLA_MATH_FINDANDTRACEAPERTURES_H

#include <vector>

#include "ndarray_fwd.h"

#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/geom/Point.h"

#include "pfs/drp/stella/DetectorMap.h"
#include "pfs/drp/stella/Controls.h"
#include "pfs/drp/stella/FiberTraces.h"

namespace pfs {
namespace drp {
namespace stella {
namespace math {

/**
 * @brief identifies and traces the fiberTraces in maskedImage, and extracts them into individual FiberTraces
 * FiberTraces in returned FiberTraceSet will be sorted by their xCenter positions
 * Set I_NTermsGaussFit to
 *       1 to look for maximum only without GaussFit
 *       3 to fit Gaussian
 *       4 to fit Gaussian plus constant (sky)
 *         Spatial profile must be at least 5 pixels wide
 *       5 to fit Gaussian plus linear term (sloped sky)
 *         Spatial profile must be at least 6 pixels wide
 * NOTE that the WCS starts at [0., 0.], so an xCenter of 1.1 refers to position 0.1 of the second pixel
 * @param maskedImage : MaskedImage in which to find and trace the FiberTraces
 * @param fiberTraceFunctionFindingControl : Control to be used in task
 **/
template<typename ImageT, typename MaskT=lsst::afw::image::MaskPixel,
         typename VarianceT=lsst::afw::image::VariancePixel>
FiberTraceSet<ImageT, MaskT, VarianceT>
findAndTraceApertures(
    lsst::afw::image::MaskedImage<ImageT, MaskT, VarianceT> const& maskedImage,
    DetectorMap const& detectorMap,
    FiberTraceFunctionFindingControl const& finding,
    FiberTraceProfileFittingControl const& fitting
);


struct FindCenterPositionsOneTraceResult {
    std::vector<float> index;
    std::vector<float> position;
    std::vector<float> error;
    lsst::afw::geom::Point<int> nextSearchStart;
};

/**
 * @brief: traces the fiberTrace closest to the bottom of the image and sets it to zero
 * @param ccdImage: image to trace
 * @param ccdImageVariance: variance of image to trace (used for fitting)
 * @param fiberTraceFunctionFindingControl: parameters to find and trace a fiberTrace
 * */
template<typename ImageT, typename VarianceT=lsst::afw::image::VariancePixel>
FindCenterPositionsOneTraceResult findCenterPositionsOneTrace(
    lsst::afw::image::Image<ImageT> & image,
    lsst::afw::image::Image<VarianceT> const& variance,
    FiberTraceFunctionFindingControl const& finding,
    lsst::afw::geom::Point<int> const& nextSearchStart
);

/**
 * @brief: returns ndarray containing the xCenters of a FiberTrace from 0 to FiberTrace.getTrace().getHeight()-1
 *         NOTE that the WCS starts at [0., 0.], so an xCenter of 1.1 refers to position 0.1 of the second pixel
 * @param function : FiberTraceFunction to use when calculating the xCenters
 */
ndarray::Array<float, 1, 1>
calculateXCenters(
    FiberTraceFunction const& function
);

/**
 * @brief: returns ndarray containing the xCenters of a FiberTrace from 0 to FiberTrace.getTrace().getHeight()-1
 *         NOTE that the WCS starts at [0., 0.], so an xCenter of 1.1 refers to position 0.1 of the second pixel
 * @param function : FiberTraceFunction to use when calculating the xCenters
 * @param yIn : This range in y will be converted to [-1.0,1.0] when calculating the xCenters
 */
ndarray::Array<float, 1, 1>
calculateXCenters(
    pfs::drp::stella::FiberTraceFunction const& function,
    ndarray::Array<float, 1, 1> const& yIn
);

}}}} // namespace pfs::drp::stella::math

#endif // include guard