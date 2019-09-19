#include <algorithm>

#include "lsst/afw/math/LeastSquares.h"

#include "pfs/drp/stella/FiberTraceSet.h"
#include "pfs/drp/stella/math/Math.h"
#include "pfs/drp/stella/math/symmetricTridiagonal.h"

//#define __DEBUG_FINDANDTRACE__ 1

namespace afwImage = lsst::afw::image;

namespace pfs { namespace drp { namespace stella {


template<typename ImageT, typename MaskT, typename VarianceT>
FiberTraceSet<ImageT, MaskT, VarianceT>::FiberTraceSet(
    FiberTraceSet<ImageT, MaskT, VarianceT> const &other,
    bool const deep
) : _traces(other.getInternal()),
    _metadata(deep ? other._metadata : std::make_shared<lsst::daf::base::PropertyList>()) {
    if (deep) {
        // Replace entries in the collection with copies
        for (auto tt : _traces) {
            tt = std::make_shared<FiberTraceT>(*tt, true);
        }
    }
}


template<typename ImageT, typename MaskT, typename VarianceT>
void FiberTraceSet<ImageT, MaskT, VarianceT >::sortTracesByXCenter()
{
    std::size_t const num = _traces.size();
    std::vector<float> xCenters(num);
    std::transform(_traces.begin(), _traces.end(), xCenters.begin(),
                   [](std::shared_ptr<FiberTraceT> ft) {
                       auto const& box = ft->getTrace().getBBox();
                       return 0.5*(box.getMinX() + box.getMaxX()); });
    std::vector<std::size_t> indices = math::sortIndices(xCenters);

    Collection sorted(num);
    std::transform(indices.begin(), indices.end(), sorted.begin(),
                   [this](std::size_t ii) { return _traces[ii]; });
    _traces = std::move(sorted);
}


template<typename ImageT, typename MaskT, typename VarianceT>
SpectrumSet FiberTraceSet<ImageT, MaskT, VarianceT>::extractSpectra(
    lsst::afw::image::MaskedImage<ImageT, MaskT, VarianceT> const& image,
    MaskT badBitMask
) {
    std::size_t const num = size();
    std::size_t const height = image.getHeight();
    SpectrumSet result{num, height};
    MaskT const require = image.getMask()->getPlaneBitMask(fiberMaskPlane);

    ndarray::Array<bool, 1, 1> useTrace = ndarray::allocate(num);  // trace overlaps this row?
    ndarray::Array<MaskT, 1, 1> maskResult = ndarray::allocate(num);  // mask value for each trace
    ndarray::Array<double, 1, 1> vector = ndarray::allocate(num);  // least-squares: model dot data

    // The least-squares matrix should be very close to tri-diagonal, because each trace principally
    // interacts with only with its neighbour.
    // Since it's also symmetric, we can represent the matrix with just two vectors: one for the diagonal,
    // and one for the off-diagonal.
    // diagonal[i] is model[i]^2
    // offDiag[i] is model[i]*model[i+1]
    // We calculate two separate versions of the matrix: one without weighting (diagonal+offDiag) and another
    // with weighting by the inverse variance (diagonalWeighted+offDiagWeighted). This allows us to use the
    // un-weighted matrix for solving for the fluxes without a bias (same reason we don't weight by the
    // variance when doing PSF photometry) and the inverse of the weighted matrix to get an estimate of the
    // variance and co-variance.
    ndarray::Array<double, 1, 1> diagonal = ndarray::allocate(num);  // diagonal of least-squares
    ndarray::Array<double, 1, 1> offDiag = ndarray::allocate(num - 1);  // off-diagonal of least-squares
    ndarray::Array<double, 1, 1> diagonalWeighted = ndarray::allocate(num);  // diagonal of weighted LS
    ndarray::Array<double, 1, 1> offDiagWeighted = ndarray::allocate(num - 1);  // off-diagonal of weighted LS
    math::SymmetricTridiagonalWorkspace<double> solutionWorkspace, inversionWorkspace;

    auto const& dataImage = *image.getImage();
    auto const& dataMask = *image.getMask();
    auto const& dataVariance = *image.getVariance();

    for (std::size_t yData = 0; yData < height; ++yData) {
         // Determine which traces are relevant for this row
        for (std::size_t ii = 0; ii < num; ++ii) {
            auto const& box = _traces[ii]->getTrace().getBBox();
            useTrace[ii] = (yData >= std::size_t(box.getMinY()) && yData <= std::size_t(box.getMaxY()));
        }

        // Construct least-squares matrix and vector
        for (std::size_t ii = 0; ii < num; ++ii) {
            if (!useTrace[ii]) {
                diagonal[ii] = 1.0;  // to avoid making the matrix singular
                diagonalWeighted[ii] = 1.0;
                vector[ii] = 0.0;
                if (ii < num - 1) {
                    offDiag[ii] = 0.0;
                    offDiagWeighted[ii] = 0.0;
                }
                continue;
            }

            double model2 = 0.0;  // model^2
            double model2Weighted = 0.0;  // model^2/sigma^2
            double modelData = 0.0;  // model dot data
            auto const& iTrace = _traces[ii]->getTrace();
            auto const iyModel = yData - iTrace.getBBox().getMinY();
            std::size_t const ixMin = iTrace.getBBox().getMinX();
            std::size_t const ixMax = iTrace.getBBox().getMaxX();
            auto const& iModelImage = *iTrace.getImage();
            auto const& iModelMask = *iTrace.getMask();
            maskResult[ii] = 0;
            for (std::size_t xModel = 0, xData = ixMin; xModel < std::size_t(iTrace.getWidth());
                 ++xModel, ++xData) {
                maskResult[ii] |= dataMask(xData, yData);
                if (dataMask(xData, yData) & badBitMask) continue;
                if (!(iModelMask(xModel, iyModel) & require)) continue;
                double const modelValue = iModelImage(xModel, iyModel);
                double const m2 = std::pow(modelValue, 2);
                model2 += m2;
                model2Weighted += m2/dataVariance(xData, yData);
                modelData += modelValue*dataImage(xData, yData);
            }
            diagonal[ii] = model2;
            diagonalWeighted[ii] = model2Weighted;
            vector[ii] = modelData;

            if (ii >= num - 1) {
                continue;
            }

            std::size_t const jj = ii + 1;
            if (!useTrace[jj]) {
                offDiag[ii] = 0.0;
                offDiagWeighted[ii] = 0.0;
                continue;
            }

            auto const& jTrace = _traces[jj]->getTrace();
            auto const& jModelImage = *jTrace.getImage();
            auto const& jModelMask = *jTrace.getMask();

            // Determine overlap
            std::size_t const jxMin = jTrace.getBBox().getMinX();
            std::size_t const jxMax = jTrace.getBBox().getMaxX();
            std::size_t const jyModel = yData - jTrace.getBBox().getMinY();
            //   |----------------|
            // ixMin            ixMax
            //            |----------------|
            //          jxMin            jxMax
            std::size_t const overlapMin = std::max(ixMin, jxMin);
            std::size_t const overlapMax = std::min(ixMax, jxMax);

            // Accumulate in overlap
            double modelModel = 0.0;  // model_i dot model_j
            double modelModelWeighted = 0.0;  // model_i dot model_j/sigma^2
            for (std::size_t xData = overlapMin, xi = overlapMin - ixMin, xj = overlapMin - jxMin;
                    xData <= overlapMax; ++xData, ++xi, ++xj) {
                if (!(iModelMask(xi, iyModel) & require)) continue;
                if (!(jModelMask(xj, jyModel) & require)) continue;
                if (dataMask(xData, yData) & badBitMask) continue;
                double const mm = double(iModelImage(xi, iyModel))*double(jModelImage(xj, jyModel));
                modelModel += mm;
                modelModelWeighted += mm/dataVariance(xData, yData);
            }
            offDiag[ii] = modelModel;
            offDiagWeighted[ii] = modelModelWeighted;
        }

        // Solve least-squares and set results
        ndarray::Array<double, 1, 1> solution = math::solveSymmetricTridiagonal(diagonal, offDiag, vector,
                                                                                solutionWorkspace);
        ndarray::Array<double, 1, 1> variance;
        ndarray::Array<double, 1, 1> covariance;
        std::tie(variance, covariance) = math::invertSymmetricTridiagonal(diagonalWeighted, offDiagWeighted,
                                                                          inversionWorkspace);
        MaskT const noData = 1 << image.getMask()->addMaskPlane("NO_DATA");
        for (std::size_t ii = 0; ii < num; ++ii) {
            auto value = solution[ii];
            if (!std::isfinite(value)) {
                value = 0.0;
                maskResult[ii] |= noData;
            }
            result[ii]->getSpectrum()[yData] = value;
            result[ii]->getMask()(yData, 0) = maskResult[ii];
            result[ii]->getCovariance()[0][yData] = variance[ii];
            result[ii]->getCovariance()[1][yData] = (ii < num - 1) ? covariance[ii] : 0.0;
            result[ii]->getCovariance()[2][yData] = (ii > 0) ? covariance[ii - 1] : 0.0;
        }
    }

    for (std::size_t ii = 0; ii < num; ++ii) {
        result[ii]->setFiberId(_traces[ii]->getFiberId());
    }

    return result;
}


// Explicit instantiation
template class FiberTraceSet<float, lsst::afw::image::MaskPixel, float>;

}}} // namespace pfs::drp::stella
