#include <numeric>
#include <algorithm>

#include "lsst/afw/math/LeastSquares.h"

#include "pfs/drp/stella/FiberTraceSet.h"
#include "pfs/drp/stella/math/symmetricTridiagonal.h"
#include "pfs/drp/stella/math/SparseSquareMatrix.h"

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


namespace {

// Functor for comparing vector values by their indices
template <typename T>
struct IndicesComparator {
    std::vector<T> const& data;
    IndicesComparator(std::vector<T> const& data_) : data(data_) {}
    bool operator()(std::size_t lhs, std::size_t rhs) const {
        return data[lhs] < data[rhs];
    }
};


/**
 * Returns an integer array of the same size like <data>,
 * containing the indixes of <data> in sorted order.
 *
 * @param[in] data       vector to sort
 **/
template<typename T>
std::vector<std::size_t> sortIndices(std::vector<T> const& data) {
    std::size_t const num = data.size();
    std::vector<std::size_t> indices(num);
    std::size_t index = 0;
    std::generate_n(indices.begin(), num, [&index]() { return index++; });
    std::sort(indices.begin(), indices.end(), IndicesComparator<T>(data));
    return indices;
}

}  // anonymous namespace


template<typename ImageT, typename MaskT, typename VarianceT>
void FiberTraceSet<ImageT, MaskT, VarianceT >::sortTracesByXCenter()
{
    std::size_t const num = _traces.size();
    std::vector<float> xCenters(num);
    std::transform(_traces.begin(), _traces.end(), xCenters.begin(),
                   [](std::shared_ptr<FiberTraceT> ft) {
                       auto const& box = ft->getTrace().getBBox();
                       return 0.5*(box.getMinX() + box.getMaxX()); });
    std::vector<std::size_t> indices = sortIndices(xCenters);

    Collection sorted(num);
    std::transform(indices.begin(), indices.end(), sorted.begin(),
                   [this](std::size_t ii) { return _traces[ii]; });
    _traces = std::move(sorted);
}


template<typename ImageT, typename MaskT, typename VarianceT>
SpectrumSet FiberTraceSet<ImageT, MaskT, VarianceT>::extractSpectra(
    lsst::afw::image::MaskedImage<ImageT, MaskT, VarianceT> const& image,
    MaskT badBitMask,
    float minFracMask,
    float minFracImage
) const {
    std::size_t const num = size();
    std::size_t const height = image.getHeight();
    std::size_t const width = image.getWidth();
    SpectrumSet result{num, height};
    MaskT const require = image.getMask()->getPlaneBitMask(fiberMaskPlane);
    std::ptrdiff_t const x0 = image.getX0();
    std::ptrdiff_t const y0 = image.getY0();

    ndarray::Array<bool, 1, 1> useTrace = ndarray::allocate(num);  // trace overlaps this row?
    ndarray::Array<MaskT, 1, 1> maskResult = ndarray::allocate(num);  // mask value for each trace
    ndarray::Array<MaskT, 1, 1> maskBadResult = ndarray::allocate(num);  // mask value if trace is bad
    ndarray::Array<double, 1, 1> vector = ndarray::allocate(num);  // least-squares: model dot data

    // This is a version of the least-squares matrix used for calculating the covariances.
    // We use a simple diagnonal matrix because we don't care about anything further than that.
    // The matrix is calculated using inverse variance weights, so we can get the covariance.
    ndarray::Array<double, 1, 1> diagonalWeighted = ndarray::allocate(num);  // diagonal of weighted LS

    auto const& dataImage = *image.getImage();
    auto const& dataMask = *image.getMask();
    auto const& dataVariance = *image.getVariance();

    MaskT const noData = 1 << dataMask.addMaskPlane("NO_DATA");
    MaskT const badFiberTrace = 1 << dataMask.addMaskPlane("BAD_FIBERTRACE");
    MaskT const suspect = 1 << dataMask.addMaskPlane("SUSPECT");

    // Initialize results, in case we miss anything
    for (auto & spectrum : result) {
        spectrum->getFlux().deep() = 0.0;
        spectrum->getMask().getArray().deep() = spectrum->getMask().getPlaneBitMask("NO_DATA");
        spectrum->getVariance().deep() = 0.0;
        spectrum->getNorm().deep() = 0.0;
    }

    // yData is the position on the image (and therefore the extracted spectrum)
    // yActual is the position on the trace
    std::ptrdiff_t yActual = y0;
    for (std::size_t yData = 0; yData < height; ++yData, ++yActual) {
         // Determine which traces are relevant for this row
        for (std::size_t ii = 0; ii < num; ++ii) {
            auto const& box = _traces[ii]->getTrace().getBBox();
            useTrace[ii] = (yActual >= box.getMinY() && yActual <= box.getMaxY());
            maskResult[ii] = 0;  // used in case of success
            maskBadResult[ii] = noData;  // used in case of failure

            if (useTrace[ii]) {
                std::size_t const yy = yData - box.getMinY();
                auto const& traceImage = _traces[ii]->getTrace();
                double sumModel = 0.0;
                for (auto iter = traceImage.row_begin(yy); iter != traceImage.row_end(yy); ++iter) {
                    if (iter.mask() & require) {
                        sumModel += iter.image();
                    }
                }
                result[ii]->getNorm()[yData] = sumModel;
                if (sumModel == 0 || !std::isfinite(sumModel)) {
                    useTrace[ii] = false;
                    maskBadResult[ii] |= badFiberTrace;
                }
            }
        }

        // Construct least-squares matrix and vector
        math::SymmetricSparseSquareMatrix matrix{num};  // least-squares matrix: model dot model
        vector.deep() = 0.0;
        diagonalWeighted.deep() = 0.0;

        for (std::size_t ii = 0; ii < num; ++ii) {
            if (!useTrace[ii]) {
                matrix.add(ii, ii, 1.0);  // to avoid making the matrix singular
                diagonalWeighted[ii] = 1.0;
                continue;
            }

            // Here we calculate two versions of the matrix diagonal: one
            // without weighting and another with weighting by the inverse
            // variance. This allows us to use the un-weighted matrix for
            // solving for the fluxes without a bias (same reason we don't
            // weight by the variance when doing PSF photometry) and the inverse
            // of the weighted matrix diagonal to get an estimate of the
            // variance.
            double model2 = 0.0;  // model^2
            double model2Weighted = 0.0;  // model^2/sigma^2
            double modelData = 0.0;  // model dot data
            double sumModel = 0.0;  // sum of model values
            auto const& iTrace = _traces[ii]->getTrace();
            auto const iyModel = yActual - iTrace.getBBox().getMinY();
            assert(iTrace.getBBox().getMinX() >= 0 && iTrace.getBBox().getMinY() >= 0);
            std::size_t const ixMin = iTrace.getBBox().getMinX();
            std::size_t const ixMax = iTrace.getBBox().getMaxX();
            auto const& iModelImage = *iTrace.getImage();
            auto const& iModelMask = *iTrace.getMask();
            double const iNorm = result[ii]->getNorm()[yData];
            std::size_t numTracePixels = 0;
            std::size_t const xStart = std::max(std::ptrdiff_t(ixMin), x0);
            for (std::size_t xModel = xStart - ixMin, xData = xStart - x0;
                 xModel < std::size_t(iTrace.getWidth()) && xData < width;
                 ++xModel, ++xData) {
                if (!(iModelMask(xModel, iyModel) & require)) continue;
                ++numTracePixels;
                double const modelValue = iModelImage(xModel, iyModel)/iNorm;
                MaskT const maskValue = dataMask(xData, yData);
                ImageT const imageValue = dataImage(xData, yData);
                VarianceT const varianceValue = dataVariance(xData, yData);
                if ((maskValue & badBitMask) || !std::isfinite(imageValue) || !std::isfinite(varianceValue) ||
                    varianceValue <= 0) {
                    if (modelValue > minFracMask) {
                        maskBadResult[ii] |= maskValue;
                    }
                    continue;
                }
                if (modelValue > minFracMask) {
                    maskResult[ii] |= maskValue;
                }
                double const m2 = std::pow(modelValue, 2);
                model2 += m2;
                model2Weighted += m2/varianceValue;
                modelData += modelValue*imageValue;
                sumModel += modelValue;
            }

            if (sumModel == 0.0 || numTracePixels == 0 || model2 == 0.0 || model2Weighted == 0.0) {
                useTrace[ii] = false;
                matrix.add(ii, ii, 1.0);  // to avoid making the matrix singular
                diagonalWeighted[ii] = 1.0;
                continue;
            } else if (sumModel < minFracImage) {
                maskResult[ii] |= suspect;
                maskBadResult[ii] |= suspect;
            }

            matrix.add(ii, ii, model2);
            diagonalWeighted[ii] = model2Weighted;
            vector[ii] = modelData;

            if (ii >= num - 1) {
                continue;
            }

            for (std::size_t jj = ii + 1; jj < num; ++jj) {
                if (!useTrace[jj]) {
                    continue;
                }

                auto const& jTrace = _traces[jj]->getTrace();
                auto const& jModelImage = *jTrace.getImage();
                auto const& jModelMask = *jTrace.getMask();
                double const jNorm = result[jj]->getNorm()[yData];

                // Determine overlap
                assert(jTrace.getBBox().getMinX() >= 0 && jTrace.getBBox().getMinY() >= 0);
                std::size_t const jxMin = jTrace.getBBox().getMinX();
                std::size_t const jxMax = jTrace.getBBox().getMaxX();
                std::size_t const jyModel = yData - jTrace.getBBox().getMinY();
                //   |----------------|
                // ixMin            ixMax
                //            |----------------|
                //          jxMin            jxMax
                std::size_t const overlapMin = std::max(ixMin, jxMin);
                std::size_t const overlapMax = std::min(ixMax, jxMax);
                std::size_t const overlapStart = std::max(std::ptrdiff_t(overlapMin), x0);

                if (overlapMax < overlapMin) {
                    // No more overlaps
                    break;
                }

                // Accumulate in overlap
                double modelModel = 0.0;  // model_i dot model_j
                for (std::size_t xData = overlapStart - x0,
                        xi = overlapStart - ixMin, xj = overlapStart - jxMin;
                        xData <= overlapMax - x0;
                        ++xData, ++xi, ++xj) {
                    if (!(iModelMask(xi, iyModel) & require)) continue;
                    if (!(jModelMask(xj, jyModel) & require)) continue;
                    MaskT const maskValue = dataMask(xData, yData);
                    ImageT const imageValue = dataImage(xData, yData);
                    VarianceT const varianceValue = dataVariance(xData, yData);
                    if ((maskValue & badBitMask) || !std::isfinite(imageValue) ||
                        !std::isfinite(varianceValue) || varianceValue == 0) {
                        continue;
                    }
                    double const iModel = iModelImage(xi, iyModel)/iNorm;
                    double const jModel = jModelImage(xj, jyModel)/jNorm;
                    double const mm = iModel*jModel;
                    modelModel += mm;
                }
                matrix.add(ii, jj, modelModel);
            }
        }

        // Solve least-squares and set results
        ndarray::Array<double, 1, 1> solution = matrix.solve(vector);
        for (std::size_t ii = 0; ii < num; ++ii) {
            auto value = solution[ii];
            auto variance = 1.0/diagonalWeighted[ii];
            MaskT maskValue = maskResult[ii];
            if (!useTrace[ii]) {
                maskValue = maskBadResult[ii];
                variance = 0.0;
            }
            if (!std::isfinite(value)) {
                value = 0.0;
                maskValue = maskBadResult[ii];
            }
            if (!std::isfinite(variance) || variance <= 0) {
                variance = 0.0;
                maskValue = maskBadResult[ii];
            }
            result[ii]->getFlux()[yData] = value;
            result[ii]->getMask()(yData, 0) = maskValue;
            result[ii]->getVariance()[yData] = variance;
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
