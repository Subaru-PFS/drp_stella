#include <numeric>
#include <algorithm>

#include "lsst/afw/math/LeastSquares.h"

#include "pfs/drp/stella/FiberTraceSet.h"
#include "pfs/drp/stella/math/symmetricTridiagonal.h"
#include "pfs/drp/stella/math/SparseSquareMatrix.h"

#include "pfs/drp/stella/utils/timer.h"


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
    float minFracMask
) const {
    utils::TimerSet timers;
    std::size_t const num = size();
    std::size_t const height = image.getHeight();
    std::size_t const width = image.getWidth();
    SpectrumSet result{num, height};
    MaskT const require = image.getMask()->getPlaneBitMask(fiberMaskPlane);
    std::ptrdiff_t const x0 = image.getX0();
    std::ptrdiff_t const y0 = image.getY0();

    ndarray::Array<bool, 1, 1> useTrace = ndarray::allocate(num);  // trace overlaps this row?
    ndarray::Array<MaskT, 1, 1> maskResult = ndarray::allocate(num);  // mask value for each trace
    ndarray::Array<double, 1, 1> vector = ndarray::allocate(num);  // least-squares: model dot data

    // This is a version of the least-squares matrix used for calculating the covariances.
    // We use a symmetric tridiagonal matrix (represented by diagonal and off-diagonal arrays) because
    // we don't care about anything further than that. The matrix is calculated using inverse variance
    // weights, so we can get the covariance.
    ndarray::Array<double, 1, 1> diagonalWeighted = ndarray::allocate(num);  // diagonal of weighted LS
    ndarray::Array<double, 1, 1> offDiagWeighted = ndarray::allocate(num - 1);  // off-diagonal of weighted LS
    math::SymmetricTridiagonalWorkspace<double> solutionWorkspace, inversionWorkspace;

    auto const& dataImage = *image.getImage();
    auto const& dataMask = *image.getMask();
    auto const& dataVariance = *image.getVariance();

    MaskT const noData = 1 << dataMask.addMaskPlane("NO_DATA");
    MaskT const badFiberTrace = 1 << dataMask.addMaskPlane("BAD_FIBERTRACE");

    // Initialize results, in case we miss anything
    for (auto & spectrum : result) {
        spectrum->getSpectrum().deep() = 0.0;
        spectrum->getMask().getArray().deep() = spectrum->getMask().getPlaneBitMask("NO_DATA");
        spectrum->getCovariance().deep() = 0.0;
        spectrum->getNorm().deep() = 0.0;
    }

    // yData is the position on the image (and therefore the extracted spectrum)
    // yActual is the position on the trace
    std::ptrdiff_t yActual = y0;
    for (std::size_t yData = 0; yData < height; ++yData, ++yActual) {
         // Determine which traces are relevant for this row
        for (std::size_t ii = 0; ii < num; ++ii) {
            auto ttt = timers.context("trace");
            auto const& box = _traces[ii]->getTrace().getBBox();
            useTrace[ii] = (yActual >= box.getMinY() && yActual <= box.getMaxY());
            maskResult[ii] = noData;

            if (useTrace[ii]) {
                std::size_t const yy = yData - box.getMinY();
                auto const& trace = ndarray::asEigenArray(_traces[ii]->getTrace().getImage()->getArray()[yy]);
                result[ii]->getNorm()[yData] = trace.template cast<double>().sum();
                if (result[ii]->getNorm()[yData] == 0) {
                    useTrace[ii] = false;
                    maskResult[ii] |= badFiberTrace;
                }
            }
        }

        // Construct least-squares matrix and vector
        math::SymmetricSparseSquareMatrix matrix{num};  // least-squares matrix: model dot model
        vector.deep() = 0.0;
        diagonalWeighted.deep() = 0.0;
        offDiagWeighted.deep() = 0.0;

        for (std::size_t ii = 0; ii < num; ++ii) {
            auto ttt = timers.context("construct matrix");
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
            auto const& iTrace = _traces[ii]->getTrace();
            auto const iyModel = yActual - iTrace.getBBox().getMinY();
            assert(iTrace.getBBox().getMinX() >= 0 && iTrace.getBBox().getMinY() >= 0);
            std::size_t const ixMin = iTrace.getBBox().getMinX();
            std::size_t const ixMax = iTrace.getBBox().getMaxX();
            auto const& iModelImage = *iTrace.getImage();
            auto const& iModelMask = *iTrace.getMask();
            double const iNorm = result[ii]->getNorm()[yData];
            maskResult[ii] = 0;
            std::size_t const xStart = std::max(std::ptrdiff_t(ixMin), x0);
            for (std::size_t xModel = xStart - ixMin, xData = xStart - x0;
                 xModel < std::size_t(iTrace.getWidth()) && xData < width;
                 ++xModel, ++xData) {
                if (!(iModelMask(xModel, iyModel) & require)) continue;
                double const modelValue = iModelImage(xModel, iyModel)/iNorm;
                MaskT const maskValue = dataMask(xData, yData);
                ImageT const imageValue = dataImage(xData, yData);
                VarianceT const varianceValue = dataVariance(xData, yData);
                if (modelValue > minFracMask) {
                    maskResult[ii] |= maskValue;
                }
                if ((maskValue & badBitMask) || !std::isfinite(imageValue) || !std::isfinite(varianceValue) ||
                    varianceValue <= 0) {
                    continue;
                }
                double const m2 = std::pow(modelValue, 2);
                model2 += m2;
                model2Weighted += m2/varianceValue;
                modelData += modelValue*imageValue;
            }

            if (model2 == 0.0 || model2Weighted == 0.0) {
                useTrace[ii] = false;
                matrix.add(ii, ii, 1.0);  // to avoid making the matrix singular
                diagonalWeighted[ii] = 1.0;
                maskResult[ii] |= noData | badFiberTrace;
                continue;
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
                double modelModelWeighted = 0.0;  // model_i dot model_j/sigma^2
                bool accumulateWeighted = (jj == ii + 1);  // do we want to calculate modelModelWeighted?
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
                    if (accumulateWeighted) {
                        modelModelWeighted += mm/varianceValue;
                    }
                }
                matrix.add(ii, jj, modelModel);
                if (accumulateWeighted) {
                    offDiagWeighted[ii] = modelModelWeighted;
                }
            }
        }

        auto ttt = timers.context("solve matrix");

        // Solve least-squares and set results
        ndarray::Array<double, 1, 1> solution = matrix.solve(vector);

        ndarray::Array<double, 1, 1> variance;
        ndarray::Array<double, 1, 1> covariance;
        std::tie(variance, covariance) = math::invertSymmetricTridiagonal(diagonalWeighted, offDiagWeighted,
                                                                          inversionWorkspace);
        for (std::size_t ii = 0; ii < num; ++ii) {
            auto value = solution[ii];
            auto varResult = variance[ii];
            auto covarResult1 = (ii < num - 1 && useTrace[ii + 1]) ? covariance[ii] : 0.0;
            auto covarResult2 = (ii > 0 && useTrace[ii - 1]) ? covariance[ii - 1] : 0.0;
            if (!useTrace[ii] || !std::isfinite(value) || !std::isfinite(varResult)) {
                value = 0.0;
                maskResult[ii] |= noData;
                varResult = 0.0;
                covarResult1 = 0.0;
                covarResult2 = 0.0;
            }
            result[ii]->getSpectrum()[yData] = value;
            result[ii]->getMask()(yData, 0) = maskResult[ii];
            result[ii]->getCovariance()[0][yData] = varResult;
            result[ii]->getCovariance()[1][yData] = covarResult1;
            result[ii]->getCovariance()[2][yData] = covarResult2;
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
