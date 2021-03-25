#include <algorithm>

#include "lsst/afw/math/LeastSquares.h"

#include "pfs/drp/stella/FiberTraceSet.h"
#include "pfs/drp/stella/math/symmetricTridiagonal.h"

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
    MaskT badBitMask
) {
    std::size_t const num = size();
    std::size_t const height = image.getHeight();
    std::size_t const width = image.getWidth();
    SpectrumSet result{num, height};
    MaskT const require = image.getMask()->getPlaneBitMask(fiberMaskPlane);
    std::size_t const x0 = image.getX0();
    std::size_t const y0 = image.getY0();

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

    MaskT const noData = 1 << dataMask.addMaskPlane("NO_DATA");
    MaskT const badFiberTrace = 1 << dataMask.addMaskPlane("BAD_FIBERTRACE");

    // Initialize results, in case we miss anything
    for (auto & spectrum : result) {
        spectrum->getSpectrum().deep() = 0.0;
        spectrum->getMask().getArray().deep() = spectrum->getMask().getPlaneBitMask("NO_DATA");
        spectrum->getCovariance().deep() = 0.0;
    }

    std::size_t yStart = std::max(y0, 0UL);
    for (std::size_t yData = yStart - y0, yActual = yStart; yData < height; ++yData, ++yActual) {
         // Determine which traces are relevant for this row
        for (std::size_t ii = 0; ii < num; ++ii) {
            auto const& box = _traces[ii]->getTrace().getBBox();
            useTrace[ii] = (yActual >= std::size_t(box.getMinY()) && yActual <= std::size_t(box.getMaxY()));
            maskResult[ii] = noData;
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
            auto const iyModel = yActual - iTrace.getBBox().getMinY();
            std::size_t const ixMin = iTrace.getBBox().getMinX();
            std::size_t const ixMax = iTrace.getBBox().getMaxX();
            auto const& iModelImage = *iTrace.getImage();
            auto const& iModelMask = *iTrace.getMask();
            maskResult[ii] = 0;
            std::size_t const xStart = std::max(ixMin, x0);
            for (std::size_t xModel = xStart - ixMin, xData = xStart - x0;
                 xModel < std::size_t(iTrace.getWidth()) && xData < width;
                 ++xModel, ++xData) {
                if (dataMask(xData, yData) & badBitMask) continue;
                if (!(iModelMask(xModel, iyModel) & require)) continue;
                double const modelValue = iModelImage(xModel, iyModel);
                if (modelValue == 0.0) continue;  // don't accumulate the mask
                maskResult[ii] |= dataMask(xData, yData);
                double const m2 = std::pow(modelValue, 2);
                model2 += m2;
                model2Weighted += m2/dataVariance(xData, yData);
                modelData += modelValue*dataImage(xData, yData);
            }

            if (model2 == 0.0) {
                useTrace[ii] = false;
                diagonal[ii] = 1.0;  // to avoid making the matrix singular
                diagonalWeighted[ii] = 1.0;
                vector[ii] = 0.0;
                if (ii < num - 1) {
                    offDiag[ii] = 0.0;
                    offDiagWeighted[ii] = 0.0;
                }
                maskResult[ii] |= noData | badFiberTrace;
                continue;
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
            std::size_t const overlapStart = std::max(overlapMin, x0);

            // Accumulate in overlap
            double modelModel = 0.0;  // model_i dot model_j
            double modelModelWeighted = 0.0;  // model_i dot model_j/sigma^2
            for (std::size_t xData = overlapStart - x0, xi = overlapStart - ixMin, xj = overlapStart - jxMin;
                    xData <= overlapMax - x0; ++xData, ++xi, ++xj) {
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

#if 0
        if (yData == 0 || yData == 1) {
            std::cerr << "Diagonal: " << diagonal << std::endl;
            std::cerr << "OffDiag: " << offDiag << std::endl;
            std::cerr << "Vector: " << vector << std::endl;
        }
#endif

        // Solve least-squares and set results
        ndarray::Array<double, 1, 1> solution = math::solveSymmetricTridiagonal(diagonal, offDiag, vector,
                                                                                solutionWorkspace);
        ndarray::Array<double, 1, 1> variance;
        ndarray::Array<double, 1, 1> covariance;
        std::tie(variance, covariance) = math::invertSymmetricTridiagonal(diagonalWeighted, offDiagWeighted,
                                                                          inversionWorkspace);
        for (std::size_t ii = 0; ii < num; ++ii) {
            auto value = solution[ii];
            auto varResult = variance[ii];
            auto covarResult1 = (ii < num - 1 && useTrace[ii + 1]) ? covariance[ii] : 0.0;
            auto covarResult2 = (ii > 0 && useTrace[ii - 1]) ? covariance[ii - 1] : 0.0;
            if (!useTrace[ii] || !std::isfinite(value)) {
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
#if 0
    std::cerr << "xy0: " << _traces[0]->getTrace().getBBox() << " " << image.getBBox() << std::endl;
    std::cerr << "Trace: " << _traces[0]->getTrace().getImage()->getArray()[0] << std::endl;
    std::cerr << "Trace: " << _traces[0]->getTrace().getImage()->getArray()[1] << std::endl;
    std::cerr << "Image: " << image.getImage()->getArray()[0] << std::endl;
    std::cerr << "Image: " << image.getImage()->getArray()[1] << std::endl;
    std::cerr << "Spectrum: " << result[0]->getSpectrum()[0] << std::endl;
    std::cerr << "Spectrum: " << result[0]->getSpectrum()[1] << std::endl;
#endif

    for (std::size_t ii = 0; ii < num; ++ii) {
        result[ii]->setFiberId(_traces[ii]->getFiberId());
    }

    return result;
}


// Explicit instantiation
template class FiberTraceSet<float, lsst::afw::image::MaskPixel, float>;

}}} // namespace pfs::drp::stella
