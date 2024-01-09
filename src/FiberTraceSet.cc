#include <numeric>
#include <algorithm>

#include "lsst/afw/math/LeastSquares.h"

#include "pfs/drp/stella/FiberTraceSet.h"
#include "pfs/drp/stella/math/symmetricTridiagonal.h"
#include "pfs/drp/stella/math/LeastSquaresEquation.h"
#include "pfs/drp/stella/backgroundIndices.h"

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

//#pragma GCC optimize("O0")

template<typename ImageT, typename MaskT, typename VarianceT>
SpectrumSet FiberTraceSet<ImageT, MaskT, VarianceT>::extractSpectra(
    lsst::afw::image::MaskedImage<ImageT, MaskT, VarianceT> const& image,
    MaskT badBitMask,
    float minFracMask,
    lsst::geom::Extent2I const& bgSize
) const {
    std::size_t const numFibers = size();
    auto const dims = image.getDimensions();
    int const height = image.getHeight();
    int const width = image.getWidth();
    SpectrumSet result{numFibers, std::size_t(height)};
    MaskT const require = image.getMask()->getPlaneBitMask(fiberMaskPlane);
    std::ptrdiff_t const y0 = image.getY0();
    std::size_t const numBackground = getNumBackgroundIndices(dims, bgSize);
    bool const doBackground = (numBackground > 0);
    ndarray::Array<int, 2, 1> const bgIndices = doBackground ?
        calculateBackgroundIndices(dims, bgSize, numFibers*height) : ndarray::Array<int, 2, 1>();

    ndarray::Array<bool, 1, 1> useTrace = ndarray::allocate(numFibers);  // trace overlaps this row?
    ndarray::Array<MaskT, 1, 1> maskResult = ndarray::allocate(numFibers);  // mask value for each trace

    // This is a version of the least-squares matrix used for calculating the covariances.
    // We use a symmetric tridiagonal matrix (represented by diagonal and off-diagonal arrays) because
    // we don't care about anything further than that. The matrix is calculated using inverse variance
    // weights, so we can get the covariance.
    ndarray::Array<double, 1, 1> diagonalWeighted = ndarray::allocate(numFibers);  // diagonal of weighted LS
    ndarray::Array<double, 1, 1> offDiagWeighted = ndarray::allocate(numFibers - 1);  // off-diagonal of WLS
    math::SymmetricTridiagonalWorkspace<double> solutionWorkspace, inversionWorkspace;

    auto const& dataImage = *image.getImage();
    auto const& dataMask = *image.getMask();
    auto const& dataVariance = *image.getVariance();

    MaskT const noData = 1 << dataMask.addMaskPlane("NO_DATA");
    MaskT const badFiberTrace = 1 << dataMask.addMaskPlane("BAD_FIBERTRACE");

    // Initialize results, in case we miss anything
    for (auto & spectrum : result) {
        spectrum->getSpectrum().deep() = 0.0;
        spectrum->getMask().getArray().deep() = 0;
        spectrum->getCovariance().deep() = 0.0;
        spectrum->getNorm().deep() = 0.0;
    }

    std::ptrdiff_t const num = numFibers*height + numBackground;  // total number of parameters
    math::LeastSquaresEquation equation{num};

    ndarray::Array<int, 1, 1> lower = ndarray::allocate(numFibers);  // lower bound pixel in trace (inclusive)
    ndarray::Array<int, 1, 1> upper = ndarray::allocate(numFibers);  // upper bound pixel in trace (inclusive)
    for (std::size_t ii = 0; ii < numFibers; ++ii) {
        auto const& box = _traces[ii]->getTrace().getBBox();
        lower[ii] = box.getMinX();
        upper[ii] = box.getMaxX();
        if (ii > 0 && (lower[ii] < lower[ii - 1] || upper[ii] < upper[ii - 1])) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::InvalidParameterError,
                "Traces are not sorted by x-center; use method sortTracesByXCenter()"
            );
        }
    }

    // yData is the position on the image (and therefore the extracted spectrum)
    // yActual is the position on the trace
    std::ptrdiff_t yActual = y0;
    for (int yData = 0; yData < height; ++yData, ++yActual) {
         // Determine which traces are relevant for this row
        for (std::size_t ii = 0; ii < numFibers; ++ii) {
            auto const& box = _traces[ii]->getTrace().getBBox();
            useTrace[ii] = (yActual >= box.getMinY() && yActual <= box.getMaxY());
            maskResult[ii] = 0;

            if (useTrace[ii]) {
                std::size_t const yy = yData - box.getMinY();
                auto const& trace = ndarray::asEigenArray(_traces[ii]->getTrace().getImage()->getArray()[yy]);
                result[ii]->getNorm()[yData] = trace.template cast<double>().sum();
                if (result[ii]->getNorm()[yData] == 0) {
                    std::cerr << "Zero norm: " << ii << " " << yData << std::endl;
                    useTrace[ii] = false;
                    maskResult[ii] |= badFiberTrace;
                }
            }
        }

        // Construct least-squares matrix and vector for error estimation
        diagonalWeighted.deep() = 0.0;
        offDiagWeighted.deep() = 0.0;

        std::size_t left = 0;  // index of left-most fiber for pixel (inclusive)
        std::size_t right = 0;  // index of right-most fiber for pixel (exclusive)
        for (int xData = 0; xData < width; ++xData) {
            while (left < numFibers && xData >= upper[left]) {
                ++left;
            }
            while (right < numFibers && xData >= lower[right]) {
                ++right;
            }

            MaskT const maskValue = dataMask(xData, yData);
            ImageT const imageValue = dataImage(xData, yData);
            VarianceT const varianceValue = dataVariance(xData, yData);

            // We don't immediately move on to the next pixel if the pixel is masked, because we
            // want to accumulate the mask values for the traces that overlap this pixel.
            bool const isBad = ((maskValue & badBitMask) || !std::isfinite(imageValue) ||
                                 !std::isfinite(varianceValue) || varianceValue <= 0);

            std::size_t const bgIndex = doBackground ? bgIndices[yData][xData] : 0;
            if (doBackground && !isBad) {
                equation.addDiagonal(bgIndex, 1.0);
                equation.addVector(bgIndex, imageValue);
            }

            std::size_t iIndex = yData*numFibers + left;
            for (std::size_t ii = left; ii < right; ++ii, ++iIndex) {
                if (!useTrace[ii]) {
                    diagonalWeighted[ii] = 1.0;
                    continue;
                }

                auto const& iTrace = _traces[ii]->getTrace();
                int const ixModel = xData - iTrace.getBBox().getMinX();
                auto const iyModel = yActual - iTrace.getBBox().getMinY();
                assert(iTrace.getBBox().getMinX() >= 0 && iTrace.getBBox().getMinY() >= 0);
                auto const& iModelImage = *iTrace.getImage();
                auto const& iModelMask = *iTrace.getMask();
                double const iNorm = result[ii]->getNorm()[yData];
                double const iModelValue = iModelImage(ixModel, iyModel)/iNorm;
                if (!(iModelMask(ixModel, iyModel) & require)) continue;
                if (iModelValue > minFracMask) {
                    maskResult[ii] |= maskValue;
                }
                if (isBad) {
                    continue;
                }

                double const model2 = std::pow(iModelValue, 2);
                equation.addDiagonal(iIndex, model2);
                equation.addVector(iIndex, iModelValue*imageValue);
                diagonalWeighted[ii] += model2/varianceValue;
                if (doBackground) {
                    equation.addOffDiagonal(iIndex, bgIndex, iModelValue);
                }

                for (std::size_t jj = ii + 1, jIndex = iIndex + 1; jj < right; ++jj, ++jIndex) {
                    if (!useTrace[jj]) {
                        continue;
                    }
                    auto const& jTrace = _traces[jj]->getTrace();
                    int const jxModel = xData - jTrace.getBBox().getMinX();
                    auto const jyModel = yActual - jTrace.getBBox().getMinY();
                    auto const& jModelImage = *jTrace.getImage();
                    auto const& jModelMask = *jTrace.getMask();
                    double const jNorm = result[jj]->getNorm()[yData];
                    double const jModelValue = jModelImage(jxModel, jyModel)/jNorm;
                    if (!(jModelMask(jxModel, jyModel) & require)) continue;

                    double const modelModel = iModelValue*jModelValue;
                    equation.addOffDiagonal(iIndex, jIndex, modelModel);
                    if (jj == ii + 1) {
                        offDiagWeighted[ii] += modelModel/varianceValue;
                    }
                }
            }
        }

        for (std::size_t ii = 0; ii < numFibers; ++ii) {
            if (diagonalWeighted[ii] == 0.0) {
                diagonalWeighted[ii] = 1.0;  // to avoid making the matrix singular
                maskResult[ii] |= badFiberTrace;
            }
        }

        // Variances are measured row by row
        ndarray::Array<double, 1, 1> variance;
        ndarray::Array<double, 1, 1> covariance;
        std::tie(variance, covariance) = math::invertSymmetricTridiagonal(diagonalWeighted, offDiagWeighted,
                                                                          inversionWorkspace);
        for (std::size_t ii = 0; ii < numFibers; ++ii) {
            auto varResult = variance[ii];
            auto covarResult1 = (ii < numFibers - 1 && useTrace[ii + 1]) ? covariance[ii] : 0.0;
            auto covarResult2 = (ii > 0 && useTrace[ii - 1]) ? covariance[ii - 1] : 0.0;
            if (!useTrace[ii] || !std::isfinite(varResult)) {
                maskResult[ii] |= noData;
                if (!useTrace[ii]) {
                    maskResult[ii] |= badFiberTrace;
                }
                std::cerr << "Bad trace: " << yData << " " << ii << " " << useTrace[ii] << " " << varResult << " " << maskResult[ii] << std::endl;
                varResult = 0.0;
                covarResult1 = 0.0;
                covarResult2 = 0.0;
            }
            result[ii]->getMask()(yData, 0) |= maskResult[ii];
            result[ii]->getCovariance()[0][yData] = varResult;
            result[ii]->getCovariance()[1][yData] = covarResult1;
            result[ii]->getCovariance()[2][yData] = covarResult2;
        }
    }

    auto const nonZero = equation.makeNonSingular();

    // Solve least-squares and set results
    using Solver = Eigen::SimplicialLDLT<
        math::LeastSquaresEquation::SparseMatrix,
        Eigen::Upper,
        Eigen::NaturalOrdering<typename math::LeastSquaresEquation::IndexT>
    >;

    ndarray::Array<double, 1, 1> solution = equation.solve<Eigen::Upper, Solver>();

    for (int yData = 0; yData < height; ++yData) {
        for (std::size_t ii = 0, iIndex = yData*numFibers; ii < numFibers; ++ii, ++iIndex) {
            double const value = solution[iIndex];
            bool const isBad = nonZero.find(iIndex) != nonZero.end();
            result[ii]->getSpectrum()[yData] = value;
            if (!std::isfinite(value) || isBad) {
                result[ii]->getMask()(yData, 0) |= noData;
//                std::cerr << "Bad value: " << ii << " " << yData << " " << value << " " << result[ii]->getMask()(yData, 0) << std::endl;
            }
            if (result[ii]->getNorm()[yData] == 0) {
                result[ii]->getMask()(yData, 0) |= badFiberTrace;
//                std::cerr << "Bad norm: " << ii << " " << yData << " " << value << " " << result[ii]->getMask()(yData, 0) << std::endl;
            }
        }
    }

    for (std::size_t ii = 0; ii < numFibers; ++ii) {
        result[ii]->setFiberId(_traces[ii]->getFiberId());
    }

    std::cerr << "Background: " << solution[ndarray::view(numFibers*height, num)] << std::endl;

    return result;
}


// Explicit instantiation
template class FiberTraceSet<float, lsst::afw::image::MaskPixel, float>;

}}} // namespace pfs::drp::stella
