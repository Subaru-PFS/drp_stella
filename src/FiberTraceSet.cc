#include <numeric>
#include <algorithm>

#include "ndarray/eigen.h"
#include "lsst/afw/math/LeastSquares.h"

#include "pfs/drp/stella/FiberTraceSet.h"
#include "pfs/drp/stella/math/symmetricTridiagonal.h"
#include "pfs/drp/stella/math/SparseSquareMatrix.h"
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


template<typename ImageT, typename MaskT, typename VarianceT>
SpectrumSet FiberTraceSet<ImageT, MaskT, VarianceT>::extractSpectra(
    lsst::afw::image::MaskedImage<ImageT, MaskT, VarianceT> const& image,
    MaskT badBitMask,
    float minFracMask,
    lsst::geom::Extent2I const& bgSize
) const {
    std::size_t const numFibers = size();
    std::size_t const height = image.getHeight();
    std::size_t const width = image.getWidth();
    SpectrumSet result{numFibers, height};
    MaskT const require = image.getMask()->getPlaneBitMask(fiberMaskPlane);
    std::ptrdiff_t const x0 = image.getX0();
    std::ptrdiff_t const y0 = image.getY0();

    // Assign each pixel to a background super-pixel
    bool doBackground = false;
    int numBackground = 0;
    ndarray::Array<int, 2, 1> bgIndices;
    // if (bgSize.getX() != 0 && bgSize.getY() != 0) {
    //     doBackground = true;
    //     numBackground = getNumBackgroundIndices(lsst::geom::Extent2I(width, height), bgSize);
    //     bgIndices = getBackgroundIndices2d(lsst::geom::Extent2I(width, height), bgSize, 0);
    // }

    std::size_t const bgOffset = numFibers*height;
//    std::size_t const bgScale = bgNum.getX();
//    auto const getBgIndex = [&xIndex, &yIndex, bgOffset, bgScale](std::ptrdiff_t xx, std::ptrdiff_t yy) {
//        return bgOffset + yIndex[yy]*bgScale + xIndex[xx];
//    };
    auto const getSpectrumIndex = [numFibers](std::size_t fiberIndex, std::ptrdiff_t yy) {
        return yy*numFibers + fiberIndex;
    };

    auto const& dataImage = *image.getImage();
    auto const& dataMask = *image.getMask();
    auto const& dataVariance = *image.getVariance();

    MaskT const noData = 1 << dataMask.addMaskPlane("NO_DATA");
    MaskT const badFiberTrace = 1 << dataMask.addMaskPlane("BAD_FIBERTRACE");

    // Initialize results, in case we miss anything
    for (std::size_t ii = 0; ii < numFibers; ++ii) {
        auto & spectrum = *result[ii];

        spectrum.getSpectrum().deep() = 0.0;
        spectrum.getMask().getArray().deep() = spectrum.getMask().getPlaneBitMask("NO_DATA");
        spectrum.getCovariance().deep() = 0.0;
        spectrum.getNorm().deep() = 0.0;

        auto const trace = ndarray::asEigenArray(_traces[ii]->getTrace().getImage()->getArray());
        auto const bbox = _traces[ii]->getTrace().getBBox();
        auto norm = ndarray::asEigenArray(spectrum.getNorm()[
            ndarray::view(bbox.getMinY(), bbox.getMaxY() + 1)
        ]);
        norm = trace.template cast<double>().rowwise().sum().template cast<float>();
    }

    // Determine bounds of each trace
    ndarray::Array<int, 1, 1> lower = ndarray::allocate(numFibers);
    ndarray::Array<int, 1, 1> upper = ndarray::allocate(numFibers);
    for (std::size_t ii = 0; ii < numFibers; ++ii) {
        auto const& box = _traces[ii]->getTrace().getBBox();
        lower[ii] = box.getMinX();
        upper[ii] = box.getMaxX();
    }

    std::size_t const numParameters = numFibers*height + numBackground;
    math::SymmetricSparseSquareMatrix matrix{numParameters};
    ndarray::Array<double, 1, 1> vector = ndarray::allocate(numParameters);
    vector.deep() = 0.0;
    ndarray::Array<bool, 1, 1> isZero = ndarray::allocate(numParameters);
    isZero.deep() = true;

    // yData is the position on the image (and therefore the extracted spectrum)
    // yActual is the position on the trace
    std::ptrdiff_t yActual = y0;
    for (std::size_t yData = 0; yData < height; ++yData, ++yActual) {
        std::ptrdiff_t xActual = x0;
        std::size_t left = 0;  // fiber index of lower bound of consideration for this pixel (inclusive)
        std::size_t right = 0;  // fiber index of upper bound of consideration for this pixel (exclusive)
        for (std::size_t xData = 0; xData < width; ++xData, ++xActual) {

            // Which fibers are we dealing with?
            // pixel:                X
            // fiber 0: |-----------|
            // fiber 1:     |-----------|
            // fiber 2:         |-----------|
            // fiber 3:             |-----------|
            // fiber 4:                 |-----------|
            // Our pixel (X, at top) overlaps 1,2,3; therefore get left=1 (inclusive), right=4 (exclusive)
            while (left < numFibers && xActual >= upper[left]) ++left;
            while (right < numFibers && xActual >= lower[right]) ++right;

            // if (yData == 123) std::cerr << xData << " " << left << " " << right << std::endl;

            ImageT const imageValue = dataImage(xData, yData);
            MaskT const maskValue = dataMask(xData, yData);
            VarianceT const varianceValue = dataVariance(xData, yData);
            double const invVariance = 1.0/varianceValue;

            // We don't immediately move on from pixels that are masked, because
            // we want to collect the mask value, which is only done if the
            // trace exceeds minFracMask; so we need to iterate over fibers.
            bool const isRejected = (
                (maskValue & badBitMask) ||
                !std::isfinite(imageValue) ||
                !std::isfinite(varianceValue) ||
                varianceValue <= 0
            );

            std::size_t bgIndex;
            if (!isRejected && doBackground) {
                bgIndex = bgIndices[yData][xData];
                matrix.add(bgIndex, bgIndex, invVariance);
                vector[bgIndex] += imageValue*invVariance;
                isZero[bgIndex] = false;
            }

            for (std::size_t iFiber = left; iFiber < right; ++iFiber) {
                auto const& iTrace = _traces[iFiber]->getTrace();
                auto const ixModel = xActual - iTrace.getBBox().getMinX();
                auto const iyModel = yActual - iTrace.getBBox().getMinY();
                assert(iTrace.getBBox().getMinX() >= 0 && iTrace.getBBox().getMinY() >= 0);
                auto const& iModelImage = *iTrace.getImage();
                auto const& iModelMask = *iTrace.getMask();
                if (!(iModelMask(ixModel, iyModel) & require)) {
                    result[iFiber]->getMask()(yData, 0) |= badFiberTrace;
                    continue;
                }
                double const iNorm = result[iFiber]->getNorm()[yData];
                double const iModelValue = iModelImage(ixModel, iyModel)/iNorm;
                if (!std::isfinite(iModelValue) || iNorm == 0) {
                    result[iFiber]->getMask()(yData, 0) |= badFiberTrace;
                    continue;
                }
                if (iModelValue > minFracMask) {
                    result[iFiber]->getMask()(yData, 0) |= maskValue;
                }
                if (isRejected) continue;
                std::size_t const iIndex = getSpectrumIndex(iFiber, yData);

                matrix.add(iIndex, iIndex, iModelValue*iModelValue*invVariance);
                vector[iIndex] += iModelValue*imageValue*invVariance;
                isZero[iIndex] = false;

                if (doBackground) {
                    matrix.add(iIndex, bgIndex, iModelValue*invVariance);
                }

                for (std::size_t jFiber = iFiber + 1; jFiber < right; ++jFiber) {
                    auto const& jTrace = _traces[jFiber]->getTrace();
                    auto const jxModel = xActual - jTrace.getBBox().getMinX();
                    auto const jyModel = yActual - jTrace.getBBox().getMinY();
                    assert(jTrace.getBBox().getMinX() >= 0 && jTrace.getBBox().getMinY() >= 0);
                    auto const& jModelImage = *jTrace.getImage();
                    auto const& jModelMask = *jTrace.getMask();
                    if (!(jModelMask(jxModel, jyModel) & require)) continue;
                    double const jNorm = result[jFiber]->getNorm()[yData];
                    double const jModelValue = jModelImage(jxModel, iyModel)/jNorm;
                    std::size_t const jIndex = getSpectrumIndex(jFiber, yData);

                    matrix.add(iIndex, jIndex, iModelValue*jModelValue*invVariance);
                }
            }
        }
    }

    for (std::size_t ii = 0; ii < numParameters; ++ii) {
        if (isZero[ii]) {
            matrix.add(ii, ii, 1.0);  // to avoid making the matrix singular
        }
    }

    // Solve least-squares and set results
    ndarray::Array<double, 1, 1> solution = matrix.solve(vector, true);
//    std::cerr << "Extraction background: " << solution[ndarray::view(bgOffset, solution.size())] << std::endl;

    yActual = y0;
    for (std::size_t yData = 0; yData < height; ++yData, ++yActual) {
        for (std::size_t iFiber = 0; iFiber < numFibers; ++iFiber) {
            std::size_t const iIndex = getSpectrumIndex(iFiber, yData);
            result[iFiber]->getFlux()[yActual] = solution[iIndex];
            result[iFiber]->getVariance()[yActual] = 1.0/matrix.get(iIndex, iIndex);
        }
    }

    for (std::size_t iFiber = 0; iFiber < numFibers; ++iFiber) {
        result[iFiber]->setFiberId(_traces[iFiber]->getFiberId());
    }

    return result;
}


// Explicit instantiation
template class FiberTraceSet<float, lsst::afw::image::MaskPixel, float>;

}}} // namespace pfs::drp::stella
