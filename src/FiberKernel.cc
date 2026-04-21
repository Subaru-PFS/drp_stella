#include <chrono>
#include <string>
#include <unordered_map>

#include "ndarray.h"
#include "ndarray/eigen.h"

#include "lsst/afw/math/LeastSquares.h"

#include "pfs/drp/stella/FiberKernel.h"
#include "pfs/drp/stella/utils/checkSize.h"
#include "pfs/drp/stella/math/NormalizedPolynomial.h"
#include "pfs/drp/stella/utils/math.h"
#include "pfs/drp/stella/math/SparseSquareMatrix.h"

namespace pfs {
namespace drp {
namespace stella {


namespace {


/// RAII timer: adds elapsed wall-clock time (seconds) into *target on destruction.
struct ScopedTimer {
    double* target;
    std::chrono::steady_clock::time_point start;

    explicit ScopedTimer(double* target_)
            : target(target_), start(std::chrono::steady_clock::now()) {}

    ScopedTimer(ScopedTimer&& other) noexcept
            : target(other.target), start(other.start) {
        other.target = nullptr;  // disarm the moved-from object
    }

    ScopedTimer(ScopedTimer const&) = delete;
    ScopedTimer& operator=(ScopedTimer const&) = delete;
    ScopedTimer& operator=(ScopedTimer&&) = delete;

    ~ScopedTimer() {
        if (target) {
            *target += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start
            ).count();
        }
    }
};


struct FiberModel {
    ndarray::Array<float const, 1, 1> values;
    ndarray::Array<bool const, 1, 1> use;
    int y;
    int xMin;  // relative to the image, inclusive
    int xMax;  // relative to the image, exclusive
    int width;
    int offset;

    static FiberModel fromFiberTrace(
        FiberTrace<float> const& fiberTrace,
        int y,
        int xStart,  // relative to trace bbox
        int xStop,  // relative to trace bbox, inclusive
        lsst::afw::image::MaskPixel requireMask
    ) {
        auto const& trace = fiberTrace.getTrace();
        ndarray::Array<bool, 1, 1> use = ndarray::allocate(xStop - xStart + 1);
        ndarray::asEigenArray(use) = ndarray::asEigenArray(
            trace.getMask()->getArray()[y][ndarray::view(xStart, xStop + 1)]
        ).unaryExpr(
            [requireMask](lsst::afw::image::MaskPixel mm) { return (mm & requireMask) != 0; }
        );
        return FiberModel{
            trace.getImage()->getArray()[y][ndarray::view(xStart, xStop + 1)],
            std::move(use),
            y,
            trace.getBBox().getMinX() + xStart,
            trace.getBBox().getMinX() + xStop + 1,
            xStop - xStart + 1,
            0
        };
    }

    static FiberModel fromImage(
        lsst::afw::image::MaskedImage<float> const& image,
        int y,
        lsst::afw::image::MaskPixel badBitMask
    ) {
        ndarray::Array<bool, 1, 1> use = ndarray::allocate(image.getWidth());
        ndarray::asEigenArray(use) = ndarray::asEigenArray(
            image.getMask()->getArray()[y]
        ).unaryExpr([badBitMask](lsst::afw::image::MaskPixel mm) { return (mm & badBitMask) == 0; });
        return FiberModel{
            image.getImage()->getArray()[y],
            use,
            y,
            0,
            image.getWidth(),
            image.getWidth(),
            0
        };
    }

    static FiberModel fromImage(
        lsst::afw::image::Image<float> const& image,
        int y
    ) {
        ndarray::Array<bool, 1, 1> use = ndarray::allocate(image.getWidth());
        use.deep() = true;
        return FiberModel{
            image.getArray()[y],
            use,
            y,
            0,
            image.getWidth(),
            image.getWidth(),
            0
        };
    }

    static FiberModel dummy() {
        return FiberModel{
            ndarray::Array<float const, 1, 1>(), ndarray::Array<bool const, 1, 1>(), 0, 0, 0, 0, 0
        };
    }

    FiberModel applyOffset(int offset, int width) const {
        if (offset == 0) {
            return *this;
        }
        int const newMin = std::max(0, xMin + std::min(0, offset));
        int const newMax = std::min(width, xMax + std::max(0, offset));  // exclusive
        int const newWidth = newMax - newMin;

        ndarray::Array<float, 1, 1> newValues = ndarray::allocate(newWidth);
        ndarray::Array<bool, 1, 1> newUse = ndarray::allocate(newWidth);
        newValues.deep() = 0.0;
        newUse.deep() = false;

        // Original model: 0.1, 0.8, 0.1
        // offset = -2 --> 0.1, 0.8, 0.0, -0.8, -0.1
        // offset = -1 --> 0.1, 0.7, -0.7, -0.1
        // offset = +1 --> -0.1, -0.7, 0.7, 0.1
        // offset = +2 --> -0.1, -0.8, 0.0, 0.8, 0.1

        // Insert the offset values
        {
            // Overlap of shifted model with image in image coordinates.
            int const shiftedMin = std::max(0, xMin + offset);
            int const shiftedMax = std::min(width, xMax + offset);  // exclusive
            // Source array indices
            int const start = shiftedMin - offset - xMin;
            int const stop = shiftedMax - offset - xMin;  // exclusive
            // Target array indices
            int const newStart = shiftedMin - newMin;
            int const newStop = shiftedMax - newMin;  // exclusive
            assert(newStart >= 0 && newStop <= newWidth);
            assert(start >= 0 && stop <= this->width);
            newValues[ndarray::view(newStart, newStop)] = values[ndarray::view(start, stop)];
            newUse[ndarray::view(newStart, newStop)] = use[ndarray::view(start, stop)];
        }

        // Subtract the non-offset values
        {
            // Relative positon on the target array for zero offset
            int const newStart = xMin - newMin;
            int const newStop = xMax - newMin;
            assert(newStart >= 0 && newStop <= newWidth);
            newValues[ndarray::view(newStart, newStop)] -= values;
            newUse[ndarray::view(newStart, newStop)] |= use;
        }

        return FiberModel{std::move(newValues), std::move(newUse), y, newMin, newMax, newWidth, offset};
    }

    template <typename T>
    FiberModel operator*(ndarray::ArrayRef<T, 1, 1> const& rhs) const {
        assert(rhs.size() == std::size_t(width));
        ndarray::Array<float, 1, 1> newValues = ndarray::copy(values);
        ndarray::asEigenArray(newValues) *= ndarray::asEigenArray(rhs).template cast<float>();
        return FiberModel{newValues, ndarray::copy(use), y, xMin, xMax, width, offset};
    }

    lsst::afw::image::MaskPixel accumulateMask(
        ndarray::Array<lsst::afw::image::MaskPixel const, 1, 1> const& dataMask,
        float threshold
    ) const {
        auto const modelGood = ndarray::asEigenArray(use);
        auto const modelAboveThreshold = ndarray::asEigenArray(values) > threshold;
        return (modelGood && modelAboveThreshold).select(
            ndarray::asEigenArray(dataMask[ndarray::view(xMin, xMax)]), 0
        ).redux([](auto left, auto right) { return left | right; });
    }

    std::size_t count(
        ndarray::Array<bool const, 1, 1> const& usePixels
    ) const {
        auto const dataGood = ndarray::asEigenArray(usePixels[ndarray::view(xMin, xMax)]);
        auto const modelGood = ndarray::asEigenArray(use);
        return (dataGood && modelGood).cast<std::size_t>().sum();
    }

    double sum() const {
        return ndarray::asEigenArray(use).select(
            ndarray::asEigenArray(values).template cast<double>(), 0.0
        ).sum();
    }

    double sum(ndarray::Array<bool const, 1, 1> const& usePixels) const {
        auto const dataGood = ndarray::asEigenArray(usePixels[ndarray::view(xMin, xMax)]);
        auto const modelGood = ndarray::asEigenArray(use);
        return (dataGood && modelGood).select(
            ndarray::asEigenArray(values).template cast<double>(), 0.0
        ).sum();
    }

    double centroid() const {
        double centroid = 0.0;
        double sum = 0.0;
        for (std::size_t ii = 0, xx = xMin; ii < values.size(); ++ii, ++xx) {
            if (use[ii]) {
                centroid += xx*values[ii];
                sum += values[ii];
            }
        }
        return centroid/sum;
    }

    double dotSelf() const {
        return ndarray::asEigenArray(use).select(
            ndarray::asEigenArray(values).template cast<double>(), 0.0
        ).square().sum();
    }

    double dotSelf(ndarray::Array<bool const, 1, 1> const& usePixels) const {
        auto const modelGood = ndarray::asEigenArray(use);
        auto const dataGood = ndarray::asEigenArray(usePixels[ndarray::view(xMin, xMax)]);
        return (modelGood && dataGood).select(
            ndarray::asEigenArray(values).template cast<double>(), 0.0
        ).square().sum();
    }

    double dotSelf(
        ndarray::Array<bool const, 1, 1> const& usePixels,
        ndarray::Array<float const, 1, 1> const& dataVariance
    ) const {
        auto const modelGood = ndarray::asEigenArray(use);
        auto const dataGood = ndarray::asEigenArray(usePixels[ndarray::view(xMin, xMax)]);
        auto const weights = (modelGood && dataGood).select(
            1.0/ndarray::asEigenArray(dataVariance[ndarray::view(xMin, xMax)]), 0.0
        ).template cast<double>();
        auto const modelValues = ndarray::asEigenArray(values).template cast<double>();
        return (weights*modelValues.square()).sum();
    }

    double dotData(
        ndarray::Array<float const, 1, 1> const& dataValues,
        ndarray::Array<bool const, 1, 1> const& usePixels,
        ndarray::Array<float const, 1, 1> const& dataVariance
    ) const {
        auto const modelGood = ndarray::asEigenArray(use);
        auto const dataGood = ndarray::asEigenArray(usePixels[ndarray::view(xMin, xMax)]);
        auto const weights = (modelGood && dataGood).select(
            1.0/ndarray::asEigenArray(dataVariance[ndarray::view(xMin, xMax)]), 0.0
        ).template cast<double>();
        auto const left = modelGood.select(
            ndarray::asEigenArray(values).template cast<double>(), 0.0
        );
        auto const right = dataGood.select(
            ndarray::asEigenArray(dataValues[ndarray::view(xMin, xMax)]).template cast<double>(), 0.0
        );
        return (weights*left*right).sum();
    }

    double dotOther(
        FiberModel const& other,
        ndarray::Array<bool const, 1, 1> const& usePixels,
        ndarray::Array<float const, 1, 1> const& dataVariance
    ) const {
        int const start = std::max(this->xMin, other.xMin);
        int const stop = std::min(this->xMax, other.xMax);  // exclusive
        if (stop <= start) {
            return 0.0;
        }
        int const thisStart = start - this->xMin;
        int const otherStart = start - other.xMin;
        int const thisStop = stop - this->xMin;  // exclusive
        int const otherStop = stop - other.xMin;  // exclusive

        auto const dataGood = ndarray::asEigenArray(usePixels[ndarray::view(start, stop)]);
        auto const useLeft = ndarray::asEigenArray(this->use[ndarray::view(thisStart, thisStop)]);
        auto const useRight = ndarray::asEigenArray(other.use[ndarray::view(otherStart, otherStop)]);
        auto const weights = (useLeft && useRight && dataGood).select(
            1.0/ndarray::asEigenArray(dataVariance[ndarray::view(start, stop)]), 0.0
        ).template cast<double>();

        auto const left = (useLeft && dataGood).select(
            ndarray::asEigenArray(
                this->values[ndarray::view(thisStart, thisStop)]
            ).template cast<double>(), 0.0
        );
        auto const right = (useRight && dataGood).select(
            ndarray::asEigenArray(
                other.values[ndarray::view(otherStart, otherStop)]
            ).template cast<double>(), 0.0
        );
        return (weights*left*right).sum();
    }

    std::pair<int, ndarray::Array<double, 1, 1>> dotBackground(
        ndarray::Array<int const, 1, 1> const& blocks,
        ndarray::Array<bool const, 1, 1> const& usePixels,
        ndarray::Array<float const, 1, 1> const& dataVariance
    ) const {
        int const imageWidth = usePixels.size();
        int const xStart = std::max(xMin, 0);
        int const xStop = std::min(xMax, imageWidth - 1);  // inclusive
        int const minBlock = blocks[xStart];
        int const maxBlock = blocks[xStop];
        int const numBlocks = maxBlock - minBlock + 1;
        ndarray::Array<double, 1, 1> bgTerms = ndarray::allocate(numBlocks);
        bgTerms.deep() = 0.0;
        for (
            int xModel = xStart - xMin, xData = xStart;
            xModel < width && xData < imageWidth;
            ++xModel, ++xData
        ) {
            if (!use[xModel]) continue;
            if (!usePixels[xData]) continue;
            std::size_t const blockIndex = blocks[xData];
            bgTerms[blockIndex - minBlock] += values[xModel]/dataVariance[xData];
        }
        return {minBlock, bgTerms};
    }

    void addToImage(
        lsst::afw::image::Image<float> & image,
        double flux
    ) const {
        int const start = xMin - image.getBBox().getMinX();
        int const stop = xMax - image.getBBox().getMinX();  // exclusive
        if (stop <= start) {
            return;
        }
        auto const rhs = ndarray::asEigenArray(use).select(
            ndarray::asEigenArray(values), 0.0
        );
        ndarray::asEigenArray(image.getArray()[ndarray::view(y)(start, stop)]) += rhs*flux;
    }
};


}  // anonymous namespace


PolynomialKernel::PolynomialKernel(
    lsst::geom::Box2D const& range,
    int halfWidth,
    int order,
    std::size_t numPoly,
    ndarray::ArrayRef<double const, 1, 1> const& coefficients
) : _halfWidth(halfWidth),
    _order(order),
    _numCoeffs(Polynomial::nParametersFromOrder(order)),
    _numPoly(numPoly),
    _numParams(numPoly*_numCoeffs),
    _coefficients(ndarray::copy(coefficients))
{
    utils::checkSize(coefficients.size(), _numParams, "coefficients");

    _polynomials.reserve(numPoly);
    std::size_t start = 0;
    std::size_t stop = _numCoeffs;
    for (std::size_t ii = 0; ii < _numPoly; ++ii, start += _numCoeffs, stop += _numCoeffs) {
        ndarray::Array<double, 1, 1> coeffs = ndarray::copy(coefficients[ndarray::view(start, stop)]);
        _polynomials.emplace_back(coeffs, range);
    }
}


FiberKernel::FiberKernel(
    lsst::geom::Box2D const& range,
    int halfWidth,
    int order,
    ndarray::ArrayRef<double const, 1, 1> const& coefficients
) : PolynomialKernel(
        range,
        halfWidth,
        order,
        2*halfWidth,
        coefficients
    ) {}


std::shared_ptr<FiberTrace<float>> FiberKernel::operator()(
    FiberTrace<float> const& trace,
    lsst::geom::Box2I const& bbox
) const {
    auto const require = trace.getTrace().getMask()->getPlaneBitMask(fiberMaskPlane);
    lsst::geom::Box2I newBox = trace.getTrace().getBBox().dilatedBy(lsst::geom::Extent2I(_halfWidth, 0));
    newBox.clip(bbox);

    lsst::afw::image::MaskedImage<float> convolved{newBox};
    lsst::afw::image::Image<float> convImage = *convolved.getImage();
    lsst::afw::image::Mask<lsst::afw::image::MaskPixel> convMask = *convolved.getMask();

    convImage = 0.0;
    *convolved.getVariance() = 0.0;

    int const xMax = trace.getTrace().getWidth() - 1;  // inclusive
    for (int yy = 0; yy < trace.getTrace().getHeight(); ++yy) {
        auto const& model = FiberModel::fromFiberTrace(trace, yy, 0, xMax, require);
        float const xCenter = model.centroid();
        model.addToImage(convImage, 1.0);

        std::size_t offsetIndex = 0;
        for (int offset = -_halfWidth; offset <= _halfWidth; ++offset, ++offsetIndex) {
            if (offset == 0) {
                --offsetIndex;
                continue;
            }
            auto kernelModel = model.applyOffset(offset, newBox.getMaxX() + 1);
            kernelModel.addToImage(convImage, _polynomials[offsetIndex](xCenter, yy));
        }
    }

    for (auto iter = convolved.begin(); iter != convolved.end(); ++iter) {
        if (iter.image() != 0.0) {
            iter.mask() |= require;
        }
    }

    return std::make_shared<FiberTrace<float>>(std::move(convolved), trace.getFiberId());
}


FiberTraceSet<float> FiberKernel::operator()(
    FiberTraceSet<float> const& trace,
    lsst::geom::Box2I const& bbox
) const {
    FiberTraceSet<float> result(trace.size());
    for (std::size_t ii = 0; ii < trace.size(); ++ii) {
        result.add(operator()(*trace[ii], bbox));
    }
    return result;
}


ndarray::Array<double, 1, 1> FiberKernel::evaluate(double x, double y) const {
    ndarray::Array<double, 1, 1> result = ndarray::allocate(2*_halfWidth + 1);
    result[_halfWidth] = 0.0;
    std::size_t offsetIndex = 0;
    for (int pixel = 0, offset = -_halfWidth; offset <= _halfWidth; ++pixel, ++offset, ++offsetIndex) {
        if (offset == 0) {
            --offsetIndex;
            continue;
        }
        double const value = _polynomials[offsetIndex](x, y);
        result[pixel] = value;
        result[_halfWidth] -= value;
    }
    return result;
}


ndarray::Array<double, 3, 3> FiberKernel::makeOffsetImages(
    lsst::geom::Extent2I const& dims
) const {
    ndarray::Array<double, 3, 3> result = ndarray::allocate(2*_halfWidth + 1, dims.getY(), dims.getX());

    ndarray::ArrayRef<double, 2, 2> center = result[_halfWidth];
    center.deep() = 1.0;

    std::size_t offsetIndex = 0;
    std::size_t imageIndex = 0;
    for (int offset = -_halfWidth; offset <= _halfWidth; ++offset, ++imageIndex, ++offsetIndex) {
        if (offset == 0) {
            --offsetIndex;
            continue;
        }

        lsst::geom::Box2D const& range = _polynomials[offsetIndex].getXYRange();
        double const dx = range.getWidth()/(dims.getX() - 1);
        double const dy = range.getHeight()/(dims.getY() - 1);

        ndarray::ArrayRef<double, 2, 2> image = result[imageIndex];
        double ySample = range.getMinY();
        for (int yy = 0; yy < dims.getY(); ++yy, ySample += dy) {
            auto iter = image[yy].begin();
            double xSample = range.getMinX();
            for (int xx = 0; xx < dims.getX(); ++xx, ++iter, xSample += dx) {
                *iter = _polynomials[offsetIndex](xSample, ySample);
            }
        }
        ndarray::asEigenArray(center) -= ndarray::asEigenArray(image);
    }
    return result;
}


ImageKernel::ImageKernel(
    lsst::geom::Box2D const& range,
    int halfWidth,
    int order,
    ndarray::ArrayRef<double const, 1, 1> const& coefficients
) : PolynomialKernel(
        range,
        halfWidth,
        order,
        2*halfWidth + 1,
        coefficients
    ) {}


std::shared_ptr<lsst::afw::image::Image<float>> ImageKernel::operator()(
    lsst::afw::image::Image<float> const& image
) const {
    std::size_t height = image.getHeight();
    auto resultPtr = std::make_shared<lsst::afw::image::Image<float>>(image.getBBox());
    lsst::afw::image::Image<float> & result = *resultPtr;
    result = 0.0;

    for (std::size_t yy = 0; yy < height; ++yy) {
        auto const base = FiberModel::fromImage(image, yy);

        std::size_t offsetIndex = 0;
        for (int offset = -_halfWidth; offset <= _halfWidth; ++offset, ++offsetIndex) {
            FiberModel model = offset == 0 ? base : base.applyOffset(offset, image.getWidth());
            auto const& poly = _polynomials[offsetIndex];

            auto inIter = model.values.begin();
            auto outIter = result.row_begin(yy) + model.xMin;
            for (int xx = model.xMin; xx < model.xMax; ++xx, ++inIter, ++outIter) {
                *outIter += (*inIter)*poly(xx, yy);
            }
        }
    }
    return resultPtr;
}


ndarray::Array<double, 3, 3> ImageKernel::makeOffsetImages(
    lsst::geom::Extent2I const& dims
) const {
    ndarray::Array<double, 3, 3> result = ndarray::allocate(2*_halfWidth + 1, dims.getY(), dims.getX());

    ndarray::ArrayRef<double, 2, 2> center = result[_halfWidth];
    center.deep() = 0.0;

    std::size_t offsetIndex = 0;
    for (int offset = -_halfWidth; offset <= _halfWidth; ++offset, ++offsetIndex) {
        lsst::geom::Box2D const& range = _polynomials[offsetIndex].getXYRange();
        double const dx = range.getWidth()/(dims.getX() - 1);
        double const dy = range.getHeight()/(dims.getY() - 1);

        ndarray::ArrayRef<double, 2, 2> image = result[offsetIndex];
        double ySample = range.getMinY();
        for (int yy = 0; yy < dims.getY(); ++yy, ySample += dy) {
            auto iter = image[yy].begin();
            double xSample = range.getMinX();
            for (int xx = 0; xx < dims.getX(); ++xx, ++iter, xSample += dx) {
                double const value = _polynomials[offsetIndex](xSample, ySample);
                if (offset == 0) {
                    *iter += value;
                } else {
                    *iter = value;
                }
            }
        }
        if (offset != 0) {
            ndarray::asEigenArray(center) -= ndarray::asEigenArray(image);
        }
    }

    return result;
}


namespace {


struct BackgroundHelper {
    BackgroundHelper(
        lsst::geom::Extent2I const& dims, int xBlockSize, int yBlockSize
    ) : xNumBlocks((dims.getX() + xBlockSize - 1) / xBlockSize),
        yNumBlocks((dims.getY() + yBlockSize - 1) / yBlockSize),
        xBlocks(ndarray::allocate(dims.getX())),
        yBlocks(ndarray::allocate(dims.getY())) {
        if (xBlockSize <= 0 || yBlockSize <= 0) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, "Block sizes must be positive");
        }
        for (int ii = 0; ii < dims.getX(); ++ii) {
            xBlocks[ii] = ii / xBlockSize;
        }

        for (int ii = 0; ii < dims.getY(); ++ii) {
            yBlocks[ii] = ii / yBlockSize;
        }
    }

    std::size_t getIndex(int x, int y) const {
        return yBlocks[y] * xNumBlocks + xBlocks[x];
    }

    std::size_t xNumBlocks;
    std::size_t yNumBlocks;
    ndarray::Array<int, 1, 1> xBlocks;
    ndarray::Array<int, 1, 1> yBlocks;
};


struct RowData {
    int y;
    ndarray::Array<bool, 1, 1> useTrace;  // trace overlaps this row?
    ndarray::Array<bool, 1, 1> usePixel;  // pixel in row should be used?
    ndarray::Array<double, 1, 1> xCenter; // center of trace in x, for kernel polynomial
    std::vector<FiberModel> models;  // model for each fiber at this row
    std::vector<std::vector<FiberModel>> kernelModels;  // kernel-offset models [fiber][offset]
    std::vector<std::vector<std::size_t>> overlaps;  // [fiber] overlapping fiber indices within kernelHalfWidth

    // Layout of models: offset=-kernelWidth, ... offset=0, ... offset=+kernelWidth
    // Note that this is different from the layout of coefficients in the matrix, which skips offset=0.
    ndarray::Array<double, 2, 2> dotData;  // [fiber][offset] model dot data

    // [fiber][otherFiber][offset][otherOffset] model dot Model; first otherFiber is the same as fiber
    std::vector<std::vector<ndarray::Array<double, 2, 2>>> dotModel;

    ndarray::Array<double, 2, 2> polyValues;  // [fiber][spatialTerm]
    std::vector<std::vector<std::pair<int, ndarray::Array<double, 1, 1>>>> dotBackground;  // [fiber][offset]

    RowData(
        int y, std::size_t numFibers, int imageWidth, std::size_t kernelHalfWidth, std::size_t numSpatialTerms
    ) : y(y),
        useTrace(ndarray::allocate(numFibers)),
        usePixel(ndarray::allocate(imageWidth)),
        xCenter(ndarray::allocate(numFibers)),
        models(numFibers, FiberModel::dummy()),
        kernelModels(numFibers, std::vector<FiberModel>(2*kernelHalfWidth + 1, FiberModel::dummy())),
        overlaps(numFibers),
        dotData(ndarray::allocate(numFibers, 2*kernelHalfWidth + 1)),
        dotModel(numFibers),
        polyValues(ndarray::allocate(numFibers, numSpatialTerms)),
        dotBackground(numFibers)
    {
        useTrace.deep() = false;
        usePixel.deep() = false;
        dotData.deep() = 0.0;
        polyValues.deep() = 0.0;
    }
};


struct FiberKernelFitter {
    using ImageT = float;
    using MaskT = lsst::afw::image::MaskPixel;
    using VarianceT = float;

    FiberKernelFitter(
        lsst::afw::image::MaskedImage<ImageT> const& image,
        FiberTraceSet<ImageT> const& fiberTraces,
        lsst::afw::image::MaskPixel badBitMask,
        int kernelHalfWidth,
        int kernelOrder,
        int xBackgroundSize,
        int yBackgroundSize,
        ndarray::Array<int, 1, 1> const& rows
    ) : _numFibers(fiberTraces.size()),
        _numRows(rows.size()),
        _image(image),
        _fiberTraces(fiberTraces),
        _badBitMask(badBitMask),
        _kernelHalfWidth(kernelHalfWidth),
        _kernelOrder(kernelOrder),
        _xBackgroundSize(xBackgroundSize),
        _yBackgroundSize(yBackgroundSize),
        _rows(rows),
        _kernelPolynomial(kernelOrder, lsst::geom::Box2D(image.getBBox())),
        _numKernels(2*_kernelHalfWidth),
        _numKernelSpatial(_kernelPolynomial.getNParameters()),
        _bg(image.getDimensions(), xBackgroundSize, yBackgroundSize),
        _bgStart(_numKernels*_numKernelSpatial),
        _requireMask(image.getMask()->getPlaneBitMask(fiberMaskPlane)),
        _noData(1 << image.getMask()->addMaskPlane("NO_DATA")),
        _badFiberTrace(1 << image.getMask()->addMaskPlane("BAD_FIBERTRACE")),
        _suspect(1 << image.getMask()->addMaskPlane("SUSPECT")),
        _numParams(_bgStart + _bg.xNumBlocks*_bg.yNumBlocks)
    {}

    // Determine which traces are relevant for this row
    RowData calculateRow(int y) const {
        RowData data(y, _numFibers, _image.getWidth(), _numKernels, _numKernelSpatial);
        for (std::size_t ii = 0; ii < _numFibers; ++ii) {
            auto const& box = _fiberTraces[ii]->getTrace().getBBox();
            int xMin = 0;
            int xMax = -1;
            auto const traceMask = _fiberTraces[ii]->getTrace().getMask()->getArray()[y];
            for (int xx = 0; xx < box.getWidth(); ++xx) {
                if (traceMask[xx] & _requireMask) {
                    xMin = xx;
                    break;
                }
            }
            for (int xx = box.getWidth() - 1; xx >= 0; --xx) {
                if (traceMask[xx] & _requireMask) {
                    xMax = xx;
                    break;
                }
            }
            if (xMax < xMin) {
                continue;
            }

            data.models[ii] = FiberModel::fromFiberTrace(
                *_fiberTraces[ii], y, xMin, xMax, _requireMask
            );
            FiberModel & model = data.models[ii];
            data.xCenter[ii] = model.centroid();
            data.useTrace[ii] = true;

            std::size_t kernelIndex = 0;
            for (int offset = -_kernelHalfWidth; offset <= _kernelHalfWidth; ++offset, ++kernelIndex) {
                data.kernelModels[ii][kernelIndex] = model.applyOffset(offset, _image.getWidth());
            }
        }

        auto iter = _image.row_begin(y);
        for (int xx = 0; xx < _image.getWidth(); ++xx, ++iter) {
            data.usePixel[xx] = isGoodImage(iter);
        }

        for (std::size_t ii = 0; ii < _numFibers; ++ii) {
            if (!data.useTrace[ii]) {
                continue;
            }
            FiberModel const& left = data.models[ii];
            int const leftMin = left.xMin - _kernelHalfWidth;
            int const leftMax = left.xMax + _kernelHalfWidth;
            std::vector<std::size_t>& overlaps = data.overlaps[ii];
            for (std::size_t jj = 0; jj < _numFibers; ++jj) {
                if (!data.useTrace[jj]) {
                    continue;
                }
                FiberModel const& right = data.models[jj];
                int const rightMin = right.xMin - _kernelHalfWidth;
                int const rightMax = right.xMax + _kernelHalfWidth;
                if (std::max(leftMin, rightMin) < std::min(leftMax, rightMax)) {
                    overlaps.push_back(jj);
                }
            }
        }

        return data;
    }

    // Layout of parameters for kernel+background fit:
    // Kernel parameters: offset (...,-2,-1,1,2,... NOTE: no zero!) runs slow, spatial polynomial term
    //     (0.._numKernelSpatial-1) runs fast; starts at index 0
    // Background parameters: x block (0.._bg.xNumBlocks-1) runs fast, y block (0.._bg.yNumBlocks-1) runs
    //     slow; starts at _bgStart

    std::size_t getKernelIndex(int offset, int spatialTerm) const {
        int const offsetIndex = offset + _kernelHalfWidth - (offset < 0 ? 0 : 1);
        return offsetIndex * _numKernelSpatial + spatialTerm;
    }

    std::size_t getBackgroundIndex(int x, int y) const {
        return _bgStart + _bg.getIndex(x, y);
    }

    // Layout of parameters for flux fit:
    // Fiber parameters: row runs slow, fiber index runs fast
    std::size_t getFluxIndex(std::size_t rowNum, std::size_t fiberIndex) const {
        return rowNum*_numFibers + fiberIndex;
    }

    /// Return a ScopedTimer that accumulates elapsed time into the named bucket.
    ScopedTimer timer(std::string const& name) const {
        return ScopedTimer(&_timings[name]);
    }

    bool isGoodImage(ImageT value, lsst::afw::image::MaskPixel mask, VarianceT variance) const {
        return (mask & _badBitMask) == 0 && std::isfinite(value) && std::isfinite(variance) && variance > 0;
    }
    bool isGoodImage(auto & iter) const {
        return isGoodImage(iter.image(), iter.mask(), iter.variance());
    }

    double getDotModel(
        RowData const& data,
        std::size_t leftFiber,
        std::size_t rightFiber,
        std::size_t leftOffsetIndex,
        std::size_t rightOffsetIndex
    ) const {
        if (leftFiber <= rightFiber) {
            std::size_t const delta = rightFiber - leftFiber;
            if (delta >= data.dotModel[leftFiber].size()) {
                return 0.0;
            }
            return data.dotModel[leftFiber][delta][leftOffsetIndex][rightOffsetIndex];
        }

        std::size_t const delta = leftFiber - rightFiber;
        if (delta >= data.dotModel[rightFiber].size()) {
            return 0.0;
        }
        // dot(A, B) == dot(B, A): swap fiber/offset order when reading reversed storage.
        return data.dotModel[rightFiber][delta][rightOffsetIndex][leftOffsetIndex];
    }

    void calculateFiber(
        int y,
        std::size_t fiberIndex,
        RowData & data
    ) const {
        if (!data.useTrace[fiberIndex]) {
            return;
        }

        auto const& dataImage = _image.getImage()->getArray()[y];
        auto const& dataVariance = _image.getVariance()->getArray()[y];

        data.polyValues[fiberIndex].deep() = utils::vectorToArray(_kernelPolynomial.getDFuncDParameters(
            data.xCenter[fiberIndex], y
        ));

        std::size_t iIndex = 0;
        for (int iOffset = -_kernelHalfWidth; iOffset <= _kernelHalfWidth; ++iOffset, ++iIndex) {
            FiberModel const& iModel = data.kernelModels[fiberIndex][iIndex];
            data.dotData[fiberIndex][iIndex] = iModel.dotData(dataImage, data.usePixel, dataVariance);
            data.dotBackground[fiberIndex].emplace_back(
                iModel.dotBackground(_bg.xBlocks, data.usePixel, dataVariance)
            );
        }

        std::vector<ndarray::Array<double, 2, 2>> dotModel;
        dotModel.reserve(_numFibers - fiberIndex);
        for (std::size_t jFiberIndex = fiberIndex; jFiberIndex < _numFibers; ++jFiberIndex) {
            ndarray::Array<double, 2, 2> dotModelFiber = ndarray::allocate(_numKernels + 1, _numKernels + 1);
            if (!data.useTrace[jFiberIndex]) {
                dotModelFiber.deep() = 0.0;
                dotModel.push_back(std::move(dotModelFiber));
                continue;
            }

            std::size_t iIndex = 0;
            for (int iOffset = -_kernelHalfWidth; iOffset <= _kernelHalfWidth; ++iOffset, ++iIndex) {
                FiberModel const& iModel = data.kernelModels[fiberIndex][iIndex];
                std::size_t jIndex = 0;
                for (int jOffset = -_kernelHalfWidth; jOffset <= _kernelHalfWidth; ++jOffset, ++jIndex) {
                    FiberModel const& jModel = data.kernelModels[jFiberIndex][jIndex];
                    dotModelFiber[iIndex][jIndex] = iModel.dotOther(jModel, data.usePixel, dataVariance);
                }
            }
            bool const isZero = (ndarray::asEigenArray(dotModelFiber) == 0.0).all();
            if (isZero) {
                // The fibers are ordered, so all subsequent fibers will also be zero
                break;
            }
            dotModel.push_back(std::move(dotModelFiber));
        }
        data.dotModel[fiberIndex] = std::move(dotModel);
    }

    // Image(x,y) = sum_i F_i(y).p_i(x,y) + sum_i sum_j a_ij.F_i(y).K_ij(x,y) + sum_b a_b B_b(x,y)
    // Where:
    // F_i(y) is the flux of fiber i at row y
    // p_i(x,y) is the fiber profile for fiber i at row y
    // K_ij(x,y) = [delta(x-offset_j) - delta(x)]*p_i(x,y).P_j(x,y) is the kernel component for fiber i,
    //     offset j and spatial polynomial P_j
    // B_b(x,y) is the background block b
    // a_ij and a_b are the kernel and background parameters we are solving for
    //
    // Instead of turning Image(x,y) into a residual (by subtracting sum_i F_i(y).p_i(x,y)) and
    // recalculating the dotData values every iteration, we can remove some cross terms from the vector.
    // The model dot data terms in the vector are:
    // F_i(y).K_j(x,y) dot [Image(x,y) - sum_i F_i(y).p_i(x,y)]
    // = F_i(y).K_j(x,y) dot Image(x,y) - sum_k F_k(y).p_k(x,y) dot F_i(y).K_j(x,y)
    // and
    // B_b(x,y) dot [Image(x,y) - sum_i F_i(y).p_i(x,y)]
    // = B_b(x,y) dot Image(x,y) - sum_i F_i(y).p_i(x,y) dot B_b(x,y)
    //
    // Model dot model terms in the matrix are:
    // F_i(y).K_j(x,y) dot F_m(y).K_n(x,y) = F_i(y).F_m(y).[K_j(x,y) dot K_n(x,y)]
    // and
    // F_i(y).K_j(x,y) dot B_b(x,y) = F_i(y).[K_j(x,y) dot B_b(x,y)]
    // and
    // B_b(x,y) dot B_c(x,y)

    void accumulateFiber(
        std::size_t fiberIndex,
        RowData const& data,
        ndarray::Array<double, 1, 1> const& flux,
        ndarray::Array<double, 2, 2> & matrix,
        ndarray::Array<double, 1, 1> & vector
    ) const {
        if (!data.useTrace[fiberIndex]) {
            return;
        }

        int const yy = data.y;
        double const iFlux = flux[fiberIndex];
        ndarray::ArrayRef<double const, 1, 1> const polyValues = data.polyValues[fiberIndex];

        std::size_t iOffsetIndex = 0;
        for (int iOffset = -_kernelHalfWidth; iOffset <= _kernelHalfWidth; ++iOffset, ++iOffsetIndex) {
            std::size_t const bgStart = data.dotBackground[fiberIndex][iOffsetIndex].first;
            auto const& bgTerms = data.dotBackground[fiberIndex][iOffsetIndex].second;
            if (iOffset == 0) {
                // Subtracting sum_i F_i(y).p_i(x,y) dot B_b(x,y) from the vector
                auto _t = timer("bg_vector");
                assert(iOffsetIndex == std::size_t(_kernelHalfWidth));
                std::size_t bgIndex = getBackgroundIndex(0, yy) + bgStart;
                for (std::size_t ii = 0; ii < bgTerms.size(); ++ii, ++bgIndex) {
                    vector[bgIndex] -= bgTerms[ii]*iFlux;
                }
                continue;
            }

            double const dotData = data.dotData[fiberIndex][iOffsetIndex]*iFlux;

            for (std::size_t iSpatial = 0; iSpatial < _numKernelSpatial; ++iSpatial) {
                std::size_t const iKernelIndex = getKernelIndex(iOffset, iSpatial);
                double const iPoly = polyValues[iSpatial];
                // Vector term from the image
                // F_i(y).p_i(x,y) dot K_j(x,y)
                {
                    auto _t = timer("vector_datum");
                    vector[iKernelIndex] += iPoly*dotData;
                }

                // Subtracting sum_k F_k(y).p_k(x,y) dot F_i(y).K_j(x,y) from the vector
                {
                    auto _t = timer("vector_cross");
                    for (std::size_t jFiberIndex : data.overlaps[fiberIndex]) {
                        double const jFlux = flux[jFiberIndex];
                        double const dotModel = getDotModel(
                            data, fiberIndex, jFiberIndex, iOffsetIndex, _kernelHalfWidth
                        );
                        vector[iKernelIndex] -= iPoly*dotModel*iFlux*jFlux;
                    }
                }

                // Kernel-kernel terms for all fibers.
                // We accumulate only the upper triangle (jKernelIndex >= iKernelIndex),
                // but include all ordered fiber pairs for the full normal equations.
                {
                    auto _t = timer("matrix_kernel_kernel");
                    for (std::size_t jFiberIndex : data.overlaps[fiberIndex]) {
                        double const jFlux = flux[jFiberIndex];

                        double dotKernel = getDotModel(
                            data, fiberIndex, jFiberIndex, iOffsetIndex, iOffsetIndex
                        );
                        if (dotKernel == 0.0) {
                            continue;
                        }
                        dotKernel *= iFlux*jFlux*iPoly;
                        std::size_t jKernelIndex = iKernelIndex;
                        auto polyIter = data.polyValues[jFiberIndex].begin() + iSpatial;
                        auto matrixIter = matrix[iKernelIndex].begin() + jKernelIndex;
                        for (
                            std::size_t jSpatial = iSpatial;
                            jSpatial < _numKernelSpatial;
                            ++jSpatial, ++jKernelIndex, ++polyIter, ++matrixIter
                        ) {
                            double const jPoly = *polyIter;
                            *matrixIter += jPoly*dotKernel;
                        }

                        std::size_t jOffsetIndex = iOffsetIndex + 1;
                        for (
                            int jOffset = iOffset + 1;
                            jOffset <= _kernelHalfWidth;
                            ++jOffset, ++jOffsetIndex
                        ) {
                            if (jOffset == 0) {
                                continue;
                            }

                            double dotKernel = getDotModel(
                                data, fiberIndex, jFiberIndex, iOffsetIndex, jOffsetIndex
                            );
                            if (dotKernel == 0.0) {
                                continue;
                            }
                            dotKernel *= iFlux*jFlux*iPoly;

                            std::size_t jKernelIndex = getKernelIndex(jOffset, 0);
                            auto polyIter = data.polyValues[jFiberIndex].begin();
                            auto matrixIter = matrix[iKernelIndex].begin() + jKernelIndex;
                            for (
                                std::size_t jSpatial = 0;
                                jSpatial < _numKernelSpatial;
                                ++jSpatial, ++jKernelIndex, ++polyIter, ++matrixIter
                            ) {
                                double const jPoly = *polyIter;
                                *matrixIter += jPoly*dotKernel;
                            }
                        }
                    }
                }

                // Kernel-background cross-terms
                // F_i(y).[K_j(x,y) dot B_b(x,y)]
                {
                    auto _t = timer("matrix_kernel_bg");
                    double const value = iFlux*iPoly;
                    std::size_t bgIndex = getBackgroundIndex(0, yy) + bgStart;
                    for (std::size_t ii = 0; ii < bgTerms.size(); ++ii, ++bgIndex) {
                        matrix[iKernelIndex][bgIndex] += value*bgTerms[ii];
                    }
                }
            }
        }
    }

    // Accumulate the background diagonal terms for a row
    void accumulateBackground(
        RowData const& data,
        ndarray::Array<double, 2, 2>& matrix,
        ndarray::Array<double, 1, 1>& vector
    ) const {
        auto _t = timer("background");
        int const yy = data.y;
        auto iter = _image.row_begin(yy);
        ndarray::Array<double, 1, 1> terms = ndarray::allocate(_bg.xNumBlocks);
        terms.deep() = 0.0;
        for (int xx = 0; xx < _image.getWidth(); ++xx, ++iter) {
            if (!data.usePixel[xx]) {
                continue;
            }

            std::size_t const block = _bg.xBlocks[xx];
            double const weight = 1.0/iter.variance();
            terms[block] += weight;

            std::size_t const bgIndex = getBackgroundIndex(xx, yy);
            vector[bgIndex] += iter.image()*weight;
        }
        for (std::size_t ii = 0, bgIndex = getBackgroundIndex(0, yy); ii < _bg.xNumBlocks; ++ii, ++bgIndex) {
            matrix[bgIndex][bgIndex] += terms[ii];
            assert(std::isfinite(terms[ii]));
        }
    }

    std::pair<ndarray::Array<double, 2, 2>, ndarray::Array<double, 1, 1>> accumulate(
        std::vector<RowData> const& data,
        ndarray::Array<double, 2, 2> const& flux
    ) const {
        ndarray::Array<double, 2, 2> matrix = ndarray::allocate(_numParams, _numParams);
        ndarray::Array<double, 1, 1> vector = ndarray::allocate(_numParams);
        matrix.deep() = 0.0;
        vector.deep() = 0.0;

        for (std::size_t ii = 0; ii < data.size(); ++ii) {
            for (std::size_t fiberIndex = 0; fiberIndex < _numFibers; ++fiberIndex) {
                accumulateFiber(fiberIndex, data[ii], flux[ii], matrix, vector);
            }
            accumulateBackground(data[ii], matrix, vector);
        }
        return {std::move(matrix), std::move(vector)};
    }

    ndarray::Array<double, 1, 1> solve(
        ndarray::Array<double, 2, 2> const& matrix,
        ndarray::Array<double, 1, 1> const& vector,
        double lsqThreshold
    ) const {
        auto _t = timer("solve_kernel");

        // Ensure we have entries for all parameters, to avoid singular matrix
        // Fill in the lower triangle of the matrix
        for (std::size_t ii = 0; ii < _numParams; ++ii) {
            if (matrix[ii][ii] == 0.0) {
                assert(vector[ii] == 0.0);
                 matrix[ii][ii] = 1.0;
            }
            for (std::size_t jj = ii + 1; jj < _numParams; ++jj) {
                assert(matrix[jj][ii] == 0.0);
                matrix[jj][ii] = matrix[ii][jj];
            }
        }

        // Solve the system of equations
        auto lsq = lsst::afw::math::LeastSquares::fromNormalEquations(matrix, vector);
        lsq.setThreshold(lsqThreshold);
        ndarray::Array<double const, 1, 1> solution = lsq.getSolution();

        return ndarray::copy(solution);
    }

    std::pair<FiberKernel, lsst::afw::image::Image<float>> extract(
        ndarray::Array<double const, 1, 1> const& solution
    ) const {
        // Extract the kernel
        FiberKernel kernel(
            lsst::geom::Box2D(_image.getBBox()),
            _kernelHalfWidth,
            _kernelOrder,
            solution[ndarray::view(0, _bgStart)]
        );

        // Extract the background
        lsst::afw::image::Image<float> background(_bg.xNumBlocks, _bg.yNumBlocks);
        std::size_t bgIndex = _bgStart;
        for (std::size_t yy = 0; yy < _bg.yNumBlocks; ++yy, bgIndex += _bg.xNumBlocks) {
            background.getArray()[yy] = solution[ndarray::view(bgIndex, bgIndex + _bg.xNumBlocks)];
        }

        return {std::move(kernel), std::move(background)};
    }

    std::vector<RowData> calculate(ndarray::Array<int, 1, 1> const& rows) const {
        auto _t = timer("calculate");

        std::vector<RowData> result;
        result.reserve(rows.size());
        for (std::size_t ii = 0; ii < rows.size(); ++ii) {
            int yy = rows[ii];
            RowData data = calculateRow(yy);
            for (std::size_t jj = 0; jj < _numFibers; ++jj) {
                calculateFiber(yy, jj, data);
            }
            result.push_back(std::move(data));
        }
        return result;
    }

    ndarray::Array<double, 1, 1> fitKernel(
        std::vector<RowData> const& data,
        ndarray::Array<double, 2, 2> const& flux,
        double lsqThreshold
    ) {
        auto _t = timer("fit_kernel");
        auto const equation = accumulate(data, flux);
        return solve(equation.first, equation.second, lsqThreshold);
    }

    ndarray::Array<double, 2, 2> calculatePolynomials(
        RowData const& data,
        ndarray::Array<double const, 1, 1> const& kernelSolution
    ) const {
        ndarray::Array<double, 2, 2> polyValues = ndarray::allocate(_numFibers, _numKernels + 1);
        for (std::size_t fiberIndex = 0; fiberIndex < _numFibers; ++fiberIndex) {
            if (!data.useTrace[fiberIndex]) {
                continue;
            }
            std::size_t offsetIndex = 0;
            for (int offset = -_kernelHalfWidth; offset <= _kernelHalfWidth; ++offset, ++offsetIndex) {
                if (offset == 0) {
                    polyValues[fiberIndex][offsetIndex] = std::numeric_limits<double>::quiet_NaN();
                    continue;
                }
                polyValues[fiberIndex][offsetIndex] = (ndarray::asEigenArray(
                    data.polyValues[fiberIndex]
                )*ndarray::asEigenArray(
                    kernelSolution[ndarray::view(
                        getKernelIndex(offset, 0),
                        getKernelIndex(offset, _numKernelSpatial)
                    )]
                )).sum();
            }
        }
        return polyValues;
    }

    void accumulateFlux(
        std::size_t fiberIndex,
        std::size_t rowNum,
        RowData const& data,
        ndarray::Array<double const, 2, 2> const& polyValues,  // [fiberIndex][offset]
        ndarray::ArrayRef<double const, 1, 1> const& background,
        math::SymmetricSparseSquareMatrix & matrix,
        ndarray::Array<double, 1, 1> & vector
    ) const {
        auto _t = timer("flux");

        if (!data.useTrace[fiberIndex]) {
            return;
        }
        std::size_t const iIndex = getFluxIndex(rowNum, fiberIndex);

        // Image(x,y) = sum_i F_i(y).[p_i(x,y) + sum_I a_I K_I(x,y)] + sum_b a_b B_b(x,y)
        // Solving for the F_i(y)
        // a_I and a_b are fixed parameters from the kernel/background fit.
        // Model_i = p_i + sum_I K_I
        // model_i dot model_j = p_i*p_j + p_i*sum_J K_J + p_j*sum_I K_I + sum_I sum_J K_I*K_J
        // model_i dot (data - sum_b a_b B_b) = p_i dot data + sum_I K_I dot data - sum_b a_b p_i dot B_b - sum_I sum_b a_b K_I dot B_b

        // Calculate model dot data
        ndarray::Array<double, 1, 1> const& dotData = data.dotData[fiberIndex];
        double modelDotData = 0.0;
        std::size_t offsetIndex = 0;
        for (int offset = -_kernelHalfWidth; offset <= _kernelHalfWidth; ++offset, ++offsetIndex) {
            std::size_t const bgStart = data.dotBackground[fiberIndex][offsetIndex].first;
            ndarray::Array<double, 1, 1> const& bgTerms = data.dotBackground[fiberIndex][offsetIndex].second;
            if (offset == 0) {
                modelDotData += dotData[offsetIndex];  // p_i dot data
                for (std::size_t ii = 0, bgIndex = bgStart; ii < bgTerms.size(); ++ii, ++bgIndex) {
                    modelDotData -= background[bgIndex]*bgTerms[ii];  // sum_b a_b p_i dot B_b
                }
                continue;
            }
            double const spatial = polyValues[fiberIndex][offsetIndex];
            modelDotData += spatial*dotData[offsetIndex];  // sum_I K_I dot data
            for (std::size_t ii = 0, bgIndex = bgStart; ii < bgTerms.size(); ++ii, ++bgIndex) {
                modelDotData -= spatial*background[bgIndex]*bgTerms[ii];  // sum_I sum_b a_b K_I dot B_b
            }
        }
        vector[iIndex] = modelDotData;

        // Calculate model dot model
        std::size_t const numOverlaps = data.dotModel[fiberIndex].size();
        for (std::size_t jj = 0, jFiberIndex = fiberIndex; jj < numOverlaps; ++jj, ++jFiberIndex) {
            if (!data.useTrace[jFiberIndex]) {
                continue;
            }
            std::size_t jIndex = getFluxIndex(rowNum, jFiberIndex);
            ndarray::Array<double, 2, 2> const& dotModel = data.dotModel[fiberIndex][jj];

            double modelDotModel = dotModel[_kernelHalfWidth][_kernelHalfWidth];

            std::size_t offsetIndex = 0;
            for (int offset = -_kernelHalfWidth; offset <= _kernelHalfWidth; ++offset, ++offsetIndex) {
                if (offset == 0) {
                    continue;
                }
                double const iSpatial = polyValues[fiberIndex][offsetIndex];
                {
                    double const jSpatial = polyValues[jFiberIndex][offsetIndex];
                    modelDotModel += iSpatial*dotModel[offsetIndex][_kernelHalfWidth];
                    modelDotModel += jSpatial*dotModel[_kernelHalfWidth][offsetIndex];
                }

                std::size_t jOffsetIndex = 0;
                for (
                    int jOffset = -_kernelHalfWidth;
                    jOffset <= _kernelHalfWidth;
                    ++jOffset, ++jOffsetIndex
                ) {
                    if (jOffset == 0) {
                        continue;
                    }
                    double const jSpatial = polyValues[jFiberIndex][jOffsetIndex];
                    modelDotModel += iSpatial*jSpatial*dotModel[offsetIndex][jOffsetIndex];
                }
            }
            matrix.add(iIndex, jIndex, modelDotModel);
        }
    }

    ndarray::Array<double, 2, 2> fitFlux(
        std::vector<RowData> const& data,
        ndarray::Array<double const, 1, 1> const& kernelSolution
    ) {
        std::size_t const numFluxParams = _numFibers*_numRows;
        math::SymmetricSparseSquareMatrix matrix(numFluxParams);
        ndarray::Array<double, 1, 1> vector = ndarray::allocate(numFluxParams);
        vector.deep() = 0.0;
        for (std::size_t ii = 0; ii < data.size(); ++ii) {
            ndarray::Array<double, 2, 2> polyValues = calculatePolynomials(data[ii], kernelSolution);
            for (std::size_t fiberIndex = 0; fiberIndex < _numFibers; ++fiberIndex) {
                accumulateFlux(
                    fiberIndex, ii, data[ii],
                    polyValues, kernelSolution[ndarray::view(_bgStart, _numParams)],
                    matrix, vector
                );
            }
        }

        auto _t = timer("solve_flux");

        // Avoid non-singular matrix from missing fibers
        for (std::size_t ii = 0; ii < numFluxParams; ++ii) {
            if (matrix.get(ii, ii) == 0.0) {
                assert(vector[ii] == 0.0);
                matrix.add(ii, ii, 1.0);
            }
        }

        ndarray::Array<double, 2, 2> flux = ndarray::allocate(_numRows, _numFibers);

        using Solver = math::SymmetricSparseSquareMatrix::SimplicialLDLTSolverUpper;
        ndarray::Array<double, 1, 1> solution = matrix.solve<Solver>(vector);

        std::size_t start = 0;
        std::size_t stop = _numFibers;
        for (std::size_t ii = 0; ii < _numRows; ++ii, start += _numFibers, stop += _numFibers) {
            flux[ii] = solution[ndarray::view(start, stop)];
        }
        return flux;
    }

    std::tuple<FiberKernel, lsst::afw::image::Image<float>, ndarray::Array<double, 2, 2>> run(
        int maxIter,
        int andersonDepth,
        double fluxTol,
        double lsqThreshold
    ) {
        std::vector<RowData> data = calculate(_rows);

        std::size_t const numFluxParams = _numRows*_numFibers;

        ndarray::Array<double, 1, 1> kernel = utils::arrayFilled<double, 1, 1>(_numParams, 0.0);
        ndarray::Array<double, 2, 2> flux = fitFlux(data, kernel);
        std::cerr << "Initial flux: " << flux[0] << std::endl;

        std::vector<ndarray::Array<double, 1, 1>> kernelHistory;
        std::vector<ndarray::Array<double, 1, 1>> fluxHistory;
        kernelHistory.reserve(andersonDepth + 2);
        fluxHistory.reserve(andersonDepth + 2);

        bool converged = false;
        for (int ii = 0; ii < maxIter; ++ii) {
            auto _t = timer("iteration");
            ndarray::Array<double, 1, 1> const fluxVector = utils::flattenArray(flux);
            kernel = fitKernel(data, flux, lsqThreshold);
            std::cerr << "New kernel: " << kernel << std::endl;

            ndarray::Array<double, 2, 2> newFlux = fitFlux(data, kernel);
            ndarray::Array<double, 1, 1> const newFluxVector = utils::flattenArray(newFlux);
            ndarray::Array<double, 1, 1> fluxResidual = ndarray::allocate(numFluxParams);
            ndarray::asEigenArray(fluxResidual) =
                ndarray::asEigenArray(newFluxVector) - ndarray::asEigenArray(fluxVector);
            double const rms = std::sqrt(
                ndarray::asEigenArray(fluxResidual).square().sum()/fluxResidual.size()
            );
            std::cerr << "Iteration " << ii << ": flux RMS change = " << rms << std::endl;
            if (rms < fluxTol) {
                flux = std::move(newFlux);
                converged = true;
                break;
            }

            kernelHistory.push_back(fluxVector);
            fluxHistory.push_back(fluxResidual);
            if (kernelHistory.size() > std::size_t(andersonDepth + 1)) {
                kernelHistory.erase(kernelHistory.begin());
                fluxHistory.erase(fluxHistory.begin());
            }

            ndarray::Array<double, 1, 1> nextFluxVector = ndarray::copy(newFluxVector);
            std::size_t const numSteps = kernelHistory.size() - 1;
            if (numSteps > 0) {
                std::size_t const mk = std::min<std::size_t>(andersonDepth, numSteps);
                ndarray::Array<double, 2, 2> dFlux = ndarray::allocate(numFluxParams, mk);
                ndarray::Array<double, 2, 2> dKernel = ndarray::allocate(numFluxParams, mk);
                std::size_t const start = kernelHistory.size() - mk - 1;
                for (std::size_t jj = 0; jj < mk; ++jj) {
                    for (std::size_t kk = 0; kk < numFluxParams; ++kk) {
                        dFlux[kk][jj] = fluxHistory[start + jj + 1][kk] - fluxHistory[start + jj][kk];
                        dKernel[kk][jj] = kernelHistory[start + jj + 1][kk] - kernelHistory[start + jj][kk];
                    }
                }

                Eigen::VectorXd const gamma = ndarray::asEigenMatrix(dFlux).colPivHouseholderQr().solve(
                    ndarray::asEigenMatrix(fluxResidual)
                );
                Eigen::VectorXd const correction = (
                    ndarray::asEigenMatrix(dKernel) + ndarray::asEigenMatrix(dFlux)
                )*gamma;
                Eigen::VectorXd const andersonFluxVector = ndarray::asEigenMatrix(newFluxVector) - correction;
                if (andersonFluxVector.array().isFinite().all()) {
                    ndarray::asEigenArray(nextFluxVector) = andersonFluxVector.array();
                }
            }

            ndarray::Array<double, 2, 1> reshapedFlux = utils::unflattenArray(
                nextFluxVector, _numRows, _numFibers
            );
            ndarray::asEigenArray(flux) = ndarray::asEigenArray(reshapedFlux);
            std::cerr << "New flux: " << flux[0] << std::endl;
        }
        if (!converged) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::RuntimeError,
                "Kernel fitting failed to converge before reaching maximum number of iterations"
            );
        }

        std::cerr << "Timing results (accumulated over all iterations, seconds):\n";
        for (auto const& [name, elapsed] : _timings) {
            std::cerr << "  " << name << ": " << elapsed << "\n";
        }

        kernel = fitKernel(data, flux, lsqThreshold);
        auto const extraction = extract(kernel);
        return {std::move(extraction.first), std::move(extraction.second), std::move(flux)};
    }

    // Inputs
    std::size_t _numFibers;
    std::size_t _numRows;
    lsst::afw::image::MaskedImage<ImageT> const& _image;
    FiberTraceSet<ImageT> const& _fiberTraces;
    lsst::afw::image::MaskPixel _badBitMask;
    int _kernelHalfWidth;
    int _kernelOrder;
    int _xBackgroundSize;
    int _yBackgroundSize;
    ndarray::Array<int, 1, 1> const& _rows;

    // Helpers
    math::NormalizedPolynomial2<double> _kernelPolynomial;
    std::size_t _numKernels;
    std::size_t _numKernelSpatial;
    BackgroundHelper _bg;
    std::size_t _bgStart;  // starting index of background parameters for current block
    lsst::afw::image::MaskPixel _requireMask;
    MaskT _noData;
    MaskT _badFiberTrace;
    MaskT _suspect;
    std::size_t _numParams;

    // Timing accumulators; mutable so timer() can be called from const methods.
    mutable std::unordered_map<std::string, double> _timings;
};


}  // anonymous namespace


std::tuple<FiberKernel, lsst::afw::image::Image<float>, ndarray::Array<double, 2, 2>> fitFiberKernel(
    lsst::afw::image::MaskedImage<float> const& image,
    FiberTraceSet<float> const& fiberTraces,
    lsst::afw::image::MaskPixel badBitMask,
    int kernelHalfWidth,
    int kernelOrder,
    int xBackgroundSize,
    int yBackgroundSize,
    ndarray::Array<int, 1, 1> const& rows,
    int maxIter,
    int andersonDepth,
    double fluxTol,
    double lsqThreshold
) {
    if (kernelHalfWidth <= 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, "Kernel half-width must be positive");
    }
    if (kernelOrder < 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, "Kernel order must be non-negative");
    }
    if (xBackgroundSize <= 0 || yBackgroundSize <= 0) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterError, "Background block sizes must be positive"
        );
    }
    if (maxIter <= 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, "maxIter must be positive");
    }
    if (andersonDepth < 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, "andersonDepth must be non-negative");
    }
    if (!(std::isfinite(fluxTol) && fluxTol > 0.0)) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterError, "fluxTol must be finite and positive"
        );
    }
    if (!(std::isfinite(lsqThreshold) && lsqThreshold > 0.0)) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterError,
            "lsqThreshold must be finite and positive"
        );
    }
    return FiberKernelFitter(
        image, fiberTraces, badBitMask,
        kernelHalfWidth, kernelOrder,
        xBackgroundSize, yBackgroundSize,
        rows.isEmpty() ? utils::arange<int>(0, image.getHeight()) : rows
    ).run(maxIter, andersonDepth, fluxTol, lsqThreshold);
}


std::pair<ImageKernel, lsst::afw::image::Image<float>> fitImageKernel(
    lsst::afw::image::MaskedImage<float> const& source,
    lsst::afw::image::MaskedImage<float> const& target,
    lsst::afw::image::MaskPixel badBitMask,
    int kernelHalfWidth,
    int kernelOrder,
    int xBackgroundSize,
    int yBackgroundSize,
    ndarray::Array<int, 1, 1> const& rows,
    double lsqThreshold
) {
    if (kernelHalfWidth <= 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, "Kernel half-width must be positive");
    }
    if (kernelOrder < 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, "Kernel order must be non-negative");
    }
    if (xBackgroundSize <= 0 || yBackgroundSize <= 0) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterError, "Background block sizes must be positive"
        );
    }
    if (!(std::isfinite(lsqThreshold) && lsqThreshold > 0.0)) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterError,
            "lsqThreshold must be finite and positive"
        );
    }

    std::unordered_map<std::string, double> timings;
    auto const makeTimer = [&timings](std::string const& name) {
        return ScopedTimer(&timings[name]);
    };

    std::size_t const width = target.getWidth();
    std::size_t const numKernels = 2*kernelHalfWidth + 1;
    std::size_t const numKernelSpatial = (kernelOrder + 1)*(kernelOrder + 2)/2;
    BackgroundHelper bg(target.getDimensions(), xBackgroundSize, yBackgroundSize);
    std::size_t const numBackground = bg.xNumBlocks*bg.yNumBlocks;
    std::size_t bgStart = numKernels*numKernelSpatial;
    std::size_t const numParams = bgStart + numBackground;
    math::NormalizedPolynomial2<double> polynomial(kernelOrder, lsst::geom::Box2D(target.getBBox()));

    ndarray::Array<double, 2, 2> matrix = ndarray::allocate(numParams, numParams);
    ndarray::Array<double, 1, 1> vector = ndarray::allocate(numParams);
    matrix.deep() = 0.0;
    vector.deep() = 0.0;

    ndarray::Array<bool, 1, 1> usePixels = ndarray::allocate(width);
    ndarray::Array<double, 2, 2> polyValues = ndarray::allocate(numKernelSpatial, width);

    std::size_t const numRows = rows.isEmpty() ? target.getHeight() : rows.size();
    for (std::size_t ii = 0; ii < numRows; ++ii) {
        int const yy = rows.isEmpty() ? ii : rows[ii];
        if (yy < 0 || yy >= target.getHeight()) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::InvalidParameterError,
                (boost::format(
                    "Row %d is out of bounds for image height %d") % yy % target.getHeight()
                ).str()
            );
        }

        ndarray::asEigenArray(usePixels) = ndarray::asEigenArray(
            target.getMask()->getArray()[yy]
        ).unaryExpr([badBitMask](auto const& mask) { return (mask & badBitMask) == 0; });
        ndarray::Array<float const, 1, 1> image = target.getImage()->getArray()[yy];
        ndarray::Array<float const, 1, 1> variance = target.getVariance()->getArray()[yy];

        for (std::size_t xx = 0; xx < width; ++xx) {
            polyValues[ndarray::view()(xx)] = utils::vectorToArray(polynomial.getDFuncDParameters(xx, yy));
        }

        std::vector<std::vector<FiberModel>> kernelModels;  // [offsetIndex][spatialIndex]
        kernelModels.reserve(numKernels);
        FiberModel sourceCenter = FiberModel::fromImage(source, yy, badBitMask);
        std::size_t offsetIndex = 0;
        for (int offset = -kernelHalfWidth; offset <= kernelHalfWidth; ++offset, ++offsetIndex) {
            auto _t = makeTimer("calculate");
            std::vector<FiberModel> models;
            models.reserve(numKernelSpatial);
            FiberModel base = (offset == 0) ? sourceCenter : sourceCenter.applyOffset(offset, width);
            for (std::size_t spatial = 0; spatial < numKernelSpatial; ++spatial) {
                models.emplace_back(
                    base*polyValues[spatial][ndarray::view(base.xMin, base.xMax)]
                );
            }
            kernelModels.emplace_back(std::move(models));
        }

        // Model is:
        // Target(x,y) = sum_i ag1_i P_i(x,y) K_i(x)*Source(x,y) + sum_b a_b B_b(x,y)
        // Where P_i(x,y) are the spatial polynomials for kernel component i,
        // K_i(x) are the kernel components (delta functions at different offsets), and
        // B_b(x,y) are the background blocks.
        // a_i and a_b are the parameters we are solving for.
        //
        // Model dot model terms in the matrix are:
        // P_i.K_i*Source dot P_j.K_j*Source
        // P_i.K_i*Source dot B_b
        // B_b dot B_c
        //
        // Model dot data terms in the matrix are:
        // P_i.K_i*Source dot Target
        // B_b dot Target
        std::size_t iOffsetIndex = 0;
        std::size_t iKernelIndex = 0;
        for (int iOffset = -kernelHalfWidth; iOffset <= kernelHalfWidth; ++iOffset, ++iOffsetIndex) {
            auto _t = makeTimer("accumulate");
            for (std::size_t iSpatial = 0; iSpatial < numKernelSpatial; ++iSpatial, ++iKernelIndex) {
                FiberModel const& iModel = kernelModels[iOffsetIndex][iSpatial];
                vector[iKernelIndex] += iModel.dotData(image, usePixels, variance);
                matrix[iKernelIndex][iKernelIndex] += iModel.dotSelf(usePixels, variance);

                std::size_t jKernelIndex = iKernelIndex + 1;
                for (
                    std::size_t jSpatial = iSpatial + 1;
                    jSpatial < numKernelSpatial;
                    ++jSpatial, ++jKernelIndex
                ) {
                    FiberModel const& jModel = kernelModels[iOffsetIndex][jSpatial];
                    matrix[iKernelIndex][jKernelIndex] += iModel.dotOther(jModel, usePixels, variance);
                }

                jKernelIndex = (iOffsetIndex + 1)*numKernelSpatial;
                for (std::size_t jOffsetIndex = iOffsetIndex + 1; jOffsetIndex < numKernels; ++jOffsetIndex) {
                    for (std::size_t jSpatial = 0; jSpatial < numKernelSpatial; ++jSpatial, ++jKernelIndex) {
                        FiberModel const& jModel = kernelModels[jOffsetIndex][jSpatial];
                        matrix[iKernelIndex][jKernelIndex] += iModel.dotOther(jModel, usePixels, variance);
                    }
                }

                auto const dotBackground = iModel.dotBackground(bg.xBlocks, usePixels, variance);
                std::size_t bgIndex = bgStart + bg.yBlocks[yy]*bg.xNumBlocks + dotBackground.first;
                ndarray::Array<double, 1, 1> const& bgTerms = dotBackground.second;
                for (std::size_t ii = 0; ii < bgTerms.size(); ++ii, ++bgIndex) {
                    matrix[iKernelIndex][bgIndex] += bgTerms[ii];
                }
            }
        }

        auto iter = target.row_begin(yy);
        std::size_t const bgIndex = bgStart + bg.yBlocks[yy]*bg.xNumBlocks;
        for (int xx = 0; xx < target.getWidth(); ++xx, ++iter) {
            if (!usePixels[xx]) {
                continue;
            }
            std::size_t const block = bgIndex + bg.xBlocks[xx];
            double const weight = 1.0/iter.variance();
            matrix[block][block] += weight;
            vector[block] += iter.image()*weight;
        }
    }

    ndarray::Array<double, 1, 1> solution;
    {
        auto _t = makeTimer("solve");
        // Ensure we have entries for all parameters, to avoid singular matrix
        // Fill in the lower triangle of the matrix
        for (std::size_t ii = 0; ii < numParams; ++ii) {
            if (matrix[ii][ii] == 0.0) {
                assert(vector[ii] == 0.0);
                matrix[ii][ii] = 1.0;
            }
            for (std::size_t jj = ii + 1; jj < numParams; ++jj) {
                assert(matrix[jj][ii] == 0.0);
                matrix[jj][ii] = matrix[ii][jj];
            }
        }

        // Solve the system of equations
        auto lsq = lsst::afw::math::LeastSquares::fromNormalEquations(matrix, vector);
        lsq.setThreshold(lsqThreshold);
        solution = ndarray::copy(lsq.getSolution());
    }

    lsst::afw::image::Image<float> background(bg.xNumBlocks, bg.yNumBlocks);
    std::size_t bgIndex = bgStart;
    for (std::size_t yy = 0; yy < bg.yNumBlocks; ++yy) {
        for (std::size_t xx = 0; xx < bg.xNumBlocks; ++xx, ++bgIndex) {
            background.getArray()[yy][xx] = solution[bgIndex];
        }
    }

    std::cerr << "Timing results (accumulated over all iterations, seconds):\n";
    for (auto const& [name, elapsed] : timings) {
        std::cerr << "  " << name << ": " << elapsed << "\n";
    }

    return {
        ImageKernel(
            lsst::geom::Box2D(target.getBBox()),
            kernelHalfWidth,
            kernelOrder,
            solution[ndarray::view(0, bgStart)]
        ),
        std::move(background)
    };
}


}}}  // namespace pfs::drp::stella
