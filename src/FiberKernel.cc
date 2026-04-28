#include <algorithm>
#include <chrono>
#include <cmath>
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


using DotTerms = std::pair<std::size_t, ndarray::Array<double, 1, 1>>;

DotTerms emptyDotTerms() {
    return {-1, ndarray::Array<double, 1, 1>()};
}

using DotBackgroundTerms = std::vector<std::tuple<std::size_t, std::size_t, double>>;


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

    FiberModel applyOffset(int newOffset, int width) const {
        assert(offset == 0);  // haven't done the math to support offsets on top of offsets
        if (newOffset == 0) {
            return *this;
        }
        int const newMin = std::max(0, xMin + std::min(0, newOffset));
        int const newMax = std::min(width, xMax + std::max(0, newOffset));  // exclusive
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
            int const shiftedMin = std::max(0, xMin + newOffset);
            int const shiftedMax = std::min(width, xMax + newOffset);  // exclusive
            // Source array indices
            int const start = shiftedMin - newOffset - xMin;
            int const stop = shiftedMax - newOffset - xMin;  // exclusive
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

        return FiberModel{std::move(newValues), std::move(newUse), y, newMin, newMax, newWidth, newOffset};
    }

    template <typename T>
    FiberModel operator*(ndarray::ArrayRef<T, 1, 1> const& rhs) const {
        assert(rhs.size() == std::size_t(width));
        ndarray::Array<float, 1, 1> newValues = ndarray::copy(values);
        ndarray::asEigenArray(newValues) *= ndarray::asEigenArray(rhs).template cast<float>();
        return FiberModel{newValues, ndarray::copy(use), y, xMin, xMax, width, offset};
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

#if 0
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
#endif

    template <typename Func>
    DotTerms dotFunction(
        detail::PiecewiseConstantInterpolator const& kernelInterp,
        ndarray::Array<bool const, 1, 1> const& usePixels,
        Func func
    ) const {
        int const imageWidth = usePixels.size();
        int const xStart = std::max(xMin, 0);
        int const xStop = std::min(xMax, imageWidth - 1);  // inclusive
        auto const blocks = kernelInterp.getXBlocks();
        std::size_t const minBlock = blocks[xStart];
        std::size_t const maxBlock = blocks[xStop];
        std::size_t const numBlocks = maxBlock - minBlock + 1;
        ndarray::Array<double, 1, 1> terms = ndarray::allocate(numBlocks);
        terms.deep() = 0.0;
        for (
            int xModel = xStart - xMin, xData = xStart;
            xModel < width && xData < imageWidth;
            ++xModel, ++xData
        ) {
            if (!use[xModel]) continue;
            if (!usePixels[xData]) continue;
            std::size_t const blockIndex = blocks[xData];
            terms[blockIndex - minBlock] += func(xModel, xData);
        }
        return {minBlock, terms};
    }

    DotTerms dotSelf(
        detail::PiecewiseConstantInterpolator const& kernelInterp,
        ndarray::Array<bool const, 1, 1> const& usePixels,
        ndarray::Array<float const, 1, 1> const& dataVariance
    ) const {
        return dotFunction(kernelInterp, usePixels, [this, &dataVariance](int xModel, int xData) {
            return values[xModel]*values[xModel]/dataVariance[xData];
        });
    }

    DotTerms dotData(
        detail::PiecewiseConstantInterpolator const& kernelInterp,
        ndarray::Array<float const, 1, 1> const& dataValues,
        ndarray::Array<bool const, 1, 1> const& usePixels,
        ndarray::Array<float const, 1, 1> const& dataVariance
    ) const {
        return dotFunction(
            kernelInterp,
            usePixels,
            [this, &dataValues, &dataVariance](int xModel, int xData) {
                return values[xModel]*dataValues[xData]/dataVariance[xData];
            }
        );
    }

    DotTerms dotOther(
        detail::PiecewiseConstantInterpolator const& kernelInterp,
        FiberModel const& other,
        ndarray::Array<bool const, 1, 1> const& usePixels,
        ndarray::Array<float const, 1, 1> const& dataVariance
    ) const {
        return dotFunction(
            kernelInterp,
            usePixels,
            [this, &other, &dataVariance](int xModel, int xData) {
                int const xOther = xData - other.xMin;
                if (xOther < 0 || xOther >= other.width) {
                    return 0.0f;
                }
                return values[xModel]*other.values[xOther]*dataVariance[xData];
            }
        );
    }

    // Dot the background with something that's not a kernel (e.g., the image)
    DotTerms dotBackground(
        detail::PiecewiseConstantInterpolator const& bgInterp,
        ndarray::Array<bool const, 1, 1> const& usePixels,
        ndarray::Array<float const, 1, 1> const& dataVariance
    ) const {
        return dotFunction(
            bgInterp,
            usePixels,
            [this, &dataVariance](int xModel, int xData) {
                return values[xModel]/dataVariance[xData];
            }
        );
    }

    // Dot the background with a kernel
    DotBackgroundTerms dotBackground(
        detail::PiecewiseConstantInterpolator const& kernelInterp,
        detail::PiecewiseConstantInterpolator const& bgInterp,
        ndarray::Array<bool const, 1, 1> const& usePixels,
        ndarray::Array<float const, 1, 1> const& dataVariance
    ) const {
        int const imageWidth = usePixels.size();
        int const xStart = std::max(xMin, 0);
        int const xStop = std::min(xMax, imageWidth - 1);  // inclusive
        auto const kernelBlocks = kernelInterp.getXBlocks();
        auto const bgBlocks = bgInterp.getXBlocks();

        std::size_t const kernelMinBlock = kernelBlocks[xStart];
        std::size_t const kernelMaxBlock = kernelBlocks[xStop];
        std::size_t const bgMinBlock = bgBlocks[xStart];
        std::size_t const bgMaxBlock = bgBlocks[xStop];

        DotBackgroundTerms terms;
        terms.reserve((kernelMaxBlock - kernelMinBlock + 1)*(bgMaxBlock - bgMinBlock + 1));
        std::size_t kernelIndex = 0;
        std::size_t bgIndex = 0;
        double sum = 0.0;
        for (
            int xModel = xStart - xMin, xData = xStart;
            xModel < width && xData < imageWidth;
            ++xModel, ++xData
        ) {
            if (!use[xModel]) continue;
            if (!usePixels[xData]) continue;
            std::size_t const kernelBlockIndex = kernelBlocks[xData];
            std::size_t const bgBlockIndex = bgBlocks[xData];

            if (kernelBlockIndex != kernelIndex || bgBlockIndex != bgIndex) {
                if (sum != 0.0) {
                    terms.emplace_back(kernelIndex, bgIndex, sum);
                }
                kernelIndex = kernelBlockIndex;
                bgIndex = bgBlockIndex;
                sum = 0.0;
            }
            sum += values[xModel]/dataVariance[xData];
        }
        if (sum != 0.0) {
            terms.emplace_back(kernelIndex, bgIndex, sum);
        }
        return terms;
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


BaseKernel::BaseKernel(
    lsst::geom::Extent2I const& dims,
    int halfWidth,
    std::size_t numParams,
    ndarray::ArrayRef<double const, 1, 1> const& coefficients
) : _dims(dims),
    _halfWidth(halfWidth),
    _numParams(numParams),
    _coefficients(ndarray::copy(coefficients))
{
    utils::checkSize(coefficients.size(), _numParams, "coefficients");
}


FiberTraceSet<float> BaseKernel::convolve(
    FiberTraceSet<float> const& traces,
    lsst::geom::Box2I const& bbox
) const {
    FiberTraceSet<float> result(traces.size());
    for (std::size_t ii = 0; ii < traces.size(); ++ii) {
        result.add(convolve(*traces[ii], bbox));
    }
    return result;
}


FiberKernel::FiberKernel(
    lsst::geom::Extent2I const& dims,
    int halfWidth,
    int xNumBlocks,
    int yNumBlocks,
    ndarray::ArrayRef<double const, 1, 1> const& coefficients
) : BaseKernel(
        dims,
        halfWidth,
        (2*halfWidth)*xNumBlocks*yNumBlocks,
        coefficients
    ),
    _interp(dims, xNumBlocks, yNumBlocks, 2*halfWidth)
    {}


std::shared_ptr<FiberTrace<float>> FiberKernel::convolveImpl(
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
    *convolved.getMask() = 0;
    *convolved.getVariance() = 0.0;

    int const xMax = trace.getTrace().getWidth() - 1;  // inclusive
    for (int yy = 0; yy < trace.getTrace().getHeight(); ++yy) {
        auto const& model = FiberModel::fromFiberTrace(trace, yy, 0, xMax, require);
        float const xCenter = model.centroid();
        model.addToImage(convImage, 1.0);

        std::size_t offsetIndex = _interp.getIndex(xCenter, yy);
        for (int offset = -_halfWidth; offset <= _halfWidth; ++offset, ++offsetIndex) {
            if (offset == 0) {
                --offsetIndex;
                continue;
            }
            auto kernelModel = model.applyOffset(offset, newBox.getMaxX() + 1);
            kernelModel.addToImage(convImage, _coefficients[offsetIndex]);
        }
    }

    for (auto iter = convolved.begin(); iter != convolved.end(); ++iter) {
        if (iter.image() != 0.0) {
            iter.mask() |= require;
        }
    }

    return std::make_shared<FiberTrace<float>>(std::move(convolved), trace.getFiberId());
}


lsst::afw::image::Image<float> FiberKernel::convolveImpl(
    lsst::afw::image::Image<float> const& image
) const {
    std::size_t height = image.getHeight();
    lsst::afw::image::Image<float> result{image.getBBox()};
    result = 0.0;

    for (std::size_t yy = 0; yy < height; ++yy) {
        auto const model = FiberModel::fromImage(image, yy);

        std::size_t offsetIndex = 0;
        for (int offset = -_halfWidth; offset <= _halfWidth; ++offset, ++offsetIndex) {
            if (offset == 0) {
                auto inIter = model.values.begin();
                auto outIter = result.row_begin(yy) + model.xMin;
                for (int xx = model.xMin; xx < model.xMax; ++xx, ++inIter, ++outIter) {
                    *outIter += *inIter;
                }
                --offsetIndex;
                continue;
            }
            FiberModel const kernelModel = model.applyOffset(offset, image.getWidth());

            auto inIter = kernelModel.values.begin();
            auto outIter = result.row_begin(yy) + kernelModel.xMin;
            for (int xx = kernelModel.xMin; xx < kernelModel.xMax; ++xx, ++inIter, ++outIter) {
                double const index = _interp.getIndex(xx, yy) + offsetIndex;
                *outIter += (*inIter)*_coefficients[index];
            }
        }
    }
    return result;
}


ndarray::Array<double, 1, 1> FiberKernel::evaluate(double x, double y) const {
    ndarray::Array<double, 1, 1> result = ndarray::allocate(2*_halfWidth + 1);
    result[_halfWidth] = 0.0;
    std::size_t offsetIndex = _interp.getIndex(x, y);
    for (int pixel = 0, offset = -_halfWidth; offset <= _halfWidth; ++pixel, ++offset, ++offsetIndex) {
        if (offset == 0) {
            --offsetIndex;
            continue;
        }
        double const value = _coefficients[offsetIndex];
        result[pixel] = value;
        result[_halfWidth] -= value;
    }
    return result;
}


ndarray::Array<double, 3, 3> FiberKernel::makeOffsetImagesImpl(
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

        double const dx = _dims.getX()/(dims.getX() - 1);
        double const dy = _dims.getY()/(dims.getY() - 1);

        ndarray::ArrayRef<double, 2, 2> image = result[imageIndex];
        double ySample = 0;
        for (int yy = 0; yy < dims.getY(); ++yy, ySample += dy) {
            auto iter = image[yy].begin();
            double xSample = 0;
            for (int xx = 0; xx < dims.getX(); ++xx, ++iter, xSample += dx) {
                std::size_t const index = _interp.getIndex(xSample, ySample);
                *iter = _coefficients[offsetIndex + index];
            }
        }
        ndarray::asEigenArray(center) -= ndarray::asEigenArray(image);
    }
    return result;
}


detail::PiecewiseConstantInterpolator::PiecewiseConstantInterpolator(
    lsst::geom::Extent2I const& dims,
    int xNumBlocks,
    int yNumBlocks,
    std::size_t step
) : _xNumBlocks(xNumBlocks),
    _yNumBlocks(yNumBlocks),
    _step(step),
    _xBlocks(ndarray::allocate(dims.getX())),
    _yBlocks(ndarray::allocate(dims.getY())) {
    if (xNumBlocks <= 0 || yNumBlocks <= 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, "Number of blocks must be positive");
    }
    double const dx = static_cast<double>(dims.getX())/xNumBlocks*step;
    double const dy = static_cast<double>(dims.getY())/yNumBlocks*step*xNumBlocks;

    for (int ii = 0; ii < dims.getX(); ++ii) {
        _xBlocks[ii] = ii*dx;
    }
    for (int ii = 0; ii < dims.getY(); ++ii) {
        _yBlocks[ii] = ii*dy;
    }
}


namespace {


struct RowData {
    int y;
    ndarray::Array<bool, 1, 1> useTrace;  // trace overlaps this row?
    ndarray::Array<bool, 1, 1> usePixel;  // pixel in row should be used?
    ndarray::Array<double, 1, 1> xCenter; // center of trace in x, for selecting kernel block
    std::vector<FiberModel> models;  // model for each fiber at this row
    std::vector<std::vector<FiberModel>> kernelModels;  // kernel-offset models [fiber][offset]
    std::vector<std::vector<std::size_t>> overlaps;  // [fiber] overlapping fiber indices within kernelHalfWidth

    // Layout of models: offset=-kernelWidth, ... offset=0, ... offset=+kernelWidth
    // Note that this is different from the layout of coefficients in the matrix, which skips offset=0.
    std::vector<std::vector<DotTerms>> dotData;  // [fiber][offset]

    // [fiber][otherFiber][offset][otherOffset] model dot Model; first otherFiber is the same as fiber
    std::vector<std::vector<std::vector<std::vector<DotTerms>>>> dotModel;

    std::vector<std::vector<DotBackgroundTerms>> dotBackground;  // [fiber][offset]
    DotTerms dotBackgroundImage;
    DotTerms dotBackgroundConstant;

    RowData(
        int y, std::size_t numFibers, int imageWidth, std::size_t kernelHalfWidth
    ) : y(y),
        useTrace(ndarray::allocate(numFibers)),
        usePixel(ndarray::allocate(imageWidth)),
        xCenter(ndarray::allocate(numFibers)),
        models(numFibers, FiberModel::dummy()),
        kernelModels(numFibers, std::vector<FiberModel>(2*kernelHalfWidth + 1, FiberModel::dummy())),
        overlaps(numFibers),
        dotData(numFibers),
        dotModel(numFibers),
        dotBackground(numFibers)
    {
        useTrace.deep() = false;
        usePixel.deep() = false;
        for (std::size_t ii = 0; ii < numFibers; ++ii) {
            dotData[ii].reserve(2*kernelHalfWidth + 1);
            dotBackground[ii].reserve(2*kernelHalfWidth + 1);
        }
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
        int xKernelNum,
        int yKernelNum,
        int xBackgroundNum,
        int yBackgroundNum,
        ndarray::Array<int, 1, 1> const& rows
    ) : _numFibers(fiberTraces.size()),
        _numRows(rows.size()),
        _image(image),
        _fiberTraces(fiberTraces),
        _badBitMask(badBitMask),
        _kernelHalfWidth(kernelHalfWidth),
        _rows(rows),
        _numKernel(2*kernelHalfWidth),
        _kernel(image.getDimensions(), xKernelNum, yKernelNum, _numKernel),
        _bg(image.getDimensions(), xBackgroundNum, yBackgroundNum),
        _bgStart(2*kernelHalfWidth*_kernel.getNumBlocks()),
        _requireMask(image.getMask()->getPlaneBitMask(fiberMaskPlane)),
        _numParams(_bgStart + _bg.getNumBlocks())
    {}

    // Determine which traces are relevant for this row
    RowData calculateRow(int y) const {
        RowData data(y, _numFibers, _image.getWidth(), _kernelHalfWidth);
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

        ndarray::Array<double, 1, 1> dotBackgroundImage = ndarray::allocate(_bg.getXNumBlocks());
        dotBackgroundImage.deep() = 0.0;
        ndarray::Array<double, 1, 1> dotBackgroundConstant = ndarray::allocate(_bg.getXNumBlocks());
        dotBackgroundConstant.deep() = 0.0;
        auto iter = _image.row_begin(y);
        for (int xx = 0; xx < _image.getWidth(); ++xx, ++iter) {
            bool const usePixel = isGoodImage(iter);
            data.usePixel[xx] = usePixel;
            if (!usePixel) {
                continue;
            }
            std::size_t const index = _bg.getIndex(xx, y);
            double const invVar = 1.0/iter.variance();
            dotBackgroundImage[index] += iter.image()*invVar;
            dotBackgroundConstant[index] += invVar;
        }

        std::size_t const bgStart = _bgStart + _bg.getIndex(0, y);
        data.dotBackgroundImage = std::make_pair(bgStart, dotBackgroundImage);
        data.dotBackgroundConstant = std::make_pair(bgStart, dotBackgroundConstant);

        return data;
    }

    // Layout of parameters for kernel+background fit:
    // Kernel parameters: kernel block runs slow, offset (...,-2,-1,1,2,... NOTE: no zero!) runs fast
    //     starts at index 0
    // Background parameters: x block (0.._bg.xNumBlocks-1) runs fast, y block (0.._bg.yNumBlocks-1) runs
    //     slow; starts at _bgStart

    std::size_t getKernelIndex(int offset, int x, int y) const {
        return getKernelIndex(offset, _kernel.getIndex(x, y));
    }

    std::size_t getKernelIndex(int offset, std::size_t blockIndex) const {
        int const offsetIndex = offset + _kernelHalfWidth - (offset < 0 ? 0 : 1);
        return blockIndex*2*_kernelHalfWidth + offsetIndex;
    }

    std::size_t getBackgroundIndex(int x, int y) const {
        return getBackgroundIndex(_bg.getIndex(x, y));
    }

    std::size_t getBackgroundIndex(std::size_t blockIndex) const {
        return _bgStart + blockIndex;
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

    DotTerms getDotModel(
        RowData const& data,
        std::size_t leftFiber,
        std::size_t rightFiber,
        std::size_t leftOffsetIndex,
        std::size_t rightOffsetIndex
    ) const {
        if (leftFiber <= rightFiber) {
            std::size_t const delta = rightFiber - leftFiber;
            if (delta >= data.dotModel[leftFiber].size()) {
                return emptyDotTerms();
            }
            return data.dotModel[leftFiber][delta][leftOffsetIndex][rightOffsetIndex];
        }

        std::size_t const delta = leftFiber - rightFiber;
        if (delta >= data.dotModel[rightFiber].size()) {
            return emptyDotTerms();
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

        std::size_t iIndex = 0;
        for (int iOffset = -_kernelHalfWidth; iOffset <= _kernelHalfWidth; ++iOffset, ++iIndex) {
            FiberModel const& iModel = data.kernelModels[fiberIndex][iIndex];
            data.dotData[fiberIndex].emplace_back(
                iModel.dotData(_kernel, dataImage, data.usePixel, dataVariance)
            );
            data.dotBackground[fiberIndex].emplace_back(
                iModel.dotBackground(_kernel, _bg, data.usePixel, dataVariance)
            );
        }

        std::vector<std::vector<std::vector<DotTerms>>> dotModel;
        dotModel.reserve(_numFibers - fiberIndex);
        for (std::size_t jFiberIndex = fiberIndex; jFiberIndex < _numFibers; ++jFiberIndex) {
            std::vector<std::vector<DotTerms>> dotModelFiber;
            dotModelFiber.reserve(2*_kernelHalfWidth + 1);
            if (!data.useTrace[jFiberIndex]) {
                dotModelFiber.emplace_back(2*_kernelHalfWidth + 1, emptyDotTerms());
                continue;
            }

            std::size_t iIndex = 0;
            bool allZero = true;
            for (int iOffset = -_kernelHalfWidth; iOffset <= _kernelHalfWidth; ++iOffset, ++iIndex) {
                FiberModel const& iModel = data.kernelModels[fiberIndex][iIndex];
                std::size_t jIndex = 0;
                for (int jOffset = -_kernelHalfWidth; jOffset <= _kernelHalfWidth; ++jOffset, ++jIndex) {
                    FiberModel const& jModel = data.kernelModels[jFiberIndex][jIndex];
                    DotTerms const terms = iModel.dotOther(_kernel, jModel, data.usePixel, dataVariance);
                    if (terms.first >= 0) {
                        allZero = false;
                    }
                    dotModelFiber[iIndex][jIndex] = std::move(terms);
                }
            }
            if (allZero) {
                // The fibers are ordered, so all subsequent fibers will also be zero
                break;
            }
            dotModel.push_back(std::move(dotModelFiber));
        }
        data.dotModel[fiberIndex] = std::move(dotModel);
    }

    // Image(x,y) = sum_i F_i(y).p_i(x,y) + sum_i sum_j sum_k a_ij.F_i(y).K_jk(x,y) + sum_b a_b B_b(x,y)
    // Where:
    // F_i(y) is the flux of fiber i at row y
    // p_i(x,y) is the fiber profile for fiber i at row y
    // K_jk(x,y) = [delta(x-offset_j) - delta(x)]*p_i(x,y).D_k(x,y) is the kernel component for offset j and
    //    block k (D_k = unity at block k, zero elsewhere).
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

        double const iFlux = flux[fiberIndex];

        std::size_t iOffsetIndex = 0;
        for (int iOffset = -_kernelHalfWidth; iOffset <= _kernelHalfWidth; ++iOffset, ++iOffsetIndex) {
            DotBackgroundTerms const& bgTerms = data.dotBackground[fiberIndex][iOffsetIndex];
            if (iOffset == 0) {
                // Subtracting sum_i F_i(y).p_i(x,y) dot B_b(x,y) from the vector
                auto _t = timer("bg_vector");
                assert(iOffsetIndex == std::size_t(_kernelHalfWidth));
                for (const auto& [kIndex, bgIndex, term] : bgTerms) {
                    vector[bgIndex] -= term*iFlux;
                }
                continue;
            }

            // Vector term from the image: F_i(y).p_i(x,y) dot K_j(x,y)
            {
                auto _t = timer("vector_datum");

                auto const& dotData = data.dotData[fiberIndex][iOffsetIndex];
                std::size_t dotStart = dotData.first;
                auto const& dotTerms = dotData.second;

                std::size_t index = dotStart + iOffsetIndex;
                for (std::size_t ii = 0; ii < dotTerms.size(); ++ii, index += _numKernel) {
                    vector[index] += iFlux*dotTerms[ii];
                }
            }

            // Subtracting sum_k F_k(y).p_k(x,y) dot F_i(y).K_j(x,y) from the vector
            {
                auto _t = timer("vector_cross");
                for (std::size_t jFiberIndex : data.overlaps[fiberIndex]) {
                    double const jFlux = flux[jFiberIndex];
                    auto const dotModel = getDotModel(
                        data, fiberIndex, jFiberIndex, iOffsetIndex, _kernelHalfWidth
                    );
                    std::size_t index = dotModel.first + iOffsetIndex;
                    ndarray::Array<double, 1, 1> const& terms = dotModel.second;
                    for (std::size_t ii = 0; ii < terms.size(); ++ii, index += _numKernel) {
                        vector[index] -= iFlux*jFlux*terms[ii];
                    }
                }
            }

            // Kernel-kernel terms for all fibers.
            // We accumulate only the upper triangle (jKernelIndex >= iKernelIndex),
            // but include all ordered fiber pairs for the full normal equations.
            {
                auto _t = timer("matrix_kernel_kernel");
                for (std::size_t jFiberIndex : data.overlaps[fiberIndex]) {
                    double const jFlux = flux[jFiberIndex];

                    std::size_t jOffsetIndex = 0;
                    for (
                        int jOffset = -_kernelHalfWidth;
                        jOffset <= _kernelHalfWidth;
                        ++jOffset, ++jOffsetIndex
                    ) {
                        if (jOffset == 0) {
                            continue;
                        }
                        auto const dotKernel = getDotModel(
                            data, fiberIndex, jFiberIndex, iOffsetIndex, jOffsetIndex
                        );
                        std::size_t const start = dotKernel.first;
                        ndarray::Array<double, 1, 1> const& terms = dotKernel.second;

                        std::size_t iIndex = start + iOffsetIndex;
                        std::size_t jIndex = start + jOffsetIndex;
                        for (
                            std::size_t ii = 0;
                            ii < terms.size();
                            ++ii, iIndex += _numKernel, jIndex += _numKernel
                        ) {
                            assert(iIndex <= jIndex);
                            matrix[iIndex][jIndex] += iFlux*jFlux*terms[ii];
                        }
                    }
                }
            }

            // Kernel-background cross-terms
            // F_i(y).[K_j(x,y) dot B_b(x,y)]
            {
                auto _t = timer("matrix_kernel_bg");
                for (const auto& [kIndex, bgIndex, term] : bgTerms) {
                    matrix[kIndex + iOffsetIndex][bgIndex] += iFlux*term;
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
        ndarray::Array<double, 1, 1> const& dotBackgroundImage = data.dotBackgroundImage.second;
        ndarray::Array<double, 1, 1> const& dotBackgroundConstant = data.dotBackgroundConstant.second;
        std::size_t bgIndex = data.dotBackgroundImage.first;
        assert(bgIndex == data.dotBackgroundConstant.first);
        assert(dotBackgroundImage.size() == dotBackgroundConstant.size());
        for (std::size_t ii = 0; ii < dotBackgroundImage.size(); ++ii, ++bgIndex) {
            vector[bgIndex] += dotBackgroundImage[ii];
            matrix[bgIndex][bgIndex] += dotBackgroundConstant[ii];
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
            _image.getDimensions(),
            _kernel.getXNumBlocks(),
            _kernel.getYNumBlocks(),
            _kernelHalfWidth,
            solution[ndarray::view(0, _bgStart)]
        );

        // Extract the background
        lsst::afw::image::Image<float> background(_bg.getXNumBlocks(), _bg.getYNumBlocks());
        std::size_t bgIndex = _bgStart;
        for (std::size_t yy = 0; yy < _bg.getYNumBlocks(); ++yy, bgIndex += _bg.getXNumBlocks()) {
            background.getArray()[yy] = solution[ndarray::view(bgIndex, bgIndex + _bg.getXNumBlocks())];
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

    // Assemble a kernel for each fiber
    ndarray::Array<double, 2, 2> calculateKernelValues(
        RowData const& data,
        ndarray::Array<double const, 1, 1> const& kernelSolution
    ) const {
        ndarray::Array<double, 2, 2> kernelValues = ndarray::allocate(_numFibers, _numKernel);
        for (std::size_t fiberIndex = 0; fiberIndex < _numFibers; ++fiberIndex) {
            if (!data.useTrace[fiberIndex]) {
                continue;
            }
            std::size_t const start = _kernel.getIndex(data.xCenter[fiberIndex], data.y);
            kernelValues[fiberIndex].deep() = kernelSolution[ndarray::view(start, start + _numKernel)];
        }
        return kernelValues;
    }

    void accumulateFlux(
        std::size_t fiberIndex,
        std::size_t rowNum,
        RowData const& data,
        ndarray::Array<double const, 1, 1> const& kernelSolution,
        math::SymmetricSparseSquareMatrix & matrix,
        ndarray::Array<double, 1, 1> & vector
    ) const {
        auto _t = timer("flux");

        if (!data.useTrace[fiberIndex]) {
            return;
        }
        std::size_t const iFluxIndex = getFluxIndex(rowNum, fiberIndex);

        // Image(x,y) = sum_i F_i(y).[p_i(x,y) + sum_I a_I K_I(x,y)] + sum_b a_b B_b(x,y)
        // Solving for the F_i(y)
        // a_I and a_b are fixed parameters from the kernel/background fit.
        // Model_i = p_i + sum_I a_I.K_I
        // model_i dot model_j = p_i dot p_j + sum_J a_J p_i dot K_J + sum_I a_I K_I dot p_j + sum_I sum_J a_I a_J K_I dot K_J
        // model_i dot (data - sum_b a_b B_b) = p_i dot data + sum_I a_I K_I dot data - sum_b a_b p_i dot B_b - sum_I sum_b a_I a_b K_I dot B_b

        // Calculate model dot data
        double modelDotData = 0.0;
        std::size_t offsetIndex = 0;
        std::size_t kernelIndex = 0;
        for (
            int offset = -_kernelHalfWidth;
            offset <= _kernelHalfWidth;
            ++offset, ++offsetIndex, ++kernelIndex
        ) {
            DotTerms const& dotData = data.dotData[fiberIndex][offsetIndex];
            DotBackgroundTerms const& dotBackground = data.dotBackground[fiberIndex][offsetIndex];

            if (offset == 0) {
                modelDotData += ndarray::asEigenArray(dotData.second).sum();  // p_i dot data
                for (auto const& [kIndex, bgIndex, term] : dotBackground) {
                    modelDotData -= term*kernelSolution[bgIndex];  // sum_b a_b p_i dot B_b
                }
                continue;
                --kernelIndex;
            }

            std::size_t index = dotData.first + kernelIndex;
            ndarray::Array<double, 1, 1> const& terms = dotData.second;
            for (std::size_t ii = 0; ii < terms.size(); ++ii, index += _numKernel) {
                modelDotData += kernelSolution[index]*terms[ii];  // sum_I a_I K_I dot data
            }

            for (auto const& [kIndex, bgIndex, term] : dotBackground) {
                // sum_I sum_b a_I a_b K_I dot B_b
                modelDotData -= kernelSolution[kIndex]*kernelSolution[bgIndex]*term;
            }
        }
        vector[iFluxIndex] = modelDotData;

        // Calculate model dot model
        std::size_t const numOverlaps = data.dotModel[fiberIndex].size();
        for (std::size_t jj = 0, jFiberIndex = fiberIndex; jj < numOverlaps; ++jj, ++jFiberIndex) {
            if (!data.useTrace[jFiberIndex]) {
                continue;
            }
            std::size_t jFluxIndex = getFluxIndex(rowNum, jFiberIndex);
            double modelDotModel = 0.0;

            std::size_t iOffsetIndex = 0;
            std::size_t iKernelIndex = 0;
            for (
                int offset = -_kernelHalfWidth;
                offset <= _kernelHalfWidth;
                ++offset, ++iOffsetIndex, ++iKernelIndex
            ) {
                if (offset == 0) {
                    --iKernelIndex;

                    DotTerms const& dotModel = getDotModel(
                        data, fiberIndex, jFiberIndex, _kernelHalfWidth, _kernelHalfWidth
                    );
                    modelDotModel += ndarray::asEigenArray(dotModel.second).sum();  // p_i dot p_j

                    std::size_t jOffsetIndex = 0;
                    std::size_t jKernelIndex = 0;
                    for (
                        int jOffset = -_kernelHalfWidth;
                        jOffset <= _kernelHalfWidth;
                        ++jOffsetIndex, ++jKernelIndex
                    ) {
                        if (jOffset == 0) {
                            --jKernelIndex;
                        }

                        DotTerms const& dotModel = getDotModel(
                            data, fiberIndex, jFiberIndex, iOffsetIndex, jOffsetIndex
                        );

                        std::size_t index = dotModel.first + jKernelIndex;
                        ndarray::Array<double, 1, 1> const& terms = dotModel.second;
                        for (std::size_t ii = 0; ii < terms.size(); ++ii, index += _numKernel) {
                            modelDotModel += kernelSolution[index]*terms[ii];  // sum_J a_J p_i dot K_J
                        }
                    }

                    continue;
                }

                std::size_t jOffsetIndex = 0;
                std::size_t jKernelIndex = 0;
                for (
                    int jOffset = -_kernelHalfWidth;
                    jOffset <= _kernelHalfWidth;
                    ++jOffsetIndex, ++jKernelIndex
                ) {
                    if (jOffset == 0) {
                        --jKernelIndex;

                        auto const dotModel = getDotModel(
                            data, fiberIndex, jFiberIndex, iOffsetIndex, _kernelHalfWidth
                        );
                        std::size_t const start = dotModel.first;
                        ndarray::Array<double, 1, 1> const& terms = dotModel.second;
                        std::size_t index = start + iKernelIndex;
                        for (std::size_t ii = 0; ii < terms.size(); ++ii, index += _numKernel) {
                            modelDotModel += kernelSolution[index]*terms[ii];  // sum_I a_I K_I dot p_j
                        }

                    }

                    auto const dotModel = getDotModel(
                        data, fiberIndex, jFiberIndex, iOffsetIndex, jOffsetIndex
                    );
                    std::size_t const start = dotModel.first;
                    ndarray::Array<double, 1, 1> const& terms = dotModel.second;
                    std::size_t iIndex = start + iKernelIndex;
                    std::size_t jIndex = start + jKernelIndex;
                    for (
                        std::size_t ii = 0;
                        ii < terms.size();
                        ++ii, iIndex += _numKernel, jIndex += _numKernel
                    ) {
                        // sum_I sum_J a_I a_J K_I dot K_J
                        modelDotModel += kernelSolution[iIndex]*kernelSolution[jIndex]*terms[ii];
                    }
                }
            }

            matrix.add(iFluxIndex, jFluxIndex, modelDotModel);
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
            ndarray::Array<double, 2, 2> kernelValues = calculateKernelValues(data[ii], kernelSolution);
            for (std::size_t fiberIndex = 0; fiberIndex < _numFibers; ++fiberIndex) {
                accumulateFlux(fiberIndex, ii, data[ii], kernelSolution, matrix, vector);
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
//        std::cerr << "Initial flux: " << flux[0] << std::endl;

        std::vector<ndarray::Array<double, 1, 1>> kernelHistory;
        std::vector<ndarray::Array<double, 1, 1>> fluxHistory;
        kernelHistory.reserve(andersonDepth + 2);
        fluxHistory.reserve(andersonDepth + 2);

        bool converged = false;
        for (int ii = 0; ii < maxIter; ++ii) {
            auto _t = timer("iteration");
            ndarray::Array<double, 1, 1> const fluxVector = utils::flattenArray(flux);
            kernel = fitKernel(data, flux, lsqThreshold);
//            std::cerr << "New kernel: " << kernel << std::endl;

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
//            std::cerr << "New flux: " << flux[0] << std::endl;
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
    ndarray::Array<int, 1, 1> const& _rows;

    // Helpers
    std::size_t _numKernel;  // Number of parameters per kernel block
    detail::PiecewiseConstantInterpolator _kernel;
    detail::PiecewiseConstantInterpolator _bg;
    std::size_t _bgStart;  // starting index of background parameters for current block
    lsst::afw::image::MaskPixel _requireMask;
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
    int xKernelNum,
    int yKernelNum,
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
    if (xKernelNum <= 0 || yKernelNum <= 0) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterError,
            "Kernel interpolation block counts must be positive"
        );
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
        kernelHalfWidth, xKernelNum, yKernelNum,
        xBackgroundSize, yBackgroundSize,
        rows.isEmpty() ? utils::arange<int>(0, image.getHeight()) : rows
    ).run(maxIter, andersonDepth, fluxTol, lsqThreshold);
}


std::pair<FiberKernel, lsst::afw::image::Image<float>> fitFiberKernel(
    lsst::afw::image::MaskedImage<float> const& source,
    lsst::afw::image::MaskedImage<float> const& target,
    lsst::afw::image::MaskPixel badBitMask,
    int kernelHalfWidth,
    int xKernelNum,
    int yKernelNum,
    int xBackgroundNum,
    int yBackgroundNum,
    ndarray::Array<int, 1, 1> const& rows,
    double lsqThreshold
) {
    if (kernelHalfWidth <= 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, "Kernel half-width must be positive");
    }
    if (xKernelNum <= 0 || yKernelNum <= 0) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterError,
            "Kernel interpolation block counts must be positive"
        );
    }
    if (xBackgroundNum <= 0 || yBackgroundNum <= 0) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterError, "Background block numbers must be positive"
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
    std::size_t const numKernels = 2*kernelHalfWidth;
    detail::PiecewiseConstantInterpolator kernel(target.getDimensions(), xKernelNum, yKernelNum);
    detail::PiecewiseConstantInterpolator bg(target.getDimensions(), xBackgroundNum, yBackgroundNum);
    std::size_t const numBackground = bg.getNumBlocks();
    std::size_t bgStart = numKernels*kernel.getNumBlocks();
    std::size_t const numParams = bgStart + numBackground;

    ndarray::Array<double, 2, 2> matrix = ndarray::allocate(numParams, numParams);
    ndarray::Array<double, 1, 1> vector = ndarray::allocate(numParams);
    matrix.deep() = 0.0;
    vector.deep() = 0.0;

    ndarray::Array<bool, 1, 1> usePixels = ndarray::allocate(width);

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
        ndarray::Array<float, 1, 1> image = ndarray::copy(target.getImage()->getArray()[yy]);
        ndarray::asEigenArray(image) -= ndarray::asEigenArray(source.getImage()->getArray()[yy]);
        ndarray::Array<float, 1, 1> variance = ndarray::copy(target.getVariance()->getArray()[yy]);
        ndarray::asEigenArray(variance) += ndarray::asEigenArray(source.getVariance()->getArray()[yy]);

        std::vector<FiberModel> kernelModels;  // [offsetIndex]
        kernelModels.reserve(numKernels);
        FiberModel sourceCenter = FiberModel::fromImage(source, yy, badBitMask);
        for (int offset = -kernelHalfWidth; offset <= kernelHalfWidth; ++offset) {
            if (offset == 0) {
                continue;
            }
            auto _t = makeTimer("calculate");
            kernelModels.emplace_back(sourceCenter.applyOffset(offset, width));
        }

        // Model is:
        // Target(x,y) = Source(x,y) + sum_i ag1_i P_i(x,y) K_i(x)*Source(x,y) + sum_b a_b B_b(x,y)
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
        for (int iOffset = -kernelHalfWidth; iOffset <= kernelHalfWidth; ++iOffset, ++iOffsetIndex) {
            if (iOffset == 0) {
                --iOffsetIndex;
                continue;
            }
            auto _t = makeTimer("accumulate");
            FiberModel const& iModel = kernelModels[iOffsetIndex];

            auto const dotData = iModel.dotData(kernel, image, usePixels, variance);
            std::size_t index = dotData.first + iOffsetIndex;
            ndarray::Array<double, 1, 1> const& terms = dotData.second;
            for (std::size_t ii = 0; ii < terms.size(); ++ii, index += numKernels) {
                vector[index] += terms[ii];
            }

            std::size_t jOffsetIndex = 0;
            for (int jOffset = -kernelHalfWidth; jOffset <= kernelHalfWidth; ++jOffset, ++jOffsetIndex) {
                if (jOffset == 0) {
                    --jOffsetIndex;
                    continue;
                }
                FiberModel const& jModel = kernelModels[jOffsetIndex];
                auto const dotKernel = iModel.dotOther(kernel, jModel, usePixels, variance);
                std::size_t iIndex = dotKernel.first + iOffsetIndex;
                std::size_t jIndex = dotKernel.first + jOffsetIndex;
                ndarray::Array<double, 1, 1> const& terms = dotKernel.second;
                for (
                    std::size_t ii = 0;
                    ii < terms.size();
                    ++ii, iIndex += numKernels, jIndex += numKernels
                ) {
                    matrix[iIndex][jIndex] += terms[ii];
                }
            }

            DotBackgroundTerms const& dotBackground = iModel.dotBackground(kernel, bg, usePixels, variance);
            for (auto const& [kIndex, bgIndex, term] : dotBackground) {
                matrix[kIndex][bgIndex] += term;
            }
        }

        auto img = image.begin();
        auto var = variance.begin();
        std::size_t const rowBgIndex = bgStart + bg.getIndex(0, yy);
        auto const xBgBlocks = bg.getXBlocks();
        for (int xx = 0; xx < target.getWidth(); ++xx, ++img, ++var) {
            if (!usePixels[xx]) {
                continue;
            }
            std::size_t const block = rowBgIndex + xBgBlocks[xx];
            double const weight = 1.0/(*var);
            matrix[block][block] += weight;
            vector[block] += (*img)*weight;
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

    lsst::afw::image::Image<float> background(bg.getXNumBlocks(), bg.getYNumBlocks());
    std::size_t bgIndex = bgStart;
    for (std::size_t yy = 0; yy < bg.getYNumBlocks(); ++yy) {
        for (std::size_t xx = 0; xx < bg.getXNumBlocks(); ++xx, ++bgIndex) {
            background.getArray()[yy][xx] = solution[bgIndex];
        }
    }

    std::cerr << "Timing results (accumulated over all iterations, seconds):\n";
    for (auto const& [name, elapsed] : timings) {
        std::cerr << "  " << name << ": " << elapsed << "\n";
    }

    return {
        FiberKernel(
            target.getDimensions(),
            kernelHalfWidth,
            xKernelNum,
            yKernelNum,
            solution[ndarray::view(0, bgStart)]
        ),
        std::move(background)
    };
}


}}}  // namespace pfs::drp::stella
