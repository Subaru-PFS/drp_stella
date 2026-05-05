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
#include "pfs/drp/stella/math/quartiles.h"

namespace pfs {
namespace drp {
namespace stella {


template <typename T>
detail::LinearInterpolationHelper<T>::LinearInterpolationHelper(
    ndarray::Array<T, 1, 1> const& x,
    std::size_t length
) : _x(x), _length(length), _index(ndarray::allocate(length)), _weight(ndarray::allocate(length)) {
    if (x.size() < 2) {
        return;  // It's constant everywhere
    }
    std::size_t const last = x.size() - 1;

    // Ensure that x is monotonic increasing
    for (std::size_t ii = 1; ii < x.size(); ++ii) {
        if (x[ii] <= x[ii - 1]) {
            throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "x must be monotonic increasing");
        }
    }

    std::size_t index = 0;
    for (std::size_t ii = 0; ii < length; ++ii) {
        while (ii >= x[index + 1] && index < last - 1) {
            ++index;
        }
        _index[ii] = index;
        _weight[ii] = 1.0 - (ii - x[index])/(x[index + 1] - x[index]);
    }
}


template <typename T>
std::pair<
    typename detail::LinearInterpolationHelper<T>::IndexWeightPair,
    typename detail::LinearInterpolationHelper<T>::IndexWeightPair
> detail::LinearInterpolationHelper<T>::operator()(std::size_t x) const {
    if (x < 0 || x >= _length) {
        throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeError, "x is out of range");
    }
    if (_x.size() < 2) {
        return {{0, 1.0}, {0, 0.0}};  // It's constant everywhere
    }
    return {{_index[x], _weight[x]}, {_index[x] + 1, 1.0 - _weight[x]}};
}


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
        if (xStop < xStart) {
            return dummy();
        }
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
        lsst::geom::Box2I const& box,
        lsst::afw::image::MaskPixel badBitMask,
        ndarray::Array<bool, 1, 1> const& usePixels=ndarray::Array<bool, 1, 1>()
    ) {
        ndarray::Array<bool, 1, 1> use = ndarray::allocate(box.getWidth());
        ndarray::asEigenArray(use) = ndarray::asEigenArray(
            image.getMask()->getArray()[y][ndarray::view(box.getMinX(), box.getMaxX() + 1)]
        ).unaryExpr([badBitMask](lsst::afw::image::MaskPixel mm) { return (mm & badBitMask) == 0; });
        if (!usePixels.empty()) {
            assert(usePixels.size() == box.getWidth());
            ndarray::asEigenArray(use) = ndarray::asEigenArray(use) && ndarray::asEigenArray(usePixels);
        }
        return FiberModel{
            image.getImage()->getArray()[y][ndarray::view(box.getMinX(), box.getMaxX() + 1)],
            use,
            y,
            box.getMinX(),
            box.getMaxX() + 1,
            box.getWidth(),
            0
        };
    }

    static FiberModel fromImage(
        lsst::afw::image::Image<float> const& image,
        int y,
        lsst::geom::Box2I const& box
    ) {
        ndarray::Array<bool, 1, 1> use = ndarray::allocate(box.getWidth());
        use.deep() = true;
        return FiberModel{
            image.getArray()[y][ndarray::view(box.getMinX(), box.getMaxX() + 1)],
            use,
            y,
            box.getMinX(),
            box.getMaxX() + 1,
            box.getWidth(),
            0
        };
    }

    static FiberModel dummy() {
        return FiberModel{
            ndarray::Array<float const, 1, 1>(), ndarray::Array<bool const, 1, 1>(), 0, 0, 0, 0, 0
        };
    }

    FiberModel applyOffset(int newOffset, int imageWidth, bool subtract=true) const {
        assert(offset == 0);  // haven't done the math to support offsets on top of offsets
        if (newOffset == 0) {
            return *this;
        }
        if (xMax <=xMin) {
            return dummy();
        }
        int const newMin = std::max(0, xMin + std::min(0, newOffset));
        int const newMax = std::min(imageWidth, xMax + std::max(0, newOffset));  // exclusive
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
            int const shiftedMax = std::min(imageWidth, xMax + newOffset);  // exclusive
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
        if (subtract) {
            // Relative positon on the target array for zero offset
            int const newStart = xMin - newMin;
            int const newStop = xMax - newMin;
            assert(newStart >= 0 && newStop <= newWidth);
            assert(newStop - newStart == this->width);
            newValues[ndarray::view(newStart, newStop)] -= values;
            newUse[ndarray::view(newStart, newStop)] &= use;
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

#if 0
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
#endif

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
    double dotFunction(
        ndarray::Array<bool const, 1, 1> const& usePixels,
        ndarray::Array<float const, 1, 1> const& dataVariance,
        Func func
    ) const {
        assert(usePixels.size() == dataVariance.size());

        int start = std::max(xMin, 0);
        int stop = std::min(xMax, static_cast<int>(usePixels.size()));  // exclusive
        if (stop <= start) {
            return 0.0;
        }

        int const thisStart = start - xMin;
        int const thisStop = stop - xMin;  // exclusive

    //    if (y == 0) std::cerr << "dotFunction: offset=" << offset << " start=" << start << " stop=" << stop << " thisStart=" << thisStart << " thisStop=" << thisStop << std::endl;

        double result = 0.0;
        for (std::size_t xx = start, xThis = thisStart; xx < stop; ++xx, ++xThis) {
            if (!use[xThis] || !usePixels[xx]) {
                continue;
            }
            double const weight = 1.0/dataVariance[xx];
            result += func(xx, xThis, weight);
        }
        return result;
    }


    double dotSelf(
        ndarray::Array<bool const, 1, 1> const& usePixels,
        ndarray::Array<float const, 1, 1> const& dataVariance
    ) const {
        // if (y == 0) {
        //     std::cerr << "dotSelf: offset=" << offset << " ..." << std::endl;
        // }
        return dotFunction(usePixels, dataVariance, [this](
            int x, int xThis, double weight
        ) {
            // if (y == 0) {
            //     std::cerr << "dotSelf: offset=" << offset << " xx=" << x << " xThis=" << xThis << " weight=" << weight << " model=" << values[xThis] << std::endl;
            // }
            return weight*std::pow(values[xThis], 2);
        });
    }

    double dotData(
        ndarray::Array<float const, 1, 1> const& dataValues,
        ndarray::Array<bool const, 1, 1> const& usePixels,
        ndarray::Array<float const, 1, 1> const& dataVariance
    ) const {
        // if (y == 0) {
        //     std::cerr << "dotData: offset=" << offset << " ..." << std::endl;
        // }

        return dotFunction(usePixels, dataVariance, [this, &dataValues](
            int x, int xThis, double weight
        ) {
            // if (y == 0) {
            //     std::cerr << "dotData: offset=" << offset << " xx=" << x << " xThis=" << xThis << " weight=" << weight << " model=" << values[xThis] << " data=" << dataValues[x] << std::endl;
            // }
            return weight*values[xThis]*dataValues[x];
        });
    }

    double dotOther(
        FiberModel const& other,
        ndarray::Array<bool const, 1, 1> const& usePixels,
        ndarray::Array<float const, 1, 1> const& dataVariance
    ) const {
        // if (y == 0) {
        //     std::cerr << "dotOther: offset=" << offset << " ..." << std::endl;
        // }

        return dotFunction(usePixels, dataVariance, [this, &other](
            int x, int xThis, double weight
        ) {
            int const xOther = x - other.xMin;
            if (xOther < 0 || xOther >= other.width || !other.use[xOther]) {
                return 0.0;
            }
            // if (y == 0) {
            //     std::cerr << "dotOther: offset=" << offset << " xx=" << x << " xThis=" << xThis << " weight=" << weight << " model=" << values[xThis] << " other=" << other.values[xOther] << std::endl;
            // }
            return weight*values[xThis]*other.values[xOther];
        });
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


std::ostream& operator<<(std::ostream& os, FiberModel const& model) {
    os << "FiberModel(y=" << model.y << " xMin=" << model.xMin << " xMax=" << model.xMax << " offset=" << model.offset << " use=" << model.use << " values=" << model.values << ")";
    return os;
}


}  // anonymous namespace


BaseKernel::BaseKernel(
    lsst::geom::Extent2I const& dims,
    int halfWidth,
    std::size_t numParams,
    ndarray::Array<double const, 1, 1> const& values
) : _dims(dims),
    _halfWidth(halfWidth),
    _numParams(numParams),
    _values(ndarray::copy(values))
{
    utils::checkSize(values.size(), _numParams, "values");
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


namespace {


detail::LinearInterpolationHelper<double> makeInterpolationHelper(
    int numBlocks,
    std::size_t length
) {
    ndarray::Array<double, 1, 1> values = ndarray::allocate(numBlocks);

    for (std::size_t ii = 0; ii < numBlocks; ++ii) {
        values[ii] = (ii + 0.5)*length/numBlocks;
    }
    return detail::LinearInterpolationHelper<double>(values, length);
}


}  // anonymous namespace


FiberKernel::FiberKernel(
    lsst::geom::Extent2I const& dims,
    int halfWidth,
    int xNumBlocks,
    int yNumBlocks,
    ndarray::Array<double const, 1, 1> const& values
) : BaseKernel(
        dims,
        halfWidth,
        (2*halfWidth)*xNumBlocks*yNumBlocks,
        values
    ),
    _xNumBlocks(xNumBlocks),
    _yNumBlocks(yNumBlocks),
    _xInterp(makeInterpolationHelper(xNumBlocks, dims.getX())),
    _yInterp(makeInterpolationHelper(yNumBlocks, dims.getY()))
{
    utils::checkSize(
            static_cast<int>(values.size()), (2*halfWidth)*xNumBlocks*yNumBlocks, "values"
        );
}


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

    ndarray::Array<double, 1, 1> kernel = ndarray::allocate(2*_halfWidth + 1);

    int const xMax = trace.getTrace().getWidth() - 1;  // inclusive
    for (int yy = 0; yy < trace.getTrace().getHeight(); ++yy) {
        auto const& model = FiberModel::fromFiberTrace(trace, yy, 0, xMax, require);
        float const xCenter = model.centroid();
        if (!std::isfinite(xCenter)) {
             continue;
        }

        _evaluate(kernel, xCenter, yy);

        std::size_t index = 0;
        for (int offset = -_halfWidth; offset <= _halfWidth; ++offset, ++index) {
            model.applyOffset(offset, newBox.getMaxX() + 1, false).addToImage(convImage, kernel[index]);
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

    int const numKernel = 2*_halfWidth;  // number of kernel offsets (excluding zero offset)
    ndarray::Array<double, 1, 1> kernel = ndarray::allocate(numKernel + 1);
    ndarray::Array<double, 1, 1> row = ndarray::allocate(image.getWidth());

    for (int yy = 0; yy < height; ++yy) {
        for (int xx = 0; xx < image.getWidth(); ++xx) {
            _evaluate(kernel, xx, yy);

            double sum = 0.0;
            std::size_t index = 0;
            for (int offset = -_halfWidth; offset <= _halfWidth; ++offset, ++index) {
                int const xKernel = xx - offset;
                if (xKernel < 0 || xKernel >= image.getWidth()) {
                    continue;
                }
                sum += kernel[index]*image(xKernel, yy);
            }
            row[xx] = sum;
        }
        result.getArray()[yy] = row;
    }
    return result;
}


ndarray::Array<double, 1, 1> FiberKernel::evaluate(double x, double y) const {
    ndarray::Array<double, 1, 1> result = ndarray::allocate(2*_halfWidth + 1);
    _evaluate(result, x, y);
    return result;
}


template <int C>
void FiberKernel::_evaluate(
    ndarray::Array<double, 1, C> & result,
    double x,
    double y
) const {
    std::size_t const numKernel = 2*_halfWidth;
    utils::checkSize(result.size(), numKernel + 1, "result");
    result[_halfWidth] = 1.0;

    auto const xInterp = _xInterp(x);
    auto const yInterp = _yInterp(y);

    std::size_t const leftIndex = xInterp.first.first;
    std::size_t const rightIndex = xInterp.second.first;
    std::size_t const lowerIndex = yInterp.first.first;
    std::size_t const upperIndex = yInterp.second.first;
    double const leftWeight = xInterp.first.second;
    double const rightWeight = xInterp.second.second;
    double const lowerWeight = yInterp.first.second;
    double const upperWeight = yInterp.second.second;

    auto getValue = [this, numKernel](std::size_t xIndex, std::size_t yIndex, std::size_t kernelIndex) {
        return _values[(yIndex*_xNumBlocks + xIndex)*numKernel + kernelIndex];
    };

    std::size_t index = 0;
    for (int offset = -_halfWidth, pixel = 0; offset <= _halfWidth; ++pixel, ++offset, ++index) {
        if (offset == 0) {
            --index;
            continue;
        }
        double const lowerLeft = getValue(leftIndex, lowerIndex, index);
        double const lowerRight = getValue(rightIndex, lowerIndex, index);
        double const upperLeft = getValue(leftIndex, upperIndex, index);
        double const upperRight = getValue(rightIndex, upperIndex, index);

        double const value = leftWeight*lowerWeight*lowerLeft +
            rightWeight*lowerWeight*lowerRight +
            leftWeight*upperWeight*upperLeft +
            rightWeight*upperWeight*upperRight;
        result[pixel] = value;
        result[_halfWidth] -= value;
    }
    ndarray::asEigenArray(result) /= ndarray::asEigenArray(result).sum();
}


ndarray::Array<double, 3, 3> FiberKernel::makeOffsetImagesImpl(
    lsst::geom::Extent2I const& dims
) const {
    ndarray::Array<double, 3, 3> result = ndarray::allocate(2*_halfWidth + 1, dims.getY(), dims.getX());
    ndarray::Array<double, 1, 1> kernel = ndarray::allocate(2*_halfWidth + 1);

    double const dx = static_cast<double>(_dims.getX() - 1)/std::max(dims.getX() - 1, 1);
    double const dy = static_cast<double>(_dims.getY() - 1)/std::max(dims.getY() - 1, 1);
    for (int yy = 0; yy < dims.getY(); ++yy) {
        double const ySample = yy*dy;
        for (int xx = 0; xx < dims.getX(); ++xx) {
            double const xSample = xx*dx;
            _evaluate(kernel, xSample, ySample);
            result[ndarray::view()(yy)(xx)] = kernel;
        }
    }
    return result;
}


namespace {


    struct RowData {
    int y;
    ndarray::Array<bool, 1, 1> useTrace;  // trace overlaps this row?
    ndarray::Array<bool, 1, 1> usePixel;  // pixel in row should be used?
    ndarray::Array<double, 1, 1> xCenter; // center of trace in x, for kernel polynomial
    std::vector<FiberModel> models;  // model for each fiber at this row
    std::vector<std::vector<FiberModel>> kernelModels;  // kernel-offset models [fiber][offset]
    std::vector<std::vector<std::size_t>> overlaps;  // [fiber][offset] overlapping fiber indices within kernelHalfWidth

    // Layout of models: offset=-kernelWidth, ... offset=0, ... offset=+kernelWidth
    // Note that this is different from the layout of values in the matrix, which skips offset=0.
    ndarray::Array<double, 2, 2> dotData;  // [fiber][offset] model dot data

    // [fiber][otherFiber][offset][otherOffset] model dot Model; first otherFiber is the same as fiber
    std::vector<std::vector<ndarray::Array<double, 2, 2>>> dotModel;

    RowData(
        int y, std::size_t numFibers, int imageWidth, std::size_t kernelHalfWidth
    ) : y(y),
        useTrace(ndarray::allocate(numFibers)),
        usePixel(ndarray::allocate(imageWidth)),
        xCenter(ndarray::allocate(numFibers)),
        models(numFibers, FiberModel::dummy()),
        kernelModels(numFibers, std::vector<FiberModel>(2*kernelHalfWidth + 1, FiberModel::dummy())),
        overlaps(numFibers),
        dotData(ndarray::allocate(numFibers, 2*kernelHalfWidth + 1)),
        dotModel(numFibers)
    {
        useTrace.deep() = false;
        usePixel.deep() = false;
        dotData.deep() = 0.0;
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
        lsst::geom::Box2I const& box,
        ndarray::Array<int, 1, 1> const& rows
    ) : _numFibers(fiberTraces.size()),
        _numRows(rows.size()),
        _image(image),
        _fiberTraces(fiberTraces),
        _badBitMask(badBitMask),
        _kernelHalfWidth(kernelHalfWidth),
        _box(box),
        _rows(rows),
        _numParams(2*_kernelHalfWidth),
        _requireMask(image.getMask()->getPlaneBitMask(fiberMaskPlane)),
        _noData(1 << image.getMask()->addMaskPlane("NO_DATA")),
        _badFiberTrace(1 << image.getMask()->addMaskPlane("BAD_FIBERTRACE")),
        _suspect(1 << image.getMask()->addMaskPlane("SUSPECT"))
    {}

    // Determine which traces are relevant for this row
    RowData calculateRow(int y) const {
        RowData data(y, _numFibers, _image.getWidth(), _kernelHalfWidth);
        for (std::size_t ii = 0; ii < _numFibers; ++ii) {
            auto const& traceBox = _fiberTraces[ii]->getTrace().getBBox();
            if (traceBox.getMaxX() < _box.getMinX() || traceBox.getMinX() >= _box.getMaxX()) {
                continue;
            }
            int xMin = 0;
            int xMax = -1;
            auto const traceMask = _fiberTraces[ii]->getTrace().getMask()->getArray()[y];
            for (int xx = 0; xx < traceBox.getWidth(); ++xx) {
                if (traceMask[xx] & _requireMask) {
                    xMin = xx;
                    break;
                }
            }
            for (int xx = traceBox.getWidth() - 1; xx >= 0; --xx) {
                if (traceMask[xx] & _requireMask) {
                    xMax = xx;
                    break;
                }
            }
            if (xMax < xMin) {
                continue;
            }

            int const x0 = traceBox.getMinX();
            int const xLow = _box.getMinX() - x0;  // position of kernel box on trace array
            int const xHigh = _box.getMaxX() - x0;  // inclusive

            data.models[ii] = FiberModel::fromFiberTrace(
                *_fiberTraces[ii], y, std::max(xMin, xLow), std::min(xMax, xHigh), _requireMask
            );
            FiberModel & model = data.models[ii];
            data.xCenter[ii] = model.centroid();
            data.useTrace[ii] = true;

            std::size_t kernelIndex = 0;
            for (int offset = -_kernelHalfWidth; offset <= _kernelHalfWidth; ++offset, ++kernelIndex) {
                data.kernelModels[ii][kernelIndex] = model.applyOffset(offset, _image.getWidth());
            }
        }

        auto iter = _image.row_begin(y) + _box.getMinX();
        for (std::size_t xx = _box.getMinX(); xx <= _box.getMaxX(); ++xx, ++iter) {
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

        std::size_t iIndex = 0;
        for (int iOffset = -_kernelHalfWidth; iOffset <= _kernelHalfWidth; ++iOffset, ++iIndex) {
            FiberModel const& iModel = data.kernelModels[fiberIndex][iIndex];
            data.dotData[fiberIndex][iIndex] = iModel.dotData(dataImage, data.usePixel, dataVariance);
        }

        std::vector<ndarray::Array<double, 2, 2>> dotModel;
        dotModel.reserve(_numFibers - fiberIndex);
        for (std::size_t jFiberIndex = fiberIndex; jFiberIndex < _numFibers; ++jFiberIndex) {
            ndarray::Array<double, 2, 2> dotModelFiber = ndarray::allocate(_numParams + 1, _numParams + 1);
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

    // Image(x,y) = sum_i F_i(y).p_i(x,y) + sum_i sum_j a_ij.F_i(y).K_ij(x,y)
    // Where:
    // F_i(y) is the flux of fiber i at row y
    // p_i(x,y) is the fiber profile for fiber i at row y
    // K_ij(x,y) = [delta(x-offset_j) - delta(x)]*p_i(x,y) is the kernel component for fiber i, offset j
    // a_ij are the kernel parameters we are solving for
    //
    // Instead of turning Image(x,y) into a residual (by subtracting sum_j F_j(y).p_j(x,y)) and
    // recalculating the dotData values every iteration, we can remove some cross terms from the vector.
    // The model dot data terms in the vector are:
    // F_i(y).K_i(x,y) dot [Image(x,y) - sum_j F_j(y).p_j(x,y)]
    // = F_i(y).K_i(x,y) dot Image(x,y) - sum_j F_j(y).p_j(x,y) dot F_i(y).K_i(x,y)
    //
    // Model dot model terms in the matrix are:
    // F_i(y).K_i(x,y) dot F_j(y).K_j(x,y) = F_i(y).F_j(y).[K_i(x,y) dot K_j(x,y)]

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

        std::size_t iIndex = 0;
        std::size_t iKernel = 0;
        for (int iOffset = -_kernelHalfWidth; iOffset <= _kernelHalfWidth; ++iOffset, ++iIndex, ++iKernel) {
            if (iOffset == 0) {
                --iKernel;
                continue;
            }
            double const dotData = data.dotData[fiberIndex][iIndex];
            vector[iKernel] += iFlux*dotData;  // F_i(y) K_i(x,y) dot Image

            // Subtracting sum_j F_j(y).p_j(x,y) dot F_i(y).K_i(x,y) from the vector
            {
                auto _t = timer("vector_cross");
                for (std::size_t jFiberIndex : data.overlaps[fiberIndex]) {
                    double const jFlux = flux[jFiberIndex];
                    double const dotModel = getDotModel(
                        data, fiberIndex, jFiberIndex, iIndex, _kernelHalfWidth
                    );
                    vector[iKernel] -= dotModel*iFlux*jFlux;
                }
            }

            // Kernel-kernel terms for all fibers (including self terms, where jFiberIndex == fiberIndex)
            // We accumulate only the upper triangle (jKernel >= iKernel),
            // but include all ordered fiber pairs for the full normal equations.
            auto _t = timer("matrix_kernel_kernel");
            for (std::size_t jFiberIndex : data.overlaps[fiberIndex]) {
                double const jFlux = flux[jFiberIndex];

                std::size_t jIndex = iIndex;
                std::size_t jKernel = iKernel;
                for (int jOffset = iOffset; jOffset <= _kernelHalfWidth; ++jOffset, ++jIndex, ++jKernel) {
                    if (jOffset == 0) {
                        --jKernel;
                        continue;
                    }

                    double const dotKernel = getDotModel(data, fiberIndex, jFiberIndex, iIndex, jIndex);
                    // F_i(y).K_j(x,y) dot F_i(y).K_j(x,y)
                    matrix[iKernel][jKernel] += dotKernel*iFlux*jFlux;
                }
            }
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

#if 0
        lsst::afw::image::Image<double>(matrix).writeFits("matrix.fits");
        std::cerr << "Matrix:\n" << matrix << std::endl;
        std::cerr << "Vector:\n" << vector << std::endl;
#endif

        // Solve the system of equations
        auto lsq = lsst::afw::math::LeastSquares::fromNormalEquations(matrix, vector);
        lsq.setThreshold(lsqThreshold);
        ndarray::Array<double const, 1, 1> solution = lsq.getSolution();

        return ndarray::copy(solution);
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

    void accumulateFlux(
        std::size_t fiberIndex,
        std::size_t rowNum,
        RowData const& data,
        ndarray::Array<double const, 1, 1> const& kernelSolution,
        math::NonsymmetricSparseSquareMatrix & matrix,
        ndarray::Array<double, 1, 1> & vector,
        ndarray::Array<bool, 1, 1> & haveParam
    ) const {
        auto _t = timer("flux");

        if (!data.useTrace[fiberIndex]) {
            return;
        }
        std::size_t const iIndex = getFluxIndex(rowNum, fiberIndex);

        // Image(x,y) = sum_i F_i(y).[p_i(x,y) + sum_I a_I K_I(x,y)]
        // Solving for the F_i(y)
        // a_I are fixed parameters from the kernel
        // Model_i = p_i + sum_I K_I
        // model_i dot model_j = p_i*p_j + p_i*sum_J K_J + p_j*sum_I K_I + sum_I sum_J K_I*K_J
        // model_i dot data = p_i dot data + sum_I K_I dot data

        // Calculate model dot data
        ndarray::Array<double, 1, 1> const& dotData = data.dotData[fiberIndex];
        double modelDotData = 0.0;
        std::size_t offsetIndex = 0;
        std::size_t kernelIndex = 0;
        for (int offset = -_kernelHalfWidth; offset <= _kernelHalfWidth; ++offset, ++offsetIndex, ++kernelIndex) {
            if (offset == 0) {
                modelDotData += dotData[offsetIndex];  // p_i dot data
                --kernelIndex;
                continue;
            }
            modelDotData += kernelSolution[kernelIndex]*dotData[offsetIndex];  // K_I dot data
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

            std::size_t iOffsetIndex = 0;
            std::size_t iKernelIndex = 0;
            for (int iOffset = -_kernelHalfWidth; iOffset <= _kernelHalfWidth; ++iOffset, ++iOffsetIndex, ++iKernelIndex) {
                if (iOffset == 0) {
                    --iKernelIndex;
                    continue;
                }

                double const iKernel = kernelSolution[iKernelIndex];
                modelDotModel += iKernel*dotModel[iOffsetIndex][_kernelHalfWidth];  // K_i dot p_j
                modelDotModel += iKernel*dotModel[_kernelHalfWidth][iOffsetIndex];  // p_i dot K_j

                std::size_t jOffsetIndex = 0;
                std::size_t jKernelIndex = 0;
                for (
                    int jOffset = -_kernelHalfWidth;
                    jOffset <= _kernelHalfWidth;
                    ++jOffset, ++jOffsetIndex, ++jKernelIndex
                ) {
                    if (jOffset == 0) {
                        --jKernelIndex;
                        continue;
                    }
                    double const jKernel = kernelSolution[jKernelIndex];
                    modelDotModel += iKernel*jKernel*dotModel[iOffsetIndex][jOffsetIndex];  // K_i dot K_j
                }
            }
            matrix.add(iIndex, jIndex, modelDotModel);
            if (iIndex != jIndex) {
                matrix.add(jIndex, iIndex, modelDotModel);
            }
            haveParam[iIndex] |= modelDotModel != 0.0;
        }
    }

    ndarray::Array<double, 2, 2> fitFlux(
        std::vector<RowData> const& data,
        ndarray::Array<double const, 1, 1> const& kernelSolution
    ) {
        std::size_t const numFluxParams = _numFibers*_numRows;
        math::NonsymmetricSparseSquareMatrix matrix(numFluxParams);
        ndarray::Array<double, 1, 1> vector = ndarray::allocate(numFluxParams);
        vector.deep() = 0.0;
        ndarray::Array<bool, 1, 1> haveParam = ndarray::allocate(numFluxParams);
        haveParam.deep() = false;
        for (std::size_t ii = 0; ii < data.size(); ++ii) {
            for (std::size_t fiberIndex = 0; fiberIndex < _numFibers; ++fiberIndex) {
                accumulateFlux(fiberIndex, ii, data[ii], kernelSolution, matrix, vector, haveParam);
            }
        }

        auto _t = timer("solve_flux");

        // Avoid non-singular matrix from missing fibers
        for (std::size_t ii = 0; ii < numFluxParams; ++ii) {
            if (!haveParam[ii]) {
                assert(vector[ii] == 0.0);
                matrix.add(ii, ii, 1.0);
            }
        }

#if 0
        std::cerr << "Flux matrix diagonal: ";
        for (std::size_t ii = 0; ii < 3; ++ii) {
            std::cerr << matrix.get(ii, ii) << " ";
        }
        std::cerr << "..." << std::endl;
        std::cerr << "Flux vector: " << vector[ndarray::view(0, 3)] << " ..." << std::endl;
#endif

        ndarray::Array<double, 2, 2> flux = ndarray::allocate(_numRows, _numFibers);
#if 0
        using Solver = math::SymmetricSparseSquareMatrix::SimplicialLDLTSolverUpper;
        ndarray::Array<double, 1, 1> solution = matrix.solve<Solver>(vector);
#else
        using Solver = Eigen::ConjugateGradient<
            typename math::NonsymmetricSparseSquareMatrix::Matrix,
            Eigen::Upper | Eigen::Lower,
            Eigen::DiagonalPreconditioner<double>
        >;
        Solver solver;
        solver.setMaxIterations(numFluxParams*10);
        solver.setTolerance(1.0e-4);
        ndarray::Array<double, 1, 1> solution = ndarray::allocate(numFluxParams);
        matrix.solve(solution, vector, solver);
#endif

        std::size_t start = 0;
        std::size_t stop = _numFibers;
        for (std::size_t ii = 0; ii < _numRows; ++ii, start += _numFibers, stop += _numFibers) {
            flux[ii] = solution[ndarray::view(start, stop)];
        }
        return flux;
    }

    ndarray::Array<double, 1, 1> run(
        int maxIter,
        int andersonDepth,
        double andersonDamping,
        double fluxTol,
        double lsqThreshold
    ) {
        std::vector<RowData> data = calculate(_rows);

        std::size_t const numFluxParams = _numRows*_numFibers;

        ndarray::Array<double, 1, 1> kernel = utils::arrayFilled<double, 1, 1>(_numParams, 0.0);
        ndarray::Array<double, 2, 2> flux = fitFlux(data, kernel);
        std::cerr << "Initial flux: " << flux[0][ndarray::view(0, 3)] << " ..." << std::endl;

        std::vector<ndarray::Array<double, 1, 1>> kernelHistory;
        std::vector<ndarray::Array<double, 1, 1>> fluxHistory;
        kernelHistory.reserve(andersonDepth + 2);
        fluxHistory.reserve(andersonDepth + 2);

        bool converged = false;
        double delta = 0.0;
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

            // Use robust estimator for RMS, to avoid outliers. I'm especially worried about fibers
            // on the edge of the box, which may fall in and out depending on the kernel.
            ndarray::Array<bool, 1, 1> const isZero = ndarray::allocate(numFluxParams);
            ndarray::asEigenArray(isZero) = ndarray::asEigenArray(fluxResidual).array() == 0.0;
            double const rms = math::robustRms<double, 1>(fluxResidual, isZero);

            double newDelta = (ndarray::asEigenMatrix(newFluxVector) - ndarray::asEigenMatrix(fluxVector)).norm();
            double const spectralRadius = newDelta/delta;
            delta = newDelta;

            std::cerr << "Iteration " << ii << ": flux RMS change = " << rms << ", spectral radius = " << spectralRadius << std::endl;
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
                    ndarray::asEigenArray(nextFluxVector) = andersonFluxVector.array()*andersonDamping +
                        ndarray::asEigenArray(newFluxVector)*(1.0 - andersonDamping);
                }
            }

            ndarray::Array<double, 2, 1> reshapedFlux = utils::unflattenArray(
                nextFluxVector, _numRows, _numFibers
            );
            ndarray::asEigenArray(flux) = ndarray::asEigenArray(reshapedFlux);
            std::cerr << "New flux: " << flux[0][ndarray::view(0, 3)] << " ..." << std::endl;
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

        return fitKernel(data, flux, lsqThreshold);
    }

    // Inputs
    std::size_t _numFibers;
    std::size_t _numRows;
    lsst::afw::image::MaskedImage<ImageT> const& _image;
    FiberTraceSet<ImageT> const& _fiberTraces;
    lsst::afw::image::MaskPixel _badBitMask;
    int _kernelHalfWidth;
    lsst::geom::Box2I _box;
    ndarray::Array<int, 1, 1> const& _rows;

    // Helpers
    std::size_t _numParams;
    lsst::afw::image::MaskPixel _requireMask;
    MaskT _noData;
    MaskT _badFiberTrace;
    MaskT _suspect;

    // Timing accumulators; mutable so timer() can be called from const methods.
    mutable std::unordered_map<std::string, double> _timings;
};


}  // anonymous namespace


FiberKernel fitFiberKernel(
    lsst::afw::image::MaskedImage<float> const& image,
    FiberTraceSet<float> const& fiberTraces,
    lsst::afw::image::MaskPixel badBitMask,
    int kernelHalfWidth,
    int xKernelNum,
    int yKernelNum,
    ndarray::Array<int, 1, 1> const& rows,
    int maxIter,
    int andersonDepth,
    double andersonDamping,
    double fluxTol,
    double lsqThreshold
) {
    if (kernelHalfWidth <= 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, "Kernel half-width must be positive");
    }
    if (xKernelNum <= 0 || yKernelNum <= 0) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterError, "Kernel block numbers must be positive"
        );
    }
    if (maxIter <= 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, "maxIter must be positive");
    }
    if (andersonDepth < 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, "andersonDepth must be non-negative");
    }
    if (andersonDamping <= 0.0 || andersonDamping >= 1.0) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterError, "andersonDamping must be between 0 and 1"
        );
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

    std::cerr << "width=" << image.getWidth() << ", height=" << image.getHeight() << std::endl;

    ndarray::Array<double, 1, 1> values = ndarray::allocate(2*kernelHalfWidth*xKernelNum*yKernelNum);
    std::size_t start = 0;
    std::size_t stop = 2*kernelHalfWidth;
    for (std::size_t ii = 0; ii < yKernelNum; ++ii) {
        for (
            std::size_t jj = 0;
            jj < xKernelNum;
            ++jj, start += 2*kernelHalfWidth, stop += 2*kernelHalfWidth
        ) {
            int const xStart = jj * image.getWidth() / xKernelNum;
            int const xStop = (jj + 1) * image.getWidth() / xKernelNum;  // exclusive
            int const yStart = ii * image.getHeight() / yKernelNum;
            int const yStop = (ii + 1) * image.getHeight() / yKernelNum;  // exclusive

            lsst::geom::Box2I box(
                lsst::geom::Point2I(xStart, yStart),
                lsst::geom::Point2I(xStop - 1, yStop - 1)
            );
            box.clip(image.getBBox());

            std::cerr << "Fitting kernel for box " << box << std::endl;
            values[ndarray::view(start, stop)] = FiberKernelFitter(
                image, fiberTraces, badBitMask,
                kernelHalfWidth, box,
                rows.isEmpty() ? utils::arange<int>(0, image.getHeight()) : rows
            ).run(maxIter, andersonDepth, andersonDamping, fluxTol, lsqThreshold);
        }
    }
    return FiberKernel(image.getDimensions(), kernelHalfWidth, xKernelNum, yKernelNum, values);
}


namespace {


ndarray::Array<double, 1, 1> _fitFiberKernel(
    lsst::afw::image::MaskedImage<float> const& source,
    lsst::afw::image::MaskedImage<float> const& target,
    lsst::afw::image::MaskPixel badBitMask,
    int kernelHalfWidth,
    lsst::geom::Box2I const& box,
    ndarray::Array<int, 1, 1> const& rows,
    double lsqThreshold
) {
    std::cerr << "Fitting kernel for box " << box << std::endl;
    std::size_t const numParams = 2*kernelHalfWidth;

    ndarray::Array<double, 2, 2> matrix = ndarray::allocate(numParams, numParams);
    ndarray::Array<double, 1, 1> vector = ndarray::allocate(numParams);
    matrix.deep() = 0.0;
    vector.deep() = 0.0;

    ndarray::Array<bool, 1, 1> usePixels = ndarray::allocate(source.getWidth());

    std::size_t const numRows = rows.isEmpty() ? target.getHeight() : rows.size();
    for (std::size_t ii = 0; ii < numRows; ++ii) {
        int const yy = rows.isEmpty() ? ii : rows[ii];
        if (yy < box.getMinY() || yy > box.getMaxY()) {
            continue;
        }

#if 0
        if (yy != 0) {
            continue;
        }
#endif

        ndarray::asEigenArray(usePixels) = ndarray::asEigenArray(
            target.getMask()->getArray()[yy]
        ).unaryExpr([badBitMask](auto const& mask) { return (mask & badBitMask) == 0; });

        // Flag pixels outside the box, so they are not used in the fit
        bool first = false;
        for (int xx = 0; xx < source.getWidth(); ++xx) {
            if (xx < box.getMinX() || xx > box.getMaxX()) {
                usePixels[xx] = false;
            } else {
                if (usePixels[xx] && !first) {
//                    std::cerr << "Row " << yy << ": first good pixel at x=" << xx << std::endl;
                    first = true;
                }
            }
        }
//        std::cerr << "Row " << yy << ": usePixels=" << usePixels << std::endl;

        ndarray::Array<float, 1, 1> image = ndarray::copy(target.getImage()->getArray()[yy]);
        ndarray::asEigenArray(image) -= ndarray::asEigenArray(source.getImage()->getArray()[yy]);
        ndarray::Array<float, 1, 1> variance = ndarray::copy(target.getVariance()->getArray()[yy]);
        ndarray::asEigenArray(variance) += ndarray::asEigenArray(source.getVariance()->getArray()[yy]);

        std::vector<FiberModel> kernelModels;  // [offsetIndex]
        kernelModels.reserve(numParams);
        // Build source model from full image to provide full support for kernel shifts.
        // Fit pixels remain restricted to the original box via usePixels.
        FiberModel sourceModel = FiberModel::fromImage(
            source, yy, source.getBBox(), badBitMask, usePixels
        );
//        std::cerr << "Source model: " << sourceModel << std::endl;
        for (int offset = -kernelHalfWidth; offset <= kernelHalfWidth; ++offset) {
            if (offset == 0) {
                continue;
            }
            kernelModels.emplace_back(std::move(sourceModel.applyOffset(offset, source.getWidth())));
            // std::cerr << "Kernel model (offset " << offset << "): " << kernelModels.back() << std::endl;
        }

        // Model is:
        // Target(x,y) - Source(x,y) = sum_i a_i K_i(x)*Source(x,y)
        // Where K_i(x) are the kernel components (delta functions at different offsets), and
        // a_i are the parameters we are solving for.
        //
        // Model dot model terms in the matrix are:
        // K_i*Source dot K_j*Source
        //
        // Model dot data terms in the matrix are:
        // K_i*Source dot Target
        std::size_t iIndex = 0;
        for (int iOffset = -kernelHalfWidth; iOffset <= kernelHalfWidth; ++iOffset, ++iIndex) {
            if (iOffset == 0) {
                --iIndex;
                continue;
            }
            FiberModel const& iModel = kernelModels[iIndex];
            vector[iIndex] += iModel.dotData(image, usePixels, variance);
            matrix[iIndex][iIndex] += iModel.dotSelf(usePixels, variance);

            std::size_t jIndex = iIndex + 1;
            for (int jOffset = iOffset + 1; jOffset <= kernelHalfWidth; ++jOffset, ++jIndex) {
                if (jOffset == 0) {
                    --jIndex;
                    continue;
                }
                FiberModel const& jModel = kernelModels[jIndex];
                matrix[iIndex][jIndex] += iModel.dotOther(jModel, usePixels, variance);
            }
        }
    }

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

    std::cerr << "Matrix:\n" << matrix << std::endl;
    std::cerr << "Vector:\n" << vector << std::endl;

    // Solve the system of equations
    auto lsq = lsst::afw::math::LeastSquares::fromNormalEquations(matrix, vector);
    lsq.setThreshold(lsqThreshold);

    std::cerr << "Solution: " << lsq.getSolution() << std::endl;

    return ndarray::copy(lsq.getSolution());
}


}  // anonymous namespace


FiberKernel fitFiberKernel(
    lsst::afw::image::MaskedImage<float> const& source,
    lsst::afw::image::MaskedImage<float> const& target,
    lsst::afw::image::MaskPixel badBitMask,
    int kernelHalfWidth,
    int xKernelNum,
    int yKernelNum,
    ndarray::Array<int, 1, 1> const& rows,
    double lsqThreshold
) {
    if (kernelHalfWidth <= 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, "Kernel half-width must be positive");
    }
    if (xKernelNum <= 0 || yKernelNum <= 0) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterError, "Kernel block numbers must be positive"
        );
    }
    if (!(std::isfinite(lsqThreshold) && lsqThreshold > 0.0)) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterError,
            "lsqThreshold must be finite and positive"
        );
    }

    ndarray::Array<double, 1, 1> values = ndarray::allocate(2*kernelHalfWidth*xKernelNum*yKernelNum);
    std::size_t start = 0;
    std::size_t stop = 2*kernelHalfWidth;
    for (std::size_t ii = 0; ii < yKernelNum; ++ii) {
        for (
            std::size_t jj = 0;
            jj < xKernelNum;
            ++jj, start += 2*kernelHalfWidth, stop += 2*kernelHalfWidth
        ) {
            // Use integer partitioning to avoid float rounding dropping pixels
            int const xStart = jj * source.getWidth() / xKernelNum;
            int const xStop = (jj + 1) * source.getWidth() / xKernelNum;  // exclusive
            int const yStart = ii * source.getHeight() / yKernelNum;
            int const yStop = (ii + 1) * source.getHeight() / yKernelNum;  // exclusive
            if (xStop <= xStart || yStop <= yStart) {
                continue;
            }
            lsst::geom::Box2I box(
                lsst::geom::Point2I(xStart, yStart),
                lsst::geom::Point2I(xStop - 1, yStop - 1)
            );
            box.clip(source.getBBox());

            values[ndarray::view(start, stop)] = _fitFiberKernel(
                source, target, badBitMask, kernelHalfWidth, box, rows, lsqThreshold
            );
        }
    }

    return FiberKernel(
        target.getDimensions(),
        kernelHalfWidth,
        xKernelNum,
        yKernelNum,
        values
    );
}


// Explicit instantiation
#define INSTANTIATE(TYPE) \
template class detail::LinearInterpolationHelper<TYPE>;


INSTANTIATE(double);


}}}  // namespace pfs::drp::stella
