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

    static FiberModel dummy() {
        return FiberModel{
            ndarray::Array<float const, 1, 1>(), ndarray::Array<bool const, 1, 1>(), 0, 0, 0, 0, 0
        };
    }

    FiberModel applyOffset(int offset) const {
        if (offset == 0) {
            return *this;
        }
        int const size = width + std::abs(offset);
        ndarray::Array<float, 1, 1> newValues = ndarray::allocate(size);
        ndarray::Array<bool, 1, 1> newUse = ndarray::allocate(size);
        newValues.deep() = 0.0;
        newUse.deep() = false;

        // Original model: 0.1, 0.8, 0.1
        // offset = -2 --> 0.1, 0.8, 0.0, -0.8, -0.1
        // offset = -1 --> 0.1, 0.7, -0.7, -0.1
        // offset = +1 --> -0.1, -0.7, 0.7, 0.1
        // offset = +2 --> -0.1, -0.8, 0.0, 0.8, 0.1

        int const newMin = xMin + std::min(0, offset);
        int const newMax = xMax + std::max(0, offset);

        // Insert the offset values
        {
            int const start = offset < 0 ? 0 : offset;
            int const stop = std::min(start + width, size);  // exclusive
            assert(start >= 0);
            assert(stop <= size);
            newValues[ndarray::view(start, stop)] = values;
            newUse[ndarray::view(start, stop)] = use;
        }

        // Subtract the non-offset values
        {
            int const start = offset > 0 ? 0 : -offset;
            int const stop = std::min(start + width, size);
            assert(start >= 0);
            assert(stop <= size);
            newValues[ndarray::view(start, stop)] -= values;
            newUse[ndarray::view(start, stop)] |= use;
        }

        return FiberModel{std::move(newValues), std::move(newUse), y, newMin, newMax, size, offset};
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

    double dotSelfWeighted(
        ndarray::Array<bool const, 1, 1> const& usePixels,
        ndarray::Array<float const, 1, 1> const& dataVariance
    ) const {
        auto const modelGood = ndarray::asEigenArray(use);
        auto const dataGood = ndarray::asEigenArray(usePixels[ndarray::view(xMin, xMax)]);
        auto const weights = (modelGood && dataGood).select(
            1.0/ndarray::asEigenArray(dataVariance[ndarray::view(xMin, xMax)]), 0.0
        ).template cast<double>();
        auto const weightedValues = weights*ndarray::asEigenArray(values).template cast<double>();
        return weightedValues.square().sum();
    }

    double dotData(
        ndarray::Array<float const, 1, 1> const& dataValues,
        ndarray::Array<bool const, 1, 1> const& usePixels
    ) const {
        auto const left = ndarray::asEigenArray(use).select(
            ndarray::asEigenArray(values).template cast<double>(), 0.0
        );
        auto const right = ndarray::asEigenArray(usePixels[ndarray::view(xMin, xMax)]).select(
            ndarray::asEigenArray(dataValues[ndarray::view(xMin, xMax)]).template cast<double>(), 0.0
        );
        return (left*right).sum();
    }

    double dotOther(
        FiberModel const& other,
        ndarray::Array<bool const, 1, 1> const& usePixels
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
        return (left*right).sum();
    }

    std::pair<int, ndarray::Array<double, 1, 1>> dotBackground(
        ndarray::Array<int const, 1, 1> const& blocks,
        ndarray::Array<bool const, 1, 1> const& usePixels
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
            bgTerms[blockIndex - minBlock] += values[xModel];
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


FiberKernel::FiberKernel(
    lsst::geom::Box2D const& range,
    int halfWidth,
    int order,
    ndarray::ArrayRef<double const, 1, 1> const& coefficients
) : _halfWidth(halfWidth),
    _order(order),
    _numPoly(Polynomial::nParametersFromOrder(order)),
    _numParams((2*halfWidth)*_numPoly),
    _coefficients(ndarray::copy(coefficients))
{
    utils::checkSize(coefficients.size(), _numParams, "coefficients");

    _polynomials.reserve(2*halfWidth);
    std::size_t offsetIndex = 0;
    std::size_t start = 0;
    std::size_t stop = _numPoly;
    for (int offset = -halfWidth; offset <= halfWidth; ++offset, ++offsetIndex) {
        if (offset == 0) {
            --offsetIndex;
            continue;
        }
        ndarray::Array<double, 1, 1> coeffs = ndarray::copy(coefficients[ndarray::view(start, stop)]);
        _polynomials.emplace_back(coeffs, range);
        start = stop;
        stop += _numPoly;
    }
}


std::shared_ptr<FiberTrace<float>> FiberKernel::operator()(FiberTrace<float> const& trace) const {
    auto const require = trace.getTrace().getMask()->getPlaneBitMask(fiberMaskPlane);
    lsst::geom::Box2I bbox = trace.getTrace().getBBox().dilatedBy(lsst::geom::Extent2I(_halfWidth, 0));

    lsst::afw::image::MaskedImage<float> convolved{bbox};
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
            auto kernelModel = model.applyOffset(offset);
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


FiberTraceSet<float> FiberKernel::operator()(FiberTraceSet<float> const& trace) const {
    FiberTraceSet<float> result(trace.size());
    for (std::size_t ii = 0; ii < trace.size(); ++ii) {
        result.add(operator()(*trace[ii]));
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


struct KernelFitter {
    using ImageT = float;
    using MaskT = lsst::afw::image::MaskPixel;
    using VarianceT = float;

    KernelFitter(
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
                data.kernelModels[ii][kernelIndex] = model.applyOffset(offset);
            }
        }

        auto iter = _image.row_begin(y);
        for (int xx = 0; xx < _image.getWidth(); ++xx, ++iter) {
            data.usePixel[xx] = isGoodImage(iter);
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

    bool isGoodImage(ImageT value, lsst::afw::image::MaskPixel mask, VarianceT variance) const {
        return (mask & _badBitMask) == 0 && std::isfinite(value) && std::isfinite(variance) && variance > 0;
    }
    bool isGoodImage(auto & iter) const {
        return isGoodImage(iter.image(), iter.mask(), iter.variance());
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
        auto const& dataMask = _image.getMask()->getArray()[y];
        // Deliberately ignoring the variance out of concern for flux biases.

        data.polyValues[fiberIndex].deep() = utils::vectorToArray(_kernelPolynomial.getDFuncDParameters(
            data.xCenter[fiberIndex], y
        ));

        std::size_t iIndex = 0;
        for (int iOffset = -_kernelHalfWidth; iOffset <= _kernelHalfWidth; ++iOffset, ++iIndex) {
            FiberModel const& iModel = data.kernelModels[fiberIndex][iIndex];
            data.dotData[fiberIndex][iIndex] = iModel.dotData(dataImage, data.usePixel);
            data.dotBackground[fiberIndex].emplace_back(iModel.dotBackground(_bg.xBlocks, data.usePixel));
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
                    dotModelFiber[iIndex][jIndex] = iModel.dotOther(jModel, data.usePixel);
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
        double const iFlux2 = iFlux*iFlux;
        ndarray::ArrayRef<double const, 1, 1> const polyValues = data.polyValues[fiberIndex];

        std::size_t iOffsetIndex = 0;
        for (int iOffset = -_kernelHalfWidth; iOffset <= _kernelHalfWidth; ++iOffset, ++iOffsetIndex) {
            std::size_t const bgStart = data.dotBackground[fiberIndex][iOffsetIndex].first;
            auto const& bgTerms = data.dotBackground[fiberIndex][iOffsetIndex].second;
            if (iOffset == 0) {
                // Subtracting sum_i F_i(y).p_i(x,y) dot B_b(x,y) from the vector
                assert(iOffsetIndex == std::size_t(_kernelHalfWidth));
                std::size_t bgIndex = getBackgroundIndex(0, yy) + bgStart;
                for (std::size_t ii = 0; ii < bgTerms.size(); ++ii, ++bgIndex) {
                    vector[bgIndex] -= bgTerms[ii]*iFlux;
                }
                continue;
            }

            double const dotSelf = data.dotModel[fiberIndex][0][iOffsetIndex][iOffsetIndex]*iFlux2;
            double const dotData = data.dotData[fiberIndex][iOffsetIndex]*iFlux;

            for (std::size_t iSpatial = 0; iSpatial < _numKernelSpatial; ++iSpatial) {
                std::size_t const iKernelIndex = getKernelIndex(iOffset, iSpatial);
                double const iPoly = polyValues[iSpatial];
                // Kernel diagonal term
                // F_i(y)^2.[K_j(x,y) dot K_j(x,y)]
                matrix[iKernelIndex][iKernelIndex] += std::pow(iPoly, 2)*dotSelf;
                // Vector term from the image
                // F_i(y).p_i(x,y) dot K_j(x,y)
                vector[iKernelIndex] += iPoly*dotData;

                // Subtracting sum_k F_k(y).p_k(x,y) dot F_i(y).K_j(x,y) from the vector
                std::size_t jFiberIndex = fiberIndex;
                for (std::size_t jj = 0; jj < data.dotModel[fiberIndex].size(); ++jj, ++jFiberIndex) {
                    if (!data.useTrace[jFiberIndex]) {
                        continue;
                    }
                    double const jFlux = flux[jFiberIndex];
                    auto const& dotModel = data.dotModel[fiberIndex][jj];
                    vector[iKernelIndex] -= iPoly*dotModel[iOffsetIndex][_kernelHalfWidth]*iFlux*jFlux;
                }

                // Kernel-spatial cross terms:
                // F_i(y).F_m(y).[K_j(x,y) dot K_n(x,y)]
                // where K_j and K_n have the same offset but different spatial polynomial
                for (std::size_t jSpatial = iSpatial + 1; jSpatial < _numKernelSpatial; ++jSpatial) {
                    std::size_t const jKernelIndex = getKernelIndex(iOffset, jSpatial);
                    double const jPoly = polyValues[jSpatial];
                    double const term = iPoly*jPoly*dotSelf;
                    matrix[iKernelIndex][jKernelIndex] += term;
                }

                // Kernel-kernel cross terms
                // F_i(y).F_m(y).[K_j(x,y) dot K_n(x,y)]
                // where K_j and K_n have different offsets
                std::size_t jOffsetIndex = iOffsetIndex + 1;
                for (
                    int jOffset = iOffset + 1;
                    jOffset <= _kernelHalfWidth;
                    ++jOffset, ++jOffsetIndex
                ) {
                    if (jOffset == 0) {
                        continue;
                    }
                    double const dotKernel = data.dotModel[fiberIndex][0][iOffsetIndex][jOffsetIndex]*iFlux2;
                    for (std::size_t jSpatial = 0; jSpatial < _numKernelSpatial; ++jSpatial) {
                        std::size_t const jKernelIndex = getKernelIndex(jOffset, jSpatial);
                        double const jPoly = polyValues[jSpatial];
                        double const term = iPoly*jPoly*dotKernel;
                        matrix[iKernelIndex][jKernelIndex] += term;
                    }
                }

                // Kernel-background cross-terms
                // F_i(y).[K_j(x,y) dot B_b(x,y)]
                std::size_t bgIndex = getBackgroundIndex(0, yy) + bgStart;
                for (std::size_t ii = 0; ii < bgTerms.size(); ++ii, ++bgIndex) {
                    matrix[iKernelIndex][bgIndex] += iFlux*iPoly*bgTerms[ii];
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
        int const y = data.y;
        auto iter = _image.row_begin(y);
        ndarray::Array<double, 1, 1> terms = ndarray::allocate(_bg.xNumBlocks);
        terms.deep() = 0.0;
        for (int xx = 0; xx < _image.getWidth(); ++xx, ++iter) {
            if (!data.usePixel[xx]) {
                continue;
            }

            std::size_t const block = _bg.xBlocks[xx];
            terms[block] += 1.0;

            std::size_t const bgIndex = getBackgroundIndex(xx, y);
            vector[bgIndex] += iter.image();
        }
        for (std::size_t ii = 0, bgIndex = getBackgroundIndex(0, y); ii < _bg.xNumBlocks; ++ii, ++bgIndex) {
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
        ndarray::Array<double, 1, 1> const& vector
    ) const {
        // Ensure we have entries for all parameters, to avoid singular matrix
        // Fill in the lower triangle of the matrix
        for (std::size_t ii = 0; ii < _numParams; ++ii) {
            if (matrix[ii][ii] == 0.0) {
                assert(vector[ii] == 0.0);
                 matrix[ii][ii] = 1.0;
            }
            for (std::size_t jj = ii + 1; jj < _numParams; ++jj) {
                matrix[jj][ii] = matrix[ii][jj];
            }
        }

        // Solve the system of equations
        auto lsq = lsst::afw::math::LeastSquares::fromNormalEquations(matrix, vector);
        /// XXX set threshold, etc.
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
        std::cerr << "Calculated " << result.size() << " rows" << std::endl;
        return result;
    }

    ndarray::Array<double, 1, 1> fitKernel(
        std::vector<RowData> const& data,
        ndarray::Array<double, 2, 2> const& flux
    ) {
        auto const equation = accumulate(data, flux);
        return solve(equation.first, equation.second);
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
                for (int jOffset = -_kernelHalfWidth; jOffset <= _kernelHalfWidth; ++jOffset, ++jOffsetIndex) {
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

    std::tuple<FiberKernel, lsst::afw::image::Image<float>, ndarray::Array<double, 2, 2>> run() {
        std::vector<RowData> data = calculate(_rows);

        int const maxIter = 10;

        ndarray::Array<double, 1, 1> kernel = utils::arrayFilled<double, 1, 1>(_numParams, 0.0);
        ndarray::Array<double, 2, 2> flux = fitFlux(data, kernel);
        std::cerr << "Initial flux: " << flux[0] << std::endl;
        for (int ii = 0; ii < maxIter; ++ii) {
            kernel = fitKernel(data, flux);
            std::cerr << "Kernel solution: " << kernel << std::endl;
            ndarray::Array<double, 2, 2> newFlux = fitFlux(data, kernel);
            std::cerr << "New flux: " << newFlux[0] << std::endl;

            auto const diff = ndarray::asEigenArray(newFlux) - ndarray::asEigenArray(flux);
            double const rms = std::sqrt((diff*diff).mean());
            std::cerr << "Iteration " << ii << ": flux RMS change = " << rms << std::endl;
            flux = std::move(newFlux);
        }
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
    ndarray::Array<int, 1, 1> const& rows
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
    std::cerr << "Fitting " << fiberTraces.size() << " fibers with kernel half-width " << kernelHalfWidth
              << ", kernel order " << kernelOrder
              << ", background block size (" << xBackgroundSize << "," << yBackgroundSize << ")"
              << " and " << rows.size() << " rows" << std::endl;
    return KernelFitter(
        image, fiberTraces,badBitMask,
        kernelHalfWidth, kernelOrder,
        xBackgroundSize, yBackgroundSize,
        rows.isEmpty() ? utils::arange<int>(0, image.getHeight()) : rows
    ).run();
}


}}}  // namespace pfs::drp::stella