#include "ndarray.h"
#include "ndarray/eigen.h"

#include "lsst/afw/math/LeastSquares.h"

#include "pfs/drp/stella/FiberKernel.h"
#include "pfs/drp/stella/utils/checkSize.h"
#include "pfs/drp/stella/math/NormalizedPolynomial.h"
#include "pfs/drp/stella/utils/math.h"

namespace pfs {
namespace drp {
namespace stella {


namespace {


struct FiberModel {
    ndarray::Array<float const, 1, 1> values;
    ndarray::Array<bool const, 1, 1> use;
    double flux;
    int xMin;  // relative to the image, inclusive
    int xMax;  // relative to the image, exclusive
    int width;
    int offset;

    static FiberModel fromFiberTrace(
        FiberTrace<float> const& fiberTrace,
        float flux,
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
            flux,
            trace.getBBox().getMinX() + xStart,
            trace.getBBox().getMinX() + xStop + 1,
            xStop - xStart + 1,
            0
        };
    }

    static FiberModel dummy() {
        return FiberModel{
            ndarray::Array<float const, 1, 1>(), ndarray::Array<bool const, 1, 1>(), 0.0, 0, 0, 0, 0
        };
    }

    FiberModel applyOffset(int offset) const {
        assert(offset != 0);
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

        return FiberModel{std::move(newValues), std::move(newUse), flux, newMin, newMax, size, offset};
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
        ).sum()*flux;
    }

    double sum(ndarray::Array<bool const, 1, 1> const& usePixels) const {
        auto const dataGood = ndarray::asEigenArray(usePixels[ndarray::view(xMin, xMax)]);
        auto const modelGood = ndarray::asEigenArray(use);
        return (dataGood && modelGood).select(
            ndarray::asEigenArray(values).template cast<double>(), 0.0
        ).sum()*flux;
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
        ).square().sum()*std::pow(flux, 2);
    }

    double dotSelf(ndarray::Array<bool const, 1, 1> const& usePixels) const {
        auto const modelGood = ndarray::asEigenArray(use);
        auto const dataGood = ndarray::asEigenArray(usePixels[ndarray::view(xMin, xMax)]);
        return (modelGood && dataGood).select(
            ndarray::asEigenArray(values).template cast<double>(), 0.0
        ).square().sum()*std::pow(flux, 2);
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
        return weightedValues.square().sum()*std::pow(flux, 2);
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
        return (left*right).sum()*flux;
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
        return (left*right).sum()*this->flux*other.flux;
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
        ndarray::asEigenArray(bgTerms) *= flux;
        return {minBlock, bgTerms};
    }

    void addToImage(
        lsst::afw::image::Image<float> & image
    ) const {
        int const start = xMin - image.getBBox().getMinX();
        int const stop = xMax - image.getBBox().getMinX();  // exclusive
        if (stop <= start) {
            return;
        }
        auto const rhs = ndarray::asEigenArray(use).select(
            ndarray::asEigenArray(values)*flux, 0.0
        );
        ndarray::asEigenArray(image.getArray()[ndarray::view(start, stop)]) += rhs;
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
    lsst::geom::Box2I bbox = trace.getTrace().getBBox().dilatedBy(_halfWidth);

    lsst::afw::image::MaskedImage<float> convolved{bbox};
    lsst::afw::image::Image<float> convImage = *convolved.getImage();
    lsst::afw::image::Mask<lsst::afw::image::MaskPixel> convMask = *convolved.getMask();

    convImage = 0.0;
    *convolved.getVariance() = 0.0;

    for (int yy = 0; yy < trace.getTrace().getHeight(); ++yy) {
        auto const& model = FiberModel::fromFiberTrace(
            trace,
            1.0,
            yy,
            trace.getTrace().getBBox().getMinX(),
            trace.getTrace().getBBox().getMaxX() - 1,
            require
        );
        float const xCenter = model.centroid();
        model.addToImage(convImage);

        std::size_t offsetIndex = 0;
        for (int offset = -_halfWidth; offset <= _halfWidth; ++offset, ++offsetIndex) {
            if (offset == 0) {
                --offsetIndex;
                continue;
            }
            auto kernelModel = model.applyOffset(offset);
            kernelModel.flux = _polynomials[offsetIndex](xCenter, yy);
            kernelModel.addToImage(convImage);
        }
    }

    ndarray::asEigenArray(convMask.getArray()) = (
        ndarray::asEigenArray(convImage.getArray()) != 0.0
    ).select(
        Eigen::VectorXd::Zero(convMask.getArray().size()), require
    ).template cast<lsst::afw::image::MaskPixel>();

    return std::make_shared<FiberTrace<float>>(std::move(convolved), trace.getFiberId());
}


FiberTraceSet<float> FiberKernel::operator()(FiberTraceSet<float> const& trace) const {
    FiberTraceSet<float> result(trace.size());
    for (std::size_t ii = 0; ii < trace.size(); ++ii) {
        result.add(operator()(*trace[ii]));
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
    ndarray::Array<bool, 1, 1> useTrace;  // trace overlaps this row?
    ndarray::Array<bool, 1, 1> usePixel;  // pixel in row should be used?
    ndarray::Array<double, 1, 1> xCenter; // center of trace in x, for kernel polynomial
    std::vector<std::vector<FiberModel>> kernelModels;  // kernel-offset models [fiber][offset]

    RowData(std::size_t numFibers, int imageWidth, std::size_t numKernels)
      : useTrace(ndarray::allocate(numFibers)),
        usePixel(ndarray::allocate(imageWidth)),
        xCenter(ndarray::allocate(numFibers)),
        kernelModels(numFibers, std::vector<FiberModel>(numKernels, FiberModel::dummy()))
    {
        useTrace.deep() = false;
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
        int yBackgroundSize
    ) : _numFibers(fiberTraces.size()),
        _numRows(image.getHeight()),
        _image(image),
        _fiberTraces(fiberTraces),
        _badBitMask(badBitMask),
        _kernelHalfWidth(kernelHalfWidth),
        _kernelOrder(kernelOrder),
        _xBackgroundSize(xBackgroundSize),
        _yBackgroundSize(yBackgroundSize),
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
    {
    }

    // Determine which traces are relevant for this row
    RowData checkRow(int y, SpectrumSet const& spectra) {
        RowData data(_numFibers, _image.getWidth(), _numKernels);

        std::vector<FiberModel> models(_numFibers, FiberModel::dummy());
        for (std::size_t ii = 0; ii < _numFibers; ++ii) {
            auto const& box = _fiberTraces[ii]->getTrace().getBBox();
            auto const& spectrum = *spectra[ii];
            assert(spectrum.getFiberId() == _fiberTraces[ii]->getFiberId());

            float const flux = spectrum.getFlux()[y];
            if ((spectrum.getMask()(y, 0) & (_badFiberTrace | _noData)) || !std::isfinite(flux)) {
                continue;
            }

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

            models[ii] = FiberModel::fromFiberTrace(*_fiberTraces[ii], flux, y, xMin, xMax, _requireMask);
            FiberModel & model = models[ii];
            data.xCenter[ii] = model.centroid();
            data.useTrace[ii] = true;

            std::size_t kernelIndex = 0;
            for (int offset = -_kernelHalfWidth; offset <= _kernelHalfWidth; ++offset, ++kernelIndex) {
                if (offset == 0) {
                    --kernelIndex;
                    continue;
                }
                data.kernelModels[ii][kernelIndex] = model.applyOffset(offset);
            }
        }

        auto iter = _image.row_begin(y);
        for (int xx = 0; xx < _image.getWidth(); ++xx, ++iter) {
            data.usePixel[xx] = isGoodImage(iter);
        }

        return data;
    }

    // Layout of parameters:
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

    bool isGoodImage(ImageT value, lsst::afw::image::MaskPixel mask, VarianceT variance) const {
        return (mask & _badBitMask) == 0 && std::isfinite(value) && std::isfinite(variance) && variance > 0;
    }
    bool isGoodImage(auto & iter) const {
        return isGoodImage(iter.image(), iter.mask(), iter.variance());
    }

    void accumulateFiber(
        int y,
        std::size_t fiberIndex,
        RowData const& data,
        ndarray::Array<double, 2, 2>& matrix,
        ndarray::Array<double, 1, 1>& vector
    ) {
        if (!data.useTrace[fiberIndex]) {
            return;
        }

        auto const& dataImage = _image.getImage()->getArray()[y];
        auto const& dataMask = _image.getMask()->getArray()[y];
        // Deliberately ignoring the variance out of concern for flux biases.

        auto polyValues = _kernelPolynomial.getDFuncDParameters(data.xCenter[fiberIndex], y);
        std::size_t iOffsetIndex = 0;
        for (int iOffset = -_kernelHalfWidth; iOffset <= _kernelHalfWidth; ++iOffset, ++iOffsetIndex) {
            if (iOffset == 0) {
                --iOffsetIndex;
                continue;
            }

            FiberModel const& iKernelModel = data.kernelModels[fiberIndex][iOffsetIndex];
            double const kernelDotSelf = iKernelModel.dotSelf(data.usePixel);  // diagonal term
            double const kernelDotData = iKernelModel.dotData(dataImage, data.usePixel);  // vector term

            auto const bgResult = iKernelModel.dotBackground(_bg.xBlocks, data.usePixel);
            auto const& bgTerms = bgResult.second;

            for (std::size_t iSpatial = 0; iSpatial < _numKernelSpatial; ++iSpatial) {
                std::size_t const iKernelIndex = getKernelIndex(iOffset, iSpatial);
                double const iPoly = polyValues[iSpatial];
                // Kernel diagonal term
                matrix[iKernelIndex][iKernelIndex] += std::pow(iPoly, 2)*kernelDotSelf;
                vector[iKernelIndex] += iPoly*kernelDotData;

                // Kernel-spatial cross terms
                for (std::size_t jSpatial = iSpatial + 1; jSpatial < _numKernelSpatial; ++jSpatial) {
                    std::size_t const jKernelIndex = getKernelIndex(iOffset, jSpatial);
                    double const jPoly = polyValues[jSpatial];
                    matrix[iKernelIndex][jKernelIndex] += iPoly*jPoly*kernelDotSelf;
                }

                // Kernel-kernel cross terms
                std::size_t jOffsetIndex = iOffsetIndex + 1;
                for (int jOffset = iOffset + 1; jOffset <= _kernelHalfWidth; ++jOffset, ++jOffsetIndex) {
                    if (jOffset == 0) {
                        --jOffsetIndex;
                        continue;
                    }
                    FiberModel const& jKernelModel = data.kernelModels[fiberIndex][jOffsetIndex];
                    double const kernelDotKernel = iKernelModel.dotOther(jKernelModel, data.usePixel);
                    for (std::size_t jSpatial = iSpatial + 1; jSpatial < _numKernelSpatial; ++jSpatial) {
                        std::size_t const jKernelIndex = getKernelIndex(jOffset, jSpatial);
                        double const jPoly = polyValues[jSpatial];
                        matrix[iKernelIndex][jKernelIndex] += iPoly*jPoly*kernelDotKernel;
                    }
                }

                // Kernel-background cross-terms
                std::size_t bgIndex = getBackgroundIndex(0, y) + bgResult.first;
                for (std::size_t ii = 0; ii < bgTerms.size(); ++ii, ++bgIndex) {
                    matrix[iKernelIndex][bgIndex] += iPoly*bgTerms[ii];
                }
            }
        }
    }

    // Accumulate the background diagonal terms
    void accumulateBackground(
        int y,
        RowData const& data,
        ndarray::Array<double, 2, 2>& matrix,
        ndarray::Array<double, 1, 1>& vector
    ) {
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

    ndarray::Array<double const, 1, 1> solve(
        ndarray::Array<double, 2, 2>& matrix,
        ndarray::Array<double, 1, 1>& vector
    ) const {
        // Ensure we have entries for all parameters, to avoid singular matrix
        for (std::size_t ii = 0; ii < _numParams; ++ii) {
            if (matrix[ii][ii] == 0.0) {
                assert(vector[ii] == 0.0);
                 matrix[ii][ii] = 1.0;
            }
        }

        // Solve the system of equations
        auto lsq = lsst::afw::math::LeastSquares::fromNormalEquations(matrix, vector);
        /// XXX set threshold, etc.
        return lsq.getSolution();
    }

    std::tuple<FiberKernel, lsst::afw::image::Image<double>> extract(
        ndarray::Array<double const, 1, 1> const& solution
    ) {
        // Extract the kernel
        FiberKernel kernel(
            lsst::geom::Box2D(_image.getBBox()),
            _kernelHalfWidth,
            _kernelOrder,
            solution[ndarray::view(0, _bgStart)]
        );

        std::cerr << "Kernel parameters:" << solution[ndarray::view(0, _numKernels*_numKernelSpatial)] << std::endl;

        // Extract the background
        lsst::afw::image::Image<double> background(_bg.xNumBlocks, _bg.yNumBlocks);
        std::size_t bgIndex = _bgStart;
        for (std::size_t yy = 0; yy < _bg.yNumBlocks; ++yy, bgIndex += _bg.xNumBlocks) {
            background.getArray()[yy] = solution[ndarray::view(bgIndex, bgIndex + _bg.xNumBlocks)];
        }

        return {std::move(kernel), std::move(background)};
    }

    std::tuple<FiberKernel, lsst::afw::image::Image<double>> run(SpectrumSet const& spectra) {
        ndarray::Array<double, 2, 2> matrix = ndarray::allocate(_numParams, _numParams);
        ndarray::Array<double, 1, 1> vector = ndarray::allocate(_numParams);
        matrix.deep() = 0.0;
        vector.deep() = 0.0;

        for (int yy = 0; yy < _numRows; ++yy) {
            if (yy % 100 == 0) {
                std::cerr << "Processing row " << yy << "/" << _numRows << std::endl;
            }
            auto const data = checkRow(yy, spectra);
            for (std::size_t ii = 0; ii < _numFibers; ++ii) {
                accumulateFiber(yy, ii, data, matrix, vector);
            }
            accumulateBackground(yy, data, matrix, vector);
        }
        auto const solution = solve(matrix, vector);
        return extract(solution);
    }

    // Inputs
    std::size_t _numFibers;
    int _numRows;
    lsst::afw::image::MaskedImage<ImageT> const& _image;
    FiberTraceSet<ImageT> const& _fiberTraces;
    lsst::afw::image::MaskPixel _badBitMask;
    int _kernelHalfWidth;
    int _kernelOrder;
    int _xBackgroundSize;
    int _yBackgroundSize;

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


std::tuple<FiberKernel, lsst::afw::image::Image<double>> fitFiberKernel(
    lsst::afw::image::MaskedImage<float> const& image,
    FiberTraceSet<float> const& fiberTraces,
    SpectrumSet const& spectra,
    lsst::afw::image::MaskPixel badBitMask,
    int kernelHalfWidth,
    int kernelOrder,
    int xBackgroundSize,
    int yBackgroundSize
) {
    return KernelFitter(
        image, fiberTraces,badBitMask,
        kernelHalfWidth, kernelOrder,
        xBackgroundSize, yBackgroundSize
    ).run(spectra);
}


}}}  // namespace pfs::drp::stella