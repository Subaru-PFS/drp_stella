#include "ndarray.h"
#include "ndarray/eigen.h"

#include "pfs/drp/stella/extractSpectra.h"
#include "pfs/drp/stella/math/NormalizedPolynomial.h"
#include "pfs/drp/stella/math/SparseSquareMatrix.h"
#include "pfs/drp/stella/utils/math.h"

namespace pfs {
namespace drp {
namespace stella {


namespace {


struct FiberModel {
    ndarray::Array<float const, 1, 1> values;
    ndarray::Array<bool const, 1, 1> use;
    double norm;
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
            1.0,
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

        return FiberModel{std::move(newValues), std::move(newUse), norm, newMin, newMax, size, offset};
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
        ).sum()*norm;
    }

    double sum(ndarray::Array<bool const, 1, 1> const& usePixels) const {
        auto const dataGood = ndarray::asEigenArray(usePixels[ndarray::view(xMin, xMax)]);
        auto const modelGood = ndarray::asEigenArray(use);
        return (dataGood && modelGood).select(
            ndarray::asEigenArray(values).template cast<double>(), 0.0
        ).sum()*norm;
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
        ).square().sum()*std::pow(norm, 2);
    }

    double dotSelf(ndarray::Array<bool const, 1, 1> const& usePixels) const {
        auto const modelGood = ndarray::asEigenArray(use);
        auto const dataGood = ndarray::asEigenArray(usePixels[ndarray::view(xMin, xMax)]);
        return (modelGood && dataGood).select(
            ndarray::asEigenArray(values).template cast<double>(), 0.0
        ).square().sum()*std::pow(norm, 2);
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
        return weightedValues.square().sum()*std::pow(norm, 2);
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
        return (left*right).sum()*norm;
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
        return (left*right).sum()*this->norm*other.norm;
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
        ndarray::asEigenArray(bgTerms) *= norm;
        return {minBlock, bgTerms};
    }
};


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


struct SpectrumExtractor {
    using ImageT = float;
    using MaskT = lsst::afw::image::MaskPixel;
    using VarianceT = float;

    SpectrumExtractor(
        lsst::afw::image::MaskedImage<ImageT> const& image,
        FiberTraceSet<ImageT> const& fiberTraces,
        lsst::afw::image::MaskPixel badBitMask,
        int kernelHalfWidth,
        int kernelOrder,
        int xBackgroundSize,
        int yBackgroundSize,
        float minFracMask,
        float minFracImage
    ) : _numFibers(fiberTraces.size()),
        _numRows(image.getHeight()),
        _image(image),
        _fiberTraces(fiberTraces),
        _badBitMask(badBitMask),
        _kernelHalfWidth(kernelHalfWidth),
        _kernelOrder(kernelOrder),
        _xBackgroundSize(xBackgroundSize),
        _yBackgroundSize(yBackgroundSize),
        _minFracMask(minFracMask),
        _minFracImage(minFracImage),
        _kernelPolynomial(kernelOrder, lsst::geom::Box2D(image.getBBox())),
        _kernelStart(_numFibers*_numRows),
        _numKernels(2*_kernelHalfWidth),
        _numKernelSpatial(_kernelPolynomial.getNParameters()),
        _kernelTerms(ndarray::allocate(_numKernels*_numKernelSpatial, _numKernels*_numKernelSpatial)),
        _bg(image.getDimensions(), xBackgroundSize, yBackgroundSize),
        _bgStart(_kernelStart + _numKernels*_numKernelSpatial),
        _requireMask(image.getMask()->getPlaneBitMask(fiberMaskPlane)),
        _noData(1 << image.getMask()->addMaskPlane("NO_DATA")),
        _badFiberTrace(1 << image.getMask()->addMaskPlane("BAD_FIBERTRACE")),
        _suspect(1 << image.getMask()->addMaskPlane("SUSPECT")),
        _fiberNorm(ndarray::allocate(_numFibers, _numRows)),
        _useTrace(ndarray::allocate(_numFibers)),
        _usePixel(ndarray::allocate(image.getWidth())),
        _xMin(ndarray::allocate(_numFibers)),
        _xMax(ndarray::allocate(_numFibers)),
        _xCenter(ndarray::allocate(_numFibers)),
        _maskResult(ndarray::allocate(_numFibers, _numRows)),
        _maskBadResult(ndarray::allocate(_numFibers, _numRows)),
        _numParams(_bgStart + _bg.xNumBlocks*_bg.yNumBlocks),
        _matrix(_numParams),
        _haveMatrixEntry(ndarray::allocate(_numParams)),
        _vector(_numParams),
        _diagonalWeighted(_numParams)
    {
        _fiberNorm.deep() = 0.0;
        _models.reserve(_numFibers);

        _kernelModels.reserve(_numFibers);
        for (std::size_t ii = 0; ii < _numFibers; ++ii) {
            _kernelModels.emplace_back();
            auto & kk = _kernelModels.back();
            for (std::size_t jj = 0; jj < _numKernels; ++jj) {
                kk.emplace_back(FiberModel::dummy());
            }
        }

        _haveMatrixEntry.deep() = false;
        _vector.deep() = 0.0;
        _diagonalWeighted.deep() = 0.0;

        _kernelTerms.deep() = 0.0;
    }

    // Determine which traces are relevant for this row
    // Calculate the normalization for each trace
    // Set the mask values for success and failure cases
    void checkRow(int y) {
        _models.clear();
        for (std::size_t ii = 0; ii < _numFibers; ++ii) {
            auto const& box = _fiberTraces[ii]->getTrace().getBBox();
            _useTrace[ii] = (y >= box.getMinY() && y <= box.getMaxY());

            _maskResult[ii][y] = 0;  // used in case of success
            _maskBadResult[ii][y] = _noData;  // used in case of failure

            _xMin[ii] = 0;
            _xMax[ii] = -1;
            _fiberNorm[ii][y] = 0.0;
            _models.emplace_back(FiberModel::dummy());

            if (!_useTrace[ii]) {
                _maskBadResult[ii][y] |= _badFiberTrace;
                continue;
            }

            auto const traceMask = _fiberTraces[ii]->getTrace().getMask()->getArray()[y];
            for (int xx = 0; xx < box.getWidth(); ++xx) {
                if (traceMask[xx] & _requireMask) {
                    _xMin[ii] = xx;
                    break;
                }
            }
            for (int xx = box.getWidth() - 1; xx >= 0; --xx) {
                if (traceMask[xx] & _requireMask) {
                    _xMax[ii] = xx;
                    break;
                }
            }
            if (_xMax[ii] < _xMin[ii]) {
                _useTrace[ii] = false;
                _maskBadResult[ii][y] |= _badFiberTrace;
                continue;
            }

            _models.back() = std::move(
                 FiberModel::fromFiberTrace(*_fiberTraces[ii], y, _xMin[ii], _xMax[ii], _requireMask)
            );

            FiberModel & model = _models.back();
            _fiberNorm[ii][y] = model.sum();
            model.norm = 1.0/_fiberNorm[ii][y];
            if (_fiberNorm[ii][y] == 0 || !std::isfinite(_fiberNorm[ii][y])) {
                _useTrace[ii] = false;
                _maskBadResult[ii][y] |= _badFiberTrace;
                continue;
            }

            _xCenter[ii] = model.centroid();
            std::size_t kernelIndex = 0;
            for (int offset = -_kernelHalfWidth; offset <= _kernelHalfWidth; ++offset, ++kernelIndex) {
                if (offset == 0) {
                    --kernelIndex;
                    continue;
                }
                _kernelModels[ii][kernelIndex] = std::move(model.applyOffset(offset));
            }
        }

        auto iter = _image.row_begin(y);
        for (int xx = 0; xx < _image.getWidth(); ++xx, ++iter) {
            _usePixel[xx] = isGoodImage(iter);
        }
    }

    // Layout of parameters:
    // Spectrum fluxes (zero-offset kernel): row (0.._numRows-1) runs fast, fiber (0.._numFibers-1) runs slow
    // Kernel parameters: offset (...,-2,-1,1,2,... NOTE: no zero!) runs slow, spatial polynomial term
    //     (0.._numKernelSpatial-1) runs fast; starts at _kernelStart
    // Background parameters: x block (0.._bg.xNumBlocks-1) runs fast, y block (0.._bg.yNumBlocks-1) runs
    //     slow; starts at _bgStart

    std::size_t getSpectrumIndex(std::size_t fiber, int yy) const {
        return yy*_numFibers + fiber;
    }

    std::size_t getKernelIndex(int offset, int spatialTerm) const {
        int const offsetIndex = offset + _kernelHalfWidth - (offset < 0 ? 0 : 1);
        return _kernelStart + offsetIndex * _numKernelSpatial + spatialTerm;
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

    // For this fiber, we calculate:
    // * the diagonal value of the matrix for this fiber (i.e. model^2)
    // * the vector value for this fiber (i.e., model dot data)
    // * the cross-terms with other fibers
    // * the cross-terms with the background
    void accumulateFiber(
        int y,
        std::size_t fiberIndex
    ) {
        if (!_useTrace[fiberIndex]) {
            return;
        }

        auto const& dataImage = _image.getImage()->getArray()[y];
        auto const& dataMask = _image.getMask()->getArray()[y];
        auto const& dataVariance = _image.getVariance()->getArray()[y];

        // Here we calculate two versions of the matrix diagonal: one
        // without weighting and another with weighting by the inverse
        // variance. This allows us to use the un-weighted matrix for
        // solving for the fluxes without a bias (same reason we don't
        // weight by the variance when doing PSF photometry) and the inverse
        // of the weighted matrix diagonal to get an estimate of the
        // variance.
        auto const& iModel = _models[fiberIndex];
        double const model2 = iModel.dotSelf(_usePixel);
        double const model2Weighted = iModel.dotSelfWeighted(_usePixel, dataVariance);
        double const modelData = iModel.dotData(dataImage, _usePixel);
        double const sumModel = iModel.sum(_usePixel);
        std::size_t const numTracePixels = iModel.count(_usePixel);
        MaskT maskCore = iModel.accumulateMask(dataMask, _minFracMask);

        if (sumModel == 0.0 || numTracePixels == 0 || model2 == 0.0 || model2Weighted == 0.0) {
            _useTrace[fiberIndex] = false;
            return;
        }
        if (sumModel < _minFracImage) {
            if (maskCore == 0) {
                // We need to give some indication of why it's suspect
                maskCore = _maskBadResult[fiberIndex][y];
            }
            _maskResult[fiberIndex][y] |= _suspect | maskCore;
            _maskBadResult[fiberIndex][y] |= _suspect | maskCore;
        }

        std::size_t const iIndex = getSpectrumIndex(fiberIndex, y);
        assert(std::isfinite(model2));
        _matrix.add(iIndex, iIndex, model2);
        _haveMatrixEntry[iIndex] |= (model2 != 0.0);
        _diagonalWeighted[iIndex] = model2Weighted;
        _vector[iIndex] = modelData;
        _maskBadResult[fiberIndex][y] |= maskCore;

        // Spectrum-spectrum cross terms
        for (std::size_t jj = fiberIndex + 1; jj < _numFibers; ++jj) {
            if (!_useTrace[jj]) {
                continue;
            }

            auto const& jModel = _models[jj];
            double const modelModel = iModel.dotOther(jModel, _usePixel);
            _matrix.add(iIndex, getSpectrumIndex(jj, y), modelModel);
            assert(std::isfinite(modelModel));
            _haveMatrixEntry[getSpectrumIndex(jj, y)] |= (modelModel != 0.0);
        }

        // Spectrum-background cross-terms
        auto bgResult = iModel.dotBackground(_bg.xBlocks, _usePixel);
        auto const& bgTerms = bgResult.second;
        std::size_t bgIndex = getBackgroundIndex(0, y) + bgResult.first;
        for (std::size_t ii = 0; ii < bgTerms.size(); ++ii, ++bgIndex) {
            _matrix.add(iIndex, bgIndex, bgTerms[ii]);
            assert(std::isfinite(bgTerms[ii]));
            _haveMatrixEntry[bgIndex] |= (bgTerms[ii] != 0.0);
        }

#if 1
        // Kernel diagonal terms, kernel-kernel cross-terms and spectrum-kernel cross-terms
        // The "model basis function" here is the kernel convolved with the fiber trace,
        // multiplied by the spatial polynomial term.
        // model = sum_ijk b_ijk profile_i(x,y) * K_j . P_k(x,y)
        auto polyValues = _kernelPolynomial.getDFuncDParameters(_xCenter[fiberIndex], y);
        std::size_t iOffsetIndex = 0;
        bgTerms.deep() = 0.0;
        for (int iOffset = -_kernelHalfWidth; iOffset <= _kernelHalfWidth; ++iOffset, ++iOffsetIndex) {
            if (iOffset == 0) {
                --iOffsetIndex;
                continue;
            }

            FiberModel const& iKernelModel = _kernelModels[fiberIndex][iOffsetIndex];
            double const kernelDotSelf = iKernelModel.dotSelf(_usePixel);  // diagonal term
            double const kernelDotData = iKernelModel.dotData(dataImage, _usePixel);  // vector term

            ndarray::Array<double, 1, 1> kernelDotFiber = ndarray::allocate(_numFibers);
            for (std::size_t jj = 0; jj < _numFibers; ++jj) {
                if (!_useTrace[jj]) {
                    kernelDotFiber[jj] = 0.0;
                    continue;
                }
                kernelDotFiber[jj] = iKernelModel.dotOther(_models[jj], _usePixel);
            }

            auto const& bgResult = iKernelModel.dotBackground(_bg.xBlocks, _usePixel);
            auto const& bgTerms = bgResult.second;

            for (std::size_t iSpatial = 0; iSpatial < _numKernelSpatial; ++iSpatial) {
                std::size_t const iKernelIndex = getKernelIndex(iOffset, iSpatial);
                std::size_t const iKernelSubIndex = iKernelIndex - _kernelStart;
                double const iPoly = polyValues[iSpatial];
                // Kernel diagonal term
                _kernelTerms[iKernelSubIndex][iKernelSubIndex] += std::pow(iPoly, 2)*kernelDotSelf;
                _vector[iKernelIndex] += iPoly*kernelDotData;

                // Kernel-spectrum cross terms
                for (
                    std::size_t jj = 0, specIndex = getSpectrumIndex(0, y);
                    jj < _numFibers;
                    ++jj, ++specIndex
                ) {
                    _matrix.add(specIndex, iKernelIndex, iPoly*kernelDotFiber[jj]);
                }

                // Kernel-spatial cross terms
                for (std::size_t jSpatial = iSpatial + 1; jSpatial < _numKernelSpatial; ++jSpatial) {
                    std::size_t const jKernelIndex = getKernelIndex(iOffset, jSpatial) - _kernelStart;
                    double const jPoly = polyValues[jSpatial];
                    _kernelTerms[iKernelSubIndex][jKernelIndex] += iPoly*jPoly*kernelDotSelf;
                }

                // Kernel-kernel cross terms
                std::size_t jOffsetIndex = iOffsetIndex + 1;
                for (int jOffset = iOffset + 1; jOffset <= _kernelHalfWidth; ++jOffset, ++jOffsetIndex) {
                    if (jOffset == 0) {
                        --jOffsetIndex;
                        continue;
                    }
                    FiberModel const& jKernelModel = _kernelModels[fiberIndex][jOffsetIndex];
                    double const kernelDotKernel = iKernelModel.dotOther(jKernelModel, _usePixel);
                    for (std::size_t jSpatial = iSpatial + 1; jSpatial < _numKernelSpatial; ++jSpatial) {
                        std::size_t const jKernelIndex = getKernelIndex(iOffset, jSpatial) - _kernelStart;
                        double const jPoly = polyValues[jSpatial];
                        _kernelTerms[iKernelSubIndex][jKernelIndex] += iPoly*jPoly*kernelDotKernel;
                    }
                }

                // Kernel-background cross-terms
                std::size_t bgIndex = getBackgroundIndex(0, y) + bgResult.first;
                for (std::size_t ii = 0; ii < bgTerms.size(); ++ii, ++bgIndex) {
                    _matrix.add(iKernelIndex, bgIndex, iPoly*bgTerms[ii]);
                }
            }
        }
#endif
    }

    // Accumulate the background diagonal terms
    void accumulateBackground(int y) {
        auto iter = _image.row_begin(y);
        ndarray::Array<double, 1, 1> terms = ndarray::allocate(_bg.xNumBlocks);
        terms.deep() = 0.0;
        for (int xx = 0; xx < _image.getWidth(); ++xx, ++iter) {
            if (!_usePixel[xx]) {
                continue;
            }

            std::size_t const block = _bg.xBlocks[xx];
            terms[block] += 1.0;

            std::size_t const bgIndex = getBackgroundIndex(xx, y);
            _vector[bgIndex] += iter.image();
            _diagonalWeighted[bgIndex] += 1.0/iter.variance();
        }
        for (std::size_t ii = 0, bgIndex = getBackgroundIndex(0, y); ii < _bg.xNumBlocks; ++ii, ++bgIndex) {
            _matrix.add(bgIndex, bgIndex, terms[ii]);
            assert(std::isfinite(terms[ii]));
            _haveMatrixEntry[bgIndex] |= (terms[ii] != 0.0);
        }
    }

    ndarray::Array<double, 1, 1> solve() {
        // Add in the kernel terms, which were accumulated serparately
        std::size_t const numKernelTerms = _numKernels*_numKernelSpatial;
        for (std::size_t ii = 0, iIndex = _kernelStart; ii < numKernelTerms; ++ii, ++iIndex) {
            _matrix.add(iIndex, iIndex, _kernelTerms[ii][ii]);
            std::cerr << "Kernel term " << ii << ": " << _kernelTerms[ii][ii] << " " << _vector[iIndex] << std::endl;
            assert(std::isfinite(_kernelTerms[ii][ii]));
            _haveMatrixEntry[iIndex] |= (_kernelTerms[ii][ii] != 0.0);
            for (std::size_t jj = ii + 1, jIndex = iIndex + 1; jj < numKernelTerms; ++jj, ++jIndex) {
                if (_kernelTerms[ii][jj] != 0.0) {
                    _matrix.add(iIndex, jIndex, _kernelTerms[ii][jj]);
                    assert(std::isfinite(_kernelTerms[ii][jj]));
                    _haveMatrixEntry[jIndex] |= (_kernelTerms[ii][jj] != 0.0);
                }
            }
        }

        std::cerr << "Matrix size: " << _numParams << " non-zero entries: " << _matrix.size() << std::endl;

        // Ensure we have entries for all parameters, to avoid singular matrix
        for (std::size_t ii = 0; ii < _numParams; ++ii) {
            if (!_haveMatrixEntry[ii]) {
                assert(_vector[ii] == 0.0);
                assert(_diagonalWeighted[ii] == 0.0);

                // Avoid singular matrix
                _matrix.add(ii, ii, 1.0);
                _vector[ii] = 0.0;
                _diagonalWeighted[ii] = 1.0;
            }
        }

        // Solve the system of equations
        using Matrix = math::NonsymmetricSparseSquareMatrix::Matrix;
        using Solver = Eigen::SimplicialLDLT<Matrix, Eigen::Upper>;

        Solver solver;
        ndarray::Array<double, 1, 1> solution = ndarray::allocate(_numParams);
        _matrix.solve(solution, _vector, solver);

#if 0
        // Require non-negative backgrounds
        ndarray::Array<bool, 1, 1> requireNonnegative = ndarray::allocate(_numParams);
        requireNonnegative[ndarray::view(_bgStart, _bgStart + _bg.xNumBlocks)] = true;
        _matrix.solveNonnegative(solution, _vector, requireNonnegative);
        std::cerr << solution[ndarray::view(_bgStart, _bgStart + _bg.xNumBlocks)] << std::endl;
#endif

        std::cerr << "Done." << std::endl;
        return solution;
    }

    std::tuple<SpectrumSet, lsst::afw::image::Image<double>> extract(
        ndarray::Array<double, 1, 1> const& solution
    ) {
        // Extract the spectra
        SpectrumSet spectra(_numFibers, _numRows);
        for (std::size_t ii = 0; ii < _numFibers; ++ii) {
            spectra[ii]->setFiberId(_fiberTraces[ii]->getFiberId());
            for (int yy = 0; yy < _numRows; ++yy) {
                std::size_t const index = getSpectrumIndex(ii, yy);
                auto value = solution[index];
                auto variance = 1.0/_diagonalWeighted[index];
                MaskT maskValue = _maskResult[ii][yy];
                if (!std::isfinite(value)) {
                    value = 0.0;
                    maskValue |= _badFiberTrace;
                }
                if (!std::isfinite(variance) || variance <= 0) {
                    variance = 0.0;
                    maskValue |= _badFiberTrace;
                }
                spectra[ii]->getFlux()[yy] = value;
                spectra[ii]->getMask()(yy, 0) = maskValue;
                spectra[ii]->getVariance()[yy] = variance;
                spectra[ii]->getNorm()[yy] = _fiberNorm[ii][yy];
            }
        }

        // Extract the kernel
//        FiberKernel kernel(_kernelHalfWidth, _kernelOrder);
        std::cerr << "Kernel parameters:" << solution[ndarray::view(_kernelStart, _kernelStart + _numKernels*_numKernelSpatial)] << std::endl;

        // Extract the background
        lsst::afw::image::Image<double> background(_bg.xNumBlocks, _bg.yNumBlocks);
        std::size_t bgIndex = _bgStart;
        for (std::size_t yy = 0; yy < _bg.yNumBlocks; ++yy, bgIndex += _bg.xNumBlocks) {
            background.getArray()[yy] = solution[ndarray::view(bgIndex, bgIndex + _bg.xNumBlocks)];
        }

        return {std::move(spectra), std::move(background)};
    }

    std::tuple<SpectrumSet, lsst::afw::image::Image<double>> run() {
        for (int yy = 0; yy < _numRows; ++yy) {
            if (yy % 100 == 0) {
                std::cerr << "Processing row " << yy << "/" << _numRows << std::endl;
            }
            checkRow(yy);
            for (std::size_t ii = 0; ii < _numFibers; ++ii) {
                accumulateFiber(yy, ii);
            }
            accumulateBackground(yy);
        }
        auto const solution = solve();
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
    float _minFracMask;
    float _minFracImage;

    // Helpers
    math::NormalizedPolynomial2<double> _kernelPolynomial;
    std::size_t _kernelStart;
    std::size_t _numKernels;
    std::size_t _numKernelSpatial;
    ndarray::Array<double, 2, 2> _kernelTerms;
    BackgroundHelper _bg;
    std::size_t _bgStart;  // starting index of background parameters for current block
    lsst::afw::image::MaskPixel _requireMask;
    MaskT _noData;
    MaskT _badFiberTrace;
    MaskT _suspect;
    ndarray::Array<double, 2, 2> _fiberNorm;  // normalization for each fiber in current row
    ndarray::Array<bool, 1, 1> _useTrace;  // trace overlaps this row?
    ndarray::Array<bool, 1, 1> _usePixel;  // pixel in row should be used?
    ndarray::Array<int, 1, 1> _xMin, _xMax;  // first/last good pixel in trace
    ndarray::Array<double, 1, 1> _xCenter;  // center of trace in x direction, used for kernel polynomial
    std::vector<FiberModel> _models;  // cache of fiber models for current row
    std::vector<std::vector<FiberModel>> _kernelModels;  // cache of kernel-convolved fiber models for current row, indexed by [fiber][kernel offset]
    ndarray::Array<MaskT, 2, 2> _maskResult;  // mask value for each trace
    ndarray::Array<MaskT, 2, 2> _maskBadResult;  // mask value if trace is bad

    // Least-squares equation
    std::size_t _numParams;
    math::NonsymmetricSparseSquareMatrix _matrix;
    ndarray::Array<bool, 1, 1> _haveMatrixEntry;
    ndarray::Array<double, 1, 1> _vector;
    ndarray::Array<double, 1, 1> _diagonalWeighted;  // diagonal of weighted least-squares matrix
};


}  // anonymous namespace


std::tuple<SpectrumSet, lsst::afw::image::Image<double>> extractSpectra(
    lsst::afw::image::MaskedImage<float> const& image,
    FiberTraceSet<float> const& fiberTraces,
    lsst::afw::image::MaskPixel badBitMask,
    int kernelHalfWidth,
    int kernelOrder,
    int xBackgroundSize,
    int yBackgroundSize,
    float minFracMask,
    float minFracImage
) {
    return SpectrumExtractor(
        image, fiberTraces, badBitMask,
        kernelHalfWidth, kernelOrder,
        xBackgroundSize, yBackgroundSize,
        minFracMask, minFracImage
    ).run();
}


}}}  // namespace pfs::drp::stella