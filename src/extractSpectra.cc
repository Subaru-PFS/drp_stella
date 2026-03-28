#include "ndarray.h"
#include "ndarray/eigen.h"

#include "pfs/drp/stella/extractSpectra.h"
#include "pfs/drp/stella/math/SparseSquareMatrix.h"
#include "pfs/drp/stella/utils/math.h"

namespace pfs {
namespace drp {
namespace stella {


namespace {


struct BackgroundHelper {
    BackgroundHelper(lsst::geom::Extent2I const& dims, int xBlockSize, int yBlockSize)
        : xNumBlocks((dims.getX() + xBlockSize - 1) / xBlockSize),
          yNumBlocks((dims.getY() + yBlockSize - 1) / yBlockSize),
          xBlocks(ndarray::allocate(dims.getX())),
          yMin(ndarray::allocate(yNumBlocks)),
          yMax(ndarray::allocate(yNumBlocks)) {
        if (xBlockSize <= 0 || yBlockSize <= 0) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, "Block sizes must be positive");
        }
        for (int ii = 0; ii < dims.getX(); ++ii) {
            xBlocks[ii] = ii / xBlockSize;
        }

        for (int ii = 0; ii < int(yNumBlocks); ++ii) {
            yMin[ii] = ii * yBlockSize;
            yMax[ii] = std::min((ii + 1) * yBlockSize, dims.getY());
        }
    }

    std::size_t xNumBlocks;
    std::size_t yNumBlocks;
    ndarray::Array<int, 1, 1> xBlocks;
    ndarray::Array<int, 1, 1> yMin;
    ndarray::Array<int, 1, 1> yMax;
};


struct SpectrumExtractor {
    using ImageT = float;
    using MaskT = lsst::afw::image::MaskPixel;
    using VarianceT = float;

    SpectrumExtractor(
        lsst::afw::image::MaskedImage<ImageT> const& image,
        FiberTraceSet<ImageT> const& fiberTraces,
        lsst::afw::image::MaskPixel badBitMask,
        int xBlockSize,
        int yBlockSize,
        float minFracMask,
        float minFracImage
    ) : _numFibers(fiberTraces.size()),
        _image(image),
        _fiberTraces(fiberTraces),
        _badBitMask(badBitMask),
        _xBlockSize(xBlockSize),
        _yBlockSize(yBlockSize),
        _minFracMask(minFracMask),
        _minFracImage(minFracImage),
        _bg(image.getDimensions(), xBlockSize, yBlockSize),
        _bgStart(_numFibers * yBlockSize),
        _requireMask(image.getMask()->getPlaneBitMask(fiberMaskPlane)),
        _noData(1 << image.getMask()->addMaskPlane("NO_DATA")),
        _badFiberTrace(1 << image.getMask()->addMaskPlane("BAD_FIBERTRACE")),
        _suspect(1 << image.getMask()->addMaskPlane("SUSPECT")),
        _useTrace(ndarray::allocate(_numFibers)),
        _traceMin(ndarray::allocate(_numFibers)),
        _traceMax(ndarray::allocate(_numFibers)),
        _maskResult(ndarray::allocate(_numFibers, yBlockSize)),
        _maskBadResult(ndarray::allocate(_numFibers, yBlockSize)),
        _numParams(_bgStart + _bg.xNumBlocks),
        _matrix(_numParams),
        _haveMatrixEntry(ndarray::allocate(_numParams)),
        _vector(_numParams),
        _diagonalWeighted(_numParams),
        _spectra(_numFibers, image.getHeight()),
        _background(_bg.xNumBlocks, _bg.yNumBlocks)
    {
        // Initialize results, in case we miss anything
        for (std::size_t ii = 0; ii < _numFibers; ++ii) {
            _spectra[ii]->setFiberId(_fiberTraces[ii]->getFiberId());
            _spectra[ii]->getFlux().deep() = 0.0;
            _spectra[ii]->getMask().getArray().deep() = _spectra[ii]->getMask().getPlaneBitMask("NO_DATA");
            _spectra[ii]->getVariance().deep() = 0.0;
            _spectra[ii]->getNorm().deep() = 0.0;
        }
        _background = 0.0;

        // Ensure we can refer to traces in order from left to right
        ndarray::Array<double, 1, 1> xCenter = ndarray::allocate(_numFibers);
        for (std::size_t ii = 0; ii < _numFibers; ++ii) {
            auto const& box = _fiberTraces[ii]->getTrace().getBBox();
            xCenter[ii] = 0.5*(box.getMinX() + box.getMaxX());
        }
        _sortFiberTraces = utils::argsort(xCenter);
    }

    FiberTrace<ImageT> const& getFiberTrace(std::size_t ii) const {
        return *_fiberTraces[_sortFiberTraces[ii]];
    }

    // Determine which traces are relevant for this row
    // Calculate the normalization for each trace
    // Set the mask values for success and failure cases
    void checkRow(int y, int yMinBlock) {
        std::ptrdiff_t const blockRow = y - yMinBlock;
        for (std::size_t ii = 0; ii < _numFibers; ++ii) {
            auto const& box = getFiberTrace(ii).getTrace().getBBox();
            _useTrace[ii] = (y >= box.getMinY() && y <= box.getMaxY());
            _traceMin[ii] = box.getMinX();
            _traceMax[ii] = box.getMaxX();
            _maskResult[ii][blockRow] = 0;  // used in case of success
            _maskBadResult[ii][blockRow] = _noData;  // used in case of failure

            if (_useTrace[ii]) {
                std::size_t const yy = y - box.getMinY();
                auto const& traceImage = getFiberTrace(ii).getTrace();
                auto const& mask = ndarray::asEigenMatrix(traceImage.getMask()->getArray()[yy]);
                auto const select = mask.unaryExpr(
                    [this](lsst::afw::image::MaskPixel value) { return value & _requireMask; }
                ).template cast<bool>();
                auto const& trace = ndarray::asEigenMatrix(traceImage.getImage()->getArray()[yy]);
                double const sumModel = select.select(trace.template cast<double>(), 0.0).sum();
                _spectra[ii]->getNorm()[y] = sumModel;
                if (sumModel == 0 || !std::isfinite(sumModel)) {
                    _useTrace[ii] = false;
                    _maskBadResult[ii][blockRow] |= _badFiberTrace;
                }
            }
        }
    }

#if 0
    double getModelValue(std::size_t fiber, int x, int y) const {
        FiberTrace<ImageT> const& trace = getFiberTrace(fiber);
        int const x0 = trace.getTrace().getBBox().getMinX();
        double const value = (*trace.getTrace().getImage())(x - x0, y);
        double const norm = _spectra[fiber]->getNorm()[y];
        return value/norm;
    }
#endif

    bool useModel(std::size_t fiber, int x, int y) const {
        FiberTrace<ImageT> const& trace = *_fiberTraces[fiber];
        int const x0 = trace.getTrace().getBBox().getMinX();
        auto const& mask = (*trace.getTrace().getMask())(x - x0, y);
        return _useTrace[fiber] && ((mask & _requireMask) != 0);
    }

    std::size_t getSpectrumIndex(std::size_t fiber, int blockRow) const {
        return blockRow*_numFibers + fiber;
    }

    std::size_t getBackgroundIndex(int x) const {
        return _bgStart + _bg.xBlocks[x];
    }

    bool isGoodImage(ImageT value, MaskT mask, VarianceT variance) const {
        return (mask & _badBitMask) == 0 && std::isfinite(value) && std::isfinite(variance) && variance > 0;
    }
    bool isGoodImage(auto & iter) const {
        return isGoodImage(iter.image(), iter.mask(), iter.variance());
    }

#if 0
    void accumulateRow(
        int y,
        int blockRow
    ) {
        std::size_t fiberMin = 0;  // index of first fiber with traceMax > column
        std::size_t fiberMax = 0;  // index of first fiber with traceMin > column
        auto iter = _image.row_begin(y);
        for (int xx = 0; xx < _image.getWidth(); ++xx, ++iter) {
            for (; fiberMin < _numFibers && _traceMax[fiberMin] < xx; ++fiberMin);  // no-op
            for (; fiberMax < _numFibers && _traceMin[fiberMax] < xx; ++fiberMax);  // no-op

            if (y == 123 && xx == 1234) {
                std::cerr << "traceMin: " << _traceMin << std::endl;
                std::cerr << "traceMax: " << _traceMax << std::endl;
                std::cerr << "fiberMin=" << fiberMin << ", fiberMax=" << fiberMax << std::endl;
            }

            ImageT const imageValue = iter.image();
            MaskT const maskValue = iter.mask();
            VarianceT const varianceValue = iter.variance();

            // Accumulate the good mask values
            // Mask values get accumulated only for "core" pixels, where the model is above threshold
            for (std::size_t ii = fiberMin; ii < fiberMax; ++ii) {
                if (useModel(ii, xx, y) && getModelValue(ii, xx, y) > _minFracMask) {
                    _maskResult[ii][blockRow] |= maskValue;
                }
            }

            // Accumulate the bad mask values
            // If the pixel is bad, we want to know why,
            // so we accumulate all bad mask values, even for non-core pixels,
            // and store it in a separate array to be used if the pixel turns out to be bad.
            if (
                (maskValue & _badBitMask)
                || !std::isfinite(imageValue)
                || !std::isfinite(varianceValue)
                || varianceValue == 0
            ) {
                for (std::size_t ii = fiberMin; ii < fiberMax; ++ii) {
                    _maskBadResult[ii][blockRow] |= maskValue;
                }
                continue;
            }

            std::size_t const bgIndex = getBackgroundIndex(xx);
            _matrix.add(bgIndex, bgIndex, 1.0);
            _haveMatrixEntry[bgIndex] = true;
            _vector[bgIndex] += imageValue;
            _diagonalWeighted[bgIndex] += 1.0/varianceValue;

            for (std::size_t ii = fiberMin; ii < fiberMax; ++ii) {
                if (!useModel(ii, xx, y)) {
                    continue;
                }

                std::size_t const iIndex = getSpectrumIndex(ii, blockRow);
                double const iModelValue = getModelValue(ii, xx, y);
                double const iModelValue2 = std::pow(iModelValue, 2);
                _matrix.add(iIndex, iIndex, iModelValue2);
                _haveMatrixEntry[iIndex] = true;
                _vector[iIndex] += iModelValue*imageValue;
                _diagonalWeighted[iIndex] += iModelValue2/varianceValue;

                _matrix.add(iIndex, bgIndex, iModelValue);

                for (std::size_t jj = ii + 1; jj < fiberMax; ++jj) {
                    if (!useModel(jj, xx, y)) {
                        continue;
                    }
                    std::size_t const jIndex = getSpectrumIndex(jj, blockRow);
                    double const jModelValue = getModelValue(jj, xx, y);
                    _matrix.add(iIndex, jIndex, iModelValue*jModelValue);
                    _haveMatrixEntry[jIndex] = true;
                }
            }
        }
    }
#endif
    void accumulateRow(
        int y,
        int blockRow
    ) {
        for (std::size_t ii = 0; ii < _numFibers; ++ii) {
            if (!_useTrace[ii]) {
                continue;
            }

            auto const& dataImage = *_image.getImage();
            auto const& dataMask = *_image.getMask();
            auto const& dataVariance = *_image.getVariance();

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
            double sumModel = 0.0;  // sum of model values
            auto const& iTrace = _fiberTraces[ii]->getTrace();
            auto const iyModel = y - iTrace.getBBox().getMinY();
            assert(iTrace.getBBox().getMinX() >= 0 && iTrace.getBBox().getMinY() >= 0);
            std::size_t const ixMin = iTrace.getBBox().getMinX();
            std::size_t const ixMax = iTrace.getBBox().getMaxX();
            auto const& iModelImage = *iTrace.getImage();
            auto const& iModelMask = *iTrace.getMask();
            double const iNorm = _spectra[ii]->getNorm()[y];
            std::size_t numTracePixels = 0;
            std::size_t const xStart = std::max(std::ptrdiff_t(ixMin), 0L);
            MaskT maskCore = 0;

            // Background cross-terms
            std::size_t const minBlock = _bg.xBlocks[xStart];
            std::size_t const maxBlock = _bg.xBlocks[xStart + iTrace.getBBox().getWidth() - 1];
            std::size_t const numBlocks = maxBlock - minBlock + 1;
            ndarray::Array<double, 1, 1> bgTerms = ndarray::allocate(numBlocks);
            bgTerms.deep() = 0.0;

            for (int xModel = xStart - ixMin, xData = xStart;
                 xModel < iTrace.getWidth() && xData < _image.getWidth();
                 ++xModel, ++xData) {
                if (!(iModelMask(xModel, iyModel) & _requireMask)) continue;
                ++numTracePixels;
                double const modelValue = iModelImage(xModel, iyModel)/iNorm;
                MaskT const maskValue = dataMask(xData, y);
                ImageT const imageValue = dataImage(xData, y);
                VarianceT const varianceValue = dataVariance(xData, y);
                bool const isCore = modelValue > _minFracMask;
                if (isCore) {
                    maskCore |= maskValue;
                }
                if (!isGoodImage(imageValue, maskValue, varianceValue)) {
                    _maskBadResult[ii][blockRow] |= maskValue;
                    continue;
                }
                if (isCore) {
                    _maskResult[ii][blockRow] |= maskValue;
                }
                double const m2 = std::pow(modelValue, 2);
                model2 += m2;
                model2Weighted += m2/varianceValue;
                modelData += modelValue*imageValue;
                sumModel += modelValue;

                std::size_t const block = _bg.xBlocks[xData];
                bgTerms[block - minBlock] += modelValue;
            }
            _maskBadResult[ii][blockRow] |= maskCore;

            if (sumModel == 0.0 || numTracePixels == 0 || model2 == 0.0 || model2Weighted == 0.0) {
                _useTrace[ii] = false;
                continue;
            } else if (sumModel < _minFracImage) {
                if (maskCore == 0) {
                    // We need to give some indication of why it's suspect
                    maskCore = _maskBadResult[ii][blockRow];
                }
                _maskResult[ii][blockRow] |= _suspect | maskCore;
                _maskBadResult[ii][blockRow] |= _suspect | maskCore;
            }

            std::size_t const iIndex = getSpectrumIndex(ii, blockRow);
            _matrix.add(iIndex, iIndex, model2);
            _haveMatrixEntry[iIndex] |= (model2 != 0.0);
            _diagonalWeighted[iIndex] = model2Weighted;
            _vector[iIndex] = modelData;

            for (std::size_t ii = 0, bgIndex = _bgStart + minBlock; ii < numBlocks; ++ii, ++bgIndex) {
                _matrix.add(iIndex, bgIndex, bgTerms[ii]);
                _haveMatrixEntry[bgIndex] |= (bgTerms[ii] != 0.0);
            }

            if (ii >= _numFibers - 1) {
                continue;
            }

            for (std::size_t jj = ii + 1; jj < _numFibers; ++jj) {
                if (!_useTrace[jj]) {
                    continue;
                }

                auto const& jTrace = _fiberTraces[jj]->getTrace();
                auto const& jModelImage = *jTrace.getImage();
                auto const& jModelMask = *jTrace.getMask();
                double const jNorm = _spectra[jj]->getNorm()[y];

                // Determine overlap
                assert(jTrace.getBBox().getMinX() >= 0 && jTrace.getBBox().getMinY() >= 0);
                std::size_t const jxMin = jTrace.getBBox().getMinX();
                std::size_t const jxMax = jTrace.getBBox().getMaxX();
                std::size_t const jyModel = y - jTrace.getBBox().getMinY();
                //   |----------------|
                // ixMin            ixMax
                //            |----------------|
                //          jxMin            jxMax
                std::size_t const overlapMin = std::max(ixMin, jxMin);
                std::size_t const overlapMax = std::min(ixMax, jxMax);
                std::size_t const overlapStart = std::max(overlapMin, 0UL);

                if (overlapMax < overlapMin) {
                    // No more overlaps
                    break;
                }

                // Accumulate in overlap
                double modelModel = 0.0;  // model_i dot model_j
                for (
                    std::size_t xData = overlapStart, xi = overlapStart - ixMin, xj = overlapStart - jxMin;
                    xData <= overlapMax;
                    ++xData, ++xi, ++xj
                ) {
                    if (!(iModelMask(xi, iyModel) & _requireMask)) continue;
                    if (!(jModelMask(xj, jyModel) & _requireMask)) continue;
                    MaskT const maskValue = dataMask(xData, y);
                    ImageT const imageValue = dataImage(xData, y);
                    VarianceT const varianceValue = dataVariance(xData, y);
                    if ((maskValue & _badBitMask) || !std::isfinite(imageValue) ||
                        !std::isfinite(varianceValue) || varianceValue == 0) {
                        continue;
                    }
                    double const iModel = iModelImage(xi, iyModel)/iNorm;
                    double const jModel = jModelImage(xj, jyModel)/jNorm;
                    double const mm = iModel*jModel;
                    modelModel += mm;
                }
                std::size_t const jIndex = getSpectrumIndex(jj, blockRow);
                _matrix.add(iIndex, jIndex, modelModel);
                _haveMatrixEntry[jIndex] |= (modelModel != 0.0);
            }
        }

        // Accumulate the background terms
        auto iter = _image.row_begin(y);
        ndarray::Array<double, 1, 1> terms = ndarray::allocate(_bg.xNumBlocks);
        terms.deep() = 0.0;
        for (int xx = 0; xx < _image.getWidth(); ++xx, ++iter) {
            if (!isGoodImage(iter)) {
                continue;
            }

            std::size_t const block = _bg.xBlocks[xx];
            terms[block] += 1.0;

            std::size_t const bgIndex = _bgStart + block;
            _vector[bgIndex] += iter.image();
            _diagonalWeighted[bgIndex] += 1.0/iter.variance();
        }
        for (std::size_t ii = 0, bgIndex = _bgStart; ii < _bg.xNumBlocks; ++ii, ++bgIndex) {
            _matrix.add(bgIndex, bgIndex, terms[ii]);
            _haveMatrixEntry[bgIndex] |= (terms[ii] != 0.0);
        }
    }

    void solveBlock(
        int yBlock
    ) {
        std::cerr << "Solving block " << yBlock << "/" << _bg.yNumBlocks;
        std::cerr << " matrix size: " << _numParams << " non-zero entries: " << _matrix.size() << std::endl;

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
#if 0
        ndarray::Array<double, 1, 1> solution = _matrix.solve(_vector);
#else
        using Matrix = math::NonsymmetricSparseSquareMatrix::Matrix;
        using Solver = Eigen::SimplicialLDLT<Matrix, Eigen::Upper>;

        Solver solver;
        ndarray::Array<double, 1, 1> solution = ndarray::allocate(_numParams);
        _matrix.solve(solution, _vector, solver);
#endif
        std::cerr << "Done." << std::endl;

        // Extract the spectra
        for (int yy = _bg.yMin[yBlock], blockRow = 0; yy < _bg.yMax[yBlock]; ++yy, ++blockRow) {
            for (std::size_t ii = 0; ii < _numFibers; ++ii) {
                auto value = solution[getSpectrumIndex(ii, blockRow)];
                auto variance = 1.0/_diagonalWeighted[getSpectrumIndex(ii, blockRow)];
                MaskT maskValue = _maskResult[ii][blockRow];
                if (!std::isfinite(value)) {
                    value = 0.0;
                    maskValue |= _badFiberTrace;
                }
                if (!std::isfinite(variance) || variance <= 0) {
                    variance = 0.0;
                    maskValue |= _badFiberTrace;
                }
                _spectra[ii]->getFlux()[yy] = value;
                _spectra[ii]->getMask()(yy, 0) = maskValue;
                _spectra[ii]->getVariance()[yy] = variance;
            }
        }
        // Extract the background
        for (std::size_t ii = 0, jj = _bgStart; ii < _bg.xNumBlocks; ++ii, ++jj) {
            _background(ii, yBlock) = solution[jj];
        }
    }

    void reset() {
        _matrix.reset();
        _haveMatrixEntry.deep() = false;
        _vector.deep() = 0.0;
        _diagonalWeighted.deep() = 0.0;
    }

    void run() {
        for (std::size_t yBlock = 0; yBlock < _bg.yNumBlocks; ++yBlock) {
            reset();
            for (int yy = _bg.yMin[yBlock]; yy < _bg.yMax[yBlock]; ++yy) {
                checkRow(yy, _bg.yMin[yBlock]);
                accumulateRow(yy, yy - _bg.yMin[yBlock]);
            }
            solveBlock(yBlock);
        }
    }

    // Inputs
    std::size_t _numFibers;
    lsst::afw::image::MaskedImage<ImageT> const& _image;
    FiberTraceSet<ImageT> const& _fiberTraces;
    lsst::afw::image::MaskPixel _badBitMask;
    int _xBlockSize;
    int _yBlockSize;
    float _minFracMask;
    float _minFracImage;

    // Helpers
    ndarray::Array<std::size_t, 1, 1> _sortFiberTraces;
    BackgroundHelper _bg;
    std::size_t _bgStart;  // starting index of background parameters for current block
    lsst::afw::image::MaskPixel _requireMask;
    MaskT _noData;
    MaskT _badFiberTrace;
    MaskT _suspect;
    ndarray::Array<bool, 1, 1> _useTrace;  // trace overlaps this row?
    ndarray::Array<int, 1, 1> _traceMin;  // minimum column for each trace in this row
    ndarray::Array<int, 1, 1> _traceMax;  // maximum column for each trace in this row
    ndarray::Array<MaskT, 2, 2> _maskResult;  // mask value for each trace
    ndarray::Array<MaskT, 2, 2> _maskBadResult;  // mask value if trace is bad

    // Equation for the current block
    std::size_t _numParams;
    math::NonsymmetricSparseSquareMatrix _matrix;
    ndarray::Array<bool, 1, 1> _haveMatrixEntry;
    ndarray::Array<double, 1, 1> _vector;
    ndarray::Array<double, 1, 1> _diagonalWeighted;  // diagonal of weighted least-squares matrix

    // Results for all blocks
    SpectrumSet _spectra;
    lsst::afw::image::Image<double> _background;
};


}  // anonymous namespace


std::pair<SpectrumSet, lsst::afw::image::Image<double>> extractSpectra(
    lsst::afw::image::MaskedImage<float> const& image,
    FiberTraceSet<float> const& fiberTraces,
    lsst::afw::image::MaskPixel badBitMask,
    int xBlockSize,
    int yBlockSize,
    float minFracMask,
    float minFracImage
) {
    SpectrumExtractor extractor(
        image, fiberTraces, badBitMask, xBlockSize, yBlockSize, minFracMask, minFracImage
    );
    extractor.run();
    return {std::move(extractor._spectra), std::move(extractor._background)};
}


}}}  // namespace pfs::drp::stella