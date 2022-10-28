#include <numeric>
#include <algorithm>

#include "lsst/afw/math/LeastSquares.h"

#include "pfs/drp/stella/FiberTraceSet.h"
#include "pfs/drp/stella/math/SparseSquareMatrix.h"
#include "pfs/drp/stella/math/NormalizedPolynomial.h"

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


template <typename ImageT, typename MaskT, typename VarianceT>
class ExtractionFit {
  public:
    ExtractionFit(
        lsst::afw::image::MaskedImage<ImageT, MaskT, VarianceT> const& image,
        FiberTraceSet<ImageT, MaskT, VarianceT> const& traces,
        MaskT badBitMask,
        unsigned int bgOrder
    ) : _image(*image.getImage()),
        _mask(*image.getMask()),
        _traces(traces),
        result(traces.size(), image.getHeight()),
        _badBitMask(badBitMask),
        _background(bgOrder, lsst::geom::Box2D(image.getBBox())),
        _numRows(image.getHeight()),
        _numFibers(_traces.size()),
        _numBg(_background.getNParameters()),
        _numTerms(_numRows*_numFibers + _numBg),
        _vector(ndarray::allocate(_numTerms)),
        _norm(ndarray::allocate(_numFibers, _numRows)),
        useTrace(ndarray::allocate(_numFibers)),
        bgTerms(ndarray::allocate(_image.getWidth(), _numBg)),
        _matrix(_numTerms, _numRows*_numFibers*10 + _numFibers*_numBg + _numBg*_numBg) {
            for (std::size_t ii = 0; ii < _numFibers; ++ii) {
                auto const& bbox = _traces[ii]->getTrace().getBBox();
                for (std::size_t yy = 0; yy < std::min<std::ptrdiff_t>(bbox.getMinY(), _numRows); ++yy) {
                    _norm[ii][yy] = 0.0;
                }
                for (std::size_t yy = bbox.getMinY(), yTrace = 0; yy <= std::min<std::ptrdiff_t>(bbox.getMaxY(), _numRows - 1); ++yy, ++yTrace) {
                    auto const& array = _traces[ii]->getTrace().getImage()->getArray()[yTrace];
                    _norm[ii][yy] = ndarray::asEigenArray(array).template cast<double>().sum();
                }
                for (std::size_t yy = bbox.getMaxY() + 1; yy < _numRows; ++yy) {
                    _norm[ii][yy] = 0.0;
                }
            }
        }

    std::size_t getSpectrumIndex(std::size_t fiberIndex, std::size_t row) {
        std::size_t result = fiberIndex*_numRows + row;
        assert(result >= 0 && result < _numTerms && result < _numRows*_numFibers);
        return result;
    }
    std::size_t getBackgroundIndex(std::size_t bgTerm) {
        std::size_t result = _numRows*_numFibers + bgTerm;
        assert(result >= 0 && result < _numTerms);
        return result;
    }

    void addRow(std::size_t yy) {
        std::ptrdiff_t const x0 = _image.getX0();
        std::ptrdiff_t const y0 = _image.getY0();
        std::ptrdiff_t yTrace = yy + y0;
        MaskT const require = _mask.getPlaneBitMask(fiberMaskPlane);
        double const minFracMask = 0.0;

        // Determine which traces are relevant for this row
        for (std::size_t ii = 0; ii < _numFibers; ++ii) {
            auto const& box = _traces[ii]->getTrace().getBBox();
            useTrace[ii] = (yTrace >= box.getMinY() && yTrace <= box.getMaxY()) && _norm[ii][yy] != 0;
        }

        for (std::size_t xx = 0; xx < _image.getWidth(); ++xx) {
            auto const terms = _background.getDFuncDParameters(xx, yy);
            std::copy(terms.begin(), terms.end(), bgTerms[xx].begin());
        }

        for (std::size_t ii = 0; ii < _numFibers; ++ii) {
            std::size_t iIndex = getSpectrumIndex(ii, yy);
            if (!useTrace[ii]) {
                _matrix.add(iIndex, iIndex, 1.0);
                _vector[iIndex] = 0.0;
                continue;
            }

            double model2 = 0.0;  // model^2
            double model2Weighted = 0.0;  // model^2/sigma^2
            double modelData = 0.0;  // model dot data
            auto const& iTrace = _traces[ii]->getTrace();
            auto const iyModel = yTrace - iTrace.getBBox().getMinY();
            assert(iTrace.getBBox().getMinX() >= 0 && iTrace.getBBox().getMinY() >= 0);
            std::size_t const ixMin = iTrace.getBBox().getMinX();
            std::size_t const ixMax = iTrace.getBBox().getMaxX();
            auto const& iModelImage = *iTrace.getImage();
            auto const& iModelMask = *iTrace.getMask();
            double const iNorm = _norm[ii][yy];
            std::size_t const xStart = std::max(std::ptrdiff_t(ixMin), x0);

            for (std::size_t xModel = xStart - ixMin, xData = xStart - x0;
                 xModel < std::size_t(iTrace.getWidth()) && xData < _image.getWidth();
                 ++xModel, ++xData) {
                if (!(iModelMask(xModel, iyModel) & require)) continue;
                double const modelValue = iModelImage(xModel, iyModel)/iNorm;
                if (_mask(xData, yy) & _badBitMask) continue;
                double const dataValue = _image(xData, yy);
                double const m2 = std::pow(modelValue, 2);
                model2 += m2;
//                model2Weighted += m2/dataVariance(xData, yy);
                modelData += modelValue*dataValue;

                for (std::size_t pp = 0; pp < _numBg; ++pp) {
                    std::size_t const pIndex = getBackgroundIndex(pp);
                    double const bgValue = bgTerms[xData][pp];
                    _matrix.add(iIndex, pIndex, modelValue*bgValue);  // model dot background
                    _matrix.add(pIndex, pIndex, std::pow(bgValue, 2));  // background dot background
                    _vector[pIndex] += bgValue*dataValue;  // background dot data
                    for (std::size_t qq = pp + 1; qq < _numBg; ++qq) {
                        std::size_t const qIndex = getBackgroundIndex(qq);
                        _matrix.add(pIndex, qIndex, bgValue*bgTerms[xData][qq]);  // bg_i dot bg_j
                    }
                }
            }

            if (model2 == 0.0) { // || model2Weighted == 0.0) {
                _matrix.add(iIndex, iIndex, 1.0);  // to avoid making the matrix singular
                _vector[iIndex] = 0.0;
                continue;
            }

            _matrix.add(iIndex, iIndex, model2);
            _vector[iIndex] = modelData;
            if (ii >= _numFibers - 1) {
                continue;
            }

            for (std::size_t jj = ii + 1; jj < _numFibers; ++jj) {
                if (!useTrace[jj]) {
                    continue;
                }

                std::size_t jIndex = getSpectrumIndex(jj, yy);
                auto const& jTrace = _traces[jj]->getTrace();
                auto const& jModelImage = *jTrace.getImage();
                auto const& jModelMask = *jTrace.getMask();
                double const jNorm = _norm[jj][yy];

                // Determine overlap
                assert(jTrace.getBBox().getMinX() >= 0 && jTrace.getBBox().getMinY() >= 0);
                std::size_t const jxMin = jTrace.getBBox().getMinX();
                std::size_t const jxMax = jTrace.getBBox().getMaxX();
                std::size_t const jyModel = yy - jTrace.getBBox().getMinY();
                //   |----------------|
                // ixMin            ixMax
                //            |----------------|
                //          jxMin            jxMax
                std::size_t const overlapMin = std::max(ixMin, jxMin);
                std::size_t const overlapMax = std::min(ixMax, jxMax);
                std::size_t const overlapStart = std::max(std::ptrdiff_t(overlapMin), x0);

                if (overlapMax < overlapMin) {
                    break;
                }

                // Accumulate in overlap
                double modelModel = 0.0;  // model_i dot model_j
                double modelModelWeighted = 0.0;  // model_i dot model_j/sigma^2
                for (std::size_t xData = overlapStart - x0,
                     xi = overlapStart - ixMin,
                     xj = overlapStart - jxMin;
                     xData <= overlapMax - x0;
                     ++xData, ++xi, ++xj
                ) {
                    if (!(iModelMask(xi, iyModel) & require)) continue;
                    if (!(jModelMask(xj, jyModel) & require)) continue;
                    if (_mask(xData, yy) & _badBitMask) continue;
                    double const iModel = iModelImage(xi, iyModel)/iNorm;
                    double const jModel = jModelImage(xj, jyModel)/jNorm;
                    double const mm = iModel*jModel;
                    modelModel += mm;
//                    modelModelWeighted += mm/dataVariance(xData, yy);
                }
                _matrix.add(iIndex, jIndex, modelModel);
            }
        }
    }

    void construct() {
        for (std::size_t yy = 0; yy < _image.getHeight(); ++yy) {
            std::cerr << "Adding row " << yy << std::endl;
            addRow(yy);
        }
    }

    SpectrumSet solve() {
        std::cerr << "Solving..." << std::endl;
        // std::cerr << _vector << std::endl;
        // std::cerr << _matrix << std::endl;
        ndarray::Array<double, 1, 1> solution = _matrix.solve(_vector);
        std::cerr << "Done!" << std::endl;

        double const NaN = std::numeric_limits<double>::quiet_NaN();
        for (std::size_t ii = 0; ii < _numFibers; ++ii) {
            auto & spectrum = *result[ii];
            for (std::size_t yy = 0; yy < _image.getHeight(); ++yy) {
                std::size_t const index = getSpectrumIndex(ii, yy);
                spectrum.getFlux()[yy] = solution[index];
                spectrum.getMask()(yy, 0) = 0;
                spectrum.getNorm()[yy] = _norm[ii][yy];
                spectrum.getCovariance()[0][yy] = NaN;
                spectrum.getCovariance()[1][yy] = NaN;
                spectrum.getCovariance()[2][yy] = NaN;
            }
        }
        return std::move(result);
    }

  private:
    lsst::afw::image::Image<ImageT> const& _image;
    lsst::afw::image::Mask<MaskT> const& _mask;
    FiberTraceSet<ImageT, MaskT, VarianceT> const& _traces;
    SpectrumSet result;
    MaskT _badBitMask;
    math::NormalizedPolynomial2<double> _background;
    std::size_t _numRows;
    std::size_t _numFibers;
    std::size_t _numBg;
    std::size_t _numTerms;
    math::SparseSquareMatrix<true> _matrix;
    ndarray::Array<double, 1, 1> _vector;
    ndarray::Array<double, 2, 1> _norm;
    ndarray::Array<bool, 1, 1> useTrace;
    ndarray::Array<double, 2, 1> bgTerms;
};


template<typename ImageT, typename MaskT, typename VarianceT>
SpectrumSet FiberTraceSet<ImageT, MaskT, VarianceT>::extractSpectra(
    lsst::afw::image::MaskedImage<ImageT, MaskT, VarianceT> const& image,
    MaskT badBitMask,
    float minFracMask,
    unsigned int bgOrder
) {
#if 0
    std::size_t const num = size();
    std::size_t const height = image.getHeight();
    std::size_t const width = image.getWidth();
    SpectrumSet result{num, height};
    MaskT const require = image.getMask()->getPlaneBitMask(fiberMaskPlane);
    std::ptrdiff_t const x0 = image.getX0();
    std::ptrdiff_t const y0 = image.getY0();

    ndarray::Array<bool, 1, 1> useTrace = ndarray::allocate(num);  // trace overlaps this row?
    ndarray::Array<MaskT, 1, 1> maskResult = ndarray::allocate(num);  // mask value for each trace

    math::NormalizedPolynomial1<double> bg{bgOrder, 0, width - 1};
    std::size_t const numBg = bg.getNParameters();
    std::size_t const numTerms = num + numBg;
    ndarray::Array<double, 2, 1> bgTerms = ndarray::allocate(width, numBg);
    for (std::size_t xx = 0; xx < width; ++xx) {
        auto const terms = bg.getDFuncDParameters(xx);
        std::copy(terms.begin(), terms.end(), bgTerms[xx].begin());
    }
    ndarray::Array<double, 1, 1> vector = ndarray::allocate(numTerms);  // least-squares: model dot data

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
        spectrum->getNorm().deep() = 0.0;
    }

    // yData is the position on the image (and therefore the extracted spectrum)
    // yActual is the position on the trace
    std::ptrdiff_t yActual = y0;
    for (std::size_t yData = 0; yData < height; ++yData, ++yActual) {
        // Determine which traces are relevant for this row
        for (std::size_t ii = 0; ii < num; ++ii) {
            auto const& box = _traces[ii]->getTrace().getBBox();
            useTrace[ii] = (yActual >= box.getMinY() && yActual <= box.getMaxY());
            maskResult[ii] = noData;

            if (useTrace[ii]) {
                std::size_t const yy = yData - box.getMinY();
                auto const& trace = ndarray::asEigenArray(_traces[ii]->getTrace().getImage()->getArray()[yy]);
                result[ii]->getNorm()[yData] = trace.template cast<double>().sum();
                if (result[ii]->getNorm()[yData] == 0) {
                    useTrace[ii] = false;
                    maskResult[ii] |= badFiberTrace;
                }
            }
        }

        // Construct least-squares matrix and vector
        math::SparseSquareMatrix<true> matrix{numTerms};  // least-squares matrix: model dot model
        vector.deep() = 0.0;

        for (std::size_t ii = 0; ii < num; ++ii) {
            if (!useTrace[ii]) {
                matrix.add(ii, ii, 1.0);
                vector[ii] = 0.0;
                continue;
            }

            double model2 = 0.0;  // model^2
            double model2Weighted = 0.0;  // model^2/sigma^2
            double modelData = 0.0;  // model dot data
            auto const& iTrace = _traces[ii]->getTrace();
            auto const iyModel = yActual - iTrace.getBBox().getMinY();
            assert(iTrace.getBBox().getMinX() >= 0 && iTrace.getBBox().getMinY() >= 0);
            std::size_t const ixMin = iTrace.getBBox().getMinX();
            std::size_t const ixMax = iTrace.getBBox().getMaxX();
            auto const& iModelImage = *iTrace.getImage();
            auto const& iModelMask = *iTrace.getMask();
            double const iNorm = result[ii]->getNorm()[yData];
            maskResult[ii] = 0;
            std::size_t const xStart = std::max(std::ptrdiff_t(ixMin), x0);
            for (std::size_t xModel = xStart - ixMin, xData = xStart - x0;
                 xModel < std::size_t(iTrace.getWidth()) && xData < width;
                 ++xModel, ++xData) {
                if (!(iModelMask(xModel, iyModel) & require)) continue;
                double const modelValue = iModelImage(xModel, iyModel)/iNorm;
                if (modelValue > minFracMask) {
                    maskResult[ii] |= dataMask(xData, yData);
                }
                if (dataMask(xData, yData) & badBitMask) continue;
                double const dataValue = dataImage(xData, yData);
                double const m2 = std::pow(modelValue, 2);
                model2 += m2;
                model2Weighted += m2/dataVariance(xData, yData);
                modelData += modelValue*dataValue;

                for (std::size_t pp = 0, ppIndex = num; pp < numBg; ++pp, ++ppIndex) {
                    double const bgValue = bgTerms[xData][pp];
                    matrix.add(ii, ppIndex, modelValue*bgValue);  // model dot background
                    matrix.add(ppIndex, ppIndex, std::pow(bgValue, 2));  // background dot background
                    vector[ppIndex] += bgValue*dataValue;  // background dot data
                    for (std::size_t qq = pp + 1, qqIndex = ppIndex + 1; qq < numBg; ++qq, ++qqIndex) {
                        matrix.add(ppIndex, qqIndex, bgValue*bgTerms[xData][qq]);  // bg_p dot bg_q
                    }
                }
            }

            if (model2 == 0.0 || model2Weighted == 0.0) {
                useTrace[ii] = false;
                matrix.add(ii, ii, 1.0);  // to avoid making the matrix singular
                vector[ii] = 0.0;
                maskResult[ii] |= noData | badFiberTrace;
                continue;
            }

            matrix.add(ii, ii, model2);
            vector[ii] = modelData;
            if (ii >= num - 1) {
                continue;
            }

            for (std::size_t jj = ii + 1; jj < num; ++jj) {
                if (!useTrace[jj]) {
                    continue;
                }

                auto const& jTrace = _traces[jj]->getTrace();
                auto const& jModelImage = *jTrace.getImage();
                auto const& jModelMask = *jTrace.getMask();
                double const jNorm = result[jj]->getNorm()[yData];

                // Determine overlap
                assert(jTrace.getBBox().getMinX() >= 0 && jTrace.getBBox().getMinY() >= 0);
                std::size_t const jxMin = jTrace.getBBox().getMinX();
                std::size_t const jxMax = jTrace.getBBox().getMaxX();
                std::size_t const jyModel = yData - jTrace.getBBox().getMinY();
                //   |----------------|
                // ixMin            ixMax
                //            |----------------|
                //          jxMin            jxMax
                std::size_t const overlapMin = std::max(ixMin, jxMin);
                std::size_t const overlapMax = std::min(ixMax, jxMax);
                std::size_t const overlapStart = std::max(std::ptrdiff_t(overlapMin), x0);

                if (overlapMax < overlapMin) {
                    break;
                }

                // Accumulate in overlap
                double modelModel = 0.0;  // model_i dot model_j
                double modelModelWeighted = 0.0;  // model_i dot model_j/sigma^2
                for (std::size_t xData = overlapStart - x0,
                     xi = overlapStart - ixMin,
                     xj = overlapStart - jxMin;
                     xData <= overlapMax - x0;
                     ++xData, ++xi, ++xj
                ) {
                    if (!(iModelMask(xi, iyModel) & require)) continue;
                    if (!(jModelMask(xj, jyModel) & require)) continue;
                    if (dataMask(xData, yData) & badBitMask) continue;
                    double const iModel = iModelImage(xi, iyModel)/iNorm;
                    double const jModel = jModelImage(xj, jyModel)/jNorm;
                    double const mm = iModel*jModel;
                    modelModel += mm;
                    modelModelWeighted += mm/dataVariance(xData, yData);
                }

                matrix.add(ii, jj, modelModel);
            }
        }

        if (yData == 2460) {
            matrix.debug = true;
        }

        ndarray::Array<double, 1, 1> solution = matrix.solve(vector);
        for (std::size_t ii = 0; ii < num; ++ii) {
            auto value = solution[ii];
            double const NaN = std::numeric_limits<double>::quiet_NaN();
            if (!useTrace[ii] || !std::isfinite(value)) {
                value = 0.0;
                maskResult[ii] |= noData;
            }
            result[ii]->getSpectrum()[yData] = value;
            result[ii]->getMask()(yData, 0) = maskResult[ii];
            result[ii]->getCovariance()[0][yData] = NaN;
            result[ii]->getCovariance()[1][yData] = NaN;
            result[ii]->getCovariance()[2][yData] = NaN;
        }
    }

    for (std::size_t ii = 0; ii < num; ++ii) {
        result[ii]->setFiberId(_traces[ii]->getFiberId());
    }

    return result;
#else
//    lsst::afw::image::MaskedImage<ImageT, MaskT, VarianceT> sub = image.subset(lsst::geom::Box2I(lsst::geom::Point2I(0, 0), lsst::geom::Extent2I(image.getWidth(), 256)));
    ExtractionFit<ImageT, MaskT, VarianceT> fitter{image, *this, badBitMask, bgOrder};
    fitter.construct();
    return fitter.solve();
#endif
}


// Explicit instantiation
template class FiberTraceSet<float, lsst::afw::image::MaskPixel, float>;

}}} // namespace pfs::drp::stella
