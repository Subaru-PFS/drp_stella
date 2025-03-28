#include <algorithm>

#include "ndarray/eigen.h"

#include "lsst/pex/exceptions.h"
#include "pfs/drp/stella/utils/checkSize.h"
#include "pfs/drp/stella/utils/math.h"

#define USE_SPLINE 1

#ifdef USE_SPLINE
#include "pfs/drp/stella/spline.h"
#else
#include "pfs/drp/stella/math/LinearInterpolation.h"
#endif

#include "pfs/drp/stella/FiberTrace.h"


namespace afwImage = lsst::afw::image;

namespace pfs { namespace drp { namespace stella {

template<typename ImageT, typename MaskT, typename VarianceT>
FiberTrace<ImageT, MaskT, VarianceT>::FiberTrace(
    MaskedImageT const& trace,
    std::size_t fiberId
) : _trace(trace),
    _fiberId(fiberId)
    {}


template<typename ImageT, typename MaskT, typename VarianceT>
FiberTrace<ImageT, MaskT, VarianceT>::FiberTrace(
    FiberTrace<ImageT, MaskT, VarianceT> const& fiberTrace,
    bool deep
) : _trace(fiberTrace.getTrace(), deep),
    _fiberId(fiberTrace.getFiberId())
    {}


template<typename ImageT, typename MaskT, typename VarianceT>
std::shared_ptr<afwImage::Image<ImageT>>
FiberTrace<ImageT, MaskT, VarianceT>::constructImage(
    Spectrum const& spectrum,
    lsst::geom::Box2I const& bbox
) const {
    auto out = std::make_shared<afwImage::Image<ImageT>>(bbox);
    *out = 0.0;
    constructImage(*out, spectrum);
    return out;
}


template<typename ImageT, typename MaskT, typename VarianceT>
void FiberTrace<ImageT, MaskT, VarianceT>::constructImage(
    afwImage::Image<ImageT> & image,
    Spectrum const& spectrum
) const {
    return constructImage(image, spectrum.getFlux());
}


template<typename ImageT, typename MaskT, typename VarianceT>
void FiberTrace<ImageT, MaskT, VarianceT>::constructImage(
    afwImage::Image<ImageT> & image,
    ndarray::Array<Spectrum::ImageT const, 1, 1> const& flux
) const {
    auto box = image.getBBox(lsst::afw::image::PARENT);
    box.clip(_trace.getBBox(lsst::afw::image::PARENT));

    if (flux.size() < std::size_t(box.getMaxY())) {
        std::ostringstream str;
        str << "Size of flux array (" << flux.size() << ") too small for box (" << box.getMaxY() << ")";
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthError, str.str());
    }

    auto const maskVal = _trace.getMask()->getPlaneBitMask(fiberMaskPlane);
    auto spec = flux.begin() + box.getMinY();
    for (std::ptrdiff_t y = box.getMinY(), row = box.getMinY() - _trace.getY0();
         y <= box.getMaxY(); ++y, ++row, ++spec) {
        std::ptrdiff_t const xStart = box.getMinX() - _trace.getX0();
        std::ptrdiff_t const xStop = box.getMaxX() - _trace.getX0();  // Inclusive
        auto profileIter = _trace.getImage()->row_begin(row) + xStart;
        auto maskIter = _trace.getMask()->row_begin(row) + xStart;
        auto imageIter = image.row_begin(row) + xStart;

        float const specValue = *spec;
        for (std::ptrdiff_t x = box.getMinX(); x <= box.getMaxX();
             ++x, ++profileIter, ++maskIter, ++imageIter) {
            if (*maskIter & maskVal) {
                *imageIter += specValue*(*profileIter);
            }
        }
    }
}


template<typename ImageT, typename MaskT, typename VarianceT>
FiberTrace<ImageT, MaskT, VarianceT> FiberTrace<ImageT, MaskT, VarianceT>::fromProfile(
    int fiberId,
    lsst::geom::Extent2I const& dims,
    int radius,
    double oversample,
    ndarray::Array<double, 1, 1> const& rows,
    ndarray::Array<double, 2, 1> const& profiles,
    ndarray::Array<bool, 2, 1> const& good,
    std::vector<std::pair<int, ndarray::Array<double, 1, 1>>> const& positions,
    ndarray::Array<Spectrum::ImageT, 1, 1> const& norm
) {
    int const width = dims.getX();
    std::size_t const height = dims.getY();
    std::size_t const numSwaths = rows.size();
    std::size_t const profileSize = 2*int((radius + 1)*oversample + 0.5) + 1;
    std::size_t const profileCenter = (radius + 1)*oversample + 0.5;
    auto const profileShape = ndarray::makeVector<std::size_t>(numSwaths, profileSize);
    utils::checkSize(profiles.getShape(), profileShape, "profiles");
    utils::checkSize(good.getShape(), profileShape, "good");
    utils::checkSize(positions.size(), height, "positions");
    if (!norm.isEmpty()) {
        utils::checkSize(norm.size(), height, "norm");
    }

    // Set up image of trace
    int xMin = width;
    int xMax = 0;
    for (std::size_t yy = 0; yy < height; ++yy) {
        std::size_t const size = positions[yy].second.size();
        if (size == 0) {
            continue;
        }
        int const xLow = positions[yy].first;
        assert(xLow >= 0);
        int const xHigh = xLow + size - 1;  // Inclusive
        assert(xHigh < width);
        xMin = std::min(xMin, xLow);
        xMax = std::max(xMax, xHigh);
    }
    if (xMin >= width - 1 || xMax <= 0) {
        // No valid centers --> no trace
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Trace does not touch image");
    }
    lsst::geom::Box2I box{lsst::geom::Point2I(xMin, 0),
                          lsst::geom::Point2I(xMax, height - 1)};
    lsst::afw::image::MaskedImage<double> image{box};
    lsst::afw::image::MaskPixel const ftMask = 1 << image.getMask()->addMaskPlane(fiberMaskPlane);

    // Set up profile interpolation
    ndarray::Array<double, 1, 1> xProfile = utils::arange<double>(0.0, profileSize, 1.0);
    ndarray::asEigenArray(xProfile) = (ndarray::asEigenArray(xProfile) - profileCenter)/oversample;
#ifdef USE_SPLINE
    using Interpolator = math::Spline<double>;
#else
    using Interpolator = math::LinearInterpolator<double, 1, 1>;
#endif
    std::vector<Interpolator> interpolators;
    interpolators.reserve(numSwaths);
    ndarray::Array<double, 1, 1> xLow = ndarray::allocate(numSwaths);
    ndarray::Array<double, 1, 1> xHigh = ndarray::allocate(numSwaths);
    for (std::size_t ii = 0; ii < numSwaths; ++ii) {
        ndarray::Array<double, 1, 1> const xProf = utils::arraySelect(xProfile, good[ii].shallow());
        ndarray::Array<double, 1, 1> const yProf = utils::arraySelect(profiles[ii].shallow(), good[ii].shallow());
#ifdef USE_SPLINE
        interpolators.emplace_back(xProf, yProf);
#else
        interpolators.emplace_back(xProf, yProf, 0.0, 0.0);
#endif
        xLow[ii] = xProf[0];
        xHigh[ii] = xProf[xProf.size() - 1];
    }

    // Calculate trace image
    std::ptrdiff_t const last = numSwaths - 1;
    std::ptrdiff_t nextIndex = 0;
    std::ptrdiff_t prevIndex = 0;
    for (std::size_t yy = 0; yy < height; ++yy) {
        if (yy <= rows[0]) {
            prevIndex = nextIndex = 0;
        } else if (yy >= rows[last]) {
            prevIndex = nextIndex = last;
        } else {
            while (rows[nextIndex] < yy && nextIndex < last) {
                ++nextIndex;
            }
            prevIndex = std::max(0L, nextIndex - 1);
        }

        image.getMask()->getArray()[yy] = 0;
        if (!norm.isEmpty() && !std::isfinite(norm[yy])) {
            image.getImage()->getArray()[yy] = norm[yy];
            continue;
        }
        image.getImage()->getArray()[yy] = 0.0;

        double const yPrev = rows[prevIndex];
        double const yNext = rows[nextIndex];
        double const prevLow = xLow[prevIndex];
        double const prevHigh = xHigh[prevIndex];
        double const nextLow = xLow[nextIndex];
        double const nextHigh = xHigh[nextIndex];
        auto const& prevInterp = interpolators[prevIndex];
        auto const& nextInterp = interpolators[nextIndex];
        double const nextWeight = (yy - yPrev)/(yNext - yPrev);
        double const prevWeight = 1.0 - nextWeight;

        int const xStart = positions[yy].first;  // inclusive
        ndarray::Array<double, 1, 1> const& dx = positions[yy].second;
        int const xStop = xStart + dx.size();  // exclusive

        auto imgIter = image.getImage()->row_begin(yy) + xStart - xMin;
        auto mskIter = image.getMask()->row_begin(yy) + xStart - xMin;
        auto dxIter = dx.begin();
        for (int xx = xStart; xx < xStop; ++xx, ++imgIter, ++mskIter, ++dxIter) {
                double const xRel = *dxIter;
                bool const prevOk = xRel >= prevLow && xRel <= prevHigh;
                bool const nextOk = xRel >= nextLow && xRel <= nextHigh;
                double const prevValue = prevOk ? prevInterp(xRel) : 0.0;
                double value;
                if (nextIndex == prevIndex) {
                    value = prevValue;
                } else {
                    double const nextValue = nextOk ? nextInterp(xRel) : 0.0;
                    value = prevValue*prevWeight + nextValue*nextWeight;
                }

                *imgIter = value;
                *mskIter = (prevOk || nextOk) ? ftMask : 0;
        }
        if (!norm.isEmpty()) {
            ndarray::asEigenArray(image.getImage()->getArray()[yy]) *= norm[yy];
        }
    }

    FiberTrace trace{lsst::afw::image::MaskedImage<float>(image, true)};
    trace.setFiberId(fiberId);
    return trace;
}


template<typename ImageT, typename MaskT, typename VarianceT>
FiberTrace<ImageT, MaskT, VarianceT> FiberTrace<ImageT, MaskT, VarianceT>::boxcar(
    int fiberId,
    lsst::geom::Extent2I const& dims,
    float radius,
    ndarray::Array<double, 1, 1> const& centers,
    ndarray::Array<double, 1, 1> const& norm
) {
    std::size_t const width = dims.getX();
    std::size_t const height = dims.getY();
    utils::checkSize(centers.size(), height, "centers");
    if (!norm.isEmpty()) {
        utils::checkSize(norm.size(), height, "norm");
    }

    // Set up image of trace
    auto const centersMinMax = std::minmax_element(centers.begin(), centers.end());
    int const xMin = std::max(0, int(*centersMinMax.first - radius));
    int const xMax = std::min(width - 1, std::size_t(std::ceil(*centersMinMax.second + radius)));  // incl.
    lsst::geom::Box2I box{lsst::geom::Point2I(xMin, 0),
                          lsst::geom::Point2I(xMax, height - 1)};
    lsst::afw::image::MaskedImage<double> image{box};
    image.getMask()->getArray().deep() = 0;
    lsst::afw::image::MaskPixel const ftMask = 1 << image.getMask()->addMaskPlane(fiberMaskPlane);

    // Pixels:   |    +    |    +    |    +    |
    // Aperture:        |---------X---------|
    // "+" marks are integers: 1, 2, 3; "|" marks are pixel boundaries at +/- 0.5
    // Center is at 2.2, aperture extends from 1.2 to 3.2.
    // Expected contributions (before row normalisation): 0.2, 1.0, 0.8
    for (std::size_t yy = 0; yy < height; ++yy) {
        float const middle = centers[yy];
        float const left = middle - radius;
        float const right = middle + radius;
        std::ptrdiff_t const low = std::max(0L, std::ptrdiff_t(left));
        std::ptrdiff_t const high = std::min(width, std::size_t(std::ceil(right)) + 1);  // exclusive
        double sum = 0;
        auto iter = image.row_begin(yy) + (low - xMin);
        // Use linear interpolation. There's probably fancier stuff with Lanczos resampling that we could do,
        // but this should be good enough.
        for (std::ptrdiff_t xx = low; xx < high; ++iter, ++xx) {
            float value = 1.0;
            if (xx < left - 0.5 || xx > right + 0.5) {
                value = 0.0;
            } else if (xx < left + 0.5) {
                value = std::max(0.0F, xx + 0.5F - left);
            } else if (xx > right - 0.5) {
                value = 1 - std::max(0.0F, xx + 0.5F - right);;
            }
            assert(value >= 0.0);
            iter.image() = value;
            iter.mask() = value > 0 ? ftMask : 0;
            sum += value;
        }
        if (sum != 0.0) {
            ndarray::asEigenArray(image.getImage()->getArray()[yy]) *= (norm.isEmpty() ? 1.0 : norm[yy])/sum;
        }
    }

    FiberTrace trace{lsst::afw::image::MaskedImage<float>(image, true)};
    trace.setFiberId(fiberId);
    return trace;
}


template<typename ImageT, typename MaskT, typename VarianceT>
Spectrum FiberTrace<ImageT, MaskT, VarianceT>::extractAperture(
    lsst::afw::image::MaskedImage<ImageT, MaskT, VarianceT> const& image,
    lsst::afw::image::MaskPixel badBitmask
) {
    std::size_t const height = image.getHeight();
    Spectrum spectrum{height, _fiberId};
    MaskT const require = image.getMask()->getPlaneBitMask(fiberMaskPlane);

    // Initialize results, in case we miss anything
    spectrum.getFlux().deep() = 0.0;
    spectrum.getMask().getArray().deep() = spectrum.getMask().getPlaneBitMask("NO_DATA");
    spectrum.getVariance().deep() = 0.0;
    spectrum.getNorm().deep() = 0.0;

    auto const trace = getTrace();
    auto bbox = trace.getBBox();
    bbox.clip(image.getBBox());
    if (bbox.getArea() == 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "No overlap between image and trace");
    }
    std::ptrdiff_t const xMin = bbox.getMinX();
    std::ptrdiff_t const xMax = bbox.getMaxX();  // inclusive

    MaskT const noData = 1 << image.getMask()->addMaskPlane("NO_DATA");
    MaskT const badFiberTrace = 1 << image.getMask()->addMaskPlane("BAD_FIBERTRACE");

    for (std::ptrdiff_t yy = bbox.getMinY(); yy <= bbox.getMaxY(); ++yy) {
        std::size_t const yImage = yy - image.getY0();
        std::size_t const yTrace = yy - trace.getY0();
        double sumImage = 0.0;
        double sumVariance = 0.0;
        double sumWeight = 0.0;
        auto imageIter = image.row_begin(yImage) + (xMin - image.getX0());
        auto traceIter = trace.row_begin(yTrace) + (xMin - trace.getX0());
        for (std::ptrdiff_t xx = xMin; xx <= xMax; ++xx, ++imageIter, ++traceIter) {
            ImageT const data = (*imageIter).image();
            VarianceT const var = (*imageIter).variance();
            double const weight = (*traceIter).image();
            if (!((*traceIter).mask() & require) || ((*imageIter).mask() & badBitmask)) {
                continue;
            }
            sumImage += data*weight;
            sumVariance += var*std::pow(weight, 2);
            sumWeight += weight;
        }
        spectrum.getFlux()[yImage] = sumImage;
        spectrum.getVariance()[yImage] = sumVariance;
        lsst::afw::image::MaskPixel maskResult = 0;
        if (sumWeight == 0 || !std::isfinite(sumWeight)) {
            maskResult = noData|badFiberTrace;
        } else if (!std::isfinite(sumImage) || !std::isfinite(sumVariance)) {
            maskResult |= noData;
        }
        spectrum.getMask().getArray()[0][yImage] = maskResult;
        spectrum.getNorm()[yImage] = sumWeight;
    }
    return spectrum;
}


// Explicit instantiation
template class FiberTrace<float, lsst::afw::image::MaskPixel, float>;

}}} // namespace pfs::drp::stella
