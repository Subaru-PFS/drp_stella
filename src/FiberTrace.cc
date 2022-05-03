#include <algorithm>

#include "ndarray/eigen.h"

#include "lsst/pex/exceptions.h"
#include "pfs/drp/stella/utils/checkSize.h"
#include "pfs/drp/stella/utils/math.h"
#include "pfs/drp/stella/spline.h"

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
    auto box = image.getBBox(lsst::afw::image::PARENT);
    box.clip(_trace.getBBox(lsst::afw::image::PARENT));

    auto const maskVal = _trace.getMask()->getPlaneBitMask(fiberMaskPlane);
    auto spec = spectrum.getSpectrum().begin() + box.getMinY();
    auto bg = spectrum.getBackground().begin() + box.getMinY();
    for (std::ptrdiff_t y = box.getMinY(); y <= box.getMaxY(); ++y, ++spec, ++bg) {
        auto profileIter = _trace.getImage()->row_begin(y - _trace.getY0()) + box.getMinX() - _trace.getX0();
        auto maskIter = _trace.getMask()->row_begin(y - _trace.getY0()) + box.getMinX() - _trace.getX0();;
        auto imageIter = image.row_begin(y - image.getY0()) + box.getMinX() - image.getX0();;
        float const bgValue = *bg;
        float const specValue = *spec;
        for (std::ptrdiff_t x = box.getMinX(); x <= box.getMaxX();
             ++x, ++profileIter, ++maskIter, ++imageIter) {
            if (*maskIter & maskVal) {
                *imageIter += bgValue + specValue*(*profileIter);
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
    ndarray::Array<double, 1, 1> const& centers,
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
    utils::checkSize(centers.size(), height, "centers");
    if (!norm.isEmpty()) {
        utils::checkSize(norm.size(), height, "norm");
    }

    // Set up image of trace
    auto const centersMinMax = std::minmax_element(centers.begin(), centers.end());
    int const xMin = std::max(0, int(*centersMinMax.first) - radius);
    int const xMax = std::min(dims.getX() - 1, int(std::ceil(*centersMinMax.second)) + radius);
    if (xMin > int(width) || xMax < 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Centers extend beyond bounds of the image");
    }
    lsst::geom::Box2I box{lsst::geom::Point2I(xMin, 0),
                          lsst::geom::Point2I(xMax, height - 1)};
    lsst::afw::image::MaskedImage<double> image{box};
    lsst::afw::image::MaskPixel const ftMask = 1 << image.getMask()->addMaskPlane(fiberMaskPlane);

    // Set up profile splines
    ndarray::Array<double, 1, 1> xProfile = utils::arange<double>(0.0, profileSize, 1.0);
    ndarray::asEigenArray(xProfile) = (ndarray::asEigenArray(xProfile) - profileCenter)/oversample;
    std::vector<math::Spline<double>> splines;
    splines.reserve(numSwaths);
    ndarray::Array<double, 1, 1> xLow = ndarray::allocate(numSwaths);
    ndarray::Array<double, 1, 1> xHigh = ndarray::allocate(numSwaths);
    for (std::size_t ii = 0; ii < numSwaths; ++ii) {
        auto const xProf = utils::arraySelect(xProfile, good[ii].shallow());
        splines.emplace_back(xProf, utils::arraySelect(profiles[ii].shallow(), good[ii].shallow()),
                             math::Spline<double>::CUBIC_NATURAL);
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

        if (!norm.isEmpty() && !std::isfinite(norm[yy])) {
            image.getImage()->getArray()[yy] = norm[yy];
            image.getMask()->getArray()[yy] = 0;
            continue;
        }

        double const yPrev = rows[prevIndex];
        double const yNext = rows[nextIndex];
        double const prevLow = xLow[prevIndex];
        double const prevHigh = xHigh[prevIndex];
        double const nextLow = xLow[nextIndex];
        double const nextHigh = xHigh[nextIndex];
        auto const& prevSpline = splines[prevIndex];
        auto const& nextSpline = splines[nextIndex];
        double const nextWeight = (yy - yPrev)/(yNext - yPrev);
        double const prevWeight = 1.0 - nextWeight;

        double xRel = box.getMinX() - centers[yy];
        auto imgIter = image.getImage()->row_begin(yy);
        auto mskIter = image.getMask()->row_begin(yy);
        double sum = 0.0;
        for (int xx = box.getMinX(); xx <= box.getMaxX(); ++xx, xRel += 1.0, ++imgIter, ++mskIter) {
                bool const prevOk = xRel >= prevLow && xRel <= prevHigh;
                bool const nextOk = xRel >= nextLow && xRel <= nextHigh;
                double const prevValue = prevOk ? prevSpline(xRel) : 0.0;
                double value;
                if (nextIndex == prevIndex) {
                    value = prevValue;
                } else {
                    double const nextValue = nextOk ? nextSpline(xRel) : 0.0;
                    value = prevValue*prevWeight + nextValue*nextWeight;
                }

                *imgIter = value;
                *mskIter = (prevOk || nextOk) ? ftMask : 0;
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


// Explicit instantiation
template class FiberTrace<float, lsst::afw::image::MaskPixel, float>;

}}} // namespace pfs::drp::stella
