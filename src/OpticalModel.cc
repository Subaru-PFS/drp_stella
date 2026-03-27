#include "ndarray.h"

#include "pfs/drp/stella/utils/checkSize.h"
#include "pfs/drp/stella/math/quartiles.h"
#include "pfs/drp/stella/OpticalModel.h"

namespace pfs {
namespace drp {
namespace stella {


double constexpr NOT_A_NUMBER = std::numeric_limits<double>::quiet_NaN();


FiberMap makeFiberMap(ndarray::Array<int, 1, 1> const& fiberId) {
    FiberMap result;
    for (std::size_t ii = 0; ii < fiberId.size(); ++ii) {
        int const ff = fiberId[ii];
        result[ff] = ii;
    }
    return result;
}


namespace {


double calculateFiberPitch(DetectorMap const& detMap) {
    double const row = detMap.getBBox().getCenterY();
    auto const& fiberId = detMap.getFiberId();
    std::size_t const num = fiberId.size();
    ndarray::Array<double, 1, 1> xCenter = ndarray::allocate(num - 1);
    double last = detMap.getXCenter(fiberId.front(), row);
    for (std::size_t ii = 0, jj = 1; jj < num; ++ii, ++jj) {
        double const next = detMap.getXCenter(fiberId[jj], row);
        xCenter[ii] = next - last;
        last = next;
    }
    return math::calculateMedian(xCenter);
}


double calculateWavelengthDispersion(DetectorMap const& detMap) {
    double const row = detMap.getBBox().getCenterY();
    int const fiberId = detMap.getFiberId()[detMap.getNumFibers()/2];
    return detMap.findWavelength(fiberId, row + 1) - detMap.findWavelength(fiberId, row);
}


}  // anonymous namespace


SlitModel::SlitModel(
    ndarray::Array<int, 1, 1> const& fiberId,
    double fiberPitch,
    double fiberMin,
    double wavelengthDispersion,
    double wavelengthMin,
    Array1D const& spatialOffsets,
    Array1D const& spectralOffsets,
    DistortionList const& distortions
) : _fiberPitch(fiberPitch),
    _fiberMin(fiberMin),
    _wavelengthDispersion(wavelengthDispersion),
    _wavelengthMin(wavelengthMin),
    _fiberId(fiberId),
    _spatialOffsets(spatialOffsets),
    _spectralOffsets(spectralOffsets),
    _fiberMap(makeFiberMap(fiberId)),
    _distortions(distortions)
{
    utils::checkSize(fiberId.size(), spatialOffsets.size(), "fiberId vs spatialOffsets");
    utils::checkSize(fiberId.size(), spectralOffsets.size(), "fiberId vs spectralOffsets");
}


SlitModel::SlitModel(
    SplinedDetectorMap const& source,
    DistortionList const& distortions
) : SlitModel(
        source.getFiberId(),
        calculateFiberPitch(source),
        utils::arrayMin(source.getFiberId()),
        calculateWavelengthDispersion(source),
        utils::arrayMin(source.getWavelength(source.getFiberId()[source.getFiberId().size()/2])),
        source.getSpatialOffsets(),
        source.getSpectralOffsets(),
        distortions
    )
{}


SlitModel SlitModel::copy() const {
    DistortionList distortions;
    distortions.reserve(_distortions.size());
    for (auto const& dd : _distortions) {
        distortions.emplace_back(dd->clone());
    }
    return SlitModel(
        ndarray::copy(getFiberId()),
        getFiberPitch(),
        getFiberMin(),
        getWavelengthDispersion(),
        getWavelengthMin(),
        ndarray::copy(getSpatialOffsets()),
        ndarray::copy(getSpectralOffsets()),
        distortions
    );
}


SlitModel SlitModel::withDistortion(std::shared_ptr<Distortion> distortion) const {
    if (!distortion) {
        return copy();
    }

    DistortionList distortions;
    distortions.reserve(_distortions.size());
    for (auto const& dd : _distortions) {
        distortions.emplace_back(dd->clone());
    }
    distortions.push_back(distortion);

    return SlitModel(
        ndarray::copy(getFiberId()),
        getFiberPitch(),
        getFiberMin(),
        getWavelengthDispersion(),
        getWavelengthMin(),
        ndarray::copy(getSpatialOffsets()),
        ndarray::copy(getSpectralOffsets()),
        distortions
    );
}


SlitModel SlitModel::withoutDistortion() const {
    return SlitModel(
        ndarray::copy(getFiberId()),
        getFiberPitch(),
        getFiberMin(),
        getWavelengthDispersion(),
        getWavelengthMin(),
        ndarray::copy(getSpatialOffsets()),
        ndarray::copy(getSpectralOffsets()),
        DistortionList()
    );
}


double SlitModel::getSpatialOffset(int fiberId) const {
    auto const iter = _fiberMap.find(fiberId);
    if (iter == _fiberMap.end()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeError, "fiberId not found");
    }
    return _spatialOffsets[iter->second];
}


double SlitModel::getSpectralOffset(int fiberId) const {
    auto const iter = _fiberMap.find(fiberId);
    if (iter == _fiberMap.end()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeError, "fiberId not found");
    }
    return _spectralOffsets[iter->second];
}


lsst::geom::Point2D SlitModel::_spectrographToPreSlit(int fiberId, double wavelength) const {
//    std::cerr << "fiberId,wavelength: " << fiberId << ", " << wavelength << std::endl;
    lsst::geom::Point2D preSlit = fiberWavelengthToPixels(fiberId, wavelength);
//    std::cerr << " --> toPixels: " << preSlit;
    preSlit += lsst::geom::Extent2D(getSpatialOffset(fiberId), getSpectralOffset(fiberId));
//    std::cerr << " + slitOffset:  " << preSlit;
    for (auto const& dd : _distortions) {
        preSlit += lsst::geom::Extent2D((*dd)(preSlit));
    }
//    std::cerr << " + distortions: " << preSlit << std::endl;
    return preSlit;
}


////////////////////////////////////////////////////////////////////////////////
// OpticsModel
////////////////////////////////////////////////////////////////////////////////


OpticsModel::OpticsModel(
    Array2D const& spatial,
    Array2D const& spectral,
    Array2D const& x,
    Array2D const& y,
    DistortionList const& distortions
) : _xOrig(x),
    _yOrig(y),
    _distortions(distortions),
    _slitToDetector(spatial, spectral, x, y, distortions),
    _detectorToSlit(_slitToDetector.inverse())
{}


namespace {


std::tuple<
    OpticsModel::Array2D, OpticsModel::Array2D, OpticsModel::Array2D, OpticsModel::Array2D
> extractGrid(SplinedDetectorMap const& source) {
    auto const& fiberId = source.getFiberId();
    std::size_t const numFibers = fiberId.size();

    using Spline = SplinedDetectorMap::Spline;

    std::vector<Spline> xCenterSplines;
    std::vector<Spline> wavelengthSplines;
    xCenterSplines.reserve(numFibers);
    wavelengthSplines.reserve(numFibers);
    for (int ff : fiberId) {
        xCenterSplines.push_back(source.getXCenterSpline(ff));
        wavelengthSplines.push_back(source.getWavelengthSpline(ff));
    }

    if (wavelengthSplines.size() == 0) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LengthError,
            "SplinedDetectorMap must contain at least one fiber"
        );
    }

    std::size_t const numWavelength = wavelengthSplines[0].getX().size();
    std::size_t numXCenter = xCenterSplines[0].getX().size();
    for (std::size_t ii = 0; ii < numFibers; ++ii) {
        if (wavelengthSplines[ii].getX().size() != numWavelength) {
            std::cerr << "Fiber " << fiberId[ii] << " has " << wavelengthSplines[ii].getX().size()
                      << " wavelength spline knots, but expected " << numWavelength << std::endl;
            throw LSST_EXCEPT(
                lsst::pex::exceptions::LengthError,
                "All fibers must have the same number of wavelength spline knots"
            );
        }
        if (xCenterSplines[ii].getX().size() != numXCenter) {
            std::cerr << "Fiber " << fiberId[ii] << " has " << xCenterSplines[ii].getX().size()
                      << " xCenter spline knots, but expected " << numXCenter << std::endl;
            throw LSST_EXCEPT(
                lsst::pex::exceptions::LengthError,
                "All fibers must have the same number of xCenter spline knots"
            );
        }
    }

    // The current detectorMaps out of the simulator have numXCenter = numWavelength + 4.
    // This is because four points have been trimmed out of the wavelength splines to avoid
    // edge effects in the spline interpolation. We'll trim the xCenter splines also.
    if ((numXCenter > numWavelength) && (numXCenter - numWavelength == 4)) {
        std::vector<Spline> newXCenterSplines;
        newXCenterSplines.reserve(numFibers);
        for (std::size_t ii = 0; ii < numFibers; ++ii) {

            Spline::ConstArray const& knots = xCenterSplines[ii].getX();
            Spline::ConstArray const& values = xCenterSplines[ii].getY();

            Spline::Array newKnots = ndarray::allocate(knots.size() - 4);
            Spline::Array newValues = ndarray::allocate(values.size() - 4);

            newKnots[0] = knots[0];
            newValues[0] = values[0];

            newKnots[ndarray::view(1, newKnots.size() - 1)] = knots[ndarray::view(3, knots.size() - 3)];
            newValues[ndarray::view(1, newValues.size() - 1)] = values[ndarray::view(3, values.size() - 3)];

            newKnots[newKnots.size() - 1] = knots[knots.size() - 1];
            newValues[newValues.size() - 1] = values[values.size() - 1];

            newXCenterSplines.emplace_back(
                newKnots,
                newValues,
                xCenterSplines[ii].getInterpolationType(),
                xCenterSplines[ii].getExtrapolationType()
            );
        }
        xCenterSplines = std::move(newXCenterSplines);
        numXCenter = xCenterSplines[0].getX().size();
    }
    if (numWavelength != numXCenter) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LengthError,
            "Number of wavelength spline knots must match number of xCenter spline knots"
        );
    }

    OpticsModel::Array2D spatial = ndarray::allocate(numFibers, numWavelength);
    OpticsModel::Array2D spectral = ndarray::allocate(numFibers, numWavelength);
    OpticsModel::Array2D xx = ndarray::allocate(numFibers, numWavelength);
    OpticsModel::Array2D yy = ndarray::allocate(numFibers, numWavelength);
    for (std::size_t ii = 0; ii < numFibers; ++ii) {
        spatial[ndarray::view(ii)()] = fiberId[ii];
        spectral[ndarray::view(ii)()] = wavelengthSplines[ii].getY();
        xx[ndarray::view(ii)()] = xCenterSplines[ii].getY();

        double maxDiff = 0.0;
        for (std::size_t jj = 0; jj < numWavelength; ++jj) {
            double const diff = std::abs(xCenterSplines[ii].getX()[jj] - wavelengthSplines[ii].getX()[jj]);
            if (diff > maxDiff) {
                maxDiff = diff;
            }
        }
        if (maxDiff > 1e-5) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::LengthError,
                "xCenter and wavelength spline knots must match for each fiber"
            );
        }
        yy[ndarray::view(ii)()] = wavelengthSplines[ii].getX();
    }

    return std::make_tuple(spatial, spectral, xx, yy);
}


}  // anonymous namespace


OpticsModel::OpticsModel(SplinedDetectorMap const& source, DistortionList const& distortions)
    : OpticsModel(extractGrid(source), distortions) {}


OpticsModel::OpticsModel(
    std::tuple<Array2D, Array2D, Array2D, Array2D> const& grid,
    DistortionList const& distortions
) : OpticsModel(std::get<0>(grid), std::get<1>(grid), std::get<2>(grid), std::get<3>(grid), distortions)
{}


OpticsModel OpticsModel::copy() const {
    DistortionList distortions;
    distortions.reserve(_distortions.size());
    for (auto const& dd : _distortions) {
        distortions.emplace_back(dd->clone());
    }
    return OpticsModel(
        ndarray::copy(getSpatial()),
        ndarray::copy(getSpectral()),
        ndarray::copy(getX()),
        ndarray::copy(getY()),
        distortions
    );
}


lsst::geom::Point2D OpticsModel::slitToDetector(double spatial, double spectral) const {
    if (!std::isfinite(spatial) || !std::isfinite(spectral)) {
        return lsst::geom::Point2D(NOT_A_NUMBER, NOT_A_NUMBER);
    }
    try {
        return _slitToDetector(spatial, spectral);
    } catch (...) {
        return lsst::geom::Point2D(NOT_A_NUMBER, NOT_A_NUMBER);
    }
}


lsst::geom::Point2D OpticsModel::detectorToSlit(double x, double y) const {
    if (!std::isfinite(x) || !std::isfinite(y)) {
        return lsst::geom::Point2D(NOT_A_NUMBER, NOT_A_NUMBER);
    }
    try {
        return _detectorToSlit(x, y);
    } catch (...) {
        return lsst::geom::Point2D(NOT_A_NUMBER, NOT_A_NUMBER);
    }
}


////////////////////////////////////////////////////////////////////////////////
// DetectorModel
////////////////////////////////////////////////////////////////////////////////


DetectorModel::DetectorModel(
    lsst::geom::Box2I const& bbox,
    lsst::geom::AffineTransform const& rightCcd,
    DistortionList const& distortions
) : DetectorModel(bbox, true, rightCcd, distortions) {}


DetectorModel::DetectorModel(
    lsst::geom::Box2I const& bbox,
    DistortionList const& distortions
) : DetectorModel(bbox, false, lsst::geom::AffineTransform(), distortions) {}


DetectorModel::DetectorModel(
    lsst::geom::Box2I const& bbox,
    bool isDivided,
    lsst::geom::AffineTransform const& rightCcd,
    DistortionList const& distortions
) : _bbox(bbox),
    _isDivided(isDivided),
    _rightCcd(rightCcd),
    _distortions(distortions),
    _xCenter(bbox.getCenterX()),
    _xOffset(0.5*(getBBox().getMinX() + getBBox().getMaxX())),
    _xScale(2.0/(getBBox().getMaxX() - getBBox().getMinX())),
    _yOffset(0.5*(getBBox().getMinY() + getBBox().getMaxY())),
    _yScale(2.0/(getBBox().getMaxY() - getBBox().getMinY()))
{
    if (!_distortions.empty()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicError,
            "Distortions are not supported"
        );
    }
}


DetectorModel DetectorModel::copy() const {
    DistortionList distortions;
    distortions.reserve(_distortions.size());
    for (auto const& dd : _distortions) {
        distortions.emplace_back(dd->clone());
    }

    lsst::geom::AffineTransform rightCcd;
    rightCcd.setParameterVector(_rightCcd.getParameterVector());

    return DetectorModel(
        lsst::geom::Box2I(getBBox()),
        getIsDivided(),
        rightCcd,
        distortions
    );
}


lsst::geom::Point2D DetectorModel::detectorToPixels(lsst::geom::Point2D const& detector) const {
    if (!_isDivided || detector.getX() < _xCenter) {
        return detector;
    }

    double const x = detector.getX();
    double const y = detector.getY();

    lsst::geom::Point2D normalized{
        (x - _xOffset)*_xScale,
        (y - _yOffset)*_yScale
    };
    lsst::geom::Point2D result = detector + lsst::geom::Extent2D(_rightCcd(normalized));
    if (result.getX() < _xCenter) {
        // Off the right detector in the chip gap
        result.setX(NOT_A_NUMBER);
    }

    return result;
}


lsst::geom::Point2D DetectorModel::pixelsToDetector(lsst::geom::Point2D const& pixels) const {
    if (!_isDivided) {
        return pixels;
    }
    lsst::geom::Point2D const nanPoint = lsst::geom::Point2D(NOT_A_NUMBER, NOT_A_NUMBER);
    if (!std::isfinite(pixels.getX()) || !std::isfinite(pixels.getY())) {
        // No idea where we are
        return nanPoint;
    }
    if (pixels.getX() < _xCenter) {
        return pixels;
    }

    // We need to find the detector point for which
    //     detectorToPixels(detector) == pixels
    lsst::geom::Point2D detector = pixels - _rightCcd.getTranslation();
    lsst::geom::Extent2D delta;
    std::size_t iter = 0;
    do {
        lsst::geom::Point2D newPixels = detectorToPixels(detector);
        if (!std::isfinite(newPixels.getX()) || !std::isfinite(newPixels.getY())) {
            // Is there something I can do here?
            return nanPoint;
        }
        delta = newPixels - pixels;
        detector -= delta;
        if (++iter > MAX_ITER) {
            return nanPoint;
        }
    } while (delta.computeSquaredNorm() > std::pow(PRECISION, 2));

    return detector;
}


std::pair<int, ndarray::Array<double, 1, 1>> DetectorModel::detectorToPixelsColumns(
    lsst::geom::Point2D const& detector,
    int halfWidth
) const {
    lsst::geom::Point2D const pixel = detectorToPixels(detector);
    double const center = pixel.getX();
    double const row = pixel.getY();

    int pStart;  // Left pixel of the fiber on the pixels; inclusive
    int pStop; // Right pixel of the fiber on the pixels; inclusive

    // Case 0: the center is on the detector --> need to find the left and right sides of the fiber
    if (std::isfinite(center)) {
        int const centerInt = std::floor(center);
        pStart = centerInt - halfWidth;
        pStop = centerInt + halfWidth + 1;
    } else {
        // The center is off the detector to the left or right, or in the chip gap
        // Check both extents to see if those are on the detector
        lsst::geom::Point2D const left = detectorToPixels(detector - lsst::geom::Extent2D(halfWidth, 0));
        lsst::geom::Point2D const right = detectorToPixels(detector + lsst::geom::Extent2D(halfWidth, 0));


        // If the left side is on the detector, we're either off the detector to the right,
        // or in the chip gap.
        // If the halfWidth is a reasonable size, we can tell the difference between these cases
        // by the position of the left side: is it on the left chip or the right chip?
        // Similarly for the right side.

        if (std::isfinite(left.getX())) {
            pStart = left.getX();
            if (std::isfinite(right.getX())) {
                pStop = right.getX() + 0.5;
            } else {
                pStop = _isDivided && (left.getX() <= _xCenter) ? _xCenter : getBBox().getMaxX();
            }
        } else if (std::isfinite(right.getX())) {
            pStart = _isDivided && (right.getX() > _xCenter) ? getBBox().getMinX() : _xCenter + 1;
            pStop = right.getX() + 0.5;
        } else {
            // If neither side is on the detector, we're so far off to the side, or the halfWidth is so small
            // that it fits in the chip gap (or so large that it spans the entire detector).
            // In any case there's nothing to do.
            return std::make_pair(0, ndarray::allocate(0));
        }
    }

    pStart = std::max(pStart, getBBox().getMinX());
    pStop = std::min(pStop, getBBox().getMaxX());
    if (pStart > pStop) {
        return std::make_pair(0, ndarray::allocate(0));
    }

    std::vector<double> pRel;  // p positions relative to the fiber center
    pRel.reserve(pStop - pStart + 1);
    int pMin = -1;
    for (int pp = pStart; pp <= pStop; ++pp) {
        lsst::geom::Point2D const point = pixelsToDetector(lsst::geom::Point2D(pp, row));
        double const dp = point.getX() - detector.getX();
        if (std::isfinite(point.getX()) && std::abs(dp) <= halfWidth) {
            if (pMin == -1) {
                pMin = pp;
            }
            pRel.push_back(dp);
        }
    }

    return std::make_pair(pMin, ndarray::copy(utils::vectorToArray(pRel)));
}


}}}  // namespace pfs::drp::stella
