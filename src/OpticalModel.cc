#include "ndarray.h"

#include "pfs/drp/stella/utils/checkSize.h"
#include "pfs/drp/stella/math/quartiles.h"
#include "pfs/drp/stella/OpticalModel.h"

namespace pfs {
namespace drp {
namespace stella {


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
    double wavelengthDispersion,
    Array1D const& spatialOffsets,
    Array1D const& spectralOffsets,
    DistortionList const& distortions
) : _fiberPitch(fiberPitch),
    _wavelengthDispersion(wavelengthDispersion),
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
    SplinedDetectorMap const& source
) : SlitModel(
        source.getFiberId(),
        calculateFiberPitch(source),
        calculateWavelengthDispersion(source),
        source.getSpatialOffsets(),
        source.getSpectralOffsets()
    )
{}


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


lsst::geom::Point2D SlitModel::spectrographToSlit(int fiberId, double wavelength) const {
    double const spatial = fiberId + getSpatialOffset(fiberId)/_fiberPitch;
    double const spectral = wavelength + getSpectralOffset(fiberId)*_wavelengthDispersion;
    lsst::geom::Point2D slit{spatial, spectral};
    for (auto const& dd : _distortions) {
         slit = (*dd)(slit);
    }
    return slit;
}


SlitModel::Array2D SlitModel::spectrographToSlit(
    Array1I const& fiberId,
    Array1D const& wavelength
) const {
    utils::checkSize(fiberId.size(), wavelength.size(), "fiberId vs wavelength");
    Array2D result = ndarray::allocate(fiberId.size(), 2);
    for (std::size_t ii = 0; ii < fiberId.size(); ++ii) {
        lsst::geom::Point2D const slit = spectrographToSlit(fiberId[ii], wavelength[ii]);
        result[ii][0] = slit.getX();
        result[ii][1] = slit.getY();
    }
    return result;
}


////////////////////////////////////////////////////////////////////////////////
// OpticalModel
////////////////////////////////////////////////////////////////////////////////


OpticalModel::OpticalModel(
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
    OpticalModel::Array2D, OpticalModel::Array2D, OpticalModel::Array2D, OpticalModel::Array2D
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
            throw LSST_EXCEPT(
                lsst::pex::exceptions::LengthError,
                "All fibers must have the same number of wavelength spline knots"
            );
        }
        if (xCenterSplines[ii].getX().size() != numXCenter) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::LengthError,
                "All fibers must have the same number of xCenter spline knots"
            );
        }
    }

    // The current detectorMaps out of the simulator have numWavelength=69 and numXCenter=73.
    // This is because four points have been trimmed out of the wavelength splines to avoid
    // edge effects in the spline interpolation. We'll trim the xCenter splines also.
    if (numWavelength == 69 && numXCenter == 73) {
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

    OpticalModel::Array2D spatial = ndarray::allocate(numFibers, numWavelength);
    OpticalModel::Array2D spectral = ndarray::allocate(numFibers, numWavelength);
    OpticalModel::Array2D xx = ndarray::allocate(numFibers, numWavelength);
    OpticalModel::Array2D yy = ndarray::allocate(numFibers, numWavelength);
    for (std::size_t ii = 0; ii < numFibers; ++ii) {
        spatial[ndarray::view(ii)()] = fiberId[ii];
        spectral[ndarray::view(ii)()] = wavelengthSplines[ii].getY();
        xx[ndarray::view(ii)()] = xCenterSplines[ii].getY();

        if (!ndarray::all(xCenterSplines[ii].getX() == wavelengthSplines[ii].getX())) {
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


OpticalModel::OpticalModel(SplinedDetectorMap const& source)
    : OpticalModel(extractGrid(source)) {}


OpticalModel::OpticalModel(
    std::tuple<Array2D, Array2D, Array2D, Array2D> const& grid,
    DistortionList const& distortions
) : OpticalModel(std::get<0>(grid), std::get<1>(grid), std::get<2>(grid), std::get<3>(grid), distortions)
{}


////////////////////////////////////////////////////////////////////////////////
// DetectorModel
////////////////////////////////////////////////////////////////////////////////


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
{}


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
        result.setX(std::numeric_limits<double>::quiet_NaN());
    }

    return result;
}


DetectorModel::Array2D DetectorModel::detectorToPixels(Array1D const& x, Array1D const& y) const {
    utils::checkSize(x.size(), y.size(), "x vs y");
    Array2D result = ndarray::allocate(x.size(), 2);
    for (std::size_t ii = 0; ii < x.size(); ++ii) {
        lsst::geom::Point2D const pixel = detectorToPixels(lsst::geom::Point2D(x[ii], y[ii]));
        result[ii][0] = pixel.getX();
        result[ii][1] = pixel.getY();
    }
    return result;
}


lsst::geom::Point2D DetectorModel::pixelsToDetector(lsst::geom::Point2D const& pixels) const {
    if (!_isDivided) {
        return pixels;
    }
    lsst::geom::Point2D const nanPoint = lsst::geom::Point2D(
        std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()
    );
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


DetectorModel::Array2D DetectorModel::pixelsToDetector(Array1D const& p, Array1D const& q) const {
    utils::checkSize(p.size(), q.size(), "p vs q");
    Array2D result = ndarray::allocate(p.size(), 2);
    for (std::size_t ii = 0; ii < p.size(); ++ii) {
        lsst::geom::Point2D const detector = pixelsToDetector(lsst::geom::Point2D(p[ii], q[ii]));
        result[ii][0] = detector.getX();
        result[ii][1] = detector.getY();
    }
    return result;
}


}}}  // namespace pfs::drp::stella
