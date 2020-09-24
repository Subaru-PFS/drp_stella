#include <set>
#include <numeric>

#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/FCNBase.h"

#include "ndarray.h"

#include "pfs/drp/stella/utils/checkSize.h"
#include "pfs/drp/stella/utils/math.h"
#include "pfs/drp/stella/GlobalDetectorModel.h"

#define NDEBUG

#ifndef NDEBUG
#include "Minuit2/MnPrint.h"
#endif


namespace pfs {
namespace drp {
namespace stella {

namespace {


// Fit a straight line, y = f(x)
template <typename T, typename U>
std::tuple<double, double, double, double, std::size_t> fitStraightLine(
    ndarray::Array<T, 1, 1> const& xx,  // x values
    ndarray::Array<U, 1, 1> const& yy,   // y values
    ndarray::Array<bool, 1, 1> const& good  // use in fit?
) {
    assert(xx.size() == yy.size());
    std::size_t const length = xx.size();
    double xMean = 0.0;
    double yMean = 0.0;
    std::size_t num = 0;
    for (std::size_t ii = 0; ii < length; ++ii) {
        if (!good[ii]) continue;
        xMean += xx[ii];
        yMean += yy[ii];
        ++num;
    }
    xMean /= num;
    yMean /= num;
    double xySum = 0.0;
    double xxSum = 0.0;
    for (std::size_t ii = 0; ii < length; ++ii) {
        if (!good[ii]) continue;
        T const dx = xx[ii] - xMean;
        T const dy = yy[ii] - yMean;
        xySum += dx*dy;
        xxSum += dx*dx;
    }
    double const slope = xySum/xxSum;
    double const intercept = yMean - slope*xMean;
    return std::make_tuple(slope, intercept, xMean, yMean, num);
}


// Construct a mapping from fiberId to index
GlobalDetectorModel::FiberMap makeFiberMap(
    ndarray::Array<int, 1, 1> const& fiberId  // List of fiberId values; may contain duplicates
) {
    std::set<int> unique(fiberId.begin(), fiberId.end());
    GlobalDetectorModel::FiberMap fiberMap;
    fiberMap.reserve(unique.size());
    std::size_t ii = 0;
    for (auto iter = unique.begin(); iter != unique.end(); ++iter, ++ii) {
        fiberMap[*iter] = ii;
    }
    return fiberMap;
}


}  // anonymous namespace


GlobalDetectorModel::GlobalDetectorModel(
    lsst::geom::Box2I const& bbox,
    int distortionOrder,
    ndarray::Array<int, 1, 1> const& fiberId,
    bool dualDetector,
    std::vector<double> const& parameters,
    bool copyParameters
) : GlobalDetectorModel(bbox, distortionOrder, makeFiberMap(fiberId), dualDetector, parameters,
                        copyParameters)
{}


GlobalDetectorModel::GlobalDetectorModel(
    lsst::geom::Box2I const& bbox,
    int distortionOrder,
    GlobalDetectorModel::FiberMap const& fiberMap,
    bool dualDetector,
    std::vector<double> const& parameters,
    bool copyParameters
) : GlobalDetectorModel(
    bbox, distortionOrder, fiberMap, dualDetector,
    copyParameters ?
        ndarray::Array<double const, 1, 1>(
            ndarray::external(
                parameters.data(),
                ndarray::makeVector(getNumParameters(distortionOrder, fiberMap.size()
        )))).deep() :
        ndarray::external(
            parameters.data(),
            ndarray::makeVector(getNumParameters(distortionOrder, fiberMap.size()))
    )
) {
    assert(parameters.size() == getNumParameters(distortionOrder, fiberMap.size()));
}


GlobalDetectorModel::GlobalDetectorModel(
    lsst::geom::Box2I const& bbox,
    int distortionOrder,
    ndarray::Array<int, 1, 1> const& fiberId,
    bool dualDetector,
    ndarray::Array<double const, 1, 1> const& parameters
) : GlobalDetectorModel(bbox, distortionOrder, makeFiberMap(fiberId), dualDetector, parameters) {}


GlobalDetectorModel::GlobalDetectorModel(
    lsst::geom::Box2I const& bbox,
    int distortionOrder,
    GlobalDetectorModel::FiberMap const& fiberMap,
    bool dualDetector,
    ndarray::Array<double const, 1, 1> const& parameters
) : GlobalDetectorModel(
    bbox, distortionOrder, fiberMap, dualDetector,
    parameters[ndarray::view(0, BULK + getNumDistortion(distortionOrder))].shallow(),
    parameters[ndarray::view(BULK + getNumDistortion(distortionOrder),
                             BULK + getNumDistortion(distortionOrder) + fiberMap.size())].shallow(),
    parameters[ndarray::view(BULK + getNumDistortion(distortionOrder) + fiberMap.size(),
                             BULK + getNumDistortion(distortionOrder) + 2*fiberMap.size())].shallow()
) {
    assert(parameters.size() == GlobalDetectorModel::getNumParameters(distortionOrder, fiberMap.size()));
}


GlobalDetectorModel::GlobalDetectorModel(
    lsst::geom::Box2I const& bbox,
    int distortionOrder,
    ndarray::Array<int, 1, 1> const& fiberId,
    bool dualDetector,
    ndarray::Array<double const, 1, 1> const& parameters,
    ndarray::Array<double const, 1, 1> const& spatialOffsets,
    ndarray::Array<double const, 1, 1> const& spectralOffsets
) : GlobalDetectorModel(bbox, distortionOrder, makeFiberMap(fiberId), dualDetector,
                        parameters, spatialOffsets, spectralOffsets)
{}


GlobalDetectorModel::GlobalDetectorModel(
    lsst::geom::Box2I const& bbox,
    int distortionOrder,
    ndarray::Array<int, 1, 1> const& fiberId,
    bool dualDetector,
    ndarray::Array<double const, 1, 1> const& parameters,
    ndarray::Array<float, 1, 1> const& spatialOffsets,
    ndarray::Array<float, 1, 1> const& spectralOffsets
) : GlobalDetectorModel(bbox, distortionOrder, makeFiberMap(fiberId), dualDetector, parameters,
                        utils::convertArray<double>(spatialOffsets),
                        utils::convertArray<double>(spectralOffsets))
{}


GlobalDetectorModel::GlobalDetectorModel(
    lsst::geom::Box2I const& bbox,
    int distortionOrder,
    GlobalDetectorModel::FiberMap const& fiberMap,
    bool dualDetector,
    ndarray::Array<double const, 1, 1> const& parameters,
    ndarray::Array<double const, 1, 1> const& spatialOffsets,
    ndarray::Array<double const, 1, 1> const& spectralOffsets
) : _bbox(bbox),
    _distortionOrder(distortionOrder),
    _fiberMap(fiberMap),
    _dualDetector(dualDetector),
    _parameters(parameters),
    _xDistortion(distortionOrder),
    _yDistortion(distortionOrder),
    _spatialOffsets(spatialOffsets),
    _spectralOffsets(spectralOffsets),
    _cosCcdRotation(std::cos(getCcdRotation())),
    _sinCcdRotation(std::sin(getCcdRotation()))
{
    std::size_t const numDistortion = getNumDistortion(distortionOrder);
    assert(parameters.size() == BULK + numDistortion);
    assert(spatialOffsets.size() == fiberMap.size());
    assert(spectralOffsets.size() == fiberMap.size());

    int const fiberMin = std::min_element(fiberMap.begin(), fiberMap.end())->first;
    int const fiberMax = std::max_element(fiberMap.begin(), fiberMap.end())->first;
    float const xiMin = getFiberPitch()*(fiberMin - getFiberCenter() - 1);
    float const xiMax = getFiberPitch()*(fiberMax - getFiberCenter() + 1);
    float const etaMin = bbox.getMinY() - getY0();
    float const etaMax = bbox.getMaxY() - getY0();
    lsst::geom::Box2D range{lsst::geom::Point2D(xiMin, etaMin), lsst::geom::Point2D(xiMax, etaMax)};
    _xDistortion = Polynomial(distortionOrder, range);
    _yDistortion = Polynomial(distortionOrder, range);
    _xDistortion.setParameters(getPolynomialCoefficients(BULK, numDistortion/2));
    _yDistortion.setParameters(getPolynomialCoefficients(BULK + numDistortion/2, numDistortion/2));
}


ndarray::Array<int, 1, 1> GlobalDetectorModel::getFiberId() const {
    ndarray::Array<int, 1, 1> fiberId = ndarray::allocate(getNumFibers());
    for (auto & ff : _fiberMap) {
        fiberId[ff.second] = ff.first;
    }
    return fiberId;
}


lsst::geom::Point2D GlobalDetectorModel::operator()(
    int fiberId,
    double wavelength,
    std::size_t fiberIndex
) const {
    // xi,eta: position relative to the slit center
    double const xi = getFiberPitch()*(fiberId - getFiberCenter()) + getSpatialOffset(fiberIndex);
    double const eta = (wavelength - getWavelengthCenter())/getDispersion() + getSpectralOffset(fiberIndex);
    // xiPrime,etaPrime: distorted position relative the slit center
    double const xiPrime = xi + getXDistortion()(xi, eta);
    double const etaPrime = eta + getYDistortion()(xi, eta);
    // x,y: detector coordinates
    double x = xiPrime + getX0();
    double y = etaPrime + getY0();
    if (x >= getXCenter()) {
        if (getDualDetector()) {
            // Right CCD: rotation+offset
            x = getCosCcdRotation()*xiPrime - getSinCcdRotation()*etaPrime + getX0() + getXGap();
            y = getSinCcdRotation()*xiPrime + getCosCcdRotation()*etaPrime + getY0() + getYGap();
        } else {
            // Apply the x gap without any rotation
            // I believe there's a fiber numbering discontinuity in the middle of the slit, which
            // translates into a negative xGap even when there's no physical chip gap.
            x += getXGap();
        }
    }
    return lsst::geom::Point2D(x, y);
}


ndarray::Array<double, 2, 1> GlobalDetectorModel::operator()(
    ndarray::Array<int, 1, 1> const& fiberId,
    ndarray::Array<double, 1, 1> const& wavelength
) const {
    std::size_t length = fiberId.size();
    utils::checkSize(wavelength.size(), length, "wavelength");
    ndarray::Array<double, 2, 1> out = ndarray::allocate(2, length);
    for (std::size_t ii = 0; ii < length; ++ii) {
        auto const point = operator()(fiberId[ii], wavelength[ii], _fiberMap.at(fiberId[ii]));
        out[0][ii] = point.getX();
        out[1][ii] = point.getY();
    }
    return out;
}


ndarray::Array<double, 2, 1> GlobalDetectorModel::operator()(
    ndarray::Array<int, 1, 1> const& fiberId,
    ndarray::Array<double, 1, 1> const& wavelength,
    ndarray::Array<std::size_t, 1, 1> const& fiberIndex
) const {
    std::size_t length = fiberId.size();
    utils::checkSize(wavelength.size(), length, "wavelength");
    utils::checkSize(fiberIndex.size(), length, "fiberIndex");
    ndarray::Array<double, 2, 1> out = ndarray::allocate(2, length);
    for (std::size_t ii = 0; ii < length; ++ii) {
        auto const point = operator()(fiberId[ii], wavelength[ii], fiberIndex[ii]);
        out[0][ii] = point.getX();
        out[1][ii] = point.getY();
    }
    return out;
}


ndarray::Array<double const, 1, 1> const GlobalDetectorModel::getXCoefficients() const {
    std::size_t const num = Polynomial::nParametersFromOrder(_distortionOrder);
    return _parameters[ndarray::view(BULK, BULK + num)].shallow();
}


ndarray::Array<double const, 1, 1> const GlobalDetectorModel::getYCoefficients() const {
    std::size_t const num = Polynomial::nParametersFromOrder(_distortionOrder);
    return _parameters[ndarray::view(BULK + num, BULK + 2*num)].shallow();
}


std::vector<double> GlobalDetectorModel::makeParameters(
    int distortionOrder,
    std::size_t numFibers,
    double fiberCenter,
    double fiberPitch,
    double wavelengthCenter,
    double dispersion,
    double ccdRotation,
    double x0,
    double y0,
    double xGap,
    double yGap,
    ndarray::Array<double, 1, 1> const& distortionCoeff,
    ndarray::Array<double, 1, 1> const& spatialOffsets,
    ndarray::Array<double, 1, 1> const& spectralOffsets
) {
    std::size_t const num = getNumParameters(distortionOrder, numFibers);
    std::vector<double> parameters(num, 0.0);
    parameters[FIBER_CENTER] = fiberCenter;
    parameters[FIBER_PITCH] = fiberPitch;
    parameters[WAVELENGTH_CENTER] = wavelengthCenter;
    parameters[DISPERSION] = dispersion;
    parameters[CCD_ROTATION] = ccdRotation;
    parameters[X0] = x0;
    parameters[Y0] = y0;
    parameters[X_GAP] = xGap;
    parameters[Y_GAP] = yGap;

    if (!distortionCoeff.isEmpty()) {
        utils::checkSize(distortionCoeff.size(), getNumDistortion(distortionOrder), "distortionCoeff");
        std::copy(distortionCoeff.begin(), distortionCoeff.end(), parameters.begin() + BULK);
    }
    if (!spatialOffsets.isEmpty()) {
        utils::checkSize(spatialOffsets.size(), numFibers, "spatialOffsets");
        std::copy(spatialOffsets.begin(), spatialOffsets.end(),
                  parameters.begin() + BULK + getNumDistortion(distortionOrder));
    }
    if (!spectralOffsets.isEmpty()) {
        utils::checkSize(spectralOffsets.size(), numFibers, "spectralOffsets");
        std::copy(spectralOffsets.begin(), spectralOffsets.end(),
                  parameters.begin() + BULK + getNumDistortion(distortionOrder) + numFibers);
    }

    return parameters;
}


std::vector<double> GlobalDetectorModel::getPolynomialCoefficients(std::size_t start, std::size_t num) {
    std::vector<double> coeff;
    coeff.reserve(num);
    std::size_t const stop = start + num;
    for (std::size_t ii = start; ii < stop; ++ii) {
        coeff.push_back(_parameters[ii]);
    }
    return coeff;
}


std::ostream& operator<<(std::ostream& os, GlobalDetectorModel const& model) {
    os << "GlobalDetectorModel(";
    os << "fiberCenter=" << model.getFiberCenter() << ", ";
    os << "fiberPitch=" << model.getFiberPitch() << ", ";
    os << "wavelengthCenter=" << model.getWavelengthCenter() << ", ";
    os << "dispersion=" << model.getDispersion() << ", ";
    os << "ccdRotation=" << model.getCcdRotation() << ", ";
    os << "xy0=(" << model.getX0() << "," << model.getY0() << "), ";
    os << "xyGap=(" << model.getXGap() << "," << model.getYGap() << "), ";
    os << "xDistortion=" << model.getXCoefficients() << ", ";
    os << "yDistortion=" << model.getYCoefficients() << ", ";
    os << "spatialOffsets=" << model.getSpatialOffsets() << ", ";
    os << "spectralOffsets=" << model.getSpectralOffsets() << ")";
    return os;
}


namespace {

// Minuit2 chi^2 evaluation, for fitting a GlobalDetectorModel
class MinimizationFunction : public ROOT::Minuit2::FCNBase {
  public:
    using FiberIds = ndarray::Array<int, 1, 1>;
    using Array = ndarray::Array<double, 1, 1>;
    using Indices = ndarray::Array<std::size_t, 1, 1>;

    /// Ctor
    ///
    /// @param bbox : detector bounding box
    /// @param distortionOrder : polynomial order for distortion
    /// @param dualDetector : detector is two individual CCDs?
    /// @param fiberId : fiberId values for arc lines
    /// @param wavelength : wavelength values for arc lines
    /// @param xx : x values for arc lines
    /// @param yy : y values for arc lines
    /// @param xErr : error in x values for arc lines
    /// @param yErr : error in y values for arc lines
    /// @param fiberIndex : index for fiber
    /// @param good : use value in fit?
    explicit MinimizationFunction(
        lsst::geom::Box2I const& bbox,
        int distortionOrder,
        bool dualDetector,
        FiberIds const& fiberId,
        Array const& wavelength,
        Array const& xx,
        Array const& yy,
        Array const& xErr,
        Array const& yErr,
        Indices const& fiberIndex,
        ndarray::Array<bool, 1, 1> const& good
    ) : _num(fiberId.getNumElements()),
        _bbox(bbox),
        _distortionOrder(distortionOrder),
        _dualDetector(dualDetector),
        _fiberId(fiberId),
        _wavelength(wavelength),
        _xx(xx),
        _yy(yy),
        _xErr(xErr),
        _yErr(yErr),
        _fiberIndex(fiberIndex),
        _good(good)
    {
        assert(fiberId.getNumElements() == _num);
        assert(wavelength.getNumElements() == _num);
        assert(xx.getNumElements() == _num);
        assert(yy.getNumElements() == _num);
        assert(xErr.getNumElements() == _num);
        assert(yErr.getNumElements() == _num);
        assert(fiberIndex.getNumElements() == _num);
        assert(good.getNumElements() == _num);

        std::set<int> unique(fiberId.begin(), fiberId.end());
        _uniqueFiberIds = ndarray::allocate(unique.size());
        std::copy(unique.begin(), unique.end(), _uniqueFiberIds.begin());
    }

    MinimizationFunction(MinimizationFunction const &) = default;
    MinimizationFunction(MinimizationFunction &&) = default;
    MinimizationFunction &operator=(MinimizationFunction const &) = default;
    MinimizationFunction &operator=(MinimizationFunction &&) = default;
    ~MinimizationFunction() override = default;

    double operator()(std::vector<double> const& parameters) const override;
    double Up() const override { return 1.0; }  // 1.0 --> fitting chi^2

  private:
    std::size_t const _num;
    lsst::geom::Box2I _bbox;
    int const _distortionOrder;
    ndarray::Array<int, 1, 1> _uniqueFiberIds;
    bool _dualDetector;
    FiberIds const& _fiberId;
    Array const& _wavelength;
    Array const& _xx;
    Array const& _yy;
    Array const& _xErr;
    Array const& _yErr;
    Indices const& _fiberIndex;
    ndarray::Array<bool, 1, 1> const& _good;
};


/// Evaluate chi^2 given model parameters
double MinimizationFunction::operator()(std::vector<double> const& parameters) const {
    GlobalDetectorModel model(_bbox, _distortionOrder, _uniqueFiberIds, _dualDetector, parameters, false);
    ndarray::Array<double, 2, 1> xy = model(_fiberId, _wavelength, _fiberIndex);

    double chi2 = 0.0;
    std::size_t num = 0;
    for (std::size_t ii = 0; ii < _num; ++ii) {
        if (!_good[ii]) continue;
        double const xMeas = _xx[ii];
        double const yMeas = _yy[ii];
        double const xErr = _xErr[ii];
        double const yErr = _yErr[ii];
        double const xFit = xy[0][ii];
        double const yFit = xy[1][ii];
        chi2 += std::pow((xMeas - xFit)/xErr, 2) + std::pow((yMeas - yFit)/yErr, 2);
        num += 2;  // one for x, one for y
    }
    return chi2/num;
}


/// Guess starting parameters for model fit
///
/// @param bbox : detector bounding box
/// @param distortionOrder : polynomial order for distortion
/// @param numFibers : number of fibers
/// @param fiberId : fiberId values for arc lines
/// @param wavelength : wavelength values for arc lines
/// @param xx : x values for arc lines
/// @param yy : y values for arc lines
/// @param good : whether value should be used in the fit
/// @return guess parameters
std::vector<double> guessParameters(
    lsst::geom::Box2I const& bbox,
    int distortionOrder,
    std::size_t numFibers,
    ndarray::Array<int, 1, 1> const& fiberId,
    ndarray::Array<double, 1, 1> const& wavelength,
    ndarray::Array<double, 1, 1> const& xx,
    ndarray::Array<double, 1, 1> const& yy,
    ndarray::Array<bool, 1, 1> const& good
) {
    auto const fiberFit = fitStraightLine(xx, fiberId, good);  // fiberId = slope*xx + intercept
    auto const wlFit = fitStraightLine(yy, wavelength, good);  // wavelength = slope*yy + intercept
    std::vector<double> guess = GlobalDetectorModel::makeParameters(
        distortionOrder,
        numFibers,
        std::get<3>(fiberFit),  // mean fiberId
        1.0/std::get<0>(fiberFit),  // columns/fiberId
        std::get<3>(wlFit),  // mean wavelength
        std::get<0>(wlFit),  // wavelength/row
        0.0,
        0.5*(bbox.getMinX() + bbox.getMaxX()),
        0.5*(bbox.getMinY() + bbox.getMaxY()),
        0.0,
        0.0
    );
    return guess;
}

/// Guess parameter steps for model fit
///
/// @param distortionOrder : polynomial order for distortion
/// @param numFibers : number of fibers
/// @param parameters : parameter values
/// @return guess parameter steps
std::vector<double> guessParameterSteps(
    int distortionOrder,
    std::size_t numFibers,
    std::vector<double> const& parameters
) {
    std::size_t const numDistortion = GlobalDetectorModel::getNumDistortion(distortionOrder);
    ndarray::Array<double, 1, 1> distortionStep = ndarray::allocate(numDistortion);
    distortionStep.deep() = 0.1;
    ndarray::Array<double, 1, 1> offsetStep = ndarray::allocate(numFibers);
    offsetStep.deep() = 0.1;
    std::vector<double> step = GlobalDetectorModel::makeParameters(
        distortionOrder,
        numFibers,
        1,
        0.1,
        0.01*parameters[2],
        0.01*parameters[3],
        1.0e-4,
        100.0,
        100.0,
        100.0,
        10.0,
        distortionStep,
        offsetStep,
        offsetStep
    );
    return step;
}


}  // anonymous namespace


GlobalDetectorModel GlobalDetectorModel::fit(
    lsst::geom::Box2I const& bbox,
    int distortionOrder,
    bool dualDetector,
    ndarray::Array<int, 1, 1> const& fiberId,
    ndarray::Array<double, 1, 1> const& wavelength,
    ndarray::Array<double, 1, 1> const& xx,
    ndarray::Array<double, 1, 1> const& yy,
    ndarray::Array<double, 1, 1> const& xErr,
    ndarray::Array<double, 1, 1> const& yErr,
    ndarray::Array<bool, 1, 1> const& goodOrig,
    ndarray::Array<double, 1, 1> const& parameters
) {
    std::size_t const length = fiberId.size();
    utils::checkSize(wavelength.size(), length, "wavelength");
    utils::checkSize(xx.size(), length, "x");
    utils::checkSize(yy.size(), length, "y");
    utils::checkSize(xErr.size(), length, "xErr");
    utils::checkSize(yErr.size(), length, "yErr");
    ndarray::Array<bool, 1, 1> good;
    if (good.isEmpty()) {
        good = ndarray::allocate(length);
        good.deep() = true;
    } else {
        good = goodOrig;
        utils::checkSize(good.size(), length, "good");
    }

#ifndef NDEBUG
    ROOT::Minuit2::MnPrint::SetLevel(3);
#endif

    FiberMap const fiberMap = makeFiberMap(fiberId);
    ndarray::Array<std::size_t, 1, 1> fiberIndex = ndarray::allocate(length);
    for (std::size_t ii = 0; ii < length; ++ii) {
        fiberIndex[ii] = fiberMap.at(fiberId[ii]);
    }
    std::size_t const numFibers = fiberMap.size();

    // Specify the strategy for getting a good minimum.
    // For each minimisation iteration, we'll fit the specified parameters only.
    // General strategy: fit a set of parameters by itself, then put them in the pot with everything else.
    // Parameters are grouped into sets of mostly-orthogonal parameters, and we work from simpler to
    // more complex.
    // Fitting a set of parameters that are degenerate will cause the fit to fail to converge.
    std::vector<std::vector<std::size_t>> fitParams = {
        {FIBER_CENTER, FIBER_PITCH, WAVELENGTH_CENTER, DISPERSION},
        {LINEAR},
        {FIBER_CENTER, FIBER_PITCH, WAVELENGTH_CENTER, DISPERSION, LINEAR},
        {X_GAP, Y_GAP},
        {FIBER_CENTER, FIBER_PITCH, WAVELENGTH_CENTER, DISPERSION, X_GAP, Y_GAP, LINEAR},
        {DISTORTION},
        {FIBER_CENTER, FIBER_PITCH, WAVELENGTH_CENTER, DISPERSION, X_GAP, Y_GAP, LINEAR, DISTORTION},
        {CCD_ROTATION, X0, Y0},
        {FIBER_CENTER, FIBER_PITCH, WAVELENGTH_CENTER, DISPERSION, X_GAP, Y_GAP,
         CCD_ROTATION, X0, Y0, LINEAR, DISTORTION},
        {OFFSETS},
    };

    std::vector<double> guess;
    if (parameters.isEmpty()) {
        guess = guessParameters(bbox, distortionOrder, numFibers, fiberId, wavelength, xx, yy, good);
    } else {
        utils::checkSize(parameters.size(), BULK + getNumDistortion(distortionOrder), "parameters");
        guess.resize(getNumParameters(distortionOrder, numFibers), 0.0);
        std::copy(parameters.begin(), parameters.end(), guess.begin());
    }
    std::vector<double> step = guessParameterSteps(distortionOrder, numFibers, guess);
    std::size_t const numParams = guess.size();
    std::size_t const numDistortion = getNumDistortion(distortionOrder);

#ifndef NDEBUG
        std::cerr << "Starting point: ";
        std::cerr << GlobalDetectorModel(bbox, distortionOrder, fiberMap, dualDetector, guess) << std::endl;
#endif

    MinimizationFunction func{bbox, distortionOrder, dualDetector,
                              fiberId, wavelength, xx, yy, xErr, yErr, fiberIndex, good};
    auto minimizer = ROOT::Minuit2::MnMigrad(func, guess, step);

    for (std::size_t iter = 0; iter < fitParams.size(); ++iter) {
        // Fix all parameters
        for (std::size_t ii = 0; ii < numParams; ++ii) {
            minimizer.Fix(ii);
        }

        // Release designated parameters
        for (auto const ff : fitParams[iter]) {
            switch (ff) {
              case FIBER_CENTER:
              case FIBER_PITCH:
              case WAVELENGTH_CENTER:
              case DISPERSION:
              case CCD_ROTATION:
              case X0:
              case Y0:
              case X_GAP:
              case Y_GAP:
                minimizer.Release(ff);
                break;
              case LINEAR:
                for (std::size_t ii = 0; ii < 3; ++ii) {
                    minimizer.Release(BULK + ii);
                    minimizer.Release(BULK + numDistortion/2 + ii);
                }
                break;
              case DISTORTION:
                for (std::size_t ii = BULK + 3; ii < BULK + numDistortion/2; ++ii) {
                    minimizer.Release(ii);
                }
                for (std::size_t ii = BULK + numDistortion/2 + 3; ii < BULK + numDistortion; ++ii) {
                    minimizer.Release(ii);
                }
                break;
              case OFFSETS:
                for (std::size_t ii = BULK + numDistortion; ii < numParams; ++ii) {
                    minimizer.Release(ii);
                }
                break;
              default:
                std::cerr << "Unrecognised parameter specified to fix" << std::endl;
                std::abort();
            }
        }

        // Do the minimization
        auto const min = minimizer();
#ifndef NDEBUG
        std::cerr << "Iteration " << iter << ": ";
        std::cerr << GlobalDetectorModel(bbox, distortionOrder, fiberMap, dualDetector,
                                         min.UserParameters().Params()) << std::endl;
        std::cerr << min.HasValidParameters() << min.HasValidCovariance() << min.HasAccurateCovar() << min.HasPosDefCovar() << min.HasMadePosDefCovar() << min.HesseFailed() << min.HasCovariance() << min.IsAboveMaxEdm() << min.HasReachedCallLimit() << std::endl;
#endif
        if (!min.HasValidParameters() || min.IsAboveMaxEdm() || min.HasReachedCallLimit() ||
            !std::isfinite(min.Fval())) {
            throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                              (boost::format("Minimisation iteration %d failed.") % iter).str());
        }
    }
#if 1
    // Release everything and do a final fit
    for (std::size_t ii = 0; ii < numParams; ++ii) {
        minimizer.Release(ii);
    }

    auto const min = minimizer();
#ifndef NDEBUG
        std::cerr << "Final iteration: ";
        std::cerr << GlobalDetectorModel(bbox, distortionOrder, fiberMap, dualDetector,
                                         min.UserParameters().Params()) << std::endl;
        std::cerr << min.HasValidParameters() << min.HasValidCovariance() << min.HasAccurateCovar() << min.HasPosDefCovar() << min.HasMadePosDefCovar() << min.HesseFailed() << min.HasCovariance() << min.IsAboveMaxEdm() << min.HasReachedCallLimit() << std::endl;
#endif
        if (!min.HasValidParameters() || !std::isfinite(min.Fval())) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Final fit failed");
    }
#endif
    std::vector<double> finalVector = minimizer.Parameters().Params();
    ndarray::Array<double, 1, 1> finalArray = ndarray::external(finalVector.data(),
                                                                ndarray::makeVector(finalVector.size()));
    GlobalDetectorModel model(bbox, distortionOrder, fiberMap, dualDetector,
                              ndarray::copy(finalArray));
    return model;
}


}}}  // namespace pfs::drp::stella
