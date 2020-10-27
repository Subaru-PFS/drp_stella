#include <set>
#include <numeric>
#include <algorithm>

#include "ndarray.h"
#include "ndarray/eigen.h"

#include "pfs/drp/stella/utils/checkSize.h"
#include "pfs/drp/stella/utils/math.h"
#include "pfs/drp/stella/GlobalDetectorModel.h"


namespace pfs {
namespace drp {
namespace stella {


lsst::geom::Point2D GlobalDetectorModelScaling::operator()(
    int fiberId,
    double wavelength
) const {
    double const xi = fiberPitch*fiberId;
    double const eta = (wavelength - wavelengthCenter)/dispersion;
    return lsst::geom::Point2D(xi, eta);
}


ndarray::Array<double, 2, 1> GlobalDetectorModelScaling::operator()(
    ndarray::Array<int, 1, 1> const& fiberId,
    ndarray::Array<double, 1, 1> const& wavelength
) const {
    std::size_t const length = fiberId.size();
    utils::checkSize(length, wavelength.size(), "wavelength");
    ndarray::Array<double, 2, 1> xiEta = ndarray::allocate(length, 2);
    double const invDispersion = 1.0/dispersion;
    for (std::size_t ii = 0; ii < length; ++ii) {
        double const xi = fiberPitch*fiberId[ii];
        double const eta = (wavelength[ii] - wavelengthCenter)*invDispersion;
        xiEta[ii][0] = xi;
        xiEta[ii][1] = eta;
    }
    return xiEta;
}


lsst::geom::Point2D GlobalDetectorModelScaling::inverse(
    lsst::geom::Point2D const& xiEta
) const {
    double const fiberId = xiEta.getX()/fiberPitch;
    double const wavelength = xiEta.getY()*dispersion + wavelengthCenter;
    return lsst::geom::Point2D(fiberId, wavelength);
}


lsst::geom::Box2D GlobalDetectorModelScaling::getRange() const {
    float const maxFactor = 1.0 + buffer;
    float const minFactor = 1.0 - buffer;
    double const xiMin = fiberPitch*(minFiberId - 1)*minFactor;
    double const xiMax = fiberPitch*(maxFiberId + 1)*maxFactor;
    double const etaMin = -0.5*height*maxFactor;
    double const etaMax = 0.5*height*maxFactor;
    return lsst::geom::Box2D(lsst::geom::Point2D(xiMin, etaMin), lsst::geom::Point2D(xiMax, etaMax));
}


std::ostream& operator<<(std::ostream& os, GlobalDetectorModelScaling const& model) {
    os << "GlobalDetectorModelScaling(";
    os << "fiberPitch=" << model.fiberPitch << ", ";
    os << "dispersion=" << model.dispersion << ", ";
    os << "wavelengthCenter=" << model.wavelengthCenter << ")";
    return os;
}


FiberMap::FiberMap(
    ndarray::Array<int, 1, 1> const& fiberId
) {
    std::set<int> unique(fiberId.begin(), fiberId.end());  // sorted, to ensure order is consistent
    _map.reserve(unique.size());
    std::size_t ii = 0;
    for (auto iter = unique.begin(); iter != unique.end(); ++iter, ++ii) {
        _map[*iter] = ii;
    }
}


ndarray::Array<std::size_t, 1, 1> FiberMap::operator()(
    ndarray::Array<int, 1, 1> const& fiberId
) const {
    std::size_t const length = fiberId.size();
    ndarray::Array<std::size_t, 1, 1> index = ndarray::allocate(length);
    std::transform(fiberId.begin(), fiberId.end(), index.begin(),
                   [this](int ff) { return operator()(ff); });
    return index;
}


std::ostream& operator<<(std::ostream& os, FiberMap const& fiberMap) {
    os << "FiberMap(";
    bool comma = false;
    for (auto const& ii : fiberMap._map) {
        if (comma) {
            os << ", ";
        } else {
            comma = true;
        }
        os << ii.first << ": " << ii.second;
    }
    os << ")";
    return os;
}


GlobalDetectorModel::GlobalDetectorModel(
    lsst::geom::Box2I const& bbox,
    int distortionOrder,
    ndarray::Array<int, 1, 1> const& fiberId,
    GlobalDetectorModelScaling const& scaling,
    ndarray::Array<double, 1, 1> const& xDistortion,
    ndarray::Array<double, 1, 1> const& yDistortion,
    ndarray::Array<double, 1, 1> const& rightCcd,
    ndarray::Array<float, 1, 1> const& spatialOffsets,
    ndarray::Array<float, 1, 1> const& spectralOffsets
) : GlobalDetectorModel(
        distortionOrder, FiberMap(fiberId), scaling,
        0.5*(bbox.getMinX() + bbox.getMaxX()), bbox.getHeight(),
        xDistortion, yDistortion, rightCcd, spatialOffsets, spectralOffsets)
{}


GlobalDetectorModel::GlobalDetectorModel(
    int distortionOrder,
    FiberMap const& fiberMap,
    GlobalDetectorModelScaling const& scaling,
    float xCenter,
    std::size_t height,
    ndarray::Array<double, 1, 1> const& xDistortion,
    ndarray::Array<double, 1, 1> const& yDistortion,
    ndarray::Array<double, 1, 1> const& rightCcd,
    ndarray::Array<float, 1, 1> const& spatialOffsets,
    ndarray::Array<float, 1, 1> const& spectralOffsets
) : _distortionOrder(distortionOrder),
    _fiberMap(fiberMap),
    _scaling(scaling),
    _xCenter(xCenter),
    _xDistortion(utils::arrayToVector(xDistortion), scaling.getRange()),
    _yDistortion(utils::arrayToVector(yDistortion), scaling.getRange()),
    _rightCcd(),
    _spatialOffsets(spatialOffsets),
    _spectralOffsets(spectralOffsets)
{
    std::size_t const numDistortion = getNumDistortion(distortionOrder);
    utils::checkSize(xDistortion.size(), numDistortion, "xDistortion");
    utils::checkSize(yDistortion.size(), numDistortion, "yDistortion");

    _rightCcd.setParameterVector(ndarray::asEigenArray(rightCcd));

    std::size_t const numFibers = fiberMap.size();
    if (spatialOffsets.isEmpty()) {
        _spatialOffsets = ndarray::allocate(numFibers);
        _spatialOffsets.deep() = 0.0;
    } else {
        utils::checkSize(spatialOffsets.size(), numFibers, "spatial offsets");
    }
    if (spectralOffsets.isEmpty()) {
        _spectralOffsets = ndarray::allocate(numFibers);
        _spectralOffsets.deep() = 0.0;
    } else {
        utils::checkSize(spectralOffsets.size(), numFibers, "spectral offsets");
    }
}


ndarray::Array<int, 1, 1> GlobalDetectorModel::getFiberId() const {
    ndarray::Array<int, 1, 1> fiberId = ndarray::allocate(getNumFibers());
    for (auto & ff : _fiberMap) {
        fiberId[ff.second] = ff.first;
    }
    return fiberId;
}


lsst::geom::Point2D GlobalDetectorModel::operator()(
    lsst::geom::Point2D const& xiEta,
    std::size_t fiberIndex
) const {
    double const xi = xiEta[0] + getSpatialOffset(fiberIndex);
    double const eta = xiEta[1] + getSpectralOffset(fiberIndex);
    if (!getXDistortion().getXYRange().contains(lsst::geom::Point2D(xi, eta))) {
        throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeError,
                          (boost::format("xi,eta=(%f,%f) fiberId,wavelength=(%s) is out of range of %s") %
                           xi % eta % getScaling().inverse(xiEta) % getXDistortion().getXYRange()).str());
    }
    // x,y: distorted position
    lsst::geom::Point2D xy(getXDistortion()(xi, eta), getYDistortion()(xi, eta));
    // x,y: detector coordinates
    if (xy.getX() >= getXCenter()) {
        xy += lsst::geom::Extent2D(getRightCcd()(xiEta));
    }
    return xy;
}


ndarray::Array<double, 2, 1> GlobalDetectorModel::operator()(
    ndarray::Array<double, 2, 1> const& xiEta,
    ndarray::Array<std::size_t, 1, 1> const& fiberIndex
) const {
    std::size_t const length = xiEta.getShape()[0];
    utils::checkSize(xiEta.getShape()[1], 2UL, "xiEta");
    utils::checkSize(fiberIndex.getNumElements(), length, "fiberIndex");
    ndarray::Array<double, 2, 1> out = ndarray::allocate(length, 2);
    for (std::size_t ii = 0; ii < length; ++ii) {
        auto const point = operator()(lsst::geom::Point2D(xiEta[ii][0], xiEta[ii][1]), fiberIndex[ii]);
        out[ii][0] = point.getX();
        out[ii][1] = point.getY();
    }
    return out;
}


ndarray::Array<double, 1, 1> GlobalDetectorModel::getXCoefficients() const {
    return ndarray::copy(utils::vectorToArray(getXDistortion().getParameters()));
}


ndarray::Array<double, 1, 1> GlobalDetectorModel::getYCoefficients() const {
    return ndarray::copy(utils::vectorToArray(getYDistortion().getParameters()));
}


ndarray::Array<double, 1, 1> GlobalDetectorModel::getRightCcdCoefficients() const {
    auto const parameters = _rightCcd.getParameterVector();  // An Eigen matrix
    assert(parameters.size() == 6);
    ndarray::Array<double, 1, 1> out = ndarray::allocate(parameters.size());
    ndarray::asEigenMatrix(out) = parameters;
    return out;
}


ndarray::Array<double, 1, 1> GlobalDetectorModel::makeRightCcdCoefficients(
    double x, double y,
    double xx, double xy,
    double yx, double yy
) {
    ndarray::Array<double, 1, 1> coeff = ndarray::allocate(6);
    using Parameters = lsst::geom::AffineTransform::Parameters;
    coeff[Parameters::X] = x;
    coeff[Parameters::Y] = y;
    coeff[Parameters::XX] = xx;
    coeff[Parameters::XY] = xy;
    coeff[Parameters::YX] = yx;
    coeff[Parameters::YY] = yy;
    return coeff;
}


std::ostream& operator<<(std::ostream& os, GlobalDetectorModel const& model) {
    os << "GlobalDetectorModel(";
    os << "fiberPitch=" << model.getScaling().fiberPitch << ", ";
    os << "dispersion=" << model.getScaling().dispersion << ", ";
    os << "wavelengthCenter=" << model.getScaling().wavelengthCenter << ", ";
    os << "xDistortion=" << model.getXCoefficients() << ", ";
    os << "yDistortion=" << model.getYCoefficients() << ", ";
    os << "rightCcd=" << model.getRightCcdCoefficients() << ", ";
    os << "spatialOffsets=" << model.getSpatialOffsets() << ", ";
    os << "spectralOffsets=" << model.getSpectralOffsets() << ")";
    return os;
}


ndarray::Array<double, 2, 1> GlobalDetectorModel::calculateDesignMatrix(
    int distortionOrder,
    lsst::geom::Box2D const& xiEtaRange,
    ndarray::Array<double, 2, 1> const& xiEta,
    ndarray::Array<std::size_t, 1, 1> const& fiberIndex,
    ndarray::Array<float, 1, 1> const& spatialOffsets,
    ndarray::Array<float, 1, 1> const& spectralOffsets
) {
    std::size_t const length = xiEta.getShape()[0];
    utils::checkSize(xiEta.getShape()[1], 2UL, "xiEta");
    utils::checkSize(fiberIndex.size(), length, "fiberIndex");
    std::size_t const numFibers = (spatialOffsets.isEmpty() ?
                                   (spectralOffsets.isEmpty() ? 0 : spectralOffsets.size()) :
                                    spatialOffsets.size());
    if (!spatialOffsets.isEmpty()) {
        utils::checkSize(spatialOffsets.size(), numFibers, "spatialOffsets");
    }
    if (!spectralOffsets.isEmpty()) {
        utils::checkSize(spectralOffsets.size(), numFibers, "spectralOffsets");
    }

    std::size_t const numTerms = GlobalDetectorModel::getNumDistortion(distortionOrder);
    ndarray::Array<double, 2, 1> matrix = ndarray::allocate(length, numTerms);
    Polynomial const poly(distortionOrder, xiEtaRange);
    for (std::size_t ii = 0; ii < length; ++ii) {
        std::size_t const index = fiberIndex[ii];
        if (numFibers > 0 && index >= numFibers) {
            throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeError,
                              (boost::format("fiberIndex[%d]=%d is out of range for %d fibers") %
                               ii % index % numFibers).str());
        }
        double const xi = xiEta[ii][0] + (spatialOffsets.isEmpty() ? 0.0 : spatialOffsets[index]);
        double const eta = xiEta[ii][1] + (spectralOffsets.isEmpty() ? 0.0 : spectralOffsets[index]);
        auto const terms = poly.getDFuncDParameters(xi, eta);
        std::copy(terms.begin(), terms.end(), matrix[ii].begin());
    }
    return matrix;
}


std::pair<float, std::size_t> GlobalDetectorModel::calculateChi2(
    ndarray::Array<double, 2, 1> const& xiEta,
    ndarray::Array<std::size_t, 1, 1> const& fiberIndex,
    ndarray::Array<double, 1, 1> const& xx,
    ndarray::Array<double, 1, 1> const& yy,
    ndarray::Array<double, 1, 1> const& xErr,
    ndarray::Array<double, 1, 1> const& yErr,
    ndarray::Array<bool, 1, 1> const& goodOrig,
    float sysErr
) const {
    std::size_t const length = xiEta.getShape()[0];
    utils::checkSize(xiEta.getShape()[1], 2UL, "xiEta");
    utils::checkSize(fiberIndex.size(), length, "fiberIndex");
    utils::checkSize(xx.size(), length, "x");
    utils::checkSize(yy.size(), length, "y");
    utils::checkSize(xErr.size(), length, "xErr");
    utils::checkSize(yErr.size(), length, "yErr");
    ndarray::Array<bool, 1, 1> good;
    if (goodOrig.isEmpty()) {
        good = ndarray::allocate(length);
        good.deep() = true;
    } else {
        good = goodOrig;
        utils::checkSize(good.size(), length, "good");
    }
    double const sysErr2 = std::pow(sysErr, 2);
    double chi2 = 0.0;
    std::size_t num = 0;
    for (std::size_t ii = 0; ii < length; ++ii) {
        if (!good[ii]) continue;
        double const xMeas = xx[ii];
        double const yMeas = yy[ii];
        double const xErr2 = std::pow(xErr[ii], 2) + sysErr2;
        double const yErr2 = std::pow(yErr[ii], 2) + sysErr2;
        std::size_t const index = fiberIndex[ii];
        lsst::geom::Point2D const fit = operator()(xiEta[ii][0], xiEta[ii][1], index);
        chi2 += std::pow(xMeas - fit.getX(), 2)/xErr2 + std::pow(yMeas - fit.getY(), 2)/yErr2;
        num += 2;  // one for x, one for y
    }
    std::size_t const numFitParams = 2*getNumDistortion() + 2*getNumFibers() + 6;
    std::size_t const dof = num - numFitParams;
    return std::make_pair(float(chi2), dof);
}


ndarray::Array<double, 2, 1> GlobalDetectorModel::measureSlitOffsets(
    ndarray::Array<double, 2, 1> const& xiEta,
    ndarray::Array<std::size_t, 1, 1> const& fiberIndex,
    ndarray::Array<double, 1, 1> const& xx,
    ndarray::Array<double, 1, 1> const& yy,
    ndarray::Array<double, 1, 1> const& xErr,
    ndarray::Array<double, 1, 1> const& yErr,
    ndarray::Array<bool, 1, 1> const& goodOrig
) {
    std::size_t const length = xiEta.getShape()[0];
    utils::checkSize(xiEta.getShape()[1], 2UL, "xiEta");
    utils::checkSize(fiberIndex.size(), length, "fiberIndex");
    utils::checkSize(xx.size(), length, "x");
    utils::checkSize(yy.size(), length, "y");
    utils::checkSize(xErr.size(), length, "xErr");
    utils::checkSize(yErr.size(), length, "yErr");
    ndarray::Array<bool, 1, 1> good;
    if (goodOrig.isEmpty()) {
        good = ndarray::allocate(length);
        good.deep() = true;
    } else {
        good = goodOrig;
        utils::checkSize(good.size(), length, "good");
    }
    std::size_t const numFibers = getNumFibers();

    ndarray::Array<double, 2, 1> offsets = ndarray::allocate(numFibers, 2);
    ndarray::Array<double, 2, 1> weights = ndarray::allocate(numFibers, 2);
    offsets.deep() = 0;
    weights.deep() = 0;
    for (std::size_t ii = 0; ii < length; ++ii) {
        if (!good[ii]) continue;
        std::size_t const index = fiberIndex[ii];
        if (index >= numFibers) {
            throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeError,
                              (boost::format("fiberIndex[%d]=%d is out of range for %d fibers") %
                               ii % index % numFibers).str());
        }
        auto const fit = operator()(lsst::geom::Point2D(xiEta[ii][0], xiEta[ii][1]), index);
        double const dx = fit.getX() - xx[ii];
        double const dy = fit.getY() - yy[ii];
        double const xWeight = 1.0/std::pow(xErr[ii], 2);
        double const yWeight = 1.0/std::pow(yErr[ii], 2);
        offsets[index][0] += dx*xWeight;
        weights[index][0] += xWeight;
        offsets[index][1] += dy*yWeight;
        weights[index][1] += yWeight;
    }

    for (std::size_t ii = 0; ii < numFibers; ++ii) {
        offsets[ii][0] /= weights[ii][0];
        offsets[ii][1] /= weights[ii][1];
        _spatialOffsets[ii] += offsets[ii][0];
        _spectralOffsets[ii] += offsets[ii][1];
    }

    return offsets;
}


}}}  // namespace pfs::drp::stella
