#include <numeric>
#include <tuple>

#include "ndarray.h"
#include "ndarray/eigen.h"

#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/FCNBase.h"

#include "lsst/afw/table.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/Persistable.cc"
#include "lsst/afw/math/LeastSquares.h"

#include "pfs/drp/stella/utils/checkSize.h"
#include "pfs/drp/stella/utils/math.h"
#include "pfs/drp/stella/RotTiltDistortion.h"
#include "pfs/drp/stella/impl/Distortion.h"
#include "pfs/drp/stella/utils/math.h"


namespace pfs {
namespace drp {
namespace stella {


RotTiltDistortion::RotTiltDistortion(
    lsst::geom::Box2D const& range,
    RotTiltDistortion::Array1D const& coeff
) : _range(range),
    _params(coeff)
{
    utils::checkSize(coeff.size(), getNumParameters(), "coeff");
    // _dxLeft = coeff[DX_LEFT];
    // _dyLeft = coeff[DY_LEFT];
    // _rotLeft = coeff[ROT_LEFT];
    // _scaleLeft = coeff[SCALE_LEFT];
    // _xTiltLeft = coeff[XTILT_LEFT];
    // _yTiltLeft = coeff[YTILT_LEFT];
    // _xTiltZeroLeft = coeff[XTILTZERO_LEFT];
    // _yTiltZeroLeft = coeff[YTILTZERO_LEFT];
    // _magTiltLeft = coeff[MAGTILT_LEFT];
    // _dxRight = coeff[DX_RIGHT];
    // _dyRight = coeff[DY_RIGHT];
    // _rotRight = coeff[ROT_RIGHT];
    // _scaleRight = coeff[SCALE_RIGHT];
    // _xTiltRight = coeff[XTILT_RIGHT];
    // _yTiltRight = coeff[YTILT_RIGHT];
    // _xTiltZeroRight = coeff[XTILTZERO_RIGHT];
    // _yTiltZeroRight = coeff[YTILTZERO_RIGHT];
    // _magTiltRight = coeff[MAGTILT_RIGHT];

    _cosRotLeft = std::cos(coeff[ROT_LEFT]);
    _sinRotLeft = std::sin(coeff[ROT_LEFT]);
    _cosRotRight = std::cos(coeff[ROT_RIGHT]);
    _sinRotRight = std::sin(coeff[ROT_RIGHT]);
}


lsst::geom::Extent2D RotTiltDistortion::evaluateRot(
    lsst::geom::Point2D const& point,
    bool onRightCcd
) const {
    if (onRightCcd) {
        return evaluateRot(
            point, _params[DX_RIGHT], _params[DY_RIGHT], _cosRotRight, _sinRotRight, _params[SCALE_RIGHT]
        );
    }
    return evaluateRot(
        point, _params[DX_LEFT], _params[DY_LEFT], _cosRotLeft, _sinRotLeft, _params[SCALE_LEFT]
    );
}


lsst::geom::Extent2D RotTiltDistortion::evaluateRot(
    lsst::geom::Point2D const& point,
    double dx,
    double dy,
    double cosRot,
    double sinRot,
    double scale
) {
    double const xx = point.getX();
    double const yy = point.getY();

    double const dxRot = dx + scale*xx*cosRot - scale*yy*sinRot;
    double const dyRot = dy + scale*xx*sinRot + scale*yy*cosRot;

    return lsst::geom::Extent2D(dxRot, dyRot);
}


lsst::geom::Extent2D RotTiltDistortion::evaluateTilt(
    lsst::geom::Point2D const& point,
    bool onRightCcd
) const {
    if (onRightCcd) {
        return evaluateTilt(
            point,
            _params[XTILT_RIGHT],
            _params[YTILT_RIGHT],
            _params[XTILTZERO_RIGHT],
            _params[YTILTZERO_RIGHT],
            _params[MAGTILT_RIGHT]
        );
    }
    return evaluateTilt(
        point,
        _params[XTILT_LEFT],
        _params[YTILT_LEFT],
        _params[XTILTZERO_LEFT],
        _params[YTILTZERO_LEFT],
        _params[MAGTILT_LEFT]
    );
}


lsst::geom::Extent2D RotTiltDistortion::evaluateTilt(
    lsst::geom::Point2D const& point,
    double xTilt,
    double yTilt,
    double xTiltZero,
    double yTiltZero,
    double magTilt
) {
    double const xx = point.getX();
    double const yy = point.getY();

    double const rTilt = std::hypot((xx - xTiltZero)*xTilt, (yy - yTiltZero)*yTilt);
    double const drTilt = magTilt*rTilt - 1;
    double const dxTilt = (xx - xTiltZero)*drTilt;
    double const dyTilt = (yy - yTiltZero)*drTilt;

    return lsst::geom::Extent2D(dxTilt, dyTilt);
}


lsst::geom::Point2D RotTiltDistortion::evaluate(
    lsst::geom::Point2D const& point,
    bool onRightCcd
) const {
    double const xx = point.getX();
    double const yy = point.getY();
    if (!getRange().contains(point)) {
        double const nan = std::numeric_limits<double>::quiet_NaN();
        return lsst::geom::Point2D(nan, nan);
        throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeError,
                          (boost::format("x,y=(%f,%f) is out of range of %s") %
                           xx % yy % getRange()).str());
    }

    return lsst::geom::Point2D(evaluateRot(point, onRightCcd) + evaluateTilt(point, onRightCcd));
}

namespace {


std::ostream& operator<<(std::ostream& os, RotTiltDistortion const& model) {
    os << "RotTiltDistortion(...)";
    return os;
}


namespace {


/// Minimization function for fitting supplementary distortion
class MinimizationFunction : public ROOT::Minuit2::FCNBase {
  public:

    /// Ctor
    ///
    /// @param order : Polynomial order in x and y
    /// @param range : Box enclosing all x,y coordinates
    /// @param xx : Expected x coordinates
    /// @param yy : Expected y coordinates
    /// @param xMeas : Measured x coordinates
    /// @param yMeas : Measured y coordinates
    /// @param xErr : Error in measured x coordinates
    /// @param yErr : Error in measured y coordinates
    /// @param useForWavelength : Whether to use this point for wavelength (y) fit
    explicit MinimizationFunction(
        int order,
        lsst::geom::Box2D const& range,
        ndarray::Array<double, 1, 1> const& xx,
        ndarray::Array<double, 1, 1> const& yy,
        ndarray::Array<double, 1, 1> const& xMeas,
        ndarray::Array<double, 1, 1> const& yMeas,
        ndarray::Array<double, 1, 1> const& xErr,
        ndarray::Array<double, 1, 1> const& yErr,
        ndarray::Array<bool, 1, 1> const& useForWavelength
    ) : _order(order),
        _range(range),
        _numData(xx.getNumElements()),
        _x(xx),
        _y(yy),
        _xMeas(xMeas),
        _yMeas(yMeas),
        _xErr(xErr),
        _yErr(yErr),
        _useForWavelength(useForWavelength)
    {
        assert(yy.getNumElements() == _numData);
        assert(xMeas.getNumElements() == _numData);
        assert(yMeas.getNumElements() == _numData);
        assert(xErr.getNumElements() == _numData);
        assert(yErr.getNumElements() == _numData);
        assert(useForWavelength.getNumElements() == _numData);
    }

    MinimizationFunction(MinimizationFunction const &) = default;
    MinimizationFunction(MinimizationFunction &&) = default;
    MinimizationFunction &operator=(MinimizationFunction const &) = default;
    MinimizationFunction &operator=(MinimizationFunction &&) = default;
    ~MinimizationFunction() override = default;

    /// Get number of parameters being fit
    ///
    /// This is different from the number of parameters for the distortion,
    /// since we are fitting a common scale factor for the x/y and left/right
    /// polynomials.
    std::size_t getNumFitParameters() const { return 18; }

    /// Construct a RotTiltDistortion from the fit parameters
    RotTiltDistortion makeDistortion(std::vector<double> const& parameters) const;

    double operator()(std::vector<double> const& parameters) const override;
    double Up() const override { return 1.0; }  // 1.0 --> fitting chi^2

  private:
    int _order;  // Polynomial order
    lsst::geom::Box2D _range;  // Box enclosing all x,y coordinates
    std::size_t _numData;  // Number of data points
    ndarray::Array<double, 1, 1> const& _x;  // Expected x coordinates
    ndarray::Array<double, 1, 1> const& _y;  // Expected y coordinates
    ndarray::Array<double, 1, 1> const& _xMeas;  // Measured x coordinates
    ndarray::Array<double, 1, 1> const& _yMeas;  // Measured y coordinates
    ndarray::Array<double, 1, 1> const& _xErr;  // Error in measured x coordinates
    ndarray::Array<double, 1, 1> const& _yErr;  // Error in measured y coordinates
    ndarray::Array<bool, 1, 1> const& _useForWavelength;  // Whether to use this point for wavelength (y) fit
};


RotTiltDistortion MinimizationFunction::makeDistortion(std::vector<double> const& parameters) const {
    ndarray::Array<double, 1, 1> params = utils::vectorToArray(parameters);
    return RotTiltDistortion(_range, params);
}


double MinimizationFunction::operator()(std::vector<double> const& parameters) const {
    RotTiltDistortion const distortion = makeDistortion(parameters);
    auto const model = distortion(_x, _y);

    double chi2 = 0.0;
    for (std::size_t ii = 0; ii < _numData; ++ii) {
        double const dx = model[ii][0] - _xMeas[ii];
        chi2 += std::pow(dx/_xErr[ii], 2);
        if (_useForWavelength[ii]) {
            double const dy = model[ii][1] - _yMeas[ii];
            chi2 += std::pow(dy/_yErr[ii], 2);
        }
    }
    return chi2;
}


}  // anonymous namespace


RotTiltDistortion AnalyticDistortion<RotTiltDistortion>::fit(
    lsst::geom::Box2D const& range,
    ndarray::Array<double, 1, 1> const& xx,
    ndarray::Array<double, 1, 1> const& yy,
    ndarray::Array<double, 1, 1> const& xMeas,
    ndarray::Array<double, 1, 1> const& yMeas,
    ndarray::Array<double, 1, 1> const& xErr,
    ndarray::Array<double, 1, 1> const& yErr,
    ndarray::Array<bool, 1, 1> const& useForWavelength
) {
    using Array1D = RotTiltDistortion::Array1D;
    using Array2D = RotTiltDistortion::Array2D;
    std::size_t const length = xx.size();
    utils::checkSize(yy.size(), length, "y");
    utils::checkSize(xMeas.size(), length, "xMeas");
    utils::checkSize(yMeas.size(), length, "yMeas");
    utils::checkSize(xErr.size(), length, "xErr");
    utils::checkSize(yErr.size(), length, "yErr");
    utils::checkSize(useForWavelength.getNumElements(), length, "useForWavelength");

    double const xCenter = range.getCenterX();
    ndarray::Array<bool, 1, 1> onRightCcd = ndarray::allocate(length);
    asEigenArray(onRightCcd) = asEigenArray(xx) >= xCenter;
    std::size_t const numRight = std::accumulate(onRightCcd.begin(), onRightCcd.end(), 0UL,
                                                 [](std::size_t sum, bool count) {
                                                    return count ? sum + 1 : sum; });
    std::size_t const numLeft = length - numRight;

    if (fitStatic) {
        // Fit base distortion with no restriction on coefficients
        FitData left(leftRange(range), distortionOrder, numLeft);
        FitData right(rightRange(range), distortionOrder, numRight);
        for (std::size_t ii = 0; ii < length; ++ii) {
            FitData & data = onRightCcd[ii] ? right : left;
            data.add(lsst::geom::Point2D(xx[ii], yy[ii]), lsst::geom::Point2D(xMeas[ii], yMeas[ii]),
                    lsst::geom::Point2D(xErr[ii], yErr[ii]));
        }

        ndarray::Array<bool, 1, 1> useLeft = ndarray::allocate(numLeft);
        ndarray::Array<bool, 1, 1> useRight = ndarray::allocate(numRight);
        for (std::size_t ii = 0, iLeft = 0, iRight = 0; ii < length; ++ii) {
            if (onRightCcd[ii]) {
                useRight[iRight] = useForWavelength[ii];
                ++iRight;
            } else {
                useLeft[iLeft] = useForWavelength[ii];
                ++iLeft;
            }
        }

        auto const leftSolution = left.getSolution(useLeft, threshold);
        auto const rightSolution = right.getSolution(useRight, threshold);
        return RotTiltDistortion(distortionOrder, range, leftSolution.first, leftSolution.second,
                                rightSolution.first, rightSolution.second);
    }

    // Fit for a common scale factor
    MinimizationFunction func(distortionOrder, range, xx, yy, xMeas, yMeas, xErr, yErr, useForWavelength);
    std::vector<double> parameters(func.getNumFitParameters(), 0.0);
    std::vector<double> steps(func.getNumFitParameters(), 0.1);
    auto const min = ROOT::Minuit2::MnMigrad(func, parameters, steps)();
    if (!min.IsValid() || !std::isfinite(min.Fval())) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Minimization failed to converge");
    }
    return func.makeDistortion(min.UserParameters().Params());
}


ndarray::Array<bool, 1, 1> RotTiltDistortion::getOnRightCcd(
    ndarray::Array<double, 1, 1> const& xx
) const {
    ndarray::Array<bool, 1, 1> out = ndarray::allocate(xx.size());
    std::transform(xx.begin(), xx.end(), out.begin(),
                   [this](double value) { return getOnRightCcd(value); });
    return out;
}


namespace {

// Singleton class that manages the persistence catalog's schema and keys
class RotTiltDistortionSchema {
    using IntArray = lsst::afw::table::Array<int>;
    using DoubleArray = lsst::afw::table::Array<double>;
  public:
    lsst::afw::table::Schema schema;
    lsst::afw::table::Key<int> distortionOrder;
    lsst::afw::table::Box2DKey range;
    lsst::afw::table::Key<DoubleArray> coefficients;
    lsst::afw::table::Key<int> visitInfo;

    static RotTiltDistortionSchema const &get() {
        static RotTiltDistortionSchema const instance;
        return instance;
    }

  private:
    RotTiltDistortionSchema()
      : schema(),
        distortionOrder(schema.addField<int>("distortionOrder", "polynomial order for distortion", "")),
        range(lsst::afw::table::Box2DKey::addFields(schema, "range", "range of input values", "pixel")),
        coefficients(schema.addField<DoubleArray>("coefficients", "distortion coefficients", "", 0))
        {}
};

}  // anonymous namespace


void RotTiltDistortion::write(lsst::afw::table::io::OutputArchiveHandle & handle) const {
    RotTiltDistortionSchema const &schema = RotTiltDistortionSchema::get();
    lsst::afw::table::BaseCatalog cat = handle.makeCatalog(schema.schema);
    std::shared_ptr<lsst::afw::table::BaseRecord> record = cat.addNew();
    record->set(schema.distortionOrder, getOrder());
    record->set(schema.range, getRange());
    ndarray::Array<double, 1, 1> xCoeff = ndarray::copy(getCoefficients());
    record->set(schema.coefficients, xCoeff);
    handle.saveCatalog(cat);
}


class RotTiltDistortion::Factory : public lsst::afw::table::io::PersistableFactory {
  public:
    std::shared_ptr<lsst::afw::table::io::Persistable> read(
        lsst::afw::table::io::InputArchive const& archive,
        lsst::afw::table::io::CatalogVector const& catalogs
    ) const override {
        static auto const& schema = RotTiltDistortionSchema::get();
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        lsst::afw::table::BaseRecord const& record = catalogs.front().front();
        LSST_ARCHIVE_ASSERT(record.getSchema() == schema.schema);

        int const distortionOrder = record.get(schema.distortionOrder);
        lsst::geom::Box2D const range = record.get(schema.range);
        ndarray::Array<double, 1, 1> coeff = ndarray::copy(record.get(schema.coefficients));

        return std::make_shared<RotTiltDistortion>(distortionOrder, range, coeff);
    }

    Factory(std::string const& name) : lsst::afw::table::io::PersistableFactory(name) {}
};


namespace {

RotTiltDistortion::Factory registration("RotTiltDistortion");

}  // anonymous namespace


// Explicit instantiation
template class AnalyticDistortion<RotTiltDistortion>;


}}}  // namespace pfs::drp::stella
