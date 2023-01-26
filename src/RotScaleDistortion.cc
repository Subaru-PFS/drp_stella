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
#include "pfs/drp/stella/math/quartiles.h"
#include "pfs/drp/stella/RotScaleDistortion.h"
#include "pfs/drp/stella/impl/Distortion.h"
#include "pfs/drp/stella/utils/math.h"
#include "pfs/drp/stella/math/quartiles.h"


namespace pfs {
namespace drp {
namespace stella {

RotScaleDistortion::RotScaleDistortion(
    lsst::geom::Box2D const& range,
    RotScaleDistortion::Array1D && parameters
) : _range(range),
    _params(parameters)
{
    utils::checkSize(parameters.size(), getNumParameters(), "parameters");
    _x0Left = 0.25*range.getWidth();
    _y0Left = 0.5*range.getHeight();
    _x0Right = 0.75*range.getWidth();
    _y0Right = 0.5*range.getHeight();
    double const cosRotLeft = std::cos(parameters[ROT_LEFT]);
    _xTermLeft = cosRotLeft - 1 + parameters[XSCALE_LEFT];
    _yTermLeft = cosRotLeft - 1 + parameters[YSCALE_LEFT];
    _sinRotLeft = std::sin(parameters[ROT_LEFT]);
    double const cosRotRight = std::cos(parameters[ROT_RIGHT]);
    _xTermRight = cosRotRight - 1 + parameters[XSCALE_RIGHT];
    _yTermRight = cosRotRight - 1 + parameters[YSCALE_RIGHT];
    _sinRotRight = std::sin(parameters[ROT_RIGHT]);
}


namespace {

/// Calculate the rotation and scale.
///
/// This is calculation at the heart of RotScaleDistortion.
lsst::geom::Extent2D evaluateRotScale(
    lsst::geom::Point2D const& point,
    double dx,
    double dy,
    double x0,
    double y0,
    double xTerm,
    double yTerm,
    double sinRot
) {
    double const xx = point.getX() - x0;
    double const yy = point.getY() - y0;

    double const xRotScale = xx*xTerm - yy*sinRot;
    double const yRotScale = xx*sinRot + yy*yTerm;

    return lsst::geom::Extent2D(dx + xRotScale, dy + yRotScale);
}


}  // anonymous namespace


lsst::geom::Point2D RotScaleDistortion::evaluate(
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

    return lsst::geom::Point2D(
        onRightCcd ?
            evaluateRotScale(
                point,
                _params[DX_RIGHT], _params[DY_RIGHT],
                _params[XCEN_RIGHT] + _x0Right, _params[YCEN_RIGHT] + _y0Right,
                _xTermRight, _yTermRight,
                _sinRotRight
            ) :
            evaluateRotScale(
                point,
                _params[DX_LEFT], _params[DY_LEFT],
                _params[XCEN_LEFT] + _x0Left, _params[YCEN_LEFT] + _y0Left,
                _xTermLeft, _yTermLeft,
                _sinRotLeft
            )
    );
}


std::ostream& operator<<(std::ostream& os, RotScaleDistortion const& model) {
    os << "RotScaleDistortion(" << model.getRange() << ", " << model.getParameters() << ")";
    return os;
}


namespace {

/// Minimization function for fitting supplementary distortion
class MinimizationFunction : public ROOT::Minuit2::FCNBase {
  public:

    /// Ctor
    ///
    /// @param range : Box enclosing all x,y coordinates
    /// @param xx : Expected x coordinates
    /// @param yy : Expected y coordinates
    /// @param xMeas : Measured x coordinates
    /// @param yMeas : Measured y coordinates
    /// @param xErr : Error in measured x coordinates
    /// @param yErr : Error in measured y coordinates
    /// @param useForWavelength : Whether to use this point for wavelength (y) fit
    explicit MinimizationFunction(
        lsst::geom::Box2D const& range,
        ndarray::Array<double, 1, 1> const& xx,
        ndarray::Array<double, 1, 1> const& yy,
        ndarray::Array<double, 1, 1> const& xMeas,
        ndarray::Array<double, 1, 1> const& yMeas,
        ndarray::Array<double, 1, 1> const& xErr,
        ndarray::Array<double, 1, 1> const& yErr,
        ndarray::Array<bool, 1, 1> const& useForWavelength
    ) : _range(range),
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
    std::size_t getNumFitParameters() const { return 14; }

    /// Construct a RotScaleDistortion from the fit parameters
    RotScaleDistortion makeDistortion(std::vector<double> const& parameters) const;

    double operator()(std::vector<double> const& parameters) const override;
    double Up() const override { return 1.0; }  // 1.0 --> fitting chi^2

  private:
    lsst::geom::Box2D _range;  // Box enclosing all x,y coordinates
    std::size_t _numData;  // Number of data points
    ndarray::Array<double, 1, 1> const& _x;  // Expected x coordinates
    ndarray::Array<double, 1, 1> const& _y;  // Expected y coordinates
    ndarray::Array<double, 1, 1> const& _xMeas;  // Measured distortion in x coordinates
    ndarray::Array<double, 1, 1> const& _yMeas;  // Measured distortion in y coordinates
    ndarray::Array<double, 1, 1> const& _xErr;  // Error in measured x distortion
    ndarray::Array<double, 1, 1> const& _yErr;  // Error in measured y distortion
    ndarray::Array<bool, 1, 1> const& _useForWavelength;  // Whether to use this point for wavelength (y) fit
};


RotScaleDistortion MinimizationFunction::makeDistortion(std::vector<double> const& parameters) const {
    ndarray::Array<double, 1, 1> params = ndarray::copy(utils::vectorToArray(parameters));
    return RotScaleDistortion(_range, params);
}


double MinimizationFunction::operator()(std::vector<double> const& parameters) const {
    RotScaleDistortion const distortion = makeDistortion(parameters);
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
#if 0
    std::cerr << distortion.getParameters() << " --> " << chi2 << std::endl;
#endif
    return chi2;
}


}  // anonymous namespace


RotScaleDistortion RotScaleDistortion::fit(
    lsst::geom::Box2D const& range,
    ndarray::Array<double, 1, 1> const& xx,
    ndarray::Array<double, 1, 1> const& yy,
    ndarray::Array<double, 1, 1> const& xMeas,
    ndarray::Array<double, 1, 1> const& yMeas,
    ndarray::Array<double, 1, 1> const& xErr,
    ndarray::Array<double, 1, 1> const& yErr,
    ndarray::Array<bool, 1, 1> const& useForWavelength,
    std::size_t maxFuncCalls
) {
    using Array1D = RotScaleDistortion::Array1D;
    using Array2D = RotScaleDistortion::Array2D;
    std::size_t const length = xx.size();
    utils::checkSize(yy.size(), length, "y");
    utils::checkSize(xMeas.size(), length, "xMeas");
    utils::checkSize(yMeas.size(), length, "yMeas");
    utils::checkSize(xErr.size(), length, "xErr");
    utils::checkSize(yErr.size(), length, "yErr");
    utils::checkSize(useForWavelength.getNumElements(), length, "useForWavelength");

    bool const haveWavelength = std::any_of(useForWavelength.begin(), useForWavelength.end(),
                                            [](bool value) { return value; });

    MinimizationFunction func(range, xx, yy, xMeas, yMeas, xErr, yErr, useForWavelength);
    std::vector<double> parameters(func.getNumFitParameters(), 0.0);
    std::vector<double> steps(func.getNumFitParameters(), 0.1);

    auto const xQuartiles = math::calculateQuartiles(xMeas);
    double const xMedian = std::get<1>(xQuartiles);
    double const xStdev = 0.741*(std::get<2>(xQuartiles) - std::get<0>(xQuartiles));
    parameters[DX_LEFT] = -xMedian;
    parameters[DX_RIGHT] = -xMedian;
    steps[DX_LEFT] = std::max(0.1*xStdev, 1.0);
    steps[DX_RIGHT] = std::max(0.1*xStdev, 1.0);

    steps[XCEN_LEFT] = 0.1*range.getWidth();
    steps[YCEN_LEFT] = 0.1*range.getHeight();
    steps[XCEN_RIGHT] = 0.1*range.getWidth();
    steps[YCEN_RIGHT] = 0.1*range.getHeight();
    steps[ROT_LEFT] = 1.0e-4;
    steps[ROT_RIGHT] = 1.0e-4;
    steps[XSCALE_LEFT] = 1.0e-6;
    steps[YSCALE_LEFT] = 1.0e-6;
    steps[XSCALE_RIGHT] = 1.0e-6;
    steps[YSCALE_RIGHT] = 1.0e-6;

    if (haveWavelength) {
        auto const yQuartiles = math::calculateQuartiles<true>(yMeas, useForWavelength);
        double const yMedian = std::get<1>(yQuartiles);
        double const yStdev = 0.741*(std::get<2>(yQuartiles) - std::get<0>(yQuartiles));
        parameters[DY_LEFT] = -yMedian;
        parameters[DY_RIGHT] = -yMedian;
        steps[DY_LEFT] = std::max(0.1*yStdev, 1.0);
        steps[DY_RIGHT] = std::max(0.1*yStdev, 1.0);
    } else {
        steps[DY_LEFT] = steps[DX_LEFT];
        steps[DY_RIGHT] = steps[DX_RIGHT];
    }

#if 0
    std::cerr << "Number of points: " << length << std::endl;
    std::cerr << "Start: " << func.makeDistortion(parameters) << std::endl;
#endif
    // Start with low precision (strategy=0), and work up
    for (int strategy = 0; strategy < 1; ++strategy) {
        ROOT::Minuit2::MnMigrad minimizer(func, parameters, steps, strategy);
        auto min = minimizer(maxFuncCalls);

#if 0
        std::cerr << "Iteration " << strategy << ": " <<
            func.makeDistortion(min.UserParameters().Params()) << std::endl;
        std::cerr << "Minimizer: " << min.IsValid() << min.HasValidParameters() << min.HasAccurateCovar() <<
            min.HasPosDefCovar() << min.HasMadePosDefCovar() << min.HesseFailed() << min.HasCovariance() <<
            min.IsAboveMaxEdm() << min.HasReachedCallLimit() << " " << min.Fval() << " " << min.Edm() <<
            " " << min.NFcn() << std::endl;
#endif

        if (!min.IsValid() || !std::isfinite(min.Fval())) {
            throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Minimization failed to converge");
        }
        auto const minParams = min.UserParameters().Params();
        std::copy(minParams.begin(), minParams.end(), parameters.begin());
    }
#if 0
    std::cerr << "Final: " << func.makeDistortion(parameters) << std::endl;
#endif

    return func.makeDistortion(parameters);
}


ndarray::Array<bool, 1, 1> RotScaleDistortion::getOnRightCcd(
    ndarray::Array<double, 1, 1> const& xx
) const {
    ndarray::Array<bool, 1, 1> out = ndarray::allocate(xx.size());
    std::transform(xx.begin(), xx.end(), out.begin(),
                   [this](double value) { return getOnRightCcd(value); });
    return out;
}


namespace {

// Singleton class that manages the persistence catalog's schema and keys
class RotScaleDistortionSchema {
    using IntArray = lsst::afw::table::Array<int>;
    using DoubleArray = lsst::afw::table::Array<double>;
  public:
    lsst::afw::table::Schema schema;
    lsst::afw::table::Box2DKey range;
    lsst::afw::table::Key<DoubleArray> parameters;
    lsst::afw::table::Key<int> visitInfo;

    static RotScaleDistortionSchema const& get() {
        static RotScaleDistortionSchema const instance;
        return instance;
    }

  private:
    RotScaleDistortionSchema()
      : schema(),
        range(lsst::afw::table::Box2DKey::addFields(schema, "range", "range of input values", "pixel")),
        parameters(schema.addField<DoubleArray>("parameters", "distortion parameters", "", 0))
        {}
};

}  // anonymous namespace


void RotScaleDistortion::write(lsst::afw::table::io::OutputArchiveHandle & handle) const {
    RotScaleDistortionSchema const &schema = RotScaleDistortionSchema::get();
    lsst::afw::table::BaseCatalog cat = handle.makeCatalog(schema.schema);
    std::shared_ptr<lsst::afw::table::BaseRecord> record = cat.addNew();
    record->set(schema.range, getRange());
    ndarray::Array<double, 1, 1> parameters = ndarray::copy(getParameters());
    record->set(schema.parameters, parameters);
    handle.saveCatalog(cat);
}


class RotScaleDistortion::Factory : public lsst::afw::table::io::PersistableFactory {
  public:
    std::shared_ptr<lsst::afw::table::io::Persistable> read(
        lsst::afw::table::io::InputArchive const& archive,
        lsst::afw::table::io::CatalogVector const& catalogs
    ) const override {
        static auto const& schema = RotScaleDistortionSchema::get();
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        lsst::afw::table::BaseRecord const& record = catalogs.front().front();
        LSST_ARCHIVE_ASSERT(record.getSchema() == schema.schema);

        lsst::geom::Box2D const range = record.get(schema.range);
        ndarray::Array<double, 1, 1> parameters = ndarray::copy(record.get(schema.parameters));

        return std::make_shared<RotScaleDistortion>(range, parameters);
    }

    Factory(std::string const& name) : lsst::afw::table::io::PersistableFactory(name) {}
};


namespace {

RotScaleDistortion::Factory registration("RotScaleDistortion");

}  // anonymous namespace


}}}  // namespace pfs::drp::stella
