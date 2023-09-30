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

#include "pfs/drp/stella/utils/checkSize.h"
#include "pfs/drp/stella/utils/math.h"
#include "pfs/drp/stella/math/quartiles.h"
#include "pfs/drp/stella/RotScaleDistortion.h"
#include "pfs/drp/stella/impl/Distortion.h"
#include "pfs/drp/stella/utils/math.h"
#include "pfs/drp/stella/math/quartiles.h"
#include "pfs/drp/stella/math/solveLeastSquares.h"


namespace pfs {
namespace drp {
namespace stella {

RotScaleDistortion::RotScaleDistortion(
    lsst::geom::Box2D const& range,
    RotScaleDistortion::Array1D && parameters
) : _range(range),
    _params(parameters),
    _x0(range.getCenterX()),
    _y0(range.getCenterY()),
    _norm(1.0/std::max(_x0, _y0))
{
    utils::checkSize(parameters.size(), getNumParameters(), "parameters");
    _x0 = range.getCenterX();
    _y0 = range.getCenterY();
}


lsst::geom::Point2D RotScaleDistortion::evaluate(
    lsst::geom::Point2D const& point
) const {
    if (!getRange().contains(point)) {
        double const nan = std::numeric_limits<double>::quiet_NaN();
        return lsst::geom::Point2D(nan, nan);
    }

    double const dx = _params[DX];
    double const dy = _params[DY];
    double const diag = _params[DIAG];
    double const offDiag = _params[OFFDIAG];

    double const xRel = (point.getX() - _x0)*_norm;
    double const yRel = (point.getY() - _y0)*_norm;

    double const xNew = dx + xRel*diag - yRel*offDiag;
    double const yNew = dy + xRel*offDiag + yRel*diag;
    return lsst::geom::Point2D(xNew, yNew);
}


std::ostream& operator<<(std::ostream& os, RotScaleDistortion const& model) {
    os << "RotScaleDistortion(" << model.getRange() << ", " << model.getParameters() << ")";
    return os;
}


RotScaleDistortion RotScaleDistortion::fit(
    lsst::geom::Box2D const& range,
    ndarray::Array<double, 1, 1> const& xx,
    ndarray::Array<double, 1, 1> const& yy,
    ndarray::Array<double, 1, 1> const& xMeas,
    ndarray::Array<double, 1, 1> const& yMeas,
    ndarray::Array<double, 1, 1> const& xErr,
    ndarray::Array<double, 1, 1> const& yErr,
    ndarray::Array<bool, 1, 1> const& isLine,
    ndarray::Array<double, 1, 1> const& slope,
    double threshold,
    ndarray::Array<bool, 1, 1> const& forced,
    ndarray::Array<double, 1, 1> const& params
) {
    using Array1D = RotScaleDistortion::Array1D;
    using Array2D = RotScaleDistortion::Array2D;
    std::size_t const length = xx.size();
    utils::checkSize(yy.size(), length, "y");
    utils::checkSize(xMeas.size(), length, "xMeas");
    utils::checkSize(yMeas.size(), length, "yMeas");
    utils::checkSize(xErr.size(), length, "xErr");
    utils::checkSize(yErr.size(), length, "yErr");
    utils::checkSize(slope.size(), length, "slope");
    utils::checkSize(isLine.getNumElements(), length, "isLine");

    // X = (x - xCenter)/xCenter
    // Y = (y - yCenter)/yCenter
    //
    // For a line:
    // dx = dx0 + diag * X - offDiag * Y
    // dy = dy0 + offDiag * X + diag * Y
    //
    // For a trace:
    // dx = dx0 + diag * X - offDiag * Y - slope*dy
    // dx = dx0 - slope*dy0 + diag*X - offDiag*Y - slope*offDiag*X - slope*diag*Y
    // dx = dx0 - slope*dy0 + diag*(X - slope*Y) + offDiag*(-slope*X - Y)
    //
    // diag = cos(rotAngle) - 1 + scale
    // offDiag = sin(rotAngle)

    std::size_t const numLines = std::count(isLine.begin(), isLine.end(), true);
    ndarray::Array<double, 2, 1> design = ndarray::allocate(length + numLines, 4);
    design.deep() = 0.0;
    ndarray::Array<double, 1, 1> measurements = ndarray::allocate(length + numLines);
    measurements[ndarray::view(0, length)] = xMeas;
    ndarray::Array<double, 1, 1> errors = ndarray::allocate(length + numLines);
    errors[ndarray::view(0, length)] = xErr;
    std::size_t index = length;  // Counter for lines

    double const xCenter = range.getCenterX();
    double const yCenter = range.getCenterY();
    double const norm = 1.0/std::max(xCenter, yCenter);

    for (std::size_t ii = 0; ii < length; ++ii) {
        double const xRel = (xx[ii] - xCenter)*norm;
        double const yRel = (yy[ii] - yCenter)*norm;
        design[ii][DX] = 1.0;
        design[ii][DIAG] = xRel;
        design[ii][OFFDIAG] = -yRel;

        if (isLine[ii]) {
            design[index][DY] = 1.0;
            design[index][DIAG] = yRel;
            design[index][OFFDIAG] = xRel;
            measurements[index] = yMeas[ii];
            errors[index] = yErr[ii];
            ++index;
        } else {
            design[ii][DY] = -slope[ii];
            design[ii][OFFDIAG] -= slope[ii]*xRel;
            design[ii][DIAG] -= slope[ii]*yRel;
        }
    }
    assert(index == length + numLines);

    auto solution = math::solveLeastSquaresDesign(design, measurements, errors, threshold, forced, params);

#if 0
    double const rot = std::asin(solution[OFFDIAG]*norm);
    double const scale = solution[DIAG]*norm - std::cos(rot) + 1.0;
    std::cerr << "Linear fit: rot=" << rot << ", scale=" << scale << std::endl;
#endif

    return RotScaleDistortion(range, solution);
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

RotScaleDistortion::Factory registerRotScaleDistortion("RotScaleDistortion");

}  // anonymous namespace


namespace {

/// Return the left-hand side of the box
lsst::geom::Box2D leftRange(lsst::geom::Box2D const& range) {
    return lsst::geom::Box2D(range.getMin(), lsst::geom::Point2D(range.getCenterX(), range.getMaxY()));
}


/// Return the right-hand side of the box
lsst::geom::Box2D rightRange(lsst::geom::Box2D const& range) {
    return lsst::geom::Box2D(lsst::geom::Point2D(range.getCenterX(), range.getMinY()), range.getMax());
}


}  // anonymous namespace


DoubleRotScaleDistortion::DoubleRotScaleDistortion(
    lsst::geom::Box2D const& range,
    Array1D const& parameters
) : DoubleRotScaleDistortion(range, DoubleRotScaleDistortion::splitParameters(parameters)) {}


DoubleRotScaleDistortion::DoubleRotScaleDistortion(
    lsst::geom::Box2D const& range,
    Array1D const& leftParameters,
    Array1D const& rightParameters
) : _range(range),
    _center(range.getCenterX()),
    _left(RotScaleDistortion(leftRange(range), leftParameters)),
    _right(RotScaleDistortion(rightRange(range), rightParameters)) {}


DoubleRotScaleDistortion::DoubleRotScaleDistortion(
    lsst::geom::Box2D const& range,
    Array2D const& parameters
) : DoubleRotScaleDistortion(range, parameters[0], parameters[1]) {}


DoubleRotScaleDistortion::Array2D DoubleRotScaleDistortion::splitParameters(
    ndarray::Array<double, 1, 1> const& parameters
) {
    utils::checkSize(parameters.size(), 8UL, "parameters");
    std::size_t const numDistortion = 4;
    Array2D split = ndarray::allocate(2, 4);
    for (std::size_t ii = 0; ii < 2; ++ii) {
        split[ndarray::view(ii)] = parameters[ndarray::view(ii*numDistortion, (ii + 1)*numDistortion)];
    }
    return split;
}


DoubleRotScaleDistortion::Array1D DoubleRotScaleDistortion::joinParameters(
    DoubleRotScaleDistortion::Array1D const& left,
    DoubleRotScaleDistortion::Array1D const& right
) {
    std::size_t const numDistortion = 4;
    utils::checkSize(left.size(), numDistortion, "left");
    utils::checkSize(right.size(), numDistortion, "right");
    Array1D coeff = ndarray::allocate(2*numDistortion);
    coeff[ndarray::view(0, numDistortion)] = left;
    coeff[ndarray::view(numDistortion, 2*numDistortion)] = right;
    return coeff;
}


DoubleRotScaleDistortion::Array1D DoubleRotScaleDistortion::getParameters() const {
    return joinParameters(_left.getParameters(), _right.getParameters());
}


ndarray::Array<bool, 1, 1> DoubleRotScaleDistortion::getOnRightCcd(
    ndarray::Array<double, 1, 1> const& xx
) const {
    ndarray::Array<bool, 1, 1> onRight = ndarray::allocate(xx.size());
    for (std::size_t ii = 0; ii < xx.size(); ++ii) {
        onRight[ii] = (xx[ii] > _center);
    }
    return onRight;
}


std::ostream& operator<<(std::ostream& os, DoubleRotScaleDistortion const& model) {
    os << "DoubleRotScaleDistortion(" << model.getRange() << ", " << model.getParameters() << ")";
    return os;
}


DoubleRotScaleDistortion DoubleRotScaleDistortion::fit(
    lsst::geom::Box2D const& range,
    ndarray::Array<double, 1, 1> const& xx,
    ndarray::Array<double, 1, 1> const& yy,
    ndarray::Array<double, 1, 1> const& xMeas,
    ndarray::Array<double, 1, 1> const& yMeas,
    ndarray::Array<double, 1, 1> const& xErr,
    ndarray::Array<double, 1, 1> const& yErr,
    ndarray::Array<bool, 1, 1> const& isLine,
    ndarray::Array<double, 1, 1> const& slope,
    double threshold,
    ndarray::Array<bool, 1, 1> const& forced,
    ndarray::Array<double, 1, 1> const& params
) {

    using Array1D = DoubleRotScaleDistortion::Array1D;
    using Array2D = DoubleRotScaleDistortion::Array2D;
    std::size_t const length = xx.size();
    utils::checkSize(yy.size(), length, "y");
    utils::checkSize(xMeas.size(), length, "xMeas");
    utils::checkSize(yMeas.size(), length, "yMeas");
    utils::checkSize(xErr.size(), length, "xErr");
    utils::checkSize(yErr.size(), length, "yErr");
    utils::checkSize(isLine.getNumElements(), length, "isLine");

    double const xCenter = range.getCenterX();
    ndarray::Array<bool, 1, 1> onLeftCcd = ndarray::allocate(length);
    asEigenArray(onLeftCcd) = asEigenArray(xx) < xCenter;
    ndarray::Array<bool, 1, 1> onRightCcd = ndarray::allocate(length);
    asEigenArray(onRightCcd) = asEigenArray(xx) >= xCenter;

    auto const left = RotScaleDistortion::fit(
        leftRange(range),
        utils::arraySelect(xx, onLeftCcd),
        utils::arraySelect(yy, onLeftCcd),
        utils::arraySelect(xMeas, onLeftCcd),
        utils::arraySelect(yMeas, onLeftCcd),
        utils::arraySelect(xErr, onLeftCcd),
        utils::arraySelect(yErr, onLeftCcd),
        utils::arraySelect(isLine, onLeftCcd),
        utils::arraySelect(slope, onLeftCcd),
        threshold,
        forced.isEmpty() ? forced : forced[ndarray::view(0, 4)],
        params.isEmpty() ? params : params[ndarray::view(0, 4)]
    );
    auto const right = RotScaleDistortion::fit(
        rightRange(range),
        utils::arraySelect(xx, onRightCcd),
        utils::arraySelect(yy, onRightCcd),
        utils::arraySelect(xMeas, onRightCcd),
        utils::arraySelect(yMeas, onRightCcd),
        utils::arraySelect(xErr, onRightCcd),
        utils::arraySelect(yErr, onRightCcd),
        utils::arraySelect(isLine, onRightCcd),
        utils::arraySelect(slope, onRightCcd),
        threshold,
        forced.isEmpty() ? forced : forced[ndarray::view(4, 8)],
        params.isEmpty() ? params : params[ndarray::view(4, 8)]
    );

    return DoubleRotScaleDistortion(range, left.getParameters(), right.getParameters());
}


namespace {

// Singleton class that manages the persistence catalog's schema and keys
class DoubleRotScaleDistortionSchema {
    using IntArray = lsst::afw::table::Array<int>;
    using DoubleArray = lsst::afw::table::Array<double>;
  public:
    lsst::afw::table::Schema schema;
    lsst::afw::table::Box2DKey range;
    lsst::afw::table::Key<DoubleArray> parameters;
    lsst::afw::table::Key<int> visitInfo;

    static DoubleRotScaleDistortionSchema const& get() {
        static DoubleRotScaleDistortionSchema const instance;
        return instance;
    }

  private:
    DoubleRotScaleDistortionSchema()
      : schema(),
        range(lsst::afw::table::Box2DKey::addFields(schema, "range", "range of input values", "pixel")),
        parameters(schema.addField<DoubleArray>("parameters", "distortion parameters", "", 0))
        {}
};

}  // anonymous namespace


void DoubleRotScaleDistortion::write(lsst::afw::table::io::OutputArchiveHandle & handle) const {
    DoubleRotScaleDistortionSchema const &schema = DoubleRotScaleDistortionSchema::get();
    lsst::afw::table::BaseCatalog cat = handle.makeCatalog(schema.schema);
    std::shared_ptr<lsst::afw::table::BaseRecord> record = cat.addNew();
    record->set(schema.range, getRange());
    ndarray::Array<double, 1, 1> parameters = ndarray::copy(getParameters());
    record->set(schema.parameters, parameters);
    handle.saveCatalog(cat);
}


class DoubleRotScaleDistortion::Factory : public lsst::afw::table::io::PersistableFactory {
  public:
    std::shared_ptr<lsst::afw::table::io::Persistable> read(
        lsst::afw::table::io::InputArchive const& archive,
        lsst::afw::table::io::CatalogVector const& catalogs
    ) const override {
        static auto const& schema = DoubleRotScaleDistortionSchema::get();
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        lsst::afw::table::BaseRecord const& record = catalogs.front().front();
        LSST_ARCHIVE_ASSERT(record.getSchema() == schema.schema);

        lsst::geom::Box2D const range = record.get(schema.range);
        ndarray::Array<double, 1, 1> parameters = ndarray::copy(record.get(schema.parameters));

        return std::make_shared<DoubleRotScaleDistortion>(range, parameters);
    }

    Factory(std::string const& name) : lsst::afw::table::io::PersistableFactory(name) {}
};


namespace {

DoubleRotScaleDistortion::Factory registerDoubleRotScaleDistortion("DoubleRotScaleDistortion");

}  // anonymous namespace


}}}  // namespace pfs::drp::stella
