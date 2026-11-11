#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <vector>

#include "ndarray.h"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/table.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/Persistable.cc"

#include "pfs/drp/stella/utils/checkSize.h"
#include "pfs/drp/stella/BSplineDistortion.h"


namespace pfs {
namespace drp {
namespace stella {


namespace {

/// Check that a knot vector is valid for a spline with a particular order
void checkKnots(ndarray::Array<double, 1, 1> const& knots, int order, char const* name) {
    if (knots.size() < std::size_t(2*(order + 1))) {
        std::ostringstream os;
        os << "Insufficient " << name << " knots (" << knots.size() << ") for order " << order;
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthError, os.str());
    }
    for (std::size_t ii = 1; ii < knots.size(); ++ii) {
        if (!(knots[ii] >= knots[ii - 1])) {
            std::ostringstream os;
            os << name << " knots are not non-decreasing at index " << ii;
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, os.str());
        }
    }
    if (!(knots[knots.size() - order - 1] > knots[order])) {
        std::ostringstream os;
        os << name << " knots have zero range";
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, os.str());
    }
}

}  // anonymous namespace


BSpline2D::BSpline2D(int order) : _order(order) {
    if (order < 0 || order > MAX_ORDER) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, "Unsupported spline order");
    }
}


BSpline2D::BSpline2D(
    int order,
    Array1D const& xKnots,
    Array1D const& yKnots,
    Array2D const& coeff
) : _order(order),
    _xKnots(ndarray::copy(xKnots)),
    _yKnots(ndarray::copy(yKnots)),
    _coeff(ndarray::copy(coeff))
{
    if (order < 0 || order > MAX_ORDER) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, "Unsupported spline order");
    }
    if (isEmpty()) {
        return;
    }
    checkKnots(_xKnots, order, "x");
    checkKnots(_yKnots, order, "y");
    std::size_t const numX = _xKnots.size() - order - 1;
    std::size_t const numY = _yKnots.size() - order - 1;
    if (_coeff.getShape()[0] != numX || _coeff.getShape()[1] != numY) {
        std::ostringstream os;
        os << "Coefficients shape (" << _coeff.getShape()[0] << ", " << _coeff.getShape()[1] <<
            ") doesn't match knots: expected (" << numX << ", " << numY << ")";
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthError, os.str());
    }
}


BSpline2D BSpline2D::copy() const {
    if (isEmpty()) {
        return BSpline2D(_order);
    }
    return BSpline2D(_order, _xKnots, _yKnots, _coeff);  // ctor copies arrays
}


lsst::geom::Box2D BSpline2D::getRange() const {
    if (isEmpty()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LogicError, "Empty spline has no range");
    }
    std::size_t const numX = _xKnots.size() - _order - 1;
    std::size_t const numY = _yKnots.size() - _order - 1;
    return lsst::geom::Box2D(
        lsst::geom::Point2D(_xKnots[_order], _yKnots[_order]),
        lsst::geom::Point2D(_xKnots[numX], _yKnots[numY])
    );
}


std::size_t BSpline2D::calculateBasis(Array1D const& knots, double value, double * basis) const {
    // Algorithms A2.1 (FindSpan) and A2.2 (BasisFuns) from "The NURBS Book" (Piegl & Tiller)
    std::size_t const order = _order;
    std::size_t const num = knots.size() - order - 1;  // number of basis functions
    double const low = knots[order];
    double const high = knots[num];
    double const xx = std::min(std::max(value, low), high);

    // Find the knot span: knots[span] <= xx < knots[span + 1], with order <= span < num
    std::size_t span;
    if (xx >= high) {
        span = num - 1;
    } else {
        auto const begin = knots.begin();
        span = std::upper_bound(begin + order, begin + num, xx) - begin - 1;
    }

    double left[MAX_ORDER + 1];
    double right[MAX_ORDER + 1];
    basis[0] = 1.0;
    for (std::size_t jj = 1; jj <= order; ++jj) {
        left[jj] = xx - knots[span + 1 - jj];
        right[jj] = knots[span + jj] - xx;
        double saved = 0.0;
        for (std::size_t rr = 0; rr < jj; ++rr) {
            double const denominator = right[rr + 1] + left[jj - rr];
            double const temp = denominator == 0.0 ? 0.0 : basis[rr]/denominator;
            basis[rr] = saved + right[rr + 1]*temp;
            saved = left[jj - rr]*temp;
        }
        basis[jj] = saved;
    }
    return span - order;
}


double BSpline2D::operator()(double x, double y) const {
    if (isEmpty()) {
        return 0.0;
    }
    if (!std::isfinite(x) || !std::isfinite(y)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double xBasis[MAX_ORDER + 1];
    double yBasis[MAX_ORDER + 1];
    std::size_t const xStart = calculateBasis(_xKnots, x, xBasis);
    std::size_t const yStart = calculateBasis(_yKnots, y, yBasis);
    double sum = 0.0;
    for (int ii = 0; ii <= _order; ++ii) {
        double rowSum = 0.0;
        for (int jj = 0; jj <= _order; ++jj) {
            rowSum += _coeff[xStart + ii][yStart + jj]*yBasis[jj];
        }
        sum += xBasis[ii]*rowSum;
    }
    return sum;
}


BSpline2D::Array1D BSpline2D::operator()(Array1D const& x, Array1D const& y) const {
    utils::checkSize(x.size(), y.size(), "x vs y");
    Array1D result = ndarray::allocate(x.size());
    for (std::size_t ii = 0; ii < x.size(); ++ii) {
        result[ii] = operator()(x[ii], y[ii]);
    }
    return result;
}


std::ostream& operator<<(std::ostream& os, BSpline2D const& spline) {
    os << "BSpline2D(order=" << spline.getOrder();
    if (spline.isEmpty()) {
        os << ", empty)";
        return os;
    }
    os << ", range=" << spline.getRange() << ", shape=(" << spline.getCoefficients().getShape()[0] <<
        ", " << spline.getCoefficients().getShape()[1] << "))";
    return os;
}


BSplineDistortion::BSplineDistortion(
    Array1D const& xBoundaries,
    SplineList const& xSplines,
    SplineList const& ySplines
) : _xBoundaries(ndarray::copy(xBoundaries)) {
    std::size_t const numRegions = xBoundaries.size() + 1;
    utils::checkSize(xSplines.size(), numRegions, "xSplines vs number of regions");
    utils::checkSize(ySplines.size(), numRegions, "ySplines vs number of regions");
    for (std::size_t ii = 1; ii < xBoundaries.size(); ++ii) {
        if (!(xBoundaries[ii] > xBoundaries[ii - 1])) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                              "xBoundaries are not ascending");
        }
    }
    _xSplines.reserve(numRegions);
    _ySplines.reserve(numRegions);
    for (std::size_t ii = 0; ii < numRegions; ++ii) {
        _xSplines.emplace_back(xSplines[ii].copy());
        _ySplines.emplace_back(ySplines[ii].copy());
    }
}


std::shared_ptr<Distortion> BSplineDistortion::clone() const {
    return std::make_shared<BSplineDistortion>(_xBoundaries, _xSplines, _ySplines);  // ctor copies
}


std::size_t BSplineDistortion::getRegion(double x) const {
    return std::upper_bound(_xBoundaries.begin(), _xBoundaries.end(), x) - _xBoundaries.begin();
}


std::size_t BSplineDistortion::getNumParameters() const {
    std::size_t num = 0;
    for (std::size_t ii = 0; ii < getNumRegions(); ++ii) {
        num += _xSplines[ii].getNumParameters() + _ySplines[ii].getNumParameters();
    }
    return num;
}


lsst::geom::Point2D BSplineDistortion::evaluate(lsst::geom::Point2D const& xy) const {
    double const x = xy.getX();
    double const y = xy.getY();
    if (!std::isfinite(x) || !std::isfinite(y)) {
        return lsst::geom::Point2D(std::numeric_limits<double>::quiet_NaN(),
                                   std::numeric_limits<double>::quiet_NaN());
    }
    std::size_t const region = getRegion(x);
    return lsst::geom::Point2D(_xSplines[region](x, y), _ySplines[region](x, y));
}


std::ostream& operator<<(std::ostream& os, BSplineDistortion const& model) {
    os << "BSplineDistortion(xBoundaries=" << model.getXBoundaries() << ", xSplines=[";
    for (std::size_t ii = 0; ii < model.getNumRegions(); ++ii) {
        os << (ii > 0 ? ", " : "") << model.getXSplines()[ii];
    }
    os << "], ySplines=[";
    for (std::size_t ii = 0; ii < model.getNumRegions(); ++ii) {
        os << (ii > 0 ? ", " : "") << model.getYSplines()[ii];
    }
    os << "])";
    return os;
}


namespace {

// Singleton class that manages the persistence catalog's schema and keys
//
// There are two catalogs: the first has a single record with the region
// boundaries, and the second has a record for each spline.
class BSplineDistortionSchema {
    using DoubleArray = lsst::afw::table::Array<double>;
  public:
    lsst::afw::table::Schema boundariesSchema;
    lsst::afw::table::Key<DoubleArray> xBoundaries;

    lsst::afw::table::Schema splineSchema;
    lsst::afw::table::Key<int> axis;
    lsst::afw::table::Key<int> region;
    lsst::afw::table::Key<int> order;
    lsst::afw::table::Key<int> numX;
    lsst::afw::table::Key<int> numY;
    lsst::afw::table::Key<DoubleArray> xKnots;
    lsst::afw::table::Key<DoubleArray> yKnots;
    lsst::afw::table::Key<DoubleArray> coefficients;

    static BSplineDistortionSchema const &get() {
        static BSplineDistortionSchema const instance;
        return instance;
    }

  private:
    BSplineDistortionSchema()
      : boundariesSchema(),
        xBoundaries(boundariesSchema.addField<DoubleArray>("xBoundaries", "region boundaries in x",
                                                           "pixel", 0)),
        splineSchema(),
        axis(splineSchema.addField<int>("axis", "axis of distortion (0=x, 1=y)", "")),
        region(splineSchema.addField<int>("region", "index of region", "")),
        order(splineSchema.addField<int>("order", "order of spline", "")),
        numX(splineSchema.addField<int>("numX", "number of coefficients in x", "")),
        numY(splineSchema.addField<int>("numY", "number of coefficients in y", "")),
        xKnots(splineSchema.addField<DoubleArray>("xKnots", "knots in x", "pixel", 0)),
        yKnots(splineSchema.addField<DoubleArray>("yKnots", "knots in y", "pixel", 0)),
        coefficients(splineSchema.addField<DoubleArray>("coefficients", "spline coefficients", "pixel", 0))
        {}
};

}  // anonymous namespace


void BSplineDistortion::write(lsst::afw::table::io::OutputArchiveHandle & handle) const {
    BSplineDistortionSchema const &schema = BSplineDistortionSchema::get();

    lsst::afw::table::BaseCatalog boundaries = handle.makeCatalog(schema.boundariesSchema);
    std::shared_ptr<lsst::afw::table::BaseRecord> record = boundaries.addNew();
    ndarray::Array<double, 1, 1> xBoundaries = ndarray::copy(_xBoundaries);
    record->set(schema.xBoundaries, xBoundaries);
    handle.saveCatalog(boundaries);

    lsst::afw::table::BaseCatalog splines = handle.makeCatalog(schema.splineSchema);
    for (int axis = 0; axis < 2; ++axis) {
        SplineList const& list = axis == 0 ? _xSplines : _ySplines;
        for (std::size_t region = 0; region < list.size(); ++region) {
            BSpline2D const& spline = list[region];
            std::shared_ptr<lsst::afw::table::BaseRecord> rr = splines.addNew();
            rr->set(schema.axis, axis);
            rr->set(schema.region, static_cast<int>(region));
            rr->set(schema.order, spline.getOrder());
            rr->set(schema.numX, static_cast<int>(spline.getCoefficients().getShape()[0]));
            rr->set(schema.numY, static_cast<int>(spline.getCoefficients().getShape()[1]));
            ndarray::Array<double, 1, 1> xKnots = ndarray::copy(spline.getXKnots());
            ndarray::Array<double, 1, 1> yKnots = ndarray::copy(spline.getYKnots());
            ndarray::Array<double, 1, 1> coeff = ndarray::allocate(spline.getNumParameters());
            // Flatten coefficients with y varying fastest
            Array2D const& splineCoeff = spline.getCoefficients();
            std::size_t index = 0;
            for (std::size_t ii = 0; ii < splineCoeff.getShape()[0]; ++ii) {
                for (std::size_t jj = 0; jj < splineCoeff.getShape()[1]; ++jj, ++index) {
                    coeff[index] = splineCoeff[ii][jj];
                }
            }
            rr->set(schema.xKnots, xKnots);
            rr->set(schema.yKnots, yKnots);
            rr->set(schema.coefficients, coeff);
        }
    }
    handle.saveCatalog(splines);
}


class BSplineDistortion::Factory : public lsst::afw::table::io::PersistableFactory {
  public:
    std::shared_ptr<lsst::afw::table::io::Persistable> read(
        lsst::afw::table::io::InputArchive const& archive,
        lsst::afw::table::io::CatalogVector const& catalogs
    ) const override {
        static auto const& schema = BSplineDistortionSchema::get();
        LSST_ARCHIVE_ASSERT(catalogs.size() == 2u);
        LSST_ARCHIVE_ASSERT(catalogs[0].size() == 1u);
        lsst::afw::table::BaseRecord const& boundaries = catalogs[0].front();
        LSST_ARCHIVE_ASSERT(boundaries.getSchema() == schema.boundariesSchema);
        ndarray::Array<double, 1, 1> xBoundaries = ndarray::copy(boundaries.get(schema.xBoundaries));
        std::size_t const numRegions = xBoundaries.size() + 1;

        lsst::afw::table::BaseCatalog const& splines = catalogs[1];
        LSST_ARCHIVE_ASSERT(splines.size() == 2*numRegions);
        SplineList xSplines(numRegions);
        SplineList ySplines(numRegions);
        for (auto const& rr : splines) {
            LSST_ARCHIVE_ASSERT(rr.getSchema() == schema.splineSchema);
            int const axis = rr.get(schema.axis);
            std::size_t const region = rr.get(schema.region);
            LSST_ARCHIVE_ASSERT(axis == 0 || axis == 1);
            LSST_ARCHIVE_ASSERT(region < numRegions);
            int const order = rr.get(schema.order);
            std::size_t const numX = rr.get(schema.numX);
            std::size_t const numY = rr.get(schema.numY);
            BSpline2D spline{order};
            if (numX*numY > 0) {
                ndarray::Array<double, 1, 1> xKnots = ndarray::copy(rr.get(schema.xKnots));
                ndarray::Array<double, 1, 1> yKnots = ndarray::copy(rr.get(schema.yKnots));
                auto const& flat = rr.get(schema.coefficients);
                LSST_ARCHIVE_ASSERT(flat.size() == numX*numY);
                ndarray::Array<double, 2, 1> coeff = ndarray::allocate(numX, numY);
                std::size_t index = 0;
                for (std::size_t ii = 0; ii < numX; ++ii) {
                    for (std::size_t jj = 0; jj < numY; ++jj, ++index) {
                        coeff[ii][jj] = flat[index];
                    }
                }
                spline = BSpline2D(order, xKnots, yKnots, coeff);
            }
            (axis == 0 ? xSplines : ySplines)[region] = spline;
        }
        return std::make_shared<BSplineDistortion>(xBoundaries, xSplines, ySplines);
    }

    Factory(std::string const& name) : lsst::afw::table::io::PersistableFactory(name) {}
};


namespace {

BSplineDistortion::Factory registration("BSplineDistortion");

}  // anonymous namespace


}}}  // namespace pfs::drp::stella
