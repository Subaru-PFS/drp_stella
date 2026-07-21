#include <numeric>
#include <tuple>

#include "ndarray.h"
#include "ndarray/eigen.h"

#include "lsst/afw/table.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/Persistable.cc"
#include "lsst/afw/math/LeastSquares.h"

#include "pfs/drp/stella/utils/checkSize.h"
#include "pfs/drp/stella/utils/math.h"
#include "pfs/drp/stella/PolynomialDistortion.h"
#include "pfs/drp/stella/impl/Distortion.h"
#include "pfs/drp/stella/math/solveLeastSquares.h"


namespace pfs {
namespace drp {
namespace stella {


PolynomialDistortion::PolynomialDistortion(
    int order,
    lsst::geom::Box2D const& range,
    PolynomialDistortion::Array1D const& coeff
) : PolynomialDistortion(order, range, splitCoefficients(order, coeff))
{}


PolynomialDistortion::PolynomialDistortion(
    int order,
    lsst::geom::Box2D const& range,
    PolynomialDistortion::Array1D const& xCoeff,
    PolynomialDistortion::Array1D const& yCoeff
) : AnalyticDistortion<PolynomialDistortion>(order, range, joinCoefficients(order, xCoeff, yCoeff)),
    _xPoly(xCoeff, range),
    _yPoly(yCoeff, range)
{}


template<> std::size_t AnalyticDistortion<PolynomialDistortion>::getNumParametersForOrder(int order) {
    return 2*PolynomialDistortion::getNumDistortionForOrder(order);
}


PolynomialDistortion::Array2D
PolynomialDistortion::splitCoefficients(
    int order,
    ndarray::Array<double, 1, 1> const& coeff
) {
    utils::checkSize(coeff.size(), PolynomialDistortion::getNumParametersForOrder(order), "coeff");
    std::size_t const numDistortion = PolynomialDistortion::getNumDistortionForOrder(order);
    Array2D split = ndarray::allocate(2, numDistortion);
    for (std::size_t ii = 0; ii < 2; ++ii) {
        split[ndarray::view(ii)] = coeff[ndarray::view(ii*numDistortion, (ii + 1)*numDistortion)];
    }
    return split;
}


PolynomialDistortion::Array1D PolynomialDistortion::joinCoefficients(
    int order,
    PolynomialDistortion::Array1D const& xCoeff,
    PolynomialDistortion::Array1D const& yCoeff
) {
    std::size_t const numDistortion = getNumDistortionForOrder(order);
    utils::checkSize(xCoeff.size(), numDistortion, "xCoeff");
    utils::checkSize(yCoeff.size(), numDistortion, "yCoeff");
    Array1D coeff = ndarray::allocate(2*numDistortion);
    coeff[ndarray::view(0, numDistortion)] = xCoeff;
    coeff[ndarray::view(numDistortion, 2*numDistortion)] = yCoeff;
    return coeff;
}


lsst::geom::Point2D PolynomialDistortion::evaluate(
    lsst::geom::Point2D const& point
) const {
    double const xx = point.getX();
    double const yy = point.getY();
    return lsst::geom::Point2D(getXPoly()(xx, yy), getYPoly()(xx, yy));
}


namespace {


/// Return distortion coefficients given a distortion
PolynomialDistortion::Array1D getDistortionCoefficients(PolynomialDistortion::Polynomial const& distortion) {
    return ndarray::copy(utils::vectorToArray(distortion.getParameters()));
}


}  // anonymous namespace


PolynomialDistortion::Array1D PolynomialDistortion::getXCoefficients() const {
    return getDistortionCoefficients(getXPoly());
}


PolynomialDistortion::Array1D PolynomialDistortion::getYCoefficients() const {
    return getDistortionCoefficients(getYPoly());
}


std::ostream& operator<<(std::ostream& os, PolynomialDistortion const& model) {
    os << "PolynomialDistortion(";
    os << "order=" << model.getOrder() << ", ";
    os << "range=" << model.getRange() << ", ";
    os << "xCoeff=" << model.getXCoefficients() << ", ";
    os << "yCoeff=" << model.getYCoefficients() << ")";
    return os;
}


namespace {

// Structure to aid in fitting a polynomial distortion
//
// We calculate the design matrix one point at a time and then solve.
struct FitData {
    using Array1D = ndarray::Array<double, 1, 1>;
    using Array2D = ndarray::Array<double, 2, 1>;

    // Ctor
    //
    // @param range : Box enclosing all x,y coordinates.
    // @param order : Polynomial order.
    // @param numLines_ : Number of arc line measurements.
    // @param numTrace_ : Number of binned trace measurements.
    FitData(
        lsst::geom::Box2D const& range, int order,
        std::size_t numLines_, std::size_t numTrace_
    ) :
        poly(order, range),
        numLines(numLines_),
        numTrace(numTrace_),
        length(2*numLines + 2*numTrace),
        measurements(ndarray::allocate(length)),
        errors(ndarray::allocate(length)),
        design(ndarray::allocate(length, 2*poly.getNParameters())),
        index(0) {
        design.deep() = 0.0;
    }

    // Add an arc line measurement to the design matrix
    //
    // Contributes one row for x and one row for y.
    //
    // @param xy : Point at which to evaluate the polynomial
    // @param meas : Measured residual
    // @param err : Error in measured residual
    void add(
        lsst::geom::Point2D const& xy,
        lsst::geom::Point2D const& meas,
        lsst::geom::Point2D const& err
    ) {
        std::size_t const ii = index++;
        assert(ii < length);

        auto const terms = poly.getDFuncDParameters(xy.getX(), xy.getY());
        assert (terms.size() == poly.getNParameters());

        // x part of the design matrix
        std::copy(terms.begin(), terms.end(), design[ii].begin());
        measurements[ii] = meas.getX();
        errors[ii] = err.getX();

        // y part of the design matrix (independent of x part)
        std::size_t const jj = index++;
        assert(jj < length);
        std::copy(terms.begin(), terms.end(), design[jj].begin() + poly.getNParameters());
        measurements[jj] = meas.getY();
        errors[jj] = err.getY();
    }

    // Add a binned trace measurement to the design matrix
    //
    // Contributes two rows, both in x-polynomial columns only:
    //   - one for the position constraint at the bin center
    //   - one for the slope constraint d(delta_x)/dy
    //
    // @param xy : Position at which to evaluate the constraints
    // @param posMeas : Measured x residual at bin center
    // @param posErr : Error in x residual
    // @param slopeMeas : Measured slope residual d(x_resid)/dy
    // @param slopeErr : Error in slope
    void addTrace(
        lsst::geom::Point2D const& xy,
        double posMeas, double posErr,
        double slopeMeas, double slopeErr
    ) {
        // Position row: x-polynomial only; y-polynomial columns remain zero.
        std::size_t const ii = index++;
        assert(ii < length);
        auto const terms = poly.getDFuncDParameters(xy.getX(), xy.getY());
        std::copy(terms.begin(), terms.end(), design[ii].begin());
        measurements[ii] = posMeas;
        errors[ii] = posErr;

        // Slope row: numerical derivative of x-polynomial w.r.t. y.
        std::size_t const jj = index++;
        assert(jj < length);
        auto const termsPlus = poly.getDFuncDParameters(xy.getX(), xy.getY() + 1.0);
        auto const termsMinus = poly.getDFuncDParameters(xy.getX(), xy.getY() - 1.0);
        std::size_t const numPoly = poly.getNParameters();
        for (std::size_t kk = 0; kk < numPoly; ++kk) {
            design[jj][kk] = 0.5*(termsPlus[kk] - termsMinus[kk]);
        }
        // y-polynomial columns remain zero; slope constrains delta_y negligibly (s*d(delta_y)/dy, s~0.016).
        measurements[jj] = slopeMeas;
        errors[jj] = slopeErr;
    }

    // Solve the least-squares problem
    //
    // @param threshold : Threshold for truncating eigenvalues (see lsst::afw::math::LeastSquares)
    // @return Solutions in x and y.
    std::pair<Array1D, Array1D> getSolution(
        double threshold=1.0e-6,
        ndarray::Array<bool, 1, 1> const& forced_=ndarray::Array<bool, 1, 1>(),
        ndarray::Array<double, 1, 1> const& params_=ndarray::Array<double, 1, 1>()
    ) const {
        assert(index == length);  // everything got added
        auto solution = math::solveLeastSquaresDesign(design, measurements, errors, threshold);
        return std::make_pair(
            solution[ndarray::view(0, poly.getNParameters())],
            solution[ndarray::view(poly.getNParameters(), 2*poly.getNParameters())]
        );
    }

    PolynomialDistortion::Polynomial poly;  // Polynomial used for calculating design
    std::size_t numLines;  // Number of arc line measurements
    std::size_t numTrace;  // Number of binned trace measurements
    std::size_t length;  // Total number of design matrix rows
    Array1D measurements;  // Measurements
    Array1D errors;  // Errors in measurements
    Array2D design;  // Design matrix
    std::size_t index;  // Next index to add
};


}  // anonymous namespace



template<>
PolynomialDistortion AnalyticDistortion<PolynomialDistortion>::fit(
    int distortionOrder,
    lsst::geom::Box2D const& range,
    ndarray::Array<double, 1, 1> const& xx,
    ndarray::Array<double, 1, 1> const& yy,
    ndarray::Array<double, 1, 1> const& xMeas,
    ndarray::Array<double, 1, 1> const& yMeas,
    ndarray::Array<double, 1, 1> const& xErr,
    ndarray::Array<double, 1, 1> const& yErr,
    double threshold,
    ndarray::Array<bool, 1, 1> const& forced,
    ndarray::Array<double, 1, 1> const& params,
    ndarray::Array<double, 1, 1> const& xTrace,
    ndarray::Array<double, 1, 1> const& yTrace,
    ndarray::Array<double, 1, 1> const& tracePos,
    ndarray::Array<double, 1, 1> const& tracePosErr,
    ndarray::Array<double, 1, 1> const& traceSlope,
    ndarray::Array<double, 1, 1> const& traceSlopeErr
) {
    std::size_t const numLines = xx.size();
    utils::checkSize(yy.size(), numLines, "y");
    utils::checkSize(xMeas.size(), numLines, "xMeas");
    utils::checkSize(yMeas.size(), numLines, "yMeas");
    utils::checkSize(xErr.size(), numLines, "xErr");
    utils::checkSize(yErr.size(), numLines, "yErr");

    std::size_t const numTrace = xTrace.size();
    utils::checkSize(yTrace.size(), numTrace, "yTrace");
    utils::checkSize(tracePos.size(), numTrace, "tracePos");
    utils::checkSize(tracePosErr.size(), numTrace, "tracePosErr");
    utils::checkSize(traceSlope.size(), numTrace, "traceSlope");
    utils::checkSize(traceSlopeErr.size(), numTrace, "traceSlopeErr");

    FitData fit(range, distortionOrder, numLines, numTrace);
    for (std::size_t ii = 0; ii < numLines; ++ii) {
        fit.add(
            lsst::geom::Point2D(xx[ii], yy[ii]),
            lsst::geom::Point2D(xMeas[ii], yMeas[ii]),
            lsst::geom::Point2D(xErr[ii], yErr[ii])
        );
    }
    for (std::size_t ii = 0; ii < numTrace; ++ii) {
        fit.addTrace(lsst::geom::Point2D(xTrace[ii], yTrace[ii]),
                     tracePos[ii], tracePosErr[ii], traceSlope[ii], traceSlopeErr[ii]);
    }

    auto const solution = fit.getSolution(threshold, forced, params);
    return PolynomialDistortion(distortionOrder, range, solution.first, solution.second);
}


namespace {

// Singleton class that manages the persistence catalog's schema and keys
class PolynomialDistortionSchema {
    using IntArray = lsst::afw::table::Array<int>;
    using DoubleArray = lsst::afw::table::Array<double>;
  public:
    lsst::afw::table::Schema schema;
    lsst::afw::table::Key<int> distortionOrder;
    lsst::afw::table::Box2DKey range;
    lsst::afw::table::Key<DoubleArray> coefficients;
    lsst::afw::table::Key<int> visitInfo;

    static PolynomialDistortionSchema const &get() {
        static PolynomialDistortionSchema const instance;
        return instance;
    }

  private:
    PolynomialDistortionSchema()
      : schema(),
        distortionOrder(schema.addField<int>("distortionOrder", "polynomial order for distortion", "")),
        range(lsst::afw::table::Box2DKey::addFields(schema, "range", "range of input values", "pixel")),
        coefficients(schema.addField<DoubleArray>("coefficients", "distortion coefficients", "", 0))
        {}
};

}  // anonymous namespace


void PolynomialDistortion::write(lsst::afw::table::io::OutputArchiveHandle & handle) const {
    PolynomialDistortionSchema const &schema = PolynomialDistortionSchema::get();
    lsst::afw::table::BaseCatalog cat = handle.makeCatalog(schema.schema);
    std::shared_ptr<lsst::afw::table::BaseRecord> record = cat.addNew();
    record->set(schema.distortionOrder, getOrder());
    record->set(schema.range, getRange());
    ndarray::Array<double, 1, 1> xCoeff = ndarray::copy(getCoefficients());
    record->set(schema.coefficients, xCoeff);
    handle.saveCatalog(cat);
}


class PolynomialDistortion::Factory : public lsst::afw::table::io::PersistableFactory {
  public:
    std::shared_ptr<lsst::afw::table::io::Persistable> read(
        lsst::afw::table::io::InputArchive const& archive,
        lsst::afw::table::io::CatalogVector const& catalogs
    ) const override {
        static auto const& schema = PolynomialDistortionSchema::get();
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        lsst::afw::table::BaseRecord const& record = catalogs.front().front();
        LSST_ARCHIVE_ASSERT(record.getSchema() == schema.schema);

        int const distortionOrder = record.get(schema.distortionOrder);
        lsst::geom::Box2D const range = record.get(schema.range);
        ndarray::Array<double, 1, 1> coeff = ndarray::copy(record.get(schema.coefficients));

        return std::make_shared<PolynomialDistortion>(distortionOrder, range, coeff);
    }

    Factory(std::string const& name) : lsst::afw::table::io::PersistableFactory(name) {}
};


namespace {

PolynomialDistortion::Factory registration("PolynomialDistortion");

}  // anonymous namespace


// Explicit instantiation
template class AnalyticDistortion<PolynomialDistortion>;


}}}  // namespace pfs::drp::stella
