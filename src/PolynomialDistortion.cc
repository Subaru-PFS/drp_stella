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
    int xOrder,
    int yOrder,
    lsst::geom::Box2D const& range,
    PolynomialDistortion::Array1D const& coeff
) : PolynomialDistortion(xOrder, yOrder, range, splitCoefficients(xOrder, yOrder, coeff))
{}


PolynomialDistortion::PolynomialDistortion(
    int xOrder,
    int yOrder,
    lsst::geom::Box2D const& range,
    PolynomialDistortion::Array1D const& xCoeff,
    PolynomialDistortion::Array1D const& yCoeff
) : AnalyticDistortion<PolynomialDistortion>(
        xOrder, yOrder, range, joinCoefficients(xOrder, yOrder, xCoeff, yCoeff)
    ),
    _xPoly(xCoeff, range),
    _yPoly(yCoeff, range)
{}


template<> std::size_t AnalyticDistortion<PolynomialDistortion>::getNumParametersForOrder(int order) {
    return 2*PolynomialDistortion::getNumDistortionForOrder(order);
}


std::pair<ndarray::Array<double, 1, 1>, ndarray::Array<double, 1, 1>>
PolynomialDistortion::splitCoefficients(
    int xOrder,
    int yOrder,
    ndarray::Array<double, 1, 1> const& coeff
) {
    std::size_t const xNumDistortion = PolynomialDistortion::getNumDistortionForOrder(xOrder);
    std::size_t const yNumDistortion = PolynomialDistortion::getNumDistortionForOrder(yOrder);
    utils::checkSize(coeff.size(), xNumDistortion + yNumDistortion, "coeff");

    Array1D xCoeff = ndarray::copy(coeff[ndarray::view(0, xNumDistortion)]);
    Array1D yCoeff = ndarray::copy(coeff[ndarray::view(xNumDistortion, xNumDistortion + yNumDistortion)]);
    return std::make_pair(xCoeff, yCoeff);
}


PolynomialDistortion::Array1D PolynomialDistortion::joinCoefficients(
    int xOrder,
    int yOrder,
    PolynomialDistortion::Array1D const& xCoeff,
    PolynomialDistortion::Array1D const& yCoeff
) {
    std::size_t const xNumDistortion = getNumDistortionForOrder(xOrder);
    std::size_t const yNumDistortion = getNumDistortionForOrder(yOrder);
    utils::checkSize(xCoeff.size(), xNumDistortion, "xCoeff");
    utils::checkSize(yCoeff.size(), yNumDistortion, "yCoeff");
    Array1D coeff = ndarray::allocate(xNumDistortion + yNumDistortion);
    coeff[ndarray::view(0, xNumDistortion)] = xCoeff;
    coeff[ndarray::view(xNumDistortion, xNumDistortion + yNumDistortion)] = yCoeff;
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
    os << "xOrder=" << model.getXOrder() << ", ";
    os << "yOrder=" << model.getYOrder() << ", ";
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
    // @param length : Number of points that will be added.
    FitData(
        lsst::geom::Box2D const& range,
        int xOrder,
        int yOrder,
        std::size_t numLines_,
        std::size_t numTraces_
    ) :
        xPoly(xOrder, range),
        yPoly(yOrder, range),
        numLines(numLines_),
        numTraces(numTraces_),
        length(2*numLines + numTraces),
        measurements(ndarray::allocate(length)),
        errors(ndarray::allocate(length)),
        design(ndarray::allocate(length, xPoly.getNParameters() + yPoly.getNParameters())),
        index(0) {
        design.deep() = 0.0;
    }

    // Add a point to the design matrix
    //
    // @param xy : Point at which to evaluate the polynomial
    // @param meas : Measured value
    // @param err : Error in measured value
    // @param isLine : True if this is a line measurement
    // @param slope : Slope of the trace
    void add(
        lsst::geom::Point2D const& xy,
        lsst::geom::Point2D const& meas,
        lsst::geom::Point2D const& err,
        bool isLine,
        double slope
    ) {
        std::size_t const ii = index++;
        assert(ii < length);

        std::vector<double> const xTerms = xPoly.getDFuncDParameters(xy.getX(), xy.getY());
        assert (xTerms.size() == xPoly.getNParameters());
        std::vector<double> yTerms;
        if (xPoly.getOrder() == yPoly.getOrder()) {
            // If the orders are the same, we can reuse the x terms for the y part of the design matrix.
            yTerms = xTerms;
        } else {
            yTerms = yPoly.getDFuncDParameters(xy.getX(), xy.getY());
            assert (yTerms.size() == yPoly.getNParameters());
        }

        // x part of the design matrix
        std::copy(xTerms.begin(), xTerms.end(), design[ii].begin());
        measurements[ii] = meas.getX();
        errors[ii] = err.getX();

        // y part of the design matrix
        if (isLine) {
            // For a line, the y part is independent of the x part.
            std::size_t const jj = index++;
            assert(jj < length);
            std::copy(yTerms.begin(), yTerms.end(), design[jj].begin() + yPoly.getNParameters());
            measurements[jj] = meas.getY();
            errors[jj] = err.getY();
        } else {
            // For a trace, the y part is linked to the x part by the slope.
            std::size_t const start = xPoly.getNParameters();
            std::size_t const stop = start + yPoly.getNParameters();
            auto lhs = design[ii][ndarray::view(start, stop)];
            ndarray::asEigenArray(lhs) = -slope*ndarray::asEigenArray(utils::vectorToArray(yTerms));
        }
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
            solution[ndarray::view(0, xPoly.getNParameters())],
            solution[ndarray::view(xPoly.getNParameters(), xPoly.getNParameters() + yPoly.getNParameters())]
        );
    }

    PolynomialDistortion::Polynomial xPoly;  // Polynomial used for calculating design in x
    PolynomialDistortion::Polynomial yPoly;  // Polynomial used for calculating design in y
    std::size_t numLines;  // Number of lines
    std::size_t numTraces;  // Number of traces
    std::size_t length;  // Number of measurements
    Array1D measurements;  // Measurements
    Array1D errors;  // Errors in measurements
    Array2D design;  // Design matrix
    std::size_t index;  // Next index to add
};


}  // anonymous namespace



template<>
PolynomialDistortion AnalyticDistortion<PolynomialDistortion>::fit(
    int xOrder,
    int yOrder,
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
    using Array1D = PolynomialDistortion::Array1D;
    using Array2D = PolynomialDistortion::Array2D;
    std::size_t const length = xx.size();
    utils::checkSize(yy.size(), length, "y");
    utils::checkSize(xMeas.size(), length, "xMeas");
    utils::checkSize(yMeas.size(), length, "yMeas");
    utils::checkSize(xErr.size(), length, "xErr");
    utils::checkSize(yErr.size(), length, "yErr");

    std::size_t const numLines = std::count(isLine.begin(), isLine.end(), true);
    std::size_t const numTraces = length - numLines;

    FitData fit(range, xOrder, yOrder, numLines, numTraces);
    for (std::size_t ii = 0; ii < length; ++ii) {
        fit.add(
            lsst::geom::Point2D(xx[ii], yy[ii]),
            lsst::geom::Point2D(xMeas[ii], yMeas[ii]),
            lsst::geom::Point2D(xErr[ii], yErr[ii]),
            isLine[ii],
            slope[ii]
        );
    }

    auto const solution = fit.getSolution(threshold, forced, params);
    return PolynomialDistortion(xOrder, yOrder, range, solution.first, solution.second);
}


namespace {

// Singleton class that manages the persistence catalog's schema and keys
class PolynomialDistortionSchema {
    using IntArray = lsst::afw::table::Array<int>;
    using DoubleArray = lsst::afw::table::Array<double>;
  public:
    lsst::afw::table::Schema schema;
    lsst::afw::table::Key<int> xOrder;
    lsst::afw::table::Key<int> yOrder;
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
        xOrder(schema.addField<int>("xOrder", "polynomial order for x distortion", "")),
        yOrder(schema.addField<int>("yOrder", "polynomial order for y distortion", "")),
        range(lsst::afw::table::Box2DKey::addFields(schema, "range", "range of input values", "pixel")),
        coefficients(schema.addField<DoubleArray>("coefficients", "distortion coefficients", "", 0))
        {}
};

}  // anonymous namespace


void PolynomialDistortion::write(lsst::afw::table::io::OutputArchiveHandle & handle) const {
    PolynomialDistortionSchema const &schema = PolynomialDistortionSchema::get();
    lsst::afw::table::BaseCatalog cat = handle.makeCatalog(schema.schema);
    std::shared_ptr<lsst::afw::table::BaseRecord> record = cat.addNew();
    record->set(schema.xOrder, getXOrder());
    record->set(schema.yOrder, getYOrder());
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

        int const xOrder = record.get(schema.xOrder);
        int const yOrder = record.get(schema.yOrder);
        lsst::geom::Box2D const range = record.get(schema.range);
        ndarray::Array<double, 1, 1> coeff = ndarray::copy(record.get(schema.coefficients));

        return std::make_shared<PolynomialDistortion>(xOrder, yOrder, range, coeff);
    }

    Factory(std::string const& name) : lsst::afw::table::io::PersistableFactory(name) {}
};


namespace {

PolynomialDistortion::Factory registration("PolynomialDistortion");

}  // anonymous namespace


// Explicit instantiation
template class AnalyticDistortion<PolynomialDistortion>;


}}}  // namespace pfs::drp::stella
