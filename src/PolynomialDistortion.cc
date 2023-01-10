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
    if (!getRange().contains(point)) {
        double const nan = std::numeric_limits<double>::quiet_NaN();
        return lsst::geom::Point2D(nan, nan);
        throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeError,
                          (boost::format("x,y=(%f,%f) is out of range of %s") %
                           xx % yy % getRange()).str());
    }
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
    // @param length : Number of points that will be added.
    FitData(lsst::geom::Box2D const& range, int order, std::size_t length_) :
        poly(order, range),
        length(length_),
        xMeas(ndarray::allocate(length)),
        yMeas(ndarray::allocate(length)),
        xErr(ndarray::allocate(length)),
        yErr(ndarray::allocate(length)),
        design(ndarray::allocate(length_, poly.getNParameters())),
        index(0)
        {}

    // Add a point to the design matrix
    //
    // @param xy : Point at which to evaluate the polynomial
    // @param meas : Measured value
    // @param err : Error in measured value
    void add(lsst::geom::Point2D const& xy, lsst::geom::Point2D const& meas, lsst::geom::Point2D const& err) {
        std::size_t const ii = index++;
        assert(ii < length);

        auto const terms = poly.getDFuncDParameters(xy.getX(), xy.getY());
        assert (terms.size() == poly.getNParameters());
        std::copy(terms.begin(), terms.end(), design[ii].begin());

        xMeas[ii] = meas.getX();
        yMeas[ii] = meas.getY();
        xErr[ii] = err.getX();
        yErr[ii] = err.getY();
    }

    // Solve the least-squares problem
    //
    // @param threshold : Threshold for truncating eigenvalues (see lsst::afw::math::LeastSquares)
    // @return Solutions in x and y.
    std::pair<Array1D, Array1D> getSolution(double threshold=1.0e-6) const {
        assert(index == length);
        return std::make_pair(
            math::solveLeastSquaresDesign(design, xMeas, xErr, threshold),
            math::solveLeastSquaresDesign(design, yMeas, yErr, threshold)
        );
    }

    PolynomialDistortion::Polynomial poly;  // Polynomial used for calculating design
    std::size_t length;  // Number of measurements
    Array1D xMeas;  // Measurements in x
    Array1D yMeas;  // Measurements in y
    Array1D xErr;  // Error in x measurement
    Array1D yErr;  // Error in y measurement
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
    bool fitStatic,
    double threshold
) {
    using Array1D = PolynomialDistortion::Array1D;
    using Array2D = PolynomialDistortion::Array2D;
    std::size_t const length = xx.size();
    utils::checkSize(yy.size(), length, "y");
    utils::checkSize(xMeas.size(), length, "xMeas");
    utils::checkSize(yMeas.size(), length, "yMeas");
    utils::checkSize(xErr.size(), length, "xErr");
    utils::checkSize(yErr.size(), length, "yErr");

    FitData fit(range, distortionOrder, length);
    for (std::size_t ii = 0; ii < length; ++ii) {
        fit.add(lsst::geom::Point2D(xx[ii], yy[ii]), lsst::geom::Point2D(xMeas[ii], yMeas[ii]),
                 lsst::geom::Point2D(xErr[ii], yErr[ii]));
    }

    auto const solution = fit.getSolution(threshold);
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
