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
#include "pfs/drp/stella/DoubleDistortion.h"
#include "pfs/drp/stella/impl/Distortion.h"
#include "pfs/drp/stella/math/solveLeastSquares.h"


namespace pfs {
namespace drp {
namespace stella {


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


DoubleDistortion::DoubleDistortion(
    int order,
    lsst::geom::Box2D const& range,
    DoubleDistortion::Array1D const& coeff
) : DoubleDistortion(order, range, splitCoefficients(order, coeff))
{}


DoubleDistortion::DoubleDistortion(
    int order,
    lsst::geom::Box2D const& range,
    DoubleDistortion::Array1D const& xLeft,
    DoubleDistortion::Array1D const& yLeft,
    DoubleDistortion::Array1D const& xRight,
    DoubleDistortion::Array1D const& yRight
) : AnalyticDistortion<DoubleDistortion>(
        order,
        range,
        joinCoefficients(order, xLeft, yLeft, xRight, yRight)
    ),
    _xLeft(xLeft, leftRange(range)),
    _yLeft(yLeft, leftRange(range)),
    _xRight(xRight, rightRange(range)),
    _yRight(yRight, rightRange(range))
{}


template<> std::size_t AnalyticDistortion<DoubleDistortion>::getNumParametersForOrder(int order) {
    return 4*DoubleDistortion::getNumDistortionForOrder(order);
}


DoubleDistortion::DoubleDistortion(
    int order,
    lsst::geom::Box2D const& range,
    Array2D const& coeff
) : DoubleDistortion(
    order,
    range,
    coeff[ndarray::view(0)],
    coeff[ndarray::view(1)],
    coeff[ndarray::view(2)],
    coeff[ndarray::view(3)]
) {
    utils::checkSize(coeff.getShape()[0], 4UL, "coefficients");
    utils::checkSize(coeff.getShape()[1], getNumDistortionForOrder(order), "coefficients vs order");
}


DoubleDistortion::Array2D DoubleDistortion::splitCoefficients(
    int order,
    ndarray::Array<double, 1, 1> const& coeff
) {
    utils::checkSize(coeff.size(), DoubleDistortion::getNumParametersForOrder(order), "coeff");
    std::size_t const numDistortion = DoubleDistortion::getNumDistortionForOrder(order);
    Array2D split = ndarray::allocate(4, numDistortion);
    for (std::size_t ii = 0; ii < 4; ++ii) {
        split[ndarray::view(ii)] = coeff[ndarray::view(ii*numDistortion, (ii + 1)*numDistortion)];
    }
    return split;
}


DoubleDistortion::Array1D DoubleDistortion::joinCoefficients(
    int order,
    DoubleDistortion::Array1D const& xLeft,
    DoubleDistortion::Array1D const& yLeft,
    DoubleDistortion::Array1D const& xRight,
    DoubleDistortion::Array1D const& yRight
) {
    std::size_t const numDistortion = getNumDistortionForOrder(order);
    utils::checkSize(xLeft.size(), numDistortion, "xLeft");
    utils::checkSize(yLeft.size(), numDistortion, "yLeft");
    utils::checkSize(xRight.size(), numDistortion, "xRight");
    utils::checkSize(yRight.size(), numDistortion, "yRight");
    Array1D coeff = ndarray::allocate(4*numDistortion);
    coeff[ndarray::view(0, numDistortion)] = xLeft;
    coeff[ndarray::view(numDistortion, 2*numDistortion)] = yLeft;
    coeff[ndarray::view(2*numDistortion, 3*numDistortion)] = xRight;
    coeff[ndarray::view(3*numDistortion, 4*numDistortion)] = yRight;
    return coeff;
}


lsst::geom::Point2D DoubleDistortion::evaluate(
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
    if (onRightCcd) {
        return lsst::geom::Point2D(getXRight()(xx, yy), getYRight()(xx, yy));
    }
    return lsst::geom::Point2D(getXLeft()(xx, yy), getYLeft()(xx, yy));
}


DoubleDistortion DoubleDistortion::removeLowOrder(int order) const {
    Array1D xLeft = getXLeftCoefficients();
    Array1D yLeft = getYLeftCoefficients();
    Array1D xRight = getXRightCoefficients();
    Array1D yRight = getYRightCoefficients();

    std::size_t const num = std::min(getNumDistortion(), getNumDistortionForOrder(order));

    xLeft[ndarray::view(0, num)] = 0.0;
    yLeft[ndarray::view(0, num)] = 0.0;
    xRight[ndarray::view(0, num)] = 0.0;
    yRight[ndarray::view(0, num)] = 0.0;

    return DoubleDistortion(getOrder(), getRange(), xLeft, yLeft, xRight, yRight);
}


DoubleDistortion DoubleDistortion::merge(DoubleDistortion const& other) const {
    if (other.getRange() != getRange()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Range mismatch");
    }
    if (other.getOrder() >= getOrder()) {
        return other;
    }

    Array1D xLeft = getXLeftCoefficients();
    Array1D yLeft = getYLeftCoefficients();
    Array1D xRight = getXRightCoefficients();
    Array1D yRight = getYRightCoefficients();

    std::size_t const numOther = other.getNumDistortion();
    xLeft[ndarray::view(0, numOther)] = other.getXLeftCoefficients();
    yLeft[ndarray::view(0, numOther)] = other.getYLeftCoefficients();
    xRight[ndarray::view(0, numOther)] = other.getXRightCoefficients();
    yRight[ndarray::view(0, numOther)] = other.getYRightCoefficients();

    return DoubleDistortion(getOrder(), getRange(), xLeft, yLeft, xRight, yRight);
}


namespace {


/// Return distortion coefficients given a distortion
DoubleDistortion::Array1D getDistortionCoefficients(DoubleDistortion::Polynomial const& distortion) {
    return ndarray::copy(utils::vectorToArray(distortion.getParameters()));
}


}  // anonymous namespace


DoubleDistortion::Array1D DoubleDistortion::getXLeftCoefficients() const {
    return getDistortionCoefficients(getXLeft());
}


DoubleDistortion::Array1D DoubleDistortion::getYLeftCoefficients() const {
    return getDistortionCoefficients(getYLeft());
}


DoubleDistortion::Array1D DoubleDistortion::getXRightCoefficients() const {
    return getDistortionCoefficients(getXRight());
}


DoubleDistortion::Array1D DoubleDistortion::getYRightCoefficients() const {
    return getDistortionCoefficients(getYRight());
}


std::ostream& operator<<(std::ostream& os, DoubleDistortion const& model) {
    os << "DoubleDistortion(";
    os << "order=" << model.getOrder() << ", ";
    os << "range=" << model.getRange() << ", ";
    os << "xLeft=" << model.getXLeftCoefficients() << ", ";
    os << "yLeft=" << model.getYLeftCoefficients() << ", ";
    os << "xRight=" << model.getXRightCoefficients() << ", ";
    os << "yRight=" << model.getYRightCoefficients() << ")";
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
    std::pair<Array1D, Array1D> getSolution(
        ndarray::Array<bool, 1, 1> const& useForWavelength,
        double threshold=1.0e-6
    ) const {
        assert(index == length);
        assert(useForWavelength.size() == length);

        auto xSolution = math::solveLeastSquaresDesign(design, xMeas, xErr, threshold);

        std::size_t const numWavelength = std::count_if(
            useForWavelength.begin(),
            useForWavelength.end(),
            [](bool ss) { return ss; }
        );

        if (numWavelength == 0) {
            Array1D ySolution = utils::arrayFilled<double, 1, 1>(poly.getNParameters(), 0.0);
            return std::make_pair(xSolution, ySolution);
        }
        if (numWavelength == length) {
            Array1D ySolution = math::solveLeastSquaresDesign(design, yMeas, yErr, threshold);
            return std::make_pair(xSolution, ySolution);
        }

        ndarray::Array<double, 2, 1> yDesign = ndarray::allocate(numWavelength, poly.getNParameters());
        ndarray::Array<double, 1, 1> yMeasUse = ndarray::allocate(numWavelength);
        ndarray::Array<double, 1, 1> yErrUse = ndarray::allocate(numWavelength);
        for (std::size_t ii = 0, jj = 0; ii < length; ++ii) {
            if (!useForWavelength[ii]) continue;
            std::copy(design[ii].begin(), design[ii].end(), yDesign[jj].begin());
            yMeasUse[jj] = yMeas[ii];
            yErrUse[jj] = yErr[ii];
            ++jj;
        }

        return std::make_pair(
            xSolution,
            math::solveLeastSquaresDesign(yDesign, yMeasUse, yErrUse, threshold)
        );
    }

    DoubleDistortion::Polynomial poly;  // Polynomial used for calculating design
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
DoubleDistortion AnalyticDistortion<DoubleDistortion>::fit(
    int distortionOrder,
    lsst::geom::Box2D const& range,
    ndarray::Array<double, 1, 1> const& xx,
    ndarray::Array<double, 1, 1> const& yy,
    ndarray::Array<double, 1, 1> const& xMeas,
    ndarray::Array<double, 1, 1> const& yMeas,
    ndarray::Array<double, 1, 1> const& xErr,
    ndarray::Array<double, 1, 1> const& yErr,
    ndarray::Array<bool, 1, 1> const& useForWavelength,
    bool fitStatic,
    double threshold
) {
    using Array1D = DoubleDistortion::Array1D;
    using Array2D = DoubleDistortion::Array2D;
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
    return DoubleDistortion(distortionOrder, range, leftSolution.first, leftSolution.second,
                            rightSolution.first, rightSolution.second);
}


ndarray::Array<bool, 1, 1> DoubleDistortion::getOnRightCcd(
    ndarray::Array<double, 1, 1> const& xx
) const {
    ndarray::Array<bool, 1, 1> out = ndarray::allocate(xx.size());
    std::transform(xx.begin(), xx.end(), out.begin(),
                   [this](double value) { return getOnRightCcd(value); });
    return out;
}


namespace {

// Singleton class that manages the persistence catalog's schema and keys
class DoubleDistortionSchema {
    using IntArray = lsst::afw::table::Array<int>;
    using DoubleArray = lsst::afw::table::Array<double>;
  public:
    lsst::afw::table::Schema schema;
    lsst::afw::table::Key<int> distortionOrder;
    lsst::afw::table::Box2DKey range;
    lsst::afw::table::Key<DoubleArray> coefficients;
    lsst::afw::table::Key<int> visitInfo;

    static DoubleDistortionSchema const &get() {
        static DoubleDistortionSchema const instance;
        return instance;
    }

  private:
    DoubleDistortionSchema()
      : schema(),
        distortionOrder(schema.addField<int>("distortionOrder", "polynomial order for distortion", "")),
        range(lsst::afw::table::Box2DKey::addFields(schema, "range", "range of input values", "pixel")),
        coefficients(schema.addField<DoubleArray>("coefficients", "distortion coefficients", "", 0))
        {}
};

}  // anonymous namespace


void DoubleDistortion::write(lsst::afw::table::io::OutputArchiveHandle & handle) const {
    DoubleDistortionSchema const &schema = DoubleDistortionSchema::get();
    lsst::afw::table::BaseCatalog cat = handle.makeCatalog(schema.schema);
    std::shared_ptr<lsst::afw::table::BaseRecord> record = cat.addNew();
    record->set(schema.distortionOrder, getOrder());
    record->set(schema.range, getRange());
    ndarray::Array<double, 1, 1> xCoeff = ndarray::copy(getCoefficients());
    record->set(schema.coefficients, xCoeff);
    handle.saveCatalog(cat);
}


class DoubleDistortion::Factory : public lsst::afw::table::io::PersistableFactory {
  public:
    std::shared_ptr<lsst::afw::table::io::Persistable> read(
        lsst::afw::table::io::InputArchive const& archive,
        lsst::afw::table::io::CatalogVector const& catalogs
    ) const override {
        static auto const& schema = DoubleDistortionSchema::get();
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        lsst::afw::table::BaseRecord const& record = catalogs.front().front();
        LSST_ARCHIVE_ASSERT(record.getSchema() == schema.schema);

        int const distortionOrder = record.get(schema.distortionOrder);
        lsst::geom::Box2D const range = record.get(schema.range);
        ndarray::Array<double, 1, 1> coeff = ndarray::copy(record.get(schema.coefficients));

        return std::make_shared<DoubleDistortion>(distortionOrder, range, coeff);
    }

    Factory(std::string const& name) : lsst::afw::table::io::PersistableFactory(name) {}
};


namespace {

DoubleDistortion::Factory registration("DoubleDistortion");

}  // anonymous namespace


// Explicit instantiation
template class AnalyticDistortion<DoubleDistortion>;


}}}  // namespace pfs::drp::stella
