#include <numeric>

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
#include "pfs/drp/stella/DetectorDistortion.h"
#include "pfs/drp/stella/impl/BaseDistortion.h"
#include "pfs/drp/stella/math/solveLeastSquares.h"


namespace pfs {
namespace drp {
namespace stella {


DetectorDistortion::DetectorDistortion(
    int order,
    lsst::geom::Box2D const& range,
    DetectorDistortion::Array1D const& xDistortion,
    DetectorDistortion::Array1D const& yDistortion,
    DetectorDistortion::Array1D const& rightCcd
) : BaseDistortion<DetectorDistortion>(order, range,
                                       joinCoefficients(order, xDistortion, yDistortion, rightCcd)),
    _xDistortion(xDistortion, range),
    _yDistortion(yDistortion, range),
    _rightCcd()
{
    _rightCcd.setParameterVector(ndarray::asEigenArray(rightCcd));
}


template<>
std::size_t BaseDistortion<DetectorDistortion>::getNumParametersForOrder(int order) {
    return 2*DetectorDistortion::getNumDistortionForOrder(order) + 6;
}


std::tuple<DetectorDistortion::Array1D, DetectorDistortion::Array1D, DetectorDistortion::Array1D>
DetectorDistortion::splitCoefficients(
    int order,
    ndarray::Array<double, 1, 1> const& coeff
) {
    utils::checkSize(coeff.size(), DetectorDistortion::getNumParametersForOrder(order), "coeff");
    std::size_t const numDistortion = DetectorDistortion::getNumDistortionForOrder(order);
    return std::tuple<Array1D, Array1D, Array1D>(
        ndarray::copy(coeff[ndarray::view(0, numDistortion)]),
        ndarray::copy(coeff[ndarray::view(numDistortion, 2*numDistortion)]),
        ndarray::copy(coeff[ndarray::view(2*numDistortion, 2*numDistortion + 6)])
    );
}


DetectorDistortion::Array1D DetectorDistortion::joinCoefficients(
    int order,
    DetectorDistortion::Array1D const& xDistortion,
    DetectorDistortion::Array1D const& yDistortion,
    DetectorDistortion::Array1D const& rightCcd
) {
    std::size_t const numDistortion = getNumDistortionForOrder(order);
    utils::checkSize(xDistortion.size(), numDistortion, "xDistortion");
    utils::checkSize(yDistortion.size(), numDistortion, "yDistortion");
    utils::checkSize(rightCcd.size(), 6UL, "rightCcd");
    Array1D coeff = ndarray::allocate(2*numDistortion + 6);
    coeff[ndarray::view(0, numDistortion)] = xDistortion;
    coeff[ndarray::view(numDistortion, 2*numDistortion)] = yDistortion;
    coeff[ndarray::view(2*numDistortion, 2*numDistortion + 6)] = rightCcd;
    return coeff;
}


lsst::geom::Point2D DetectorDistortion::evaluate(
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
    // x,y: distorted position
    lsst::geom::Point2D xy(getXDistortion()(xx, yy), getYDistortion()(xx, yy));
    // x,y: detector coordinates
    if (onRightCcd) {
        xy += lsst::geom::Extent2D(getRightCcd()(point));
    }
    return xy;
}


DetectorDistortion DetectorDistortion::removeLowOrder(int order) const {
    Array1D xDistortion = getXCoefficients();
    Array1D yDistortion = getYCoefficients();

    std::size_t const num = std::max(getNumDistortion(), getNumDistortionForOrder(order));
    xDistortion[ndarray::view(0, num)] = 0.0;
    yDistortion[ndarray::view(0, num)] = 0.0;
    
    return DetectorDistortion(getOrder(), getRange(), xDistortion, yDistortion, getRightCcdCoefficients());
}


DetectorDistortion DetectorDistortion::merge(DetectorDistortion const& other) const {
    if (other.getRange() != getRange()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Range mismatch");
    }
    if (other.getOrder() >= getOrder()) {
        return other;
    }

    Array1D xDistortion = getXCoefficients();
    Array1D yDistortion = getYCoefficients();

    std::size_t const numOther = other.getNumDistortion();
    xDistortion[ndarray::view(0, numOther)] = other.getXCoefficients();
    yDistortion[ndarray::view(0, numOther)] = other.getYCoefficients();

    return DetectorDistortion(getOrder(), getRange(), xDistortion, yDistortion, getRightCcdCoefficients());
}


ndarray::Array<double, 1, 1> DetectorDistortion::getXCoefficients() const {
    return ndarray::copy(utils::vectorToArray(getXDistortion().getParameters()));
}


ndarray::Array<double, 1, 1> DetectorDistortion::getYCoefficients() const {
    return ndarray::copy(utils::vectorToArray(getYDistortion().getParameters()));
}


ndarray::Array<double, 1, 1> DetectorDistortion::getRightCcdCoefficients() const {
    auto const parameters = _rightCcd.getParameterVector();  // An Eigen matrix
    assert(parameters.size() == 6);
    ndarray::Array<double, 1, 1> out = ndarray::allocate(parameters.size());
    ndarray::asEigenMatrix(out) = parameters;
    return out;
}


ndarray::Array<double, 1, 1> DetectorDistortion::makeRightCcdCoefficients(
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


std::ostream& operator<<(std::ostream& os, DetectorDistortion const& model) {
    os << "DetectorDistortion(";
    os << "xDistortion=" << model.getXCoefficients() << ", ";
    os << "yDistortion=" << model.getYCoefficients() << ", ";
    os << "rightCcd=" << model.getRightCcdCoefficients() << ", " << ")";
    return os;
}


template<>
DetectorDistortion BaseDistortion<DetectorDistortion>::fit(
    int distortionOrder,
    lsst::geom::Box2D const& range,
    ndarray::Array<double, 1, 1> const& xx,
    ndarray::Array<double, 1, 1> const& yy,
    ndarray::Array<double, 1, 1> const& xMeas,
    ndarray::Array<double, 1, 1> const& yMeas,
    ndarray::Array<double, 1, 1> const& xErr,
    ndarray::Array<double, 1, 1> const& yErr,
    bool fitStatic
) {
    std::size_t const length = xx.size();
    utils::checkSize(yy.size(), length, "y");
    utils::checkSize(xMeas.size(), length, "xMeas");
    utils::checkSize(yMeas.size(), length, "yMeas");
    utils::checkSize(xErr.size(), length, "xErr");
    utils::checkSize(yErr.size(), length, "yErr");

    std::size_t const numDistortion = DetectorDistortion::getNumDistortionForOrder(distortionOrder);
    std::size_t const numTerms = numDistortion + (fitStatic ? 3 : 0);
    ndarray::Array<double, 2, 1> design = ndarray::allocate(length, numTerms);
    DetectorDistortion::Polynomial const poly(distortionOrder, range);
    double const xCenter = range.getCenterX();
    for (std::size_t ii = 0; ii < length; ++ii) {
        auto const terms = poly.getDFuncDParameters(xx[ii], yy[ii]);
        assert(terms.size() == numDistortion);
        std::copy(terms.begin(), terms.end(), design[ii].begin());
        if (fitStatic) {
            if (xx[ii] > xCenter) {
                design[ii][numTerms - 3] = 1.0;
                design[ii][numTerms - 2] = xx[ii];
                design[ii][numTerms - 1] = yy[ii];
            } else {
                design[ii][ndarray::view(numTerms - 3, numTerms)] = 0.0;
            }
        }
    }

    ndarray::Array<double const, 1, 1> xSolution = math::solveLeastSquaresDesign(design, xMeas, xErr);
    ndarray::Array<double const, 1, 1> ySolution = math::solveLeastSquaresDesign(design, yMeas, yErr);
    ndarray::Array<double, 1, 1> xCoeff = copy(xSolution[ndarray::view(0, numDistortion)]);
    ndarray::Array<double, 1, 1> yCoeff = copy(ySolution[ndarray::view(0, numDistortion)]);
    ndarray::Array<double, 1, 1> rightCcd;
    if (fitStatic) {
        rightCcd = DetectorDistortion::makeRightCcdCoefficients(xSolution[numTerms - 3],
        ySolution[numTerms - 3], xSolution[numTerms - 2], xSolution[numTerms - 1],
        ySolution[numTerms - 2], ySolution[numTerms - 1]);
    } else {
        rightCcd = utils::arrayFilled<double, 1, 1>(6, 0);
    }
    return DetectorDistortion(distortionOrder, range, xCoeff, yCoeff, rightCcd);
}


ndarray::Array<bool, 1, 1> DetectorDistortion::getOnRightCcd(
    ndarray::Array<double, 1, 1> const& xx
) const {
    ndarray::Array<bool, 1, 1> out = ndarray::allocate(xx.size());
    std::transform(xx.begin(), xx.end(), out.begin(),
                   [this](double value) { return getOnRightCcd(value); });
    return out;
}


namespace {

// Singleton class that manages the persistence catalog's schema and keys
class DetectorDistortionSchema {
    using IntArray = lsst::afw::table::Array<int>;
    using DoubleArray = lsst::afw::table::Array<double>;
  public:
    lsst::afw::table::Schema schema;
    lsst::afw::table::Key<int> distortionOrder;
    lsst::afw::table::Box2DKey range;
    lsst::afw::table::Key<DoubleArray> xCoefficients;
    lsst::afw::table::Key<DoubleArray> yCoefficients;
    lsst::afw::table::Key<DoubleArray> rightCcd;
    lsst::afw::table::Key<int> visitInfo;

    static DetectorDistortionSchema const &get() {
        static DetectorDistortionSchema const instance;
        return instance;
    }

  private:
    DetectorDistortionSchema()
      : schema(),
        distortionOrder(schema.addField<int>("distortionOrder", "polynomial order for distortion", "")),
        range(lsst::afw::table::Box2DKey::addFields(schema, "range", "range of input values", "pixel")),
        xCoefficients(schema.addField<DoubleArray>("xCoefficients", "x distortion coefficients", "", 0)),
        yCoefficients(schema.addField<DoubleArray>("yCoefficients", "y distortion coefficients", "", 0)),
        rightCcd(schema.addField<DoubleArray>("rightCcd", "affine transform for right Ccd", "", 0)) {
            schema.getCitizen().markPersistent();
    }
};

}  // anonymous namespace


void DetectorDistortion::write(lsst::afw::table::io::OutputArchiveHandle & handle) const {
    DetectorDistortionSchema const &schema = DetectorDistortionSchema::get();
    lsst::afw::table::BaseCatalog cat = handle.makeCatalog(schema.schema);
    PTR(lsst::afw::table::BaseRecord) record = cat.addNew();
    record->set(schema.distortionOrder, getOrder());
    record->set(schema.range, getRange());
    ndarray::Array<double, 1, 1> xCoeff = ndarray::copy(getXCoefficients());
    record->set(schema.xCoefficients, xCoeff);
    ndarray::Array<double, 1, 1> yCoeff = ndarray::copy(getYCoefficients());
    record->set(schema.yCoefficients, yCoeff);
    ndarray::Array<double, 1, 1> rightCcd = ndarray::copy(getRightCcdCoefficients());
    record->set(schema.rightCcd, rightCcd);
    handle.saveCatalog(cat);
}


class DetectorDistortion::Factory : public lsst::afw::table::io::PersistableFactory {
  public:
    std::shared_ptr<lsst::afw::table::io::Persistable> read(
        lsst::afw::table::io::InputArchive const& archive,
        lsst::afw::table::io::CatalogVector const& catalogs
    ) const override {
        static auto const& schema = DetectorDistortionSchema::get();
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        lsst::afw::table::BaseRecord const& record = catalogs.front().front();
        LSST_ARCHIVE_ASSERT(record.getSchema() == schema.schema);

        int const distortionOrder = record.get(schema.distortionOrder);
        lsst::geom::Box2D const range = record.get(schema.range);
        ndarray::Array<double, 1, 1> xCoeff = ndarray::copy(record.get(schema.xCoefficients));
        ndarray::Array<double, 1, 1> yCoeff = ndarray::copy(record.get(schema.yCoefficients));
        ndarray::Array<double, 1, 1> rightCcd = ndarray::copy(record.get(schema.rightCcd));

        return std::make_shared<DetectorDistortion>(distortionOrder, range, xCoeff, yCoeff, rightCcd);
    }

    Factory(std::string const& name) : lsst::afw::table::io::PersistableFactory(name) {}
};


namespace {

DetectorDistortion::Factory registration("DetectorDistortion");

}  // anonymous namespace


// Explicit instantiation
template class BaseDistortion<DetectorDistortion>;


}}}  // namespace pfs::drp::stella
