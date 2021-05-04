#include <set>
#include <numeric>
#include <algorithm>

#include "ndarray.h"
#include "ndarray/eigen.h"

#include "lsst/afw/table.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/Persistable.cc"

#include "pfs/drp/stella/utils/checkSize.h"
#include "pfs/drp/stella/utils/math.h"
#include "pfs/drp/stella/DetectorDistortion.h"


namespace pfs {
namespace drp {
namespace stella {


DetectorDistortion::DetectorDistortion(
    int distortionOrder,
    lsst::geom::Box2D const& range,
    ndarray::Array<double, 1, 1> const& xDistortion,
    ndarray::Array<double, 1, 1> const& yDistortion,
    ndarray::Array<double, 1, 1> const& rightCcd
) : _distortionOrder(distortionOrder),
    _range(range),
    _xDistortion(utils::arrayToVector(xDistortion), range),
    _yDistortion(utils::arrayToVector(yDistortion), range),
    _rightCcd()
{
    std::size_t const numDistortion = getNumDistortion(distortionOrder);
    utils::checkSize(xDistortion.size(), numDistortion, "xDistortion");
    utils::checkSize(yDistortion.size(), numDistortion, "yDistortion");
    utils::checkSize(rightCcd.size(), 6UL, "rightCcd");
    _rightCcd.setParameterVector(ndarray::asEigenArray(rightCcd));
}


lsst::geom::Point2D DetectorDistortion::operator()(
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


ndarray::Array<double, 2, 1> DetectorDistortion::operator()(
    ndarray::Array<double, 1, 1> const& xx,
    ndarray::Array<double, 1, 1> const& yy,
    ndarray::Array<bool, 1, 1> const& onRightCcd
) const {
    std::size_t const length = xx.size();
    utils::checkSize(yy.size(), length, "y");
    utils::checkSize(onRightCcd.size(), length, "onRightCcd");
    ndarray::Array<double, 2, 1> out = ndarray::allocate(length, 2);
    for (std::size_t ii = 0; ii < length; ++ii) {
        auto const point = operator()(lsst::geom::Point2D(xx[ii], yy[ii]), onRightCcd[ii]);
        out[ii][0] = point.getX();
        out[ii][1] = point.getY();
    }
    return out;
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


ndarray::Array<double, 2, 1> DetectorDistortion::calculateDesignMatrix(
    int distortionOrder,
    lsst::geom::Box2D const& range,
    ndarray::Array<double, 1, 1> const& xx,
    ndarray::Array<double, 1, 1> const& yy
) {
    utils::checkSize(xx.size(), yy.size(), "x vs y");
    std::size_t const length = xx.size();

    std::size_t const numTerms = DetectorDistortion::getNumDistortion(distortionOrder);
    ndarray::Array<double, 2, 1> matrix = ndarray::allocate(length, numTerms);
    Polynomial const poly(distortionOrder, range);
    for (std::size_t ii = 0; ii < length; ++ii) {
        auto const terms = poly.getDFuncDParameters(xx[ii], yy[ii]);
        std::copy(terms.begin(), terms.end(), matrix[ii].begin());
    }
    return matrix;
}


std::pair<double, std::size_t> DetectorDistortion::calculateChi2(
    ndarray::Array<double, 1, 1> const& xOrig,
    ndarray::Array<double, 1, 1> const& yOrig,
    ndarray::Array<bool, 1, 1> const& onRightCcd,
    ndarray::Array<double, 1, 1> const& xMeas,
    ndarray::Array<double, 1, 1> const& yMeas,
    ndarray::Array<double, 1, 1> const& xErr,
    ndarray::Array<double, 1, 1> const& yErr,
    ndarray::Array<bool, 1, 1> const& goodOrig,
    float sysErr
) const {
    std::size_t const length = xOrig.size();
    utils::checkSize(yOrig.size(), length, "yOrig");
    utils::checkSize(onRightCcd.size(), length, "onRightCcd");
    utils::checkSize(xMeas.size(), length, "xMeas");
    utils::checkSize(yMeas.size(), length, "yMeas");
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
        double const xErr2 = std::pow(xErr[ii], 2) + sysErr2;
        double const yErr2 = std::pow(yErr[ii], 2) + sysErr2;
        lsst::geom::Point2D const fit = operator()(lsst::geom::Point2D(xOrig[ii], yOrig[ii]), onRightCcd[ii]);
        chi2 += std::pow(xMeas[ii] - fit.getX(), 2)/xErr2 + std::pow(yMeas[ii] - fit.getY(), 2)/yErr2;
        num += 2;  // one for x, one for y
    }
    std::size_t const numFitParams = getNumParameters();
    std::size_t const dof = num - numFitParams;
    return std::make_pair(chi2, dof);
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
    record->set(schema.distortionOrder, getDistortionOrder());
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



}}}  // namespace pfs::drp::stella
