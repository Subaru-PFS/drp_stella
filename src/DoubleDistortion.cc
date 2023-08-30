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
    _left(order, leftRange(range), xLeft, yLeft),
    _right(order, rightRange(range), xRight, yRight)
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
    ndarray::Array<bool, 1, 1> const& isLine,
    ndarray::Array<double, 1, 1> const& slope,
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
    utils::checkSize(isLine.getNumElements(), length, "isLine");

    double const xCenter = range.getCenterX();
    ndarray::Array<bool, 1, 1> onLeftCcd = ndarray::allocate(length);
    asEigenArray(onLeftCcd) = asEigenArray(xx) < xCenter;
    ndarray::Array<bool, 1, 1> onRightCcd = ndarray::allocate(length);
    asEigenArray(onRightCcd) = asEigenArray(xx) >= xCenter;

    auto const left = PolynomialDistortion::fit(
        distortionOrder,
        leftRange(range),
        utils::arraySelect(xx, onLeftCcd),
        utils::arraySelect(yy, onLeftCcd),
        utils::arraySelect(xMeas, onLeftCcd),
        utils::arraySelect(yMeas, onLeftCcd),
        utils::arraySelect(xErr, onLeftCcd),
        utils::arraySelect(yErr, onLeftCcd),
        utils::arraySelect(isLine, onLeftCcd),
        utils::arraySelect(slope, onLeftCcd),
        threshold
    );
    auto const right = PolynomialDistortion::fit(
        distortionOrder,
        rightRange(range),
        utils::arraySelect(xx, onRightCcd),
        utils::arraySelect(yy, onRightCcd),
        utils::arraySelect(xMeas, onRightCcd),
        utils::arraySelect(yMeas, onRightCcd),
        utils::arraySelect(xErr, onRightCcd),
        utils::arraySelect(yErr, onRightCcd),
        utils::arraySelect(isLine, onRightCcd),
        utils::arraySelect(slope, onRightCcd),
        threshold
    );

    return DoubleDistortion(distortionOrder, range, left.getXCoefficients(), left.getYCoefficients(),
                            right.getXCoefficients(), right.getYCoefficients());
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
    ndarray::Array<double, 1, 1> coeff = ndarray::copy(getCoefficients());
    record->set(schema.coefficients, coeff);
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
