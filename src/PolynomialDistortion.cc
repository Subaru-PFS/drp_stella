#include <algorithm>
#include <numeric>
#include <vector>
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
    ndarray::Array<bool, 1, 1> const& isLine,
    double threshold,
    ndarray::Array<bool, 1, 1> const& forced,
    ndarray::Array<double, 1, 1> const& params
) {
    using Array1D = PolynomialDistortion::Array1D;
    using Array2D = ndarray::Array<double, 2, 1>;
    std::size_t const length = xx.size();
    utils::checkSize(yy.size(), length, "y");
    utils::checkSize(xMeas.size(), length, "xMeas");
    utils::checkSize(yMeas.size(), length, "yMeas");
    utils::checkSize(xErr.size(), length, "xErr");
    utils::checkSize(yErr.size(), length, "yErr");
    utils::checkSize(isLine.size(), length, "isLine");

    PolynomialDistortion::Polynomial const poly(distortionOrder, range);
    std::size_t const numPoly = poly.getNParameters();
    if (!forced.isEmpty()) {
        utils::checkSize(forced.size(), 2*numPoly, "forced");
    }
    if (!params.isEmpty()) {
        utils::checkSize(params.size(), 2*numPoly, "params");
    }

    // The x and y dimensions are independent, so we fit them separately. This
    // ensures that the eigenvalue threshold for each dimension is relative to
    // that dimension only, which is important when the number of measurements
    // in each dimension is very different (traces only provide x).
    std::size_t const numLines = std::count(isLine.begin(), isLine.end(), true);
    Array2D xDesign = ndarray::allocate(length, numPoly);
    Array2D yDesign = ndarray::allocate(numLines, numPoly);
    Array1D yMeasLines = ndarray::allocate(numLines);
    Array1D yErrLines = ndarray::allocate(numLines);
    for (std::size_t ii = 0, jj = 0; ii < length; ++ii) {
        auto const terms = poly.getDFuncDParameters(xx[ii], yy[ii]);
        assert(terms.size() == numPoly);
        std::copy(terms.begin(), terms.end(), xDesign[ii].begin());
        if (isLine[ii]) {
            std::copy(terms.begin(), terms.end(), yDesign[jj].begin());
            yMeasLines[jj] = yMeas[ii];
            yErrLines[jj] = yErr[ii];
            ++jj;
        }
    }

    // forced and params are ordered: x coefficients, y coefficients
    std::vector<std::size_t> xIndices(numPoly);
    std::vector<std::size_t> yIndices(numPoly);
    std::iota(xIndices.begin(), xIndices.end(), 0);
    std::iota(yIndices.begin(), yIndices.end(), numPoly);

    Array1D const xSolution = math::solveLeastSquaresDesign(
        xDesign, xMeas, xErr, threshold,
        detail::takeOptional(forced, xIndices), detail::takeOptional(params, xIndices)
    );
    Array1D const ySolution = math::solveLeastSquaresDesign(
        yDesign, yMeasLines, yErrLines, threshold,
        detail::takeOptional(forced, yIndices), detail::takeOptional(params, yIndices)
    );
    return PolynomialDistortion(distortionOrder, range, xSolution, ySolution);
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
