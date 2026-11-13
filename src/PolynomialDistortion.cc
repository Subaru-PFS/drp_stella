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


template<> std::size_t AnalyticDistortion<PolynomialDistortion>::getNumParametersForOrder(
    int xOrder, int yOrder
) {
    return PolynomialDistortion::getNumDistortionForOrder(xOrder)
        + PolynomialDistortion::getNumDistortionForOrder(yOrder);
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

    PolynomialDistortion::Polynomial const xPoly(xOrder, range);
    PolynomialDistortion::Polynomial const yPoly(yOrder, range);
    std::size_t const xNumPoly = xPoly.getNParameters();
    std::size_t const yNumPoly = yPoly.getNParameters();
    if (!forced.isEmpty()) {
        utils::checkSize(forced.size(), xNumPoly + yNumPoly, "forced");
    }
    if (!params.isEmpty()) {
        utils::checkSize(params.size(), xNumPoly + yNumPoly, "params");
    }

    // The x and y dimensions are independent, so we fit them separately. This
    // ensures that the eigenvalue threshold for each dimension is relative to
    // that dimension only, which is important when the number of measurements
    // in each dimension is very different (traces only provide x).
    std::size_t const numLines = std::count(isLine.begin(), isLine.end(), true);
    Array2D xDesign = ndarray::allocate(length, xNumPoly);
    Array2D yDesign = ndarray::allocate(numLines, yNumPoly);
    Array1D yMeasLines = ndarray::allocate(numLines);
    Array1D yErrLines = ndarray::allocate(numLines);
    for (std::size_t ii = 0, jj = 0; ii < length; ++ii) {
        auto const xTerms = xPoly.getDFuncDParameters(xx[ii], yy[ii]);
        assert(xTerms.size() == xNumPoly);
        std::copy(xTerms.begin(), xTerms.end(), xDesign[ii].begin());
        if (isLine[ii]) {
            auto const yTerms = yPoly.getDFuncDParameters(xx[ii], yy[ii]);
            assert(yTerms.size() == yNumPoly);
            std::copy(yTerms.begin(), yTerms.end(), yDesign[jj].begin());
            yMeasLines[jj] = yMeas[ii];
            yErrLines[jj] = yErr[ii];
            ++jj;
        }
    }

    // forced and params are ordered: x coefficients, y coefficients
    std::vector<std::size_t> xIndices(xNumPoly);
    std::vector<std::size_t> yIndices(yNumPoly);
    std::iota(xIndices.begin(), xIndices.end(), 0);
    std::iota(yIndices.begin(), yIndices.end(), xNumPoly);

    Array1D const xSolution = math::solveLeastSquaresDesign(
        xDesign, xMeas, xErr, threshold,
        detail::takeOptional(forced, xIndices), detail::takeOptional(params, xIndices)
    );
    Array1D const ySolution = math::solveLeastSquaresDesign(
        yDesign, yMeasLines, yErrLines, threshold,
        detail::takeOptional(forced, yIndices), detail::takeOptional(params, yIndices)
    );
    return PolynomialDistortion(xOrder, yOrder, range, xSolution, ySolution);
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
