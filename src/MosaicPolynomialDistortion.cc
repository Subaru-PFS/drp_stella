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
#include "pfs/drp/stella/MosaicPolynomialDistortion.h"
#include "pfs/drp/stella/impl/Distortion.h"
#include "pfs/drp/stella/math/AffineTransform.h"
#include "pfs/drp/stella/math/solveLeastSquares.h"


namespace pfs {
namespace drp {
namespace stella {

namespace {

std::size_t const NUM_AFFINE = math::NUM_AFFINE_PARAMS;

}  // anonymous namespace



MosaicPolynomialDistortion::MosaicPolynomialDistortion(
    int xOrder,
    int yOrder,
    lsst::geom::Box2D const& range,
    MosaicPolynomialDistortion::Array1D const& coeff
) : MosaicPolynomialDistortion(xOrder, yOrder, range, splitCoefficients(xOrder, yOrder, coeff))
{}


MosaicPolynomialDistortion::MosaicPolynomialDistortion(
    int xOrder,
    int yOrder,
    lsst::geom::Box2D const& range,
    MosaicPolynomialDistortion::Array1D const& affineCoeff,
    MosaicPolynomialDistortion::Array1D const& xCoeff,
    MosaicPolynomialDistortion::Array1D const& yCoeff
) : AnalyticDistortion<MosaicPolynomialDistortion>(
        xOrder, yOrder, range, joinCoefficients(xOrder, yOrder, affineCoeff, xCoeff, yCoeff)
    ),
    _affine(std::move(math::makeAffineTransform(affineCoeff))),
    _poly(xOrder, yOrder, range, xCoeff, yCoeff)
{}


template<> std::size_t AnalyticDistortion<MosaicPolynomialDistortion>::getNumParametersForOrder(
    int xOrder, int yOrder
) {
    return NUM_AFFINE + MosaicPolynomialDistortion::getNumDistortionForOrder(xOrder)
        + MosaicPolynomialDistortion::getNumDistortionForOrder(yOrder);
}


std::tuple<
    MosaicPolynomialDistortion::Array1D,
    MosaicPolynomialDistortion::Array1D,
    MosaicPolynomialDistortion::Array1D
> MosaicPolynomialDistortion::splitCoefficients(
    int xOrder,
    int yOrder,
    ndarray::Array<double, 1, 1> const& coeff
) {
    std::size_t const xNumDistortion = MosaicPolynomialDistortion::getNumDistortionForOrder(xOrder);
    std::size_t const yNumDistortion = MosaicPolynomialDistortion::getNumDistortionForOrder(yOrder);
    utils::checkSize(coeff.size(), xNumDistortion + yNumDistortion + NUM_AFFINE, "coeff");
    std::size_t const xStart = NUM_AFFINE;
    std::size_t const yStart = NUM_AFFINE + xNumDistortion;
    Array1D affineCoeff = coeff[ndarray::view(0, NUM_AFFINE)];
    Array1D xPolyCoeff = coeff[ndarray::view(xStart, xStart + xNumDistortion)];
    Array1D yPolyCoeff = coeff[ndarray::view(yStart, yStart + yNumDistortion)];
    return std::make_tuple(affineCoeff, xPolyCoeff, yPolyCoeff);
}


MosaicPolynomialDistortion::Array1D MosaicPolynomialDistortion::joinCoefficients(
    int xOrder,
    int yOrder,
    MosaicPolynomialDistortion::Array1D const& affineCoeff,
    MosaicPolynomialDistortion::Array1D const& xCoeff,
    MosaicPolynomialDistortion::Array1D const& yCoeff
) {
    utils::checkSize(affineCoeff.size(), NUM_AFFINE, "affineCoeff");
    std::size_t const xNumDistortion = getNumDistortionForOrder(xOrder);
    std::size_t const yNumDistortion = getNumDistortionForOrder(yOrder);
    utils::checkSize(xCoeff.size(), xNumDistortion, "xCoeff");
    utils::checkSize(yCoeff.size(), yNumDistortion, "yCoeff");
    std::size_t const xStart = NUM_AFFINE;
    std::size_t const yStart = NUM_AFFINE + xNumDistortion;
    Array1D coeff = ndarray::allocate(NUM_AFFINE + xNumDistortion + yNumDistortion);
    coeff[ndarray::view(0, NUM_AFFINE)] = affineCoeff;
    coeff[ndarray::view(xStart, xStart + xNumDistortion)] = xCoeff;
    coeff[ndarray::view(yStart, yStart + yNumDistortion)] = yCoeff;
    return coeff;
}


ndarray::Array<bool, 1, 1> MosaicPolynomialDistortion::getOnRightCcd(
    ndarray::Array<double, 1, 1> const& xx
) const {
    ndarray::Array<bool, 1, 1> out = ndarray::allocate(xx.size());
    std::transform(xx.begin(), xx.end(), out.begin(),
                   [this](double value) { return getOnRightCcd(value); });
    return out;
}


namespace {


/// Return distortion coefficients given a distortion
MosaicPolynomialDistortion::Array1D getDistortionCoefficients(
    MosaicPolynomialDistortion::Polynomial const& distortion
) {
    return ndarray::copy(utils::vectorToArray(distortion.getParameters()));
}


}  // anonymous namespace

MosaicPolynomialDistortion::Array1D MosaicPolynomialDistortion::getAffineCoefficients() const {
    ndarray::Array<double, 1, 1> coeff = ndarray::allocate(NUM_AFFINE);
    ndarray::asEigenMatrix(coeff) = _affine.getParameterVector();
    return coeff;
}


MosaicPolynomialDistortion::Array1D MosaicPolynomialDistortion::getXCoefficients() const {
    return getDistortionCoefficients(getXPoly());
}


MosaicPolynomialDistortion::Array1D MosaicPolynomialDistortion::getYCoefficients() const {
    return getDistortionCoefficients(getYPoly());
}


std::ostream& operator<<(std::ostream& os, MosaicPolynomialDistortion const& model) {
    os << "MosaicPolynomialDistortion(";
    os << "xOrder=" << model.getXOrder() << ", ";
    os << "yOrder=" << model.getYOrder() << ", ";
    os << "range=" << model.getRange() << ", ";
    os << "affineCoeff=" << model.getAffineCoefficients() << ", ";
    os << "xCoeff=" << model.getXCoefficients() << ", ";
    os << "yCoeff=" << model.getYCoefficients() << ")";
    return os;
}


lsst::geom::Point2D MosaicPolynomialDistortion::evaluate(lsst::geom::Point2D const& xy) const {
    lsst::geom::Point2D point = _poly.evaluate(xy);
    if (getOnRightCcd(xy.getX())) {
        point += lsst::geom::Extent2D(_affine(_poly.getXPoly().normalize(xy)));
    }
    return point;
}


template<>
MosaicPolynomialDistortion AnalyticDistortion<MosaicPolynomialDistortion>::fit(
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
    using Array1D = MosaicPolynomialDistortion::Array1D;
    using Array2D = ndarray::Array<double, 2, 1>;
    using Parameters = lsst::geom::AffineTransform::Parameters;
    std::size_t const length = xx.size();
    utils::checkSize(yy.size(), length, "y");
    utils::checkSize(xMeas.size(), length, "xMeas");
    utils::checkSize(yMeas.size(), length, "yMeas");
    utils::checkSize(xErr.size(), length, "xErr");
    utils::checkSize(yErr.size(), length, "yErr");
    utils::checkSize(isLine.size(), length, "isLine");

    MosaicPolynomialDistortion::Polynomial const xPoly(xOrder, range);
    MosaicPolynomialDistortion::Polynomial const yPoly(yOrder, range);
    std::size_t const xNumPoly = xPoly.getNParameters();
    std::size_t const yNumPoly = yPoly.getNParameters();
    std::size_t const numAffine = NUM_AFFINE/2;  // Affine parameters per dimension
    std::size_t const xNumParams = xNumPoly + numAffine;  // Parameters for x
    std::size_t const yNumParams = yNumPoly + numAffine;  // Parameters for y
    if (!forced.isEmpty()) {
        utils::checkSize(forced.size(), xNumParams + yNumParams, "forced");
    }
    if (!params.isEmpty()) {
        utils::checkSize(params.size(), xNumParams + yNumParams, "params");
    }
    double const xMiddle = range.getCenterX();  // Boundary between CCDs

    // The x and y dimensions are independent, so we fit them separately. This
    // ensures that the eigenvalue threshold for each dimension is relative to
    // that dimension only, which is important when the number of measurements
    // in each dimension is very different (traces only provide x).
    //
    // For each dimension, the parameters are the polynomial coefficients
    // followed by the affine terms (constant, x, y) for the right CCD.
    std::size_t const numLines = std::count(isLine.begin(), isLine.end(), true);
    Array2D xDesign = ndarray::allocate(length, xNumParams);
    Array2D yDesign = ndarray::allocate(numLines, yNumParams);
    xDesign.deep() = 0.0;
    yDesign.deep() = 0.0;
    Array1D yMeasLines = ndarray::allocate(numLines);
    Array1D yErrLines = ndarray::allocate(numLines);
    for (std::size_t ii = 0, jj = 0; ii < length; ++ii) {
        lsst::geom::Point2D const xy(xx[ii], yy[ii]);
        bool const onRightCcd = xy.getX() > xMiddle;
        lsst::geom::Point2D const xyNorm = xPoly.normalize(xy);  // same range for x and y

        auto const xTerms = xPoly.getDFuncDParameters(xy.getX(), xy.getY());
        assert(xTerms.size() == xNumPoly);
        std::copy(xTerms.begin(), xTerms.end(), xDesign[ii].begin());
        if (onRightCcd) {
            xDesign[ii][xNumPoly] = 1.0;
            xDesign[ii][xNumPoly + 1] = xyNorm.getX();
            xDesign[ii][xNumPoly + 2] = xyNorm.getY();
        }

        if (isLine[ii]) {
            auto const yTerms = yPoly.getDFuncDParameters(xy.getX(), xy.getY());
            assert(yTerms.size() == yNumPoly);
            std::copy(yTerms.begin(), yTerms.end(), yDesign[jj].begin());
            if (onRightCcd) {
                yDesign[jj][yNumPoly] = 1.0;
                yDesign[jj][yNumPoly + 1] = xyNorm.getX();
                yDesign[jj][yNumPoly + 2] = xyNorm.getY();
            }
            yMeasLines[jj] = yMeas[ii];
            yErrLines[jj] = yErr[ii];
            ++jj;
        }
    }

    // forced and params are ordered: x coefficients, y coefficients, then the
    // affine terms X, XX, XY, Y, YX, YY.
    std::vector<std::size_t> xIndices(xNumParams);
    std::vector<std::size_t> yIndices(yNumParams);
    std::iota(xIndices.begin(), xIndices.begin() + xNumPoly, 0);
    std::iota(yIndices.begin(), yIndices.begin() + yNumPoly, xNumPoly);
    std::iota(xIndices.begin() + xNumPoly, xIndices.end(), xNumPoly + yNumPoly);
    std::iota(yIndices.begin() + yNumPoly, yIndices.end(), xNumPoly + yNumPoly + numAffine);

    Array1D const xSolution = math::solveLeastSquaresDesign(
        xDesign, xMeas, xErr, threshold,
        detail::takeOptional(forced, xIndices), detail::takeOptional(params, xIndices)
    );
    Array1D const ySolution = math::solveLeastSquaresDesign(
        yDesign, yMeasLines, yErrLines, threshold,
        detail::takeOptional(forced, yIndices), detail::takeOptional(params, yIndices)
    );

    Array1D affine = ndarray::allocate(NUM_AFFINE);
    affine[Parameters::X] = xSolution[xNumPoly];
    affine[Parameters::XX] = xSolution[xNumPoly + 1];
    affine[Parameters::XY] = xSolution[xNumPoly + 2];
    affine[Parameters::Y] = ySolution[yNumPoly];
    affine[Parameters::YX] = ySolution[yNumPoly + 1];
    affine[Parameters::YY] = ySolution[yNumPoly + 2];

    return MosaicPolynomialDistortion(
        xOrder, yOrder, range, affine,
        ndarray::copy(xSolution[ndarray::view(0, xNumPoly)]),
        ndarray::copy(ySolution[ndarray::view(0, yNumPoly)])
    );
}


namespace {

// Singleton class that manages the persistence catalog's schema and keys
class MosaicPolynomialDistortionSchema {
    using IntArray = lsst::afw::table::Array<int>;
    using DoubleArray = lsst::afw::table::Array<double>;
  public:
    lsst::afw::table::Schema schema;
    lsst::afw::table::Key<int> xOrder;
    lsst::afw::table::Key<int> yOrder;
    lsst::afw::table::Box2DKey range;
    lsst::afw::table::Key<DoubleArray> coefficients;
    lsst::afw::table::Key<int> visitInfo;

    static MosaicPolynomialDistortionSchema const &get() {
        static MosaicPolynomialDistortionSchema const instance;
        return instance;
    }

  private:
    MosaicPolynomialDistortionSchema()
      : schema(),
        xOrder(schema.addField<int>("xOrder", "polynomial order for x distortion", "")),
        yOrder(schema.addField<int>("yOrder", "polynomial order for y distortion", "")),
        range(lsst::afw::table::Box2DKey::addFields(schema, "range", "range of input values", "pixel")),
        coefficients(schema.addField<DoubleArray>("coefficients", "distortion coefficients", "", 0))
        {}
};

}  // anonymous namespace


void MosaicPolynomialDistortion::write(lsst::afw::table::io::OutputArchiveHandle & handle) const {
    MosaicPolynomialDistortionSchema const &schema = MosaicPolynomialDistortionSchema::get();
    lsst::afw::table::BaseCatalog cat = handle.makeCatalog(schema.schema);
    std::shared_ptr<lsst::afw::table::BaseRecord> record = cat.addNew();
    record->set(schema.xOrder, getXOrder());
    record->set(schema.yOrder, getYOrder());
    record->set(schema.range, getRange());
    ndarray::Array<double, 1, 1> xCoeff = ndarray::copy(getCoefficients());
    record->set(schema.coefficients, xCoeff);
    handle.saveCatalog(cat);
}


class MosaicPolynomialDistortion::Factory : public lsst::afw::table::io::PersistableFactory {
  public:
    std::shared_ptr<lsst::afw::table::io::Persistable> read(
        lsst::afw::table::io::InputArchive const& archive,
        lsst::afw::table::io::CatalogVector const& catalogs
    ) const override {
        static auto const& schema = MosaicPolynomialDistortionSchema::get();
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        lsst::afw::table::BaseRecord const& record = catalogs.front().front();
        LSST_ARCHIVE_ASSERT(record.getSchema() == schema.schema);

        int const xOrder = record.get(schema.xOrder);
        int const yOrder = record.get(schema.yOrder);
        lsst::geom::Box2D const range = record.get(schema.range);
        ndarray::Array<double, 1, 1> coeff = ndarray::copy(record.get(schema.coefficients));

        return std::make_shared<MosaicPolynomialDistortion>(xOrder, yOrder, range, coeff);
    }

    Factory(std::string const& name) : lsst::afw::table::io::PersistableFactory(name) {}
};


namespace {

MosaicPolynomialDistortion::Factory registration("MosaicPolynomialDistortion");

}  // anonymous namespace


// Explicit instantiation
template class AnalyticDistortion<MosaicPolynomialDistortion>;


}}}  // namespace pfs::drp::stella
