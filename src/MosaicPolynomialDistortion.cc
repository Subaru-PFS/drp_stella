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


template<> std::size_t AnalyticDistortion<MosaicPolynomialDistortion>::getNumParametersForOrder(int order) {
    return NUM_AFFINE + 2*MosaicPolynomialDistortion::getNumDistortionForOrder(order);
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
    // @param xOrder : Polynomial order for x.
    // @param yOrder : Polynomial order for y.
    // @param length : Number of points that will be added.
    FitData(lsst::geom::Box2D const& range, int xOrder, int yOrder, std::size_t numLines_, std::size_t numTraces_) :
        xMiddle(range.getCenterX()),
        xPoly(xOrder, range),
        yPoly(yOrder, range),
        affineStart(0),
        xStart(NUM_AFFINE),
        yStart(NUM_AFFINE + xPoly.getNParameters()),
        numLines(numLines_),
        numTraces(numTraces_),
        length(2*numLines + numTraces),
        measurements(ndarray::allocate(length)),
        errors(ndarray::allocate(length)),
        design(ndarray::allocate(length, xPoly.getNParameters() + yPoly.getNParameters() + NUM_AFFINE)),
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

        std::size_t const xNumPoly = xPoly.getNParameters();
        std::size_t const yNumPoly = yPoly.getNParameters();

        std::vector<double> const xTerms = xPoly.getDFuncDParameters(xy.getX(), xy.getY());
        assert (xTerms.size() == xNumPoly);
        std::vector<double> yTerms;
        if (xPoly.getOrder() == yPoly.getOrder()) {
            // If the orders are the same, we can reuse the terms.
            yTerms = xTerms;
        } else {
            yTerms = yPoly.getDFuncDParameters(xy.getX(), xy.getY());
            assert (yTerms.size() == yNumPoly);
        }

        lsst::geom::Point2D const xyNorm = xPoly.normalize(xy);  // Normalization should be same for x and y

        // Affine part of the design matrix
        bool const onRightCcd = xy.getX() > xMiddle;
        if (onRightCcd) {
            design[ii][affineStart] = 1.0;
            design[ii][affineStart + 1] = xyNorm.getX();
            design[ii][affineStart + 2] = xyNorm.getY();
        }

        // x part of the design matrix
        std::copy(xTerms.begin(), xTerms.end(), design[ii].begin() + xStart);
        measurements[ii] = meas.getX();
        errors[ii] = err.getX();

        // y part of the design matrix
        if (isLine) {
            // For a line, the y part is independent of the x part.
            std::size_t const jj = index++;
            assert(jj < length);
            std::copy(yTerms.begin(), yTerms.end(), design[jj].begin() + yStart);
            measurements[jj] = meas.getY();
            errors[jj] = err.getY();

            // Affine part of the design matrix
            if (onRightCcd) {
                design[jj][affineStart + 3] = 1.0;
                design[jj][affineStart + 4] = xyNorm.getX();
                design[jj][affineStart + 5] = xyNorm.getY();
            }
        } else {
            // For a trace, the y part is linked to the x part by the slope.
            auto lhs = design[ii][ndarray::view(yStart, yStart + yNumPoly)];
            ndarray::asEigenArray(lhs) = -slope*ndarray::asEigenArray(utils::vectorToArray(yTerms));

            // Affine part of the design matrix
            if (onRightCcd) {
                auto lhs = design[ii][ndarray::view(affineStart + 3, affineStart + 6)];
                auto const rhs = design[ii][ndarray::view(affineStart, affineStart + 3)];                ndarray::asEigenArray(lhs) = -slope*ndarray::asEigenArray(rhs);
            }
        }
    }

    // Solve the least-squares problem
    //
    // @param threshold : Threshold for truncating eigenvalues (see lsst::afw::math::LeastSquares)
    // @return Solutions in x and y.
    std::tuple<Array1D, Array1D, Array1D> getSolution(
        double threshold=1.0e-6,
        ndarray::Array<bool, 1, 1> const& forced=ndarray::Array<bool, 1, 1>(),
        ndarray::Array<double, 1, 1> const& params=ndarray::Array<double, 1, 1>()
    ) const {
        assert(index == length);  // everything got added
        auto solution = math::solveLeastSquaresDesign(
            design, measurements, errors, threshold, forced, params
        );

        ndarray::Array<double, 1, 1> affine = ndarray::allocate(NUM_AFFINE);
        affine[lsst::geom::AffineTransform::Parameters::X] = solution[affineStart];
        affine[lsst::geom::AffineTransform::Parameters::XX] = solution[affineStart + 1];
        affine[lsst::geom::AffineTransform::Parameters::XY] = solution[affineStart + 2];
        affine[lsst::geom::AffineTransform::Parameters::Y] = solution[affineStart + 3];
        affine[lsst::geom::AffineTransform::Parameters::YX] = solution[affineStart + 4];
        affine[lsst::geom::AffineTransform::Parameters::YY] = solution[affineStart + 5];

        return std::make_tuple(
            affine,
            solution[ndarray::view(xStart, xStart + xPoly.getNParameters())],
            solution[ndarray::view(yStart, yStart + yPoly.getNParameters())]
        );
    }

    float xMiddle;  // Middle x value: boundary between CCDs
    MosaicPolynomialDistortion::Polynomial xPoly, yPoly;  // Polynomials used for calculating design
    std::size_t affineStart, xStart, yStart;  // Starting indices for affine, x, y in the design matrix
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
    ndarray::Array<double, 1, 1> const& slope,
    double threshold,
    ndarray::Array<bool, 1, 1> const& forced,
    ndarray::Array<double, 1, 1> const& params
) {
    using Array1D = MosaicPolynomialDistortion::Array1D;
    using Array2D = MosaicPolynomialDistortion::Array2D;
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
    return MosaicPolynomialDistortion(
        xOrder, yOrder, range, std::get<0>(solution), std::get<1>(solution), std::get<2>(solution)
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
