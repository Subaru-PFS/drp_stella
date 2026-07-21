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
    int order,
    lsst::geom::Box2D const& range,
    MosaicPolynomialDistortion::Array1D const& coeff
) : MosaicPolynomialDistortion(order, range, splitCoefficients(order, coeff))
{}


MosaicPolynomialDistortion::MosaicPolynomialDistortion(
    int order,
    lsst::geom::Box2D const& range,
    MosaicPolynomialDistortion::Array1D const& affineCoeff,
    MosaicPolynomialDistortion::Array1D const& xCoeff,
    MosaicPolynomialDistortion::Array1D const& yCoeff
) : AnalyticDistortion<MosaicPolynomialDistortion>(
        order, range, joinCoefficients(order, affineCoeff, xCoeff, yCoeff)
    ),
    _affine(std::move(math::makeAffineTransform(affineCoeff))),
    _poly(order, range, xCoeff, yCoeff)
{}


template<> std::size_t AnalyticDistortion<MosaicPolynomialDistortion>::getNumParametersForOrder(int order) {
    return 6 + 2*MosaicPolynomialDistortion::getNumDistortionForOrder(order);
}


std::tuple<MosaicPolynomialDistortion::Array1D, MosaicPolynomialDistortion::Array2D>
MosaicPolynomialDistortion::splitCoefficients(
    int order,
    ndarray::Array<double, 1, 1> const& coeff
) {
    utils::checkSize(coeff.size(), MosaicPolynomialDistortion::getNumParametersForOrder(order), "coeff");
    std::size_t const numDistortion = MosaicPolynomialDistortion::getNumDistortionForOrder(order);
    Array1D affineCoeff = coeff[ndarray::view(0, NUM_AFFINE)];
    Array2D polyCoeff = ndarray::allocate(2, numDistortion);
    for (std::size_t ii = 0; ii < 2; ++ii) {
        std::size_t const start = NUM_AFFINE + ii*numDistortion;
        std::size_t const stop = NUM_AFFINE + (ii + 1)*numDistortion;
        polyCoeff[ndarray::view(ii)] = coeff[ndarray::view(start, stop)];
    }
    return std::make_tuple(affineCoeff, polyCoeff);
}


MosaicPolynomialDistortion::Array1D MosaicPolynomialDistortion::joinCoefficients(
    int order,
    MosaicPolynomialDistortion::Array1D const& affineCoeff,
    MosaicPolynomialDistortion::Array1D const& xCoeff,
    MosaicPolynomialDistortion::Array1D const& yCoeff
) {
    utils::checkSize(affineCoeff.size(), NUM_AFFINE, "affineCoeff");
    std::size_t const numDistortion = getNumDistortionForOrder(order);
    utils::checkSize(xCoeff.size(), numDistortion, "xCoeff");
    utils::checkSize(yCoeff.size(), numDistortion, "yCoeff");
    Array1D coeff = ndarray::allocate(NUM_AFFINE + 2*numDistortion);
    coeff[ndarray::view(0, NUM_AFFINE)] = affineCoeff;
    coeff[ndarray::view(NUM_AFFINE, NUM_AFFINE + numDistortion)] = xCoeff;
    coeff[ndarray::view(NUM_AFFINE + numDistortion, NUM_AFFINE + 2*numDistortion)] = yCoeff;
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
    os << "order=" << model.getOrder() << ", ";
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
    // @param order : Polynomial order.
    // @param numLines_ : Number of arc line measurements.
    // @param numTrace_ : Number of binned trace measurements.
    FitData(
        lsst::geom::Box2D const& range, int order,
        std::size_t numLines_, std::size_t numTrace_
    ) :
        xMiddle(range.getCenterX()),
        poly(order, range),
        numLines(numLines_),
        numTrace(numTrace_),
        length(2*numLines + 2*numTrace),
        measurements(ndarray::allocate(length)),
        errors(ndarray::allocate(length)),
        design(ndarray::allocate(length, 2*poly.getNParameters() + NUM_AFFINE)),
        index(0) {
        design.deep() = 0.0;
    }

    // Add an arc line measurement to the design matrix
    //
    // Contributes one row for x and one row for y.
    //
    // @param xy : Point at which to evaluate the polynomial
    // @param meas : Measured residual
    // @param err : Error in measured residual
    void add(
        lsst::geom::Point2D const& xy,
        lsst::geom::Point2D const& meas,
        lsst::geom::Point2D const& err
    ) {
        std::size_t const ii = index++;
        assert(ii < length);

        auto const terms = poly.getDFuncDParameters(xy.getX(), xy.getY());
        std::size_t const numPoly = poly.getNParameters();
        assert (terms.size() == numPoly);

        // x part of the design matrix
        std::copy(terms.begin(), terms.end(), design[ii].begin());
        measurements[ii] = meas.getX();
        errors[ii] = err.getX();

        lsst::geom::Point2D const xyNorm = poly.normalize(xy);

        // Affine part of the design matrix
        bool const onRightCcd = xy.getX() > xMiddle;
        std::size_t const affineStart = 2*numPoly;
        if (onRightCcd) {
            design[ii][affineStart] = 1.0;
            design[ii][affineStart + 1] = xyNorm.getX();
            design[ii][affineStart + 2] = xyNorm.getY();
        }

        // y part of the design matrix (independent of x part)
        std::size_t const jj = index++;
        assert(jj < length);
        std::copy(terms.begin(), terms.end(), design[jj].begin() + numPoly);
        measurements[jj] = meas.getY();
        errors[jj] = err.getY();

        if (onRightCcd) {
            design[jj][affineStart + 3] = 1.0;
            design[jj][affineStart + 4] = xyNorm.getX();
            design[jj][affineStart + 5] = xyNorm.getY();
        }
    }

    // Add a binned trace measurement to the design matrix
    //
    // Contributes two rows, both in x-polynomial columns only:
    //   - one for the position constraint at the bin center
    //   - one for the slope constraint d(delta_x)/dy
    //
    // @param xy : Position at which to evaluate the constraints
    // @param posMeas : Measured x residual at bin center
    // @param posErr : Error in x residual
    // @param slopeMeas : Measured slope residual d(x_resid)/dy
    // @param slopeErr : Error in slope
    void addTrace(
        lsst::geom::Point2D const& xy,
        double posMeas, double posErr,
        double slopeMeas, double slopeErr
    ) {
        std::size_t const numPoly = poly.getNParameters();
        std::size_t const affineStart = 2*numPoly;
        bool const onRightCcd = xy.getX() > xMiddle;
        lsst::geom::Point2D const xyNorm = poly.normalize(xy);

        // Position row: x-polynomial only; y-polynomial and affine-y columns remain zero.
        std::size_t const ii = index++;
        assert(ii < length);
        auto const terms = poly.getDFuncDParameters(xy.getX(), xy.getY());
        std::copy(terms.begin(), terms.end(), design[ii].begin());
        if (onRightCcd) {
            design[ii][affineStart] = 1.0;
            design[ii][affineStart + 1] = xyNorm.getX();
            design[ii][affineStart + 2] = xyNorm.getY();
        }
        measurements[ii] = posMeas;
        errors[ii] = posErr;

        // Slope row: numerical derivative of x-polynomial w.r.t. y.
        // Affine x-part: d(a0 + a1*x_norm + a2*y_norm)/dy via finite difference.
        // Affine y-part: slope constrains delta_y so poorly (via s*d(delta_y)/dy, s~0.016) it's negligible.
        std::size_t const jj = index++;
        assert(jj < length);
        auto const termsPlus = poly.getDFuncDParameters(xy.getX(), xy.getY() + 1.0);
        auto const termsMinus = poly.getDFuncDParameters(xy.getX(), xy.getY() - 1.0);
        for (std::size_t kk = 0; kk < numPoly; ++kk) {
            design[jj][kk] = 0.5*(termsPlus[kk] - termsMinus[kk]);
        }
        if (onRightCcd) {
            lsst::geom::Point2D const xyNormPlus = poly.normalize(
                lsst::geom::Point2D(xy.getX(), xy.getY() + 1.0));
            lsst::geom::Point2D const xyNormMinus = poly.normalize(
                lsst::geom::Point2D(xy.getX(), xy.getY() - 1.0));
            design[jj][affineStart + 2] = 0.5*(xyNormPlus.getY() - xyNormMinus.getY());
        }
        measurements[jj] = slopeMeas;
        errors[jj] = slopeErr;
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

        std::size_t const nPolyParams = poly.getNParameters();
        std::size_t const affineStart = 2*nPolyParams;
        ndarray::Array<double, 1, 1> affine = ndarray::allocate(NUM_AFFINE);
        affine[lsst::geom::AffineTransform::Parameters::X] = solution[affineStart];
        affine[lsst::geom::AffineTransform::Parameters::XX] = solution[affineStart + 1];
        affine[lsst::geom::AffineTransform::Parameters::XY] = solution[affineStart + 2];
        affine[lsst::geom::AffineTransform::Parameters::Y] = solution[affineStart + 3];
        affine[lsst::geom::AffineTransform::Parameters::YX] = solution[affineStart + 4];
        affine[lsst::geom::AffineTransform::Parameters::YY] = solution[affineStart + 5];

        return std::make_tuple(
            affine,
            solution[ndarray::view(0, nPolyParams)],
            solution[ndarray::view(nPolyParams, affineStart)]
        );
    }

    float xMiddle;  // Middle x value: boundary between CCDs
    MosaicPolynomialDistortion::Polynomial poly;  // Polynomial used for calculating design
    std::size_t numLines;  // Number of arc line measurements
    std::size_t numTrace;  // Number of binned trace measurements
    std::size_t length;  // Total number of design matrix rows
    Array1D measurements;  // Measurements
    Array1D errors;  // Errors in measurements
    Array2D design;  // Design matrix
    std::size_t index;  // Next index to add
};


}  // anonymous namespace



template<>
MosaicPolynomialDistortion AnalyticDistortion<MosaicPolynomialDistortion>::fit(
    int distortionOrder,
    lsst::geom::Box2D const& range,
    ndarray::Array<double, 1, 1> const& xx,
    ndarray::Array<double, 1, 1> const& yy,
    ndarray::Array<double, 1, 1> const& xMeas,
    ndarray::Array<double, 1, 1> const& yMeas,
    ndarray::Array<double, 1, 1> const& xErr,
    ndarray::Array<double, 1, 1> const& yErr,
    double threshold,
    ndarray::Array<bool, 1, 1> const& forced,
    ndarray::Array<double, 1, 1> const& params,
    ndarray::Array<double, 1, 1> const& xTrace,
    ndarray::Array<double, 1, 1> const& yTrace,
    ndarray::Array<double, 1, 1> const& tracePos,
    ndarray::Array<double, 1, 1> const& tracePosErr,
    ndarray::Array<double, 1, 1> const& traceSlope,
    ndarray::Array<double, 1, 1> const& traceSlopeErr
) {
    std::size_t const length = xx.size();
    utils::checkSize(yy.size(), length, "y");
    utils::checkSize(xMeas.size(), length, "xMeas");
    utils::checkSize(yMeas.size(), length, "yMeas");
    utils::checkSize(xErr.size(), length, "xErr");
    utils::checkSize(yErr.size(), length, "yErr");

    std::size_t const numLines = length;
    std::size_t const numTrace = xTrace.size();
    utils::checkSize(yTrace.size(), numTrace, "yTrace");
    utils::checkSize(tracePos.size(), numTrace, "tracePos");
    utils::checkSize(tracePosErr.size(), numTrace, "tracePosErr");
    utils::checkSize(traceSlope.size(), numTrace, "traceSlope");
    utils::checkSize(traceSlopeErr.size(), numTrace, "traceSlopeErr");

    FitData fit(range, distortionOrder, numLines, numTrace);
    for (std::size_t ii = 0; ii < length; ++ii) {
        fit.add(
            lsst::geom::Point2D(xx[ii], yy[ii]),
            lsst::geom::Point2D(xMeas[ii], yMeas[ii]),
            lsst::geom::Point2D(xErr[ii], yErr[ii])
        );
    }
    for (std::size_t ii = 0; ii < numTrace; ++ii) {
        fit.addTrace(lsst::geom::Point2D(xTrace[ii], yTrace[ii]),
                     tracePos[ii], tracePosErr[ii], traceSlope[ii], traceSlopeErr[ii]);
    }

    auto const solution = fit.getSolution(threshold, forced, params);
    return MosaicPolynomialDistortion(
        distortionOrder, range, std::get<0>(solution), std::get<1>(solution), std::get<2>(solution)
    );
}


namespace {

// Singleton class that manages the persistence catalog's schema and keys
class MosaicPolynomialDistortionSchema {
    using IntArray = lsst::afw::table::Array<int>;
    using DoubleArray = lsst::afw::table::Array<double>;
  public:
    lsst::afw::table::Schema schema;
    lsst::afw::table::Key<int> distortionOrder;
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
        distortionOrder(schema.addField<int>("distortionOrder", "polynomial order for distortion", "")),
        range(lsst::afw::table::Box2DKey::addFields(schema, "range", "range of input values", "pixel")),
        coefficients(schema.addField<DoubleArray>("coefficients", "distortion coefficients", "", 0))
        {}
};

}  // anonymous namespace


void MosaicPolynomialDistortion::write(lsst::afw::table::io::OutputArchiveHandle & handle) const {
    MosaicPolynomialDistortionSchema const &schema = MosaicPolynomialDistortionSchema::get();
    lsst::afw::table::BaseCatalog cat = handle.makeCatalog(schema.schema);
    std::shared_ptr<lsst::afw::table::BaseRecord> record = cat.addNew();
    record->set(schema.distortionOrder, getOrder());
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

        int const distortionOrder = record.get(schema.distortionOrder);
        lsst::geom::Box2D const range = record.get(schema.range);
        ndarray::Array<double, 1, 1> coeff = ndarray::copy(record.get(schema.coefficients));

        return std::make_shared<MosaicPolynomialDistortion>(distortionOrder, range, coeff);
    }

    Factory(std::string const& name) : lsst::afw::table::io::PersistableFactory(name) {}
};


namespace {

MosaicPolynomialDistortion::Factory registration("MosaicPolynomialDistortion");

}  // anonymous namespace


// Explicit instantiation
template class AnalyticDistortion<MosaicPolynomialDistortion>;


}}}  // namespace pfs::drp::stella
