#include <set>
#include <numeric>
#include <algorithm>

#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/FCNBase.h"

//#define DEBUG

#ifdef DEBUG
#include "Minuit2/MnPrint.h"
#endif

#include "ndarray.h"
#include "ndarray/eigen.h"

#include "lsst/afw/table.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/Persistable.cc"

#include "pfs/drp/stella/utils/checkSize.h"
#include "pfs/drp/stella/utils/math.h"
#include "pfs/drp/stella/GlobalDetectorModel.h"


namespace pfs {
namespace drp {
namespace stella {


lsst::geom::Point2D GlobalDetectorModelScaling::operator()(
    int fiberId,
    double wavelength
) const {
    double const xi = fiberPitch*fiberId;
    double const eta = (wavelength - wavelengthCenter)/dispersion;
    return lsst::geom::Point2D(xi, eta);
}


ndarray::Array<double, 2, 1> GlobalDetectorModelScaling::operator()(
    ndarray::Array<int, 1, 1> const& fiberId,
    ndarray::Array<double, 1, 1> const& wavelength
) const {
    std::size_t const length = fiberId.size();
    utils::checkSize(length, wavelength.size(), "wavelength");
    ndarray::Array<double, 2, 1> xiEta = ndarray::allocate(length, 2);
    double const invDispersion = 1.0/dispersion;
    for (std::size_t ii = 0; ii < length; ++ii) {
        double const xi = fiberPitch*fiberId[ii];
        double const eta = (wavelength[ii] - wavelengthCenter)*invDispersion;
        xiEta[ii][0] = xi;
        xiEta[ii][1] = eta;
    }
    return xiEta;
}


lsst::geom::Point2D GlobalDetectorModelScaling::inverse(
    lsst::geom::Point2D const& xiEta
) const {
    double const fiberId = xiEta.getX()/fiberPitch;
    double const wavelength = xiEta.getY()*dispersion + wavelengthCenter;
    return lsst::geom::Point2D(fiberId, wavelength);
}


lsst::geom::Box2D GlobalDetectorModelScaling::getRange() const {
    float const maxFactor = 1.0 + buffer;
    float const minFactor = 1.0 - buffer;
    double const xiMin = fiberPitch*(minFiberId - 1)*minFactor;
    double const xiMax = fiberPitch*(maxFiberId + 1)*maxFactor;
    double const etaMin = -0.5*height*maxFactor;
    double const etaMax = 0.5*height*maxFactor;
    return lsst::geom::Box2D(lsst::geom::Point2D(xiMin, etaMin), lsst::geom::Point2D(xiMax, etaMax));
}


std::ostream& operator<<(std::ostream& os, GlobalDetectorModelScaling const& model) {
    os << "GlobalDetectorModelScaling(";
    os << "fiberPitch=" << model.fiberPitch << ", ";
    os << "dispersion=" << model.dispersion << ", ";
    os << "wavelengthCenter=" << model.wavelengthCenter << ")";
    return os;
}


FiberMap::FiberMap(
    ndarray::Array<int, 1, 1> const& fiberId
) {
    std::set<int> unique(fiberId.begin(), fiberId.end());  // sorted, to ensure order is consistent
    _map.reserve(unique.size());
    std::size_t ii = 0;
    for (auto iter = unique.begin(); iter != unique.end(); ++iter, ++ii) {
        _map[*iter] = ii;
    }
}


ndarray::Array<std::size_t, 1, 1> FiberMap::operator()(
    ndarray::Array<int, 1, 1> const& fiberId
) const {
    std::size_t const length = fiberId.size();
    ndarray::Array<std::size_t, 1, 1> index = ndarray::allocate(length);
    std::transform(fiberId.begin(), fiberId.end(), index.begin(),
                   [this](int ff) { return operator()(ff); });
    return index;
}


std::ostream& operator<<(std::ostream& os, FiberMap const& fiberMap) {
    os << "FiberMap(";
    bool comma = false;
    for (auto const& ii : fiberMap._map) {
        if (comma) {
            os << ", ";
        } else {
            comma = true;
        }
        os << ii.first << ": " << ii.second;
    }
    os << ")";
    return os;
}


GlobalDetectorModel::GlobalDetectorModel(
    lsst::geom::Box2I const& bbox,
    int distortionOrder,
    ndarray::Array<int, 1, 1> const& fiberId,
    GlobalDetectorModelScaling const& scaling,
    float fiberCenter,
    ndarray::Array<double, 1, 1> const& xDistortion,
    ndarray::Array<double, 1, 1> const& yDistortion,
    ndarray::Array<double, 1, 1> const& highCcd,
    ndarray::Array<double, 1, 1> const& spatialOffsets,
    ndarray::Array<double, 1, 1> const& spectralOffsets
) : GlobalDetectorModel(
        distortionOrder, FiberMap(fiberId), scaling, fiberCenter, bbox.getHeight(),
        xDistortion, yDistortion, highCcd, spatialOffsets, spectralOffsets)
{}


GlobalDetectorModel::GlobalDetectorModel(
    int distortionOrder,
    FiberMap const& fiberMap,
    GlobalDetectorModelScaling const& scaling,
    float fiberCenter,
    std::size_t height,
    ndarray::Array<double, 1, 1> const& xDistortion,
    ndarray::Array<double, 1, 1> const& yDistortion,
    ndarray::Array<double, 1, 1> const& highCcd,
    ndarray::Array<double, 1, 1> const& spatialOffsets,
    ndarray::Array<double, 1, 1> const& spectralOffsets
) : _distortionOrder(distortionOrder),
    _fiberMap(fiberMap),
    _scaling(scaling),
    _fiberCenter(fiberCenter),
    _xDistortion(utils::arrayToVector(xDistortion), scaling.getRange()),
    _yDistortion(utils::arrayToVector(yDistortion), scaling.getRange()),
    _highCcd(),
    _spatialOffsets(spatialOffsets),
    _spectralOffsets(spectralOffsets)
{
    std::size_t const numDistortion = getNumDistortion(distortionOrder);
    utils::checkSize(xDistortion.size(), numDistortion, "xDistortion");
    utils::checkSize(yDistortion.size(), numDistortion, "yDistortion");

    _highCcd.setParameterVector(ndarray::asEigenArray(highCcd));

    std::size_t const numFibers = fiberMap.size();
    if (spatialOffsets.isEmpty()) {
        _spatialOffsets = ndarray::allocate(numFibers);
        _spatialOffsets.deep() = 0.0;
    } else {
        utils::checkSize(spatialOffsets.size(), numFibers, "spatial offsets");
    }
    if (spectralOffsets.isEmpty()) {
        _spectralOffsets = ndarray::allocate(numFibers);
        _spectralOffsets.deep() = 0.0;
    } else {
        utils::checkSize(spectralOffsets.size(), numFibers, "spectral offsets");
    }
}


ndarray::Array<int, 1, 1> GlobalDetectorModel::getFiberId() const {
    ndarray::Array<int, 1, 1> fiberId = ndarray::allocate(getNumFibers());
    for (auto & ff : _fiberMap) {
        fiberId[ff.second] = ff.first;
    }
    return fiberId;
}


lsst::geom::Point2D GlobalDetectorModel::operator()(
    lsst::geom::Point2D const& xiEta,
    std::size_t fiberIndex,
    bool onHighCcd
) const {
    double const xi = xiEta[0] + getSpatialOffset(fiberIndex);
    double const eta = xiEta[1] + getSpectralOffset(fiberIndex);
    if (!getXDistortion().getXYRange().contains(lsst::geom::Point2D(xi, eta))) {
        throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeError,
                          (boost::format("xi,eta=(%f,%f) fiberId,wavelength=(%s) is out of range of %s") %
                           xi % eta % getScaling().inverse(xiEta) % getXDistortion().getXYRange()).str());
    }
    // x,y: distorted position
    lsst::geom::Point2D xy(getXDistortion()(xi, eta), getYDistortion()(xi, eta));
    // x,y: detector coordinates
    if (onHighCcd) {
        xy += lsst::geom::Extent2D(getHighCcd()(xiEta));
    }
    return xy;
}


ndarray::Array<double, 2, 1> GlobalDetectorModel::operator()(
    ndarray::Array<double, 2, 1> const& xiEta,
    ndarray::Array<std::size_t, 1, 1> const& fiberIndex,
    ndarray::Array<bool, 1, 1> const& onHighCcd
) const {
    std::size_t const length = xiEta.getShape()[0];
    utils::checkSize(xiEta.getShape()[1], 2UL, "xiEta");
    utils::checkSize(fiberIndex.getNumElements(), length, "fiberIndex");
    utils::checkSize(onHighCcd.getNumElements(), length, "onHighCcd");
    ndarray::Array<double, 2, 1> out = ndarray::allocate(length, 2);
    for (std::size_t ii = 0; ii < length; ++ii) {
        auto const point = operator()(lsst::geom::Point2D(xiEta[ii][0], xiEta[ii][1]),
                                      fiberIndex[ii], onHighCcd[ii]);
        out[ii][0] = point.getX();
        out[ii][1] = point.getY();
    }
    return out;
}


ndarray::Array<double, 1, 1> GlobalDetectorModel::getXCoefficients() const {
    return ndarray::copy(utils::vectorToArray(getXDistortion().getParameters()));
}


ndarray::Array<double, 1, 1> GlobalDetectorModel::getYCoefficients() const {
    return ndarray::copy(utils::vectorToArray(getYDistortion().getParameters()));
}


ndarray::Array<double, 1, 1> GlobalDetectorModel::getHighCcdCoefficients() const {
    auto const parameters = _highCcd.getParameterVector();  // An Eigen matrix
    assert(parameters.size() == 6);
    ndarray::Array<double, 1, 1> out = ndarray::allocate(parameters.size());
    ndarray::asEigenMatrix(out) = parameters;
    return out;
}


ndarray::Array<double, 1, 1> GlobalDetectorModel::makeHighCcdCoefficients(
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


std::ostream& operator<<(std::ostream& os, GlobalDetectorModel const& model) {
    os << "GlobalDetectorModel(";
    os << "fiberPitch=" << model.getScaling().fiberPitch << ", ";
    os << "dispersion=" << model.getScaling().dispersion << ", ";
    os << "wavelengthCenter=" << model.getScaling().wavelengthCenter << ", ";
    os << "xDistortion=" << model.getXCoefficients() << ", ";
    os << "yDistortion=" << model.getYCoefficients() << ", ";
    os << "highCcd=" << model.getHighCcdCoefficients() << ", ";
    os << "spatialOffsets=" << model.getSpatialOffsets() << ", ";
    os << "spectralOffsets=" << model.getSpectralOffsets() << ")";
    return os;
}


ndarray::Array<double, 2, 1> GlobalDetectorModel::calculateDesignMatrix(
    int distortionOrder,
    lsst::geom::Box2D const& xiEtaRange,
    ndarray::Array<double, 2, 1> const& xiEta
) {
    std::size_t const length = xiEta.getShape()[0];
    utils::checkSize(xiEta.getShape()[1], 2UL, "xiEta");

    std::size_t const numTerms = GlobalDetectorModel::getNumDistortion(distortionOrder);
    ndarray::Array<double, 2, 1> matrix = ndarray::allocate(length, numTerms);
    Polynomial const poly(distortionOrder, xiEtaRange);
    for (std::size_t ii = 0; ii < length; ++ii) {
        double const xi = xiEta[ii][0];
        double const eta = xiEta[ii][1];
        auto const terms = poly.getDFuncDParameters(xi, eta);
        std::copy(terms.begin(), terms.end(), matrix[ii].begin());
    }
    return matrix;
}


std::pair<double, std::size_t> GlobalDetectorModel::calculateChi2(
    ndarray::Array<double, 2, 1> const& xiEta,
    ndarray::Array<std::size_t, 1, 1> const& fiberIndex,
    ndarray::Array<bool, 1, 1> const& onHighCcd,
    ndarray::Array<double, 1, 1> const& xx,
    ndarray::Array<double, 1, 1> const& yy,
    ndarray::Array<double, 1, 1> const& xErr,
    ndarray::Array<double, 1, 1> const& yErr,
    ndarray::Array<bool, 1, 1> const& goodOrig,
    float sysErr
) const {
    std::size_t const length = xiEta.getShape()[0];
    utils::checkSize(xiEta.getShape()[1], 2UL, "xiEta");
    utils::checkSize(fiberIndex.size(), length, "fiberIndex");
    utils::checkSize(xx.size(), length, "x");
    utils::checkSize(yy.size(), length, "y");
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
        double const xMeas = xx[ii];
        double const yMeas = yy[ii];
        double const xErr2 = std::pow(xErr[ii], 2) + sysErr2;
        double const yErr2 = std::pow(yErr[ii], 2) + sysErr2;
        std::size_t const index = fiberIndex[ii];
        lsst::geom::Point2D const fit = operator()(lsst::geom::Point2D(xiEta[ii][0], xiEta[ii][1]),
                                                   index, onHighCcd[ii]);
        chi2 += std::pow(xMeas - fit.getX(), 2)/xErr2 + std::pow(yMeas - fit.getY(), 2)/yErr2;
        num += 2;  // one for x, one for y
    }
    std::size_t const numFitParams = 2*getNumDistortion() + 2*getNumFibers() + 6;
    std::size_t const dof = num - numFitParams;
    return std::make_pair(chi2, dof);
}


namespace {

class SlitOffsetMinimization : public ROOT::Minuit2::FCNBase {
  public:
    SlitOffsetMinimization(
        GlobalDetectorModel const& model,
        std::size_t selectFiberIndex,
        ndarray::Array<double, 2, 1> const& xiEta,
        ndarray::Array<std::size_t, 1, 1> const& fiberIndex,
        ndarray::Array<bool, 1, 1> const& onHighCcd,
        ndarray::Array<double, 1, 1> const& x,
        ndarray::Array<double, 1, 1> const& y,
        ndarray::Array<double, 1, 1> const& xErr,
        ndarray::Array<double, 1, 1> const& yErr
    ) : _model(model),
        _selectFiberIndex(selectFiberIndex),
        _num(fiberIndex.size()),
        _xiEta(xiEta),
        _fiberIndex(fiberIndex),
        _onHighCcd(onHighCcd),
        _x(x),
        _y(y),
        _xErr(xErr),
        _yErr(yErr),
        _select(ndarray::allocate(fiberIndex.size())) {
            assert(selectFiberIndex < model.getNumFibers());
            assert(xiEta.getShape()[0] == _num);
            assert(xiEta.getShape()[1] == 2);
            assert(fiberIndex.size() == _num);
            assert(onHighCcd.size() == _num);
            assert(_x.size() == _num);
            assert(_y.size() == _num);
            assert(_xErr.size() == _num);
            assert(_yErr.size() == _num);
            for (std::size_t ii = 0; ii < _num; ++ii) {
                _select[ii] = fiberIndex[ii] == selectFiberIndex;
            }
        }

    SlitOffsetMinimization(SlitOffsetMinimization const &) = default;
    SlitOffsetMinimization(SlitOffsetMinimization &&) = default;
    SlitOffsetMinimization &operator=(SlitOffsetMinimization const &) = default;
    SlitOffsetMinimization &operator=(SlitOffsetMinimization &&) = default;
    virtual ~SlitOffsetMinimization() override = default;

    double calculateChi2(lsst::geom::Point2D offset) const;

    lsst::geom::Point2D minimize() const;

    double operator()(std::vector<double> const& parameters) const override;
    double Up() const override { return 1.0; }  // 1.0 --> fitting chi^2/dof

  protected:
    mutable GlobalDetectorModel _model;
    std::size_t _selectFiberIndex;
    std::size_t _num;
    ndarray::Array<double, 2, 1> const _xiEta;
    ndarray::Array<std::size_t, 1, 1> const& _fiberIndex;
    ndarray::Array<bool, 1, 1> const& _onHighCcd;
    ndarray::Array<double, 1, 1> const _x;
    ndarray::Array<double, 1, 1> const _y;
    ndarray::Array<double, 1, 1> const _xErr;
    ndarray::Array<double, 1, 1> const _yErr;
    ndarray::Array<bool, 1, 1> const _select;
};


double SlitOffsetMinimization::operator()(std::vector<double> const& parameters) const {
    // Modify model's slit offsets
    _model.getSpatialOffsets()[_selectFiberIndex] = parameters[0];
    _model.getSpectralOffsets()[_selectFiberIndex] = parameters[1];

    double chi2;
    std::size_t const dof = _num - 2;
    try {
        chi2 = _model.calculateChi2(_xiEta, _fiberIndex, _onHighCcd, _x, _y, _xErr, _yErr, _select).first;
    } catch (...) {
        chi2 = std::numeric_limits<double>::infinity();
    }
    if (std::isnan(chi2)) {
        chi2 = std::numeric_limits<double>::infinity();
    }
    return chi2/dof;
}


lsst::geom::Point2D SlitOffsetMinimization::minimize() const {
    std::vector<double> parameters = {0.0, 0.0};
    std::vector<double> steps = {0.1, 0.1};

#ifdef DEBUG
    ROOT::Minuit2::MnPrint::SetLevel(3);
#endif
    auto const min = ROOT::Minuit2::MnMigrad(*this, parameters, steps)();
    assert(min.UserParameters().Params().size() == 2);

    if (!min.IsValid() || !std::isfinite(min.Fval())) {
        return lsst::geom::Point2D(0.0, 0.0);
    }
    return lsst::geom::Point2D(min.UserParameters().Params()[0], min.UserParameters().Params()[1]);
}

}  // anonymous namespace


ndarray::Array<double, 2, 1> GlobalDetectorModel::measureSlitOffsets(
    ndarray::Array<double, 2, 1> const& xiEta,
    ndarray::Array<std::size_t, 1, 1> const& fiberIndex,
    ndarray::Array<bool, 1, 1> const& onHighCcd,
    ndarray::Array<double, 1, 1> const& xx,
    ndarray::Array<double, 1, 1> const& yy,
    ndarray::Array<double, 1, 1> const& xErr,
    ndarray::Array<double, 1, 1> const& yErr
) {
    std::size_t const length = xiEta.getShape()[0];
    utils::checkSize(xiEta.getShape()[1], 2UL, "xiEta");
    utils::checkSize(fiberIndex.size(), length, "fiberIndex");
    utils::checkSize(xx.size(), length, "x");
    utils::checkSize(yy.size(), length, "y");
    utils::checkSize(xErr.size(), length, "xErr");
    utils::checkSize(yErr.size(), length, "yErr");
    std::size_t const numFibers = getNumFibers();

    ndarray::Array<double, 2, 1> offsets = ndarray::allocate(numFibers, 2);
    for (std::size_t ii = 0; ii < numFibers; ++ii) {
        lsst::geom::Point2D const result = SlitOffsetMinimization(*this, ii, xiEta, fiberIndex, onHighCcd,
                                                                  xx, yy, xErr, yErr).minimize();
        offsets[ii][0] = result.getX();
        offsets[ii][1] = result.getY();
    }

    // Modify model's slit offsets
    for (std::size_t ii = 0; ii < numFibers; ++ii) {
        _spatialOffsets[ii] = offsets[ii][0];
        _spectralOffsets[ii] = offsets[ii][1];
    }

    return offsets;
}


ndarray::Array<bool, 1, 1> GlobalDetectorModel::getOnHighCcd(
    ndarray::Array<int, 1, 1> const& fiberId
) const {
    ndarray::Array<bool, 1, 1> out = ndarray::allocate(fiberId.size());
    std::transform(fiberId.begin(), fiberId.end(), out.begin(),
                   [this](int ff) { return getOnHighCcd(ff); });
    return out;
}


namespace {

// Singleton class that manages the persistence catalog's schema and keys
class GlobalDetectorModelSchema {
    using IntArray = lsst::afw::table::Array<int>;
    using DoubleArray = lsst::afw::table::Array<double>;
  public:
    lsst::afw::table::Schema schema;
    lsst::afw::table::Key<int> distortionOrder;
    lsst::afw::table::Key<double> fiberPitch;
    lsst::afw::table::Key<double> dispersion;
    lsst::afw::table::Key<double> wavelengthCenter;
    lsst::afw::table::Key<int> height;
    lsst::afw::table::Key<float> buffer;
    lsst::afw::table::Key<float> fiberCenter;
    lsst::afw::table::Key<IntArray> fiberId;
    lsst::afw::table::Key<DoubleArray> xCoefficients;
    lsst::afw::table::Key<DoubleArray> yCoefficients;
    lsst::afw::table::Key<DoubleArray> highCcd;
    lsst::afw::table::Key<DoubleArray> spatialOffset;
    lsst::afw::table::Key<DoubleArray> spectralOffset;
    lsst::afw::table::Key<int> visitInfo;

    static GlobalDetectorModelSchema const &get() {
        static GlobalDetectorModelSchema const instance;
        return instance;
    }

  private:
    GlobalDetectorModelSchema()
      : schema(),
        distortionOrder(schema.addField<int>("distortionOrder", "polynomial order for distortion", "")),
        fiberPitch(schema.addField<double>("fiberPitch", "distance between fibers", "pixel")),
        dispersion(schema.addField<double>("dispersion", "wavelength dispersion", "nm/pixel")),
        wavelengthCenter(schema.addField<double>("wavelengthCenter", "central wavelength", "nm")),
        height(schema.addField<int>("height", "height of detector", "pixel")),
        buffer(schema.addField<float>("buffer", "fraction by which to expand wavelength range", "")),
        fiberCenter(schema.addField<float>("fiberCenter", "central x value", "pixel")),
        fiberId(schema.addField<IntArray>("fiberId", "fiber identifiers", "", 0)),
        xCoefficients(schema.addField<DoubleArray>("xCoefficients", "x distortion coefficients", "", 0)),
        yCoefficients(schema.addField<DoubleArray>("yCoefficients", "y distortion coefficients", "", 0)),
        highCcd(schema.addField<DoubleArray>("highCcd", "affine transform for high-fiberId Ccd", "", 0)),
        spatialOffset(schema.addField<DoubleArray>("spatialOffset", "slit offsets in x", "micron", 0)),
        spectralOffset(schema.addField<DoubleArray>("spectralOffset", "slit offsets in y", "micron", 0)) {
            schema.getCitizen().markPersistent();
    }
};

}  // anonymous namespace


void GlobalDetectorModel::write(lsst::afw::table::io::OutputArchiveHandle & handle) const {
    GlobalDetectorModelSchema const &schema = GlobalDetectorModelSchema::get();
    lsst::afw::table::BaseCatalog cat = handle.makeCatalog(schema.schema);
    PTR(lsst::afw::table::BaseRecord) record = cat.addNew();
    record->set(schema.distortionOrder, getDistortionOrder());
    record->set(schema.fiberPitch, getFiberPitch());
    record->set(schema.dispersion, getDispersion());
    record->set(schema.wavelengthCenter, getWavelengthCenter());
    record->set(schema.height, getHeight());
    record->set(schema.buffer, getBuffer());
    record->set(schema.buffer, _fiberCenter);
    ndarray::Array<int, 1, 1> const fiberId = ndarray::copy(getFiberId());
    record->set(schema.fiberId, fiberId);
    ndarray::Array<double, 1, 1> xCoeff = ndarray::copy(getXCoefficients());
    record->set(schema.xCoefficients, xCoeff);
    ndarray::Array<double, 1, 1> yCoeff = ndarray::copy(getYCoefficients());
    record->set(schema.yCoefficients, yCoeff);
    ndarray::Array<double, 1, 1> highCcd = ndarray::copy(getHighCcdCoefficients());
    record->set(schema.highCcd, highCcd);
    ndarray::Array<double, 1, 1> spatialOffset = ndarray::copy(getSpatialOffsets());
    record->set(schema.spatialOffset, spatialOffset);
    ndarray::Array<double, 1, 1> spectralOffset = ndarray::copy(getSpectralOffsets());
    record->set(schema.spectralOffset, spectralOffset);
    handle.saveCatalog(cat);
}


class GlobalDetectorModel::Factory : public lsst::afw::table::io::PersistableFactory {
  public:
    std::shared_ptr<lsst::afw::table::io::Persistable> read(
        lsst::afw::table::io::InputArchive const& archive,
        lsst::afw::table::io::CatalogVector const& catalogs
    ) const override {
        static auto const& schema = GlobalDetectorModelSchema::get();
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        lsst::afw::table::BaseRecord const& record = catalogs.front().front();
        LSST_ARCHIVE_ASSERT(record.getSchema() == schema.schema);

        int const distortionOrder = record.get(schema.distortionOrder);
        double const fiberPitch = record.get(schema.fiberPitch);
        double const dispersion = record.get(schema.dispersion);
        double const wavelengthCenter = record.get(schema.wavelengthCenter);
        int const height = record.get(schema.height);
        float const buffer = record.get(schema.buffer);
        float const fiberCenter = record.get(schema.fiberCenter);
        ndarray::Array<int, 1, 1> fiberId = ndarray::copy(record.get(schema.fiberId));
        ndarray::Array<double, 1, 1> xCoeff = ndarray::copy(record.get(schema.xCoefficients));
        ndarray::Array<double, 1, 1> yCoeff = ndarray::copy(record.get(schema.yCoefficients));
        ndarray::Array<double, 1, 1> highCcd = ndarray::copy(record.get(schema.highCcd));
        ndarray::Array<double, 1, 1> spatialOffset = ndarray::copy(record.get(schema.spatialOffset));
        ndarray::Array<double, 1, 1> spectralOffset = ndarray::copy(record.get(schema.spectralOffset));
        assert(spatialOffset.getNumElements() == fiberId.size());
        assert(spectralOffset.getNumElements() == fiberId.size());

        return std::make_shared<GlobalDetectorModel>(
            distortionOrder, FiberMap(fiberId),
            GlobalDetectorModelScaling(
                fiberPitch, dispersion, wavelengthCenter,
                *std::min_element(fiberId.begin(), fiberId.end()),
                *std::max_element(fiberId.begin(), fiberId.end()),
                height, buffer),
            fiberCenter, height, xCoeff, yCoeff, highCcd, spatialOffset, spectralOffset
        );
    }

    Factory(std::string const& name) : lsst::afw::table::io::PersistableFactory(name) {}
};


namespace {

GlobalDetectorModel::Factory registration("GlobalDetectorModel");

}  // anonymous namespace



}}}  // namespace pfs::drp::stella
