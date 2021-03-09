#include <set>
#include <numeric>

#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/FCNBase.h"

//#define DEBUG

#ifdef DEBUG
#include "Minuit2/MnPrint.h"
#endif

#include "ndarray.h"

#include "lsst/utils/Cache.h"
#include "lsst/afw/table.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/Persistable.cc"
#include "lsst/afw/math/Statistics.h"

#include "pfs/drp/stella/utils/checkSize.h"
#include "pfs/drp/stella/GlobalDetectorMap.h"


namespace pfs {
namespace drp {
namespace stella {


GlobalDetectorMap::GlobalDetectorMap(
    lsst::geom::Box2I const& bbox,
    GlobalDetectorModel const& model,
    VisitInfo const& visitInfo,
    std::shared_ptr<lsst::daf::base::PropertySet> metadata
) : ModelBasedDetectorMap(bbox, model.getWavelengthCenter(), model.getDispersion(),
                          model.getFiberId(), model.getSpatialOffsets(), model.getSpectralOffsets(),
                          visitInfo, metadata),
    _model(model)
    {}


GlobalDetectorMap::GlobalDetectorMap(
    lsst::geom::Box2I const& bbox,
    int distortionOrder,
    ndarray::Array<int, 1, 1> const& fiberId,
    double fiberPitch,
    double dispersion,
    double wavelengthCenter,
    float buffer,
    float fiberCenter,
    ndarray::Array<double, 1, 1> const& xCoefficients,
    ndarray::Array<double, 1, 1> const& yCoefficients,
    ndarray::Array<double, 1, 1> const& highCcd,
    ndarray::Array<double, 1, 1> const& spatialOffsets,
    ndarray::Array<double, 1, 1> const& spectralOffsets,
    VisitInfo const& visitInfo,
    std::shared_ptr<lsst::daf::base::PropertySet> metadata
) : GlobalDetectorMap(
        bbox,
        GlobalDetectorModel(
            distortionOrder, fiberId,
            GlobalDetectorModelScaling(
                fiberPitch, dispersion, wavelengthCenter, *std::min_element(fiberId.begin(), fiberId.end()),
                *std::max_element(fiberId.begin(), fiberId.end()), buffer),
            fiberCenter, xCoefficients, yCoefficients, highCcd, spatialOffsets, spectralOffsets),
        visitInfo,
        metadata)
{}


std::shared_ptr<DetectorMap> GlobalDetectorMap::clone() const {
    return std::make_shared<GlobalDetectorMap>(getBBox(), getModel(), getVisitInfo(),
                                               getMetadata()->deepCopy());
}


lsst::geom::PointD GlobalDetectorMap::findPointImpl(
    int fiberId,
    double wavelength
) const {
    return _model(fiberId, wavelength);
}


void GlobalDetectorMap::_resetSlitOffsets() {
    _model = GlobalDetectorModel(
        _model.getDistortionOrder(), _model.getFiberId(),
        _model.getScaling(), _model.getFiberCenter(),
        _model.getXCoefficients(), _model.getYCoefficients(), _model.getHighCcdCoefficients(),
        getSpatialOffsets(), getSpectralOffsets()
    );
    ModelBasedDetectorMap::_resetSlitOffsets();
}



namespace {

class SlitOffsetMinimization : public ROOT::Minuit2::FCNBase {
  public:
    SlitOffsetMinimization(
        DetectorMap const& detectorMap,
        ndarray::Array<int, 1, 1> const& fiberId,
        ndarray::Array<double, 1, 1> const& wavelength,
        ndarray::Array<double, 1, 1> const& x,
        ndarray::Array<double, 1, 1> const& y,
        ndarray::Array<double, 1, 1> const& xErr,
        ndarray::Array<double, 1, 1> const& yErr
    ) : _detectorMap(detectorMap.clone()),
        _num(fiberId.size()),
        _fiberId(fiberId),
        _wavelength(wavelength),
        _x(x),
        _y(y),
        _xErr2(ndarray::allocate(fiberId.size())),
        _yErr2(ndarray::allocate(fiberId.size())) {
            assert(fiberId.size() == _num);
            assert(_wavelength.size() == _num);
            assert(_x.size() == _num);
            assert(_y.size() == _num);
            assert(xErr.size() == _num);
            assert(yErr.size() == _num);
            std::transform(xErr.begin(), xErr.end(), _xErr2.begin(),
                           [](double err) { return std::pow(err, 2); });
            std::transform(yErr.begin(), yErr.end(), _yErr2.begin(),
                           [](double err) { return std::pow(err, 2); });
        }

    SlitOffsetMinimization(SlitOffsetMinimization const &) = default;
    SlitOffsetMinimization(SlitOffsetMinimization &&) = default;
    SlitOffsetMinimization &operator=(SlitOffsetMinimization const &) = default;
    SlitOffsetMinimization &operator=(SlitOffsetMinimization &&) = default;
    virtual ~SlitOffsetMinimization() override = default;

    virtual void applyOffset(double spatial, double spectral) const {
        _detectorMap->applySlitOffset(spatial, spectral);
    }
    virtual void removeOffset(double spatial, double spectral) const {
        _detectorMap->applySlitOffset(-spatial, -spectral);
    }

    double calculateChi2(lsst::geom::Point2D const& offset) const;

    lsst::geom::Point2D minimize() const;

    double operator()(std::vector<double> const& parameters) const override;
    double Up() const override { return 1.0; }  // 1.0 --> fitting chi^2

  protected:
    std::shared_ptr<DetectorMap> _detectorMap;
    std::size_t _num;
    ndarray::Array<int, 1, 1> const _fiberId;
    ndarray::Array<double, 1, 1> const _wavelength;
    ndarray::Array<double, 1, 1> const _x;
    ndarray::Array<double, 1, 1> const _y;
    ndarray::Array<double, 1, 1> const _xErr2;
    ndarray::Array<double, 1, 1> const _yErr2;
    mutable lsst::utils::Cache<lsst::geom::Point2D, double> _cache;
};

double SlitOffsetMinimization::calculateChi2(lsst::geom::Point2D const& offset) const {
    double const spatial = offset.getX();
    double const spectral = offset.getY();
    applyOffset(spatial, spectral);

    double chi2 = 0.0;
    std::size_t num = 0;
    for (std::size_t ii = 0; ii < _num; ++ii) {
        lsst::geom::Point2D const point = _detectorMap->findPoint(_fiberId[ii], _wavelength[ii]);
        double const dx = _x[ii] - point.getX();
        double const dy = _y[ii] - point.getY();
        chi2 += std::pow(dx, 2)/_xErr2[ii];
        chi2 += std::pow(dy, 2)/_yErr2[ii];
        ++num;
    }
    removeOffset(spatial, spectral);  // no need to wrap in RAII since the detectorMap is a copy just for us

    if (std::isnan(chi2)) {
        chi2 = std::numeric_limits<double>::infinity();
    }

    return chi2;
}


double SlitOffsetMinimization::operator()(std::vector<double> const& parameters) const {
    assert(parameters.size() == 2);
    return _cache(lsst::geom::Point2D(parameters[0], parameters[1]),
                  [this](lsst::geom::Point2D const& offset) { return calculateChi2(offset); });
}


lsst::geom::Point2D SlitOffsetMinimization::minimize() const {
    std::vector<double> dx(_num);
    std::vector<double> dy(_num);
    for (std::size_t ii = 0, jj = 0; ii < _num; ++ii) {
        lsst::geom::Point2D const point = _detectorMap->findPoint(_fiberId[ii], _wavelength[ii]);
        dx[jj] = _x[ii] - point.getX();
        dy[jj] = _y[ii] - point.getY();
        ++jj;
    }
    double const dxMean = lsst::afw::math::makeStatistics(dx, lsst::afw::math::MEDIAN).getValue();
    double const dyMean = lsst::afw::math::makeStatistics(dy, lsst::afw::math::MEDIAN).getValue();

    std::vector<double> parameters = {dxMean, dyMean};
    std::vector<double> steps = {0.1, 0.1};

#ifdef DEBUG
    ROOT::Minuit2::MnPrint::SetLevel(3);
#endif
    auto const min = ROOT::Minuit2::MnMigrad(*this, parameters, steps)();
    assert(min.UserParameters().Params().size() == 2);
    return lsst::geom::Point2D(min.UserParameters().Params()[0], min.UserParameters().Params()[1]);
}

}  // anonymous namespace


void GlobalDetectorMap::measureSlitOffsets(
    ndarray::Array<int, 1, 1> const& fiberId,
    ndarray::Array<double, 1, 1> const& wavelength,
    ndarray::Array<double, 1, 1> const& x,
    ndarray::Array<double, 1, 1> const& y,
    ndarray::Array<double, 1, 1> const& xErr,
    ndarray::Array<double, 1, 1> const& yErr
) {
    std::size_t const num = fiberId.size();
    utils::checkSize(wavelength.size(), num, "wavelength");
    utils::checkSize(x.size(), num, "x");
    utils::checkSize(y.size(), num, "y");
    utils::checkSize(xErr.size(), num, "xErr");
    utils::checkSize(yErr.size(), num, "yErr");

    auto const offset = SlitOffsetMinimization(*this, fiberId, wavelength, x, y, xErr, yErr).minimize();
    applySlitOffset(offset.getX(), offset.getY());
}


namespace {

// Singleton class that manages the persistence catalog's schema and keys
class GlobalDetectorMapSchema {
  public:
    lsst::afw::table::Schema schema;
    lsst::afw::table::Box2IKey bbox;
    lsst::afw::table::Key<int> model;
    lsst::afw::table::Key<int> visitInfo;

    static GlobalDetectorMapSchema const &get() {
        static GlobalDetectorMapSchema const instance;
        return instance;
    }

  private:
    GlobalDetectorMapSchema()
      : schema(),
        bbox(lsst::afw::table::Box2IKey::addFields(schema, "bbox", "bounding box", "pixel")),
        model(schema.addField<int>("model", "model reference", "")),
        visitInfo(schema.addField<int>("visitInfo", "visitInfo reference", "")) {
            schema.getCitizen().markPersistent();
    }
};

}  // anonymous namespace


void GlobalDetectorMap::write(lsst::afw::table::io::OutputArchiveHandle & handle) const {
    GlobalDetectorMapSchema const &schema = GlobalDetectorMapSchema::get();
    lsst::afw::table::BaseCatalog cat = handle.makeCatalog(schema.schema);
    PTR(lsst::afw::table::BaseRecord) record = cat.addNew();
    record->set(schema.bbox, getBBox());
    record->set(schema.model, handle.put(getModel()));
    record->set(schema.visitInfo, handle.put(getVisitInfo()));
    // XXX dropping metadata on the floor, since we can't write a header
    handle.saveCatalog(cat);
}


class GlobalDetectorMap::Factory : public lsst::afw::table::io::PersistableFactory {
  public:
    std::shared_ptr<lsst::afw::table::io::Persistable> read(
        lsst::afw::table::io::InputArchive const& archive,
        lsst::afw::table::io::CatalogVector const& catalogs
    ) const override {
        static auto const& schema = GlobalDetectorMapSchema::get();
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        lsst::afw::table::BaseRecord const& record = catalogs.front().front();
        LSST_ARCHIVE_ASSERT(record.getSchema() == schema.schema);

        lsst::geom::Box2I const bbox = record.get(schema.bbox);
        auto model = archive.get<GlobalDetectorModel>(record.get(schema.model));
        auto visitInfo = archive.get<lsst::afw::image::VisitInfo>(record.get(schema.visitInfo));

        return std::make_shared<GlobalDetectorMap>(bbox, *model, *visitInfo);
    }

    Factory(std::string const& name) : lsst::afw::table::io::PersistableFactory(name) {}
};


namespace {

GlobalDetectorMap::Factory registration("GlobalDetectorMap");

}  // anonymous namespace


}}}  // namespace pfs::drp::stella
