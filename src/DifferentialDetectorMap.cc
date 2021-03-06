#include <unordered_set>
#include "ndarray.h"

#include "lsst/afw/table.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/Persistable.cc"

#include "pfs/drp/stella/DifferentialDetectorMap.h"
#include "pfs/drp/stella/utils/math.h"
#include "pfs/drp/stella/utils/checkSize.h"


namespace pfs {
namespace drp {
namespace stella {


DifferentialDetectorMap::DifferentialDetectorMap(
    std::shared_ptr<SplinedDetectorMap> base,
    GlobalDetectorModel const& model,
    VisitInfo const& visitInfo,
    std::shared_ptr<lsst::daf::base::PropertySet> metadata,
    float samplingFactor
) : ModelBasedDetectorMap(base->getBBox(), model.getWavelengthCenter(), samplingFactor*model.getDispersion(),
                          base->getFiberId(), base->getSpatialOffsets(), base->getSpectralOffsets(),
                          visitInfo, metadata),
    _base(base),
    _model(model)
{}


std::shared_ptr<DetectorMap> DifferentialDetectorMap::clone() const {
    return std::make_shared<DifferentialDetectorMap>(getBase(), getModel(), getVisitInfo(),
                                                     getMetadata()->deepCopy());
}


lsst::geom::PointD DifferentialDetectorMap::findPointImpl(
    int fiberId,
    double wavelength
) const {
    lsst::geom::PointD const base = _base->findPoint(fiberId, wavelength);
    lsst::geom::PointD const model = _model(fiberId, wavelength);
    return base + lsst::geom::Extent2D(model);
}


void DifferentialDetectorMap::measureSlitOffsets(
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

    // Remove the distortion field
    ndarray::Array<double, 2, 1> distortion = _model(fiberId, wavelength);
    ndarray::Array<double, 1, 1> xPrime = ndarray::allocate(num);
    ndarray::Array<double, 1, 1> yPrime = ndarray::allocate(num);
    for (std::size_t ii = 0; ii < num; ++ii) {
        xPrime[ii] = x[ii] - distortion[ii][0];
        yPrime[ii] = y[ii] - distortion[ii][1];
    }

    _base->measureSlitOffsets(fiberId, wavelength, xPrime, yPrime, xErr, yErr);

    // Get offsets measured by the base into the main list of offsets
    for (auto ff : getFiberId()) {
        setSlitOffsets(ff, _base->getSpatialOffset(ff), _base->getSpectralOffset(ff));
    }
}


void DifferentialDetectorMap::_resetSlitOffsets() {
    for (auto fiberId : getFiberId()) {
        _base->setSlitOffsets(fiberId, getSpatialOffset(fiberId), getSpectralOffset(fiberId));
    }
    ModelBasedDetectorMap::_resetSlitOffsets();
}


namespace {

// Singleton class that manages the persistence catalog's schema and keys
class DifferentialDetectorMapSchema {
  public:
    lsst::afw::table::Schema schema;
    lsst::afw::table::Key<int> base;
    lsst::afw::table::Key<int> model;
    lsst::afw::table::Key<int> visitInfo;

    static DifferentialDetectorMapSchema const &get() {
        static DifferentialDetectorMapSchema const instance;
        return instance;
    }

  private:
    DifferentialDetectorMapSchema()
      : schema(),
        base(schema.addField<int>("base", "reference to base", "")),
        model(schema.addField<int>("model", "reference to model", "")),
        visitInfo(schema.addField<int>("visitInfo", "visitInfo reference", "")) {
            schema.getCitizen().markPersistent();
    }
};

}  // anonymous namespace


void DifferentialDetectorMap::write(lsst::afw::table::io::OutputArchiveHandle & handle) const {
    DifferentialDetectorMapSchema const &schema = DifferentialDetectorMapSchema::get();
    lsst::afw::table::BaseCatalog cat = handle.makeCatalog(schema.schema);
    PTR(lsst::afw::table::BaseRecord) record = cat.addNew();
    record->set(schema.base, handle.put(_base));
    record->set(schema.model, handle.put(_model));
    record->set(schema.visitInfo, handle.put(getVisitInfo()));
    // XXX dropping metadata on the floor, since we can't write a header
    handle.saveCatalog(cat);
}


class DifferentialDetectorMap::Factory : public lsst::afw::table::io::PersistableFactory {
  public:
    std::shared_ptr<lsst::afw::table::io::Persistable> read(
        lsst::afw::table::io::InputArchive const& archive,
        lsst::afw::table::io::CatalogVector const& catalogs
    ) const override {
        static auto const& schema = DifferentialDetectorMapSchema::get();
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        lsst::afw::table::BaseRecord const& record = catalogs.front().front();
        LSST_ARCHIVE_ASSERT(record.getSchema() == schema.schema);

        auto base = archive.get<SplinedDetectorMap>(record.get(schema.base));
        auto model = archive.get<GlobalDetectorModel>(record.get(schema.model));
        auto visitInfo = archive.get<lsst::afw::image::VisitInfo>(record.get(schema.visitInfo));
        return std::make_shared<DifferentialDetectorMap>(base, *model, *visitInfo);
    }

    Factory(std::string const& name) : lsst::afw::table::io::PersistableFactory(name) {}
};


namespace {

DifferentialDetectorMap::Factory registration("DifferentialDetectorMap");

}  // anonymous namespace


}}}  // namespace pfs::drp::stella
