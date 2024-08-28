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
    float sampling
) : ModelBasedDetectorMap(
        base->getBBox(),
        model.getWavelengthCenter(),
        model.getDispersion(),
        sampling,
        base->getFiberId(),
        base->getSpatialOffsets(),
        base->getSpectralOffsets(),
        visitInfo,
        metadata
    ),
    _base(base),
    _model(model)
{}


std::shared_ptr<DetectorMap> DifferentialDetectorMap::clone() const {
    return std::make_shared<DifferentialDetectorMap>(getBase(), getModel(), getVisitInfo(),
                                                     getMetadata()->deepCopy());
}


lsst::geom::PointD DifferentialDetectorMap::evalModel(
    int fiberId,
    double wavelength
) const {
    lsst::geom::PointD const base = _base->findPoint(fiberId, wavelength);
    lsst::geom::PointD const model = _model(fiberId, wavelength);
    return base + lsst::geom::Extent2D(model);
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
        visitInfo(schema.addField<int>("visitInfo", "visitInfo reference", ""))
        {}
};

}  // anonymous namespace


void DifferentialDetectorMap::write(lsst::afw::table::io::OutputArchiveHandle & handle) const {
    DifferentialDetectorMapSchema const &schema = DifferentialDetectorMapSchema::get();
    lsst::afw::table::BaseCatalog cat = handle.makeCatalog(schema.schema);
    std::shared_ptr<lsst::afw::table::BaseRecord> record = cat.addNew();
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
