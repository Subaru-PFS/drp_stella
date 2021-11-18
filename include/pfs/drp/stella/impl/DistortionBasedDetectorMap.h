#ifndef PFS_DRP_STELLA_IMPL_DISTORTIONBASEDDETECTORMAP_H
#define PFS_DRP_STELLA_IMPL_DISTORTIONBASEDDETECTORMAP_H

#include "lsst/afw/table.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/Persistable.cc"


#include "pfs/drp/stella/DistortionBasedDetectorMap.h"


namespace pfs {
namespace drp {
namespace stella {


namespace {

double calculateWavelengthCenter(DetectorMap const& detMap) {
    lsst::geom::Box2I const box = detMap.getBBox();
    double const row = 0.5*(box.getMinY() + box.getMaxY());
    int const fiberId = detMap.getFiberId()[detMap.getNumFibers()/2];
    return detMap.findWavelength(fiberId, row);
}


double calculateDispersion(DetectorMap const& detMap) {
    lsst::geom::Box2I const box = detMap.getBBox();
    double const row = 0.5*(box.getMinY() + box.getMaxY());
    int const fiberId = detMap.getFiberId()[detMap.getNumFibers()/2];
    double const wavelength1 = detMap.findWavelength(fiberId, row);
    double const wavelength2 = detMap.findWavelength(fiberId, row + 1);
    return std::abs(wavelength2 - wavelength1);
}

} // anonymous namespace


template <typename DistortionT>
DistortionBasedDetectorMap<DistortionT>::DistortionBasedDetectorMap(
    SplinedDetectorMap const& base,
    DistortionT const& distortion,
    VisitInfo const& visitInfo,
    std::shared_ptr<lsst::daf::base::PropertySet> metadata,
    float samplingFactor
) : ModelBasedDetectorMap(base.getBBox(), calculateWavelengthCenter(base),
                          samplingFactor*calculateDispersion(base), base.getFiberId(),
                          base.getSpatialOffsets(), base.getSpectralOffsets(),
                          visitInfo, metadata),
    _base(base),
    _distortion(distortion)
{}


template <typename DistortionT>
std::shared_ptr<DetectorMap> DistortionBasedDetectorMap<DistortionT>::clone() const {
    return std::make_shared<DistortionBasedDetectorMap>(getBase(), getDistortion(), getVisitInfo(),
                                                        getMetadata()->deepCopy());
}


template <typename DistortionT>
lsst::geom::PointD DistortionBasedDetectorMap<DistortionT>::findPointImpl(
    int fiberId,
    double wavelength
) const {
    lsst::geom::PointD const base = _base.findPoint(fiberId, wavelength);
    lsst::geom::PointD const distortion = _distortion(base);
    return base + lsst::geom::Extent2D(distortion);
}


template <typename DistortionT>
void DistortionBasedDetectorMap<DistortionT>::_resetSlitOffsets() {
    for (auto fiberId : getFiberId()) {
        _base.setSlitOffsets(fiberId, getSpatialOffset(fiberId), getSpectralOffset(fiberId));
    }
    ModelBasedDetectorMap::_resetSlitOffsets();
}


namespace {

// Singleton class that manages the persistence catalog's schema and keys
class DistortionBasedDetectorMapSchema {
  public:
    lsst::afw::table::Schema schema;
    lsst::afw::table::Key<int> base;
    lsst::afw::table::Key<int> distortion;
    lsst::afw::table::Key<int> visitInfo;

    static DistortionBasedDetectorMapSchema const &get() {
        static DistortionBasedDetectorMapSchema const instance;
        return instance;
    }

  private:
    DistortionBasedDetectorMapSchema()
      : schema(),
        base(schema.addField<int>("base", "reference to base", "")),
        distortion(schema.addField<int>("distortion", "reference to distortion", "")),
        visitInfo(schema.addField<int>("visitInfo", "visitInfo reference", "")) {
            schema.getCitizen().markPersistent();
    }
};

}  // anonymous namespace


template <typename DistortionT>
void DistortionBasedDetectorMap<DistortionT>::write(
    lsst::afw::table::io::OutputArchiveHandle & handle
) const {
    DistortionBasedDetectorMapSchema const &schema = DistortionBasedDetectorMapSchema::get();
    lsst::afw::table::BaseCatalog cat = handle.makeCatalog(schema.schema);
    PTR(lsst::afw::table::BaseRecord) record = cat.addNew();
    record->set(schema.base, handle.put(getBase()));
    record->set(schema.distortion, handle.put(getDistortion()));
    record->set(schema.visitInfo, handle.put(getVisitInfo()));
    // XXX dropping metadata on the floor, since we can't write a header
    handle.saveCatalog(cat);
}


template <typename Derived>
std::shared_ptr<lsst::afw::table::io::Persistable>
readDistortionBasedDetectorMap(
    lsst::afw::table::io::InputArchive const& archive,
    lsst::afw::table::io::CatalogVector const& catalogs
) {
    static auto const& schema = DistortionBasedDetectorMapSchema::get();
    LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
    lsst::afw::table::BaseRecord const& record = catalogs.front().front();
    LSST_ARCHIVE_ASSERT(record.getSchema() == schema.schema);

    auto base = archive.get<SplinedDetectorMap>(record.get(schema.base));
    auto model = archive.get<typename Derived::Distortion>(record.get(schema.distortion));
    auto visitInfo = archive.get<lsst::afw::image::VisitInfo>(record.get(schema.visitInfo));
    return std::make_shared<Derived>(*base, *model, *visitInfo);
}


}}}  // namespace pfs::drp::stella

#endif  // include guard
