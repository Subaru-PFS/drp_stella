#include "lsst/afw/table.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/Persistable.cc"


#include "pfs/drp/stella/MultipleDistortionsDetectorMap.h"


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


MultipleDistortionsDetectorMap::MultipleDistortionsDetectorMap(
    SplinedDetectorMap const& base,
    DistortionList const& distortions,
    VisitInfo const& visitInfo,
    std::shared_ptr<lsst::daf::base::PropertySet> metadata,
    float samplingFactor
) : ModelBasedDetectorMap(base.getBBox(), calculateWavelengthCenter(base),
                          samplingFactor*calculateDispersion(base), base.getFiberId(),
                          base.getSpatialOffsets(), base.getSpectralOffsets(),
                          visitInfo, metadata),
    _base(base),
    _distortions(distortions)
{
    for (auto const& distortion : _distortions) {
        if (!distortion) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                              "Null pointer in distortion list");
        }
    }
}


std::shared_ptr<DetectorMap> MultipleDistortionsDetectorMap::clone() const {
    return std::make_shared<MultipleDistortionsDetectorMap>(getBase(), getDistortions(), getVisitInfo(),
                                                            getMetadata()->deepCopy());
}


lsst::geom::PointD MultipleDistortionsDetectorMap::findPointImpl(
    int fiberId,
    double wavelength
) const {
    lsst::geom::PointD point = _base.findPoint(fiberId, wavelength);
    for (auto const& distortion : _distortions) {
        point += lsst::geom::Extent2D((*distortion)(point));
    }
    return point;
}


void MultipleDistortionsDetectorMap::_resetSlitOffsets() {
    for (auto fiberId : getFiberId()) {
        _base.setSlitOffsets(fiberId, getSpatialOffset(fiberId), getSpectralOffset(fiberId));
    }
    ModelBasedDetectorMap::_resetSlitOffsets();
}


namespace {

// Singleton class that manages the persistence catalog's schema and keys
class MultipleDistortionsDetectorMapSchema {
  public:
    lsst::afw::table::Schema schema;
    lsst::afw::table::Key<int> base;
    lsst::afw::table::Key<lsst::afw::table::Array<int>> distortions;
    lsst::afw::table::Key<int> visitInfo;

    static MultipleDistortionsDetectorMapSchema const &get() {
        static MultipleDistortionsDetectorMapSchema const instance;
        return instance;
    }

  private:
    MultipleDistortionsDetectorMapSchema()
      : schema(),
        base(schema.addField<int>("base", "reference to base", "")),
        distortions(schema.addField<lsst::afw::table::Array<int>>(
            "distortions", "reference to distortions", ""
        )),
        visitInfo(schema.addField<int>("visitInfo", "visitInfo reference", ""))
        {}
};

}  // anonymous namespace


void MultipleDistortionsDetectorMap::write(
    lsst::afw::table::io::OutputArchiveHandle & handle
) const {
    MultipleDistortionsDetectorMapSchema const &schema = MultipleDistortionsDetectorMapSchema::get();
    lsst::afw::table::BaseCatalog cat = handle.makeCatalog(schema.schema);
    std::shared_ptr<lsst::afw::table::BaseRecord> record = cat.addNew();
    ndarray::Array<int, 1, 1> distortions = ndarray::allocate(_distortions.size());
    for (std::size_t ii = 0; ii < _distortions.size(); ++ii) {
        distortions[ii] = handle.put(*_distortions[ii]);
    }
    record->set(schema.base, handle.put(getBase()));
    record->set(schema.distortions, distortions);
    record->set(schema.visitInfo, handle.put(getVisitInfo()));
    // XXX dropping metadata on the floor, since we can't write a header
    handle.saveCatalog(cat);
}


class MultipleDistortionsDetectorMap::Factory : public lsst::afw::table::io::PersistableFactory {
  public:
    std::shared_ptr<lsst::afw::table::io::Persistable> read(
        lsst::afw::table::io::InputArchive const& archive,
        lsst::afw::table::io::CatalogVector const& catalogs
    ) const override {
        static auto const& schema = MultipleDistortionsDetectorMapSchema::get();
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        lsst::afw::table::BaseRecord const& record = catalogs.front().front();
        LSST_ARCHIVE_ASSERT(record.getSchema() == schema.schema);

        auto base = archive.get<SplinedDetectorMap>(record.get(schema.base));

        ndarray::Array<int const, 1, 1> const distortionPtrs = record.get(schema.distortions);
        std::size_t const numDistortions = distortionPtrs.size();
        MultipleDistortionsDetectorMap::DistortionList distortions;
        distortions.reserve(numDistortions);
        for (std::size_t ii = 0; ii < numDistortions; ++ii) {
            distortions.emplace_back(archive.get<Distortion>(distortionPtrs[ii]));
        }

        auto visitInfo = archive.get<lsst::afw::image::VisitInfo>(record.get(schema.visitInfo));
        return std::make_shared<MultipleDistortionsDetectorMap>(*base, distortions, *visitInfo);
    }

    Factory(std::string const& name) : lsst::afw::table::io::PersistableFactory(name) {}
};


namespace {

MultipleDistortionsDetectorMap::Factory registration("MultipleDistortionsDetectorMap");

}  // anonymous namespace


}}}  // namespace pfs::drp::stella
