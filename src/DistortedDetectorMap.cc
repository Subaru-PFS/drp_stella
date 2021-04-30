#include <unordered_set>
#include "ndarray.h"

#include "lsst/afw/table.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/Persistable.cc"

#include "pfs/drp/stella/DistortedDetectorMap.h"
#include "pfs/drp/stella/utils/math.h"
#include "pfs/drp/stella/utils/checkSize.h"


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


DistortedDetectorMap::DistortedDetectorMap(
    SplinedDetectorMap const& base,
    DetectorDistortion const& distortion,
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


std::shared_ptr<DetectorMap> DistortedDetectorMap::clone() const {
    return std::make_shared<DistortedDetectorMap>(getBase(), getDistortion(), getVisitInfo(),
                                                  getMetadata()->deepCopy());
}


lsst::geom::PointD DistortedDetectorMap::findPointImpl(
    int fiberId,
    double wavelength
) const {
    lsst::geom::PointD const base = _base.findPoint(fiberId, wavelength);
    lsst::geom::PointD const distortion = _distortion(base);
    return base + lsst::geom::Extent2D(distortion);
}


void DistortedDetectorMap::measureSlitOffsets(
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
    ndarray::Array<double, 2, 1> base = _base.findPoint(fiberId, wavelength);
    ndarray::Array<double, 2, 1> distortion = _distortion(base);
    ndarray::Array<double, 1, 1> xPrime = ndarray::allocate(num);
    ndarray::Array<double, 1, 1> yPrime = ndarray::allocate(num);
    for (std::size_t ii = 0; ii < num; ++ii) {
        xPrime[ii] = x[ii] - distortion[ii][0];
        yPrime[ii] = y[ii] - distortion[ii][1];
    }

    _base.measureSlitOffsets(fiberId, wavelength, xPrime, yPrime, xErr, yErr);

    // Get offsets measured by the base into the main list of offsets
    for (auto ff : getFiberId()) {
        setSlitOffsets(ff, _base.getSpatialOffset(ff), _base.getSpectralOffset(ff));
    }
}


void DistortedDetectorMap::_resetSlitOffsets() {
    for (auto fiberId : getFiberId()) {
        _base.setSlitOffsets(fiberId, getSpatialOffset(fiberId), getSpectralOffset(fiberId));
    }
    ModelBasedDetectorMap::_resetSlitOffsets();
}


namespace {

// Singleton class that manages the persistence catalog's schema and keys
class DistortedDetectorMapSchema {
  public:
    lsst::afw::table::Schema schema;
    lsst::afw::table::Key<int> base;
    lsst::afw::table::Key<int> distortion;
    lsst::afw::table::Key<int> visitInfo;

    static DistortedDetectorMapSchema const &get() {
        static DistortedDetectorMapSchema const instance;
        return instance;
    }

  private:
    DistortedDetectorMapSchema()
      : schema(),
        base(schema.addField<int>("base", "reference to base", "")),
        distortion(schema.addField<int>("distortion", "reference to distortion", "")),
        visitInfo(schema.addField<int>("visitInfo", "visitInfo reference", "")) {
            schema.getCitizen().markPersistent();
    }
};

}  // anonymous namespace


void DistortedDetectorMap::write(lsst::afw::table::io::OutputArchiveHandle & handle) const {
    DistortedDetectorMapSchema const &schema = DistortedDetectorMapSchema::get();
    lsst::afw::table::BaseCatalog cat = handle.makeCatalog(schema.schema);
    PTR(lsst::afw::table::BaseRecord) record = cat.addNew();
    record->set(schema.base, handle.put(_base));
    record->set(schema.distortion, handle.put(_distortion));
    record->set(schema.visitInfo, handle.put(getVisitInfo()));
    // XXX dropping metadata on the floor, since we can't write a header
    handle.saveCatalog(cat);
}


class DistortedDetectorMap::Factory : public lsst::afw::table::io::PersistableFactory {
  public:
    std::shared_ptr<lsst::afw::table::io::Persistable> read(
        lsst::afw::table::io::InputArchive const& archive,
        lsst::afw::table::io::CatalogVector const& catalogs
    ) const override {
        static auto const& schema = DistortedDetectorMapSchema::get();
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        lsst::afw::table::BaseRecord const& record = catalogs.front().front();
        LSST_ARCHIVE_ASSERT(record.getSchema() == schema.schema);

        auto base = archive.get<SplinedDetectorMap>(record.get(schema.base));
        auto model = archive.get<DetectorDistortion>(record.get(schema.distortion));
        auto visitInfo = archive.get<lsst::afw::image::VisitInfo>(record.get(schema.visitInfo));
        return std::make_shared<DistortedDetectorMap>(*base, *model, *visitInfo);
    }

    Factory(std::string const& name) : lsst::afw::table::io::PersistableFactory(name) {}
};


namespace {

DistortedDetectorMap::Factory registration("DistortedDetectorMap");

}  // anonymous namespace


}}}  // namespace pfs::drp::stella
