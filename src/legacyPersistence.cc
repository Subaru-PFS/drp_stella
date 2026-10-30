// Readers for persisted objects whose classes have been removed
//
// Data written by earlier versions of this package contains these objects, and
// we need to be able to read that data. Each factory reads the persisted form
// and returns the closest object that still exists.

#include "lsst/afw/table.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/Persistable.h"
#include "lsst/afw/detection/Psf.h"

namespace pfs {
namespace drp {
namespace stella {

namespace {

// ImagingSpectralPsf wrapped an imaging PSF together with a DetectorMap, and
// was attached to every calexp written by the pipeline. We return only the
// imaging PSF. The DetectorMap is not read, so that a change in the
// persistence of DetectorMaps cannot prevent reading the PSF.
class LegacyImagingSpectralPsfFactory : public lsst::afw::table::io::PersistableFactory {
  public:
    std::shared_ptr<lsst::afw::table::io::Persistable> read(
        InputArchive const& archive,
        CatalogVector const& catalogs
    ) const override {
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        lsst::afw::table::BaseRecord const& record = catalogs.front().front();
        auto const base = record.getSchema().find<int>("base").key;
        return archive.get<lsst::afw::detection::Psf>(record.get(base));
    }

    LegacyImagingSpectralPsfFactory(std::string const& name)
      : lsst::afw::table::io::PersistableFactory(name) {}
};

LegacyImagingSpectralPsfFactory imagingSpectralPsfRegistration("ImagingSpectralPsf");

}  // anonymous namespace

}}}  // namespace pfs::drp::stella
