#include "pfs/drp/stella/DistortedDetectorMap.h"
#include "pfs/drp/stella/impl/DistortionBasedDetectorMap.h"

namespace pfs {
namespace drp {
namespace stella {


class DistortedDetectorMap::Factory : public lsst::afw::table::io::PersistableFactory {
  public:
    std::shared_ptr<lsst::afw::table::io::Persistable> read(
        lsst::afw::table::io::InputArchive const& archive,
        lsst::afw::table::io::CatalogVector const& catalogs
    ) const override {
        return readDistortionBasedDetectorMap<DistortedDetectorMap>(archive, catalogs);
    }

    Factory(std::string const& name) : lsst::afw::table::io::PersistableFactory(name) {}
};


namespace {

DistortedDetectorMap::Factory registration("DistortedDetectorMap");

}  // anonymous namespace


}}}  // namespace pfs::drp::stella
