#include "pfs/drp/stella/DoubleDetectorMap.h"
#include "pfs/drp/stella/impl/DistortionBasedDetectorMap.h"

namespace pfs {
namespace drp {
namespace stella {


class DoubleDetectorMap::Factory : public lsst::afw::table::io::PersistableFactory {
  public:
    std::shared_ptr<lsst::afw::table::io::Persistable> read(
        lsst::afw::table::io::InputArchive const& archive,
        lsst::afw::table::io::CatalogVector const& catalogs
    ) const override {
        return readDistortionBasedDetectorMap<DoubleDetectorMap>(archive, catalogs);
    }

    Factory(std::string const& name) : lsst::afw::table::io::PersistableFactory(name) {}
};



namespace {

DoubleDetectorMap::Factory registration("DoubleDetectorMap");

}  // anonymous namespace


// Explicit instantiation
template class DistortionBasedDetectorMap<pfs::drp::stella::DoubleDistortion>;


}}}  // namespace pfs::drp::stella
