#ifndef PFS_DRP_STELLA_DISTORTEDDETECTORMAP_H
#define PFS_DRP_STELLA_DISTORTEDDETECTORMAP_H

#include "pfs/drp/stella/DetectorDistortion.h"
#include "pfs/drp/stella/DistortionBasedDetectorMap.h"


namespace pfs {
namespace drp {
namespace stella {


/// DetectorMap referenced to a base detectorMap and a DetectorDistortion
class DistortedDetectorMap : public DistortionBasedDetectorMap<DetectorDistortion> {
  public:

    // Inherit Ctor
    using DistortionBasedDetectorMap<DetectorDistortion>::DistortionBasedDetectorMap;

    virtual ~DistortedDetectorMap() {}
    DistortedDetectorMap(DistortedDetectorMap const&) = default;
    DistortedDetectorMap(DistortedDetectorMap &&) = default;
    DistortedDetectorMap & operator=(DistortedDetectorMap const&) = default;
    DistortedDetectorMap & operator=(DistortedDetectorMap &&) = default;

    class Factory;

  protected:
    std::string getPersistenceName() const override { return "DistortedDetectorMap"; }
};


}}}

#endif // include guard
