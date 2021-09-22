#ifndef PFS_DRP_STELLA_DOUBLEDETECTORMAP_H
#define PFS_DRP_STELLA_DOUBLEDETECTORMAP_H

#include "pfs/drp/stella/DoubleDistortion.h"
#include "pfs/drp/stella/DistortionBasedDetectorMap.h"


namespace pfs {
namespace drp {
namespace stella {


/// DetectorMap referenced to a base detectorMap and a DoubleDistortion
class DoubleDetectorMap : public DistortionBasedDetectorMap<DoubleDistortion> {
  public:

    // Inherit Ctor
    using DistortionBasedDetectorMap<DoubleDistortion>::DistortionBasedDetectorMap;

    virtual ~DoubleDetectorMap() {}
    DoubleDetectorMap(DoubleDetectorMap const&) = default;
    DoubleDetectorMap(DoubleDetectorMap &&) = default;
    DoubleDetectorMap & operator=(DoubleDetectorMap const&) = default;
    DoubleDetectorMap & operator=(DoubleDetectorMap &&) = default;

    class Factory;

  protected:
    std::string getPersistenceName() const override { return "DoubleDetectorMap"; }
};


}}}

#endif // include guard
