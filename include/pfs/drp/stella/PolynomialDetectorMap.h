#ifndef PFS_DRP_STELLA_POLYNOMIALDETECTORMAP_H
#define PFS_DRP_STELLA_POLYNOMIALDETECTORMAP_H

#include "pfs/drp/stella/PolynomialDistortion.h"
#include "pfs/drp/stella/DistortionBasedDetectorMap.h"


namespace pfs {
namespace drp {
namespace stella {


/// DetectorMap referenced to a base detectorMap and a PolynomialDistortion
class PolynomialDetectorMap : public DistortionBasedDetectorMap<PolynomialDistortion> {
  public:

    // Inherit Ctor
    using DistortionBasedDetectorMap<PolynomialDistortion>::DistortionBasedDetectorMap;

    virtual ~PolynomialDetectorMap() {}
    PolynomialDetectorMap(PolynomialDetectorMap const&) = default;
    PolynomialDetectorMap(PolynomialDetectorMap &&) = default;
    PolynomialDetectorMap & operator=(PolynomialDetectorMap const&) = default;
    PolynomialDetectorMap & operator=(PolynomialDetectorMap &&) = default;

    class Factory;

  protected:
    std::string getPersistenceName() const override { return "PolynomialDetectorMap"; }
};


}}}

#endif // include guard
