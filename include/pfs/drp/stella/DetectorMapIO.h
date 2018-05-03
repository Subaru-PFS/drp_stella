#if !defined(PFS_DRP_STELLA_DETECTORMAPIO_H)
#define PFS_DRP_STELLA_DETECTORMAPIO_H

#include "pfs/drp/stella/DetectorMap.h"

namespace pfs { namespace drp { namespace stella {
/**
 * @brief Provide access to the insides of a DetectorMap to permit us to persist/unpersist to disk
 */
class DetectorMapIO {
public:
    /// \brief Create a DetectorMapIO with an empty DetectorMap
    explicit DetectorMapIO(lsst::afw::geom::Box2I bbox,                    // detector's bounding box
                           ndarray::Array<int, 1, 1> const& fiberIds,      // 1-indexed IDs for each fibre
                           std::size_t nKnot                               // number of knots
                          ) :
        _detectorMap(bbox, fiberIds, nKnot) { }
        
    /// \brief Create a DetectorMapIO from a DetectorMap
    explicit DetectorMapIO(DetectorMap const& detectorMap)
        : _detectorMap(detectorMap) {}

    /// \brief return a fully initialised DetectorMap -- providing that you set the fields!
    DetectorMap getDetectorMap()
        {
            _detectorMap._set_xToFiberId();
            return _detectorMap;
        }

    /** \brief return the DetectorMap's bounding box */
    lsst::afw::geom::Box2I getBBox() const { return _detectorMap.getBBox(); }
    /** \brief return the DetectorMap's fiberIds */
    DetectorMap::FiberMap const & getFiberIds() const { return _detectorMap._fiberIds; }

    /**
     * Get the offsets of the wavelengths and x-centres (in floating-point pixels) and focus (in microns)
     * of each fibre
     */
    ndarray::Array<float, 2, 1> const& getSlitOffsets() const {
        return _detectorMap.getSlitOffsets();
    }
    
    /**
     * set the offsets of the wavelengths and x-centres (in floating-point pixels) and focus (in microns)
     * of each fibre
     */
    void setSlitOffsets(ndarray::Array<float, 2, 1> const& slitOffsets ///< new values of offsets
                       ) {
        _detectorMap.setSlitOffsets(slitOffsets);
    }

    /** Return the knots and values at the knots for the XCenter spline for the specified fibre */
    std::pair<std::vector<float>, std::vector<float>> getXCenter(const int fiberId ///< desired fiberId
                                                                ) {
        return std::make_pair(_detectorMap._yToXCenter[_detectorMap.getFiberIdx(fiberId)].getX(),
                              _detectorMap._yToXCenter[_detectorMap.getFiberIdx(fiberId)].getY());
    }

    /** Set the XCenter spline for specified fibre */
    void setXCenter(const int fiberId,  ///< desired fiberId
                    std::vector<float> const& xCenterKnots, ///< knots
                    std::vector<float> const& xCenterValues ///< values
                   ) {
        _detectorMap._yToXCenter[_detectorMap.getFiberIdx(fiberId)] =
            math::spline<float>(xCenterKnots, xCenterValues);
    }

    /** Return the knots and values at the knots for the Wavelength spline for the specified fibre */
    std::pair<std::vector<float>, std::vector<float>> getWavelength(const int fiberId ///< desired fiberId
                                                                ) {
        return std::make_pair(_detectorMap._yToWavelength[_detectorMap.getFiberIdx(fiberId)].getX(),
                              _detectorMap._yToWavelength[_detectorMap.getFiberIdx(fiberId)].getY());
    }

    /** Set the Wavelength spline for specified fibre */
    void setWavelength(const int fiberId,  ///< desired fiberId
                       std::vector<float> const& WavelengthKnots, ///< knots
                       std::vector<float> const& WavelengthValues ///< values
                      ) {
        _detectorMap._yToWavelength[_detectorMap.getFiberIdx(fiberId)] =
            math::spline<float>(WavelengthKnots, WavelengthValues);
    }

private:
    DetectorMap _detectorMap;
};

}}}

#endif
