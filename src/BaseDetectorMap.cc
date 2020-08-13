#include <algorithm>

#include "lsst/pex/exceptions/Exception.h"

#include "pfs/drp/stella/utils/checkSize.h"

#include "pfs/drp/stella/BaseDetectorMap.h"

namespace pfs { namespace drp { namespace stella {


BaseDetectorMap::BaseDetectorMap(
    lsst::geom::Box2I bbox,
    BaseDetectorMap::FiberIds const& fiberId,
    BaseDetectorMap::Array1D const& spatialOffsets,
    BaseDetectorMap::Array1D const& spectralOffsets,
    BaseDetectorMap::VisitInfo const& visitInfo,
    std::shared_ptr<lsst::daf::base::PropertySet> metadata
) : _bbox(bbox),
    _fiberId(ndarray::allocate(fiberId.size())),
    _spatialOffsets(ndarray::allocate(fiberId.size())),
    _spectralOffsets(ndarray::allocate(fiberId.size())),
    _visitInfo(visitInfo),
    _metadata(metadata ? metadata : std::make_shared<lsst::daf::base::PropertyList>())
{
    // Ensure the fiberIds are sorted, since some algorithms depend on that.
    ndarray::Array<std::size_t, 1, 1> indices = ndarray::allocate(getNumFibers());
    for (std::size_t ii = 0; ii < getNumFibers(); ++ii) {
        indices[ii] = ii;
    }
    std::sort(indices.begin(), indices.end(),
              [&fiberId](std::size_t left, std::size_t right) { return fiberId[left] < fiberId[right]; });
    for (std::size_t ii = 0; ii < fiberId.size(); ++ii) {
        _fiberId[ii] = fiberId[indices[ii]];
        _fiberMap[_fiberId[ii]] = ii;
    }
    if (spatialOffsets.isEmpty()) {
        _spatialOffsets.deep() = 0.0;
    } else {
        utils::checkSize(spatialOffsets.size(), fiberId.size(), "spatialOffsets");
        for (std::size_t ii = 0; ii < fiberId.size(); ++ii) {
            _spatialOffsets[ii] = spatialOffsets[indices[ii]];
        }
    }
    if (spectralOffsets.isEmpty()) {
        _spectralOffsets.deep() = 0.0;
    } else {
        utils::checkSize(spectralOffsets.size(), fiberId.size(), "spectralOffsets");
        for (std::size_t ii = 0; ii < fiberId.size(); ++ii) {
            _spectralOffsets[ii] = spectralOffsets[indices[ii]];
        }
    }
}


void BaseDetectorMap::applySlitOffset(
    float spatial,
    float spectral
) {
    if (spatial != 0.0) {
        std::for_each(_spatialOffsets.begin(), _spatialOffsets.end(),
                      [spatial](float& value) { value += spatial; });
    }
    if (spectral != 0.0) {
        std::for_each(_spectralOffsets.begin(), _spectralOffsets.end(),
                      [spectral](float& value) { value += spectral; });
    }
    _resetSlitOffsets();
}


void BaseDetectorMap::setSlitOffsets(
    BaseDetectorMap::Array1D const& spatial,
    BaseDetectorMap::Array1D const& spectral
) {
    _spatialOffsets.deep() = spatial;
    _spectralOffsets.deep() = spectral;
    _resetSlitOffsets();
}


void BaseDetectorMap::setSlitOffsets(
    int fiberId,
    float spatial,
    float spectral
) {
    std::size_t const index = getFiberIndex(fiberId);
    _spatialOffsets[index] = spatial;
    _spectralOffsets[index] = spectral;
    _resetSlitOffsets();
}


BaseDetectorMap::Array2D BaseDetectorMap::getWavelength() const {
    Array2D wavelength = ndarray::allocate(getNumFibers(), _bbox.getHeight());
    for (std::size_t ii = 0; ii < getNumFibers(); ++ii) {
        wavelength[ii].deep() = getWavelength(_fiberId[ii]);
    }
    return wavelength;
}


BaseDetectorMap::Array2D BaseDetectorMap::getXCenter() const {
    Array2D xCenter = ndarray::allocate(getNumFibers(), _bbox.getHeight());
    for (std::size_t ii = 0; ii < getNumFibers(); ++ii) {
        xCenter[ii].deep() = getXCenter(_fiberId[ii]);
    }
    return xCenter;
}


}}}  // namespace pfs::drp::stella
