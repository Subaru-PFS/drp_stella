#include "lsst/afw/table.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/Persistable.cc"

#include "pfs/drp/stella/math/AffineTransform.h"
#include "pfs/drp/stella/math/quartiles.h"
#include "pfs/drp/stella/LayeredDetectorMap.h"

//#define DEBUG_FIBERID 1


namespace pfs {
namespace drp {
namespace stella {


namespace {


double calculateFiberPitch(DetectorMap const& detMap) {
    double const row = detMap.getBBox().getCenterY();
    auto const& fiberId = detMap.getFiberId();
    std::size_t const num = fiberId.size();
    ndarray::Array<double, 1, 1> xCenter = ndarray::allocate(num - 1);
    double last = detMap.getXCenter(fiberId.front(), row);
    for (std::size_t ii = 0, jj = 1; jj < num; ++ii, ++jj) {
        double const next = detMap.getXCenter(fiberId[jj], row);
        xCenter[ii] = next - last;
        last = next;
    }
    return math::calculateMedian(xCenter);
}


double calculateWavelengthCenter(DetectorMap const& detMap) {
    double const row = detMap.getBBox().getCenterY();
    int const fiberId = detMap.getFiberId()[detMap.getNumFibers()/2];
    return detMap.findWavelength(fiberId, row);
}


double calculateDispersion(DetectorMap const& detMap) {
    double const row = detMap.getBBox().getCenterY();
    int const fiberId = detMap.getFiberId()[detMap.getNumFibers()/2];
    return detMap.findWavelength(fiberId, row + 1) - detMap.findWavelength(fiberId, row);
}


} // anonymous namespace


LayeredDetectorMap::LayeredDetectorMap(
    lsst::geom::Box2I const& bbox,
    ndarray::Array<double, 1, 1> const& spatialOffsets,
    ndarray::Array<double, 1, 1> const& spectralOffsets,
    SplinedDetectorMap const& base,
    DistortionList const& distortions,
    bool dividedDetector,
    lsst::geom::AffineTransform const& rightCcd,
    VisitInfo const& visitInfo,
    std::shared_ptr<lsst::daf::base::PropertySet> metadata,
    float sampling,
    float precision
) : ModelBasedDetectorMap(
        bbox,
        calculateWavelengthCenter(base),
        calculateDispersion(base),
        std::abs(sampling),
        base.getFiberId(),
        spatialOffsets,
        spectralOffsets,
        visitInfo,
        metadata,
        Spline::EXTRAPOLATE_ALL,
        precision
    ),
    _fiberPitch(calculateFiberPitch(base)),
    _wavelengthDispersion(calculateDispersion(base)),
    _base(base),
    _distortions(distortions),
    _dividedDetector(dividedDetector),
    _rightCcd(rightCcd),
    _precisionBBox(base.getBBox()),
    _fiberIndexCache(1000)
{
    for (auto const& distortion : _distortions) {
        if (!distortion) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                              "Null pointer in distortion list");
        }
    }

    // Ensure fiberIds are sorted
    for (std::size_t ii = 1; ii < getNumFibers(); ++ii) {
        if (getFiberId()[ii] <= getFiberId()[ii - 1]) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                              "FiberIds are not sorted");
        }
    }

    float const yCenter = _base.getBBox().getCenterY();

    {
        float const xFirst = _base.getXCenter(getFiberId().front(), yCenter);
        float const xLast = _base.getXCenter(getFiberId().back(), yCenter);
        _increasing = xFirst < xLast;
    }

    if (dividedDetector) {
        // Find the fibers that fall at the detector divide
        float const xCenter = 0.5*(_base.getXCenter(getFiberId().front(), yCenter) +
                                   _base.getXCenter(getFiberId().back(), yCenter));

        auto const lower = std::lower_bound(
            getFiberId().begin(), getFiberId().end(), xCenter,
            [yCenter, this](int fiberId, float xCenter) {
                float const xx = _base.getXCenter(fiberId, yCenter);
                return _increasing ? (xx < xCenter) : (xx > xCenter);
            }
        );
        if (lower == getFiberId().end()) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                              "Unable to find lower fiberId for detector center");
        }
        auto const upper = lower + (_increasing ? 1 : -1);
        if (_increasing) {
            _lowMiddleFiber = *lower;
            _highMiddleFiber = *upper;
            // lowMiddleFiber-1, lowMiddleFiber, highMiddleFiber, highMiddleFiber+1
            _lowMiddleIndex = lower - getFiberId().begin() - 1;  // selects lowMiddleFiber-1
            _highMiddleIndex = upper - getFiberId().begin();  // selects highMiddleFiber
        } else {
            _lowMiddleFiber = *upper;
            _highMiddleFiber = *lower;
            // highMiddleFiber+1, highMiddleFiber, lowMiddleFiber, lowMiddleFiber-1
            _lowMiddleIndex = upper - getFiberId().begin() - 1;  // selects lowMiddleFiber-1
            _highMiddleIndex = lower - getFiberId().begin();  // selects highMiddleFiber
        }
        _middleFiber = 0.5*(_lowMiddleFiber + _highMiddleFiber);
    } else {
        // These values will be unused, but set them to something not random
        _lowMiddleFiber = _highMiddleFiber = 0;
        _lowMiddleIndex = _highMiddleIndex = 0;
        _middleFiber = 0;
    }

    _precisionBBox.grow(getPrecision());

#if 0
    std::cerr << "LayeredDetectorMap: fiberPitch = " << _fiberPitch << std::endl;
    std::cerr << "LayeredDetectorMap: wavelengthDispersion = " << _wavelengthDispersion << std::endl;
    std::cerr << "LayeredDetectorMap: dividedDetector = " << _dividedDetector << std::endl;
    std::cerr << "LayeredDetectorMap: rightCcd = " << _rightCcd << std::endl;
    std::cerr << "LayeredDetectorMap: bbox = " << getBBox() << std::endl;
    std::cerr << "LayeredDetectorMap: base box = " << _base.getBBox() << std::endl;
    std::cerr << "LayeredDetectorMap: middle: " << _lowMiddleFiber << " " << _middleFiber << " " <<
        _highMiddleFiber << " " << getFiberId()[_lowMiddleIndex] << " " << getFiberId()[_highMiddleIndex] <<
        std::endl;
#endif
}


LayeredDetectorMap::LayeredDetectorMap(
    lsst::geom::Box2I const& bbox,
    ndarray::Array<double, 1, 1> const& spatialOffsets,
    ndarray::Array<double, 1, 1> const& spectralOffsets,
    SplinedDetectorMap const& base,
    DistortionList const& distortions,
    bool dividedDetector,
    ndarray::Array<double, 1, 1> const& rightCcd,
    VisitInfo const& visitInfo,
    std::shared_ptr<lsst::daf::base::PropertySet> metadata,
    float sampling,
    float precision
) : LayeredDetectorMap(
        bbox,
        spatialOffsets,
        spectralOffsets,
        base,
        distortions,
        dividedDetector,
        math::makeAffineTransform(rightCcd),
        visitInfo,
        metadata,
        sampling,
        precision
    )
{}


std::shared_ptr<DetectorMap> LayeredDetectorMap::clone() const {
    DistortionList distortions;
    distortions.reserve(getDistortions().size());
    for (auto const& dd : getDistortions()) {
        distortions.emplace_back(dd->clone());
    }
    return std::make_shared<LayeredDetectorMap>(
        getBBox(),
        ndarray::copy(getSpatialOffsets()),
        ndarray::copy(getSpectralOffsets()),
        *std::dynamic_pointer_cast<SplinedDetectorMap>(getBase().clone()),
        distortions,
        getDividedDetector(),
        ndarray::copy(getRightCcdParameters()),
        lsst::afw::image::VisitInfo(getVisitInfo()),  // copy
        getMetadata()->deepCopy()
    );
}


ndarray::Array<double, 1, 1> LayeredDetectorMap::getRightCcdParameters() const {
    return math::getAffineParameters(_rightCcd);
}


bool LayeredDetectorMap::isRightCcd(float x) const {
    if (!_dividedDetector) {
        return false;
    }
    return (x > getBBox().getCenterX());
}


std::pair<std::pair<int, double>, std::pair<int, double>>
LayeredDetectorMap::_interpolateFiberIndices(double fiberId) const {
    std::size_t prevIndex;
    ndarray::Array<int, 1, 1> const& fiberIdArray = getFiberId();

    if (_dividedDetector && fiberId >= _lowMiddleFiber && fiberId <= _highMiddleFiber) {
        // We're in the middle of the detector, so we need to interpolate between the two fibers on
        // the side closest to the target fiberId.
        if (fiberId < _middleFiber) {
            prevIndex = _lowMiddleIndex;
        } else {
            prevIndex = _highMiddleIndex;
        }
    } else {
        // We're not in the middle of the detector, so simply use the two closest fibers
        float const multiplier = 4;
        std::ptrdiff_t const integerizedFiberId = fiberId*multiplier;  // truncates
        prevIndex = _fiberIndexCache(
            integerizedFiberId,
            [fiberId, fiberIdArray, multiplier, this](std::ptrdiff_t intFiberId) {
                int const value = intFiberId/multiplier;  // truncates
                auto const iter = std::lower_bound(fiberIdArray.begin(), fiberIdArray.end(), value);
                std::size_t const index = iter - fiberIdArray.begin();
                return std::max(std::size_t(0), std::min(index, fiberIdArray.size() - 2));
            }
        );
    }

    std::size_t const nextIndex = prevIndex + 1;
    int const prevFiberId = fiberIdArray[prevIndex];
    int const nextFiberId = fiberIdArray[nextIndex];
    double const nextWeight = (fiberId - prevFiberId)/(nextFiberId - prevFiberId);
    double const prevWeight = 1.0 - nextWeight;

    return std::make_pair(std::make_pair(prevFiberId, prevWeight), std::make_pair(nextFiberId, nextWeight));
}


lsst::geom::PointD LayeredDetectorMap::evalModel(
    int fiberId,
    double wavelength
) const {
#ifdef DEBUG_FIBERID
    if (fiberId == DEBUG_FIBERID) {
        std::cerr << "fiberId,wavelength = " << fiberId << "," << wavelength;
        std::cerr << " + (" << getSpatialOffset(fiberId) << "," << getSpectralOffset(fiberId) << ")";
    }
#endif

    // Layer 1: apply the slit offsets
    double const fiber = fiberId + getSpatialOffset(fiberId)/_fiberPitch;
    double const wl = wavelength + getSpectralOffset(fiberId)*_wavelengthDispersion;

#ifdef DEBUG_FIBERID
    if (fiberId == DEBUG_FIBERID) {
        std::cerr << " --> fiber,wl = " << fiber << "," << wl;
    }
#endif

    // Layer 2: apply the base detectorMap
    auto const interp = _interpolateFiberIndices(fiber);

#ifdef DEBUG_FIBERID
    if (fiberId == DEBUG_FIBERID) {
        std::cerr << " --> interp = (" << interp.first.first << "," << interp.first.second << ") (" <<
            interp.second.first << "," << interp.second.second << ")";
    }
#endif

    lsst::geom::Extent2D const prev{_base.findPoint(interp.first.first, wl, false, false)};
    lsst::geom::Extent2D const next{_base.findPoint(interp.second.first, wl, false, false)};
    lsst::geom::Point2D const base{
        prev*interp.first.second + lsst::geom::Extent2D(next)*interp.second.second
    };
    lsst::geom::Point2D point{base};

#ifdef DEBUG_FIBERID
    if (fiberId == DEBUG_FIBERID) {
        std::cerr << " prev=" << prev << " next=" << next << " --> point=" << point;
    }
#endif

    // Layer 3: apply the distortions
    for (auto const& dd : _distortions) {
        lsst::geom::Extent2D const dist{(*dd)(base)};
        point += dist;
#ifdef DEBUG_FIBERID
        if (fiberId == DEBUG_FIBERID) {
            std::cerr << " + distortion " << dist << " --> point = " << point;
        }
#endif
    }

#ifdef DEBUG_FIBERID
    if (fiberId == DEBUG_FIBERID) {
        std::cerr << " --> distorted = " << point << std::endl;
    }
#endif

    // Layer 4: transform to the detector frame
    // This layer is handled in findPointImpl
    // The intent is that ModelBasedDetectorMap (which uses evalModel) handles
    // everything above the detectors layer, where everything is continuous.
    // Adding the detector layer introduces the potential for discontinuity
    // at the chip gap.
    return point;
}


lsst::geom::PointD LayeredDetectorMap::findPointPermissive(int fiberId, double wavelength) const {
    lsst::geom::Point2D point = ModelBasedDetectorMap::findPointImpl(fiberId, wavelength);

    // Layer 4: transform to the detector frame
    if (_dividedDetector) {
        float const xCenterBase = _base.getBBox().getCenterX();
        float const xCenterThis = getBBox().getCenterX();
        if (point.getX() > xCenterBase) {
            double const xOffset = 0.5*(getBBox().getMinX() + getBBox().getMaxX());
            double const xScale = 2.0/(getBBox().getMaxX() - getBBox().getMinX());
            double const yOffset = 0.5*(getBBox().getMinY() + getBBox().getMaxY());
            double const yScale = 2.0/(getBBox().getMaxY() - getBBox().getMinY());
            lsst::geom::Point2D normalized{
                (point.getX() - xOffset)*xScale,
                (point.getY() - yOffset)*yScale
            };
            point += lsst::geom::Extent2D(_rightCcd(normalized));
            if (point.getX() < xCenterThis) {
                // Off the right detector in the chip gap
                point.setX(std::numeric_limits<double>::quiet_NaN());
            }
        } else if (point.getX() > xCenterThis) {
            // Off the left detector in the chip gap
            point.setX(std::numeric_limits<double>::quiet_NaN());
        }
    }

#ifdef DEBUG_FIBERID
    if (fiberId == DEBUG_FIBERID) {
        std::cerr << "findPoint: fiberId,wavelength = " << fiberId << "," << wavelength;
        std::cerr << " --> point = " << point << std::endl;
    }
#endif
    return point;
}


lsst::geom::Point2D LayeredDetectorMap::findPointImpl(int fiberId, double wavelength) const {
    lsst::geom::Point2D const point = findPointPermissive(fiberId, wavelength);

    if (!lsst::geom::Box2D(_base.getBBox()).contains(point)) {
        double const nan = std::numeric_limits<double>::quiet_NaN();
        return lsst::geom::Point2D(nan, nan);
    }

    return point;
}


double LayeredDetectorMap::getXCenterImpl(int fiberId, double row) const {
    // We need to find the col for which
    //     findPoint(fiberId, wavelength).getX() == col
    //     findPoint(fiberId, wavelength).getY() == row
    // by iterating over the wavelength until we have the right row.
    // This is the same thing that we do in findWavelengthImpl.

#ifdef DEBUG_FIBERID
    if (fiberId == DEBUG_FIBERID) {
        std::cerr << "getXCenter: fiberId,row = " << fiberId << "," << row << std::endl << "  ";
    }
#endif

    double const wavelength = findWavelengthImpl(fiberId, row);
    if (!std::isfinite(wavelength)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    lsst::geom::Point2D const point = findPointPermissive(fiberId, wavelength);

#ifdef DEBUG_FIBERID
    if (fiberId == DEBUG_FIBERID) {
        std::cerr << " --> wavelength = " << wavelength << " --> point = " << point;
    }
#endif

    // Allow the point to be slightly off the detector
    double const xx = point.getX();
    double const yy = point.getY();
    if (!std::isfinite(yy) || yy < _precisionBBox.getMinY() || yy > _precisionBBox.getMaxY() ||
        xx < getBBox().getMinX() || xx > getBBox().getMaxX()) {
#ifdef DEBUG_FIBERID
        if (fiberId == DEBUG_FIBERID) {
            std::cerr << " vs " << _precisionBBox << " --> xCenter = NAN" << std::endl;
        }
#endif
        return std::numeric_limits<double>::quiet_NaN();
    }

#ifdef DEBUG_FIBERID
    if (fiberId == DEBUG_FIBERID) {
        std::cerr << " vs " << _precisionBBox << " --> xCenter = " << point.getX() << std::endl;
    }
#endif

    return xx;
}


double LayeredDetectorMap::findWavelengthImpl(int fiberId, double row) const {
    // We need to find the wavelength for which
    //     findPoint(fiberId, wavelength).getY() == row
    // by iterating over the wavelength until we have the right row.

    double const originalRow = row;

    double wavelength;
    double dy = 0.0;
    int iter = 0;
    do {
        if (++iter > 100) {
            return std::numeric_limits<double>::quiet_NaN();
            throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "findWavelength did not converge");
        }
        // This is the wavelength for layer 3 (distorted, above the detector)
        wavelength = ModelBasedDetectorMap::findWavelengthImpl(fiberId, originalRow - dy);
        // This is the position for layer 4 (on the detector)
        lsst::geom::Point2D const point = findPointPermissive(fiberId, wavelength);
        double const yy = point.getY();
#ifdef DEBUG_FIBERID
        if (fiberId == DEBUG_FIBERID) {
            std::cerr << "findWavelength: iter=" << iter << " fiberId,row,wavelength,yy = " << fiberId << "," << row << "," << wavelength << "," << yy << std::endl;
        }
#endif
        if (!std::isfinite(yy)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        // This is the difference between layer 3 and layer 4
        dy = point.getY() - originalRow;
    } while (std::abs(dy) > getPrecision());

    return wavelength;
}


namespace {

// Singleton class that manages the persistence catalog's schema and keys
class LayeredDetectorMapSchema {
    using IntArray = lsst::afw::table::Array<int>;
    using DoubleArray = lsst::afw::table::Array<double>;
  public:
    lsst::afw::table::Schema schema;
    lsst::afw::table::Box2IKey bbox;
    lsst::afw::table::Key<DoubleArray> spatialOffsets;
    lsst::afw::table::Key<DoubleArray> spectralOffsets;
    lsst::afw::table::Key<int> base;
    lsst::afw::table::Key<IntArray> distortions;
    lsst::afw::table::Key<int> dividedDetector;
    lsst::afw::table::Key<DoubleArray> rightCcd;
    lsst::afw::table::Key<int> visitInfo;

    static LayeredDetectorMapSchema const &get() {
        static LayeredDetectorMapSchema const instance;
        return instance;
    }

  private:
    LayeredDetectorMapSchema()
      : schema(),
        bbox(bbox.addFields(schema, "bbox", "bounding box", "")),
        spatialOffsets(schema.addField<DoubleArray>("spatialOffsets", "spatial offsets", "")),
        spectralOffsets(schema.addField<DoubleArray>("spectralOffsets", "spectral offsets", "")),
        base(schema.addField<int>("base", "reference to base", "")),
        distortions(schema.addField<IntArray>("distortions", "reference to distortions", "")),
        dividedDetector(schema.addField<int>("dividedDetector", "divided detector", "")),
        rightCcd(schema.addField<DoubleArray>("rightCcd", "upper CCD transform", "")),
        visitInfo(schema.addField<int>("visitInfo", "visitInfo reference", ""))
        {}
};

}  // anonymous namespace


void LayeredDetectorMap::write(
    lsst::afw::table::io::OutputArchiveHandle & handle
) const {
    LayeredDetectorMapSchema const &schema = LayeredDetectorMapSchema::get();
    lsst::afw::table::BaseCatalog cat = handle.makeCatalog(schema.schema);
    std::shared_ptr<lsst::afw::table::BaseRecord> record = cat.addNew();

    ndarray::Array<int, 1, 1> distortions = ndarray::allocate(_distortions.size());
    for (std::size_t ii = 0; ii < _distortions.size(); ++ii) {
        distortions[ii] = handle.put(*_distortions[ii]);
    }

    schema.bbox.set(*record, getBBox());
    record->set(schema.spatialOffsets, getSpatialOffsets());
    record->set(schema.spectralOffsets, getSpectralOffsets());
    record->set(schema.base, handle.put(getBase()));
    record->set(schema.distortions, distortions);
    record->set(schema.dividedDetector, _dividedDetector);
    ndarray::Array<double, 1, 1> rightCcd = ndarray::allocate(6);
    ndarray::asEigenMatrix(rightCcd) = _rightCcd.getParameterVector();
    record->set(schema.rightCcd, rightCcd);
    record->set(schema.visitInfo, handle.put(getVisitInfo()));
    // XXX dropping metadata on the floor, since we can't write a header
    handle.saveCatalog(cat);
}


class LayeredDetectorMap::Factory : public lsst::afw::table::io::PersistableFactory {
  public:
    std::shared_ptr<lsst::afw::table::io::Persistable> read(
        lsst::afw::table::io::InputArchive const& archive,
        lsst::afw::table::io::CatalogVector const& catalogs
    ) const override {
        static auto const& schema = LayeredDetectorMapSchema::get();
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        lsst::afw::table::BaseRecord const& record = catalogs.front().front();
        LSST_ARCHIVE_ASSERT(record.getSchema() == schema.schema);

        lsst::geom::Box2I bbox = schema.bbox.get(record);
        ndarray::Array<double, 1, 1> spatialOffsets = ndarray::copy(record.get(schema.spatialOffsets));
        ndarray::Array<double, 1, 1> spectralOffsets = ndarray::copy(record.get(schema.spectralOffsets));

        auto base = archive.get<SplinedDetectorMap>(record.get(schema.base));

        ndarray::Array<int const, 1, 1> const distortionPtrs = record.get(schema.distortions);
        std::size_t const numDistortions = distortionPtrs.size();
        LayeredDetectorMap::DistortionList distortions;
        distortions.reserve(numDistortions);
        for (std::size_t ii = 0; ii < numDistortions; ++ii) {
            distortions.emplace_back(archive.get<Distortion>(distortionPtrs[ii]));
        }

        bool dividedDetector = record.get(schema.dividedDetector);
        ndarray::Array<double, 1, 1> rightCcd = ndarray::copy(record.get(schema.rightCcd));

        auto visitInfo = archive.get<lsst::afw::image::VisitInfo>(record.get(schema.visitInfo));
        return std::make_shared<LayeredDetectorMap>(
            bbox, spatialOffsets, spectralOffsets, *base, distortions, dividedDetector, rightCcd, *visitInfo
        );
    }

    Factory(std::string const& name) : lsst::afw::table::io::PersistableFactory(name) {}
};


namespace {

LayeredDetectorMap::Factory registration("LayeredDetectorMap");

}  // anonymous namespace


}}}  // namespace pfs::drp::stella
