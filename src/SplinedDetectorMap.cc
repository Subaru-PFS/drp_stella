#include <unordered_set>

#include "lsst/afw/table.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/Persistable.cc"

#include "pfs/drp/stella/utils/checkSize.h"

#include "pfs/drp/stella/SplinedDetectorMap.h"

namespace pfs { namespace drp { namespace stella {


SplinedDetectorMap::SplinedDetectorMap(
    lsst::geom::Box2I bbox,
    FiberIds const& fiberId,
    std::vector<ndarray::Array<double, 1, 1>> const& xCenterKnots,
    std::vector<ndarray::Array<double, 1, 1>> const& xCenterValues,
    std::vector<ndarray::Array<double, 1, 1>> const& wavelengthKnots,
    std::vector<ndarray::Array<double, 1, 1>> const& wavelengthValues,
    Array1D const& spatialOffsets,
    Array1D const& spectralOffsets,
    VisitInfo const& visitInfo,
    std::shared_ptr<lsst::daf::base::PropertySet> metadata
) : DetectorMap(bbox, fiberId, spatialOffsets, spectralOffsets, visitInfo, metadata),
    _xToFiberId(bbox.getWidth())
{
    utils::checkSize(xCenterKnots.size(), fiberId.size(), "DetectorMap: xCenterKnots");
    utils::checkSize(xCenterValues.size(), fiberId.size(), "DetectorMap: xCenterValues");
    utils::checkSize(wavelengthKnots.size(), fiberId.size(), "DetectorMap: wavelengthKnots");
    utils::checkSize(wavelengthValues.size(), fiberId.size(), "DetectorMap: wavelengthValues");

    // findFiberId requires that the fiberIds be sorted
    if (fiberId.size() > 1) {
        for (std::size_t ii = 1; ii < fiberId.size(); ++ii) {
            if (fiberId[ii] < fiberId[ii - 1]) {
                throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                                  "fiberId array is not sorted");
            }
        }
    }

    _xCenter.reserve(fiberId.size());
    _wavelength.reserve(fiberId.size());
    for (std::size_t ii = 0; ii < fiberId.size(); ++ii) {
        _xCenter.emplace_back(xCenterKnots[ii], xCenterValues[ii]);
        _wavelength.emplace_back(wavelengthKnots[ii], wavelengthValues[ii]);
        _row.emplace_back(wavelengthValues[ii], wavelengthKnots[ii]);
    }

    _set_xToFiberId();
}


SplinedDetectorMap::SplinedDetectorMap(
    lsst::geom::Box2I bbox,
    FiberIds const& fiberId,
    std::vector<SplinedDetectorMap::Spline> const& xCenter,
    std::vector<SplinedDetectorMap::Spline> const& wavelength,
    Array1D const& spatialOffsets,
    Array1D const& spectralOffsets,
    VisitInfo const& visitInfo,
    std::shared_ptr<lsst::daf::base::PropertySet> metadata
)  : DetectorMap(bbox, fiberId, spatialOffsets, spectralOffsets, visitInfo, metadata),
    _xCenter(xCenter),
    _wavelength(wavelength),
    _xToFiberId(bbox.getWidth())
{
    utils::checkSize(xCenter.size(), fiberId.size(), "xCenter");
    utils::checkSize(wavelength.size(), fiberId.size(), "wavelength");

    // findFiberId requires that the fiberIds be sorted
    if (fiberId.size() > 1) {
        for (std::size_t ii = 1; ii < fiberId.size(); ++ii) {
            if (fiberId[ii] < fiberId[ii - 1]) {
                throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                                  "fiberId array is not sorted");
            }
        }
    }

    for (std::size_t ii = 0; ii < fiberId.size(); ++ii) {
        _row.emplace_back(_wavelength[ii].getY(), _wavelength[ii].getX());
    }

    _set_xToFiberId();
}


std::shared_ptr<DetectorMap> SplinedDetectorMap::clone() const {
    std::vector<Spline> xCenter(_xCenter.begin(), _xCenter.end());
    std::vector<Spline> wavelength(_wavelength.begin(), _wavelength.end());
    return std::make_shared<SplinedDetectorMap>(
        getBBox(), getFiberId(), xCenter, wavelength,
        ndarray::copy(getSpatialOffsets()), ndarray::copy(getSpectralOffsets()),
        getVisitInfo(), getMetadata()->deepCopy()
    );
}


double SplinedDetectorMap::getXCenterImpl(
    int fiberId,
    double row
) const {
    auto const& spline = getXCenterSpline(fiberId);
    double const spatial = getSpatialOffset(fiberId);
    double const spectral = getSpectralOffset(fiberId);
    return spline(row - spectral) + spatial;
}


lsst::geom::PointD SplinedDetectorMap::findPointImpl(
    int fiberId,
    double wavelength
) const {
    std::size_t const index = getFiberIndex(fiberId);
    double const y = _row[index](wavelength) + getSpectralOffset(fiberId);
    if (y < 0 || y > getBBox().getHeight() - 1) {
        double const NaN = std::numeric_limits<double>::quiet_NaN();
        return lsst::geom::PointD(NaN, NaN);
    }
    double const x = getXCenter(fiberId, y);
    return lsst::geom::PointD(x, y);
}


double SplinedDetectorMap::findWavelengthImpl(
    int fiberId,
    double row
) const {
    if (row < 0 || row > getBBox().getHeight() - 1) {
        std::ostringstream os;
        os << "Row " << row << " is not in range [0, " << getBBox().getHeight() - 1 << "]";
        throw LSST_EXCEPT(lsst::pex::exceptions::RangeError, os.str());
    }
    auto const& spline = getWavelengthSpline(fiberId);
    return spline(row - getSpectralOffset(fiberId));
}


math::Spline<double> const&
SplinedDetectorMap::getXCenterSpline(
    int fiberId
) const {
    return _xCenter[getFiberIndex(fiberId)];
}


math::Spline<double> const&
SplinedDetectorMap::getWavelengthSpline(
    int fiberId
) const {
    return _wavelength[getFiberIndex(fiberId)];
}


void SplinedDetectorMap::setXCenter(
    int fiberId,
    ndarray::Array<double, 1, 1> const& knots,
    ndarray::Array<double, 1, 1> const& xCenter
) {
    _xCenter[getFiberIndex(fiberId)] = math::Spline<double>(knots, xCenter);
}


void SplinedDetectorMap::setWavelength(
    int fiberId,
    ndarray::Array<double, 1, 1> const& knots,
    ndarray::Array<double, 1, 1> const& wavelength
) {
    std::size_t const index = getFiberIndex(fiberId);
    _wavelength[index] = math::Spline<double>(knots, wavelength);
    _row[index] = math::Spline<double>(wavelength, knots);
}


void SplinedDetectorMap::_set_xToFiberId() {
    int const iy = getBBox().getHeight()/2;

    double lastCenter = -1e30;            // center of the last fibre we passed scanning to the right
    std::size_t lastIndex = -1;          // index of the last fibre we passed.  Special case starting with 0

    double nextCenter = _xCenter[0](iy); // center of the next fiber we're looking at
    for (std::size_t index = 0, x = 0; x != std::size_t(getBBox().getWidth()); ++x) {
        if (x - lastCenter > nextCenter - x) { // x is nearer to next_xc than last_xc
            lastCenter = nextCenter;
            lastIndex = index;

            if (index < getNumFibers() - 1) {
                ++index;
            }
            nextCenter = _xCenter[index](iy);
        }

        _xToFiberId[x] = getFiberId()[lastIndex];
    }
}


namespace {

// Singleton class that manages the persistence catalog's schema and keys
class SplinedDetectorMapSchema {
    using IntArray = lsst::afw::table::Array<int>;
    using DoubleArray = lsst::afw::table::Array<double>;
  public:
    lsst::afw::table::Schema schema;
    lsst::afw::table::Box2IKey bbox;
    lsst::afw::table::Key<IntArray> fiberId;
    lsst::afw::table::Key<IntArray> xCenter;
    lsst::afw::table::Key<IntArray> wavelength;
    lsst::afw::table::Key<DoubleArray> spatialOffset;
    lsst::afw::table::Key<DoubleArray> spectralOffset;
    lsst::afw::table::Key<int> visitInfo;

    static SplinedDetectorMapSchema const &get() {
        static SplinedDetectorMapSchema const instance;
        return instance;
    }

  private:
    SplinedDetectorMapSchema()
      : schema(),
        bbox(lsst::afw::table::Box2IKey::addFields(schema, "bbox", "bounding box", "pixel")),
        fiberId(schema.addField<IntArray>("fiberId", "fiber identifiers", "", 0)),
        xCenter(schema.addField<IntArray>("xCenter", "xCenter spline references", "", 0)),
        wavelength(schema.addField<IntArray>("wavelength", "wavelength spline references", "", 0)),
        spatialOffset(schema.addField<DoubleArray>("spatialOffset", "slit offsets in x", "micron", 0)),
        spectralOffset(schema.addField<DoubleArray>("spectralOffset", "slit offsets in y", "micron", 0)),
        visitInfo(schema.addField<int>("visitInfo", "visitInfo reference", ""))
        {}
};

}  // anonymous namespace


void SplinedDetectorMap::write(lsst::afw::table::io::OutputArchiveHandle & handle) const {
    SplinedDetectorMapSchema const &schema = SplinedDetectorMapSchema::get();
    lsst::afw::table::BaseCatalog cat = handle.makeCatalog(schema.schema);
    std::shared_ptr<lsst::afw::table::BaseRecord> record = cat.addNew();
    record->set(schema.bbox, getBBox());
    ndarray::Array<int, 1, 1> const fiberId = ndarray::copy(getFiberId());
    record->set(schema.fiberId, fiberId);

    ndarray::Array<int, 1, 1> xCenter = ndarray::allocate(getNumFibers());
    ndarray::Array<int, 1, 1> wavelength = ndarray::allocate(getNumFibers());
    for (std::size_t ii = 0; ii < getNumFibers(); ++ii) {
        xCenter[ii] = handle.put(_xCenter[ii]);
        wavelength[ii] = handle.put(_wavelength[ii]);
    }

    record->set(schema.xCenter, xCenter);
    record->set(schema.wavelength, wavelength);
    ndarray::Array<double, 1, 1> spatialOffset = ndarray::copy(getSpatialOffsets());
    ndarray::Array<double, 1, 1> spectralOffset = ndarray::copy(getSpectralOffsets());
    record->set(schema.spatialOffset, spatialOffset);
    record->set(schema.spectralOffset, spectralOffset);
    record->set(schema.visitInfo, handle.put(getVisitInfo()));
    // XXX dropping metadata on the floor, since we can't write a header
    handle.saveCatalog(cat);
}


class SplinedDetectorMap::Factory : public lsst::afw::table::io::PersistableFactory {
  public:
    std::shared_ptr<lsst::afw::table::io::Persistable> read(
        lsst::afw::table::io::InputArchive const& archive,
        lsst::afw::table::io::CatalogVector const& catalogs
    ) const override {
        static auto const& schema = SplinedDetectorMapSchema::get();
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        lsst::afw::table::BaseRecord const& record = catalogs.front().front();
        LSST_ARCHIVE_ASSERT(record.getSchema() == schema.schema);

        lsst::geom::Box2I bbox = record.get(schema.bbox);
        ndarray::Array<int, 1, 1> fiberId = ndarray::copy(record.get(schema.fiberId));
        std::size_t const numFibers = fiberId.size();

        std::vector<Spline> xCenter;
        std::vector<Spline> wavelength;
        xCenter.reserve(numFibers);
        wavelength.reserve(numFibers);
        for (std::size_t ii = 0; ii < numFibers; ++ii) {
            xCenter.emplace_back(*archive.get<Spline>(record.get(schema.xCenter)[ii]));
            wavelength.emplace_back(*archive.get<Spline>(record.get(schema.wavelength)[ii]));
        }

        ndarray::Array<double, 1, 1> spatialOffset = ndarray::copy(record.get(schema.spatialOffset));
        ndarray::Array<double, 1, 1> spectralOffset = ndarray::copy(record.get(schema.spectralOffset));
        assert (wavelength.size() == numFibers);
        assert(spatialOffset.getNumElements() == numFibers);
        assert(spectralOffset.getNumElements() == numFibers);
        auto visitInfo = archive.get<lsst::afw::image::VisitInfo>(record.get(schema.visitInfo));

        return std::make_shared<SplinedDetectorMap>(bbox, fiberId, xCenter, wavelength,
                                                    spatialOffset, spectralOffset, *visitInfo);
    }

    Factory(std::string const& name) : lsst::afw::table::io::PersistableFactory(name) {}
};


namespace {

SplinedDetectorMap::Factory registration("SplinedDetectorMap");

}  // anonymous namespace


}}}  // namespace pfs::drp::stella
