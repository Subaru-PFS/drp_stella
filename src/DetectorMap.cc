#include <algorithm>
#include <cassert>
#include <cmath>
#include <sstream>

#include "lsst/pex/exceptions/Exception.h"
#include "lsst/daf/base.h"
#include "lsst/afw/table.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/Persistable.cc"

#include "pfs/drp/stella/utils/checkSize.h"
#include "pfs/drp/stella/DetectorMap.h"

namespace pfs { namespace drp { namespace stella {

DetectorMap::DetectorMap(
    lsst::geom::Box2I bbox,
    FiberMap const& fiberId,
    std::vector<ndarray::Array<float, 1, 1>> const& centerKnots,
    std::vector<ndarray::Array<float, 1, 1>> const& centerValues,
    std::vector<ndarray::Array<float, 1, 1>> const& wavelengthKnots,
    std::vector<ndarray::Array<float, 1, 1>> const& wavelengthValues,
    Array2D const& slitOffsets,
    VisitInfo const& visitInfo,
    std::shared_ptr<lsst::daf::base::PropertySet> metadata
) : _nFiber(fiberId.getShape()[0]),
    _bbox(bbox),
    _fiberId(ndarray::copy(fiberId)),
    _yToXCenter(_nFiber),
    _yToWavelength(_nFiber),
    _xToFiberId(_bbox.getWidth()),
    _slitOffsets(ndarray::makeVector(std::size_t(3), _nFiber)),
    _visitInfo(visitInfo),
    _metadata(metadata ? metadata : std::make_shared<lsst::daf::base::PropertyList>())
{
    utils::checkSize(centerKnots.size(), _nFiber, "DetectorMap: centerKnots");
    utils::checkSize(centerValues.size(), _nFiber, "DetectorMap: centerValues");
    utils::checkSize(wavelengthKnots.size(), _nFiber, "DetectorMap: wavelengthKnots");
    utils::checkSize(wavelengthValues.size(), _nFiber, "DetectorMap: wavelengthValues");
    if (!slitOffsets.isEmpty()) {
        setSlitOffsets(slitOffsets);   // actually this is where we check slitOffsets
    } else {
        _slitOffsets.deep() = 0.0;      // Assume that all the fibres are aligned perfectly
    }

    for (std::size_t ii = 0; ii < _nFiber; ++ii) {
        _yToXCenter[ii] = std::make_shared<math::Spline<float>>(centerKnots[ii], centerValues[ii]);
        _yToWavelength[ii] = std::make_shared<math::Spline<float>>(wavelengthKnots[ii], wavelengthValues[ii]);
    }
    _set_xToFiberId();
}


DetectorMap::DetectorMap(
    lsst::geom::Box2I bbox,
    FiberMap const& fiberId,
    std::vector<std::shared_ptr<DetectorMap::Spline const>> const& center,
    std::vector<std::shared_ptr<DetectorMap::Spline const>> const& wavelength,
    Array2D const& slitOffsets,
    VisitInfo const& visitInfo,
    std::shared_ptr<lsst::daf::base::PropertySet> metadata
) : _nFiber(fiberId.getShape()[0]),
    _bbox(bbox),
    _fiberId(ndarray::copy(fiberId)),
    _yToXCenter(center),
    _yToWavelength(wavelength),
    _xToFiberId(_bbox.getWidth()),
    _slitOffsets(ndarray::makeVector(std::size_t(3), _nFiber)),
    _visitInfo(visitInfo),
    _metadata(metadata ? metadata : std::make_shared<lsst::daf::base::PropertyList>())
{
    utils::checkSize(center.size(), _nFiber, "DetectorMap: center");
    utils::checkSize(wavelength.size(), _nFiber, "DetectorMap: wavelength");
    if (!slitOffsets.isEmpty()) {
        setSlitOffsets(slitOffsets);   // actually this is where we check slitOffsets
    } else {
        _slitOffsets.deep() = 0.0;      // Assume that all the fibres are aligned perfectly
    }
    _set_xToFiberId();
}


/*
 * Return a fiberIdx given a fiberId
 */
std::size_t
DetectorMap::getFiberIndex(int fiberId) const
{
    auto el = std::lower_bound(_fiberId.begin(), _fiberId.end(), fiberId);
    if (el == _fiberId.end() || *el != fiberId) {
        std::ostringstream os;
        os << "Unknown fiberId " << fiberId;
        throw LSST_EXCEPT(lsst::pex::exceptions::RangeError, os.str());
    }
    return el - _fiberId.begin();      // index into _fiberId
}

/*
 * Set the offsets of the wavelengths of each fibre (in floating-point pixels)
 */
void
DetectorMap::setSlitOffsets(ndarray::Array<float, 2, 1> const& slitOffsets)
{
    if (slitOffsets.getShape()[0] != 3) {
        std::ostringstream os;
        os << "Number of types of offsets == " << slitOffsets.getShape()[0] <<
            ".  Expected three (dx, dy, dfocus)";
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthError, os.str());
    }
    if (slitOffsets.getShape()[1] != _nFiber) {
        std::ostringstream os;
        os << "Number of offsets == " << slitOffsets.getShape()[1] <<
            " != number of fibres == " << _nFiber;
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthError, os.str());
    }

    _slitOffsets.deep() = slitOffsets;
}

math::Spline<float> const&
DetectorMap::getWavelengthSpline(std::size_t index) const {
    if (index < 0 || index >= _nFiber) {
        std::ostringstream os;
        os << "Fiber index " << index << " out of range [" << 0 << "," << _nFiber << ")";
        throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundError, os.str());
    }
    auto spline = _yToWavelength[index];
    if (!spline) {
        std::ostringstream os;
        os << "Wavelength spline for fiber index " << index << " not set";
        throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundError, os.str());
    }
    return *spline;
}

math::Spline<float> const&
DetectorMap::getCenterSpline(std::size_t index) const {
    if (index < 0 || index >= _nFiber) {
        std::ostringstream os;
        os << "Fiber index " << index << " out of range [" << 0 << "," << _nFiber << ")";
        throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundError, os.str());
    }
    auto spline = _yToXCenter[index];
    if (!spline) {
        std::ostringstream os;
        os << "Center spline for fiber index " << index << " not set";
        throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundError, os.str());
    }
    return *spline;
}

/*
 * Return the wavelength values for a fibre
 */
ndarray::Array<float, 1, 1>
DetectorMap::getWavelength(std::size_t fiberId) const
{
    std::size_t const index = getFiberIndex(fiberId);
    auto const & spline = getWavelengthSpline(index);
    const float slitOffsetY = _slitOffsets[DY][index];

    ndarray::Array<float, 1, 1> res(_bbox.getHeight());
    for (int i = 0; i != _bbox.getHeight(); ++i) {
        res[i] = spline(i - slitOffsetY);
    }

    return res;
}

std::vector<DetectorMap::Array1D> DetectorMap::getWavelength() const {
    std::size_t numFibers = _fiberId.getNumElements();
    std::vector<Array1D> result{numFibers};
    for (std::size_t ii = 0; ii < numFibers; ++ii) {
        result[ii] = ndarray::copy(getWavelength(_fiberId[ii]));
    }
    return result;
}

float DetectorMap::getWavelength(std::size_t fiberId, float y) const {
    std::size_t const index = getFiberIndex(fiberId);
    auto const & spline = getWavelengthSpline(index);
    float const slitOffsetY = _slitOffsets[DY][index];
    return spline(y - slitOffsetY);
}

void DetectorMap::setWavelength(
    std::size_t fiberId,
    Array1D const& wavelength
) {
    Array1D rows = ndarray::allocate(_bbox.getHeight());
    int ii = _bbox.getMinY();
    for (auto rr = rows.begin(); rr != rows.end(); ++rr, ++ii) {
        *rr = ii;
    }
    setWavelength(fiberId, rows, wavelength);
}


/*
 * Return the wavelength values for a fibre
 */
void
DetectorMap::setWavelength(std::size_t fiberId,
                           ndarray::Array<float, 1, 1> const& knots,
                           ndarray::Array<float, 1, 1> const& wavelength)
{
    int const index = getFiberIndex(fiberId);
    _yToWavelength[index] = std::make_shared<Spline>(knots, wavelength);
}

/*
 * Return the xCenter values for a fibre
 */
ndarray::Array<float, 1, 1>
DetectorMap::getXCenter(std::size_t fiberId) const
{
    int const index = getFiberIndex(fiberId);
    auto const & spline = getCenterSpline(index);
    const float slitOffsetX = _slitOffsets[DX][index];
    const float slitOffsetY = _slitOffsets[DY][index];

    ndarray::Array<float, 1, 1> res(_bbox.getHeight());
    for (int i = 0; i != _bbox.getHeight(); ++i) {
        res[i] = spline(i - slitOffsetY) + slitOffsetX;
    }

    return res;
}

std::vector<DetectorMap::Array1D> DetectorMap::getXCenter() const {
    std::size_t numFibers = _fiberId.getNumElements();
    std::vector<Array1D> result{numFibers};
    for (std::size_t ii = 0; ii < numFibers; ++ii) {
        result[ii] = ndarray::copy(getXCenter(_fiberId[ii]));
    }
    return result;
}

/*
 * Return the xCenter values for a fibre
 */
float
DetectorMap::getXCenter(std::size_t fiberId, float y) const
{
    int const index = getFiberIndex(fiberId);
    auto const & spline = getCenterSpline(index);
    const float slitOffsetX = _slitOffsets[DX][index];
    const float slitOffsetY = _slitOffsets[DY][index];

    return spline(y - slitOffsetY) + slitOffsetX;
}


void DetectorMap::setXCenter(
    std::size_t fiberId,
    Array1D const& xCenter
) {
    Array1D rows = ndarray::allocate(_bbox.getHeight());
    int ii = _bbox.getMinY();
    for (auto rr = rows.begin(); rr != rows.end(); ++rr, ++ii) {
        *rr = ii;
    }
    setXCenter(fiberId, rows, xCenter);
}


/*
 * Update the xcenter values for a fibre
 */
void
DetectorMap::setXCenter(std::size_t fiberId,
                        ndarray::Array<float, 1, 1> const& knots,
                        ndarray::Array<float, 1, 1> const& xCenter)
{
    int const index = getFiberIndex(fiberId);
    _yToXCenter[index] = std::make_shared<Spline>(knots, xCenter);
}

/*
 * Return the position of the fiber trace on the detector, given a fiberId and wavelength
 */
lsst::geom::PointD
DetectorMap::findPoint(int fiberId,               ///< Desired fibreId
                       float wavelength           ///< desired wavelength
                      ) const
{
    auto fiberWavelength = getWavelength(fiberId);

    auto begin = fiberWavelength.begin();
    auto end = fiberWavelength.end();
    auto onePast = std::lower_bound(begin, end, wavelength); // pointer to element just larger than wavelength

    if (onePast == end) {
        double const NaN = std::numeric_limits<double>::quiet_NaN();
        return lsst::geom::PointD(NaN, NaN);
    }

    auto fiberXCenter = getXCenter(fiberId);
    const auto iy = onePast - begin;

    double x = fiberXCenter[iy];
    double y = iy;                      // just past the desired point

    if (iy > 0) {                       // interpolate to get better accuracy
        const float dy = -(fiberWavelength[iy] - wavelength)/(fiberWavelength[iy] - fiberWavelength[iy - 1]);
        x += dy*(fiberXCenter[iy] - fiberXCenter[iy - 1]);
        y += dy;
    }

    return lsst::geom::PointD(x, y);
}

/************************************************************************************************************/
/*
 * Return the position of the fiber trace on the detector, given a fiberId and wavelength
 */
float
DetectorMap::findWavelength(int fiberId,               ///< Desired fibreId
                            float row                  ///< desired row
                           ) const
{
    if (row < 0 || row > _bbox.getHeight() - 1) {
        std::ostringstream os;
        os << "Row " << row << " is not in range [0, " << _bbox.getHeight() - 1 << "]";
        throw LSST_EXCEPT(lsst::pex::exceptions::RangeError, os.str());
    }
    int const index = getFiberIndex(fiberId);
    auto const & spline = getWavelengthSpline(index);

    return spline(row - getSlitOffsets(fiberId)[DY]);
}

/************************************************************************************************************/
/*
 * Return the fiberId given a position on the detector
 */
int
DetectorMap::findFiberId(lsst::geom::PointD pixelPos // position on detector
                          ) const
{
    float const x = pixelPos[0], y = pixelPos[1];
    std::size_t const maxIter = 2*_fiberId.getNumElements();  // maximum number of iterations

    if (!_bbox.contains(lsst::geom::PointI(x, y))) {
        std::ostringstream os;
        os << "Point " << pixelPos << " does not lie within BBox " << _bbox;
        throw LSST_EXCEPT(lsst::pex::exceptions::RangeError, os.str());
    }

    int fiberId = _xToFiberId[int(x)];    // first guess at the fiberId
    std::size_t index = getFiberIndex(fiberId);
    std::size_t iterations = 0;
    for (bool updatedIndex = true; updatedIndex; ++iterations) {
        auto const & spline0 = getCenterSpline(index);
        float const xCenter0 = spline0(y - getSlitOffsets()[DY][index]) + getSlitOffsets()[DX][index];

        updatedIndex = false;
        if (index > 0) {
            auto const & splineMinus = getCenterSpline(index - 1);
            float const xCenterMinus = splineMinus(y - getSlitOffsets()[DY][index - 1]) +
                getSlitOffsets()[DX][index - 1];

            if (std::fabs(x - xCenterMinus) < std::fabs(x - xCenter0)) {
                --index;
                updatedIndex = true;
                continue;
            }
        }
        if (index < _yToXCenter.size() - 1) {
            auto const & splinePlus = getCenterSpline(index + 1);
            float const xCenterPlus = splinePlus(y - getSlitOffsets()[DY][index]) +
                getSlitOffsets()[DX][index + 1];

            if (std::fabs(x - xCenterPlus) < std::fabs(x - xCenter0)) {
                ++index;
                updatedIndex = true;
                continue;
            }
        }
    }
    if (iterations > maxIter) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Exceeded maximum number of iterations");
    }

    return _fiberId[index];
}

/************************************************************************************************************/

/*
 * set _xToFiberId, an array giving the fiber ID for each pixel across the centre of the chip
 */
void
DetectorMap::_set_xToFiberId()
{
    int const iy = _bbox.getHeight()/2;

    float lastCenter = -1e30;            // center of the last fibre we passed scanning to the right
    std::size_t lastIndex = -1;          // index of the last fibre we passed.  Special case starting with 0

    float nextCenter = getXCenter(_fiberId[0], iy); // center of the next fiber we're looking at
    for (std::size_t index = 0, x = 0; x != std::size_t(_bbox.getWidth()); ++x) {
        if (x - lastCenter > nextCenter - x) { // x is nearer to next_xc than last_xc
            lastCenter = nextCenter;
            lastIndex = index;

            if (index < _fiberId.size() - 1) {
                ++index;
            }
            nextCenter = getXCenter(_fiberId[index], iy);
        }

        _xToFiberId[x] = _fiberId[lastIndex];
    }
}


namespace {

// Singleton class that manages the persistence catalog's schema and keys
class DetectorMapSchema {
    using IntArray = lsst::afw::table::Array<int>;
    using FloatArray = lsst::afw::table::Array<float>;
  public:
    lsst::afw::table::Schema schema;
    lsst::afw::table::Box2IKey bbox;
    lsst::afw::table::Key<IntArray> fiberId;
    lsst::afw::table::Key<IntArray> center;
    lsst::afw::table::Key<IntArray> wavelength;
    lsst::afw::table::Key<FloatArray> slitOffset1;
    lsst::afw::table::Key<FloatArray> slitOffset2;
    lsst::afw::table::Key<FloatArray> slitOffset3;
    lsst::afw::table::Key<int> visitInfo;

    static DetectorMapSchema const &get() {
        static DetectorMapSchema const instance;
        return instance;
    }

  private:
    DetectorMapSchema()
      : schema(),
        bbox(lsst::afw::table::Box2IKey::addFields(schema, "bbox", "bounding box", "pixel")),
        fiberId(schema.addField<IntArray>("fiberId", "fiber identifiers", "", 0)),
        center(schema.addField<IntArray>("center", "center spline references", "", 0)),
        wavelength(schema.addField<IntArray>("wavelength", "wavelength spline references", "", 0)),
        slitOffset1(schema.addField<FloatArray>("slitOffset1", "slit offsets in x", "micron", 0)),
        slitOffset2(schema.addField<FloatArray>("slitOffset2", "slit offsets in y", "micron", 0)),
        slitOffset3(schema.addField<FloatArray>("slitOffset3", "slit offsets in focus", "micron", 0)),
        visitInfo(schema.addField<int>("visitInfo", "visitInfo reference", "")) {
            schema.getCitizen().markPersistent();
    }
};

}  // anonymous namespace


void DetectorMap::write(lsst::afw::table::io::OutputArchiveHandle & handle) const {
    DetectorMapSchema const &schema = DetectorMapSchema::get();
    lsst::afw::table::BaseCatalog cat = handle.makeCatalog(schema.schema);
    PTR(lsst::afw::table::BaseRecord) record = cat.addNew();
    record->set(schema.bbox, _bbox);
    ndarray::Array<int, 1, 1> const fiberId = ndarray::copy(_fiberId);
    record->set(schema.fiberId, fiberId);

    ndarray::Array<int, 1, 1> center = ndarray::allocate(_nFiber);
    ndarray::Array<int, 1, 1> wavelength = ndarray::allocate(_nFiber);
    for (std::size_t ii = 0; ii < _nFiber; ++ii) {
        center[ii] = handle.put(_yToXCenter[ii]);
        wavelength[ii] = handle.put(_yToWavelength[ii]);
    }

    record->set(schema.center, center);
    record->set(schema.wavelength, wavelength);
    ndarray::Array<float, 1, 1> dx = ndarray::copy(_slitOffsets[DX]);
    ndarray::Array<float, 1, 1> dy = ndarray::copy(_slitOffsets[DY]);
    ndarray::Array<float, 1, 1> dFocus = ndarray::copy(_slitOffsets[DFOCUS]);
    record->set(schema.slitOffset1, dx);
    record->set(schema.slitOffset2, dy);
    record->set(schema.slitOffset3, dFocus);
    record->set(schema.visitInfo, handle.put(_visitInfo));
    /// XXX dropping metadata on the floor, since we can't write a header
    handle.saveCatalog(cat);
}


class DetectorMap::Factory : public lsst::afw::table::io::PersistableFactory {
  public:
    std::shared_ptr<lsst::afw::table::io::Persistable> read(
        lsst::afw::table::io::InputArchive const& archive,
        lsst::afw::table::io::CatalogVector const& catalogs
    ) const override {
        static auto const& schema = DetectorMapSchema::get();
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        lsst::afw::table::BaseRecord const& record = catalogs.front().front();
        LSST_ARCHIVE_ASSERT(record.getSchema() == schema.schema);

        lsst::geom::Box2I bbox = record.get(schema.bbox);
        FiberMap fiberId = record.get(schema.fiberId);
        std::size_t const numFibers = fiberId.size();

        std::vector<std::shared_ptr<Spline const>> center;
        std::vector<std::shared_ptr<Spline const>> wavelength;
        center.reserve(numFibers);
        wavelength.reserve(numFibers);
        for (std::size_t ii = 0; ii < numFibers; ++ii) {
            center.emplace_back(archive.get<Spline>(record.get(schema.center)[ii]));
            wavelength.emplace_back(archive.get<Spline>(record.get(schema.wavelength)[ii]));
        }

        ndarray::Array<float const, 1, 1> dx = record.get(schema.slitOffset1);
        ndarray::Array<float const, 1, 1> dy = record.get(schema.slitOffset1);
        ndarray::Array<float const, 1, 1> dFocus = record.get(schema.slitOffset1);
        assert (wavelength.size() == numFibers);
        assert(dx.getNumElements() == numFibers);
        assert(dy.getNumElements() == numFibers);
        assert(dFocus.getNumElements() == numFibers);
        Array2D slitOffsets = ndarray::allocate(3u, numFibers);
        slitOffsets[DX].deep() = dx;
        slitOffsets[DY].deep() = dy;
        slitOffsets[DFOCUS].deep() = dFocus;
        auto visitInfo = archive.get<lsst::afw::image::VisitInfo>(record.get(schema.visitInfo));

        return std::make_shared<DetectorMap>(bbox, fiberId, center, wavelength, slitOffsets, *visitInfo);
    }

    Factory(std::string const& name) : lsst::afw::table::io::PersistableFactory(name) {}
};


DetectorMap::Factory registration("DetectorMap");


}}}
