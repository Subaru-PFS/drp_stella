#include <set>
#include <numeric>

#include "ndarray.h"

#include "lsst/afw/image.h"
#include "lsst/afw/table.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/Persistable.cc"

#include "pfs/drp/stella/utils/checkSize.h"
#include "pfs/drp/stella/utils/math.h"
#include "pfs/drp/stella/GlobalDetectorMap.h"


namespace pfs {
namespace drp {
namespace stella {


GlobalDetectorMap::GlobalDetectorMap(
    lsst::geom::Box2I bbox,
    FiberIds const& fiberId,
    int distortionOrder,
    bool dualDetector,
    ndarray::Array<double, 1, 1> const& parameters,
    Array1D const& spatialOffsets,
    Array1D const& spectralOffsets,
    VisitInfo const& visitInfo,
    std::shared_ptr<lsst::daf::base::PropertySet> metadata
) : DetectorMap(bbox, fiberId, spatialOffsets, spectralOffsets, visitInfo, metadata),
    _model(bbox, distortionOrder, fiberId, dualDetector, parameters.deep(),
           spatialOffsets.deep(), spectralOffsets.deep())
{
    _setSplines();
}


GlobalDetectorMap::GlobalDetectorMap(
    GlobalDetectorModel const& model,
    VisitInfo const& visitInfo,
    std::shared_ptr<lsst::daf::base::PropertySet> metadata
) : DetectorMap(model.getBBox(), model.getFiberId(),
                    utils::convertArray<float>(model.getSpatialOffsets()),
                    utils::convertArray<float>(model.getSpectralOffsets()),
                    visitInfo, metadata),
    _model(model)
{
    _setSplines();
}


void GlobalDetectorMap::_setSplines() {
    _rowToWavelength.clear();
    _rowToXCenter.clear();
    _rowToWavelength.reserve(getNumFibers());
    _rowToXCenter.reserve(getNumFibers());

    ParamType const wavelengthCenter = _model.getWavelengthCenter();
    ParamType const dispersion = _model.getDispersion();
    assert(dispersion > 0);  // to prevent infinite loops
    std::size_t const height = getBBox().getHeight();
    for (std::size_t ii = 0; ii < getNumFibers(); ++ii) {
        std::vector<ParamType> wavelength;
        std::vector<ParamType> xx;
        std::vector<ParamType> yy;

        wavelength.reserve(height);
        xx.reserve(height);
        yy.reserve(height);
        int fiberId = getFiberId()[ii];

        // Iterate up in wavelength until we drop off the edge of the detector
        for (ParamType wl = wavelengthCenter; true; wl += dispersion) {
            auto const point = _model(fiberId, wl, ii);
            wavelength.push_back(wl);
            xx.push_back(point.getX());
            yy.push_back(point.getY());
            if (point.getY() > height || point.getY() < 0) {
                break;
            }
        }
        // Iterate down in wavelength until we drop off the edge of the detector
        for (ParamType wl = wavelengthCenter - dispersion; true; wl -= dispersion) {
            auto const point = _model(fiberId, wl, ii);
            wavelength.push_back(wl);
            xx.push_back(point.getX());
            yy.push_back(point.getY());
            if (point.getY() < 0 || point.getY() > height) {
                break;
            }
        }
        std::size_t const length = wavelength.size();

        // Sort into monotonic ndarrays
        // With some care we could simply rearrange, but easier to code the sort
        // and performance isn't critical.
        ndarray::Array<std::size_t, 1, 1> indices = ndarray::allocate(length);
        for (std::size_t ii = 0; ii < length; ++ii) {
            indices[ii] = ii;
        }
        std::sort(indices.begin(), indices.end(),
                  [&yy](std::size_t left, std::size_t right) { return yy[left] < yy[right]; });

        ndarray::Array<ParamType, 1, 1> wlArray = ndarray::allocate(length);
        ndarray::Array<ParamType, 1, 1> xArray = ndarray::allocate(length);
        ndarray::Array<ParamType, 1, 1> yArray = ndarray::allocate(length);
        for (std::size_t ii = 0; ii < length; ++ii) {
            std::size_t const index = indices[ii];
            wlArray[ii] = wavelength[index];
            xArray[ii] = xx[index];
            yArray[ii] = yy[index];
        }
        _rowToWavelength.emplace_back(yArray, wlArray);
        _rowToXCenter.emplace_back(yArray, xArray);
    }
}


DetectorMap::Array1D GlobalDetectorMap::getXCenter(
    int fiberId
) const {
    Spline const& spline = _rowToXCenter[getFiberIndex(fiberId)];
    std::size_t const height = getBBox().getHeight();
    Array1D out = ndarray::allocate(height);
    for (std::size_t yy = 0; yy < height; ++yy) {
        out[yy] = spline(yy);
    }
    return out;
}


float GlobalDetectorMap::getXCenter(
    int fiberId,
    float row
) const {
    Spline const& spline = _rowToXCenter[getFiberIndex(fiberId)];
    return spline(row);
}


DetectorMap::Array1D GlobalDetectorMap::getWavelength(
    int fiberId
) const {
    Spline const& spline = _rowToWavelength[getFiberIndex(fiberId)];
    std::size_t const height = getBBox().getHeight();
    Array1D out = ndarray::allocate(height);
    for (std::size_t yy = 0; yy < height; ++yy) {
        out[yy] = spline(yy);
    }
    return out;
}


float GlobalDetectorMap::getWavelength(
    int fiberId,
    float row
) const {
    Spline const& spline = _rowToWavelength[getFiberIndex(fiberId)];
    return spline(row);
}


int GlobalDetectorMap::findFiberId(lsst::geom::PointD const& point) const {
    if (getNumFibers() == 1) {
        return getFiberId()[0];
    }
    ParamType const xx = point.getX();
    ParamType const yy = point.getY();

    // We know x as a function of fiberId (given y),
    // and x is monotonic with fiberId (for fixed y),
    // so we can find fiberId by bisection.
    std::size_t lowIndex = 0;
    std::size_t highIndex = getNumFibers() - 1;
    ParamType xLow = _rowToXCenter[lowIndex](yy);
    ParamType xHigh = _rowToXCenter[highIndex](yy);
    bool const increasing = xHigh > xLow;  // Does x increase with increasing fiber index?
    while (highIndex - lowIndex > 1) {
        std::size_t newIndex = lowIndex + (highIndex - lowIndex)/2;
        ParamType xNew = _rowToXCenter[newIndex](yy);
        if (increasing) {
            assert(xNew > xLow && xNew < xHigh);
            if (xx > xNew) {
                lowIndex = newIndex;
                xLow = xNew;
            } else {
                highIndex = newIndex;
                xHigh = xNew;
            }
        } else {
            assert(xNew < xLow && xNew > xHigh);
            if (xx < xNew) {
                lowIndex = newIndex;
                xLow = xNew;
            } else {
                highIndex = newIndex;
                xHigh = xNew;
            }
        }
    }
    return std::abs(xx - xLow) < std::abs(xx - xHigh) ? getFiberId()[lowIndex] : getFiberId()[highIndex];
}


lsst::geom::PointD GlobalDetectorMap::findPoint(
    int fiberId,
    float wavelength
) const {
    return _model(fiberId, wavelength);
}


GlobalDetectorMap::Array2D GlobalDetectorMap::findPoint(
    GlobalDetectorMap::FiberIds const& fiberId,
    GlobalDetectorMap::Array1D const& wavelength
) const {
    std::size_t const length = fiberId.size();
    utils::checkSize(wavelength.size(), length, "wavelength");
    Array2D out = ndarray::allocate(2, length);
    for (std::size_t ii = 0; ii < length; ++ii) {
        auto const point = _model(fiberId[ii], wavelength[ii]);
        out[0][ii] = point.getX();
        out[1][ii] = point.getY();
    }
    return out;
}


float GlobalDetectorMap::findWavelength(int fiberId, float row) const {
    Spline const& spline = _rowToWavelength[getFiberIndex(fiberId)];
    return spline(row);
}


void GlobalDetectorMap::_resetSlitOffsets() {
    _model = GlobalDetectorModel(_model.getBBox(), _model.getDistortionOrder(), getFiberId(),
                                 _model.getDualDetector(), _model.getParameters().deep(),
                                 getSpatialOffsets(), getSpectralOffsets());
    _setSplines();
}


namespace {

// Singleton class that manages the persistence catalog's schema and keys
class GlobalDetectorMapSchema {
    using IntArray = lsst::afw::table::Array<int>;
    using FloatArray = lsst::afw::table::Array<float>;
    using DoubleArray = lsst::afw::table::Array<double>;
  public:
    lsst::afw::table::Schema schema;
    lsst::afw::table::Box2IKey bbox;
    lsst::afw::table::Key<int> distortionOrder;
    lsst::afw::table::Key<IntArray> fiberId;
    lsst::afw::table::Key<int> dualDetector;
    lsst::afw::table::Key<DoubleArray> parameters;
    lsst::afw::table::Key<FloatArray> spatialOffset;
    lsst::afw::table::Key<FloatArray> spectralOffset;
    lsst::afw::table::Key<int> visitInfo;

    static GlobalDetectorMapSchema const &get() {
        static GlobalDetectorMapSchema const instance;
        return instance;
    }

  private:
    GlobalDetectorMapSchema()
      : schema(),
        bbox(lsst::afw::table::Box2IKey::addFields(schema, "bbox", "bounding box", "pixel")),
        distortionOrder(schema.addField<int>("distortionOrder", "polynomial order for distortion", "")),
        fiberId(schema.addField<IntArray>("fiberId", "fiber identifiers", "", 0)),
        dualDetector(schema.addField<int>("dualDetector", "have two individual detectors?", "", 0)),
        parameters(schema.addField<DoubleArray>("parameters", "distortion parameters", "", 0)),
        spatialOffset(schema.addField<FloatArray>("spatialOffset", "slit offsets in x", "micron", 0)),
        spectralOffset(schema.addField<FloatArray>("spectralOffset", "slit offsets in y", "micron", 0)),
        visitInfo(schema.addField<int>("visitInfo", "visitInfo reference", "")) {
            schema.getCitizen().markPersistent();
    }
};

}  // anonymous namespace


void GlobalDetectorMap::write(lsst::afw::table::io::OutputArchiveHandle & handle) const {
    GlobalDetectorMapSchema const &schema = GlobalDetectorMapSchema::get();
    lsst::afw::table::BaseCatalog cat = handle.makeCatalog(schema.schema);
    PTR(lsst::afw::table::BaseRecord) record = cat.addNew();
    record->set(schema.bbox, getBBox());
    record->set(schema.distortionOrder, getDistortionOrder());
    ndarray::Array<int, 1, 1> const fiberId = ndarray::copy(getFiberId());
    record->set(schema.fiberId, fiberId);
    record->set(schema.dualDetector, getDualDetector());
    ndarray::Array<double, 1, 1> parameters = ndarray::copy(getParameters());
    record->set(schema.parameters, parameters);
    ndarray::Array<float, 1, 1> spatialOffset = ndarray::copy(getSpatialOffsets());
    record->set(schema.spatialOffset, spatialOffset);
    ndarray::Array<float, 1, 1> spectralOffset = ndarray::copy(getSpectralOffsets());
    record->set(schema.spectralOffset, spectralOffset);
    record->set(schema.visitInfo, handle.put(getVisitInfo()));
    // XXX dropping metadata on the floor, since we can't write a header
    handle.saveCatalog(cat);
}


class GlobalDetectorMap::Factory : public lsst::afw::table::io::PersistableFactory {
  public:
    std::shared_ptr<lsst::afw::table::io::Persistable> read(
        lsst::afw::table::io::InputArchive const& archive,
        lsst::afw::table::io::CatalogVector const& catalogs
    ) const override {
        static auto const& schema = GlobalDetectorMapSchema::get();
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        lsst::afw::table::BaseRecord const& record = catalogs.front().front();
        LSST_ARCHIVE_ASSERT(record.getSchema() == schema.schema);

        lsst::geom::Box2I bbox = record.get(schema.bbox);
        int distortionOrder = record.get(schema.distortionOrder);
        ndarray::Array<int, 1, 1> fiberId = ndarray::copy(record.get(schema.fiberId));
        bool dualDetector = record.get(schema.dualDetector);
        ndarray::Array<double, 1, 1> parameters = ndarray::copy(record.get(schema.parameters));
        ndarray::Array<float, 1, 1> spatialOffset = ndarray::copy(record.get(schema.spatialOffset));
        ndarray::Array<float, 1, 1> spectralOffset = ndarray::copy(record.get(schema.spectralOffset));        assert(spatialOffset.getNumElements() == fiberId.size());
        assert(spectralOffset.getNumElements() == fiberId.size());
        auto visitInfo = archive.get<lsst::afw::image::VisitInfo>(record.get(schema.visitInfo));

        return std::make_shared<GlobalDetectorMap>(bbox, fiberId, distortionOrder, dualDetector,
                                                   parameters, spatialOffset, spectralOffset, *visitInfo);
    }

    Factory(std::string const& name) : lsst::afw::table::io::PersistableFactory(name) {}
};


namespace {

GlobalDetectorMap::Factory registration("GlobalDetectorMap");

}  // anonymous namespace


}}}  // namespace pfs::drp::stella