#include "lsst/afw/table.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/Persistable.cc"

#include "pfs/drp/stella/math/AffineTransform.h"
#include "pfs/drp/stella/utils/math.h"
#include "pfs/drp/stella/OpticalModelDetectorMap.h"

namespace pfs {
namespace drp {
namespace stella {


////////////////////////////////////////////////////////////////////////////////
// OpticalDetectorMap::Data
////////////////////////////////////////////////////////////////////////////////


OpticalModelDetectorMap::Array1D OpticalModelDetectorMap::Data::getArray(Coordinate coord) const {
    switch (coord) {
        case WAVELENGTH:
            return wavelength;
        case SLIT_SPATIAL:
            return slit[ndarray::view(0)];
        case SLIT_SPECTRAL:
            return slit[ndarray::view(1)];
        case DETECTOR_X:
            return detector[ndarray::view(0)];
        case DETECTOR_Y:
            return detector[ndarray::view(1)];
        case PIXELS_P:
            return pixels[ndarray::view(0)];
        case PIXELS_Q:
            return pixels[ndarray::view(1)];
        default:
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, "Invalid system");
    }
}


////////////////////////////////////////////////////////////////////////////////
// OpticalModelDetectorMap
////////////////////////////////////////////////////////////////////////////////


namespace {


double constexpr NOT_A_NUMBER = std::numeric_limits<double>::quiet_NaN();


/// Get minimum and maximum values from the 2D wavelength array
std::pair<double, double> getWavelengthRange(GridTransform::Array2D const& wavelengths) {
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();
    for (std::size_t ii = 0; ii < wavelengths.size(); ++ii) {
        for (std::size_t jj = 0; jj < wavelengths[ii].size(); ++jj) {
            double const wavelength = wavelengths[ii][jj];
            min = std::min(min, wavelength);
            max = std::max(max, wavelength);
        }
    }
    return {min, max};
}


}  // anonymous namespace


OpticalModelDetectorMap::OpticalModelDetectorMap(
    SlitModel const& slitModel,
    OpticsModel const& opticsModel,
    DetectorModel const& detectorModel,
    VisitInfo const& visitInfo,
    std::shared_ptr<lsst::daf::base::PropertySet> metadata
) : DetectorMap(
        detectorModel.getBBox(),
        slitModel.getFiberId(),
        slitModel.getSpatialOffsets(),
        slitModel.getSpectralOffsets(),
        visitInfo,
        metadata
    ),
    _slitModel(slitModel),
    _opticsModel(opticsModel),
    _detectorModel(detectorModel),
    _wavelengthRange(getWavelengthRange(opticsModel.getSpectral())),
    _numKnots(75)
{}


std::shared_ptr<DetectorMap> OpticalModelDetectorMap::clone() const {
    using DistortionList = std::vector<std::shared_ptr<Distortion>>;

    return std::make_shared<OpticalModelDetectorMap>(
        _slitModel.copy(),
        _opticsModel.copy(),
        _detectorModel.copy(),
        lsst::afw::image::VisitInfo(getVisitInfo()),  // copy
        getMetadata()->deepCopy()
    );
}


lsst::geom::Point2D OpticalModelDetectorMap::findPointFull(int fiberId, double wavelength) const {
    lsst::geom::Point2D const slit = _slitModel.spectrographToSlit(fiberId, wavelength);
    lsst::geom::Point2D const detector = _opticsModel.slitToDetector(slit);
    lsst::geom::Point2D const pixels = _detectorModel.detectorToPixels(detector);
    if (!lsst::geom::Box2D(getBBox()).contains(pixels)) {
        return lsst::geom::Point2D(NOT_A_NUMBER, NOT_A_NUMBER);
    }
    return pixels;
}


lsst::geom::PointD OpticalModelDetectorMap::findPointImpl(int fiberId, double wavelength) const {
    double y;
    try {
        y = getRowSpline(fiberId)(wavelength);
    } catch (...) {
        return lsst::geom::Point2D(NOT_A_NUMBER, NOT_A_NUMBER);
    }
    if (!std::isfinite(y)) {
        return lsst::geom::Point2D(NOT_A_NUMBER, NOT_A_NUMBER);
    }
    double const x = getXCenterImpl(fiberId, y);
    return lsst::geom::Point2D(x, y);
}


double OpticalModelDetectorMap::findWavelengthImpl(int fiberId, double row) const {
    try {
        return getWavelengthSpline(fiberId)(row);
    } catch (...) {
        return NOT_A_NUMBER;
    }
}


double OpticalModelDetectorMap::getXCenterImpl(int fiberId, double row) const {
    lsst::geom::Point2D detector;
    try {
        detector = lsst::geom::Point2D(getXDetectorSpline(fiberId)(row), getYDetectorSpline(fiberId)(row));
    } catch (...) {
        return NOT_A_NUMBER;
    }
    lsst::geom::Point2D const pixels = _detectorModel.detectorToPixels(detector);
    if (!std::isfinite(pixels.getX()) || !std::isfinite(pixels.getY())) {
        return NOT_A_NUMBER;
    }
    if (!lsst::geom::Box2D(getBBox()).contains(pixels)) {
        return NOT_A_NUMBER;
    }
    return pixels.getX();
}


std::pair<int, ndarray::Array<double, 1, 1>> OpticalModelDetectorMap::getTracePositionImpl(
    int fiberId,
    int row,
    int halfWidth
) const {
    double const xDetector = getXDetectorSpline(fiberId)(row);
    return _detectorModel.detectorToPixelsColumns(lsst::geom::Point2D(xDetector, row), halfWidth);
}


double OpticalModelDetectorMap::calculate(
    int fiberId, Coordinate coordFrom, Coordinate coordTo, double value
) const {
    try {
        return getSpline(fiberId, coordFrom, coordTo)(value);
    } catch (...) {
        return NOT_A_NUMBER;
    }
}


OpticalModelDetectorMap::Array1D OpticalModelDetectorMap::calculate(
    FiberIds const& fiberId, Coordinate coordFrom, Coordinate coordTo, Array1D const& value
) const {
    Array1D result = ndarray::allocate(value.size());
    for (std::size_t ii = 0; ii < value.size(); ++ii) {
        result[ii] = calculate(fiberId[ii], coordFrom, coordTo, value[ii]);
    }
    return result;
}


OpticalModelDetectorMap::Data OpticalModelDetectorMap::makeData(int fiberId) const {
    // we drop the 2nd, 3rd, 3rd-last and 2nd-last points to avoid edge effects
    std::size_t const numKnots = _numKnots - 4;
    ndarray::Array<double, 1, 1> wavelength = ndarray::allocate(numKnots);

    double const minWavelength = _wavelengthRange.first;
    double const maxWavelength = _wavelengthRange.second;
    wavelength[0] = minWavelength;
    wavelength[numKnots - 1] = maxWavelength;
    double const step = (maxWavelength - minWavelength) / (_numKnots - 1);
    for (std::size_t ii = 1, jj = 3; ii < numKnots - 1; ++ii, ++jj) {
        wavelength[ii] = minWavelength + jj*step;
    }

    ndarray::Array<double, 2, 2> slit = ndarray::allocate(2, numKnots);
    ndarray::Array<double, 2, 2> detector = ndarray::allocate(2, numKnots);
    ndarray::Array<double, 2, 2> pixels = ndarray::allocate(2, numKnots);

    ndarray::Array<double, 1, 1> xDetector = ndarray::allocate(numKnots);
    ndarray::Array<double, 1, 1> yDetector = ndarray::allocate(numKnots);
    ndarray::Array<double, 1, 1> row = ndarray::allocate(numKnots);
    for (std::size_t ii = 0; ii < numKnots; ++ii) {
        lsst::geom::Point2D const slitCoord = _slitModel.spectrographToSlit(fiberId, wavelength[ii]);
        lsst::geom::Point2D const detectorCoord = _opticsModel.slitToDetector(slitCoord);
        lsst::geom::Point2D const pixelsCoord = _detectorModel.detectorToPixels(detectorCoord);
        slit[0][ii] = slitCoord.getX();
        slit[1][ii] = slitCoord.getY();
        detector[0][ii] = detectorCoord.getX();
        detector[1][ii] = detectorCoord.getY();
        pixels[0][ii] = pixelsCoord.getX();
        pixels[1][ii] = pixelsCoord.getY();
    }
    return Data(wavelength, slit, detector, pixels);
}


void OpticalModelDetectorMap::_resetSlitOffsets() {
    _data.flush();
    _splines.flush();
    DetectorMap::_resetSlitOffsets();
    _slitModel = std::move(SlitModel(
        getFiberId(),
        _slitModel.getFiberPitch(),
        _slitModel.getWavelengthDispersion(),
        getSpatialOffsets(),
        getSpectralOffsets(),
        _slitModel.getDistortions()
    ));
}


namespace {


// Singleton class that manages the persistence catalog's schema and keys
class OpticalModelDetectorMapSchema {
    using IntArray = lsst::afw::table::Array<int>;
    using DoubleArray = lsst::afw::table::Array<double>;
  public:
    lsst::afw::table::Schema schema;
    lsst::afw::table::Key<IntArray> fiberId;
    lsst::afw::table::Key<double> fiberPitch;
    lsst::afw::table::Key<double> wavelengthDispersion;
    lsst::afw::table::Key<DoubleArray> spatialOffsets;
    lsst::afw::table::Key<DoubleArray> spectralOffsets;
    lsst::afw::table::Key<IntArray> slitDistortions;
    lsst::afw::table::Point2IKey opticsSize;
    lsst::afw::table::Key<DoubleArray> opticsSpatial;
    lsst::afw::table::Key<DoubleArray> opticsSpectral;
    lsst::afw::table::Key<DoubleArray> opticsX;
    lsst::afw::table::Key<DoubleArray> opticsY;
    lsst::afw::table::Key<IntArray> opticsDistortions;
    lsst::afw::table::Box2IKey bbox;
    lsst::afw::table::Key<int> dividedDetector;
    lsst::afw::table::Key<DoubleArray> rightCcd;
    lsst::afw::table::Key<IntArray> detectorDistortions;
    lsst::afw::table::Key<int> visitInfo;

    static OpticalModelDetectorMapSchema const &get() {
        static OpticalModelDetectorMapSchema const instance;
        return instance;
    }

  private:
    OpticalModelDetectorMapSchema()
      : schema(),
        fiberId(schema.addField<IntArray>("fiberId", "fiber identifier", "")),
        fiberPitch(schema.addField<double>("fiberPitch", "fiber pitch", "pixels")),
        wavelengthDispersion(schema.addField<double>(
            "wavelengthDispersion", "wavelength dispersion", "nm/pixel"
        )),
        spatialOffsets(schema.addField<DoubleArray>("spatialOffsets", "spatial offsets", "pixels")),
        spectralOffsets(schema.addField<DoubleArray>("spectralOffsets", "spectral offsets", "pixels")),
        slitDistortions(schema.addField<IntArray>("slitDistortions", "reference to slit distortions", "")),
        opticsSize(lsst::afw::table::Point2IKey::addFields(schema, "opticsSize", "size of optics grid", "")),
        opticsSpatial(schema.addField<DoubleArray>("opticsSpatial", "optics spatial coordinates", "")),
        opticsSpectral(schema.addField<DoubleArray>("opticsSpectral", "optics spectral coordinates", "")),
        opticsX(schema.addField<DoubleArray>("opticsX", "optics x coordinates", "")),
        opticsY(schema.addField<DoubleArray>("opticsY", "optics y coordinates", "")),
        opticsDistortions(schema.addField<IntArray>(
            "opticsDistortions", "reference to optics distortions", ""
        )),
        bbox(bbox.addFields(schema, "bbox", "bounding box", "")),
        dividedDetector(schema.addField<int>("dividedDetector", "divided detector", "")),
        rightCcd(schema.addField<DoubleArray>("rightCcd", "upper CCD transform", "")),
        detectorDistortions(schema.addField<IntArray>(
            "detectorDistortions", "reference to detector distortions", ""
        )),
        visitInfo(schema.addField<int>("visitInfo", "visitInfo reference", ""))
        {}
};

}  // anonymous namespace


void OpticalModelDetectorMap::write(
    lsst::afw::table::io::OutputArchiveHandle & handle
) const {
    OpticalModelDetectorMapSchema const &schema = OpticalModelDetectorMapSchema::get();
    lsst::afw::table::BaseCatalog cat = handle.makeCatalog(schema.schema);
    std::shared_ptr<lsst::afw::table::BaseRecord> record = cat.addNew();

    // Slit model
    ndarray::Array<int, 1, 1> fiberId = ndarray::copy(getFiberId());
    record->set(schema.fiberId, fiberId);
    record->set(schema.fiberPitch, _slitModel.getFiberPitch());
    record->set(schema.wavelengthDispersion, _slitModel.getWavelengthDispersion());
    ndarray::Array<double, 1, 1> spatialOffsets = ndarray::copy(getSpatialOffsets());
    record->set(schema.spatialOffsets, spatialOffsets);
    ndarray::Array<double, 1, 1> spectralOffsets = ndarray::copy(getSpectralOffsets());
    record->set(schema.spectralOffsets, spectralOffsets);
    ndarray::Array<int, 1, 1> slitDistortions = ndarray::allocate(_slitModel.getDistortions().size());
    for (std::size_t ii = 0; ii < _slitModel.getDistortions().size(); ++ii) {
        slitDistortions[ii] = handle.put(*_slitModel.getDistortions()[ii]);
    }
    record->set(schema.slitDistortions, slitDistortions);

    // Optics model
    auto const shape = _opticsModel.getSpatial().getShape();
    schema.opticsSize.set(*record, lsst::geom::Point2I(shape[0], shape[1]));
    Array1D opticsSpatial = utils::flattenArray(_opticsModel.getSpatial());
    Array1D opticsSpectral = utils::flattenArray(_opticsModel.getSpectral());
    Array1D opticsX = utils::flattenArray(_opticsModel.getX());
    Array1D opticsY = utils::flattenArray(_opticsModel.getY());
    record->set(schema.opticsSpatial, opticsSpatial);
    record->set(schema.opticsSpectral, opticsSpectral);
    record->set(schema.opticsX, opticsX);
    record->set(schema.opticsY, opticsY);
    ndarray::Array<int, 1, 1> opticsDistortions = ndarray::allocate(_opticsModel.getDistortions().size());
    for (std::size_t ii = 0; ii < _opticsModel.getDistortions().size(); ++ii) {
        opticsDistortions[ii] = handle.put(*_opticsModel.getDistortions()[ii]);
    }
    record->set(schema.opticsDistortions, opticsDistortions);

    // Detector model
    schema.bbox.set(*record, getBBox());
    record->set(schema.dividedDetector, _detectorModel.getIsDivided());
    Array1D rightCcd = math::getAffineParameters(_detectorModel.getRightCcd());
    record->set(schema.rightCcd, rightCcd);
    ndarray::Array<int, 1, 1> detectorDistortions = ndarray::allocate(_detectorModel.getDistortions().size());
    for (std::size_t ii = 0; ii < _detectorModel.getDistortions().size(); ++ii) {
        detectorDistortions[ii] = handle.put(*_detectorModel.getDistortions()[ii]);
    }
    record->set(schema.detectorDistortions, detectorDistortions);

    record->set(schema.visitInfo, handle.put(getVisitInfo()));
    // XXX dropping metadata on the floor, since we can't write a header
    handle.saveCatalog(cat);
}


class OpticalModelDetectorMap::Factory : public lsst::afw::table::io::PersistableFactory {
  public:
    std::shared_ptr<lsst::afw::table::io::Persistable> read(
        lsst::afw::table::io::InputArchive const& archive,
        lsst::afw::table::io::CatalogVector const& catalogs
    ) const override {
        static auto const& schema = OpticalModelDetectorMapSchema::get();
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        lsst::afw::table::BaseRecord const& record = catalogs.front().front();
        LSST_ARCHIVE_ASSERT(record.getSchema() == schema.schema);

        lsst::geom::Box2I bbox = schema.bbox.get(record);

        // Slit model
        ndarray::Array<int, 1, 1> fiberId = ndarray::copy(record.get(schema.fiberId));
        double fiberPitch = record.get(schema.fiberPitch);
        double wavelengthDispersion = record.get(schema.wavelengthDispersion);
        ndarray::Array<double, 1, 1> spatialOffsets = ndarray::copy(record.get(schema.spatialOffsets));
        ndarray::Array<double, 1, 1> spectralOffsets = ndarray::copy(record.get(schema.spectralOffsets));
        ndarray::Array<int const, 1, 1> slitDistortionPtrs = record.get(schema.slitDistortions);
        std::size_t const numSlitDistortions = slitDistortionPtrs.size();
        SlitModel::DistortionList slitDistortions;
        slitDistortions.reserve(numSlitDistortions);
        for (std::size_t ii = 0; ii < numSlitDistortions; ++ii) {
            slitDistortions.emplace_back(archive.get<Distortion>(slitDistortionPtrs[ii]));
        }
        SlitModel slitModel(
            fiberId, fiberPitch, wavelengthDispersion, spatialOffsets, spectralOffsets, slitDistortions
        );

        // Optics model
        lsst::geom::Point2I opticsSize = schema.opticsSize.get(record);
        ndarray::Array<double, 1, 1> opticsSpatial = ndarray::copy(record.get(schema.opticsSpatial));
        ndarray::Array<double, 1, 1> opticsSpectral = ndarray::copy(record.get(schema.opticsSpectral));
        ndarray::Array<double, 1, 1> opticsX = ndarray::copy(record.get(schema.opticsX));
        ndarray::Array<double, 1, 1> opticsY = ndarray::copy(record.get(schema.opticsY));
        ndarray::Array<int const, 1, 1> opticsDistortionPtrs = record.get(schema.opticsDistortions);
        std::size_t const numOpticsDistortions = opticsDistortionPtrs.size();
        OpticsModel::DistortionList opticsDistortions;
        opticsDistortions.reserve(numOpticsDistortions);
        for (std::size_t ii = 0; ii < numOpticsDistortions; ++ii) {
            opticsDistortions.emplace_back(archive.get<Distortion>(opticsDistortionPtrs[ii]));
        }
        OpticsModel opticsModel(
            utils::unflattenArray(opticsSpatial, opticsSize.getX(), opticsSize.getY()),
            utils::unflattenArray(opticsSpectral, opticsSize.getX(), opticsSize.getY()),
            utils::unflattenArray(opticsX, opticsSize.getX(), opticsSize.getY()),
            utils::unflattenArray(opticsY, opticsSize.getX(), opticsSize.getY()),
            opticsDistortions
        );

        // Detector model
        bool dividedDetector = record.get(schema.dividedDetector);
        lsst::geom::AffineTransform rightCcd = math::makeAffineTransform(record.get(schema.rightCcd));
        ndarray::Array<int const, 1, 1> detectorDistortionPtrs = record.get(schema.detectorDistortions);
        std::size_t const numDetectorDistortions = detectorDistortionPtrs.size();
        DetectorModel::DistortionList detectorDistortions;
        detectorDistortions.reserve(numDetectorDistortions);
        for (std::size_t ii = 0; ii < numDetectorDistortions; ++ii) {
            detectorDistortions.emplace_back(archive.get<Distortion>(detectorDistortionPtrs[ii]));
        }
        DetectorModel detectorModel = dividedDetector
            ? DetectorModel(bbox, rightCcd, detectorDistortions)
            : DetectorModel(bbox, detectorDistortions);

        auto visitInfo = archive.get<lsst::afw::image::VisitInfo>(record.get(schema.visitInfo));
        // dropping metadata on the floor, since we can't write a header
        return std::make_shared<OpticalModelDetectorMap>(
            slitModel, opticsModel, detectorModel, *visitInfo, nullptr
        );
    }

    Factory(std::string const& name) : lsst::afw::table::io::PersistableFactory(name) {}
};


namespace {

OpticalModelDetectorMap::Factory registration("OpticalModelDetectorMap");

}  // anonymous namespace


}}}  // namespace pfs::drp::stella
