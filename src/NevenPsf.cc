#include "lsst/afw/table.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/Persistable.cc"
#include "lsst/afw/math/offsetImage.h"
#include "lsst/afw/fits.h"

#include "pfs/drp/stella/NevenPsf.h"


namespace pfs {
namespace drp {
namespace stella {


NevenPsf::NevenPsf(
    std::shared_ptr<DetectorMap> detMap,
    ndarray::Array<float const, 1, 1> const& xx,
    ndarray::Array<float const, 1, 1> const& yy,
    std::vector<ndarray::Array<double const, 2, 1>> const& images,
    int oversampleFactor,
    lsst::geom::Extent2I const& targetSize,
    float xMaxDistance
) : SpectralPsf(detMap),
    OversampledPsf(oversampleFactor, targetSize),
    _xx(xx),
    _yy(yy),
    _images(images),
    _xMaxDistance(xMaxDistance) {
        std::size_t num = images.size();
        if (_xx.getNumElements() != num || _yy.getNumElements() != num) {
            throw LSST_EXCEPT(lsst::pex::exceptions::LengthError,
                              (boost::format("Length mismatch: %ld,%ld,%ld") %
                               _xx.getNumElements() % _yy.getNumElements() % num).str());
        }
        if (num == 0) {
            throw LSST_EXCEPT(lsst::pex::exceptions::LengthError, "No images provided");
        }
        auto const shape = _images[0].getShape();
        for (std::size_t ii = 1; ii < num; ++ii) {
            if (_images[ii].getShape() != shape) {
                throw LSST_EXCEPT(lsst::pex::exceptions::LengthError,
                                  (boost::format("Shape mismatch for image %d: %s vs %s") %
                                   ii % _images[ii].getShape() % shape).str());
            }
        }
    }


std::shared_ptr<OversampledPsf::Image> NevenPsf::doComputeOversampledKernelImage(
    lsst::geom::Point2D const& position
) const {
    float const xPosition = position.getX();
    float const yPosition = position.getY();
    std::size_t const num = _images.size();

    // Ideally, we'd select by fiberId and then y or wavelength would already
    // be sorted so we wouldn't have to sort. But this is how the algorithm was
    // specified, and at the moment performance isn't critical.
    std::vector<std::pair<std::size_t, float>> candidates;
    candidates.reserve(num);
    for (std::size_t ii = 0; ii < num; ++ii) {
        if (std::abs(_xx[ii] - xPosition) < _xMaxDistance) {
            candidates.emplace_back(ii, _yy[ii] - yPosition);
        }
    }
    if (candidates.size() < 2) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          (boost::format("Unable to find 2 images within %f of x=%f") %
                           _xMaxDistance % xPosition).str());
    }

    std::sort(candidates.begin(), candidates.end(),
              [](auto const& a, auto const& b) { return a.second < b.second; });
    auto const& above = std::lower_bound(
        candidates.begin(), candidates.end(), 0.0,
        [](auto const& elem, float const value) { return elem.second < value; });

    std::shared_ptr<OversampledPsf::Image> out;
    if (above == candidates.end()) {
        // There is no y value above the desired position
        out = std::make_shared<OversampledPsf::Image>(ndarray::copy(_images[candidates.back().first]));
    } else if (above->second == 0.0 || above == candidates.begin()) {
        // Found an exact y value match, or there is no y value below the desired position
        out = std::make_shared<OversampledPsf::Image>(ndarray::copy(_images[above->first]));
    } else {
        auto const& below = std::prev(above);
        std::size_t const index1 = below->first;
        std::size_t const index2 = above->first;
        float const distance1 = below->second;
        float const distance2 = above->second;
        auto const& image1 = _images[index1];
        auto const& image2 = _images[index2];
        auto const shape = _images[index1].getShape();
        assert(_images[index2].getShape() == shape);

        out = std::make_shared<OversampledPsf::Image>(shape[0], shape[1]);
        auto array = out->getArray();
        assert(array.getShape() == shape);

        double const ratio = distance2/distance1;
        for (std::size_t y = 0; y < shape[1]; ++y) {
            for (std::size_t x = 0; x < shape[0]; ++x) {
                array[y][x] = (image2[y][x] - image1[y][x]*ratio)/(1.0 - ratio);
            }
        }

    }
    out->setXY0(-out->getWidth()/2, -out->getHeight()/2);

    return out;
}


std::shared_ptr<lsst::afw::detection::Psf>
NevenPsf::clone() const {
    return std::make_shared<NevenPsf>(getDetectorMap(), _xx, _yy, _images, getOversampleFactor(),
                                      getTargetSize(), _xMaxDistance);
}


std::shared_ptr<lsst::afw::detection::Psf>
NevenPsf::resized(int width, int height) const {
    return std::make_shared<NevenPsf>(getDetectorMap(), _xx, _yy, _images, getOversampleFactor(),
                                      lsst::geom::Extent2I(width, height), _xMaxDistance);
}


class PsfImages : public lsst::afw::table::io::Persistable {
    using Images = std::vector<ndarray::Array<double const, 2, 1>>;
  public:
    explicit PsfImages(Images const& images) : _images(images) {}

    Images getImages() const { return _images; }

    bool isPersistable() const noexcept override { return true; }

    class PsfImagesSchema {
        using DoubleArray = lsst::afw::table::Array<double>;
      public:
        lsst::afw::table::Schema schema;
        lsst::afw::table::Key<int> width;
        lsst::afw::table::Key<int> height;
        lsst::afw::table::Key<DoubleArray> images;

        static PsfImagesSchema const &get() {
            static PsfImagesSchema const instance;
            return instance;
        }

      private:
        PsfImagesSchema()
          : schema(),
            width(schema.addField<int>("width", "PSF image width", "pixel")),
            height(schema.addField<int>("height", "PSF image height", "pixel")),
            images(schema.addField<DoubleArray>("images", "PSF images", "count", 0)) {
                schema.getCitizen().markPersistent();
            }
    };

    class Factory : public lsst::afw::table::io::PersistableFactory {
      public:
        Factory(std::string const& name) : lsst::afw::table::io::PersistableFactory(name) {}

        std::shared_ptr<lsst::afw::table::io::Persistable> read(
            lsst::afw::table::io::InputArchive const& archive,
            lsst::afw::table::io::CatalogVector const& catalogs
        ) const override {
            static auto const& schema = PsfImagesSchema::get();
            LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
            lsst::afw::table::BaseCatalog const& cat = catalogs.front();
            LSST_ARCHIVE_ASSERT(cat.getSchema() == schema.schema);
            std::size_t numImages = cat.size();
            Images images;
            images.reserve(numImages);
            for (std::size_t ii = 0; ii < numImages; ++ii) {
                int const width = cat[ii].get(schema.width);
                int const height = cat[ii].get(schema.height);
                ndarray::Array<double const, 1, 1> const flat = cat[ii].get(schema.images);
                ndarray::Array<double, 2, 2> img = ndarray::allocate(width, height);
                ndarray::flatten<1>(img) = flat;
                images.emplace_back(ndarray::static_dimension_cast<1>(img));
            }
            return std::make_shared<PsfImages>(images);
        }
    };

  protected:
    std::string getPersistenceName() const override { return "NevenPsfImages"; }

    std::string getPythonModule() const override { return "pfs.drp.stella"; }

    void write(lsst::afw::table::io::OutputArchiveHandle & handle) const override {
        static auto const& schema = PsfImagesSchema::get();
        lsst::afw::table::BaseCatalog cat = handle.makeCatalog(schema.schema);
        for (std::size_t ii = 0; ii < _images.size(); ++ii) {
            PTR(lsst::afw::table::BaseRecord) record = cat.addNew();
            record->set(schema.width, _images[ii].getShape()[0]);
            record->set(schema.height, _images[ii].getShape()[1]);
            ndarray::Array<double const, 2, 2> const img = lsst::afw::fits::makeContiguousArray(_images[ii]);
            ndarray::Array<double, 1, 1> flat = ndarray::copy(ndarray::flatten<1>(img));
            record->set(schema.images, flat);
        }
        handle.saveCatalog(cat);
    }

  private:
    Images _images;
};


PsfImages::Factory nevenPsfImagesFactory("NevenPsfImages");


namespace {

// Singleton class that manages the persistence catalog's schema and keys
class NevenPsfSchema {
    using FloatArray = lsst::afw::table::Array<float>;
    using DoubleArray = lsst::afw::table::Array<double>;

  public:
    lsst::afw::table::Schema schema;
    lsst::afw::table::Key<int> detMap;
    lsst::afw::table::Key<FloatArray> xx;
    lsst::afw::table::Key<FloatArray> yy;
    lsst::afw::table::Key<int> imagesRef;
    lsst::afw::table::Key<int> oversampleFactor;
    lsst::afw::table::PointKey<int> targetSize;
    lsst::afw::table::Key<float> xMaxDistance;

    static NevenPsfSchema const &get() {
        static NevenPsfSchema const instance;
        return instance;
    }

  private:
    NevenPsfSchema()
      : schema(),
        detMap(schema.addField<int>("detectorMap", "detector mapping", "")),
        xx(schema.addField<FloatArray>("x", "x position for PSF image", "pixel", 0)),
        yy(schema.addField<FloatArray>("y", "y position for PSF image", "pixel", 0)),
        imagesRef(schema.addField<int>("images", "reference to images", "")),
        oversampleFactor(schema.addField<int>("oversampleFactor", "factor by which the PSF is oversampled")),
        targetSize(lsst::afw::table::PointKey<int>::addFields(schema, "targetSize", "size of PSF image",
                                                              "pixel")),
        xMaxDistance(schema.addField<float>("xMaxDistance", "max x distance for image selection", "pixel")) {
            schema.getCitizen().markPersistent();
        }
};

}  // anonymous namespace


void NevenPsf::write(lsst::afw::table::io::OutputArchiveHandle & handle) const {
    NevenPsfSchema const &schema = NevenPsfSchema::get();
    lsst::afw::table::BaseCatalog cat = handle.makeCatalog(schema.schema);
    PTR(lsst::afw::table::BaseRecord) record = cat.addNew();
    record->set(schema.detMap, handle.put(getDetectorMap()));
    ndarray::Array<float, 1, 1> const xx = ndarray::copy(_xx);
    ndarray::Array<float, 1, 1> const yy = ndarray::copy(_yy);
    record->set(schema.xx, xx);
    record->set(schema.yy, yy);
    record->set(schema.imagesRef, handle.put(PsfImages(_images)));
    record->set(schema.oversampleFactor, getOversampleFactor());
    record->set(schema.targetSize, lsst::geom::Point2I(getTargetSize()));
    record->set(schema.xMaxDistance, _xMaxDistance);
    handle.saveCatalog(cat);
}


class NevenPsf::Factory : public lsst::afw::table::io::PersistableFactory {
  public:
    std::shared_ptr<lsst::afw::table::io::Persistable> read(
        lsst::afw::table::io::InputArchive const& archive,
        lsst::afw::table::io::CatalogVector const& catalogs
    ) const override {
        static auto const& schema = NevenPsfSchema::get();
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        lsst::afw::table::BaseRecord const& record = catalogs.front().front();
        LSST_ARCHIVE_ASSERT(record.getSchema() == schema.schema);
        std::shared_ptr<DetectorMap> detMap = archive.get<DetectorMap>(record.get(schema.detMap));
        ndarray::Array<float const, 1, 1> const xx = record.get(schema.xx);
        ndarray::Array<float const, 1, 1> const yy = record.get(schema.yy);
        std::shared_ptr<PsfImages> images = archive.get<PsfImages>(record.get(schema.imagesRef));
        int const oversampleFactor = record.get(schema.oversampleFactor);
        lsst::geom::Extent2I const targetSize{record.get(schema.targetSize)};
        float const xMaxDistance = record.get(schema.xMaxDistance);
        return std::make_shared<NevenPsf>(detMap, xx, yy, images->getImages(), oversampleFactor, targetSize,
                                          xMaxDistance);
    }

    Factory(std::string const& name) : lsst::afw::table::io::PersistableFactory(name) {}
};


NevenPsf::Factory nevenPsfFactory("NevenPsf");


}}}  // namespace pfs::drp::stella
