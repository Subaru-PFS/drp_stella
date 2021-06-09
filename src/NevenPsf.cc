#include <numeric>
#include "boost/format.hpp"

#include "lsst/afw/table.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/Persistable.cc"
#include "lsst/afw/math/offsetImage.h"
#include "lsst/afw/fits.h"

#include "pfs/drp/stella/utils/checkSize.h"
#include "pfs/drp/stella/utils/math.h"
#include "pfs/drp/stella/NevenPsf.h"


namespace pfs {
namespace drp {
namespace stella {


NevenPsf::NevenPsf(
    std::shared_ptr<DetectorMap> detMap,
    ndarray::Array<int, 1, 1> const& fiberId,
    ndarray::Array<double, 1, 1> const& wavelength,
    std::vector<ndarray::Array<float, 2, 1>> const& images,
    int oversampleFactor,
    lsst::geom::Extent2I const& targetSize
) : SpectralPsf(detMap),
    OversampledPsf(oversampleFactor, targetSize) {
        std::size_t const num = fiberId.size();
        utils::checkSize(wavelength.size(), num, "wavelength");
        utils::checkSize(images.size(), num, "images");
        if (num == 0) {
            throw LSST_EXCEPT(lsst::pex::exceptions::LengthError, "No images provided");
        }

        // Check shape of each image, and determine number of images for each fiberId
        auto const shape = images[0].getShape();
        std::map<int, std::size_t> lengths;
        for (std::size_t ii = 0; ii < num; ++ii) {
            if (images[ii].getShape() != shape) {
                throw LSST_EXCEPT(lsst::pex::exceptions::LengthError,
                                  (boost::format("Shape mismatch for image %d: %s vs %s") %
                                   ii % images[ii].getShape() % shape).str());
            }
            int const ff = fiberId[ii];
            ++lengths[ff];
        }

        // Allocate space for each fiberId
        for (auto ff : fiberId) {
            _data[ff] = DataArray(lengths[ff]);
        }

        // Sorting the entire y array means the y values for the individual fibers are sorted too,
        // so we won't have to sort during lookup.
        auto const sorted = utils::argsort(wavelength);

        // Insert the y values and images
        std::map<int, std::size_t> indices;  // current insertion index for each fiber
        for (std::size_t ii = 0; ii < num; ++ii) {
            std::size_t const index = sorted[ii];
            int const ff = fiberId[index];
            std::size_t const jj = indices[ff]++;
            double const wl = wavelength[index];
            double const yy = detMap->findPoint(ff, wl).getY();
            _data[ff][jj] = Data{wl, yy, ndarray::copy(images[index])};
        }
    }


std::shared_ptr<OversampledPsf::Image> NevenPsf::doComputeOversampledKernelImage(
    lsst::geom::Point2D const& position
) const {
    // We've probably just converted fiberId,wavelength to x,y, and now we want to go back again.
    // We might be able to address this inefficiency with a better design, but we'll bear with it until it
    // becomes clear it's adversely affecting performance.
    int const fiberId = _detMap->findFiberId(position);
    double const yy = position.getY();
    double const wavelength = _detMap->getWavelength(fiberId, yy);

    auto const iter = _data.find(fiberId);
    if (iter == _data.end()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundError,
                          (boost::format("No PSF data for fiberId=%d") % fiberId).str());
    }
    auto const& data = iter->second;
    auto const& above = std::lower_bound(
        data.begin(), data.end(), wavelength,
        [](auto const& elem, double value) { return elem.wavelength < value; }
    );

    std::shared_ptr<OversampledPsf::Image> out;
    if (above == data.end()) {
        // There is no wavelength value above the desired position
        out = std::make_shared<OversampledPsf::Image>(utils::convertArray<double>(data.back().image));
    } else if (above->wavelength == wavelength || above == data.begin()) {
        // Found an exact wavelength match, or there is no wavelength value below the desired position
        out = std::make_shared<OversampledPsf::Image>(utils::convertArray<double>(above->image));
    } else {
        auto const& below = std::prev(above);

        double const y1 = below->y;
        double const y2 = above->y;
        double const distance1 = yy - y1;
        double const distance2 = yy - y2;

        auto const& image1 = below->image;
        auto const& image2 = above->image;
        auto const shape = image1.getShape();
        assert(image2.getShape() == shape);

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


std::size_t NevenPsf::size() const {
    return std::accumulate(_data.begin(), _data.end(), 0UL,
                           [](std::size_t size, auto const& dd) { return size + dd.second.size(); });
}


ndarray::Array<int, 1, 1> NevenPsf::getFiberId() const {
    ndarray::Array<int, 1, 1> fiberId = ndarray::allocate(size());
    std::size_t start = 0;
    for (auto const& dd : _data) {
        std::size_t const stop = start + dd.second.size();
        fiberId[ndarray::view(start, stop)] = dd.first;
        start = stop;
    }
    return fiberId;
}


ndarray::Array<double, 1, 1> NevenPsf::getWavelength() const {
    ndarray::Array<double, 1, 1> yy = ndarray::allocate(size());
    std::size_t ii = 0;
    for (auto const& dd : _data) {
        for (auto const& elem : dd.second) {
            yy[ii++] = elem.wavelength;
        }
    }
    return yy;
}


std::vector<ndarray::Array<float, 2, 1>> NevenPsf::getImages() const {
    std::vector<ndarray::Array<float, 2, 1>> images;
    images.reserve(size());
    for (auto const& dd : _data) {
        for (auto const& elem : dd.second) {
            images.emplace_back(ndarray::copy(elem.image));
        }
    }
    return images;
}


std::shared_ptr<lsst::afw::detection::Psf>
NevenPsf::clone() const {
    return std::make_shared<NevenPsf>(getDetectorMap(), getFiberId(), getWavelength(), getImages(),
                                      getOversampleFactor(), getTargetSize());
}


std::shared_ptr<lsst::afw::detection::Psf>
NevenPsf::resized(int width, int height) const {
    return std::make_shared<NevenPsf>(getDetectorMap(), getFiberId(), getWavelength(), getImages(),
                                      getOversampleFactor(), lsst::geom::Extent2I(width, height));
}


class PsfImages : public lsst::afw::table::io::Persistable {
    using Images = std::vector<ndarray::Array<float, 2, 1>>;
  public:
    explicit PsfImages(Images const& images) : _images(images) {}

    Images getImages() const { return _images; }

    bool isPersistable() const noexcept override { return true; }

    class PsfImagesSchema {
        using FloatArray = lsst::afw::table::Array<float>;
      public:
        lsst::afw::table::Schema schema;
        lsst::afw::table::Key<int> width;
        lsst::afw::table::Key<int> height;
        lsst::afw::table::Key<FloatArray> images;

        static PsfImagesSchema const &get() {
            static PsfImagesSchema const instance;
            return instance;
        }

      private:
        PsfImagesSchema()
          : schema(),
            width(schema.addField<int>("width", "PSF image width", "pixel")),
            height(schema.addField<int>("height", "PSF image height", "pixel")),
            images(schema.addField<FloatArray>("images", "PSF images", "count", 0)) {
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
                ndarray::Array<float const, 1, 1> const flat = cat[ii].get(schema.images);
                ndarray::Array<float, 2, 2> img = ndarray::allocate(width, height);
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
            ndarray::Array<float const, 2, 2> const img = lsst::afw::fits::makeContiguousArray(_images[ii]);
            ndarray::Array<float, 1, 1> flat = ndarray::copy(ndarray::flatten<1>(img));
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
    using IntArray = lsst::afw::table::Array<int>;
    using DoubleArray = lsst::afw::table::Array<double>;

  public:
    lsst::afw::table::Schema schema;
    lsst::afw::table::Key<int> detMap;
    lsst::afw::table::Key<IntArray> fiberId;
    lsst::afw::table::Key<DoubleArray> wavelength;
    lsst::afw::table::Key<int> imagesRef;
    lsst::afw::table::Key<int> oversampleFactor;
    lsst::afw::table::PointKey<int> targetSize;

    static NevenPsfSchema const &get() {
        static NevenPsfSchema const instance;
        return instance;
    }

  private:
    NevenPsfSchema()
      : schema(),
        detMap(schema.addField<int>("detectorMap", "detector mapping", "")),
        fiberId(schema.addField<IntArray>("fiberId", "fiberId for PSF image", "", 0)),
        wavelength(schema.addField<DoubleArray>("wavelength", "wavelength for PSF image", "nm", 0)),
        imagesRef(schema.addField<int>("images", "reference to images", "")),
        oversampleFactor(schema.addField<int>("oversampleFactor", "factor by which the PSF is oversampled")),
        targetSize(lsst::afw::table::PointKey<int>::addFields(schema, "targetSize", "size of PSF image",
                                                              "pixel")) {
            schema.getCitizen().markPersistent();
        }
};

}  // anonymous namespace


void NevenPsf::write(lsst::afw::table::io::OutputArchiveHandle & handle) const {
    NevenPsfSchema const &schema = NevenPsfSchema::get();
    lsst::afw::table::BaseCatalog cat = handle.makeCatalog(schema.schema);
    PTR(lsst::afw::table::BaseRecord) record = cat.addNew();
    record->set(schema.detMap, handle.put(getDetectorMap()));
    record->set(schema.fiberId, getFiberId());
    record->set(schema.wavelength, getWavelength());
    record->set(schema.imagesRef, handle.put(PsfImages(getImages())));
    record->set(schema.oversampleFactor, getOversampleFactor());
    record->set(schema.targetSize, lsst::geom::Point2I(getTargetSize()));
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
        ndarray::Array<int, 1, 1> const fiberId = ndarray::copy(record.get(schema.fiberId));
        ndarray::Array<double, 1, 1> const yy = ndarray::copy(record.get(schema.wavelength));
        std::shared_ptr<PsfImages> images = archive.get<PsfImages>(record.get(schema.imagesRef));
        int const oversampleFactor = record.get(schema.oversampleFactor);
        lsst::geom::Extent2I const targetSize{record.get(schema.targetSize)};
        return std::make_shared<NevenPsf>(detMap, fiberId, yy, images->getImages(),
                                          oversampleFactor, targetSize);
    }

    Factory(std::string const& name) : lsst::afw::table::io::PersistableFactory(name) {}
};


NevenPsf::Factory nevenPsfFactory("NevenPsf");


}}}  // namespace pfs::drp::stella
