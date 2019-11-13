#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"

#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/Persistable.cc"
#include "lsst/afw/table.h"

#include "pfs/drp/stella/SpectralPsf.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {

// Singleton class that manages the persistence catalog's schema and keys
class GaussianOversampledPsfPersistenceHelper {
  public:
    lsst::afw::table::Schema schema;
    lsst::afw::table::Key<float> sigma;
    lsst::afw::table::Key<int> oversampleFactor;
    lsst::afw::table::PointKey<int> targetSize;

    static GaussianOversampledPsfPersistenceHelper const &get() {
        static GaussianOversampledPsfPersistenceHelper const instance;
        return instance;
    }

  private:
    GaussianOversampledPsfPersistenceHelper()
      : schema(),
        sigma(schema.addField<float>("sigma", "Gaussian sigma")),
        oversampleFactor(schema.addField<int>("oversampleFactor", "factor by which the PSF is oversampled")),
        targetSize(lsst::afw::table::PointKey<int>::addFields(schema, "targetSize", "size of PSF image",
                                                              "pixel")) {
            schema.getCitizen().markPersistent();
    }
};

}  // anonymous namespace


// An oversampled Gaussian, for testing the OversampledPsf class
class GaussianOversampledPsf :
    public lsst::afw::table::io::PersistableFacade<GaussianOversampledPsf>,
    public OversampledPsf {
  public:
    // Ctor
    //
    // @param sigma : Gaussian sigma (native pixels)
    // @param oversampleFactor : integer factor by which the PSF is oversampled.
    // @param targetSize : desired size of the PSF kernel image after resampling.
    GaussianOversampledPsf(float sigma, int oversampleFactor, lsst::geom::Extent2I const& targetSize) :
        OversampledPsf(oversampleFactor, targetSize),
        _sigma(sigma)
        {}

    virtual ~GaussianOversampledPsf() = default;
    GaussianOversampledPsf(GaussianOversampledPsf const&) = default;
    GaussianOversampledPsf(GaussianOversampledPsf&&) = default;
    GaussianOversampledPsf& operator=(GaussianOversampledPsf const&) = delete;
    GaussianOversampledPsf& operator=(GaussianOversampledPsf&&) = delete;

    double getSigma() const { return _sigma; }

    virtual std::shared_ptr<Psf> clone() const override {
        return std::make_shared<GaussianOversampledPsf>(_sigma, getOversampleFactor(), getTargetSize());
    }
    virtual std::shared_ptr<lsst::afw::detection::Psf> resized(int width, int height) const override {
        return std::make_shared<GaussianOversampledPsf>(_sigma, getOversampleFactor(),
                                                        lsst::geom::Extent2I(width, height));
    }

    bool isPersistable() const noexcept override { return true; }

    class Factory;

  protected:
    std::string getPersistenceName() const override { return "GaussianOversampledPsf"; }

    std::string getPythonModule() const override { return "pfs.drp.stella.tests"; }

    void write(lsst::afw::table::io::OutputArchiveHandle & handle) const override {
        GaussianOversampledPsfPersistenceHelper const &keys = GaussianOversampledPsfPersistenceHelper::get();
        lsst::afw::table::BaseCatalog cat = handle.makeCatalog(keys.schema);
        PTR(lsst::afw::table::BaseRecord) record = cat.addNew();
        record->set(keys.sigma, _sigma);
        record->set(keys.oversampleFactor, getOversampleFactor());
        record->set(keys.targetSize, lsst::geom::Point2I(getTargetSize()));
        handle.saveCatalog(cat);
    }

  private:
    // Return an oversampled image of the PSF, with the center at 0,0
    virtual std::shared_ptr<Image> doComputeOversampledKernelImage(
        lsst::geom::Point2D const& position
    ) const {
        auto const targetBox = computeBBox(position);
        int const oversampleFactor = getOversampleFactor();
        lsst::geom::Box2I bbox{lsst::geom::Point2I(-targetBox.getWidth()*oversampleFactor/2,
                                                   -targetBox.getHeight()*oversampleFactor/2),
                               lsst::geom::Extent2I(targetBox.getWidth()*oversampleFactor,
                                                    targetBox.getHeight()*oversampleFactor)};
        auto kernel = std::make_shared<OversampledPsf::Image>(bbox);
        double const factor = -0.5/std::pow(oversampleFactor*_sigma, 2);
        for (std::ptrdiff_t jj = 0, yy = bbox.getMinY(); jj < bbox.getHeight(); ++jj, ++yy) {
            for (std::ptrdiff_t ii = 0, xx = bbox.getMinX(); ii < bbox.getWidth(); ++ii, ++xx) {
                double const value = std::exp(factor*(xx*xx + yy*yy));
                kernel->get(lsst::geom::Point2I(ii, jj), lsst::afw::image::LOCAL) = value;
            }
        }
        return kernel;
    }

    float const _sigma;  // Gaussian sigma (native pixels)
};


class GaussianOversampledPsf::Factory : public lsst::afw::table::io::PersistableFactory {
  public:
    std::shared_ptr<lsst::afw::table::io::Persistable> read(
        lsst::afw::table::io::InputArchive const& archive,
        lsst::afw::table::io::CatalogVector const& catalogs
    ) const override {
        static auto const& keys = GaussianOversampledPsfPersistenceHelper::get();
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        lsst::afw::table::BaseRecord const& record = catalogs.front().front();
        LSST_ARCHIVE_ASSERT(record.getSchema() == keys.schema);
        float const sigma = record.get(keys.sigma);
        int const oversampleFactor = record.get(keys.oversampleFactor);
        lsst::geom::Extent2I const targetSize{record.get(keys.targetSize)};
        return std::make_shared<GaussianOversampledPsf>(sigma, oversampleFactor, targetSize);
    }

    Factory(std::string const& name) : lsst::afw::table::io::PersistableFactory(name) {}
};

GaussianOversampledPsf::Factory registration("GaussianOversampledPsf");


namespace {

PYBIND11_PLUGIN(oversampledPsf) {
    py::module mod("oversampledPsf");
    pybind11::module::import("pfs.drp.stella.SpectralPsf");

    py::class_<GaussianOversampledPsf, std::shared_ptr<GaussianOversampledPsf>,
               OversampledPsf, lsst::afw::detection::Psf> cls(mod, "GaussianOversampledPsf");
    cls.def(py::init<float, int, lsst::geom::Extent2I>(), "sigma"_a, "oversampleFactor"_a, "targetSize"_a);
    cls.def("getSigma", &GaussianOversampledPsf::getSigma);

    return mod.ptr();
}


} // anonymous namespace

}}} // pfs::drp::stella
