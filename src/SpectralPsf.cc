#include <algorithm>
#include <numeric>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/table.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/Persistable.cc"
#include "lsst/afw/math/offsetImage.h"
#include "lsst/afw/image/ImageUtils.h"
#include "lsst/afw/geom/ellipses.h"
#include "lsst/afw/geom/SpanSet.h"

#include "pfs/drp/stella/SpectralPsf.h"
#include "pfs/drp/stella/centroidImage.h"

namespace pfs {
namespace drp {
namespace stella {

template <typename T>
std::shared_ptr<lsst::afw::image::Image<T>>
resampleKernelImage(
    lsst::afw::image::Image<T> const& image,
    int binning,
    lsst::geom::Box2I const& bbox,
    lsst::geom::Point2D const& center
) {
    using Image = lsst::afw::image::Image<T>;
    auto const xPosition = lsst::afw::image::positionToIndex(center.getX(), true);
    auto const yPosition = lsst::afw::image::positionToIndex(center.getY(), true);
    lsst::geom::Box2I targetBox = bbox;
    targetBox.shift(lsst::geom::Extent2I(xPosition.first, yPosition.first));
    auto resampled = std::make_shared<Image>(targetBox);
    auto array = resampled->getArray();
    array.deep() = 0.0;

    std::ptrdiff_t const x0 = image.getX0();
    std::ptrdiff_t const y0 = image.getY0();
    std::ptrdiff_t const width = image.getWidth();
    std::ptrdiff_t const height = image.getHeight();

    double sum = 0.0;
    for (std::ptrdiff_t yy = 0; yy < targetBox.getHeight(); ++yy) {
        std::ptrdiff_t jLow = (yy - 0.5 + targetBox.getMinY())*binning - y0 + 1;
        std::ptrdiff_t jHigh = (yy + 0.5 + targetBox.getMinY())*binning - y0 + 1;
        for (std::ptrdiff_t jj = std::max(0L, jLow); jj < std::min(jHigh, height); ++jj) {
            for (std::ptrdiff_t xx = 0; xx < targetBox.getWidth(); ++xx) {
                std::ptrdiff_t iLow = (xx - 0.5 + targetBox.getMinX())*binning - x0 + 1;
                std::ptrdiff_t iHigh = (xx + 0.5 + targetBox.getMinX())*binning - x0 + 1;
                for (std::ptrdiff_t ii = std::max(0L, iLow); ii < std::min(iHigh, width); ++ii) {
                    auto const value = image.get(lsst::geom::Point2I(ii, jj), lsst::afw::image::LOCAL);
                    array[yy][xx] += value;
                    sum += value;
                }
            }
        }
    }

    // Divide through by kernel sum
    for (int yy = 0; yy < targetBox.getHeight(); ++yy) {
        std::transform(array[yy].begin(), array[yy].end(), array[yy].begin(),
                       [sum](T const& value) { return value/sum; });
    }

    return resampled;
}


template<typename T>
std::shared_ptr<lsst::afw::image::Image<T>>
recenterOversampledKernelImage(
    lsst::afw::image::Image<T> const& image,
    int binning,
    lsst::geom::Point2D const& target
) {
    // Centroid using SdssCentroid: same as for our arc line measurements
    // This may be slow, as it does an initial first and second moment measurement before running
    // SdssCentroid. If it turns out to be too slow, we can have this function be provided a guess
    // PSF sigma.
    auto centroid = centroidImage(image);

    // Binning by an odd factor requires the centroid at the center of a pixel.
    // Binning by an even factor requires the centroid on the edge of a pixel.
    if (binning % 2 == 0) {
        centroid -= lsst::geom::Extent2D(0.5, 0.5);
    }

    auto const xPosition = lsst::afw::image::positionToIndex(target.getX(), true);
    auto const yPosition = lsst::afw::image::positionToIndex(target.getY(), true);
    double const xOffset = xPosition.second*binning - centroid.getX();
    double const yOffset = yPosition.second*binning - centroid.getY();

    std::string const warpAlgorithm = "lanczos5";
    unsigned int const warpBuffer = 5;
    auto recentered = lsst::afw::math::offsetImage(image, xOffset, yOffset, warpAlgorithm, warpBuffer);
    recentered->setXY0(recentered->getX0() + xPosition.first*binning,
                       recentered->getY0() + yPosition.first*binning);

    return recentered;
}


std::shared_ptr<OversampledPsf::Image>
OversampledPsf::doComputeKernelImage(
    lsst::geom::Point2D const& position,
    lsst::afw::image::Color const& color
) const {
    return resampleKernelImage(
        *recenterOversampledKernelImage(*doComputeOversampledKernelImage(position), _oversampleFactor),
        _oversampleFactor, computeBBox());
}


std::shared_ptr<OversampledPsf::Image>
OversampledPsf::doComputeImage(
    lsst::geom::Point2D const& position,
    lsst::afw::image::Color const& color
) const {
    return resampleKernelImage(
        *recenterOversampledKernelImage(*doComputeOversampledKernelImage(position), _oversampleFactor,
                                        position),
        _oversampleFactor, computeBBox(), position);
}


// This does a brute force calculation with no clever sinc apertures. Need to
// think about whether and how to pull over the sinc apertures code from LSST.
double OversampledPsf::doComputeApertureFlux(
    double radius,
    lsst::geom::Point2D const& position,
    lsst::afw::image::Color const& color
) const {
    std::shared_ptr<Image> const kernelPtr = doComputeOversampledKernelImage(position);
    Image const& kernel = *kernelPtr;
    double const r2 = std::pow(radius*_oversampleFactor, 2);
    double sumAperture = 0.0;
    double sumAll = 0.0;
    std::size_t num = 0;
    std::ptrdiff_t yy = kernel.getBBox().getMinY();
    for (std::ptrdiff_t jj = 0; jj < kernel.getHeight(); ++jj, ++yy) {
        std::ptrdiff_t xx = kernel.getBBox().getMinX();
        for (std::ptrdiff_t ii = 0; ii < kernel.getWidth(); ++ii, ++xx) {
            double const value = kernel.get(lsst::geom::Point2I(ii, jj), lsst::afw::image::LOCAL);
            sumAll += value;
            if (std::pow(double(xx), 2) + std::pow(double(yy), 2) < r2) {
                sumAperture += value;
                ++num;
            }
        }
    }
    return sumAperture/sumAll*M_PI*std::pow(radius*_oversampleFactor, 2)/num;
}


// This does a brute force calculation, rather than the usual adaptive moments,
// so it is subject to noise. Need to think about whether and how to pull over
// the adaptive moments code from LSST or GalSim.
lsst::afw::geom::ellipses::Quadrupole
OversampledPsf::doComputeShape(
    lsst::geom::Point2D const& position,
    lsst::afw::image::Color const& color
) const {
    std::shared_ptr<Image> const kernelPtr = doComputeOversampledKernelImage(position);
    Image const& kernel = *kernelPtr;

    // First moments
    double xSum = 0;
    double ySum = 0;
    double sum0 = 0;
    {
        std::ptrdiff_t yy = kernel.getY0();
        for (int jj = 0; jj < kernel.getHeight(); ++jj, ++yy) {
            std::ptrdiff_t xx = kernel.getX0();
            for (int ii = 0; ii < kernel.getWidth(); ++ii, ++xx) {
                double const value = kernel.get(lsst::geom::Point2I(ii, jj), lsst::afw::image::LOCAL);
                xSum += value*xx;
                ySum += value*yy;
                sum0 += value;
            }
        }
    }
    xSum /= sum0;
    ySum /= sum0;
    double const norm = 1.0/sum0;

    // Second moments
    double xxSum = 0;
    double xySum = 0;
    double yySum = 0;
    {
        std::ptrdiff_t yy = kernel.getY0();
        for (int jj = 0; jj < kernel.getHeight(); ++jj, ++yy) {
            double xx = kernel.getX0();
            for (int ii = 0; ii < kernel.getWidth(); ++ii, ++xx) {
                double const value = kernel.get(lsst::geom::Point2I(ii, jj), lsst::afw::image::LOCAL)*norm;
                double const dx = xx - xSum;
                double const dy = yy - ySum;
                xxSum += dx*dx*value;
                xySum += dx*dy*value;
                yySum += dy*dy*value;
            }
        }
    }
    double const factor = 1.0/std::pow(double(_oversampleFactor), 2);

    return lsst::afw::geom::ellipses::Quadrupole(xxSum*factor, yySum*factor, xySum*factor);
}


lsst::geom::Box2I OversampledPsf::doComputeBBox(
    lsst::geom::Point2D const& position,
    lsst::afw::image::Color const& color
) const {
    return lsst::geom::Box2I(lsst::geom::Point2I(-_targetSize.getX()/2, -_targetSize.getY()/2), _targetSize);
}


lsst::geom::Point2D SpectralPsf::getPosition(
    int fiberId,
    double wavelength
) const {
    lsst::geom::Point2D position = _detMap->findPoint(fiberId, wavelength);
    if (!std::isfinite(position.getX()) || !std::isfinite(position.getY())) {
        throw LSST_EXCEPT(lsst::pex::exceptions::DomainError,
                          (boost::format("Non-finite position: %f,%f") %
                           position.getX() % position.getY()).str());
    }
    if (!lsst::geom::Box2D(_detMap->getBBox()).contains(position)) {
        throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeError,
                          (boost::format("Position is off detector: %f,%f") %
                           position.getX() % position.getY()).str());
    }
    return position;
}


namespace {


struct ImagingSpectralPsfPersistenceHelper {
    lsst::afw::table::Schema schema;
    lsst::afw::table::Key<int> base;
    lsst::afw::table::Key<int> detectorMap;

    static ImagingSpectralPsfPersistenceHelper const& get() {
        static ImagingSpectralPsfPersistenceHelper instance;
        return instance;
    }

    // No copying or moving
    ImagingSpectralPsfPersistenceHelper(const ImagingSpectralPsfPersistenceHelper&) = delete;
    ImagingSpectralPsfPersistenceHelper& operator=(const ImagingSpectralPsfPersistenceHelper&) = delete;
    ImagingSpectralPsfPersistenceHelper(ImagingSpectralPsfPersistenceHelper&&) = delete;
    ImagingSpectralPsfPersistenceHelper& operator=(ImagingSpectralPsfPersistenceHelper&&) = delete;

private:
    ImagingSpectralPsfPersistenceHelper()
            : schema(),
              base(schema.addField<int>("base", "base imaging PSF", "")),
              detectorMap(schema.addField<int>("detectorMap", "detectorMap", "")) {
        schema.getCitizen().markPersistent();
    }
};


class ImagingSpectralPsfFactory : public lsst::afw::table::io::PersistableFactory {
public:
    std::shared_ptr<lsst::afw::table::io::Persistable> read(InputArchive const& archive,
                                                            CatalogVector const& catalogs) const override {
        static ImagingSpectralPsfPersistenceHelper const& keys = ImagingSpectralPsfPersistenceHelper::get();
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        lsst::afw::table::BaseRecord const& record = catalogs.front().front();
        LSST_ARCHIVE_ASSERT(record.getSchema() == keys.schema);
        auto psf = archive.get<lsst::afw::detection::Psf>(record.get(keys.base));
        auto detMap = archive.get<DetectorMap>(record.get(keys.detectorMap));
        return std::make_shared<ImagingSpectralPsf>(psf, detMap);
    }

    ImagingSpectralPsfFactory(std::string const& name) : lsst::afw::table::io::PersistableFactory(name) {}
};

}

ImagingSpectralPsfFactory registration("ImagingSpectralPsf");


void ImagingSpectralPsf::write(OutputArchiveHandle& handle) const {
    static ImagingSpectralPsfPersistenceHelper const& keys = ImagingSpectralPsfPersistenceHelper::get();
    lsst::afw::table::BaseCatalog catalog = handle.makeCatalog(keys.schema);
    std::shared_ptr<lsst::afw::table::BaseRecord> record = catalog.addNew();
    record->set(keys.base, handle.put(getBase()));
    record->set(keys.detectorMap, handle.put(getDetectorMap()));
    handle.saveCatalog(catalog);
}


// Explicit instantiation
#define INSTANTIATE(TYPE) \
std::shared_ptr<lsst::afw::image::Image<TYPE>> \
resampleKernelImage( \
    lsst::afw::image::Image<TYPE> const& image, \
    int binning, \
    lsst::geom::Box2I const& bbox, \
    lsst::geom::Point2D const& center \
); \
std::shared_ptr<lsst::afw::image::Image<TYPE>> \
recenterOversampledKernelImage( \
    lsst::afw::image::Image<TYPE> const& image, \
    int binning, \
    lsst::geom::Point2D const& center \
);

INSTANTIATE(lsst::afw::detection::Psf::Pixel);


}}}  // namespace pfs::drp::stella
