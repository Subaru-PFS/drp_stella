#include <vector>

#include "lsst/afw/table.h"
#include "lsst/afw/math/LeastSquares.h"
#include "pfs/drp/stella/photometry.h"

namespace pfs {
namespace drp {
namespace stella {


lsst::afw::table::BaseCatalog photometer(
    lsst::afw::image::MaskedImage<float> const& image,
    ndarray::Array<int, 1, 1> const& fiberId,
    ndarray::Array<double, 1, 1> const& wavelength,
    pfs::drp::stella::SpectralPsf const& psf,
    lsst::afw::image::MaskPixel badBitMask    
) {
    // Set up output
    lsst::afw::table::Schema schema;
    auto const fiberIdKey = schema.addField<int>("fiberId", "fiber identifier");
    auto const wavelengthKey = schema.addField<double>("wavelength", "line wavelength", "nm");
    auto const flagKey = schema.addField<lsst::afw::table::Flag>("flag", "bad measurement?");
    auto const fluxKey = schema.addField<double>("flux", "measured line flux");
    auto const fluxErrKey = schema.addField<double>("fluxErr", "measured line flux error");
    lsst::afw::table::BaseCatalog catalog{schema};
    std::size_t const numWavelengths = wavelength.size();
    std::size_t const num = fiberId.size()*numWavelengths;
    catalog.reserve(num);

    // Realise PSF models
    using Image = lsst::afw::detection::Psf::Image;
    std::vector<std::shared_ptr<Image>> models{num};
    std::size_t index = 0;
    for (auto ff : fiberId) {
        for (std::size_t ii = 0; ii < numWavelengths; ++ii, ++index) {
            double const wl = wavelength[ii];
            auto row = catalog.addNew();
            row->set(fiberIdKey, ff);
            row->set(wavelengthKey, wl);
            row->set(flagKey, false);
            std::shared_ptr<Image> psfImage = psf.computeImage(ff, wl);
            if (!psfImage) {
                continue;
            }
            lsst::geom::Box2I box = psfImage->getBBox();
            box.clip(image.getBBox());
            if (box.isEmpty()) {
                continue;
            }
            auto psfArray = ndarray::asEigenArray(psfImage->getArray());
            psfArray /= psfArray.sum();
            models[index] = std::make_shared<Image>(psfImage->subset(box));
        }
    }

    // Generate least-squares equations
    // The unweighted version (matrix,vector) is used for the measurement (no weighting to avoid
    // flux-dependent bias). The weighted version (weightedMatrix,weightedVector) is used for the
    // measurement errors.
    ndarray::Array<double, 2, 1> matrix = ndarray::allocate(num, num);
    ndarray::Array<double, 1, 1> vector = ndarray::allocate(num);
    ndarray::Array<double, 2, 1> weightedMatrix = ndarray::allocate(num, num);
    ndarray::Array<double, 1, 1> weightedVector = ndarray::allocate(num);
    matrix.deep() = 0.0;
    vector.deep() = 0.0;
    weightedMatrix.deep() = 0.0;
    weightedVector.deep() = 0.0;

    auto const imgArray = image.getImage()->getArray();
    auto const varArray = image.getVariance()->getArray();
    for (std::size_t ii = 0; ii < num; ++ii) {
        if (!models[ii]) {
            // avoid matrix singularity
            matrix[ii][ii] = 1.0;
            weightedMatrix[ii][ii] = 1.0;
            catalog[ii].set(flagKey, true);
            continue;
        }
        Image const& iModel = *models[ii];
        auto const iBox = iModel.getBBox();
        {
            auto const spans = lsst::afw::geom::SpanSet(iBox).intersectNot(*image.getMask(), badBitMask);
            // Hold arrays in scope while we use Eigen for math
            auto const mm_ = spans->flatten(iModel.getArray(), iModel.getXY0());
            auto const dd_ = spans->flatten(imgArray, image.getXY0());
            auto const vv_ = spans->flatten(varArray, image.getXY0());
            auto const mm = ndarray::asEigenArray(mm_);
            auto const dd = ndarray::asEigenArray(dd_).cast<double>();
            auto const vv = ndarray::asEigenArray(vv_).cast<double>();

            matrix[ii][ii] = mm.square().sum();  // model dot model
            vector[ii] = (mm*dd).sum();  // model dot data
            weightedMatrix[ii][ii] = (mm.square()/vv).sum();
            weightedVector[ii] = (mm*dd/vv).sum();
        }

        for (std::size_t jj = ii + 1; jj < num; ++jj) {
            if (!models[jj]) {
                continue;
            }
            Image const& jModel = *models[jj];
            auto jBox = jModel.getBBox();
            if (!jBox.overlaps(iBox)) {
                continue;
            }
            jBox.clip(iBox);
            auto spans = lsst::afw::geom::SpanSet(jBox).intersectNot(*image.getMask(), badBitMask);
            // Hold arrays in scope while we use Eigen for math
            auto const im_ = spans->flatten(iModel.getArray(), iModel.getXY0());
            auto const jm_ = spans->flatten(jModel.getArray(), jModel.getXY0());
            auto const vv_ = spans->flatten(varArray, image.getXY0());
            auto const im = ndarray::asEigenArray(im_);
            auto const jm = ndarray::asEigenArray(jm_);
            auto const vv = ndarray::asEigenArray(vv_).cast<double>();

            matrix[ii][jj] = matrix[jj][ii] = (im*jm).sum();  // model dot model
            weightedMatrix[ii][jj] = weightedMatrix[jj][ii] = (im*jm/vv).sum();
        }
    }

    // Solve least-squares equation and extract output
    using LeastSquares = lsst::afw::math::LeastSquares;
    auto const solution = LeastSquares::fromNormalEquations(matrix, vector).getSolution();
    auto const covar = LeastSquares::fromNormalEquations(weightedMatrix, weightedVector).getCovariance();
    for (std::size_t ii = 0; ii < num; ++ii) {
        catalog[ii].set(fluxKey, solution[ii]);
        catalog[ii].set(fluxErrKey, std::sqrt(covar[ii][ii]));
    }

    return catalog;
}


}}}  // namespace pfs::drp::stella