#include <vector>
#include <unordered_map>

#include "Eigen/Sparse"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/table.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/math/LeastSquares.h"
#include "pfs/drp/stella/utils/checkSize.h"
#include "pfs/drp/stella/photometry.h"

namespace pfs {
namespace drp {
namespace stella {


namespace {


/// Return an image of the PSF for the nominated emission line
//
/// @param psf : The point-spread function model
/// @param point : The point at which to evaluate the PSF model
/// @param bbox : Bounding box of the exposure
/// @returns PSF image
std::shared_ptr<lsst::afw::detection::Psf::Image> getPsfImage(
    pfs::drp::stella::SpectralPsf const& psf,
    lsst::geom::Point2D const& point,
    lsst::geom::Box2I const& bbox
) {
    std::shared_ptr<pfs::drp::stella::SpectralPsf::Image> psfImage;
    try {
        psfImage = psf.computeImage(point);
    } catch (lsst::pex::exceptions::Exception const&) {
        return nullptr;
    }
    if (!psfImage) return nullptr;
    lsst::geom::Box2I psfBox = psfImage->getBBox();
    psfBox.clip(bbox);
    if (psfBox.isEmpty()) return nullptr;
    return std::make_shared<lsst::afw::detection::Psf::Image>(psfImage->subset(psfBox));
}


/// Sparse representation of a square matrix
///
/// Used for solving matrix problems.
class SparseSquareMatrix {
  public:
    using ElemT = double;
    using IndexT = std::ptrdiff_t;

    /// Ctor
    ///
    /// The matrix is initialised to zero.
    ///
    /// @param num : Number of columns/rows
    /// @param nonZeroPerCol : Estimated mean number of non-zero entries per column
    SparseSquareMatrix(std::size_t num, float nonZeroPerCol=2.0)
      : _num(num) {
        _triplets.reserve(std::size_t(num*nonZeroPerCol));
    }

    virtual ~SparseSquareMatrix() {}
    SparseSquareMatrix(SparseSquareMatrix const&) = delete;
    SparseSquareMatrix(SparseSquareMatrix &&) = default;
    SparseSquareMatrix & operator=(SparseSquareMatrix const&) = delete;
    SparseSquareMatrix & operator=(SparseSquareMatrix &&) = default;

    /// Add an entry to the matrix
    void add(IndexT ii, IndexT jj, ElemT value) {
        assert(ii >= 0 && jj >= 0 && ii < std::ptrdiff_t(_num) && jj < std::ptrdiff_t(_num));
        _triplets.emplace_back(ii, jj, value);
    }

    /// Solve the matrix equation
    ndarray::Array<ElemT, 1, 1> solve(ndarray::Array<ElemT, 1, 1> const& rhs) {
        using Matrix = Eigen::SparseMatrix<ElemT, 0, IndexT>;
        utils::checkSize(rhs.size(), std::size_t(_num), "rhs");
        Matrix matrix(_num, _num);
        matrix.setFromTriplets(_triplets.begin(), _triplets.end());
        Eigen::SparseQR<Matrix, Eigen::NaturalOrdering<IndexT>> solver;
        solver.compute(matrix);
        if (solver.info() != Eigen::Success) {
            throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Sparse matrix decomposition failed.");
        }
        ndarray::Array<ElemT, 1, 1> solution = ndarray::allocate(_num);
        ndarray::asEigenMatrix(solution) = solver.solve(ndarray::asEigenMatrix(rhs));
        if (solver.info() != Eigen::Success) {
            throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Sparse matrix solving failed.");
        }
        return solution;
    }

  private:
    std::size_t _num;  ///< Number of rows/columns
    std::vector<Eigen::Triplet<ElemT>> _triplets;  ///< Triplets (two coordinates and a value) of matrix elems
};


}  // anonymous namespace


lsst::afw::table::BaseCatalog photometer(
    lsst::afw::image::MaskedImage<float> const& image,
    ndarray::Array<int, 1, 1> const& fiberId,
    ndarray::Array<double, 1, 1> const& wavelength,
    pfs::drp::stella::SpectralPsf const& psf,
    lsst::afw::image::MaskPixel badBitMask,
    ndarray::Array<double, 2, 1> const& positions_
) {
    utils::checkSize(fiberId.size(), wavelength.size(), "fiberId vs wavelength");
    std::size_t const num = fiberId.size();

    ndarray::Array<double, 2, 1> const& positions = positions_.isEmpty() ?
        psf.getDetectorMap()->findPoint(fiberId, wavelength) : positions_;
    utils::checkSize(positions.getShape(), ndarray::makeVector<ndarray::Size>(num, 2),
                     "fiberId vs positions");

    // Set up output
    lsst::afw::table::Schema schema;
    auto const fiberIdKey = schema.addField<int>("fiberId", "fiber identifier");
    auto const wavelengthKey = schema.addField<double>("wavelength", "line wavelength", "nm");
    auto const flagKey = schema.addField<lsst::afw::table::Flag>("flag", "bad measurement?");
    auto const fluxKey = schema.addField<double>("flux", "measured line flux");
    auto const fluxErrKey = schema.addField<double>("fluxErr", "measured line flux error");
    lsst::afw::table::BaseCatalog catalog{schema};
    catalog.reserve(num);
    for (std::size_t ii = 0; ii < num; ++ii) {
        auto & row = *catalog.addNew();
        row.set(fiberIdKey, fiberId[ii]);
        row.set(wavelengthKey, wavelength[ii]);
        row.set(flagKey, true);
        row.set(fluxKey, std::numeric_limits<double>::quiet_NaN());
        row.set(fluxErrKey, std::numeric_limits<double>::quiet_NaN());
    }

    // Identify blends: PSFs that touch each other
    std::unordered_map<std::size_t, std::vector<std::size_t>> blendComponents;
    {
        lsst::afw::image::Image<std::size_t> blendImage{image.getBBox()};
        blendImage = 0;  // 0 means no blend at this position
        std::unordered_map<std::size_t, std::size_t> blendAliases;
        std::size_t blendIndex = 1;  // 0 in the blendImage means no blend, so start at 1
        for (std::size_t ii = 0; ii < num; ++ii) {
            double const xx = positions[ii][0];
            double const yy = positions[ii][1];
            if (!std::isfinite(xx) || !std::isfinite(yy)) {
                // Bad position: not even a blend.
                continue;
            }
            lsst::geom::Point2D const point{xx, yy};
            lsst::geom::Box2D box;
            try {
                box = lsst::geom::Box2D(psf.computeBBox(point));
            } catch (lsst::pex::exceptions::Exception const&) {
                // Bad PSF: not even a blend.
                continue;
            }
            box.shift(lsst::geom::Extent2D(point));  // put the center on the point
            box.grow(1);  // for good measure
            box.clip(lsst::geom::Box2D(blendImage.getBBox()));
            lsst::afw::image::Image<std::size_t> subImage{blendImage, lsst::geom::Box2I(box)};
            std::set<std::size_t> blends{subImage.begin(), subImage.end()};
            blends.erase(0);
            switch (blends.size()) {
              case 0:
                // No overlaps. This is a new blend all by itself.
                subImage = blendIndex;
                blendComponents[blendIndex].push_back(ii);
                ++blendIndex;
                continue;
              case 1: {
                // A single overlap. This is a blend with the other source already there.
                std::size_t bb = *(blends.begin());
                while (blendAliases.find(bb) != blendAliases.end()) bb = blendAliases[bb];
                subImage = bb;
                blendComponents[bb].push_back(ii);
                continue;
              }
              default: {
                // Multiple overlaps. We've joined multiple blends into a much larger blend.
                // Take the first (lowest value), and have all the others point to that now.
                std::size_t bb = *(blends.begin());
                while (blendAliases.find(bb) != blendAliases.end()) bb = blendAliases[bb];
                subImage = bb;
                auto & target = blendComponents[bb];
                target.push_back(ii);
                auto iter = blends.begin();
                ++iter;  // Don't want the first, as that's the target.
                for (; iter != blends.end(); ++iter) {
                    std::size_t aa = *iter;
                    while (blendAliases.find(aa) != blendAliases.end()) aa = blendAliases[aa];
                    if (aa == bb) continue;  // must have already done this one
                    auto & source = blendComponents[aa];
                    target.insert(target.end(), source.begin(), source.end());
                    blendComponents.erase(aa);
                    blendAliases[aa] = bb;
                }
              }
            }
        }
    }

    for (auto & blend : blendComponents) {
        auto const& indices = blend.second;
        std::size_t const blendSize = indices.size();

        // Generate least-squares equations
        SparseSquareMatrix matrix{blendSize};
        ndarray::Array<double, 1, 1> vector = ndarray::allocate(blendSize);
        ndarray::Array<double, 1, 1> errors = ndarray::allocate(blendSize);
        vector.deep() = 0.0;

        using Image = lsst::afw::detection::Psf::Image;
        auto const imgArray = image.getImage()->getArray();
        auto const varArray = image.getVariance()->getArray();
        auto const bbox = image.getBBox();
        for (std::size_t ii = 0; ii < blendSize; ++ii) {
            std::size_t const iIndex = indices[ii];
            lsst::geom::Point2D const iPoint{positions[iIndex][0], positions[iIndex][1]};
            auto & row = catalog[iIndex];
            auto const iPsfImage = getPsfImage(psf, iPoint, bbox);
            if (!iPsfImage) {
                matrix.add(ii, ii, 1.0);  // avoid matrix singularity
                errors[ii] = std::numeric_limits<double>::quiet_NaN();
                row.set(flagKey, true);
                continue;
            }
            row.set(flagKey, false);
            Image const& iModel = *iPsfImage;
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

                double const modelDotModel = mm.square().sum();  // model dot model
                matrix.add(ii, ii, modelDotModel);
                vector[ii] = (mm*dd).sum();  // model dot data
                errors[ii] = std::sqrt((mm.square()*vv).sum())/modelDotModel;
            }

            // Check for masked pixels in the central area
            {
                lsst::geom::Box2I central(lsst::geom::Point2I(iPoint) - lsst::geom::Extent2I(1, 1),
                                          lsst::geom::Extent2I(3, 3));
                central.clip(bbox);
                auto const spans = lsst::afw::geom::SpanSet(central).intersect(*image.getMask(), badBitMask);
                if (spans->getArea() > 0) {
                    row.set(flagKey, true);
                }
            }

            auto const bounds = lsst::geom::Box2D::makeCenteredBox(iPoint, 2.0*iBox.getDimensions());
            for (std::size_t jj = ii + 1; jj < blendSize; ++jj) {
                std::size_t const jIndex = indices[jj];
                lsst::geom::Point2D const jPoint{positions[jIndex][0], positions[jIndex][1]};
                if (!bounds.contains(jPoint)) {
                    continue;
                }
                auto const jPsfImage = getPsfImage(psf, jPoint, bbox);
                if (!jPsfImage) {
                    continue;
                }
                Image const& jModel = *jPsfImage;
                auto jBox = jModel.getBBox();
                if (!jBox.overlaps(iBox)) {
                    continue;
                }
                jBox.clip(iBox);
                auto spans = lsst::afw::geom::SpanSet(jBox).intersectNot(*image.getMask(), badBitMask);
                // Hold arrays in scope while we use Eigen for math
                auto const im_ = spans->flatten(iModel.getArray(), iModel.getXY0());
                auto const jm_ = spans->flatten(jModel.getArray(), jModel.getXY0());
                auto const im = ndarray::asEigenArray(im_);
                auto const jm = ndarray::asEigenArray(jm_);

                double const modelDotModel = (im*jm).sum();  // model dot model
                matrix.add(ii, jj, modelDotModel);
                matrix.add(jj, ii, modelDotModel);
            }
        }

        // Solve least-squares equation and extract output
        auto const solution = matrix.solve(vector);

        for (std::size_t ii = 0; ii < blendSize; ++ii) {
            auto & row = catalog[indices[ii]];
            row.set(fluxKey, solution[ii]);
            row.set(fluxErrKey, errors[ii]);
        }
    }

    return catalog;
}


}}}  // namespace pfs::drp::stella