#ifndef PFS_DRP_STELLA_MATH_AFFINETRANSFORM_H
#define PFS_DRP_STELLA_MATH_AFFINETRANSFORM_H

// Some helpful conversions between AffineTransform and ndarray that were
// unfortunately left out of the lsst::geom::AffineTransform class.


#include "ndarray.h"
#include "ndarray/eigen.h"
#include "lsst/geom/AffineTransform.h"

namespace pfs {
namespace drp {
namespace stella {
namespace math {


/// Number of parameters for affine transform
constexpr std::size_t NUM_AFFINE_PARAMS { 6 };


/// Construct AffineTransformation from an array of parameters
inline lsst::geom::AffineTransform makeAffineTransform(
    ndarray::Array<double, 1, 1> const& parameters
) {
    if (parameters.size() != NUM_AFFINE_PARAMS) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LengthError,
            "Affine transformation parameters must have 6 elements"
        );
    }
    lsst::geom::AffineTransform result;
    result.setParameterVector(ndarray::asEigenMatrix(parameters));
    return result;
}


/// Convert AffineTransformation to an array of parameters
inline ndarray::Array<double, 1, 1> getAffineParameters(
    lsst::geom::AffineTransform const& transform
) {
    ndarray::Array<double, 1, 1> result = ndarray::allocate(ndarray::makeVector(NUM_AFFINE_PARAMS));
    ndarray::asEigenMatrix(result) = transform.getParameterVector();
    return result;
}


}}}}  // namespace pfs::drp::stella::math

#endif  // include guard
