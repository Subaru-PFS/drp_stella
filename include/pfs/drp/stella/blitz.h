#if !defined(PFS_DRP_STELLA_BLITZ_H)
#define PFS_DRP_STELLA_BLITZ_H

#include "ndarray.h"
#include "ndarray/eigen.h"
#include "blitz/array.h"

namespace pfs { namespace drp { namespace stella {
namespace utils {

/**
 * \brief Make a shallow copy of an ndarray::Array as a blitz::Array
 *
 * \note The ownership of the memory is not transferred to blitz
 */
template<class T>
blitz::Array<T, 2>
ndarrayToBlitz(ndarray::Array<T, 2, 1> arr)
{
    ndarray::EigenView<T, 2, 1> eim = arr.asEigen();
    int const stride = arr.template getStride<0>();
    return blitz::Array<T, 2>(eim.data(),
                              blitz::shape(eim.cols(), eim.rows()),
                              blitz::shape(1, stride),
                              blitz::neverDeleteData);
}

/**
 * \brief Make a shallow copy of an blitz::Array as an ndarray::Array
 *
 * \note The ownership of the memory is not transferred to eigen
 */
template<class T>
ndarray::Array<T, 2, 1>
blitzToNdarray(blitz::Array<T, 2> &bim)
{
    int const width = bim.extent(0);
    int const height = bim.extent(1);
    
    typename ndarray::Array<T, 2, 1>::Index shape = ndarray::makeVector(height, width);
    typename ndarray::Array<T, 2, 1>::Index strides = ndarray::makeVector(1, width);
    typename ndarray::Array<T, 2, 1> a = ndarray::external(bim.data(), shape, strides);
    
    return a;
}

}}}}
#endif
