#if !defined(PFS_DRP_STELLA_BLITZ_H)
#define PFS_DRP_STELLA_BLITZ_H

#include <iostream>
#include <stdio.h>
#include "ndarray.h"
#include "ndarray/eigen.h"
#include "blitz/array.h"
#include <memory>

namespace pfs { namespace drp { namespace stella {
namespace utils {

/**
 * \brief Make a shallow copy of an ndarray::Array as a blitz::Array
 *
 * \note The ownership of the memory is not transferred to blitz
 */
template<class T>
blitz::Array<T, 1>
ndarrayToBlitz(ndarray::Array<T, 1, 1> arr)
{
    ndarray::EigenView<T, 1, 1> eim = arr.asEigen();
//    int const stride = arr.template getStride<0>();
    int const i_shape = arr.template getShape<0>();
    blitz::Array<T, 2> barr(eim.data(),
                            blitz::shape(i_shape),
//                            blitz::shape(1),
                            blitz::neverDeleteData);
//    barr.transposeSelf(blitz::secondDim, blitz::firstDim);
    return barr;
}
template<class T>
blitz::Array<T, 2>
ndarrayToBlitz(ndarray::Array<T, 2, 1> arr)
{
  ndarray::EigenView<T, 2, 1> eim = arr.asEigen();
  int const stride = arr.template getStride<0>();
  blitz::Array<T, 2> barr(eim.data(),
                          blitz::shape(eim.cols(), eim.rows()),
                          blitz::shape(1, stride),
                          blitz::neverDeleteData);
  barr.transposeSelf(blitz::secondDim, blitz::firstDim);
  return barr;
}

// Convert a Numpy array to a blitz one, using the original's data (no copy)
/*template<class T, int N>
blitz::Array<T,N> py_to_blitz(const ndarray::Array<T, N, 1> &arr_obj)
{
    blitz::TinyVector<int,N> shape(0);
    blitz::TinyVector<int,N> strides(0);

    for (int i=0;i<N;i++) {
        shape[i] = arr_obj.dimensions[i];
        strides[i] = arr_obj.strides[i]/sizeof(T);
    }
    return blitz::Array<T,N>((T*) arr_obj.data(),shape,strides,blitz::neverDeleteData);
}
*/

/*
 * // Convert a Numpy array to a blitz one, using the original's data (no copy)
 * template<class T, int N>
 * static blitz::Array<T,N> py_to_blitz(PyArrayObject *arr_obj)
 * {
 *     blitz::TinyVector<int,N> shape(0);
 *     blitz::TinyVector<int,N> strides(0);
 * 
 *     for (int i=0;i<N;i++) {
 *         shape[i] = arr_obj->dimensions[i];
 *         strides[i] = arr_obj->strides[i]/sizeof(T);
}
return blitz::Array<T,N>((T*) arr_obj->data,shape,strides,
blitz::neverDeleteData);
}
*/

/**
 * \brief Make a shallow copy of an blitz::Array as an ndarray::Array
 *
 * \note The ownership of the memory is not transferred to eigen
 */
template<class T>
ndarray::Array<T, 1, 1>
blitzToNdarray(blitz::Array<T, 1> &bim)
{
  int const height = bim.extent(0);
  
  typename ndarray::Array<T, 1, 1>::Index shape = ndarray::makeVector(height);
  typename ndarray::Array<T, 1, 1>::Index strides = ndarray::makeVector(1);
  typename ndarray::Array<T, 1, 1> a = ndarray::external(bim.data(), shape, strides);
  
  return a;
}

template<class T>
ndarray::Array<T, 2, 1>
blitzToNdarray(blitz::Array<T, 2> &bim)
{
    int const width = bim.extent(1);
    int const height = bim.extent(0);
    
    typename ndarray::Array<T, 2, 1>::Index shape = ndarray::makeVector(height, width);
    typename ndarray::Array<T, 2, 1>::Index strides = ndarray::makeVector(width, 1);
    typename ndarray::Array<T, 2, 1> a = ndarray::external(bim.data(), shape, strides);
    
    return a;
}

/**
 * \brief Make a deep copy of an blitz::Array as an ndarray::Array
 *
 * \note The ownership of the memory IS transferred to eigen
 */
template<class T>
ndarray::Array<T, 1, 1> copyBlitzToNdarray(blitz::Array<T, 1> &bim){
  int const height = bim.extent(0);
  
  typename ndarray::Array<T, 1, 1>::Index shape = ndarray::makeVector(height);
  typename ndarray::Array<T, 1, 1>::Index strides = ndarray::makeVector(1);
  typename ndarray::Array<T, 1, 1> a = ndarray::external(bim.data(), shape, strides);
  
  return ndarray::copy(a);
}

template<class T>
ndarray::Array<T, 2, 1> copyBlitzToNdarray(blitz::Array<T, 2> &bim){
  int const height = bim.extent(0);
  int const width = bim.extent(1);
  
  typename ndarray::Array<T, 2, 1>::Index shape = ndarray::makeVector(height, width);
  typename ndarray::Array<T, 2, 1>::Index strides = ndarray::makeVector(width, 1);
  typename ndarray::Array<T, 2, 1> a = ndarray::external(bim.data(), shape, strides);
  
  return ndarray::copy(a);
}

//template shared_pointer<typename ndarray::Array<unsigned short, 2, 1>> copyBlitzToNdarray(blitz::Array<unsigned short, 2> &bim);
//template shared_pointer<typename ndarray::Array<int, 2, 1>> copyBlitzToNdarray(blitz::Array<int, 2> &bim);
//template shared_pointer<typename ndarray::Array<long, 2, 1>> copyBlitzToNdarray(blitz::Array<long, 2> &bim);
//template shared_pointer<typename ndarray::Array<float, 2, 1>> copyBlitzToNdarray(blitz::Array<float, 2> &bim);
//template shared_pointer<typename ndarray::Array<double, 2, 1>> copyBlitzToNdarray(blitz::Array<double, 2> &bim);

}}}}
#endif
