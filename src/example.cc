#include "pfs/drp/stella/blitz.h"
#include "pfs/drp/stella/Example.h"

namespace pfs { namespace drp { namespace stella {

/**
 * \brief Add two EigenViews (as returned by ndarray::Array::asEigen())
 * \returns The sum
 */
template<class T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
addImages(ndarray::EigenView<const T, 2, 1> const& im1, ///< First Eigen object
          ndarray::EigenView<const T, 2, 1> const& im2) ///< Second Eigen object
{
    return im1 + im2;
}

/**
 * \brief Add two EigenViews (as returned by ndarray::Array::asEigen()) into sum
 *
 * \note Using this 3-argument version may be more efficient, although the compiler should be able to use the
 * return-value-optimisation with the 2-argument version
 */
template<class T>
void
addImages(ndarray::EigenView<T, 2, 1> &sum,             ///< the desired sum.
                                        /// \note should be an rvalue reference in C++11
          ndarray::EigenView<T const, 2, 1> const& im1, ///< First Eigen object
          ndarray::EigenView<T const, 2, 1> const& im2) ///< Second Eigen object
{
    sum = im1 + im2;
}

/**
 * \brief Add two images using Eigen
 * \returns The sum
 */
template<class T>
lsst::afw::image::Image<T>
addImagesWithEigen(lsst::afw::image::Image<T> const & im1, ///< First image
                   lsst::afw::image::Image<T> const & im2) ///< Second image
{
    ndarray::EigenView<T const, 2, 1> const& eim1 = im1.getArray().asEigen();
    ndarray::EigenView<T const, 2, 1> const& eim2 = im2.getArray().asEigen();

    lsst::afw::image::Image<T> ans = lsst::afw::image::Image<T>(im1.getDimensions());
#if 0                                   // may require an extra copy
    ans.getArray().asEigen() = addImages(eim1, eim2);
#elif 0                                 // requires an rvalue reference for first argument (C++11)
    addImages(ans.getArray().asEigen(), eim1, eim2);
#else
    ndarray::EigenView<T, 2, 1> eans = ans.getArray().asEigen();
    addImages(eans, eim1, eim2);
#endif

    return ans;
}

/*
 * Now a Blitz++ version.
 * \note that arrays can't be const as blitz doesn't seem to support const arrays
 */
            
/**
 * \brief Add two Blitz++ arrays
 * \returns the sum
 */
template<class T>
blitz::Array<T, 2>
addImages(blitz::Array<T, 2> const& im1, ///< first Blitz++ array
          blitz::Array<T, 2> const& im2) ///< second Blitz++ array
{
    blitz::Array<T, 2> sum(im1.extent(0), im1.extent(1));
    sum = im1 + im2;
    return sum;
}
            
/**
 * \brief Add two Blitz++ arrays setting sum
 *
 * \note Using this 3-argument version may be more efficient, although the compiler should be able to use the
 * return-value-optimisation with the 2-argument version
 */
template<class T>
void
addImages(blitz::Array<T, 2> & sum,      ///< the desired sum
          blitz::Array<T, 2> const& im1, ///< first Blitz++ array
          blitz::Array<T, 2> const& im2) ///< second Blitz++ array
{
    sum = im1 + im2;
}
            
/**
 * \brief Add two images using Blitz++
 *
 * \note Blitz doesn't handle const arrays well, so these are not const
 */
template<class T>
lsst::afw::image::Image<T>
addImagesWithBlitz(lsst::afw::image::Image<T> & im1, ///< [in] First input image
                   lsst::afw::image::Image<T> & im2) ///< [in] Second input image
{
    blitz::Array<T, 2> bim1 = utils::ndarrayToBlitz(im1.getArray());
    blitz::Array<T, 2> bim2 = utils::ndarrayToBlitz(im2.getArray());

    lsst::afw::image::Image<T> ans = lsst::afw::image::Image<T>(im1.getDimensions());
    blitz::Array<T, 2> bans = utils::ndarrayToBlitz(ans.getArray());
#if 0                                   // may require an extra copy
    bans = addImages(bim1, bim2);
#else
    addImages(bans, bim1, bim2);
#endif

#if 0
    std::cout << "bim1 = " << bim1 << " bim2 = " << bim2 << " sum = " << bans << std::endl;
#endif
    
    return ans;
}

/*
 * Explicit instantiations
 * \cond
 */
#define INSTANTIATE(T) \
template \
lsst::afw::image::Image<T> \
addImagesWithBlitz(lsst::afw::image::Image<T> &, \
                   lsst::afw::image::Image<T> &); \
template \
lsst::afw::image::Image<T> \
addImagesWithEigen(lsst::afw::image::Image<T> const &, \
                   lsst::afw::image::Image<T> const &)

INSTANTIATE(double);
INSTANTIATE(float);
INSTANTIATE(int);
/// \endcond
}}}
