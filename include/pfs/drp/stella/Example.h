#if !defined(PFS_DRP_STELLA_EXAMPLE)
#define PFS_DRP_STELLA_EXAMPLE

#include "lsst/afw/image/Image.h"

namespace pfs { namespace drp { namespace stella {

template<class T>
lsst::afw::image::Image<T>
addImagesWithEigen(lsst::afw::image::Image<T> const & im1,
                   lsst::afw::image::Image<T> const & im2);

template<class T>
lsst::afw::image::Image<T>
addImagesWithBlitz(lsst::afw::image::Image<T> & im1,
                   lsst::afw::image::Image<T> & im2);

}}}

#endif
