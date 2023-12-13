#include "ndarray.h"

#include "lsst/pex/exceptions.h"
#include "pfs/drp/stella/backgroundIndices.h"

namespace pfs {
namespace drp {
namespace stella {


namespace {


ndarray::Array<int, 1, 1> getBackgroundIndices1d(
    int size,
    int bgSize
) {
    ndarray::Array<int, 1, 1> result = ndarray::allocate(size);

    float const center = 0.5*size;
    int const num = std::ceil(size/float(bgSize));
    float const scale = float(num - 1)/size;
    float const offset = 0.5*num;
    for (int ii = 0; ii < size; ++ii) {
        result[ii] = (ii - center)*scale + offset;
    }

    return result;
}


}  // anonymous namespace


int getNumBackgroundIndices(
    lsst::geom::Extent2I const& dims,
    lsst::geom::Extent2I const& bgSize
) {
    if (bgSize.getX() <= 0 || bgSize.getY() <= 0) {
        return 0;
    }
    int const xNum = std::ceil(dims.getX()/float(bgSize.getX()));
    int const yNum = std::ceil(dims.getY()/float(bgSize.getY()));
    return xNum*yNum;
}


ndarray::Array<int, 2, 1> calculateBackgroundIndices(
    lsst::geom::Extent2I const& dims,
    lsst::geom::Extent2I const& bgSize,
    int indexOffset
) {
    if (bgSize.getX() <= 0 || bgSize.getY() <= 0) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterError,
            "Background size must be positive"
        );
    }
    ndarray::Array<int, 2, 1> result = ndarray::allocate(dims.getY(), dims.getX());

    ndarray::Array<int, 1, 1> xIndex = getBackgroundIndices1d(dims.getX(), bgSize.getX());
    ndarray::Array<int, 1, 1> yIndex = getBackgroundIndices1d(dims.getY(), bgSize.getY());
    int const xNum = xIndex[xIndex.size() - 1] + 1;

    for (int yy = 0; yy < dims.getY(); ++yy) {
        int const yBg = yIndex[yy];
        for (int xx = 0; xx < dims.getX(); ++xx) {
            result[yy][xx] = indexOffset + yBg*xNum + xIndex[xx];
        }
    }

    return result;
}


template<typename ValueT>
std::shared_ptr<lsst::afw::image::Image<float>> makeBackgroundImage(
    lsst::geom::Extent2I const& dims,
    lsst::geom::Extent2I const& bgSize,
    ndarray::Array<ValueT, 1, 1> const& values,
    int indexOffset
) {
    int const xNum = std::ceil(dims.getX()/float(bgSize.getX()));
    int const yNum = std::ceil(dims.getY()/float(bgSize.getY()));
    auto result = std::make_shared<lsst::afw::image::Image<float>>(unsigned(xNum), unsigned(yNum));
    auto image = result->getArray();

    for (int yy = 0; yy < yNum; ++yy) {
        for (int xx = 0; xx < xNum; ++xx) {
            std::size_t const index = indexOffset + yy*xNum + xx;
            image[yy][xx] = values[index];
        }
    }

    return result;
}


// Explicit instantiation
#define INSTANTIATE(TYPE) \
template std::shared_ptr<lsst::afw::image::Image<float>> makeBackgroundImage<TYPE>( \
    lsst::geom::Extent2I const& dims, \
    lsst::geom::Extent2I const& bgSize, \
    ndarray::Array<TYPE, 1, 1> const& values, \
    int indexOffset \
);

INSTANTIATE(double);


}}}  // namespace pfs::drp::stella
