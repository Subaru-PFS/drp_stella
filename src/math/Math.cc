#include <Eigen/Eigen>

#include "ndarray/eigen.h"

#include "lsst/pex/exceptions.h"

#include "pfs/drp/stella/math/Math.h"
#include "pfs/drp/stella/cmpfit-1.2/MPFitting_ndarray.h"

namespace pfs {
namespace drp {
namespace stella {
namespace math {

template <typename T>
std::vector<std::size_t> getIndicesInValueRange(
    ndarray::Array<T const, 1, 1> const& array,
    T lowRange,
    T highRange
) {
    std::vector<std::size_t> indices;
    indices.reserve(array.getNumElements());
    std::size_t pos = 0;
    for (auto it = array.begin(); it != array.end(); ++it, ++pos) {
        if ((lowRange <= *it) && (*it < highRange)) {
            indices.push_back(pos);
        }
    }
    return indices;
}


template <typename T>
std::vector<lsst::geom::Point2I> getIndicesInValueRange(
    ndarray::Array<T, 2, 1> const& array,
    T lowRange,
    T highRange
) {
    std::vector<lsst::geom::Point2I> indices;
    indices.reserve(array.getNumElements());
    int const height = array.getShape()[0], width = array.getShape()[1];
    for (int yy = 0; yy < height; ++yy) {
        for (int xx = 0; xx < width; ++xx) {
            T const value = array[yy][xx];
            if ((lowRange <= value) && (value < highRange)) {
                indices.emplace_back(xx, yy);
            }
        }
    }
    return indices;
}


template <typename T>
ndarray::Array<T, 1, 1> getSubArray(
    ndarray::Array<T, 1, 1> const& array,
    std::vector<std::size_t> const& indices
) {
    ndarray::Array<T, 1, 1> out = ndarray::allocate(indices.size());
    std::transform(indices.begin(), indices.end(), out.begin(),
                   [array](std::size_t ii) { return array[ii]; });
    return out;
}


template <typename T>
ndarray::Array<T, 1, 1> getSubArray(
    ndarray::Array<T, 2, 1> const& array,
    std::vector<lsst::geom::Point2I> const& indices
) {
    ndarray::Array<T, 1, 1> out = ndarray::allocate(indices.size());
    std::transform(indices.begin(), indices.end(), out.begin(),
                   [array](lsst::geom::Point2I const& pp) {
                       return array[pp.getY()][pp.getX()];
                   });
    return out;
}


template<typename T>
ndarray::Array<T, 1, 1> moment(ndarray::Array<T const, 1, 1> const& array, int maxMoment)
{
    assert(maxMoment >= 0 && maxMoment <= 4);
    ndarray::Array<T, 1, 1> out = ndarray::allocate(maxMoment);
    out.deep() = 0.;
    assert(maxMoment >= 1 && maxMoment <= 4);
    assert(array.getShape()[0] >= 2);

    std::size_t const num = array.getShape()[0];
    T mean = asEigenArray(array).mean();
    out[0] = mean;
    if (maxMoment == 1) {
        return out;
    }

    // Variance
    ndarray::Array<T, 1, 1> resid = ndarray::copy(array - mean);
    Eigen::Array<T, Eigen::Dynamic, 1> eigenResid = asEigenArray(resid);
    T var = (eigenResid.pow(2).sum() - std::pow(ndarray::sum(resid), 2)/T(num))/(T(num) - 1.);
    out[1] = var;
    if (maxMoment <= 2) {
        return out;
    }

    T std = std::sqrt(var);
    if (std == 0.) {
        return out;
    }

    // Skew
    out[2] = eigenResid.pow(3).sum()/(num*std::pow(std, 3));

    if (maxMoment <= 3) {
        return out;
    }

    // Kurtosis
    out[3] = eigenResid.pow(4).sum()/(num*pow(std, 4)) - 3.;;
    return out;
}


namespace {

// Functor for comparing vector values by their indices
template <typename T>
struct IndicesComparator {
    std::vector<T> const& data;
    IndicesComparator(std::vector<T> const& data_) : data(data_) {}
    bool operator()(std::size_t lhs, std::size_t rhs) const {
        return data[lhs] < data[rhs];
    }
};

} // anonymous namespace


template<typename T>
std::vector<std::size_t> sortIndices(std::vector<T> const& data) {
    std::size_t const num = data.size();
    std::vector<std::size_t> indices(num);
    std::size_t index = 0;
    std::generate_n(indices.begin(), num, [&index]() { return index++; });
    std::sort(indices.begin(), indices.end(), IndicesComparator<T>(data));
    return indices;
}


template <typename T>
std::ptrdiff_t firstIndexWithValueGEFrom(
    ndarray::Array<T, 1, 1> const& array,
    T minValue,
    std::size_t fromIndex
) {
    assert(array.getNumElements() > 0 && fromIndex < array.getNumElements());
    auto const iter = std::find_if(array.begin() + fromIndex, array.end(),
                                   [minValue](T value) { return value >= minValue; });
    return iter == array.end() ? -1 : iter - array.begin();
}


template <typename T>
std::ptrdiff_t lastIndexWithZeroValueBefore(
    ndarray::Array<T, 1, 1> const& array,
    std::ptrdiff_t startPos
) {
    if (startPos < 0 || std::size_t(startPos) >= array.getNumElements()) {
        return -1;
    }
    std::ptrdiff_t index = startPos;
    for (auto i = array.begin() + startPos; i != array.begin(); --i, --index) {
        if (std::fabs(*i) < 0.00000000000000001) {
            return index;
        }
    }
    return -1;
}


template <typename T>
std::ptrdiff_t firstIndexWithZeroValueFrom(
    ndarray::Array<T, 1, 1> const& array,
    std::ptrdiff_t startPos
) {
    if (startPos < 0 || std::size_t(startPos) >= array.getNumElements()) {
        return -1;
    }
    auto const iter = std::find_if(array.begin() + startPos, array.end(),
                                   [](T value) { return std::fabs(value) < 0.00000000000000001; });
    return iter == array.end() ? -1 : iter - array.begin();
}


// Explicit instantiations
template std::vector<std::size_t> getIndicesInValueRange(
    ndarray::Array<float const, 1, 1> const&,
    float,
    float
);

template std::vector<lsst::geom::Point2I> getIndicesInValueRange(
    ndarray::Array<float, 2, 1> const&,
    float,
    float
);

template ndarray::Array<float, 1, 1> getSubArray(
    ndarray::Array<float, 1, 1> const&,
    std::vector<std::size_t> const&
);

template ndarray::Array<float, 1, 1> getSubArray(
    ndarray::Array<float, 2, 1 > const&,
    std::vector<lsst::geom::Point2I> const&
);

template ndarray::Array<float, 1, 1> moment(
    ndarray::Array<float const, 1, 1> const&,
    int maxMoment
);

template std::vector<std::size_t> sortIndices(std::vector<float> const&);

template std::ptrdiff_t firstIndexWithValueGEFrom(
    ndarray::Array<std::size_t, 1, 1> const& array,
    std::size_t minValue,
    std::size_t fromIndex
);

template std::ptrdiff_t lastIndexWithZeroValueBefore(
    ndarray::Array<std::size_t, 1, 1> const& array,
    std::ptrdiff_t startPos
);

template std::ptrdiff_t firstIndexWithZeroValueFrom(
    ndarray::Array<std::size_t, 1, 1> const&,
    std::ptrdiff_t startPos
);


}}}} // namespace pfs::drp::stella::math

template <typename T>
std::ostream& operator<<(std::ostream& os, std::vector<T> const& obj) {
    for (auto const& ii : obj) {
        os << ii << " ";
    }
    os << endl;
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, lsst::geom::Point<T, 2> const& point) {
    os << "(" << point.getX() << "," << point.getY() << ")";
    return os;
}

// Explicit instantiations
template std::ostream& operator<<(std::ostream&, std::vector<float> const&);
template std::ostream& operator<<(std::ostream&, std::vector<lsst::geom::Point2I> const&);
template std::ostream& operator<<(std::ostream&, lsst::geom::Point2I const&);
template std::ostream& operator<<(std::ostream&, lsst::geom::Point2D const&);
