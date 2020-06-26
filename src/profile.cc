#include "pfs/drp/stella/profile.h"

#include "pfs/drp/stella/math/quartiles.h"
#include "pfs/drp/stella/utils/checkSize.h"

namespace pfs {
namespace drp {
namespace stella {

std::pair<ndarray::Array<double, 1, 1>, ndarray::Array<bool, 1, 1>>
calculateSwathProfile(
    ndarray::Array<double, 2, 1> const& values,
    ndarray::Array<bool, 2, 1> const& mask,
    int rejIter,
    float rejThresh
) {
    utils::checkSize(mask.getShape(), values.getShape(), "mask");
    std::size_t const height = values.getShape()[0];
    std::size_t const width = values.getShape()[1];
    ndarray::Array<double, 1, 1> outValues = ndarray::allocate(width);
    ndarray::Array<bool, 1, 1> outMask = ndarray::allocate(width);

    for (std::size_t col = 0; col < width; ++col) {
        bool anyRejected = true;
        for (int ii = 0; ii < rejIter && anyRejected; ++ii) {
            double lower, median, upper;
            std::tie(lower, median, upper) = math::calculateQuartiles(values[ndarray::view()(col)],
                                                                      mask[ndarray::view()(col)]);
            double const threshold = 0.741*(upper - lower)*rejThresh;
            anyRejected = false;
            for (std::size_t row = 0; row < height; ++row) {
                if (!mask[row][col] && std::abs(values[row][col] - median) > threshold) {
                    mask[row][col] = true;
                    anyRejected = true;
                }
            }
        }
        double sum = 0.0;
        std::size_t num = 0;
        for (std::size_t row = 0; row < height; ++row) {
            if (!mask[row][col]) {
                sum += values[row][col];
                ++num;
            }
        }
        outValues[col] = num > 0 ? sum/num : 0.0;
        outMask[col] = num == 0 || !std::isfinite(sum);
    }

    return std::make_pair(outValues, outMask);
}


}}} // namespace pfs::drp::stella
