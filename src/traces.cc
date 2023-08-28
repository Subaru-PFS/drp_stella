#include "ndarray.h"
#include "ndarray/eigen.h"

#include "pfs/drp/stella/traces.h"
#include "pfs/drp/stella/utils/math.h"

namespace pfs {
namespace drp {
namespace stella {


std::ostream& operator<<(std::ostream& os, TracePeak const& tp) {
    os << "TracePeak(" << tp.span.getY() << ", " << tp.span.getX0() << ", ";
    os << tp.peak << ", " << tp.span.getX1() << ", " << tp.peakErr << ", ";
    os << tp.flux << ", " << tp.fluxErr << ")";
    return os;
}


namespace {

// State of finding peaks
//
// Used for findTracePeaks.
enum PeakState {
    NONE,  // We're not tracking a peak
    BEFORE,  // We're ascending to a peak
    AFTER,  // We're descending from a peak
};

} // anonymous namespace


std::vector<std::vector<std::shared_ptr<TracePeak>>> findTracePeaks(
    lsst::afw::image::MaskedImage<float> const& image,
    float threshold,
    lsst::afw::image::MaskPixel badBitMask
) {
    std::size_t const height = image.getHeight();
    std::size_t const width = image.getWidth();
    std::vector<std::vector<std::shared_ptr<TracePeak>>> peaks{height};
    for (int yy = 0; yy < int(height); ++yy) {
        auto iter = image.row_begin(yy);
        auto & rowPeaks = peaks[yy];
        rowPeaks.reserve(width/4);
        PeakState state = NONE;
        int low = -1;
        int peak = -1;
        float lastValue = 0;
        int xLast = -1;
        for (int xx = 0; xx < int(width); ++xx, ++iter) {
            if (iter.mask() & badBitMask) {
                // Masked pixel means we don't change state
                continue;
            }
            float const value = iter.image();
            if (value > threshold) {
                switch (state) {
                  case NONE:
                    // We've found a trace
                    state = BEFORE;
                    low = xx;
                    break;
                  case BEFORE:
                    if (value < lastValue) {
                        // We found the peak
                        state = AFTER;
                        peak = xLast;
                    }
                    break;
                  case AFTER:
                    if (value > lastValue) {
                        // We found the end of this trace and the start of the next
                        rowPeaks.push_back(std::make_shared<TracePeak>(yy, low, peak, xLast));
                        state = BEFORE;
                        low = xx;
                        peak = -1;
                    }
                    break;
                }
            } else {
                switch (state) {
                  case BEFORE:
                    // The last pixel must be the peak
                    rowPeaks.push_back(std::make_shared<TracePeak>(yy, low, xLast, xLast));
                    state = NONE;
                    low = -1;
                    peak = -1;
                    break;
                  case AFTER:
                    // We found the end of this trace
                    rowPeaks.push_back(std::make_shared<TracePeak>(yy, low, peak, xLast));
                    state = NONE;
                    low = -1;
                    peak = -1;
                    break;
                  case NONE:
                    // Nothing changed
                    break;
                }
            }
            lastValue = value;
            xLast = xx;
        }
        // Forcibly end the search
        switch (state) {
          case BEFORE:
            // Last pixel must be the peak
            rowPeaks.push_back(std::make_shared<TracePeak>(yy, low, width - 1, width - 1));
            break;
          case AFTER:
            // We found the end of this trace
            rowPeaks.push_back(std::make_shared<TracePeak>(yy, low, peak, width - 1));
            break;
          case NONE:
            // Nothing changed
            break;
        }
    }
    return peaks;
}


std::map<int, std::vector<std::shared_ptr<TracePeak>>> findTracePeaks(
    lsst::afw::image::MaskedImage<float> const& image,
    DetectorMap const& detectorMap,
    float threshold,
    float radius,
    lsst::afw::image::MaskPixel badBitMask,
    ndarray::Array<int, 1, 1> const& fiberId_
) {
    std::size_t const height = image.getHeight();
    std::size_t const width = image.getWidth();
    ndarray::Array<int, 1, 1> const& fiberId = fiberId_.isEmpty() ? detectorMap.getFiberId() : fiberId_;
    std::map<int, std::vector<std::shared_ptr<TracePeak>>> peaks;
    for (int ff : fiberId) {
        ndarray::Array<double, 1, 1> expectedCenter = detectorMap.getXCenter(ff);
        if (expectedCenter.size() != height) {
            throw LSST_EXCEPT(lsst::pex::exceptions::LengthError, "Mismatch between detectorMap and image");
        }
        std::vector<std::shared_ptr<TracePeak>> fiberPeaks;
        for (std::size_t yy = 0; yy < height; ++yy) {
            double const xCenter = expectedCenter[yy];
            std::ptrdiff_t const xStart = std::max(0L, std::ptrdiff_t(xCenter - radius));
            std::ptrdiff_t const xStop = std::min(width - 1, std::size_t(xCenter + radius + 0.5));
            float valuePeak = -std::numeric_limits<float>::infinity();
            std::ptrdiff_t xPeak = std::numeric_limits<std::ptrdiff_t>::lowest();
            auto iter = image.row_begin(yy) + xStart;
            for (std::ptrdiff_t xx = xStart; xx <= xStop; ++xx, ++iter) {
                if (iter.mask() & badBitMask) {
                    continue;
                }
                if (iter.image() > valuePeak) {
                    valuePeak = iter.image();
                    xPeak = xx;
                }
            }
            if (valuePeak > threshold) {
                int const low = xPeak - radius;
                int const high = xPeak + radius + 0.5;
                fiberPeaks.emplace_back(std::make_shared<TracePeak>(yy, low, xPeak, high));
            }
        }
        peaks[ff] = std::move(fiberPeaks);
    }
    return peaks;
}


std::map<int, ndarray::Array<double, 2, 2>> extractTraceData(
    std::map<int, std::vector<std::shared_ptr<TracePeak>>> const& peaks
) {
    std::map<int, ndarray::Array<double, 2, 2>> result;
    for (auto const& item : peaks) {
        int const fiberId = item.first;
        auto const& fiberPeaks = item.second;
        ndarray::Array<double, 2, 2> data = ndarray::allocate(fiberPeaks.size(), 5);
        for (std::size_t ii = 0; ii < fiberPeaks.size(); ++ii) {
            auto const& peak = *fiberPeaks[ii];
            data[ii][0] = peak.span.getY();
            data[ii][1] = peak.peak;
            data[ii][2] = peak.peakErr;
            data[ii][3] = peak.flux;
            data[ii][4] = peak.fluxErr;
        }
        result[fiberId] = std::move(data);
    }
    return result;
}


namespace {

/// Convolve a pixel along the row
///
/// Returns convolved image and variance values for the pixel.
std::pair<double, double> convolvePixelRow(
    lsst::afw::image::MaskedImage<float> const& image,  ///< Unconvolved image
    lsst::geom::Point2I pixel,  ///< Pixel at the center of the convolution kernel
    float psfSigma,  ///< Gaussian sigma of PSF
    lsst::afw::image::MaskPixel badBitMask,  ///< Bitmask for bad pixels
    float extent  ///< Extent of convolution kernel, relative to psfSigma
) {
    int const xx = pixel.getX();
    int const yy = pixel.getY();
    int const size = psfSigma*extent + 0.5;  // truncated
    int const xMin = std::max(0, xx - size);  // inclusive
    int const xMax = std::min(image.getWidth() - 1, xx + size);  // inclusive
    double const invSigma = 1.0/psfSigma;
    double sumImage = 0.0;
    double sumVariance = 0.0;
    double sumKernel = 0.0;
    auto iter = image.row_begin(yy) + xMin;
    for (int ii = xMin; ii <= xMax; ++ii, ++iter) {
        if (iter.mask() & badBitMask) {
            continue;
        }
        double const kernel = std::exp(-0.5*std::pow((ii - xx)*invSigma, 2));
        sumImage += iter.image()*kernel;
        sumVariance += iter.variance()*std::pow(kernel, 2);
        sumKernel += kernel;
    }
    return std::make_pair(sumImage/sumKernel, sumVariance/std::pow(sumKernel, 2));
}


}  // anonymous namespace


void centroidPeak(
    TracePeak & peak,
    lsst::afw::image::MaskedImage<float> const& image,
    float psfSigma,
    lsst::afw::image::MaskPixel badBitMask,
    float extent,
    float ampAst4
) {
    // SdssCentroid-like Gaussian quartic correction
    // We apply a Gaussian convolution to the central three pixels, and then apply the quartic correction
    int const x0 = peak.peak + 0.5;  // truncated
    if (x0 < 1 || x0 > image.getWidth() - 2) {
        peak.peak = std::numeric_limits<double>::quiet_NaN();
        peak.peakErr = std::numeric_limits<double>::quiet_NaN();
        return;
    }
    int const row = peak.span.getY();
    lsst::geom::Point2I const left{x0 - 1, row};
    lsst::geom::Point2I const middle{x0, row};
    lsst::geom::Point2I const right{x0 + 1, row};
    auto const leftConv = convolvePixelRow(image, left, psfSigma, badBitMask, extent);
    auto const middleConv = convolvePixelRow(image, middle, psfSigma, badBitMask, extent);
    auto const rightConv = convolvePixelRow(image, right, psfSigma, badBitMask, extent);

    // Calculate quartic correction:
    // From RHL's photo lite paper:
    // slope: s = 0.5*(right - left)
    // deriv: d = 2*middle - left - right
    // peak flux: A
    // ampAst4: k
    // center: x = s/d.(1 + kd/4A.(1 - 4s^2/d^2))
    double const slope = 0.5*(rightConv.first - leftConv.first);
    double const deriv = 2*middleConv.first - leftConv.first - rightConv.first;
    double const aa = middleConv.first;
    double const slope2 = std::pow(slope, 2);
    double const invDeriv = 1.0/deriv;
    double const invDeriv2 = std::pow(invDeriv, 2);
    double const kOnFourA = ampAst4/(4*aa);

    if (deriv <= 0.0 || middleConv.first <= 0.0) {
        peak.peak = std::numeric_limits<double>::quiet_NaN();
        peak.peakErr = std::numeric_limits<double>::quiet_NaN();
        return;
    }

    double const corr = slope*invDeriv*(1 + kOnFourA*deriv*(1 - 4*slope2*invDeriv2));
    double const center = x0 + corr;
    if (std::abs(corr) > 1) {
        peak.peak = std::numeric_limits<double>::quiet_NaN();
        peak.peakErr = std::numeric_limits<double>::quiet_NaN();
        return;
    }

    // Calculate errors
    // From RHL's photo lite paper:
    // var(x) = var(s)(1/d + k/4A.(1 - 12s^2/d^2))^2 + var(d)(s/d^2 - k/4A.8s^2/d^3)^2
    double const slopeVar = 0.25*std::hypot(rightConv.second, leftConv.second);
    double const derivVar = std::sqrt(4*std::pow(middleConv.second, 2) +
                                        std::pow(leftConv.second, 2) +
                                        std::pow(rightConv.second, 2));
    double const slopeTerm = invDeriv + kOnFourA*(1 - 12*slope2*invDeriv2);
    double const derivTerm = slope*invDeriv2 - kOnFourA*8*slope2*invDeriv*invDeriv2;
    double const centerVar = slopeVar*std::pow(slopeTerm, 2) + derivVar*std::pow(derivTerm, 2);

    double const aaVar = middleConv.second;

    peak.peak = center;
    peak.peakErr = std::sqrt(centerVar);
    peak.flux = aa;
    peak.fluxErr = std::sqrt(aaVar);
}


template <typename T>
ndarray::Array<T, 2, 1> medianFilterColumns(
    ndarray::Array<T, 2, 1> const& image,
    ndarray::Array<bool, 2, 1> const& mask,
    int halfHeight
) {
    ndarray::Array<T, 2, 1> result = ndarray::allocate(image.getShape());
    int const height = result.getShape()[0];
    int const width = result.getShape()[1];
    for (int yy = 0; yy < height; ++yy) {
        int const yLow = std::max(0, yy - halfHeight);
        int const yHigh = std::min(height, yy + halfHeight + 1);  // exclusive
        for (int xx = 0; xx < width; ++xx) {
            result[yy][xx] = math::calculateMedian(
                image[ndarray::view(yLow, yHigh)(xx)], mask[ndarray::view(yLow, yHigh)(xx)]
            );
        }
    }

    return result;
}


// Explicit instantiations
#define INSTANTIATE(TYPE) \
template ndarray::Array<TYPE, 2, 1> medianFilterColumns( \
    ndarray::Array<TYPE, 2, 1> const& image, \
    ndarray::Array<bool, 2, 1> const& mask, \
    int halfHeight \
);

INSTANTIATE(float);

}}} // namespace pfs::drp::stella
