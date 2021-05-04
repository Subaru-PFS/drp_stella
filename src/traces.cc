#include "pfs/drp/stella/traces.h"

namespace pfs {
namespace drp {
namespace stella {


std::ostream& operator<<(std::ostream& os, TracePeak const& tp) {
    os << "TracePeak(" << tp.span.getY() << ", " << tp.span.getX0() << ", ";
    os << tp.peak << ", " << tp.span.getX1() << ", " << tp.peakErr << ")";
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


void centroidTrace(
    std::vector<std::shared_ptr<TracePeak>> & peaks,
    lsst::afw::image::MaskedImage<float> const& image,
    int radius,
    lsst::afw::image::MaskPixel badBitMask
) {
    for (auto & pp : peaks) {
        // Going with a basic centroid for now, since RHL's book says that's the best for high S/N.
        // Later, might try a Gaussian fit, or SdssCentroid's Gaussian quartic correction.
        int const xMin = std::max(int(pp->peak - radius), 0);
        int const xMax = std::min(int(pp->peak + radius + 0.5), image.getWidth() - 1);
        double xSum = 0.0;
        double sum = 0.0;
        auto iter = image.row_begin(pp->span.getY()) + xMin;
        for (int xx = xMin; xx <= xMax; ++xx, ++iter) {
            if (iter.mask() & badBitMask) continue;
            sum += iter.image();
            xSum += xx*iter.image();
        }
        double const xMean = xSum/sum;

        // Variance in centroid = sum(variance*(x - xMean)^2)/sum(image)^2
        double dx2Var = 0.0;
        iter = image.row_begin(pp->span.getY()) + xMin;
        for (int xx = xMin; xx <= xMax; ++xx, ++iter) {
            if (iter.mask() & badBitMask) continue;
            double const dx = xx - xMean;
            dx2Var += std::pow(dx, 2)*iter.variance();
        }
        double const xErr = std::sqrt(dx2Var)/sum;

        pp->peak = xMean;
        pp->peakErr = xErr;
    }
}


}}} // namespace pfs::drp::stella
