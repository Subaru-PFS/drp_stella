#include "pfs/drp/stella/Spectrum.h"

namespace pfs {
namespace drp {
namespace stella {


struct FitLineResult {
    double rms;  ///< RMS of the residuals
    bool isValid;  ///< Did the fit succeed?
    double amplitude;  ///< Amplitude of the Gaussian
    double center;  ///< Center of the Gaussian (pixels)
    double rmsSize;  ///< Gaussian RMS (sigma) of the line (pixels)
    double bg0;  ///< Background intercept
    double bg1;  ///< Background slope
    std::size_t num;  ///< Number of pixels

    FitLineResult(double _rms, bool _isValid, double _amplitude, double _center, double _rmsSize,
                  double _bg0, double _bg1, std::size_t _num
        ) : rms(_rms), isValid(_isValid), amplitude(_amplitude), center(_center), rmsSize(_rmsSize),
            bg0(_bg0), bg1(_bg1), num(_num) {}
};


/** Fit a line to the spectrum
 *
 * We fit a Gaussian line plus linear continuum to a nominated portion of the
 * spectrum. The linear continuum is expressed as bg0 + bg1*(x - center).
 *
 * @param spectrum : Spectrum to fit.
 * @param peakPosition : estimated position of line to fit (pixels).
 * @param rmsSize : estimated Gaussian RMS (sigma) of line to fit (pixels).
 * @param badBitMask : ignore pixels where (value & badBitmask) != 0
 * @param fittingHalfSize : Half-size of the fitting domain (pixels).
 * @return Fit result.
 */
FitLineResult fitLine(
    Spectrum const& spectrum,
    float peakPosition,
    float rmsSize,
    lsst::afw::image::MaskPixel badBitMask,
    std::size_t fittingHalfSize
);


}}}  // namespace pfs::drp::stella
