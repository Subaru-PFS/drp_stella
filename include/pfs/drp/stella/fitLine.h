#include "pfs/drp/stella/Spectrum.h"

namespace pfs {
namespace drp {
namespace stella {


/// Results of fitting a line to an array
struct FitLineResult {
    // Fit quality
    double rms;  ///< RMS of the residuals
    bool isValid;  ///< Did the fit succeed?
    std::size_t num;  ///< Number of pixels
    // Fit parameters
    double amplitude;  ///< Amplitude of the Gaussian
    double center;  ///< Center of the Gaussian (pixels)
    double rmsSize;  ///< Gaussian RMS (sigma) of the line (pixels)
    double bg0;  ///< Background intercept
    double bg1;  ///< Background slope
    // Fit errors
    double amplitudeErr;  ///< Error in amplitude of the Gaussian
    double centerErr;  ///< Error in center of the Gaussian (pixels)
    double rmsSizeErr;  ///< Error in Gaussian RMS (sigma) of the line (pixels)
    double bg0Err;  ///< Error in background intercept
    double bg1Err;  ///< Error in background slope

    FitLineResult(double _rms, bool _isValid, std::size_t _num,
                  double _amplitude, double _center, double _rmsSize, double _bg0, double _bg1,
                  double _amplitudeErr, double _centerErr, double _rmsSizeErr, double _bg0Err, double _bg1Err
        ) : rms(_rms), isValid(_isValid), num(_num),
            amplitude(_amplitude), center(_center), rmsSize(_rmsSize), bg0(_bg0), bg1(_bg1),
            amplitudeErr(_amplitudeErr), centerErr(_centerErr), rmsSizeErr(_rmsSizeErr),
            bg0Err(_bg0Err), bg1Err(_bg1Err)
        {}
};


/** Fit a line to an array
 *
 * We fit a Gaussian line plus linear continuum to a nominated portion of the
 * array. The linear continuum is expressed as bg0 + bg1*(x - center).
 *
 * We deliberately don't use the variance, to avoid introducing biases as a
 * function of flux, so the error values that come out should not be interpreted
 * as absolute expressions of the parameter errors.
 *
 * @param flux : Flux values to fit.
 * @param mask : Corresponding bitmask values.
 * @param peakPosition : estimated position of line to fit (pixels).
 * @param rmsSize : estimated Gaussian RMS (sigma) of line to fit (pixels).
 * @param badBitMask : ignore pixels where (value & badBitmask) != 0
 * @param fittingHalfSize : Half-size of the fitting domain (pixels; defaults to entire range).
 * @return Fit result.
 */
template <typename T>
FitLineResult fitLine(
    ndarray::Array<T const, 1, 1> const& flux,
    ndarray::Array<lsst::afw::image::MaskPixel const, 1, 1> const& mask,
    float peakPosition,
    float rmsSize,
    lsst::afw::image::MaskPixel badBitMask,
    std::size_t fittingHalfSize=0
);


/** Fit a line to the spectrum
 *
 * We fit a Gaussian line plus linear continuum to a nominated portion of the
 * spectrum. The linear continuum is expressed as bg0 + bg1*(x - center).
 *
 * @param spectrum : Spectrum to fit.
 * @param peakPosition : estimated position of line to fit (pixels).
 * @param rmsSize : estimated Gaussian RMS (sigma) of line to fit (pixels).
 * @param badBitMask : ignore pixels where (value & badBitmask) != 0
 * @param fittingHalfSize : Half-size of the fitting domain (pixels; defaults to entire range).
 * @return Fit result.
 */
FitLineResult fitLine(
    Spectrum const& spectrum,
    float peakPosition,
    float rmsSize,
    lsst::afw::image::MaskPixel badBitMask,
    std::size_t fittingHalfSize=0
) {
    return fitLine(
        spectrum.getFlux(), spectrum.getMask().getArray()[0].shallow(),
        peakPosition, rmsSize, badBitMask, fittingHalfSize
    );
}


}}}  // namespace pfs::drp::stella
