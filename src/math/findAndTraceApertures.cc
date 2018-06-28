#include "ndarray.h"

#include "lsst/log/Log.h"

#include "pfs/drp/stella/math/CurveFitting.h"
#include "pfs/drp/stella/cmpfit-1.2/MPFitting_ndarray.h"
#include "pfs/drp/stella/math/findAndTraceApertures.h"

//#define __DEBUG_FINDANDTRACE__ 1
//#define __DEBUG_XCENTERS__ 1

namespace afwImage = lsst::afw::image;

namespace pfs {
namespace drp {
namespace stella {
namespace math {

template<typename ImageT, typename MaskT, typename VarianceT>
FiberTraceSet<ImageT, MaskT, VarianceT> findAndTraceApertures(
    afwImage::MaskedImage<ImageT, MaskT, VarianceT> const& maskedImage,
    DetectorMap const& detectorMap,
    FiberTraceFindingControl const& finding,
    FiberTraceFunctionControl const& function,
    FiberTraceProfileFittingControl const& fitting
) {
    LOG_LOGGER _log = LOG_GET("pfs.drp.stella.FiberTrace.findAndTraceApertures");
    LOGLS_TRACE(_log, "::pfs::drp::stella::math::findAndTraceApertures started");

    if (int(finding.apertureFwhm * 2.) + 1 <= int(finding.nTermsGaussFit)) {
        std::string message("finding.apertureFwhm too small for GaussFit ");
        message += "-> Try lower finding.nTermsGaussFit!";
        throw LSST_EXCEPT(pexExcept::LogicError, message.c_str());
    }

    afwImage::MaskedImage<ImageT, MaskT, VarianceT> maskedImageCopy(maskedImage, true);
    afwImage::Image<ImageT> & image = *maskedImageCopy.getImage();
    afwImage::Image<VarianceT> const& variance = *maskedImageCopy.getVariance();

    std::size_t const expectTraces = maskedImage.getWidth()/(function.xLow + function.xHigh);
    FiberTraceSet<ImageT, MaskT, VarianceT> traces(expectTraces + 1);
    lsst::afw::geom::Point2I nextSearchStart(0, 0);
    for (;;) {
        FiberTraceFunction func{function};
        LOGLS_TRACE(_log, "fiberTraceFunction.fiberTraceFunctionControl set");
        auto result = findCenterPositionsOneTrace(image, variance, finding, nextSearchStart);
        std::size_t const length = result.index.size();
        if (length < std::size_t(finding.minLength)) {
            break;
        }

        nextSearchStart = result.nextSearchStart;

        LOGLS_TRACE(_log, "D_A1_ApertureCenterIndex = " << result.index);
        LOGLS_TRACE(_log, "D_A1_ApertureCenterPos = " << result.position);

        std::size_t middle = 0.5*length;
        func.xCenter = result.position[middle];
        func.yCenter = result.index[middle];
        func.yHigh = std::size_t(result.index[length - 1]) - func.yCenter;
        func.yLow = std::ptrdiff_t(result.index[0]) - func.yCenter;
        LOGLS_TRACE(_log, "fiberTraceFunction->xCenter = " << func.xCenter);
        LOGLS_TRACE(_log, "fiberTraceFunction->yCenter = " << func.yCenter);
        LOGLS_TRACE(_log, "fiberTraceFunction->yHigh = " << func.yHigh);
        LOGLS_TRACE(_log, "fiberTraceFunction->yLow = " << func.yLow);
        LOGLS_TRACE(_log, "fiberTraceFunction bbox low = " << func.yCenter + func.yLow);

        /// Fit Polynomial
        float xRangeMin = result.index[0];
        float xRangeMax = result.index[length - 1];
        auto const fit = fitPolynomial(vectorToArray(result.index), vectorToArray(result.position),
                                       function.order, xRangeMin, xRangeMax);
        func.coefficients.deep() = fit.coeffs;
        LOGLS_TRACE(_log, "after PolyFit: fiberTraceFunction->coefficients = " << func.coefficients);

        // Find the fiberId
        lsst::afw::geom::Point2D const center(func.xCenter, func.yCenter);
        std::size_t const fiberId = detectorMap.findFiberId(center);
        traces.add(maskedImage, func, fitting, fiberId);
    }

    traces.sortTracesByXCenter();

    return traces;
}


namespace {

struct PointCompare {
    bool operator()(lsst::afw::geom::Point2D const& left, lsst::afw::geom::Point2D const& right) {
        return std::tie(left.getX(), left.getY()) < std::tie(right.getX(), right.getY());
    }
};

} // anonymous namespace


template<typename ImageT, typename VarianceT>
FindCenterPositionsOneTraceResult findCenterPositionsOneTrace(
    afwImage::Image<ImageT> & image,
    afwImage::Image<VarianceT> const& variance,
    FiberTraceFindingControl const& finding,
    lsst::afw::geom::Point<int> const& start
) {
    using FloatArray = ndarray::Array<float, 1, 1>;
    using IntArray = ndarray::Array<std::size_t, 1, 1>;

    std::size_t const width = image.getWidth();
    std::size_t const height = image.getHeight();

    lsst::afw::geom::Point<int> nextSearchStart(start);
    ndarray::Array<ImageT, 2, 1> const imageArray = image.getArray();
    ndarray::Array<VarianceT const, 2, 1> const varianceArray = variance.getArray();
    std::size_t minWidth = std::max(int(1.5*finding.apertureFwhm), int(finding.nTermsGaussFit));
    float maxTimesApertureWidth = 4.0; // XXX make configurable
    std::vector<float> gaussFitVariances;
    std::vector<float> gaussFitMean;
    int const nInd = 100; // XXX make configurable
    ndarray::Array<float, 2, 1> indGaussArr = ndarray::allocate(nInd, 2);

    std::ptrdiff_t maxApertureWidth = maxTimesApertureWidth*finding.apertureFwhm;

    std::ptrdiff_t firstWideSignal;
    std::ptrdiff_t firstWideSignalEnd;
    std::ptrdiff_t firstWideSignalStart;
    std::size_t rowBak;
    bool apertureFound;
    ndarray::Array<float, 1, 1> indexCol = ndarray::allocate(width);
    indexCol.deep() = ndarray::arange(0, width);
    #if defined(__DEBUG_FINDANDTRACE__) && __DEBUG_FINDANDTRACE__ > 2
    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: D_A1_IndexCol = " << indexCol << endl;
    #endif
    FloatArray guess = ndarray::allocate(finding.nTermsGaussFit);
    FloatArray gaussFitCoeffs = ndarray::allocate(finding.nTermsGaussFit);
    FloatArray gaussFitCoeffsBak = ndarray::allocate(finding.nTermsGaussFit);
    ndarray::Array<std::size_t, 1, 1> signal = ndarray::allocate(width);
    signal.deep() = 0;

    FloatArray apertureCenter = ndarray::allocate(height);
    FloatArray apertureCenterErr = ndarray::allocate(height);
    IntArray apertureCenterIndex = ndarray::allocate(height);
    apertureCenter.deep() = 0.;
    apertureCenterErr.deep() = 0.;
    apertureCenterIndex.deep() = 0;

    IntArray indSignal;
    #if defined(__DEBUG_FINDANDTRACE__)
    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: started" << endl;
    #endif
    IntArray ind;
    IntArray where;
    #if defined(__DEBUG_FINDANDTRACE__)
    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: fiberTraceFunctionFindingControl->signalThreshold = " <<
        finding.signalThreshold << endl;
    #endif

    FindCenterPositionsOneTraceResult result;
    result.index.reserve(height);
    result.position.reserve(height);
    result.error.reserve(height);

    // Search for Apertures

    int startIndex = start.getX();
    for (std::size_t row = nextSearchStart.getY(); row < height; ++row) {
        #if defined(__DEBUG_FINDANDTRACE__) && __DEBUG_FINDANDTRACE__ > 2
        cout << "i_Row = " << row << ": ccdArray[i_Row][*] = " << imageArray[ndarray::view(row)()] << endl;
        #elif defined(__DEBUG_FINDANDTRACE__)
        cout << "i_Row = " << row << " starting" << endl;
        #endif

        nextSearchStart.setY(row);
        nextSearchStart.setX(startIndex);
        rowBak = row;
        for (std::size_t col = 0; col < width; ++col) {
            if (col == 0) {
                signal[col] = (imageArray[row][col] > finding.signalThreshold) ? 1 : 0;
            } else if (imageArray[row][col] > finding.signalThreshold) {
                signal[col] = signal[col - 1] + 1;
            }
        }

        apertureFound = false;
        while (!apertureFound) {
            gaussFitVariances.resize(0);
            gaussFitMean.resize(0);
            #if defined(__DEBUG_FINDANDTRACE__) && __DEBUG_FINDANDTRACE__ > 2
            cout << "pfs::drp::stella::math::findAndTraceApertures: while: I_A1_Signal = " << signal << endl;
            cout << "pfs::drp::stella::math::findAndTraceApertures: while: I_MinWidth = " << minWidth << endl;
            cout << "pfs::drp::stella::math::findAndTraceApertures: while: I_StartIndex = " << startIndex << endl;
            #endif
            firstWideSignal = firstIndexWithValueGEFrom(signal, minWidth, startIndex);
            #if defined(__DEBUG_FINDANDTRACE__)
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << row <<
                ": I_FirstWideSignal found at index " << firstWideSignal << ", I_StartIndex = " <<
                startIndex << endl;
            #endif
            if (firstWideSignal < 0) {
                startIndex = 0;
                #if defined(__DEBUG_FINDANDTRACE__)
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << row <<
                    ": No Aperture found in row " << row << ", trying next row" << endl;
                #endif
                break;
            }

            firstWideSignalStart = lastIndexWithZeroValueBefore(signal, firstWideSignal) + 1;
            #if defined(__DEBUG_FINDANDTRACE__)
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << row <<
                ": 1. I_FirstWideSignalStart = " << firstWideSignalStart << endl;
            #endif

            firstWideSignalEnd = firstIndexWithZeroValueFrom(signal, firstWideSignal) - 1;
            #if defined(__DEBUG_FINDANDTRACE__)
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << row <<
                ": I_FirstWideSignalEnd = " << firstWideSignalEnd << endl;
            #endif

            if (firstWideSignalStart < 0) {
                #if defined(__DEBUG_FINDANDTRACE__)
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << row <<
                    ": WARNING: No start of aperture found -> Going to next Aperture." << endl;
                #endif

                if (firstWideSignalEnd < 0) {
                    #if defined(__DEBUG_FINDANDTRACE__)
                    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << row <<
                        ": 1. WARNING: No end of aperture found -> Going to next row." << endl;
                    #endif
                    break;
                }
                #if defined(__DEBUG_FINDANDTRACE__)
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << row <<
                    ": End of first wide signal found at index " << firstWideSignalEnd << endl;
                #endif
                // Set start index for next run
                startIndex = firstWideSignalEnd + 1;
                continue;
            }
            // Fit Gaussian and Trace Aperture
            if (firstWideSignalEnd < 0) {
                #if defined(__DEBUG_FINDANDTRACE__)
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << row <<
                    ": 2. WARNING: No end of aperture found -> Going to next row." << endl;
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << row <<
                    ": B_ApertureFound = " << apertureFound << endl;
                #endif
                break;
            }
            #if defined(__DEBUG_FINDANDTRACE__)
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << row <<
                ": End of first wide signal found at index " << firstWideSignalEnd << endl;
            #endif

            if (firstWideSignalEnd - firstWideSignalStart + 1 > maxApertureWidth) {
                firstWideSignalEnd = firstWideSignalStart + maxApertureWidth;
            }
            #if defined(__DEBUG_FINDANDTRACE__)
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << row <<
                ": I_FirstWideSignalStart = " << firstWideSignalStart << ", I_FirstWideSignalEnd = " <<
                firstWideSignalEnd << endl;
            #endif
            signal[ndarray::view(firstWideSignalStart, firstWideSignalEnd + 1)] = 0;

            // Set start index for next run
            startIndex = firstWideSignalEnd + 1;
            std::size_t const length = firstWideSignalEnd - firstWideSignalStart + 1;

            #if defined(__DEBUG_FINDANDTRACE__)
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << row <<
                ": I_Length = " << length << endl;
            #endif
            if (finding.nTermsGaussFit == 0) { // look for maximum only
                apertureCenter.deep() = 0.;
                apertureCenterErr.deep() = 0.;
                apertureFound = true;
                std::size_t maxPos = 0;
                ImageT tMax = 0.;
                for (std::ptrdiff_t i = firstWideSignalStart; i <= firstWideSignalEnd; ++i) {
                    if (imageArray[row][i] > tMax){
                        tMax = imageArray[row][i];
                        maxPos = i;
                    }
                }
                apertureCenter[row] = maxPos;
                apertureCenterErr[row] = 0.5;
                #if defined(__DEBUG_FINDANDTRACE__)
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << row <<
                    ": Aperture found at " << apertureCenter[row] << endl;
                #endif

                // Set signal to zero
                if (firstWideSignalEnd - 1 >= firstWideSignalStart + 1) {
                    imageArray[ndarray::view(row)(firstWideSignalStart + 1, firstWideSignalEnd)] = 0.;
                }
                continue;
            }
            // Fitting multiple Gaussian terms
            if (length <= std::size_t(finding.nTermsGaussFit)) {
                #if defined(__DEBUG_FINDANDTRACE__)
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << row <<
                    ": WARNING: Width of aperture <= " << finding.nTermsGaussFit <<
                    "-> abandoning aperture" << endl;
                #endif

                // Set signal to zero
                if (firstWideSignalEnd - 1 >= firstWideSignalStart + 1) {
                  imageArray[ndarray::view(row)(firstWideSignalStart + 1, firstWideSignalEnd)] = 0.;
                }
                continue;
            }
            // populate Arrays for GaussFit
            FloatArray x = copy(indexCol[ndarray::view(firstWideSignalStart, firstWideSignalEnd + 1)]);
            FloatArray y = copy(imageArray[ndarray::view(row)(firstWideSignalStart, firstWideSignalEnd + 1)]);
            for (auto it = y.begin(); it != y.end(); ++it) {
                if (*it < 0.000001) {
                    *it = 1.;
                }
            }
            #if defined(__DEBUG_FINDANDTRACE__)
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: 1. D_A1_Y set to " <<
                y << endl;
            #endif
            ndarray::Array<VarianceT const, 1, 1> const varianceSlice =
                varianceArray[ndarray::view(row)(firstWideSignalStart, firstWideSignalEnd + 1)];
            FloatArray yErr = ndarray::allocate(length);
            std::transform(varianceSlice.begin(), varianceSlice.end(), yErr.begin(),
                           [](ImageT value) { return value > 0 ? std::sqrt(value) : 1.0; });

            // Guess values for GaussFit
            if (finding.nTermsGaussFit == 3) {
                guess[0] = *std::max_element(y.begin(), y.end());
                guess[1] = firstWideSignalStart + 0.5*(firstWideSignalEnd - firstWideSignalStart);
                guess[2] = 0.5*finding.apertureFwhm;
            } else if (finding.nTermsGaussFit > 3) {
                guess[3] = std::max(std::min(y[0], y[length - 1]), float(0.1));
                if (finding.nTermsGaussFit > 4) {
                    guess[4] = (y[length - 1] - y[0])/(length - 1);
                }
            }

            gaussFitCoeffs.deep() = 0.;
            FloatArray gaussFitCoeffsErr = ndarray::allocate(gaussFitCoeffs.size());
            gaussFitCoeffsErr.deep() = 0.;

            #if defined(__DEBUG_FINDANDTRACE__)
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << row <<
                ": D_A1_X = " << x << ", D_A1_Y = " << y << endl;
            #endif

            ndarray::Array<int, 2, 1> limited = ndarray::allocate(finding.nTermsGaussFit, 2);
            limited.deep() = 1;
            ndarray::Array<float, 2, 1> limits = ndarray::allocate(finding.nTermsGaussFit, 2);
            limits[0][0] = 0.; // Peak lower limit
            limits[0][1] = 2.*guess[0]; // Peak upper limit
            limits[1][0] = firstWideSignalStart; // Centroid lower limit
            limits[1][1] = firstWideSignalEnd; // Centroid upper limit
            limits[2][0] = 0.25*finding.apertureFwhm; // Sigma lower limit
            limits[2][1] = finding.apertureFwhm; // Sigma upper limit
            if (finding.nTermsGaussFit > 3) {
                limits[3][0] = 0.;
                limits[3][1] = 2.*guess[3];
                if (finding.nTermsGaussFit > 4) {
                    limits[4][0] = guess[4]/10.;
                    limits[4][1] = guess[4]*10.;
                }
            }
            #if defined(__DEBUG_FINDANDTRACE__)
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << row <<
                ": 1. starting MPFitGaussLim: D_A1_Guess = " << guess << ", I_A2_Limited = " <<
                limited << ", D_A2_Limits = " << limits << endl;
            #endif

            if (!MPFitGaussLim(x, y, yErr, guess, limited, limits, 0, false,
                               gaussFitCoeffs, gaussFitCoeffsErr)) {
                #if defined(__DEBUG_FINDANDTRACE__)
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                    ": while: WARNING: GaussFit FAILED -> abandoning aperture" << endl;
                #endif

                // Set start index for next run
                startIndex = firstWideSignalEnd + 1;
                imageArray[ndarray::view(row)(firstWideSignalStart + 1, firstWideSignalEnd)] = 0.;
                continue;
            }
            #if defined(__DEBUG_FINDANDTRACE__)
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                ": while: D_A1_GaussFit_Coeffs = " << gaussFitCoeffs << endl;
            if (gaussFitCoeffs[0] > finding.saturationLevel){
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                    ": while: WARNING: Signal appears to be saturated" << endl;
            }
            if (gaussFitCoeffs[1] < firstWideSignalStart + 0.25*length ||
                gaussFitCoeffs[1] > firstWideSignalStart + 0.75*length) {
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                    ": while: Warning: Center of Gaussian far away from middle of signal" << endl;
            }
            #endif
            if (gaussFitCoeffs[1] < firstWideSignalStart ||
                gaussFitCoeffs[1] > firstWideSignalEnd) {
                #if defined(__DEBUG_FINDANDTRACE__)
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                    ": while: Warning: Center of Gaussian too far away from middle of signal -> abandoning aperture" << endl;
                #endif

                // Set signal to zero
                imageArray[ndarray::view(row)(firstWideSignalStart + 1, firstWideSignalEnd)] = 0.;

                /// Set start index for next run
                startIndex = firstWideSignalEnd + 1;
                continue;
            }
            if (gaussFitCoeffs[2] < finding.apertureFwhm/4. ||
                gaussFitCoeffs[2] > finding.apertureFwhm) {
                #if defined(__DEBUG_FINDANDTRACE__)
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                    ": while: WARNING: FWHM = " << gaussFitCoeffs[2] <<
                    " outside range -> abandoning aperture" << endl;
                #endif

                // Set signal to zero
                imageArray[ndarray::view(row)(firstWideSignalStart + 1, firstWideSignalEnd)] = 0.;
                #if defined(__DEBUG_FINDANDTRACE__)
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                    ": while: B_ApertureFound = " << apertureFound <<
                    ": 1. Signal set to zero from I_FirstWideSignalStart = " <<
                    firstWideSignalStart << " to I_FirstWideSignalEnd = " <<
                    firstWideSignalEnd << endl;
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                    ": while: 1. ccdArray(i_Row = " << row << ", Range(I_FirstWideSignalStart = " <<
                    firstWideSignalStart << ", I_FirstWideSignalEnd = " << firstWideSignalEnd <<
                    ")) set to " <<
                    imageArray[ndarray::view(row)(firstWideSignalStart, firstWideSignalEnd)] << endl;
                #endif
                // Set start index for next run
                startIndex = firstWideSignalEnd + 1;
                continue;
            }
            apertureCenter.deep() = 0.;
            apertureCenterErr.deep() = 0.;
            apertureFound = true;
            apertureCenter[row] = gaussFitCoeffs[1];
            apertureCenterErr[row] = gaussFitCoeffsErr[1];
            #if defined(__DEBUG_FINDANDTRACE__)
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << row <<
                ": Aperture found at " << apertureCenter[row] << endl;
            #endif
            imageArray[ndarray::view(row)(firstWideSignalStart + 1, firstWideSignalEnd)] = 0.;
        } // while no aperture found

        if (apertureFound) {
            /// Trace Aperture
            std::set<lsst::afw::geom::Point2D, PointCompare> xySorted;
            int apertureLength = 1;
            int length = 1;
            int apertureLost = 0;
            #if defined(__DEBUG_FINDANDTRACE__)
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                ": Starting to trace aperture " << endl;
            #endif
            gaussFitCoeffsBak.deep() = gaussFitCoeffs;
            while (apertureFound && (apertureLost < finding.nLost) && (row < height - 1) &&
                   length < finding.maxLength) {
                ++row;
                ++apertureLength;
                ++length;
                if (finding.nTermsGaussFit == 0) {
                    // look for maximum only
                    apertureFound = true;
                    std::size_t maxPos = 0;
                    ImageT tMax = 0.;
                    for (std::ptrdiff_t i = firstWideSignalStart; i <= firstWideSignalEnd; ++i) {
                        if (imageArray[row][i] > tMax) {
                            tMax = imageArray[row][i];
                            maxPos = i;
                        }
                    }
                    if (tMax < finding.signalThreshold) {
                        apertureLost++;
                    } else {
                        apertureCenter[row] = maxPos;
                        apertureCenterErr[row] = 0.5; // Half a pixel uncertainty
                        #if defined(__DEBUG_FINDANDTRACE__)
                        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                            ": Aperture found at " << apertureCenter[row] << endl;
                        #endif
                        if (apertureCenter[row] < apertureCenter[row - 1]) {
                            --firstWideSignalStart;
                            --firstWideSignalEnd;
                        } else if (apertureCenter[row] > apertureCenter[row - 1]) {
                            ++firstWideSignalStart;
                            ++firstWideSignalEnd;
                        }
                        imageArray[ndarray::view(row)(firstWideSignalStart + 1, firstWideSignalEnd)] = 0.;
                    }
                } else {
                    firstWideSignalStart = gaussFitCoeffsBak[1] - 1.6*gaussFitCoeffsBak[2];
                    firstWideSignalEnd = gaussFitCoeffsBak[1] + 1.6 * gaussFitCoeffsBak[2] + 1;
                    if (firstWideSignalStart < 0. || firstWideSignalEnd >= std::ptrdiff_t(width)) {
                        #if defined(__DEBUG_FINDANDTRACE__)
                        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                            ": start or end of aperture outside CCD -> Aperture lost" << endl;
                        #endif
                        // Set signal to zero
                        if (firstWideSignalStart < 0) {
                            firstWideSignalStart = 0;
                        }
                        if (firstWideSignalEnd >= std::ptrdiff_t(width)) {
                            firstWideSignalEnd = width - 1;
                        }
                        imageArray[ndarray::view(row)(firstWideSignalStart + 1, firstWideSignalEnd)] = 0.;
                        ++apertureLost;
                    } else {
                        length = firstWideSignalEnd - firstWideSignalStart + 1;
                        std::size_t const fitSize = firstWideSignalEnd - firstWideSignalStart + 1;

                        if (length <= finding.nTermsGaussFit) {
                            #if defined(__DEBUG_FINDANDTRACE__)
                            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                ": Warning: Width of Aperture <= " << finding.nTermsGaussFit <<
                                " -> Lost Aperture" << endl;
                            #endif
                            // Set signal to zero
                            imageArray[ndarray::view(row)(firstWideSignalStart + 1, firstWideSignalEnd)] = 0.;
                            ++apertureLost;
                        } else {
                            FloatArray x = copy(indexCol[ndarray::view(firstWideSignalStart, firstWideSignalEnd + 1)]);
                            FloatArray y = copy(imageArray[ndarray::view(row)(firstWideSignalStart, firstWideSignalEnd + 1)]);
                            FloatArray yErr = ndarray::allocate(fitSize);
                            auto const var = varianceArray[ndarray::view(row)(firstWideSignalStart, firstWideSignalEnd + 1)];
                            #if 0
                            std::transform(var.begin(), var.end(), yErr.begin(),
                                          [](ImageT value) { value > 0 ? std::sqrt(value) : 1.0; });
                            #else
                            yErr.deep() = 1.0;
                            #endif

                            std::size_t numGood = 0;
                            for (auto it = y.begin(); it != y.end(); ++it) {
                                if (*it < 0.000001) {
                                    *it = 1.;
                                }
                                if (*it >= finding.signalThreshold) {
                                    ++numGood;
                                }
                            }
                            if (numGood < minWidth) {
                                // Set signal to zero
                                imageArray[ndarray::view(row)(firstWideSignalStart + 1, firstWideSignalEnd)] = 0.;
                                ++apertureLost;
                                #if defined(__DEBUG_FINDANDTRACE__)
                                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                    ": Signal not wide enough => Aperture lost" << endl;
                                #endif
                            } else {
                                #if defined(__DEBUG_FINDANDTRACE__)
                                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                    ": 2. D_A1_Y set to " << y << endl;
                                #endif
                                guess.deep() = gaussFitCoeffsBak;
                                gaussFitCoeffs.deep() = 0.;
                                ndarray::Array<float, 1, 1> gaussFitCoeffsErr = ndarray::allocate(gaussFitCoeffs.size());
                                gaussFitCoeffsErr.deep() = 0.;

                                #if defined(__DEBUG_FINDANDTRACE__)
                                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                    ": while: D_A1_X = " << x << ", D_A1_Y = " << y << endl;
                                #endif

                                ndarray::Array<int, 2, 1> limited = ndarray::allocate(finding.nTermsGaussFit, 2);
                                limited.deep() = 1;
                                ndarray::Array<float, 2, 1> limits = ndarray::allocate(finding.nTermsGaussFit, 2);
                                limits[0][0] = 0.; // Peak lower limit
                                limits[0][1] = 2.*guess[0]; // Peak upper limit
                                limits[1][0] = firstWideSignalStart; // Centroid lower limit
                                limits[1][1] = firstWideSignalEnd; // Centroid upper limit
                                limits[2][0] = 0.25*finding.apertureFwhm; // Sigma lower limit
                                limits[2][1] = finding.apertureFwhm; // Sigma upper limit

                                // Restrict values from wandering once we're established
                                // XXX hard-coded values here should be made configurable
                                if (gaussFitVariances.size() > 15) {
                                    double const sum = std::accumulate(gaussFitMean.end() - 10, gaussFitMean.end(), 0.0);
                                    double const mean = sum/10.; // we're using the last 10 points; see previous line
                                    #if defined(__DEBUG_FINDANDTRACE__) && __DEBUG_FINDANDTRACE__ > 2
                                    for (int iMean = 0; iMean < gaussFitMean.size(); ++iMean){
                                        cout << "gaussFitMean[" << iMean << "] = " << gaussFitMean[iMean] << endl;
                                        cout << "gaussFitVariances[" << iMean << ") = " << gaussFitVariances[iMean] << endl;
                                    }
                                    cout << "sum = " << sum << ", mean = " << mean << endl;
                                    #endif
                                    double const sqSum = std::inner_product(gaussFitMean.end() - 10, gaussFitMean.end(),
                                                                            gaussFitMean.end() - 10, 0.0);
                                    double const stdev = std::sqrt(sqSum/10 - mean*mean);
                                    #if defined(__DEBUG_FINDANDTRACE__)
                                    cout << "GaussFitMean: sq_sum = " << sqSum << ", stdev = " << stdev << endl;
                                    #endif
                                    guess[1] = mean;
                                    limits[1][0] = mean - (3. * stdev) - 0.1;
                                    limits[1][1] = mean + (3. * stdev) + 0.1;
                                    #if defined(__DEBUG_FINDANDTRACE__)
                                    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                        ": while: D_A1_Guess[1] = " << guess[1] << ", Limits = " <<
                                        limits[ndarray::view(1)()] << endl;
                                    #endif
                                    // for (int iMean = 0; iMean < gaussFitMean.size(); ++iMean) // Inserted in error?
                                    double const sumVar = std::accumulate(gaussFitVariances.end() - 10, gaussFitVariances.end(), 0.0);
                                    double const meanVar = sumVar/10.;
                                    #if defined(__DEBUG_FINDANDTRACE__)
                                    cout << "GaussFitVariance: sum = " << sumVar << ", mean = " << meanVar << endl;
                                    #endif
                                    double const sqSumVar = std::inner_product(gaussFitVariances.end() - 10, gaussFitVariances.end(),
                                                                               gaussFitVariances.end() - 10, 0.0);
                                    double const stdevVar = std::sqrt(sqSumVar/10 - meanVar*meanVar);
                                    #if defined(__DEBUG_FINDANDTRACE__)
                                    cout << "sq_sum = " << sqSumVar << ", stdev = " << stdevVar << endl;
                                    #endif
                                    guess[2] = meanVar;
                                    limits[2][0] = meanVar - (3.*stdevVar);
                                    limits[2][1] = meanVar + (3.*stdevVar);
                                    #if defined(__DEBUG_FINDANDTRACE__)
                                    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                        ": while: D_A1_Guess[2] = " << guess[2] << ", Limits = " <<
                                        limits[ndarray::view(2)()] << endl;
                                    #endif
                                }
                                if (finding.nTermsGaussFit > 3) {
                                    limits[3][0] = 0.;
                                    limits[3][1] = 2.*guess[3];
                                    if (finding.nTermsGaussFit > 4) {
                                        limits[4][0] = guess[4]/10.;
                                        limits[4][1] = guess[4]*10.;
                                    }
                                }
                                #if defined(__DEBUG_FINDANDTRACE__)
                                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                    ": while: 2. starting MPFitGaussLim: D_A2_Limits = " << limits << endl;
                                #endif
                                if (limits[1][0] > *std::max_element(x.begin(), x.end()) ||
                                    limits[1][1] < *std::min_element(x.begin(), x.end())) {
                                    string message("pfs::drp::stella::math::findCenterPositionsOneTrace: ERROR: (limits[1][0](=");
                                    message += to_string(limits[1][0]) + ") > max(x)(=";
                                    message += to_string(*std::max_element(x.begin(), x.end()));
                                    message += ")) || (limits[1][1](=";
                                    message += to_string(limits[1][1]) + ") < min(x)(=";
                                    message += to_string(*std::min_element(x.begin(), x.end())) + "))";
                                    throw LSST_EXCEPT(pexExcept::RuntimeError, message.c_str());
                                }
                                if (!MPFitGaussLim(x, y, yErr, guess, limited, limits, 0, false, gaussFitCoeffs,
                                                   gaussFitCoeffsErr)) {
                                    #if defined(__DEBUG_FINDANDTRACE__)
                                    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                        ": Warning: GaussFit FAILED" << endl;
                                    #endif
                                    // Set signal to zero
                                    imageArray[ndarray::view(row)(firstWideSignalStart + 1, firstWideSignalEnd)] = 0.;
                                    ++apertureLost;
                                } else {
                                    gaussFitMean.push_back(gaussFitCoeffs[1]);
                                    gaussFitVariances.push_back(gaussFitCoeffs[2]);

                                    #if defined(__DEBUG_FINDANDTRACE__)
                                    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                        ": D_A1_GaussFit_Coeffs = " << gaussFitCoeffs << endl;
                                    if (gaussFitCoeffs[0] < finding.saturationLevel/5.){
                                        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                            ": WARNING: Signal less than 20% of saturation level" << endl;
                                    }
                                    if (gaussFitCoeffs[0] > finding.saturationLevel){
                                      cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                          ": WARNING: Signal appears to be saturated" << endl;
                                    }
                                    #endif
                                    if (gaussFitCoeffs[0] < finding.signalThreshold){
                                        #if defined(__DEBUG_FINDANDTRACE__)
                                        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                            ": WARNING: peak = " << gaussFitCoeffs[1] << " lower than signalThreshold -> abandoning aperture" << endl;
                                        #endif
                                        // Set signal to zero
                                        imageArray[ndarray::view(row)(firstWideSignalStart + 1, firstWideSignalEnd)] = 0.;
                                        #if defined(__DEBUG_FINDANDTRACE__)
                                        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                            ": 2. Signal set to zero from I_FirstWideSignalStart = " << firstWideSignalStart <<
                                            " to I_FirstWideSignalEnd = " << firstWideSignalEnd << endl;
                                        #endif
                                        ++apertureLost;
                                    } else if ((gaussFitCoeffs[1] < gaussFitCoeffsBak[1] - 1.) ||
                                        (gaussFitCoeffs[1] > gaussFitCoeffsBak[1] + 1.)) {
                                        #if defined(__DEBUG_FINDANDTRACE__)
                                        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                            ": Warning: Center of Gaussian too far away from middle of signal -> abandoning aperture" << endl;
                                        #endif
                                        /// Set signal to zero
                                        imageArray[ndarray::view(row)(firstWideSignalStart + 1, firstWideSignalEnd)] = 0.;
                                        ++apertureLost;
                                    } else if ((gaussFitCoeffs[2] < finding.apertureFwhm/4.) ||
                                        (gaussFitCoeffs[2] > finding.apertureFwhm)) {
                                        #if defined(__DEBUG_FINDANDTRACE__)
                                        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                            ": WARNING: FWHM = " << gaussFitCoeffs[2] << " outside range -> abandoning aperture" << endl;
                                        #endif
                                        /// Set signal to zero
                                        imageArray[ndarray::view(row)(firstWideSignalStart + 1, firstWideSignalEnd)] = 0.;
                                        #if defined(__DEBUG_FINDANDTRACE__)
                                        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                            ": 2. Signal set to zero from I_FirstWideSignalStart = " << firstWideSignalStart <<
                                            " to I_FirstWideSignalEnd = " << firstWideSignalEnd << endl;
                                        #endif
                                        ++apertureLost;
                                    } else {
                                        apertureLost = 0;
                                        apertureFound = true;
                                        apertureCenter[row] = gaussFitCoeffs[1];
                                        apertureCenterErr[row] = gaussFitCoeffsErr[1];
                                        #if defined(__DEBUG_FINDANDTRACE__)
                                        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                            ": Aperture found at " << apertureCenter[row] << endl;
                                        #endif
                                        // XXX Most of the rest of this block appears vestigial:
                                        // populating data structures that aren't used.
                                        gaussFitCoeffsBak.deep() = gaussFitCoeffs;
                                        std::size_t const xCorMinPos = 0.;
                                        std::size_t const xySize = fitSize + 2;
                                        ndarray::Array<float, 2, 1> xyRelativeToCenter = ndarray::allocate(xySize, 2);
                                        xyRelativeToCenter[0][0] = x[0] - gaussFitCoeffs[1] - 1.;
                                        xyRelativeToCenter[0][1] = 0.;
                                        xyRelativeToCenter[xySize - 1][0] = x[fitSize - 1] - gaussFitCoeffs[1] + 1.;
                                        xyRelativeToCenter[xySize - 1][1] = 0.;
                                        xyRelativeToCenter[ndarray::view(1, xySize - 1)(0)] = x - gaussFitCoeffs[1];
                                        xyRelativeToCenter[ndarray::view(1, xySize - 1)(1)] = y;

                                        // Oversampled Gaussian
                                        indGaussArr[ndarray::view()(0)] = xyRelativeToCenter[0][0];
                                        std::size_t ind = 0;
                                        float const factor = (xyRelativeToCenter[xySize - 1][0] - xyRelativeToCenter[0][0])/nInd;
                                        for (auto iter = indGaussArr.begin(); iter != indGaussArr.end(); ++iter, ++ind) {
                                            float const value = (*iter)[0] + ind*factor;
                                            (*iter)[0] = value;
                                            (*iter)[1] = gaussFitCoeffs[0] * exp(-(value*value)/(2.*guess[2]*guess[2]));
                                        }
                                        if (gaussFitVariances.size() > 20) {
                                            ndarray::Array<float, 2, 1> xysRelativeToCenter = ndarray::allocate(xySorted.size(), 2);
                                            auto itSorted = xySorted.begin();
                                            for (auto iter = xysRelativeToCenter.begin(); iter != xysRelativeToCenter.end();
                                                ++iter, ++itSorted) {
                                                (*iter)[0] = itSorted->getX();
                                                (*iter)[1] = itSorted->getY();
                                            }
                                            #if defined(__DEBUG_FINDANDTRACE__)
                                            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                                ": xCorMinPos = " << xCorMinPos << endl;
                                            #endif
                                        }
                                        if (gaussFitVariances.size() > 10) {
                                            for (auto iter = xyRelativeToCenter.begin(); iter != xyRelativeToCenter.end(); ++iter) {
                                                xySorted.insert(lsst::afw::geom::Point2D((*iter)[0] + xCorMinPos, (*iter)[1]));
                                            }
                                            apertureCenter[row] = apertureCenter[row] + xCorMinPos;
                                            #if defined(__DEBUG_FINDANDTRACE__)
                                            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                                ": Aperture position corrected to " << apertureCenter[row] << endl;
                                            #endif
                                        }
                                    } // end else if (D_A1_GaussFit_Coeffs(0) >= signalThreshold
                                } // end else if (GaussFit(D_A1_X, D_A1_Y, D_A1_GaussFit_Coeffs, S_A1_KeyWords_GaussFit, PP_Args_GaussFit))
                                imageArray[ndarray::view(row)(firstWideSignalStart + 1, firstWideSignalEnd)] = 0.;
                            } // end else if (sum(I_A1_Signal) >= I_MinWidth){
                        } // end if (I_Length > 3)
                    } // end else if (I_ApertureStart >= 0. && I_ApertureEnd < ccdArray.getShape()[1])
                } // end else if GaussFit
                if (apertureLost == finding.nLost && apertureLength < finding.minLength) {
                    row = rowBak;
                    #if defined(__DEBUG_FINDANDTRACE__)
                    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row set to " << row << endl;
                    #endif
                }
            } //end while(B_ApertureFound && (I_ApertureLost < 3) && i_Row < ccdArray.getShape()[0] - 2))

            /// Fit Polynomial to traced aperture positions
            #if defined(__DEBUG_FINDANDTRACE__) && __DEBUG_FINDANDTRACE__ > 2
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                ": D_A1_ApertureCenter = " << apertureCenter << endl;
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                ": I_A1_ApertureCenterIndex.getShape() = " << apertureCenterIndex.getShape() << endl;
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                ": D_A1_ApertureCenter.getShape() = " << apertureCenter.getShape() << endl;
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                ": D_A1_EApertureCenter.getShape() = " << apertureCenterErr.getShape() << endl;
            #endif

            auto itInd = apertureCenterIndex.begin();
            auto itCen = apertureCenter.begin();
            auto itECen = apertureCenterErr.begin();
            apertureLength = 0;
            result.index.resize(0);
            result.position.resize(0);
            result.error.resize(0);
            result.nextSearchStart = nextSearchStart;
            for (std::size_t iInd = 0; iInd < height; ++iInd) {
                if (*(itCen + iInd) > 0.) {
                    (*(itInd + iInd)) = 0;
                    ++apertureLength;
                    result.index.push_back(iInd);
                    result.position.push_back((*(itCen + iInd)));
                    result.error.push_back((*(itECen + iInd)));
                    #if defined(__DEBUG_FINDANDTRACE__)
                    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                        ": result.apertureCenterIndex[" << result.index.size() - 1 <<
                        "] set to " << result.index[result.index.size() - 1] << endl;
                    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                        ": result.apertureCenterPos[" << result.position.size() - 1 <<
                        "] set to " << result.position[result.position.size() - 1] << endl;
                    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                        ": result.eApertureCenterPos[" << result.error.size() - 1 <<
                        "] set to " << result.error[result.error.size() - 1] << endl;
                    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                        ": result.nextSearchStart set to " << result.nextSearchStart << endl;
                    #endif
                }
            }
            if (apertureLength > finding.minLength){
                #if defined(__DEBUG_FINDANDTRACE__)
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                    ": result.apertureCenterIndex.size() = " << result.index.size() << endl;
                #endif
                return result;
            }
        } // if aperture found
    } // for each row
    #if defined(__DEBUG_FINDANDTRACE__)
    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: result.apertureCenterIndex.size() = " <<
        result.index.size() << endl;
    #endif
    return result;
}


ndarray::Array<float, 1, 1>
calculateXCenters(FiberTraceFunction const& function) {
    ndarray::Array<float, 1, 1> xRowIndex = ndarray::allocate(function.yHigh - function.yLow + 1);
    float xRowInd = function.yCenter + function.yLow;
    for (auto i = xRowIndex.begin(); i != xRowIndex.end(); ++i, ++xRowInd) {
        *i = xRowInd;
    }
    return calculateXCenters(function, xRowIndex);
}


ndarray::Array<float, 1, 1>
calculateXCenters(
    pfs::drp::stella::FiberTraceFunction const& function,
    ndarray::Array<float, 1, 1> const& yIn
) {
    #ifdef __DEBUG_XCENTERS__
    cout << "pfs::drp::stella::calculateXCenters: function.ctrl.order = " << function.ctrl.order << endl;
    #endif
    float const rangeMin = function.yCenter + function.yLow;
    float const rangeMax = function.yCenter + function.yHigh;
    #ifdef __DEBUG_XCENTERS__
    cout << "pfs::drp::stella::calculateXCenters: range = " << rangeMin << "," << rangeMax << endl;
    #endif
    #ifdef __DEBUG_XCENTERS__
      cout << "pfs::drp::stella::math::calculateXCenters: Calculating Polynomial" << endl;
      cout << "pfs::drp::stella::calculateXCenters: Function = Polynomial" << endl;
      cout << "pfs::drp::stella::calculateXCenters: function.coefficients = " <<
         function.coefficients << endl;
    #endif
    return calculatePolynomial(yIn, function.coefficients, rangeMin, rangeMax);
}


// Explicit instantiations
template FiberTraceSet<float, lsst::afw::image::MaskPixel, float> findAndTraceApertures(
    afwImage::MaskedImage<float, lsst::afw::image::MaskPixel, float> const&,
    DetectorMap const&,
    FiberTraceFindingControl const&,
    FiberTraceFunctionControl const&,
    FiberTraceProfileFittingControl const&
);

template FindCenterPositionsOneTraceResult findCenterPositionsOneTrace(
    afwImage::Image<float> &,
    afwImage::Image<float> const&,
    FiberTraceFindingControl const&,
    lsst::afw::geom::Point<int> const&
);

}}}} // namespace pfs::drp::stella::math
