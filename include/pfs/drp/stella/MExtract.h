/*
author: Andreas Ritter
created: 04/12/2007
last edited: 05/08/2012
compiler: g++ 4.0
basis machine: Ubuntu Linux 6.06
*/

#ifndef __MEXTRACT_H__
#define __MEXTRACT_H__

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <iostream>
#include <cstdlib>
#include <time.h>

//#include "CAny.h"
#include "CFits.h"
//#include "pfs/drp/stella/blitz.h"

using namespace std;
namespace pfs { namespace drp { namespace stella {
int optextract(const string &fitsFileName_In,
               const string &databaseFileName_In,
               const string &fitsFileName_Out,
               const double &gain_In,
               const double &readNoise_In,
               const blitz::Array<string, 1> &parameterKeywords_In,
               const void *parameterValues_In[]);
///USAGE: optextract char[] FitsFileName_In, char[] DatabaseFileName_In, char[] FitsFileName_Out, double Gain, double ReadOutNoise[, TELLURIC=int[0 - none, 1 - Piskunov, 2 - LinFit]][, MAX_ITER_SF=int][, MAX_ITER_SKY=int][, MAX_ITER_SIG=int][, SWATH_WIDTH=int][, SF_SMOOTH=double][, SP_SMOOTH=int][, ERR_IN=char[]][, ERR_OUT_2D=char[]][, ERR_OUT_EC=char[]][, SKY_OUT_EC=char[]][, SKY_OUT_2D=char[]][, SKY_ERR_OUT_EC=char[]][, PROFILE_OUT=char[]][, IM_REC_OUT=char[]][, REC_FIT_OUT=char[]][, MASK_OUT=char[]][, SPFIT_OUT_EC=char[]][, EC_FROM_PROFILE_OUT=char[]][,AREA=[int(xmin),int(xmax),int(ymin),int(ymax)]]//"[, ERR_FROM_PROFILE_OUT=char[]])
///FitsFileName_In: image to extract
///DatabaseFileName_In: aperture-definition file to use for extraction
///FitsFileName_Out: output filename containing extracted spectra
///Gain: CCD gain
///ReadOutNoise: CCD readout noise
///TELLURIC: 0 - none, 1 - Piskunov, 2 - LinFit
///MAX_ITER_SF: maximum number of iterations calculating the slit function (spatial profile)
///MAX_ITER_SKY: maximum number of iterations calculating the sky (TELLURIC = 2 only)
///MAX_ITER_SIG: maximum number of iterations rejecting cosmic-ray hits
///SWATH_WIDTH: width of swath (bin) for which an individual profile shall be calculated
///SMOOTH_SF: Width of median SlitFunc smoothing
///SMOOTH_SP: Width of median Spectrum/Blaze smoothing
///ERR_IN: input image containing the uncertainties in the pixel values of FitsFileName_In
///ERR_OUT_2D: output uncertainty image - same as ERR_IN, but with detected cosmic-ray hits set to 10,000
///ERR_OUT_EC: output file containing the uncertainties in the extracted spectra's pixel values
///SKY_OUT_EC: output sky-only spectra (TELLURIC > 0 only)
///SKY_OUT_2D: reconstructed sky-only image
///SKY_ERR_OUT_EC: uncertainties in the calculated sky-only values
///PROFILE_OUT: reconstructed image of the spatial profiles
///IM_REC_OUT: reconstructed input image from the profile-fitting/extraction
///SPFIT_OUT_EC: extracted spectra from linear fit of spatial profiles to input spectra with 3-sigma rejection (ignoring mask), with sky if TELLURIC>0, without sky if TELLURIC=0
///REC_FIT_OUT: reconstructed input image for SPFIT_OUT_EC
///MASK_OUT: output mask with detected cosmic-ray hits set to 0, good pixels set to 1
///EC_FROM_PROFILE_OUT: extracted spectra from simply multiplying the input image with the profile image as weight and summing up///SHALL I REJECT COSMIC-RAY HITS???????????????
///AREA: Area from which to extract spectra if center of aperture is in specified area
}}}
#endif
