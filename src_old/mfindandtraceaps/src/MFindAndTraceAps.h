/*
author: Andreas Ritter
created: 04/12/2007
last edited: 05/05/2007
compiler: g++ 4.0
basis machine: Ubuntu Linux 6.06
*/

#ifndef __MFINDANDTRACEAPS_H__
#define __MFINDANDTRACEAPS_H__

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <iostream>
#include <cstdlib>
#include <time.h>

#include "../../cfits/src/CAny.h"
#include "../../cfits/src/CString.h"
#include "../../cfits/src/CFits.h"

using namespace std;

int main(int argc, char *argv[]);
/// "USAGE: findandtraceaps(
///Parameter 1: char[] FitsFileName_In,
///Parameter 2: char[] ReferenceScatteredLightImage_In,
///Parameter 3: int CentralSize_X_In,
///Parameter 4: int CentralSize_Y_In,
///Parameter 5: double SaturationLevel_In,
///Parameter 6: double SignalThreshold_In,
///Parameter 7: double ApertureFWHM_In,
///Parameter 8: int NTermsGaussFit_In,
///Parameter 9: int PolyFitOrder_In,
///Parameter 10: int NLost_In,
///Parameter 11: double XLow_In,
///Parameter 12: double XMin_In,
///Parameter 13: double XHigh_In,
///Parameter 14: double XMax_In,
///Parameter 15: char[] DataBaseFileName_Out[,
///Parameter 16: FitsFileName_Out=(char[] FitsFileName_Out)[,
///Parameter 17: FitsFileName_CentersMarked_Out=(char[] FitsFileName_CentersMarked_Out)]]
#endif
