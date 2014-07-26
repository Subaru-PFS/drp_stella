/*
author: Andreas Ritter
created: 03/20/2007
last edited: 03/20/2007
compiler: g++ 4.0
basis machine: Ubuntu Linux 6.06
*/

#ifndef __MMAKENORMFLAT_H__
#define __MMAKENORMFLAT_H__

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <iostream>
#include <cstdlib>

#include "../../cfits/src/CFits.h"
#include "../../cfits/src/CString.h"

using namespace std;

int main(int argc, char *argv[]);
///USAGE: makenormflat char[] FitsFileName_In, char[] DatabaseFileName_In, char[] NormalisedFlat_Out, char[] BlazeFits_Out, double Gain, double ReadOutNoise, int SmoothSP, double MinSNR[, int SwathWidth][,SMOOTH_SF=double][,AREA=[int(xmin),int(xmax),int(ymin),int(ymax)]][,char[] MASK_OUT]
#endif
