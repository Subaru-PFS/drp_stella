/*
author: Andreas Ritter
created: 17/01/2014
last edited: 17/01/2014
compiler: g++ 4.8
basis machine: Arch Linux
*/

#ifndef __MXCOR_H__
#define __MXCOR_H__

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
/// USAGE: createerrorimage char[] FitsFileName_In, char[] FitsFileName_Out, double RdNoise, double Gain[, CCDSEC=[x1,x2,y1,y2]
#endif
