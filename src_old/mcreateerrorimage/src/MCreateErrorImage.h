/*
author: Andreas Ritter
created: 05/08/2012
last edited: 05/08/2012
compiler: g++ 4.4
basis machine: Arch Linux
*/

#ifndef __MCREATEERRORIMAGE_H__
#define __MCREATEERRORIMAGE_H__

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
