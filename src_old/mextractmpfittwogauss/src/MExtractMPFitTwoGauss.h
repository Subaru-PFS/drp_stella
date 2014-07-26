/*
author: Andreas Ritter
created: 07/10/2013
last edited: 07/10/2013
compiler: g++ 4.4
basis machine: Arch Linux
*/

#ifndef __MEXTRACTMPFITTWOGAUSS_H__
#define __MEXTRACTMPFITTWOGAUSS_H__

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
/// USAGE: extractsum char[] FitsFileName_In, char[] DatabaseFileName_In, char[] FitsFileName_Out[, ERR_IN=char[]][, ERR_OUT_EC=char[]][,AREA=[int(xmin),int(xmax),int(ymin),int(ymax)]]
#endif
