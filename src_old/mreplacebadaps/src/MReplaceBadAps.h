/*
author: Andreas Ritter
created: 03/20/2007
last edited: 03/20/2007
compiler: g++ 4.0
basis machine: Ubuntu Linux 6.06
*/

#ifndef __MREPLACEBADAPS_H__
#define __MREPLACEBADAPS_H__

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <iostream>
#include <cstdlib>

#include "../../cfits/src/CFits.h"
#include "../../cfits/src/CString.h"

using namespace std;

int main(int argc, char *argv[]);
/// USAGE: replacebadaps char[] FitsFileName_In, char[] DatabaseFileName_In, char[] FitsFileName_Out, [int(xmin),int(xmax),int(ymin),int(ymax)]
#endif
