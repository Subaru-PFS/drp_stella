/*
author: Andreas Ritter
created: 04/12/2007
last edited: 05/05/2007
compiler: g++ 4.0
basis machine: Ubuntu Linux 6.06
*/

#ifndef __MSUBTRACTOVERSCANANDTRIM_H__
#define __MSUBTRACTOVERSCANANDTRIM_H__

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
