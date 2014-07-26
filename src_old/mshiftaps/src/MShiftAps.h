/*
author: Andreas Ritter
created: 03/20/2007
last edited: 03/20/2007
compiler: g++ 4.0
basis machine: Ubuntu Linux 6.06
*/

#ifndef __MSHIFTAPS_H__
#define __MSHIFTAPS_H__

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <iostream>
#include <cstdlib>

#include "../../cfits/src/CFits.h"
#include "../../cfits/src/CString.h"

using namespace std;

/** Task: 
 * reads FitsFileName_In and DatabaseFileName_In, shifts the apertures by ApertureShift pixels, 
 * and writes new aperture definitions to DatabaseFileName_Out
 **/
int main(int argc, char *argv[]);
//cout << "USAGE: shiftaps <char[] FitsFileName_In> <char[] DatabaseFileName_In> <double<>0 ApertureShift> //char[] DatabaseFileName_Out" << endl;
//cout << "MShiftAps: ApertureLength < 0: shift apertures to the left" << endl;
//cout << "MShiftAps: ApertureLength > 0: shift apertures to the right" << endl;
#endif
