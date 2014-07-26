/*
author: Andreas Ritter
created: 23/12/2011
last edited: 23/12/2011
compiler: g++ 4.0
basis machine: Ubuntu Linux 9.04
*/

#ifndef __MFINDCURVATURE_H__
#define __MFINDCURVATURE_H__

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
/// string input_fits_file_name: in,
/// string input_database_file_name: in,
/// string output_fits_file_name: out,
/// double Gain: in,
/// double ReadOutNoise: in,
/// int[0,1] Telluric: in,   --- subtract sky?
/// [, int SwathWidth: in]
#endif
