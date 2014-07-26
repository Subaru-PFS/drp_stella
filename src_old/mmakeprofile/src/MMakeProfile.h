/*
author: Andreas Ritter
created: 01/12/2007
last edited: 01/12/2007
compiler: g++ 4.0
basis machine: Ubuntu Linux 6.06
*/

#ifndef __MMAKEPROFILE_H__
#define __MMAKEPROFILE_H__

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <iostream>
#include <cstdlib>

#include "../../cfits/src/CFits.h"
#include "../../cfits/src/CString.h"

using namespace std;

int main(int argc, char *argv[]);
/// string input_fits_file_name: in,
/// string input_database_file_name: in,
/// string output_fits_file_name: out,
/// double Gain: in,
/// double ReadOutNoise: in,
/// int[0,1] Telluric
/// [, int SwathWidth: in]
#endif
