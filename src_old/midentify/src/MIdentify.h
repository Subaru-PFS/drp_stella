/*
author: Andreas Ritter
created: 05/08/2012
last edited: 05/08/2012
compiler: g++ 4.4
basis machine: Arch Linux
*/

#ifndef __MIDENTIFY_H__
#define __MIDENTIFY_H__

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
/// USAGE: identify char[] FitsFileName_Ec_In, char[] TextFileName_EcD_Out, char[] TextFileName_Coeffs_Out, char[] LineList_In, int Radius_In, double FWHM_In, int Order_In
#endif
