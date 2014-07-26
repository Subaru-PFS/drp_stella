/*
author: Andreas Ritter
created: 05/08/2012
last edited: 05/08/2012
compiler: g++ 4.4
basis machine: Arch Linux
*/

#ifndef __MIDENTIFYLIST_H__
#define __MIDENTIFYLIST_H__

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <iostream>
#include <cstdlib>
#include <time.h>

//#include "python2.7/Python.h"

#include "../../cfits/src/CAny.h"
#include "../../cfits/src/CString.h"
#include "../../cfits/src/CFits.h"

//#define __DEBUG_IDENTIFYLIST__

using namespace std;

int main(int argc, char *argv[]);
/// USAGE: identify char[] FitsFileName_Ec_List_In, char[] FitsFileName_Ref_In, char[] TextFileName_EcD_List_Out, char[] TextFileName_Coeffs_List_Out, char[] LineList_In, int Radius_In, double FWHM_In, int Order_In
#endif
