/*
author: Andreas Ritter
created: 30/07/2012
last edited: 30/07/2012
compiler: g++ 4.0
basis machine: Arch Linux
*/

#ifndef __MSUBSCATTERKRIGING_H__
#define __MSUBSCATTERKRIGING_H__

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <iostream>
#include <cstdlib>

#include "../../cfits/src/CFits.h"
#include "../../cfits/src/CString.h"

using namespace std;

int main(int argc, char *argv[]);
/// USAGE: subscatterkriging char[] FitsFileName_In, char[] DatabaseFileName_In, char[] FitsFileName_Out, int[] ClusterSizes_In, int AddNPixToAp_X, int AddNPixToAp_Y, int NRectangles_X, int NRectangles_Y[, FileName_ApZero_Out=FileName_ApZero_Out][, FileName_Scatter_Out=FileName_Scatter_Out][,FileName_Clustered_Out=FileName_Clustered_Out]
#endif
