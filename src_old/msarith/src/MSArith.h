/*
author: Andreas Ritter
created: 05/08/2012
last edited: 05/08/2012
compiler: g++ 4.4
basis machine: Arch Linux
*/

#ifndef __MSARITH_H__
#define __MSARITH_H__

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
/// USAGE: sarith <char[] [@]FitsFileName_Op1> <char Op> <char[] [@]FitsFileName_ArithBy_Op2> <char[] [@]FitsFileName_Out>
/// possible operators: '+' '-' '*' '/' 'sqrt' 'sum'
/// if operator == 'sqrt' or 'sum' then <char[] [@]FitsFileName_ArithBy_Op2> is ignored, but must be present
/// if operator == 'sum' then all files in @FitsFileName_Op1 will be summed up. In this case all input spectra should have the same size and be re-binned to the same wavelength

#endif
