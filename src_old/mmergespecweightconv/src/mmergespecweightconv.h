/*
author:        Andreas Ritter
created:       01/08/2007
last edited:   01/08/2007
compiler:      gcc 4.0
basis machine: Ubuntu Linux 6.06 LTS
*/

#ifndef __MERGESPECWEIGHTCONV_H__
#define __MERGESPECWEIGHTCONV_H__

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef __DEBUG__
#define __DEBUG_MERGESPECWEIGHTCONV__
#endif

//  #include <stdio.h>
  #include <math.h>
  #include "MBasis.h"

  double calcnoisefromoverlap(const double* warr, const double* varra, const double* varrb, const long starta, const long enda, const long endb, const char* out);
  int main(int argc, char *argv[]);

#endif
