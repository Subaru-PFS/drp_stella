/*
author: Andreas Ritter
created: 01/12/2007
last edited: 01/12/2007
compiler: g++ 4.0
basis machine: Ubuntu Linux 6.06
*/

#ifndef __MTESTAPP_H__
#define __MTESTAPP_H__

#define _DEBUG_TESTAPP_

#include <stdio.h>
#include <string>
#include "nrutil.h"
#include <fstream>
#include "CString.h"
#include "CFormatedString.h"
#include "CFits.h"
#include <random/normal.h>
#include <random/chisquare.h>
#include <random/F.h>

using std::string;
using std::cin;
using std::cout;
using std::endl;
using namespace ranlib;

namespace MTestApp{

  ofstream *P_Log;

  int main(int argc, char *argv[]);
  bool Run(CString &cn);

  bool TestCStringConstructors();
  bool TestCString();
  bool TestCFormatedString();
  bool TestCFSConstructors();
  bool TestCFSCalcFSAndEqualOpAndEVal();
  bool TestCFSClassInvAndCopyAndEVal();

  bool TestCFits();
  bool TestCFitsConstructors();
  bool TestCFitsClassInvAndEValAndCopy();
  bool TestCFitsMedian();
  bool TestCFitsReformMultInvert();
  bool TestCFitsInterPolUniqCeil();
  bool TestCFitsPiskunov();
  bool TestCFitsMkProf();
  bool TestCFitsMkScatter();
  bool TestCFitsCrossCorrelate();
  bool TestCFitsFindAndTrace();
}
#endif
