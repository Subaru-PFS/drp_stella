/*
author: Andreas Ritter
created: 12/06/2013
last edited: 12/06/2013
compiler: g++ 4.4
basis machine: Arch Linux
*/

#include "MWriteFitsHeader.h"

using namespace std;

int main(int argc, char *argv[])
{
  cout << "MWriteFitsHeader::main: argc = " << argc << endl;
  if (argc < 3)
  {
    cout << "MWriteFitsHeader::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: writefitsheader <char[] FitsFileName> <char[] HeaderFileName_Out>0" << endl;
    exit(EXIT_FAILURE);
  }

  char *P_CharArr_In = (char*)argv[1];
  cout << "MWriteFitsHeader::main: P_CharArr_In set to " << P_CharArr_In << endl;

  char *P_CharArr_Out = (char*)argv[2];
  cout << "MWriteFitsHeader::main: P_CharArr_Out set to " << P_CharArr_Out << endl;

  /// read parameters
  CString CS_FitsFileName_In;
    CS_FitsFileName_In.Set(P_CharArr_In);
  cout << "MWriteFitsHeader::main: CS_FitsFileName_In set to " << CS_FitsFileName_In << endl;

  CString CS_HeaderFileName_Out;
    CS_HeaderFileName_Out.Set(P_CharArr_Out);
  cout << "MWriteFitsHeader::main: CS_HeaderFileName_Out set to " << CS_HeaderFileName_Out << endl;

  CFits F_Image;
  cout << "MWriteFitsHeader::main: Starting F_Image.SetFileName(" << CS_FitsFileName_In.Get() << ")" << endl;
  if (!F_Image.SetFileName(CS_FitsFileName_In))
  {
    cout << "MWriteFitsHeader::main: ERROR: F_Image.SetFileName(" << CS_FitsFileName_In.Get() << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  
  /// Read FitsFile
  cout << "MWriteFitsHeader::main: Starting F_Image.ReadArray()" << endl;
  if (!F_Image.ReadArray())
  {
    cout << "MWriteFitsHeader::main: ERROR: F_Image.ReadArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }
  
  /// Write aperture header information
  F_Image.WriteApHead(CS_HeaderFileName_Out);

  /// clean up

  return EXIT_SUCCESS;
}

