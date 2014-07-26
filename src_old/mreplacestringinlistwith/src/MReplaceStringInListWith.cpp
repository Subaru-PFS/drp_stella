/*
author: Andreas Ritter
created: 05/08/2012
last edited: 05/08/2012
compiler: g++ 4.4
basis machine: Arch Linux
*/

#include "MReplaceStringInListWith.h"

using namespace std;

int main(int argc, char *argv[])
{
  cout << "MReplaceStringInListWith::main: argc = " << argc << endl;
  if (argc < 5)
  {
    cout << "MReplaceStringInListWith::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: replacestringinlistwith <char[] TextList_In> <char[] TextList_Out> <char[] StringToReplace> <char[] StringReplacement>" << endl;
    exit(EXIT_FAILURE);
  }

  char *P_CharArr_TextList_In = (char*)argv[1];
  cout << "MReplaceStringInListWith::main: P_CharArr_TextList_In set to " << P_CharArr_TextList_In << endl;

  char *P_CharArr_TextList_Out = (char*)argv[2];
  cout << "MReplaceStringInListWith::main: P_CharArr_TextList_Out set to " << P_CharArr_TextList_Out << endl;

  char *P_CharArr_StringToReplace = (char*)argv[3];
  cout << "MReplaceStringInListWith::main: P_CharArr_StringToReplace set to " << P_CharArr_StringToReplace << endl;

  char *P_CharArr_StringReplacement = (char*)argv[4];
  cout << "MReplaceStringInListWith::main: P_CharArr_StringReplacement set to " << P_CharArr_StringReplacement << endl;

  /// read parameters
  CString CS_TextList_In;
  CS_TextList_In.Set(P_CharArr_TextList_In);
  cout << "MReplaceStringInListWith::main: CS_TextList_In set to " << CS_TextList_In << endl;

  CString CS_TextList_Out;
  CS_TextList_Out.Set(P_CharArr_TextList_Out);
  cout << "MReplaceStringInListWith::main: CS_TextList_Out set to " << CS_TextList_Out << endl;

  CString CS_StringToReplace;
  CS_StringToReplace.Set(P_CharArr_StringToReplace);
  cout << "MReplaceStringInListWith::main: CS_StringToReplace set to " << CS_StringToReplace << endl;

  CString CS_StringReplacement;
  CS_StringReplacement.Set(P_CharArr_StringReplacement);
  cout << "MReplaceStringInListWith::main: CS_StringReplacement set to " << CS_StringReplacement << endl;

  if (!CS_TextList_In.StrReplaceInList(CS_TextList_In, CS_StringToReplace, CS_StringReplacement, CS_TextList_Out)){
    cout << "MReplaceStringInListWith::main: ERROR: StrReplaceInList(" << CS_TextList_In << ", " << CS_StringToReplace << ", " << CS_StringReplacement << ", " << CS_TextList_Out << ") returned FALSE" << endl;
    exit(EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}

