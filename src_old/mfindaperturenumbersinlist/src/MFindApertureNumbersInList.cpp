/**
author: Andreas Ritter
created: 05/08/2012
last edited: 05/08/2012
compiler: g++ 4.4
basis machine: Arch Linux
*/

#include "MFindApertureNumbersInList.h"

using namespace std;

int main(int argc, char *argv[])
{
  cout << "MFindApertureNumbersInList::main: argc = " << argc << endl;
  if (argc < 4)
  {
    cout << "MFindApertureNumbersInList::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: findaperturenumbersinlist <char[] FileList_In> <char[] ApertureNumbers.list> <char[] FileList_Out>" << endl;
    exit(EXIT_FAILURE);
  }

  CString CS_FileList_In((char*)argv[1]);
  cout << "MFindApertureNumbersInList::main: CS_FileList_In set to " << CS_FileList_In << endl;

  CString CS_ApertureNumbersList_In((char*)argv[2]);
  cout << "MFindApertureNumbersInList::main: CS_ApertureNumbersList_In set to " << CS_ApertureNumbersList_In << endl;

  CString CS_FileList_Out((char*)argv[3]);
  cout << "MFindApertureNumbersInList::main: CS_FileList_Out set to " << CS_FileList_Out << endl;

  CFits F_Image;

  ///Read FileList_In
  Array<CString, 1> CS_A1_FileNames_In(1);
  if (!F_Image.ReadFileLinesToStrArr(CS_FileList_In, CS_A1_FileNames_In)){
    cout << "MFindApertureNumbersInList: ERROR: ReadFileLinesToStrArr(" << CS_FileList_In << ") returned FALSE" << endl;
    exit(EXIT_FAILURE);
  }

  ///Read ApertureNumbers
  Array<CString, 1> CS_A1_ApertureNumbers(1);
  if (!F_Image.ReadFileLinesToStrArr(CS_ApertureNumbersList_In, CS_A1_ApertureNumbers)){
    cout << "MFindApertureNumbersInList: ERROR: ReadFileLinesToStrArr(" << CS_ApertureNumbersList_In << ") returned FALSE" << endl;
    exit(EXIT_FAILURE);
  }

  CString *P_CS;
  CString CS;
  Array<CString, 1> CS_A1_FileNames_Out(1);
  CS_A1_FileNames_Out = CString(" ");
  Array<CString, 1> CS_A1_Temp(1);
  CS_A1_Temp = CString("");
  int I_Found = 0;
  for (int i_apnum=0; i_apnum<CS_A1_ApertureNumbers.size(); i_apnum++){
    for (int i_file=0; i_file<CS_A1_FileNames_In.size(); i_file++){
      CS.Set("_");
      P_CS = CS_A1_FileNames_In(i_file).SubString(CS_A1_FileNames_In(i_file).StrPos(CString("_ap"))+3, CS_A1_FileNames_In(i_file).StrPosFrom(CS.Get(), CS_A1_FileNames_In(i_file).StrPos(CString("_ap"))+2)-1);
//      cout << "MFindApertureNumbersInList::main: CS_A1_ApertureNumbers(i_apnum=" << i_apnum << ")=" << CS_A1_ApertureNumbers(i_apnum) << ": CS_A1_FileNames_In(i_file=" << i_file << ") P_CS = " << *P_CS << endl;
      if (P_CS->EqualValue(CS_A1_ApertureNumbers(i_apnum))){
        cout << "MFindApertureNumbersInList: i_apnum = " << i_apnum << ": Aperture Number " << CS_A1_ApertureNumbers(i_apnum) << " found in FileName " << CS_A1_FileNames_In(i_file) << endl;
        I_Found++;
        if (I_Found > 1){
          CS_A1_Temp.resize(I_Found-1);
          CS_A1_Temp = CS_A1_FileNames_Out;
          CS_A1_FileNames_Out.resize(I_Found);
          CS_A1_FileNames_Out(Range(0, I_Found-2)) = CS_A1_Temp;
          CS_A1_FileNames_Out(I_Found-1) = CS_A1_FileNames_In(i_file);
        }
        else{
          CS_A1_FileNames_Out(0) = CS_A1_FileNames_In(i_file);
        }
      }
      delete(P_CS);
    }
  }
  
  if (!CS.WriteStrListToFile(CS_A1_FileNames_Out, CS_FileList_Out)){
    cout << "MFindApertureNumbersInList::main: ERROR: WriteArrayToFile(CS_A1_FileNames_Out = " << CS_A1_FileNames_Out << ", " << CS_FileList_Out << ") returned FALSE" << endl;
    exit(EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}
