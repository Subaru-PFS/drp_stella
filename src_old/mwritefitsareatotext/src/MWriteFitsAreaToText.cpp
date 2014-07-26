/*
author: Andreas Ritter
created: 03/20/2007
last edited: 03/20/2007
compiler: g++ 4.0
basis machine: Ubuntu Linux 6.06
*/

#include "MWriteFitsAreaToText.h"

int main(int argc, char *argv[])
{
  cout << "MWriteFitsAreaToText::main: argc = " << argc << endl;
  if (argc < 7)
  {
    cout << "MWriteFitsAreaToText::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: writefitsareatotext <char[] [@]FitsFileName_In> <char[] [@]TextFileName_Out> <int XMin> <int XMax> <int YMin> <int YMax>" << endl;
    exit(EXIT_FAILURE);
  }
  CString CS_FileName_In((char*)argv[1]);
  CString CS_FileName_Out((char*)argv[2]);
  int I_XMin = atoi((char*)argv[3]);
  cout << "MWriteFitsAreaToText: I_XMin = " << I_XMin << endl;
  int I_XMax = atoi((char*)argv[4]);
  cout << "MWriteFitsAreaToText: I_XMax = " << I_XMax << endl;
  int I_YMin = atoi((char*)argv[5]);
  cout << "MWriteFitsAreaToText: I_YMin = " << I_YMin << endl;
  int I_YMax = atoi((char*)argv[6]);
  cout << "MWriteFitsAreaToText: I_YMax = " << I_YMax << endl;
  int i_xmin, i_xmax, i_ymin,i_ymax;
  
  CString CS_FitsFileName_In(" ");
  CString CS_FitsFileName_Out(" ");

  CFits F_Image;
  
  CString *P_CS_FirstLetter = CS_FileName_In.SubString(0,0);
  cout << "MWriteFitsAreaToText::main: *P_CS_FirstLetter = <" << *P_CS_FirstLetter << ">" << endl;
//  exit(EXIT_FAILURE);
  CString *P_CS_Temp;
  CString CS_Comp("@");
  bool B_InputIsList = false;
  Array<CString, 1> CS_A1_InputFileNames(1);
  Array<CString, 1> CS_A1_OutputFileNames(1);
  if (P_CS_FirstLetter->EqualValue(CS_Comp)){
    delete(P_CS_FirstLetter);
    P_CS_FirstLetter = CS_FileName_Out.SubString(0,0);
    if (!P_CS_FirstLetter->EqualValue(CS_Comp)){
      cout << "MWriteFitsAreaToText::main: ERROR: FileName_In is list but FileName_Out is not" << endl;
      exit(EXIT_FAILURE);
    }
    delete(P_CS_FirstLetter);
    P_CS_Temp = CS_FileName_In.SubString(1);
    CS_FileName_In.Set(*P_CS_Temp);
    delete(P_CS_Temp);
    P_CS_Temp = CS_FileName_Out.SubString(1);
    CS_FileName_Out.Set(*P_CS_Temp);
    delete(P_CS_Temp);
    if (!CS_Comp.ReadFileLinesToStrArr(CS_FileName_In, CS_A1_InputFileNames)){
      cout << "MWriteFitsAreaToText::main: ERROR: ReadFileLinesToStrArr(" << CS_FileName_In << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    if (!CS_Comp.ReadFileLinesToStrArr(CS_FileName_Out, CS_A1_OutputFileNames)){
      cout << "MWriteFitsAreaToText::main: ERROR: ReadFileLinesToStrArr(" << CS_FileName_Out << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
  }
  else{
    CS_A1_InputFileNames(0) = CS_FileName_In;
    CS_A1_OutputFileNames(0) = CS_FileName_Out;
  }

  Array<double, 2> D_A2_PixArray_In(2, 2);
  Array<double, 2> D_A2_PixArray_Out(2, 2);
  for (int i_file=0; i_file < CS_A1_InputFileNames.size(); i_file++){
    i_xmin = I_XMin;
    i_xmax = I_XMax;
    i_ymin = I_YMin;
    i_ymax = I_YMax;
    if (!F_Image.SetFileName(CS_A1_InputFileNames(i_file)))
    {
      cout << "MWriteFitsAreaToText::main: ERROR: F_Image.SetFileName(" << CS_A1_InputFileNames(i_file).Get() << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
    cout << "MWriteFitsAreaToText::main: FileName <" << CS_A1_InputFileNames(i_file).Get() << "> set" << endl;

    /// Read FitsFile
    if (!F_Image.ReadArray())
    {
      cout << "MWriteFitsAreaToText::main: ERROR: F_Image.ReadArray() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }
    cout << "MWriteFitsAreaToText::main: F_Image: Array read" << endl;

    D_A2_PixArray_In.resize(F_Image.GetNRows(), F_Image.GetNCols());
    D_A2_PixArray_In = F_Image.GetPixArray();

    if (i_xmin < 0)
      i_xmin = 0;
    if (i_xmin >= F_Image.GetNCols()){
      cout << "MWriteFitsAreaToText::main: ERROR: i_xmin(=" << i_xmin << ") >= F_Image.cols()=" << F_Image.GetNCols() << endl;
      exit(EXIT_FAILURE);
    }
    if (i_xmax > F_Image.GetNCols())
      i_xmax = F_Image.GetNCols()-1;
    if (i_xmax < i_xmin){
      cout << "MWriteFitsAreaToText::main: ERROR: i_xmax(=" << i_xmax << ") < i_xmin(=" << i_xmin << ")" << endl;
      exit(EXIT_FAILURE);
    }
    if (i_ymin < 0)
      i_ymin = 0;
    if (i_ymin >= F_Image.GetNRows()){
      cout << "MWriteFitsAreaToText::main: ERROR: i_ymin(=" << i_ymin << ") >= F_Image.rows()=" << F_Image.GetNRows() << endl;
      exit(EXIT_FAILURE);
    }
    if (i_ymax > F_Image.GetNRows())
      i_ymax = F_Image.GetNRows()-1;
    if (i_ymax < i_ymin){
      cout << "MWriteFitsAreaToText::main: ERROR: i_ymax(=" << i_ymax << ") < i_ymin(=" << i_ymin << ")" << endl;
      exit(EXIT_FAILURE);
    }
    D_A2_PixArray_Out.resize(i_ymax - i_ymin + 1, i_xmax - i_xmin + 1);
    D_A2_PixArray_Out(Range(0,i_ymax - i_ymin), Range(0,i_xmax - i_xmin)) = D_A2_PixArray_In(Range(i_ymin, i_ymax), Range(i_xmin, i_xmax));
    cout << "MWriteFitsAreaToText::main: i_xmin = " << i_xmin << ", i_xmax = " << i_xmax << ", i_ymin = " << i_ymin << ", i_ymax = " << i_ymax << endl;
    //cout << "MWriteFitsAreaToText::main: D_A2_PixArray_Out = " << D_A2_PixArray_Out << endl;
    
    if (!F_Image.WriteArrayToFile(D_A2_PixArray_Out, CS_A1_OutputFileNames(i_file), CString("ascii"))){
      cout << "MWriteFitsAreaToText::main: ERROR: WriteArrayToFile(D_A2_PixArray_Out = " << D_A2_PixArray_Out << ", " << CS_A1_OutputFileNames(i_file) << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
  }  
  return EXIT_SUCCESS;
}
