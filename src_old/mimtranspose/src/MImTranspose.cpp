/*
author: Andreas Ritter
created: 12/06/2013
last edited: 12/06/2013
compiler: g++ 4.4
basis machine: Arch Linux
*/

#include "MImTranspose.h"

using namespace std;

int main(int argc, char *argv[])
{
  cout << "MImTranspose::main: argc = " << argc << endl;
  if (argc < 3)
  {
    cout << "MImTranspose::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: imtranspose <char[] [@]FitsFileName_In> <char[] [@]FitsFileName_Out>" << endl;
    exit(EXIT_FAILURE);
  }

  char *P_CharArr_In = (char*)argv[1];
  cout << "MImTranspose::main: P_CharArr_Op1_In set to " << P_CharArr_In << endl;

  char *P_CharArr_Out = (char*)argv[2];
  cout << "MImTranspose::main: P_CharArr_Out set to " << P_CharArr_Out << endl;

  /// read parameters
  CString CS_FitsFileName_In;
  CS_FitsFileName_In.Set(P_CharArr_In);
  Array<CString, 1> CS_A1_FitsFileNames_In(1);
  CS_A1_FitsFileNames_In(0) = CS_FitsFileName_In;
  cout << "MImTranspose::main: CS_FitsFileName_Op1_In set to " << CS_FitsFileName_In << endl;

  CString CS_FitsFileName_Out;
  CS_FitsFileName_Out.Set(P_CharArr_Out);
  cout << "MImTranspose::main: CS_FitsFileName_Out set to " << CS_FitsFileName_Out << endl;
  Array<CString, 1> CS_A1_FitsFileNames_Out(1);
  CS_A1_FitsFileNames_Out(0) = CS_FitsFileName_Out;
  CString *P_CS;
  if (CS_FitsFileName_In.IsList()){
    P_CS = CS_FitsFileName_In.SubString(1);
    if (!P_CS->ReadFileLinesToStrArr(*P_CS, CS_A1_FitsFileNames_In)){
      cout << "MImTranspose::main: ERROR: ReadFileLinesToStrArr(" << *P_CS << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    if (!CS_FitsFileName_Out.IsList()){
      cout << "MImTranspose::main: ERROR: FileName_In is list but FileName_Out is not" << endl;
      exit(EXIT_FAILURE);
    }
    delete(P_CS);
    P_CS = CS_FitsFileName_Out.SubString(1);
    if (!P_CS->ReadFileLinesToStrArr(*P_CS, CS_A1_FitsFileNames_Out)){
      cout << "MImTranspose::main: ERROR: ReadFileLinesToStrArr(" << *P_CS << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    delete(P_CS);
  }

  CFits F_Image;

  for (int i_file = 0; i_file < CS_A1_FitsFileNames_In.size(); i_file++){
    cout << "MImTranspose::main: Starting F_Image.SetFileName(" << CS_A1_FitsFileNames_In(i_file).Get() << ")" << endl;
    if (!F_Image.SetFileName(CS_A1_FitsFileNames_In(i_file)))
    {
      cout << "MImTranspose::main: ERROR: F_Image.SetFileName(" << CS_A1_FitsFileNames_In(i_file).Get() << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    /// Read FitsFile
    cout << "MImTranspose::main: Starting F_Image.ReadArray()" << endl;
    if (!F_Image.ReadArray())
    {
      cout << "MImTranspose::main: ERROR: F_Image.ReadArray() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    /// Write aperture header information
    F_Image.WriteApHead(CString("aphead_")+CS_A1_FitsFileNames_In(i_file)+CString(".head"));

    Array<double, 2> D_A2_PixArray(F_Image.GetNRows(), F_Image.GetNCols());
    D_A2_PixArray = F_Image.GetPixArray();

    int i_nrows = F_Image.GetNCols();
    int i_ncols = F_Image.GetNRows();
    F_Image.SetNRows(D_A2_PixArray.cols());
    F_Image.SetNCols(D_A2_PixArray.rows());
    Array<double, 2> D_A2_PixArrayNew(i_nrows, i_ncols);

    for (int i_row=0; i_row<D_A2_PixArray.rows(); i_row++){
      for (int i_col=0; i_col<D_A2_PixArray.cols(); i_col++){
        D_A2_PixArrayNew(i_col, i_row) = D_A2_PixArray(i_row, i_col);
        cout << "MImTranspose: D_A2_PixArrayNew(i_col =" << i_col << ", i_row=" << i_row << ") set to " << D_A2_PixArrayNew(i_col, i_row) << endl;
      }
    }
    F_Image.GetPixArray() = D_A2_PixArrayNew;

    cout << "MImTranspose::main: Starting F_Image.SetFileName(" << CS_A1_FitsFileNames_Out(i_file).Get() << ")" << endl;
    if (!F_Image.SetFileName(CS_A1_FitsFileNames_Out(i_file)))
    {
      cout << "MImTranspose::main: ERROR: F_Image.SetFileName(" << CS_A1_FitsFileNames_Out(i_file).Get() << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    ///write output file
    if (!F_Image.WriteArray()){
      cout << "MImTranspose::main: ERROR: WriteArray() returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
  }
  /// clean up

  return EXIT_SUCCESS;
}

