/*
author: Andreas Ritter
created: 12/06/2013
last edited: 12/06/2013
compiler: g++ 4.4
basis machine: Arch Linux
*/

#include "MImRotate.h"

using namespace std;

int main(int argc, char *argv[])
{
  cout << "MImRotate::main: argc = " << argc << endl;
  if (argc < 4)
  {
    cout << "MImRotate::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: imrotate <char[] (@)FitsFileName_In> <int I_AngleClockwise[0,90,180,270]> <char[] (@)FitsFileName_Out>" << endl;
    exit(EXIT_FAILURE);
  }

  char *P_CharArr_Op1_In = (char*)argv[1];
  cout << "MImRotate::main: P_CharArr_Op1_In set to " << P_CharArr_Op1_In << endl;

  int I_AngleClockwise = atoi((char*)argv[2]);
  cout << "MImRotate::main: I_AngleClockwise set to " << I_AngleClockwise << endl;

  char *P_CharArr_Out = (char*)argv[3];
  cout << "MImRotate::main: P_CharArr_Out set to " << P_CharArr_Out << endl;

  /// read parameters
  CString CS_FitsFileName_In;
  CS_FitsFileName_In.Set(P_CharArr_Op1_In);
  cout << "MImRotate::main: CS_FitsFileName_Op1_In set to " << CS_FitsFileName_In << endl;

  CString CS_FitsFileName_Out;
  CS_FitsFileName_Out.Set(P_CharArr_Out);
  cout << "MImRotate::main: CS_FitsFileName_Out set to " << CS_FitsFileName_Out << endl;

  Array<CString, 1> CS_A1_FitsFileNames_In(1);
  CS_A1_FitsFileNames_In(0) = CS_FitsFileName_In;
  Array<CString, 1> CS_A1_FitsFileNames_Out(1);
  CS_A1_FitsFileNames_Out(0) = CS_FitsFileName_Out;

  if (CS_FitsFileName_In.IsList()){
    if (!CS_FitsFileName_In.ReadFileLinesToStrArr(CS_FitsFileName_In, CS_A1_FitsFileNames_In)){
      cout << "MImRotate::main: ERROR: ReadFileLinesToStrArr(" << CS_FitsFileName_In << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    if (!CS_FitsFileName_Out.IsList()){
      cout << "MImRotate::main: ERROR: " << CS_FitsFileName_In << " is list, but " << CS_FitsFileName_Out << " is not" << endl;
      exit(EXIT_FAILURE);
    }
    if (!CS_FitsFileName_In.ReadFileLinesToStrArr(CS_FitsFileName_Out, CS_A1_FitsFileNames_Out)){
      cout << "MImRotate::main: ERROR: ReadFileLinesToStrArr(" << CS_FitsFileName_Out << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
  }
  if (CS_A1_FitsFileNames_In.size() != CS_A1_FitsFileNames_Out.size()){
    cout << "MImRotate::main: ERROR: " << CS_A1_FitsFileNames_In << " and " << CS_A1_FitsFileNames_Out << " have different numbers of elements" << endl;
    exit(EXIT_FAILURE);
  }

  CFits F_Image;
  for (int i_file = 0; i_file < CS_A1_FitsFileNames_In.size(); i_file++){
    cout << "MImRotate::main: Starting F_Image.SetFileName(" << CS_A1_FitsFileNames_In(i_file).Get() << ")" << endl;
    if (!F_Image.SetFileName(CS_A1_FitsFileNames_In(i_file)))
    {
      cout << "MImRotate::main: ERROR: F_Image.SetFileName(" << CS_A1_FitsFileNames_In(i_file).Get() << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    /// Read FitsFile
    cout << "MImRotate::main: Starting F_Image.ReadArray()" << endl;
    if (!F_Image.ReadArray())
    {
      cout << "MImRotate::main: ERROR: F_Image.ReadArray() returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    /// Write aperture header information
    CString CS_HeaderFileName("aphead");
    CS_HeaderFileName.Add(CS_A1_FitsFileNames_In(i_file));
    CS_HeaderFileName.Add(CString(".head"));
    F_Image.WriteApHead(CS_HeaderFileName);

    Array<double, 2> D_A2_PixArray(F_Image.GetNRows(), F_Image.GetNCols());
    D_A2_PixArray = F_Image.GetPixArray();

    int i_ncols=F_Image.GetNCols();
    int i_nrows=F_Image.GetNRows();
    if (I_AngleClockwise != 180){
      i_nrows = F_Image.GetNCols();
      i_ncols = F_Image.GetNRows();
      F_Image.SetNRows(D_A2_PixArray.cols());
      F_Image.SetNCols(D_A2_PixArray.rows());
    }
    Array<double, 2> D_A2_PixArrayNew(i_nrows, i_ncols);

    for (int i_row=0; i_row<D_A2_PixArray.rows(); i_row++){
      for (int i_col=0; i_col<D_A2_PixArray.cols(); i_col++){
        if (I_AngleClockwise == 90){
	  if (i_nrows - i_col - 1 < 0){
	    cout << "MImRotate: ERROR: i_nrows(=" << i_nrows << ") - i_col(=" << i_col << ") - 1 = " << i_nrows - i_col - 1 << " < 0" << endl;
	    exit(EXIT_FAILURE);
  	  }
  	  if (i_nrows - i_col - 1 >= i_nrows){
	    cout << "MImRotate: ERROR: i_nrows(=" << i_nrows << ") - i_col(=" << i_col << ") - 1 = " << i_nrows - i_col - 1 << " >= i_nrows = " << i_nrows << endl;
	    exit(EXIT_FAILURE);
	  }
	  if (i_row < 0){
	    cout << "MImRotate: ERROR: i_row(=" << i_row << ") < 0" << endl;
	    exit(EXIT_FAILURE);
	  }
	  if (i_row >= i_ncols){
	    cout << "MImRotate: ERROR: i_row(=" << i_row << ") >= i_ncols(=" << i_ncols << ")" << endl;
	    exit(EXIT_FAILURE);
	  }
	  D_A2_PixArrayNew(i_nrows - i_col - 1, i_row) = D_A2_PixArray(i_row, i_col);
//	cout << "MImRotate: D_A2_PixArrayNew(i_nrows - i_col - 1 =" << i_nrows - i_col - 1 << ", i_row=" << i_row << ") set to " << D_A2_PixArrayNew(i_nrows - i_col - 1, i_row) << endl;
        }
        else if (I_AngleClockwise == 180){
	  D_A2_PixArrayNew(i_nrows - i_row - 1, i_ncols - i_col - 1) = D_A2_PixArray(i_row, i_col);
        }
        else{// if (I_AngleClockwise == 270){
	  if (i_col >= i_nrows){
	    cout << "MImRotate: ERROR: i_col(=" << i_col << ") >= i_nrows(=" << i_nrows << " )" << endl;
	    exit(EXIT_FAILURE);
	  }
	  if (i_ncols - i_row - 1 < 0){
	    cout << "MImRotate: ERROR: i_ncols(=" << i_ncols << ") - i_row(=" << i_row << ") - 1 = " << i_ncols - i_row - 1 << " < 0" << endl;
	    exit(EXIT_FAILURE);
	  }
	  D_A2_PixArrayNew(i_col, i_ncols - i_row - 1) = D_A2_PixArray(i_row, i_col);
        }
      }
    }
    F_Image.GetPixArray() = D_A2_PixArrayNew;

    cout << "MImRotate::main: Starting F_Image.SetFileName(" << CS_A1_FitsFileNames_Out(i_file).Get() << ")" << endl;
    if (!F_Image.SetFileName(CS_A1_FitsFileNames_Out(i_file)))
    {
      cout << "MImRotate::main: ERROR: F_Image.SetFileName(" << CS_A1_FitsFileNames_Out(i_file).Get() << ") returned FALSE!" << endl;
      exit(EXIT_FAILURE);
    }

    ///write output file
    if (!F_Image.WriteArray()){
      cout << "MImRotate::main: ERROR: WriteArray() returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }

    if (!F_Image.AddHeaderToFitsFile(CS_A1_FitsFileNames_Out(i_file), CS_HeaderFileName)){
      cout << "MImRotate::main: ERROR: AddHeaderToFitsFile(" << CS_FitsFileName_Out << ", " << CS_HeaderFileName << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
  }
  /// clean up

  return EXIT_SUCCESS;
}

