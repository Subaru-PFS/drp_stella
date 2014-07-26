/*
author: Andreas Ritter
created: 05/08/2012
last edited: 05/08/2012
compiler: g++ 4.4
basis machine: Arch Linux
*/

#include "MDispCorList.h"

using namespace std;

int main(int argc, char *argv[])
{
  cout << "MDispCorList::main: argc = " << argc << endl;
  if (argc < 5)
  {
    cout << "MDispCorList::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: dispcorlist <char[] FitsFileName_Ec_List_In> <char[] TextFileName_Coeffs_List_In> <char[] TextFileName_EcD_List_Out> <double D_MaxRMS_In>" << endl;
    exit(EXIT_FAILURE);
  }

  char *P_CharArr_FitsFiles_Ec_List_In = (char*)argv[1];
  cout << "MDispCorList::main: P_CharArr_FitsFiles_Ec_List_In set to " << P_CharArr_FitsFiles_Ec_List_In << endl;

  char *P_CharArr_TextFiles_Coeffs_List_In = (char*)argv[2];
  cout << "MDispCorList::main: P_CharArr_TextFiles_Coeffs_List_In set to " << P_CharArr_TextFiles_Coeffs_List_In << endl;

  char *P_CharArr_TextFiles_EcD_List_Out = (char*)argv[3];
  cout << "MDispCorList::main: P_CharArr_TextFiles_EcD_List_Out set to " << P_CharArr_TextFiles_EcD_List_Out << endl;

  char *P_CharArr_MaxRMS_In = (char*)argv[4];
  cout << "MDispCorList::main: P_CharArr_MaxRMS_In set to " << P_CharArr_MaxRMS_In << endl;

  /// read parameters
  CString CS_FitsFileNames_Ec_List_In;
  CS_FitsFileNames_Ec_List_In.Set(P_CharArr_FitsFiles_Ec_List_In);
  cout << "MDispCorList::main: CS_FitsFileNames_Ec_List_In set to " << CS_FitsFileNames_Ec_List_In << endl;

  CString CS_TextFileNames_Coeffs_List_In;
  CS_TextFileNames_Coeffs_List_In.Set(P_CharArr_TextFiles_Coeffs_List_In);
  cout << "MDispCorList::main: CS_TextFileNames_Coeffs_List_In set to " << CS_TextFileNames_Coeffs_List_In << endl;

  CString CS_TextFileNames_EcD_List_Out;
  CS_TextFileNames_EcD_List_Out.Set(P_CharArr_TextFiles_EcD_List_Out);
  cout << "MDispCorList::main: CS_TextFileNames_EcD_List_Out set to " << CS_TextFileNames_EcD_List_Out << endl;

  double D_MaxRMS_In = double(atof(P_CharArr_MaxRMS_In));
  cout << "MDispCorList::main: D_MaxRMS_In set to " << D_MaxRMS_In << endl;

  CFits F_Image, F_ImageRef;

  /// Read FileNames
  Array<CString, 1> CS_A1_FitsFiles_Ec_In(1);
  cout << "MDispCorList::main: starting ReadFileLinesToStrArr(" << CS_FitsFileNames_Ec_List_In << "), CS_A1_FitsFiles_Ec_In" << endl;
  if (!F_Image.ReadFileLinesToStrArr(CS_FitsFileNames_Ec_List_In, CS_A1_FitsFiles_Ec_In)){
    cout << "MDispCorList::main: ERROR: ReadFileLinesToStrArr(" << CS_FitsFileNames_Ec_List_In << ") returned FALSE" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MDispCorList::main: CS_A1_FitsFiles_Ec_In set to " << CS_A1_FitsFiles_Ec_In << endl;

  Array<CString, 1> CS_A1_TextFiles_Coeffs_In(1);
  if (!F_Image.ReadFileLinesToStrArr(CS_TextFileNames_Coeffs_List_In, CS_A1_TextFiles_Coeffs_In)){
    cout << "MDispCorList::main: ERROR: ReadFileLinesToStrArr(" << CS_TextFileNames_Coeffs_List_In << ") returned FALSE" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MDispCorList::main: CS_A1_TextFiles_Coeffs_In set to " << CS_A1_TextFiles_Coeffs_In << endl;

  Array<CString, 1> CS_A1_TextFiles_EcD_Out(1);
  if (!F_Image.ReadFileLinesToStrArr(CS_TextFileNames_EcD_List_Out, CS_A1_TextFiles_EcD_Out)){
    cout << "MDispCorList::main: ERROR: ReadFileLinesToStrArr(" << CS_TextFileNames_EcD_List_Out << ") returned FALSE" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "MDispCorList::main: CS_A1_TextFiles_EcD_Out set to " << CS_A1_TextFiles_EcD_Out << endl;

  if (CS_A1_FitsFiles_Ec_In.size() != CS_A1_TextFiles_EcD_Out.size()){
    cout << "MDispCorList::main: ERROR: CS_A1_FitsFiles_Ec_In.size() != CS_A1_TextFiles_EcD_Out.size()" << endl;
    exit(EXIT_FAILURE);
  }

  /// read aperture number of CS_A1_TextFiles_Coeffs_In
  Array<int, 1> I_A1_ApNums_Coeffs(CS_A1_TextFiles_Coeffs_In.size());
  CString *P_CS_ApNum;
  CString CS_UnderScore("_");
  int I_StrStart, I_StrEnd;
  for (int i=0; i < CS_A1_TextFiles_Coeffs_In.size(); i++){
    I_StrStart = (CS_A1_TextFiles_Coeffs_In(i)).StrPos(CString("_ap"))+3;
    if (I_StrStart < 4){
      cout << "MDispCorList::main: i=" << i << ": ERROR: aperture number not found in filename " << CS_A1_TextFiles_Coeffs_In(i) << endl;
      exit(EXIT_FAILURE);
    }
//    cout << "MDispCorList::main: i=" << i << ": I_StrStart = " << I_StrStart << endl;

    I_StrEnd = (CS_A1_TextFiles_Coeffs_In(i)).StrPosFrom(CS_UnderScore.GetPChar(),I_StrStart)-1;
    if (I_StrEnd < 4){
      cout << "MDispCorList::main: i=" << i << ": ERROR: aperture number not found in filename " << CS_A1_TextFiles_Coeffs_In(i) << endl;
      exit(EXIT_FAILURE);
    }
//    cout << "MDispCorList::main: i=" << i << ": I_StrEnd = " << I_StrEnd << endl;

    P_CS_ApNum = (CS_A1_TextFiles_Coeffs_In(i)).SubString(I_StrStart, I_StrEnd);
    I_A1_ApNums_Coeffs(i) = atoi(P_CS_ApNum->GetPChar());
    delete(P_CS_ApNum);
//    cout << "MDispCorList::main: i=" << i << ": I_A1_ApNums_Coeffs(i) set to " << I_A1_ApNums_Coeffs(i) << endl;
  }
//  cout << "MDispCorList::main: I_A1_ApNums_Coeffs set to " << I_A1_ApNums_Coeffs << endl;

  Array<CString, 1> CS_A1_Coeffs(2);
  int I_ApNum, I_CoeffFileNo, I_NInd;
  Array<double, 1> D_A1_Spec(1);
  Array<double, 2> D_A2_SpecCalib_Out(2,2);
  Array<double, 1> D_A1_PolyFitCoeffs_In(2);
  double D_RMS_In;
  Array<int, 1> I_A1_Ind(I_A1_ApNums_Coeffs.size());
  Array<int, 1> *P_I_A1_IndPos;
  CString *P_CS_RMS;
  Array<double, 1> *P_D_A1_PixNum;
  Array<double, 1> *P_D_A1_WLen;
  Array<CString, 1> CS_A1_Spec(1);

  for (int i_file=0; i_file < CS_A1_FitsFiles_Ec_In.size(); i_file++){
    /// Read spectrum
    if (!F_Image.SetFileName(CS_A1_FitsFiles_Ec_In(i_file))){
      cout << "MDispCorList::main: i_file=" << i_file << ": ERROR: F_Image.SetFileName(" << CS_A1_FitsFiles_Ec_In(i_file) << ") returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }
    if (!F_Image.ReadArray()){
      cout << "MDispCorList::main: ERROR: F_Image.ReadArray() returned FALSE" << endl;
      exit(EXIT_FAILURE);
    }

    /// Read F_Image.PixArray to Array<double, 1>
    if (F_Image.GetNRows() == 1){
      D_A1_Spec.resize(F_Image.GetNCols());
      D_A1_Spec = (F_Image.GetPixArray())(0,Range::all());
    }
    else if (F_Image.GetNCols() == 1){
      D_A1_Spec.resize(F_Image.GetNRows());
      D_A1_Spec = (F_Image.GetPixArray())(Range::all(),0);
    }
    else{
      cout << "MDispCorList::main: ERROR: Input spectrum not one-dimensional" << endl;
      exit(EXIT_FAILURE);
    }
//    cout << "MDispCorList::main: D_A1_Spec set to " << D_A1_Spec << endl;

    /// read aperture number of CS_A1_FitsFiles_Ec_In
    I_StrStart = (CS_A1_FitsFiles_Ec_In(i_file)).StrPos(CString("_ap"))+3;
    if (I_StrStart < 4){
      cout << "MDispCorList::main: i_file=" << i_file << ": ERROR: aperture number not found in filename " << CS_A1_FitsFiles_Ec_In(i_file) << endl;
      exit(EXIT_FAILURE);
    }
//    cout << "MDispCorList::main: i_file=" << i_file << ": I_StrStart = " << I_StrStart << endl;

    I_StrEnd = (CS_A1_FitsFiles_Ec_In(i_file)).StrPosFrom(CS_UnderScore.GetPChar(),I_StrStart)-1;
    if (I_StrEnd < 4){
      cout << "MDispCorList::main: i_file=" << i_file << ": ERROR: aperture number not found in filename " << CS_A1_FitsFiles_Ec_In(i_file) << endl;
      exit(EXIT_FAILURE);
    }
//    cout << "MDispCorList::main: i_file=" << i_file << ": I_StrEnd = " << I_StrEnd << endl;

    P_CS_ApNum = (CS_A1_FitsFiles_Ec_In(i_file)).SubString(I_StrStart, I_StrEnd);
    I_ApNum = atoi(P_CS_ApNum->GetPChar());
    delete(P_CS_ApNum);
//    cout << "MDispCorList::main: i_file=" << i_file << ": I_ApNum set to " << I_ApNum << endl;

    /// check aperture number of coefficients files (I_A1_ApNums_Coeffs) for I_ApNum
    I_A1_Ind = where(I_A1_ApNums_Coeffs == I_ApNum,1,0);
    P_I_A1_IndPos = F_Image.GetIndex(I_A1_Ind, I_NInd);
//    cout << "MDispCorList::main: i_file = " << i_file << " :P_I_A1_IndPos = " << *P_I_A1_IndPos << endl;
    if (I_NInd != 1){
      cout << "MDispCorList::main: i_file = " << i_file << ": I_ApNum = " << I_ApNum << ": WARNING: I_ApNum not found in coefficients files" << endl;
      delete(P_I_A1_IndPos);
    }
    else{
      I_CoeffFileNo = (*P_I_A1_IndPos)(0);
      delete(P_I_A1_IndPos);
      /// Read Polynomial coefficients
//      cout << "MDispCorList::main: i_file=" << i_file << ": Reading coeff file " << CS_A1_TextFiles_Coeffs_In(I_CoeffFileNo) << endl;
      if (!F_Image.ReadFileLinesToStrArr(CS_A1_TextFiles_Coeffs_In(I_CoeffFileNo), CS_A1_Coeffs)){
        cout << "MDispCorList::main: WARNING: ReadFileLinesToStrArr(" << CS_TextFileNames_EcD_List_Out << ") returned FALSE" << endl;
      }
      else{
//        cout << "MDispCorList::main: i_file=" << i_file << ": CS_A1_Coeffs = " << CS_A1_Coeffs << endl;

        /// populate D_A1_PolyFitCoeffs_In array
        D_A1_PolyFitCoeffs_In.resize(CS_A1_Coeffs.size()-1);
        if (!CS_FitsFileNames_Ec_List_In.AToD(CS_A1_Coeffs(Range(0,CS_A1_Coeffs.size()-2)), D_A1_PolyFitCoeffs_In)){
          cout << "MDispCorList::main: i_file = " << i_file << ": ERROR: Could not convert CS_A1_Coeffs to double array" << endl;
          exit(EXIT_FAILURE);
        }
//        cout << "MDispCorList::main: i_file = " << i_file << ": D_A1_PolyFitCoeffs_In set to " << D_A1_PolyFitCoeffs_In << endl;

        /// Read RMS
        P_CS_RMS = (CS_A1_Coeffs(CS_A1_Coeffs.size()-1)).SubString(4);
        D_RMS_In = P_CS_RMS->AToD();
        delete(P_CS_RMS);
        cout << "MDispCorList::main: i_file = " << i_file << ": D_RMS_In = " << D_RMS_In << endl;

        if ((D_RMS_In <= D_MaxRMS_In) && (D_RMS_In >= 0.000000001)){
//          P_D_A1_PixNum = F_Image.DIndGenArr(D_A1_Spec.size());
//          P_D_A1_WLen = F_Image.Poly(*P_D_A1_PixNum, D_A1_PolyFitCoeffs_In);
//          D_A2_SpecCalib_Out.resize(P_D_A1_WLen->size(),2);
//          D_A2_SpecCalib_Out(Range::all(),0) = *P_D_A1_WLen;
//          D_A2_SpecCalib_Out(Range::all(),1) = D_A1_Spec;
          if (!F_Image.DispCor(D_A1_Spec, D_A1_PolyFitCoeffs_In, D_A2_SpecCalib_Out)){
            cout << "MDispCorList::main: ERROR: DispCor returned FALSE" << endl;
            exit(EXIT_FAILURE);
          }
//          cout << "MDispCorList::main: i_file = " << i_file << ": D_A2_SpecCalib_Out = " << D_A2_SpecCalib_Out << endl;
          cout << "MDispCorList::main: Writing " << CS_A1_TextFiles_EcD_Out(i_file) << endl;
          if (!F_Image.WriteArrayToFile(D_A2_SpecCalib_Out, CS_A1_TextFiles_EcD_Out(i_file), CString("ascii"))){
            cout << "MDispCorList::main: i_file = " << i_file << ": Error: Could not write " << CS_A1_TextFiles_EcD_Out(i_file) << endl;
            exit(EXIT_FAILURE);
          }
//          delete(P_D_A1_PixNum);
//          delete(P_D_A1_WLen);
        }
      }
    }
  }

  /// clean up

  return EXIT_SUCCESS;
}

