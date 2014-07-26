/*
author: Andreas Ritter
created: 05/08/2012
last edited: 05/08/2012
compiler: g++ 4.4
basis machine: Arch Linux
*/

#include "MCreateErrorImage.h"

using namespace std;

int main(int argc, char *argv[])
{
  cout << "MCreateErrorImage::main: argc = " << argc << endl;
  if (argc < 4)
  {
    cout << "MCreateErrorImage::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: createerrorimage char[] FitsFileName_In, char[] FitsFileName_Out, double RdNoise, double Gain[, CCDSEC=[x1,x2,y1,y2]" << endl;
    exit(EXIT_FAILURE);
  }
  Array<int, 1> I_A1_CCDSec(4);

  CString CS(" ");
  CString CS_comp(" ");
  CString *P_CS = new CString(" ");
  CString *P_CS_In = new CString(" ");
  CString *P_CS_ErrOut = new CString(" ");

  char *P_CharArr_In = (char*)argv[1];
  char *P_CharArr_Out = (char*)argv[2];
  double D_RdNoise = (double)(atof((char*)argv[3]));
//  double D_SNoise = (double)(atof((char*)argv[4]));
  double D_Gain = (double)(atof((char*)argv[4]));

  /// read optional parameters
  bool B_CCDSec_Set = false;
  for (int i = 4; i <= argc; i++){
    CS.Set((char*)argv[i]);
    cout << "MCreateErrorImage: Reading Parameter " << CS << endl;

    /// AREA
    CS_comp.Set("CCDSEC");
    if (CS.GetLength() > CS_comp.GetLength()){
      delete(P_CS);
      P_CS = CS.SubString(0,CS.CharPos('=')-1);
      cout << "MCreateErrorImage::main: *P_CS set to " << *P_CS << endl;
      if (P_CS->EqualValue(CS_comp)){
        B_CCDSec_Set = true;
        CString cs_temp;
        cs_temp.Set(",");
        int i_pos_a = CS_comp.GetLength()+2;
        int i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        cout << "MCreateErrorImage: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
        delete(P_CS);
        P_CS = CS.SubString(i_pos_a,i_pos_b-1);
        cout << "MCreateErrorImage: P_CS set to " << *P_CS << endl;
        I_A1_CCDSec(0) = (int)(atoi(P_CS->Get()));
        cout << "MCreateErrorImage: I_A1_CCDSec(0) set to " << I_A1_CCDSec(0) << endl;

        i_pos_a = i_pos_b+1;
        i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        cout << "MCreateErrorImage: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
        delete(P_CS);
        P_CS = CS.SubString(i_pos_a,i_pos_b-1);
        cout << "MCreateErrorImage: P_CS set to " << *P_CS << endl;
        I_A1_CCDSec(1) = (int)(atoi(P_CS->Get()));
        cout << "MCreateErrorImage: I_A1_CCDSec(1) set to " << I_A1_CCDSec(1) << endl;

        i_pos_a = i_pos_b+1;
        i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        cout << "MCreateErrorImage: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
        delete(P_CS);
        P_CS = CS.SubString(i_pos_a,i_pos_b-1);
        cout << "MCreateErrorImage: P_CS set to " << *P_CS << endl;
        I_A1_CCDSec(2) = (int)(atoi(P_CS->Get()));
        cout << "MCreateErrorImage: I_A1_CCDSec(2) set to " << I_A1_CCDSec(2) << endl;

        i_pos_a = i_pos_b+1;
        cs_temp.Set("]");
        i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        if (i_pos_b < 0){
          cs_temp.Set(")");
          i_pos_b = CS.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
        }
        cout << "MCreateErrorImage: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
        delete(P_CS);
        P_CS = CS.SubString(i_pos_a,i_pos_b-1);
        cout << "MCreateErrorImage: P_CS set to " << *P_CS << endl;
        I_A1_CCDSec(3) = (int)(atoi(P_CS->Get()));
        cout << "MCreateErrorImage: I_A1_CCDSec(3) set to " << I_A1_CCDSec(3) << endl;
      }
    }
  }
  time_t seconds;
//  if (argc == 8)
//  {
//    I_SwathWidth = (int)(atoi((char*)argv[7]));
//    cout << "MCreateErrorImage::main: I_SwathWidth set to " << I_SwathWidth << endl;
  CString CS_FitsFileName_In;
  CS_FitsFileName_In.Set(P_CharArr_In);
  CString CS_FitsFileName_Out;
  CS_FitsFileName_Out.Set(P_CharArr_Out);

  CFits F_Image;
  cout << "MCreateErrorImage::main: Starting F_Image.SetFileName(" << CS_FitsFileName_In.Get() << ")" << endl;
  if (!F_Image.SetFileName(CS_FitsFileName_In))
  {
    cout << "MCreateErrorImage::main: ERROR: F_Image.SetFileName(" << CS_FitsFileName_In.Get() << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Read FitsFile
  cout << "MCreateErrorImage::main: Starting F_Image.ReadArray()" << endl;
  if (!F_Image.ReadArray())
  {
    cout << "MCreateErrorImage::main: ERROR: F_Image.ReadArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  if (!B_CCDSec_Set){
    I_A1_CCDSec(0) = 0;
    I_A1_CCDSec(1) = F_Image.GetNCols()-1;
    I_A1_CCDSec(2) = 0;
    I_A1_CCDSec(3) = F_Image.GetNRows()-1;
  }

  /// Write aperture header information
  F_Image.WriteApHead(CString("aphead_")+CS_FitsFileName_In+CString(".head"));

  CFits F_OutImage;
  /// Set CS_FitsFileName_In
  cout << "MCreateErrorImage::main: Starting F_OutImage.SetFileName(" << CS_FitsFileName_In << ")" << endl;
  if (!F_OutImage.SetFileName(CS_FitsFileName_In))
  {
    cout << "MCreateErrorImage::main: ERROR: F_OutImage.SetFileName(" << CS_FitsFileName_In << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  ///Read FitsFile
  cout << "MCreateErrorImage::main: Starting F_OutImage.ReadArray()" << endl;
  if (!F_OutImage.ReadArray())
  {
    cout << "MCreateErrorImage::main: ERROR: F_OutImage.ReadArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Set CS_FitsFileName_Out
  cout << "MCreateErrorImage::main: Starting F_OutImage.SetFileName(" << CS_FitsFileName_Out << ")" << endl;
  if (!F_OutImage.SetFileName(CS_FitsFileName_Out))
  {
    cout << "MCreateErrorImage::main: ERROR: F_OutImage.SetFileName(" << CS_FitsFileName_Out << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// change size of F_OutImage to (NApertures x NRows)
  if (!F_OutImage.SetNCols(I_A1_CCDSec(1)-I_A1_CCDSec(0)+1))
  {
    cout << "MCreateErrorImage::main: ERROR: F_OutImage.SetNCols(" << I_A1_CCDSec(1)-I_A1_CCDSec(0)+1 << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  if (!F_OutImage.SetNRows(I_A1_CCDSec(3)-I_A1_CCDSec(2)+1))
  {
    cout << "MCreateErrorImage::main: ERROR: F_OutImage.SetNRows(" << I_A1_CCDSec(3)-I_A1_CCDSec(2)+1 << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Calculate Uncertainties
  F_OutImage.GetPixArray() = sqrt(fabs(((F_Image.GetPixArray())(Range(I_A1_CCDSec(0),I_A1_CCDSec(1)),
					 		        Range(I_A1_CCDSec(2),I_A1_CCDSec(3)))) * D_Gain) + pow2(D_RdNoise));
  if (!F_OutImage.WriteArray())
  {
    cout << "MCreateErrorImage::main: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Write aperture header information
  F_Image.WriteApHead(CString("aphead_")+CS_FitsFileName_In+CString(".head"));

  cout << "MCreateErrorImage::main: Starting F_OutImage.WriteArray()" << endl;
  if (!F_OutImage.WriteArray())
  {
    cout << "MCreateErrorImage::main: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  delete(P_CS);
  delete(P_CS_In);
  delete(P_CS_ErrOut);
  return EXIT_SUCCESS;
}
