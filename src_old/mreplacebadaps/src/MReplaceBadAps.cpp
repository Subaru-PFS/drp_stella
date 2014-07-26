/*
author: Andreas Ritter
created: 03/20/2007
last edited: 03/20/2007
compiler: g++ 4.0
basis machine: Ubuntu Linux 6.06
*/
#include "MReplaceBadAps.h"

int main(int argc, char *argv[])
    /// string input_fits_file_name: in, string input_database_file_name: in, string output_fits_file_name: out, string output_blaze_file_name: out, double Gain: in, double ReadOutNoise: in, int LambdaSP: in, double MinSNR: in[, int SwathWidth: in]
{
  cout << "MReplaceBadAps::main: argc = " << argc << endl;
  if (argc < 5)
  {
    cout << "MReplaceBadAps::main: ERROR: Not enough parameters specified!" << endl;
    cout << "USAGE: replacebadaps char[] FitsFileName_In, char[] DatabaseFileName_In, char[] FitsFileName_Out, [int(xmin),int(xmax),int(ymin),int(ymax)] " << endl;
    exit(EXIT_FAILURE);
  }
  char *P_CharArr_In = (char*)argv[1];
  char *P_CharArr_DB = (char*)argv[2];
  char *P_CharArr_Out = (char*)argv[3];
  char *P_CharArr_Area = (char*)argv[4];

  CString CS_FitsFileName_In;
  CS_FitsFileName_In.Set(P_CharArr_In);
  CString CS_FitsFileName_Out;
  CS_FitsFileName_Out.Set(P_CharArr_Out);
  CString CS_DatabaseFileName_In;
  CS_DatabaseFileName_In.Set(P_CharArr_DB);
  Array<int, 1> I_A1_Area(4);
  CString CS_Area;
  CS_Area.Set(P_CharArr_Area);
  CString CS_Temp;
  CString cs_temp;
  cs_temp.Set(",");
  CString *P_CS = new CString(" ");
  int i_pos_a = 1;
  int i_pos_b = CS_Area.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
  cout << "MReplaceBadAps: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
  delete(P_CS);
  P_CS = CS_Area.SubString(i_pos_a,i_pos_b-1);
  cout << "MReplaceBadAps: P_CS set to " << *P_CS << endl;
  I_A1_Area(0) = (int)(atoi(P_CS->Get()));
  cout << "MReplaceBadAps: I_A1_Area(0) set to " << I_A1_Area(0) << endl;

  i_pos_a = i_pos_b+1;
  i_pos_b = CS_Area.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
  cout << "MReplaceBadAps: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
  delete(P_CS);
  P_CS = CS_Area.SubString(i_pos_a,i_pos_b-1);
  cout << "MReplaceBadAps: P_CS set to " << *P_CS << endl;
  I_A1_Area(1) = (int)(atoi(P_CS->Get()));
  cout << "MReplaceBadAps: I_A1_Area(1) set to " << I_A1_Area(1) << endl;

  i_pos_a = i_pos_b+1;
  i_pos_b = CS_Area.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
  cout << "MReplaceBadAps: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
  delete(P_CS);
  P_CS = CS_Area.SubString(i_pos_a,i_pos_b-1);
  cout << "MReplaceBadAps: P_CS set to " << *P_CS << endl;
  I_A1_Area(2) = (int)(atoi(P_CS->Get()));
  cout << "MReplaceBadAps: I_A1_Area(2) set to " << I_A1_Area(2) << endl;

  i_pos_a = i_pos_b+1;
  cs_temp.Set("]");
  i_pos_b = CS_Area.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
  if (i_pos_b < 0){
    cs_temp.Set(")");
    i_pos_b = CS_Area.StrPosFrom(cs_temp.GetPChar(),i_pos_a+1);
  }
  cout << "MReplaceBadAps: i_pos_a set to " << i_pos_a << ", i_pos_b set to " << i_pos_b << endl;
  delete(P_CS);
  P_CS = CS_Area.SubString(i_pos_a,i_pos_b-1);
  cout << "MReplaceBadAps: P_CS set to " << *P_CS << endl;
  I_A1_Area(3) = (int)(atoi(P_CS->Get()));
  cout << "MReplaceBadAps: I_A1_Area(3) set to " << I_A1_Area(3) << endl;

  CFits F_Image;
  if (!F_Image.SetFileName(CS_FitsFileName_In))
  {
    cout << "MReplaceBadAps::main: ERROR: F_Image.SetFileName(" << CS_FitsFileName_In.Get() << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Read FitsFile
  if (!F_Image.ReadArray())
  {
    cout << "MReplaceBadAps::main: ERROR: F_Image.ReadArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Set DatabaseFileName_In
  if (!F_Image.SetDatabaseFileName(CS_DatabaseFileName_In))
  {
    cout << "MReplaceBadAps::main: ERROR: F_Image.SetDatabaseFileName(" << CS_DatabaseFileName_In << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Read DatabaseFileName_In
  if (!F_Image.ReadDatabaseEntry())
  {
    cout << "MReplaceBadAps::main: ERROR: F_Image.ReadDatabaseEntry() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Calculate Trace Functions
  if (!F_Image.CalcTraceFunctions())
  {
    cout << "MReplaceBadAps::main: ERROR: F_Image.CalcTraceFunctions() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  Array<double, 2> D_A2_PixArray(F_Image.GetNRows(), F_Image.GetNCols());
  D_A2_PixArray = F_Image.GetPixArray();
  Array<double, 1> *P_D_A1_YCenter = F_Image.Get_YCenter();
  Array<double, 1> *P_D_A1_YLow = F_Image.Get_YLow();
  Array<double, 1> *P_D_A1_YHigh = F_Image.Get_YHigh();
  Array<double, 1> *P_D_A1_XCenter = F_Image.Get_XCenter();
  Array<double, 1> *P_D_A1_XLow = F_Image.Get_XLow();
  Array<double, 1> *P_D_A1_XHigh = F_Image.Get_XHigh();
  Array<double, 2> *P_D_A2_XCenters = F_Image.Get_XCenters();

  /// Find bad apertures and replace with good ones
  Array<double, 3> D_A3_Aps(F_Image.Get_NApertures(), max(*P_D_A1_YHigh) - min(*P_D_A1_YLow) + 1, max(*P_D_A1_XHigh) - min(*P_D_A1_XLow) + 1);
  int i_pix_row, i_pix_col;
  CString CS_FileNameAp(" ");
  CString *P_CS_A;
  Array<double, 2> D_A2_Ap(max(*P_D_A1_YHigh) - min(*P_D_A1_YLow) + 1, max(*P_D_A1_XHigh) - min(*P_D_A1_XLow) + 1);
  for (int i_ap=0; i_ap<F_Image.Get_NApertures(); i_ap++){
    i_pix_row = 0;
    for (int i_row=(*P_D_A1_YCenter)(i_ap) + (*P_D_A1_YLow)(i_ap); i_row <= (*P_D_A1_YCenter)(i_ap) + (*P_D_A1_YHigh)(i_ap); i_row++){
      i_pix_col = 0;
      for (int i_col=(*P_D_A2_XCenters)(i_ap, i_row) + (*P_D_A1_XLow)(i_ap); i_col<=(*P_D_A2_XCenters)(i_ap, i_row) + (*P_D_A1_XHigh)(i_ap); i_col++){
        D_A3_Aps(i_ap, i_pix_row, i_pix_col) = D_A2_PixArray(i_row, i_col);
        cout << "MReplaceBadAps::main: i_row=" << i_row << ", i_col=" << i_col << ": D_A3_Aps(" << i_ap << ", " << i_pix_row << ", " << i_pix_col << ") set to " << D_A3_Aps(i_ap, i_pix_row, i_pix_col) << endl;
        i_pix_col++;
      }
      i_pix_row++;
    }
    delete(P_CS);
    P_CS = CS_FitsFileName_In.SubString(0,CS_FitsFileName_In.LastStrPos(CString("."))-1);
    *P_CS += CString("_");
    P_CS_A = P_CS->IToA(i_ap);
    *P_CS += *P_CS_A;
    delete(P_CS_A);
    *P_CS += CString(".fits");
    F_Image.WriteApToFile(D_A2_PixArray, i_ap, *P_CS);

    D_A2_Ap = D_A3_Aps(i_ap, Range::all(), Range::all());
    delete(P_CS);
    P_CS = CS_FitsFileName_In.SubString(0,CS_FitsFileName_In.LastStrPos(CString("."))-1);
    *P_CS += CString("_ap");
    P_CS_A = P_CS->IToA(i_ap);
    *P_CS += *P_CS_A;
    delete(P_CS_A);
    *P_CS += CString(".fits");
    F_Image.WriteFits(&D_A2_Ap, *P_CS);

  }

  Array<double, 1> D_A1_MeanAp(F_Image.Get_NApertures());
  double D_Mean_Aps;
  for (int i_ap=0; i_ap<F_Image.Get_NApertures(); i_ap++){
    D_A1_MeanAp(i_ap) = mean(D_A3_Aps(i_ap,Range::all(), Range::all()));
    cout << "MReplaceBadAps::main: i_ap = " << i_ap << ": D_A1_MeanAp = " << D_A1_MeanAp(i_ap) << endl;
  }
  D_Mean_Aps = mean(D_A1_MeanAp);
  cout << "MReplaceBadAps::main: D_Mean_Aps = " << D_Mean_Aps << endl;

  int I_XMin = I_A1_Area(0);
  int I_XMax = I_A1_Area(1);
  int I_YMin = I_A1_Area(2);
  int I_YMax = I_A1_Area(3);
  int i_ap_temp;
  double D_Mean;

  for (int i_ap=0; i_ap<F_Image.Get_NApertures(); i_ap++){
    if (((*P_D_A1_XCenter)(i_ap) >= I_XMin)
     && ((*P_D_A1_XCenter)(i_ap) <= I_XMax)
     && ((((*P_D_A1_YCenter)(i_ap)+(*P_D_A1_YLow)(i_ap) >= I_YMin)
       && ((*P_D_A1_YCenter)(i_ap)+(*P_D_A1_YLow)(i_ap) <= I_YMax))
      || (((*P_D_A1_YCenter)(i_ap)+(*P_D_A1_YHigh)(i_ap) >= I_YMin)
       && ((*P_D_A1_YCenter)(i_ap)+(*P_D_A1_YHigh)(i_ap) <= I_YMax)))){
      cout << "MReplaceBadAps::main: Aperture " << i_ap << " within Area[" << I_XMin << ", " << I_XMax << ", " << I_YMin << ", " << I_YMax << "]: D_A1_MeanAp(i_ap) = " << D_A1_MeanAp(i_ap) << endl;
      if (D_A1_MeanAp(i_ap) < D_Mean_Aps){
        cout << "MReplaceBadAps::main: D_A1_MeanAp(i_ap=" << i_ap << ")=" << D_A1_MeanAp(i_ap) << " < D_Mean_Aps=" << D_Mean_Aps << endl;
        D_Mean = 0;
        i_ap_temp = i_ap;
        do{
          i_ap_temp++;
          D_Mean = D_A1_MeanAp(i_ap_temp);
          cout << "MReplaceBadAps::main: D_A1_MeanAp(i_ap_temp=" << i_ap_temp << ") = " << D_A1_MeanAp(i_ap_temp) << endl;
        } while (D_Mean < D_Mean_Aps);
        i_pix_row = 0;
        for (int i_row=(*P_D_A1_YCenter)(i_ap) + (*P_D_A1_YLow)(i_ap); i_row<=(*P_D_A1_YCenter)(i_ap) + (*P_D_A1_YHigh)(i_ap); i_row++){
          i_pix_col = 0;
          for (int i_col=(*P_D_A2_XCenters)(i_ap, i_row)+(*P_D_A1_XLow)(i_ap); i_col<=(*P_D_A2_XCenters)(i_ap, i_row)+(*P_D_A1_XHigh)(i_ap); i_col++){
            F_Image.GetPixArray()(i_row, i_col) = D_A3_Aps(i_ap_temp, i_pix_row, i_pix_col);
            cout << "MReplaceBadAps::main: F_Image.GetPixArray()(i_row=" << i_row << ", i_col=" << i_col << ") set to D_A3_Aps(i_ap_temp=" << i_ap_temp << ", i_pix_row=" << i_pix_row << ", i_pix_col=" << i_pix_col << ") = " << D_A3_Aps(i_ap_temp, i_pix_row, i_pix_col) << endl;
            i_pix_col++;
          }
          i_pix_row++;
        }
      }
    }
  }

//  exit(EXIT_FAILURE);

  /// Set CS_FitsFileName_Out
  if (!F_Image.SetFileName(CS_FitsFileName_Out))
  {
    cout << "MReplaceBadAps::main: ERROR: F_Image.SetFileName(" << CS_FitsFileName_Out << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// Write NormFlat Image
  if (!F_Image.WriteArray())
  {
    cout << "MReplaceBadAps::main: ERROR: F_Image.WriteArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  delete(P_D_A1_XCenter);
  delete(P_D_A1_XHigh);
  delete(P_D_A1_XLow);
  delete(P_D_A1_YCenter);
  delete(P_D_A1_YHigh);
  delete(P_D_A1_YLow);
  delete(P_D_A2_XCenters);
  return EXIT_SUCCESS;
}
