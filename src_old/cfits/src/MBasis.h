/*
author:        Andreas Ritter
created:       01/08/2007
last edited:   01/08/2007
compiler:      gcc 4.0
basis machine: Ubuntu Linux 6.06 LTS
*/
/*#ifndef __cplusplus
#define __cplusplus
#endif
#ifdef __cplusplus
extern "C" {
#endif
*/
#ifndef __AZURI_BASIS__
#define __AZURI_BASIS__

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef __DEBUG__
#define __DEBUG_MBASIS__
#endif

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

extern float truncf(float x);
extern float roundf(float x);

using namespace std;

namespace MBasis
{

  //#define _ISOC99_SOURCE

  long countlines(const char fname[255]);
  double calcintens(const double x1, const double x2, const double xn, const double y1, const double y2);
  double absd(double val);
  char* StrTrimChar(const char* P_Str, const char Char, int Mode);
  char* StrTrim(const char* P_Str, int Mode);
  char* SubString(const char*P_Str, int Start, int End);
  int strcatstr(char* inout, const char* strtoapp);
  int charrcat(char inoutarr[255], const char *strtoapp, int oldlength);
  int charposincharrfrom(const char inarr[255], const char lookfor, int start);
  int charposincharr(const char inarr[255], const char lookfor);
  int strposincharr(const char inarr[255], const char* lookfor);
  int lastcharposincharr(const char inarr[255], const char lookfor);
  int strposincharrfrom(const char inarr[255], const char* lookfor, int startpos);
  /**
  function int CharrPosInStr(char *P_InStr: in, const char LookFor: in)
  Returns first position of String 'LookFor' in 'P_InStr', looking from position Start, if found, else '-1'
   **/
  int CharPosInStrFrom(const char* P_InStr, const char LookFor, int Start);

  /**
  function int CharrPosInStr(char *P_InStr: in, const char LookFor: in)
  Returns first position of String 'P_LookFor' in 'P_InStr', if found, else '-1'
   **/
  int CharPosInStr(const char* P_InStr, const char LookFor);

  /**
  function int StrPosInStr(char *P_InStr: in, const char* P_LookFor: in)
  Returns first position of String 'P_LookFor' in 'P_InStr', if found, else '-1'
   **/
  int StrPosInStr(const char* P_InStr, const char* P_LookFor);

  /**
  function int LastCharPosInStr(char *P_InStr: in, const char LookFor: in)
  Returns last position of character 'P_LookFor' in 'Str', if found, else '-1'
   **/
  int LastCharPosInString(const string InStr, const char LookFor);

  /**
  function int LastCharPosInStr(char *P_InStr: in, const char LookFor: in)
  Returns last position of character 'P_LookFor' in 'P_InStr', if found, else '-1'
   **/
  int LastCharPosInStr(const char* P_InStr, const char LookFor);

  /**
  function int StrPosInStrFrom(char *P_InStr: in, char* P_LookFor: in, int Start)
  Returns first position of String 'P_LookFor' in 'P_InStr', looking from position Start, if found, else '-1'
   **/
  int StrPosInStrFrom(const char* P_InStr, const char* P_LookFor, int Start);
  char* StringToPChar(const string &str);
  string PCharToString(const char* p_str);
  int read2dfiletoarrays(const char* filename, double* warr, double* varr);
  int WriteArraysToFile(const char* P_FileName, const double* P_XArray, double* P_YArray, long Start, long End);
  int IsDouble(const char* P_Str);
  int IsInt(const char* P_Str);
  int IsNumber(const char* P_Str);
  double Mean(const double *P_X, const long XStart, const long XEnd);
  double Median(double *P_Arr, long Start, long End);
  double* CopyDblArray(const double *P_ArrToCopy, long Start, long End);
  void Swap(double &A,double &B);
  void BubbleSort(double *P_Arr,long Size);
  void ShellSort(double *P_Arr,long Size);
  int IfParameterSet(const char *P_Parameters, int SizeOfParameters, const char *P_ParameterName);
  double Round(const double ToRound, int DigitsBehindDot);
  long RoundToLong(const double ToRound);
  int* WhereEqual(const double* P_Arr, const int Size, const double Equal, int* NEqual);
  int* WhereGreater(const double* P_Arr, const int Size, const double Min, int* NGreater);
  int* WhereLower(const double* P_Arr, const int Size, const double Max, int* NLower);
  double** Total(const double*** PP_Arr, const int NDims, const int* P_Size, const int Dimension);
  void* Replicate(void* Content, int XSize, int YSize);
}
#endif
//#ifdef __cplusplus
//} /* closing brace for extern "C" */
//#endif
