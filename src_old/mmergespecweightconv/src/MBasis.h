/*
author:        Andreas Ritter
created:       01/08/2007
last edited:   01/08/2007
compiler:      gcc 4.0
basis machine: Ubuntu Linux 6.06 LTS
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
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

extern float truncf(float x);
extern float roundf(float x);

//#define _ISOC99_SOURCE

/**
  function long CountLines(const char fname[255]: in)
  Returns number of lines in file <fname>, if <fname> exists, else returns 0.
 **/
long CountLines(const char fname[255]);

/**
  function int CalcIntens(const double x1: in, const double x2: in, const double xn: in, const double y1: in, const double y2: in)
  Interpolates intensity between two pixels 1 and 2.
 **/
double CalcIntens(const double D_X1_In, const double D_X2_In, const double D_XN_In, const double D_Y1_In, const double D_Y2_In);


/**
  function double AbsD(double D_Val_In: in)
  Returns absolute double value of D_Val_In.
 **/
double AbsD(double D_Val_In);

/**
  function char* StrTrimChar(const char* P_Str: in, int Mode: in)
  Mode == 0: Remove starting spaces
  Mode == 1: Remove trailing spaces
  Mode == 2: Remove starting and trailing spaces
  Removes empty leading and/or trailing <Char_In> from P_Str.
 **/
char* StrTrimChar(const char* P_Str_In, const char Char_In, int Mode_In);

/**
  function char* StrTrim(const char* P_Str: in, int I_Mode: in)
  Mode == 0: Remove starting spaces
  Mode == 1: Remove trailing spaces
  Mode == 2: Remove starting and trailing spaces
  Removes empty spaces from P_Str.
 **/
char* StrTrim(const char* P_Str_In, int I_Mode_In);

/**
  function char* SubString(const char *P_Str_In: in, int I_Start_In, int I_End_In: in)
  Returns P_Str_In[I_Start_In..I_End_In].
 **/
char* SubString(const char *P_Str_In, int I_Start_In, int I_End_In);

/**
  function int StrCatStr(char* inout: inout, char *strtoapp: in)
  Appends strtoapp at inout and a '\0' behind and returns position of '\0'.
 **/
int StrCatStr(char* P_C_InOut, const char* P_ChArrToApp);

/**
  function int ChArrCat(char ChArr_InOut[255]: inout, const char *P_ChArrToApp: in)
  Appends strtoapp at inout[I_OldLength] and a '\0' behind and returns position of '\0'.
 **/
int ChArrCat(char ChArr_InOut[255], const char *P_ChArrToApp, int I_OldLength);

/**
  function int CharPosInChArrFrom{CHARacter POSition IN CHar ARR}(char inarr[255]: in, char lookfor: in, int start: in)
  Returns first position of character 'in' in 'inarr', beginning at position 'start', if found, else '-1'
 **/
int CharPosInChArrFrom(const char ChArr_In[255], const char C_LookFor_In, int I_Start_In);

/**
  function int CharPosInChArr{CHARacter POSition IN CHar ARR}(char inarr[255]: in, char lookfor: in, int start: in)
  Returns first position of character 'in' in 'inarr', if found, else '-1'
 **/
int CharPosInChArr(const char ChArr_In[255], const char C_LookFor_In);

/**
  function int StrPosInChArr{STRing POSition IN CHar ARR}(char inarr[255]: in, char* lookfor: in)
  Returns first position of string 'in' in 'inarr', if found, else '-1'
 **/
int StrPosInChArr(const char ChArr_In[255], const char* P_C_LookFor_In);

/**
  function int LastCharPosInChArr{LAST CHARacter POSition IN CHar ARR}(char inarr[255]: in, char lookfor: in)
  Returns last position of character 'in' in 'inarr', if found, else '-1'
 **/
int LastCharPosInChArr(const char ChArr_In[255], const char C_LookFor_In);

/**
  function int StrPosInChArrFrom{LAST STRing POSition IN CHar ARR}(char inarr[255]: in, char* lookfor: in, int startpos)
  Returns last position of character 'in' in 'inarr', if found, else '-1'
 **/
int StrPosInChArrFrom(const char ChArr_In[255], const char* P_C_LookFor_In, int I_Start_In);

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
  Returns last position of character 'P_LookFor' in 'P_InStr', if found, else '-1'
 **/
int LastCharPosInStr(const char* P_InStr, const char LookFor);

/**
  function int StrPosInStrFrom(char *P_InStr: in, char* P_LookFor: in, int Start)
  Returns first position of String 'P_LookFor' in 'P_InStr', looking from position Start, if found, else '-1'
 **/
int StrPosInStrFrom(const char* P_InStr, const char* P_LookFor, int Start);

/**
  function int Read2DFileToArrays(const char* P_FileName: in, const double* P_XArray: in, const double* P_YArray)
  Returns 1, if successfull, else 0.
 **/
int Read2DFileToArrays(const char* filename, double* warr, double* varr);

/**
  function int WriteArraysToFile(const char* P_FileName: in, const double* P_XArray: in, const double* P_YArray)
  Returns 1, if successfull, else 0.
 **/
int WriteArraysToFile(const char* P_FileName, const double* P_XArray,
                      double* P_YArray, long Start, long End);

/**
  function int IsDouble(const char* P_Str: in)
  Returns 1, if P_Str is a double, else 0.
 **/
int IsDouble(const char* P_Str);

/**
  function int IsInt(const char* P_Str: in)
  Returns 1, if P_Str is an integer, else 0.
 **/
int IsInt(const char* P_Str);

/**
  function int IsNumber(const char* P_Str: in)
  Returns 1, if P_Str is a double or an integer, else 0.
 **/
int IsNumber(const char* P_Str);

/**
  function double Mean(const double *P_D_Arr: in, long L_Start, long L_End)
  Returns mean of subarray P_D_Arr[L_Start..L_End].
 **/
double Mean(const double *P_D_Arr, long L_Start, long L_End);

/**
  function double Median(const double *P_D_Arr: in, long L_Start, long L_End)
  Returns median of subarray P_D_Arr[L_Start..L_End].
 **/
double Median(const double *P_D_Arr, long L_Start, long L_End);

/**
  function double* CopyDblArray(const double *P_D_Arr: in, long L_Start, long L_End)
  Returns copy of subarray P_D_ArrToCopy[L_Start..L_End].
 **/
double* CopyDblArray(const double *P_D_ArrToCopy, long L_Start, long L_End);

/**
  function double Swap(double *P_D_A: inout, double *P_D_B: inout)
  Swaps values of P_D_A and P_D_B.
 **/
void Swap(double *P_D_A, double *P_D_B);

/**
  function double BubbleSort(double *P_D_Arr: inout, long L_Size)
  Sorts values of array of doubles P_D_Arr.
 **/
void BubbleSort(double *P_D_Arr, long Size);

/**
  function double Round(double D_ToRound: in, int I_DigitsBehindDot: in)
  Returns rounded double value of D_ToRound with I_DigitsBehindDot digits behind the dot.
 **/
double Round(const double D_ToRound, int I_DigitsBehindDot);

/**
  function double RoundToLong(double D_ToRound: in)
  Returns rounded long value of D_ToRound.
 **/
long RoundToLong(const double ToRound);
/*
int* WhereEqual(const double* P_Arr, const int Size, const double Equal, int* NEqual);

int* WhereGreater(const double* P_Arr, const int Size, const double Min, int* NGreater);

int* WhereLower(const double* P_Arr, const int Size, const double Max, int* NLower);

double** Total(const double*** PP_Arr, const int NDims, const int* P_Size, const int Dimension);

void* Replicate(void* Content, int XSize, int YSize);*/

//int IfParameterSet(const char *P_Parameters, int SizeOfParameters, const char *P_ParameterName);
#endif
