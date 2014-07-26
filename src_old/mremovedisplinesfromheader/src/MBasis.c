/*
author:        Andreas Ritter
created:       01/08/2007
last edited:   01/08/2007
compiler:      gcc 4.0
basis machine: Ubuntu Linux 6.06 LTS
*/

#include "MBasis.h"

/**
function int charrcat{CHar Array CAT}(char inoutarr[255]: inout, char *strtoapp: in, int OLDLENGTHofinoutarr: in)
Appends strtoapp at inoutarr[oldlength] and a '\0' behind and returns position of '\0'.
 **/
long CountLines(const char fname[255])
{
#ifdef __DEBUG_MBASIS__
  //  fprintf(logfile,"mbasis.CountLines: method started\n");
#endif
  FILE *ffile;
  long nelements;
  char oneword[255];
  char *line;

#ifdef __DEBUG_MBASIS__
  printf("mbasis.CountLines: function started\n");
#endif
  ffile = fopen(fname, "r");
  if (ffile == NULL)
  {
    printf("mbasis.CountLines: Failed to open file fname (=<%s>)\n", fname);
//    exit (EXIT_FAILURE);
    return 0;
  }
#ifdef __DEBUG_MBASIS__
  printf("mbasis.CountLines: File fname(=<%s>) opened\n", fname);
#endif

  nelements = 0;
  // --- read file <fname> named <ffile>
  do
  {
    line = fgets(oneword, 255, ffile);
    if (line != NULL)
    {
#ifdef __DEBUG_MBASIS__
      //      printf("mbasis.CountLines: oneword = <%s>\n", oneword);
#endif
      // --- count lines
      nelements++;
    }
  }
  while (line != NULL);
#ifdef __DEBUG_MBASIS__
  printf("mbasis.CountLines: File fname(=<%s>) contains %d data lines\n",fname,nelements);
#endif
  // --- close input file
  fclose(ffile);
#ifdef __DEBUG_MBASIS__
  printf("mbasis.CountLines: File fname (=<%s>) closed\n", fname);
#endif
  return nelements;
}

/**
function int charrcat{CHar Array CAT}(char inoutarr[255]: inout, char *strtoapp: in, int OLDLENGTHofinoutarr: in)
Appends strtoapp at inoutarr[oldlength] and a '\0' behind and returns position of '\0'.
 **/
double CalcIntens(const double x1, const double x2, const double xn, const double y1, const double y2)
{
  double intensn = 0.;
  intensn = y1;
  intensn += ((y2 - y1) * (xn - x1)/(x2 - x1));
  return intensn;
}

/**
function int charrcat{CHar Array CAT}(char inoutarr[255]: inout, char *strtoapp: in, int OLDLENGTHofinoutarr: in)
Appends strtoapp at inoutarr[oldlength] and a '\0' behind and returns position of '\0'.
 **/
double AbsD(double val)
{
  if (val >= 0)
    return val;
  return (0. - val);
}

/**
function char* StrTrim(const char* P_Str: in, int Mode: in)
Mode == 0: Remove starting spaces
Mode == 1: Remove trailing spaces
Mode == 2: Remove starting and trailing spaces
Removes empty spaces of P_Str.
 **/
char* StrTrimChar(const char* P_Str, const char Char, int Mode)
{
  int i;
  int NewLength = 0;
  int Before = 1;
  //  int Behind = 1;
  char *P_NewStr;
  int Size = strlen(P_Str);

#ifdef __DEBUG_MBASIS__
  printf("MBasis.StrTrim: P_Str = %s, strlen(P_Str) = %d, Size = %d, Mode = %d\n", P_Str, strlen(P_Str), Size, Mode);
#endif
  P_NewStr = (char*)malloc(sizeof(char) * (Size + 1));
  if (P_NewStr == NULL)
  {
    printf("MBasis.StrTrim: ERROR: Cannot allocate memory for P_NewStr!\n");
    exit(EXIT_FAILURE);
  }
#ifdef __DEBUG_MBASIS__
  printf("MBasis.StrTrim: Memory for P_NewStr allocated: strlen(P_NewStr) returns %d, Size = %d\n", strlen(P_NewStr), Size);
#endif
  if (Mode == 0 || Mode == 2)
  {
#ifdef __DEBUG_MBASIS__
    printf("MBasis.StrTrim: Mode == 0 || 2, strlen(P_NewStr=%s) = %d\n", P_NewStr, strlen(P_NewStr));
#endif
    for (i = 0; i < Size; i++)
    {
#ifdef __DEBUG_MBASIS__
      printf("MBasis.StrTrim: i = %d: P_Str[i] = <%c>\n", i, P_Str[i]);
#endif
      if (Before == 1 && P_Str[i] == Char)// || (Before == 0 && Mode < 2))
      {
#ifdef __DEBUG_MBASIS__
        printf("MBasis.StrTrim: Skipping Character P_Str[i=%d] = <%c>\n", i, P_Str[i]);
#endif
      }
      else
      {
        Before = 0;
        P_NewStr[NewLength] = P_Str[i];
        NewLength++;
        P_NewStr[NewLength] = '\0';
#ifdef __DEBUG_MBASIS__
        printf("MBasis.StrTrim: NewLength = %d: P_NewStr = <%s>\n", NewLength, P_NewStr);
#endif
      }
      if (P_Str[i] == '\0')
        i = Size;
    }
  }
  else
  {
    P_NewStr = (char*)malloc(sizeof(char) * (strlen(P_Str) + 1));
    strcpy(P_NewStr, P_Str);
  }
  if (Mode == 1 || Mode == 2)
  {
#ifdef __DEBUG_MBASIS__
    printf("MBasis.StrTrim: Mode == 1 || 2, strlen(P_NewStr=%s) = %d\n", P_NewStr, strlen(P_NewStr));
#endif
    for (i = strlen(P_NewStr) - 1; i > 0; i--)
    {
#ifdef __DEBUG_MBASIS__
      printf("MBasis.StrTrim: P_NewStr[i=%d] = <%c>\n", i, P_NewStr[i]);
#endif
      if (P_NewStr[i] == Char)
      {
#ifdef __DEBUG_MBASIS__
        printf("MBasis.StrTrim: Removing Character P_Str[i=%d] = <%c>\n", i, P_Str[i]);
#endif
        P_NewStr[i] = '\0';
      }
      else
      {
        i = 0;
      }
#ifdef __DEBUG_MBASIS__
      printf("MBasis.StrTrim: i = %d: P_NewStr = <%s>\n", i, P_NewStr);
#endif
    }
  }

  return P_NewStr;
}

/**
function int charrcat{CHar Array CAT}(char inoutarr[255]: inout, char *strtoapp: in, int OLDLENGTHofinoutarr: in)
Appends strtoapp at inoutarr[oldlength] and a '\0' behind and returns position of '\0'.
 **/
char* StrTrim(const char* P_Str, int Mode)
{
  return StrTrimChar(P_Str, ' ', Mode);
}

/**
function int charrcat{CHar Array CAT}(char inoutarr[255]: inout, char *strtoapp: in, int OLDLENGTHofinoutarr: in)
Appends strtoapp at inoutarr[oldlength] and a '\0' behind and returns position of '\0'.
 **/
char* SubString(const char*P_Str, int Start, int End)
{
  int i, Pos, Length;
  char* P_OutStr;

  if (Start < 0)
    Start = 0;

  P_OutStr = (char*)malloc(sizeof(char) * (End - Start + 1));

  Pos = 0;
  for (i = Start; i <= End; i++)
  {
    P_OutStr[Pos] = P_Str[i];
    Pos++;
    if (P_Str[i] == '\0')
      i = End;
  }
  P_OutStr[Pos] = '\0';
  return P_OutStr;
}

/**
function int charrcat{CHar Array CAT}(char inoutarr[255]: inout, char *strtoapp: in, int OLDLENGTHofinoutarr: in)
Appends strtoapp at inoutarr[oldlength] and a '\0' behind and returns position of '\0'.
 **/
int ChArrCat(char inoutarr[255], const char *strtoapp, int oldlength)
{
#ifdef __DEBUG_MBASIS__
  printf("mbasis.charcat: function started: intoutarr = %s, strtoapp = %s, oldlength =%d\n", inoutarr, strtoapp, oldlength);
#endif
  int newlength, i;
  newlength = oldlength;
  if (oldlength + strlen(strtoapp) > 255)
  {
#ifdef __DEBUG_MBASIS__
    printf("mbasis.charcat: oldlength(=%d) + strlen(strtoapp=%s)(=%d) > 255 => returning 0\n", oldlength, strtoapp, strlen(strtoapp));
#endif
    return 0;
  }
  for (i = 0; i < strlen(strtoapp); i++)
  {
    inoutarr[i+oldlength] = strtoapp[i];
    //#ifdef __DEBUG_MBASIS__
    //    printf("mbasis.charrcat: inoutarr = %s\n",inoutarr);
    //#endif

  }
  inoutarr[oldlength + strlen(strtoapp)] = '\0';
  newlength += strlen(strtoapp);
  return newlength;
}

/**
function int strcat{STRing CAT}(char* inout: inout, char *strtoapp: in)
Appends strtoapp at inout and a '\0' behind and returns position of '\0'.
 **/
int StrCatStr(char* inout, const char* strtoapp)
{
#ifdef __DEBUG_MBASIS__
  printf("mbasis.strcatstr: function started: inout = %s, strtoapp = %s\n", inout, strtoapp);
#endif

  int oldlength, newlength, i;
  char TempCharArr[255];

  oldlength = strlen(inout);
  newlength = strlen(inout);

  if (newlength + strlen(strtoapp) > 255)
  {
#ifdef __DEBUG_MBASIS__
    printf("mbasis.strcatstr: newlength(=%d) + strlen(strtoapp=%s)(=%d) > 255 => returning 0\n", newlength, strtoapp, strlen(strtoapp));
#endif
    return 0;
  }

  for (i = 0; i < strlen(inout); i++)
  {
    TempCharArr[i] = inout[i];
  }
  for (i = 0; i < strlen(strtoapp); i++)
  {
    TempCharArr[i+newlength] = strtoapp[i];
    //#ifdef __DEBUG_MBASIS__
    //    printf("mbasis.charrcat: inoutarr = %s\n",inoutarr);
    //#endif

  }
  newlength += strlen(strtoapp);
  TempCharArr[newlength] = '\0';
  inout[0] = '\0';
  for (i = 0; i <= newlength; i++)
  {
    inout[i] = TempCharArr[i];
  }
#ifdef __DEBUG_MBASIS__
  printf("mbasis.strcatstr: newlength =%d strlen(inout=%s)(=%d)\n", newlength, inout, strlen(inout));
#endif
  return 1;

}

/**
function int CharPosInChArrFrom{CHARacter POSition IN CHar ARR}(char inarr[255]: in, char lookfor: in, int start: in)
Returns first position of character 'in' in 'inarr', beginning at position 'start', if found, else '-1'
 **/
int CharPosInChArrFrom(const char inarr[255], const char lookfor, int start)
{
  int pos = start;
  do
  {
    if (inarr[pos] == '\0')
      return -1;
    if (inarr[pos] == lookfor)
    {
#ifdef __DEBUG_MBASIS__
      printf("mbasis.CharPosInChArrFrom: inarr[pos=%d] == lookfor(=%c) returned TRUE\n", pos, lookfor);
#endif
      break;
    }
    pos++;
  }
  while (pos < 255);
  if (pos == 255)
    pos = -1;
#ifdef __DEBUG_MBASIS__
  printf("mbasis.CharPosInChArrFrom: returning pos = %d\n", pos);
#endif
  return pos;
}

/**
function int CharPosInChArr{CHARacter POSition IN CHar ARR}(char inarr[255]: in, char lookfor: in, int start: in)
Returns first position of character 'in' in 'inarr', if found, else '-1'
 **/
int CharPosInChArr(const char inarr[255], const char lookfor)
{
  return CharPosInChArrFrom(inarr, lookfor, 0);
}

/**
function int LastCharPosInChArr{LAST CHARacter POSition IN CHar ARR}(char inarr[255]: in, char lookfor: in)
Returns last position of character 'in' in 'inarr', if found, else '-1'
 **/
int LastCharPosInChArr(const char inarr[255], const char lookfor)
{
  int pos = 0;
  int lastpos = -1;
#ifdef __DEBUG_MBASIS__
  printf("mbasis.LastCharPosInChArr: function started\n");
#endif
  do
  {
    pos = CharPosInChArrFrom(inarr, lookfor, pos+1);
#ifdef __DEBUG_MBASIS__
    printf("mbasis.LastCharPosInChArr: CharPosInChArrFrom(..) returned pos=%d\n", pos);
#endif
    if (pos >= 0)
      lastpos = pos;
  }
  while (pos >= 0);
#ifdef __DEBUG_MBASIS__
  printf("mbasis.LastCharPosInChArr: returning lastpos = %d\n", lastpos);
#endif
  return lastpos;
}

/**
function int StrPosInChArr{STRing POSition IN CHar ARR}(char inarr[255]: in, char* lookfor: in)
Returns first position of string 'in' in 'inarr', if found, else '-1'
 **/
int StrPosInChArr(const char inarr[255], const char* lookfor)
{
  return StrPosInChArrFrom(inarr, lookfor, 0);
}

/**
function int StrPosInChArrFrom{LAST STRing POSition IN CHar ARR}(char inarr[255]: in, char* lookfor: in, int startpos)
Returns last position of character 'in' in 'inarr', if found, else '-1'
 **/
int StrPosInChArrFrom(const char inarr[255], const char* lookfor, int startpos)
{
  int i;
  int pos = startpos;
#ifdef __DEBUG_MBASIS__
  printf("mbasis.StrPosInChArrFrom: STARTED: inarr = <%s>, lookfor = <%s>, startpos = %d\n", inarr, lookfor, startpos);
#endif
  if (pos >= strlen(inarr) - strlen(lookfor) || strlen(lookfor) > strlen(inarr))
    return -1;
  do
  {
    if (inarr[pos] == '\0')
      return -1;
#ifdef __DEBUG_MBASIS__
    printf("mbasis.StrPosInChArrFrom: inarr[pos=%d] = %c\n", pos, inarr[pos]);
#endif
    for (i = 0; i < strlen(lookfor); i++)
    {
      if (inarr[pos + i] == lookfor[i])
      {
#ifdef __DEBUG_MBASIS__
        printf("mbasis.StrPosInChArrFrom: inarr[pos=%d + i=%d](=%c) == lookfor[i=%d](=%c) returned TRUE\n", pos, i, inarr[pos + i], i, lookfor[i]);
        //      printf("mbasis.StrPosInChArrFrom: inarr[pos(=%d) + strlen(lookfor=<%s>(=%d)) - 1 = %d] = %s\n", pos, lookfor, strlen(lookfor), pos + strlen(lookfor) - 1, inarr[pos + strlen(lookfor) - 1]);
#endif
        if (i == strlen(lookfor) - 1)
        {
#ifdef __DEBUG_MBASIS__
          printf("mbasis.StrPosInChArrFrom: i = %d, strlen(lookfor=%s) = %d: Returning pos = %d\n", i, lookfor, strlen(lookfor), pos);
#endif
          return pos;
        }
      }
      else
        i = strlen(lookfor);
    }
    pos++;
    if (pos >= strlen(inarr))
      return -1;
  }
  while (pos < strlen(inarr) - strlen(lookfor) + 1);

  //  if (pos == strlen(inarr) - strlen(lookfor))
  pos = -1;
#ifdef __DEBUG_MBASIS__
  printf("mbasis.StrPosInChArrFrom: returning pos = %d\n", pos);
#endif
  return pos;
}

/**
function int Read2DFileToArrays(const char* P_FileName: in, const double* P_XArray: in, const double* P_YArray)
Returns 1, if successfull, else 0.
 **/
int Read2DFileToArrays(const char* filename, double* warr, double* varr)
{

  FILE *fname;
  char *linea;
  long i, nelements;
  char oneword[200];

  // --- CountLines
  nelements = CountLines(filename);
#ifdef __DEBUG_MBASIS__
  printf("mbasis.read2dfiletoarrays: file %s contains %d data lines\n",filename,nelements);
#endif

  // --- open file <fname> with name <filename> for reading
  fname = fopen(filename, "r");
  if (fname == NULL)
  {
    printf("Failed to open file filename (=<%s>)\n", filename);
    exit (EXIT_FAILURE);
    //return 0;
  }

  varr = (double*)malloc(sizeof(double) * nelements);
  if (varr == NULL)
  {
    printf("mbasis.read2dfiletoarrays: NOT ENOUGH MEMORY FOR varr\n");
    exit (EXIT_FAILURE);
    //return 0;
  }
  warr = (double*)malloc(sizeof(double) * nelements);
  if (warr == NULL)
  {
    printf("mbasis.read2dfiletoarrays: NOT ENOUGH MEMORY FOR warr\n");
    exit (EXIT_FAILURE);
    //return 0;
  }

  // --- read file <fname> named <filename> in <warray> and <varray>
  i = 0;
  do
  {
    linea = fgets(oneword, 200, fname);
    if (linea != NULL)
    {
#ifdef __DEBUG_MBASIS__
      printf("mbasis.read2dfiletoarrays: linea =<%s>\n", linea);
#endif
      warr[i] = atof(strtok(linea," "));
      varr[i] = atof(strtok(NULL," "));
#ifdef __DEBUG_MBASIS__
      printf("mbasis.read2dfiletoarrays: warr[%d]=%.7f, varr[%d]=%.7f\n", i, warr[i], i, varr[i]);
#endif

    }
    i++;
  }
  while (linea != NULL);
  // --- close input file
  fclose(fname);
#ifdef __DEBUG_MBASIS__
  printf("mbasis.read2dfiletoarrays: File filename (=<%s>) closed\n", filename);
#endif
  return 1;
}

/**
function int WriteArraysToFile(const char* P_FileName: in, const double* P_XArray: in, const double* P_YArray)
Returns 1, if successfull, else 0.
 **/
int WriteArraysToFile(const char* P_FileName, const double* P_XArray, double* P_YArray, long Start, long End)
{
  FILE *P_File;
  long i;

  P_File = fopen(P_FileName, "w");
  if (P_File == NULL)
  {
    printf("MBasis.WriteArrayToFile: ERROR: Cannot File P_FileName = <%s>!\n", P_FileName);
    exit(0);
  }
  for (i = Start; i <= End; i++)
  {
    fprintf(P_File, "%.7f %.7f\n", P_XArray[i], P_YArray[i]);
  }
  fclose(P_File);
  return 1;
}

/**
function int WriteArraysToFile(const char* P_FileName: in, const double* P_XArray: in, const double* P_YArray)
Returns 1, if successfull, else 0.
 **/
void Swap(double *P_A,double *P_B)
{
  double tmp = *P_A;

  *P_A = *P_B;
  *P_B = tmp;
}

/**
function int WriteArraysToFile(const char* P_FileName: in, const double* P_XArray: in, const double* P_YArray)
Returns 1, if successfull, else 0.
 **/
void BubbleSort(double *P_Arr,long Size)
{
  long UpperLimit = Size - 1;
  long LastSwap;
  long Pos;

  while(UpperLimit > 0)
  {
    LastSwap = 0;
    for(Pos = 0;Pos < UpperLimit; ++Pos)
    {
      if(P_Arr[Pos] > P_Arr[Pos+1])
      {
        Swap(&P_Arr[Pos], &P_Arr[Pos+1]);
        LastSwap = Pos;
      }
    }
    UpperLimit = LastSwap;
  }
}

/**
function int WriteArraysToFile(const char* P_FileName: in, const double* P_XArray: in, const double* P_YArray)
Returns 1, if successfull, else 0.
 **/
void ShellSort(double *P_Arr,long Size)
{
  long Step = Size;
  long Start, i, j;

  /*  while (Step < Size)
    {
    Step = (3 * Step) + 1;
  }*/
#ifdef __DEBUG_MBASIS__
  printf("MBasis.ShellSort: Step = %d\n", Step);
#endif

#ifdef __DEBUG_MBASIS__
  for (i = 0; i < Size; i++)
  {
    printf("MBasis.ShellSort: P_Arr[i=%d] = %.7f\n", i, P_Arr[i]);
  }
  printf("\n");
#endif

  do// while(step>0)
  {
    Step=Step/2;
    for(Start=0; Start < Step; Start++)
    {
      for(i = Start+1; i < Size; i += Step)
      {
        for(j = i - Step; j >= 0; j -= Step)
        {
          if(P_Arr[j] > P_Arr[j+Step])
          {
#ifdef __DEBUG_MBASIS__
            printf("MBasis.ShellSort: Swapping P_Arr[j = %d](\t\t\t=%.7f)\n and P_Arr[j(=%d) + Step(=%d) = %d](\t\t=%.7f)\n", j, P_Arr[j], j, Step, j + Step, P_Arr[j+Step]);
#endif
            Swap(&P_Arr[j],&P_Arr[j+Step]);
#ifdef __DEBUG_MBASIS__
            printf("MBasis.ShellSort: Swapped P_Arr[j = %d](\t\t\t=%.7f)\n and P_Arr[j(=%d) + Step(=%d) = %d](\t\t=%.7f)\n", j, P_Arr[j], j, Step, j + Step, P_Arr[j+Step]);
#endif
          }
          else
            break;
        }
      }
    }
#ifdef __DEBUG_MBASIS__
    for (i = 0; i < Size; i++)
    {
      printf("MBasis.ShellSort: P_Arr[i=%d] = %.7f\n", i, P_Arr[i]);
    }
    printf("\n");
#endif
  }
  while(Step>0);
}

/**
function int WriteArraysToFile(const char* P_FileName: in, const double* P_XArray: in, const double* P_YArray)
Returns 1, if successfull, else 0.
 **/
double Mean(const double *P_X, long XStart, long XEnd)
{
  double Sum = 0.;
  double Mean;
  long i;
  long NElements = 0;// = XEnd - XStart + 1;
  for (i = XStart; i <= XEnd; i++)
  {
    NElements++;
    Sum += P_X[i];
  }
  Mean = Sum / NElements;
#ifdef __DEBUG_MBASIS__
  printf("MBasis.Mean: Returning Mean = %.7f\n", Mean);
#endif
  return Mean;
}

/**
function int WriteArraysToFile(const char* P_FileName: in, const double* P_XArray: in, const double* P_YArray)
Returns 1, if successfull, else 0.
 **/
double Median(const double *P_Arr, long Start, long End)
{
  double *P_TempArr;
  long Size = End - Start + 1;
  double HalfSizeDbl = (Size / 2.);
  long HalfSizeInt = (Size / 2);

#ifdef __DEBUG_MBASIS__
  printf("MBasis.Median: HalfSizeInt = %d, HalfSizeDbl = %.5f\n", HalfSizeInt, HalfSizeDbl);
#endif
  P_TempArr = CopyDblArray(P_Arr, Start, End);
  ShellSort(P_TempArr, Size);
  if (fabs(HalfSizeDbl - HalfSizeInt) < 0.2)
  {
#ifdef __DEBUG_MBASIS__
    printf("MBasis.Median: Returning (P_TempArr[HalfSizeInt = %d - 1] = %.7f + P_TempArr[HalfSizeInt = %d] = %.7f) / 2. = %.7f\n", HalfSizeInt, P_TempArr[HalfSizeInt-1], HalfSizeInt, P_TempArr[HalfSizeInt], (P_TempArr[HalfSizeInt-1] + P_TempArr[HalfSizeInt]) / 2.);
#endif
    return (P_TempArr[HalfSizeInt-1] + P_TempArr[HalfSizeInt]) / 2.;
  }
#ifdef __DEBUG_MBASIS__
  printf("MBasis.Median: Returning P_TempArr[(Size = %d + 1)/2 - 1=%d] = %.7f\n", Size, (Size + 1)/2 - 1, P_TempArr[((Size+1)/2) - 1]);
#endif
  return P_TempArr[((Size + 1) / 2)-1];
}

/**
function int WriteArraysToFile(const char* P_FileName: in, const double* P_XArray: in, const double* P_YArray)
Returns 1, if successfull, else 0.
 **/
double* CopyDblArray(const double *P_ArrToCopy, long Start, long End)
{
  long i, Pos;
  double *P_OutArr;
  P_OutArr = (double*)malloc(sizeof(double) * (End - Start + 1));
  if (P_OutArr == NULL)
  {
    printf("MBasis.CopyDblArray: ERROR: Cannot allocate memory for P_OutArr!\n");
    exit(0);
  }
  Pos = 0;
  for (i = Start; i <= End; i++)
  {
    /*    P_OutArr[i] = (double)malloc(sizeof(double));
        if (P_OutArr[i] == NULL)
        {
          printf("MBasis.CopyDblArray: ERROR: Cannot allocate memory for P_OutArr[i=%d]!\n", i);
          exit(0);
      }*/
    P_OutArr[Pos] = P_ArrToCopy[i];
    Pos++;
  }
  return P_OutArr;
}


/**
function int IsDouble(const char* P_Str: in)
Returns 1, if P_Str is a double, else 0.
 **/
int IsDouble(const char* P_Str)
{
  char PointChar = '.';
  int Pos;
  int IntArrPos;
  char NumberCharArr[12];

  NumberCharArr[0] = '0';
  NumberCharArr[1] = '1';
  NumberCharArr[2] = '2';
  NumberCharArr[3] = '3';
  NumberCharArr[4] = '4';
  NumberCharArr[5] = '5';
  NumberCharArr[6] = '6';
  NumberCharArr[7] = '7';
  NumberCharArr[8] = '8';
  NumberCharArr[9] = '9';
  NumberCharArr[10] = '.';
  NumberCharArr[11] = '\0';

  for (Pos = 0; Pos < strlen(P_Str); Pos++)
  {
    if (CharPosInChArr(NumberCharArr, P_Str[Pos]) < 0)
      return 0;
  }

  if (CharPosInStr(P_Str, PointChar) < 0)
    return 0;

  return 1;
}

/**
function int IsInt(const char* P_Str: in)
Returns 1, if P_Str is a integer, else 0.
 **/
int IsInt(const char* P_Str)
{
  int Pos;
  int IntArrPos;
  char NumberCharArr[11];
  char P_NumberString[255];

  NumberCharArr[0] = '0';
  NumberCharArr[1] = '1';
  NumberCharArr[2] = '2';
  NumberCharArr[3] = '3';
  NumberCharArr[4] = '4';
  NumberCharArr[5] = '5';
  NumberCharArr[6] = '6';
  NumberCharArr[7] = '7';
  NumberCharArr[8] = '8';
  NumberCharArr[9] = '9';
  NumberCharArr[10] = '\0';

  for (Pos = 0; Pos < strlen(P_Str); Pos++)
  {
    if (CharPosInChArr(NumberCharArr, P_Str[Pos]) < 0)
      return 0;
  }
  return 1;
}

/**
function int IsNumber(const char* P_Str: in)
Returns 1, if P_Str is a double or an integer, else 0.
 **/
int IsNumber(const char* P_Str)
{
  if (IsDouble(P_Str) == 1 || IsInt(P_Str) == 1)
    return 1;
  return 0;
}

/**
function int CharrPosInStr(char *P_InStr: in, const char LookFor: in)
Returns first position of String 'LookFor' in 'P_InStr', looking from position Start, if found, else '-1'
 **/
int CharPosInStrFrom(const char* P_InStr, const char LookFor, int Start)
{
  char StrArr[255];
  strcpy(StrArr, P_InStr);
  return (CharPosInChArrFrom(StrArr, LookFor, Start));
}

/**
function int CharrPosInStr(char *P_InStr: in, const char LookFor: in)
Returns first position of String 'P_LookFor' in 'P_InStr', if found, else '-1'
 **/
int CharPosInStr(const char* P_InStr, const char LookFor)
{
  char StrArr[255];
  strcpy(StrArr, P_InStr);
  return (CharPosInChArr(StrArr, LookFor));
}

/**
function int StrPosInStr(char *P_InStr: in, char* P_LookFor: in)
Returns first position of String 'P_LookFor' in 'P_InStr', if found, else '-1'
 **/
int StrPosInStr(const char* P_InStr, const char* P_LookFor)
{
  char StrArr[255];
  strcpy(StrArr, P_InStr);
  return (StrPosInChArr(StrArr, P_LookFor));
}

/**
function int LastCharPosInStr(char *P_InStr: in, const char LookFor: in)
Returns last position of character 'P_LookFor' in 'P_InStr', if found, else '-1'
 **/
int LastCharPosInStr(const char* P_InStr, const char LookFor)
{
  char StrArr[255];
  strcpy(StrArr, P_InStr);
  return (LastCharPosInChArr(StrArr, LookFor));
}

/**
function int StrPosInStrFrom(char *P_InStr: in, char* P_LookFor: in, int Start)
Returns first position of String 'P_LookFor' in 'P_InStr', looking from position Start, if found, else '-1'
 **/
int StrPosInStrFrom(const char* P_InStr, const char* P_LookFor, int Start)
{
  char StrArr[255];
  strcpy(StrArr, P_InStr);
#ifdef __DEBUG_MBASIS__
  printf("MBasis.StrPosInStrFrom(P_InStr = <%s>, P_LookFor = <%s>, Start = %d) started\n", P_InStr, P_LookFor, Start);
#endif
  return (StrPosInChArrFrom(StrArr, P_LookFor, Start));
}

/**
  Function int IfParameterSet(const char *P_Parameters: in, int SizeOfParameters: in, const char *P_ParameterName: in)
  Returns Position of <P_ParameterName> in array <P_Parameters>, if found, else returns -1.
**
int IfParameterSet(const char *P_Parameters, int SizeOfParameters, const char *P_ParameterName)
{
  int i;
  for (i = 0; i < SizeOfParameters; i++)
  {
    if (StrPosInStr(&(P_Parameters[i]), P_ParameterName) == 0)
      return i;
  }
  return -1;
}*/

/**
  Function double Round(const double ToRound: in, int DigitsBehindDot: in)
  Returns rounded value of <ToRound>.
 **/
double Round(const double ToRound, int DigitsBehindDot)
{
  long TempLong;
  int TempInt, m;
  double TempDbl;

  TempDbl = ToRound;
  for (m = 0; m < DigitsBehindDot; m++)
    TempDbl *= 10;
  TempLong = (long)TempDbl;

  TempDbl -= TempLong;
  TempInt = (int)(TempDbl * 10);
  if (TempInt > 4)
    TempLong++;
  TempDbl = (double)TempLong;
  for (m = 0; m < DigitsBehindDot; m++)
    TempDbl /= 10.;
  return TempDbl;
}

/**
  Function double Round(const double ToRound: in, int DigitsBehindDot: in)
  Returns rounded long value of <ToRound>.
 **/
long RoundToLong(const double ToRound)
{
  return (long)Round(ToRound, 0);
}

/**
  Function int* WhereEqual(const double* P_DblArr: in, const int Size: in, const double Equal: in, int* NEqual: out)
  Returns Array containing the indezes of the elements in <P_DblArr> which are equal to <Equal>.
 **
int* WhereEqual(const double* P_Arr, const int Size, const double Equal, int* NEqual)
{
  int *P_Indizes;// = (int*)malloc(sizeof(int));
  int i;

  *NEqual = 0;

  for (i = 0; i < Size; i++)
  {
    if (fabs(P_Arr[i] - Equal) < 0.0000000000001)
    {
      (*NEqual)++;
    }
  }
  if (*NEqual == 0)
  {
    return NULL;
    *NEqual = 0;
  }
  P_Indizes = (int*)malloc(sizeof(int) * (*NEqual));
  *NEqual = 0;
  for (i = 0; i < Size; i++)
  {
    if (fabs(P_Arr[i] - Equal) < 0.0000000000001)
    {
      P_Indizes[*NEqual] = i;
      (*NEqual)++;
    }
  }

  return P_Indizes;
}

/**
  Function int* WhereEqual(const double* P_DblArr: in, const int Size: in, const double Equal: in, int* NEqual: out)
  Returns Array containing the indezes of the elements in <P_DblArr> which are greater than <Min>.
 **
int* WhereGreater(const double* P_Arr, const int Size, const double Min, int* NGreater)
{
  int *P_Indizes;// = (int*)malloc(sizeof(int));
  int i;

  *NGreater = 0;

  for (i = 0; i < Size; i++)
  {
    if (P_Arr[i] > Min)
    {
      (*NGreater)++;
    }
  }
  if (*NGreater == 0)
  {
    return NULL;
    *NGreater = 0;
  }
  P_Indizes = (int*)malloc(sizeof(int) * (*NGreater));
  *NGreater = 0;
  for (i = 0; i < Size; i++)
  {
    if (P_Arr[i] > Min)
    {
      P_Indizes[*NGreater] = i;
      (*NGreater)++;
    }
  }
  return P_Indizes;
}

/**
  Function int* WhereEqual(const double* P_DblArr: in, const int Size: in, const double Equal: in, int* NEqual: out)
  Returns Array containing the indezes of the elements in <P_DblArr> which are lower than <Max>.
 **
int* WhereLower(const double* P_Arr, const int Size, const double Max, int* NLower)
{
  int *P_Indizes;// = (int*)malloc(sizeof(int));
  int i;

  *NLower = 0;

  for (i = 0; i < Size; i++)
  {
    if (P_Arr[i] < Max)
    {
      (*NLower)++;
    }
  }
  if (*NLower == 0)
  {
    return NULL;
    *NLower = 0;
  }
  P_Indizes = (int*)malloc(sizeof(int) * (*NLower));
  *NLower = 0;
  for (i = 0; i < Size; i++)
  {
    if (P_Arr[i] < Max)
    {
      P_Indizes[*NLower] = i;
      (*NLower)++;
#ifdef __DEBUG_MBASIS__
      printf("MBasis.WhereLower: P_Arr[i=%d] = %.7f < Max = %.7f\n", i, P_Arr[i], Max);
#endif
    }
  }
  return P_Indizes;
}

/**
 Procedure Total(const double** PP_Arr: in, const int Size: in, const int NDims: in, const int Dimension: in)
<PP_Arr>: Array to sum.
<Size>:   Number of elements of <PP_Arr> is <NDims> * <Size>
<NDims>:              -    ii   -

Returns the sum of the elements of <PP_Arr>. if <Dimension> is greater than -1 and lower than <NDims>, the sum over the given <Dimension> is returned. The result then will be an array with one less dimensions than <PP_Arr>. For example, if the dimensions of <PP_Arr> is <NDims=3> * <Size=10> and <Dimension> is 1, the dimensions of the result is (<NDims=3>-1) * <Size=10>, and element [i][j] contains the sum

  PP_Total[i][j] = sum_{k=0}^{Size-1} PP_Arr[i][k][j]

**
double** Total(const double*** PP_Arr, const int NDims, const int* P_Size, const int Dimension)
{
  long i, j, k;
  double **PP_Total = (double**)malloc(sizeof(double*) * (NDims - 1));

  for (i = 0; i < NDims - 1; i++)
  {
    PP_Total[i] = (double*)malloc(sizeof(double) * P_Size[i]);
//  }
//  for (i = 0; i < NDims - 1; i++)
//  {
    for (j = 0; j < P_Size[i]; j++)
    {
/*      PP_Total[i][j] = (double)malloc(sizeof(double));
      if (PP_Total[i][j] == NULL){
        printf("MBasis.Total: ERROR: Cannot allocate space for PP_Total[i=%d][j=%d]\n", i, j);
        exit(0);
    }*
      PP_Total[i][j] = 0;
      for (k = 0; k < P_Size[Dimension]; k++)
      {
        PP_Total[i][j] += PP_Arr[i][k][j];
      }
    }
  }
  return PP_Total;
}

void* Replicate(void* Content, int XSize, int YSize)
{
//  return void*;
}
*/
