/*
author: Andreas Ritter
created: 01/12/2007
last edited: 01/12/2007
compiler: gcc 4.0
basis machine: Ubuntu Linux 6.06
*/
#include "MRemoveDispLinesFromHeader.h"

int main(int argc, char *argv[])
{

  FILE *P_HeaderFile, *P_HeaderOutFile;
  long NLines, tempStrLen;
  long i, j;
  int  Remove = 0;
  int  NLinesRemoved = 0;
  int  NHeaderKeyWords = 12;
  char *P_Line, *P_TempLine;
  char *P_HeaderKeyWord;
  char OneWord[200];
  char HeaderFileName[250], HeaderOutFileName[250];
  char *TempInFile = "/yoda/feros/077.D0235B_F/2006-04-11/lq_hya5_2006-04-12T00-21-05.674_840s_botzfxs_ec2_bld_head.text";
  char **PP_HeaderKeyWords;

  if (argc < 2)
  {
    printf("MRemoveDispLinesFromHeader.main: NOT ENOUGH PARAMETERS SPECIFIED!\n");
    printf("MRemoveDispLinesFromHeader.main: USAGE:\n");
    printf("MRemoveDispLinesFromHeader.main: removedisplinesfromheader (char*)headerfile.text\n");
    printf("\n headerfile.text:\n");
    printf("    EXTEND  =                    T / File may contain extensions\n");
    printf("    ORIGIN  = 'NOAO-IRAF FITS Image Kernel July 2003' / FITS file originator\n");
    printf("    ...\n");
//    exit(0);
    argv[1] = (char*)malloc(sizeof(char)*(strlen(TempInFile)+1));
    strcpy(argv[1], TempInFile);
    argc = 2;
  }

  P_TempLine = "";

  PP_HeaderKeyWords = (char**)malloc(sizeof(char*) * NHeaderKeyWords);
  if (PP_HeaderKeyWords == NULL)
  {
    printf("MRemoveDispLinesFromHeader.main: Failed to allocate memory for PP_HeaderKeyWords\n");
    exit (EXIT_FAILURE);
  }

  PP_HeaderKeyWords[0] = "CTYPE";
  PP_HeaderKeyWords[1] = "CRVAL";
  PP_HeaderKeyWords[2] = "CRPIX";
  PP_HeaderKeyWords[3] = "CD";
  PP_HeaderKeyWords[4] = "WCSDIM";
  PP_HeaderKeyWords[5] = "DC-FLAG";
  PP_HeaderKeyWords[6] = "EXTEND";
  PP_HeaderKeyWords[7] = "APNUM";
  PP_HeaderKeyWords[8] = "IRAF-TLM";
  PP_HeaderKeyWords[9] = "LTM";
  PP_HeaderKeyWords[10] = "WAT";
  PP_HeaderKeyWords[11] = "DATE  ";

  HeaderFileName[0] = '\0';
  ChArrCat(HeaderFileName, argv[1], 0);
#ifdef __DEBUG_MREMOVEDISPLINESFROMHEADER__
  printf("MRemoveDispLinesFromHeader.main: File HeaderFileName = <%s>\n", HeaderFileName);
#endif

  HeaderOutFileName[0] = '\0';
  tempStrLen = ChArrCat(HeaderOutFileName, argv[1], 0);
  tempStrLen = ChArrCat(HeaderOutFileName, ".new", tempStrLen);
#ifdef __DEBUG_MREMOVEDISPLINESFROMHEADER__
  printf("MRemoveDispLinesFromHeader.main: File HeaderOutFileName = <%s>\n", HeaderOutFileName);
#endif


  // --- count Lines of P_HeaderFiP_HeaderFileName
  NLines = CountLines(HeaderFileName);
#ifdef __DEBUG_MREMOVEDISPLINESFROMHEADER__
  printf("MRemoveDispLinesFromHeader: HeaderFileName = <%s> contains %d Lines\n", HeaderFileName, NLines);
#endif

  // --- open file <P_HeaderFileName> for reading
  P_HeaderFile = fopen(HeaderFileName, "r");
  if (P_HeaderFile == NULL)
  {
    printf("MRemoveDispLinesFromHeader.main: Failed to open file HeaderFileName =(<%s>)\n", HeaderFileName);
    exit (EXIT_FAILURE);
  }
#ifdef __DEBUG_MREMOVEDISPLINESFROMHEADER__
  printf("MRemoveDispLinesFromHeader.main: File HeaderFileName =(<%s>) opened\n", HeaderFileName);
#endif


  // --- open file <P_HeaderOutFile> for reading
  P_HeaderOutFile = fopen(HeaderOutFileName, "w");
  if (P_HeaderOutFile == NULL)
  {
    printf("MRemoveDispLinesFromHeader.main: Failed to open file HeaderOutFileName =(<%s>)\n", HeaderOutFileName);
    exit (EXIT_FAILURE);
  }
#ifdef __DEBUG_MREMOVEDISPLINESFROMHEADER__
  printf("MRemoveDispLinesFromHeader.main: File HeaderOutFileName =(<%s>) opened\n", HeaderOutFileName);
#endif

  for (i = 0; i < NLines; i++)
  {
    // --- read Line of P_HeaderFP_HeaderFile
    P_Line = fgets(OneWord, 200, P_HeaderFile);
    if (P_Line == NULL)
    {
      printf("MRemoveDispLinesFromHeader.main: Failed to read Line %d of file HeaderFileName =(<%s>)\n", i, HeaderFileName);
      exit (EXIT_FAILURE);
    }
#ifdef __DEBUG_MREMOVEDISPLINESFROMHEADER__
    printf("MRemoveDispLinesFromHeader.main: OneWord No i=%d of file HeaderFileName(=%s) = <%s>\n", i, HeaderFileName, OneWord);
    printf("MRemoveDispLinesFromHeader.main: P_Line = <%s>, P_TempLine = <%s>\n", P_Line, P_TempLine);
#endif

    P_TempLine = (char*)malloc(sizeof(char) * 255);
    StrCatStr(P_TempLine, P_Line);

    // --- read HeaderKeyWord
    P_HeaderKeyWord = strtok(P_Line,"='\0\t\n");
#ifdef __DEBUG_MREMOVEDISPLINESFROMHEADER__
    printf("MRemoveDispLinesFromHeader.main: P_HeaderKeyWord = <%s>\n", P_HeaderKeyWord);
    printf("MRemoveDispLinesFromHeader.main: StrPosInChArr(P_HeaderKeyWord = <%s>, 'WAT') returns <%d>\n", P_HeaderKeyWord, StrPosInChArr(P_HeaderKeyWord, "WAT"));
#endif

    if (strlen(P_TempLine) > 1)
    {
      Remove = 0;
      for (j = 0; j < (NHeaderKeyWords); j++)
      {
        if (StrPosInChArr(P_HeaderKeyWord, PP_HeaderKeyWords[j]) == 0)
        {
          Remove = 1;
#ifdef __DEBUG_MREMOVEDISPLINESFROMHEADER__
          printf("MRemoveDispLinesFromHeader.main: StrPosInChArr(%s, %s) returned = %d\n", P_HeaderKeyWord, PP_HeaderKeyWords[j]);
          printf("MRemoveDispLinesFromHeader.main: j = %d: HeaderKeyWords[%d] = <%s>, HeaderKeyWord = <%s>: Removing Line <%s> from HeaderFile!\n", j, j, PP_HeaderKeyWords[j], P_HeaderKeyWord, P_Line);
#endif
          NLinesRemoved++;
        }
      }// end for (j = 0; j < NHeaderKeyWords - 1; j++)
      if (StrPosInChArr(P_HeaderKeyWord, "(old)") == 0)
      {
        Remove = 1;
#ifdef __DEBUG_MREMOVEDISPLINESFROMHEADER__
          printf("MRemoveDispLinesFromHeader.main: StrPosInChArr(%s, %s) returned = %d\n", P_HeaderKeyWord, PP_HeaderKeyWords[j]);
          printf("MRemoveDispLinesFromHeader.main: j = %d: HeaderKeyWords[%d] = <%s>, HeaderKeyWord = <%s>: Removing Line <%s> from HeaderFile!\n", j, j, PP_HeaderKeyWords[j], P_HeaderKeyWord, P_Line);
#endif
          NLinesRemoved++;
      }
      if (Remove == 0)
        fprintf(P_HeaderOutFile, "%s", P_TempLine);
    }// end if (strlen(P_TempLine) > 1)
  }//end for (i = 0; i < NLines; i++)
  fprintf(P_HeaderOutFile, "\n");
  fclose(P_HeaderOutFile);
  fclose(P_HeaderFile);
  printf("MRemoveDispLinesFromHeader.main: %d lines removed from HeaderFileName\n <%s>\n", NLinesRemoved, HeaderFileName);
  printf("MRemoveDispLinesFromHeader.main: Output written to HeaderOutFileName\n <%s>\n", HeaderOutFileName);

  return EXIT_SUCCESS;
}
