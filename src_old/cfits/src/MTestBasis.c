/*
author: Andreas Ritter
created: 01/12/2007
last edited: 01/12/2007
compiler: g++ 4.0
basis machine: Ubuntu Linux 6.06
*
#import "MTestBasis.h"

long i, j, k, DblArrSize;
double *P_DblArr;
double TempDbl, Expected;

/**
 * Procedure CompareStrings
 **
int CompareStrings(const char* P_StrA, const char* P_StrB)
{
  if (strcmp(P_StrA, P_StrB) == 0)
  {
    printf("MTestBasis.CompareStrings: P_StrA (=%s) == P_StrB (=%s) (EXPECTED)\n\n", P_StrA, P_StrB);
  }
  else
  {
    printf("MTestBasis.CompareStrings: P_StrA (=%s) != P_StrB (=%s) (UNEXPECTED)\n", P_StrA, P_StrB);
    return 0;
  }
  return 1;
}

/**
 * Procedure CompareInts
 **
int CompareInts(int IntA, int IntB)
{
  if (IntA == IntB)
  {
    printf("MTestBasis.CompareInts: IntA (=%d) == IntB (=%d) (EXPECTED)\n\n", IntA, IntB);
  }
  else
  {
    printf("MTestBasis.CompareInts: IntA (=%d) != IntB (=%d) (UNEXPECTED)\n", IntA, IntB);
    return 0;
  }
  return 1;
}

/**
 * Procedure CompareDoubles
 **
int CompareDoubles(double DblA, double DblB)
{
  if (fabs(DblA - DblB) < 0.000000000001)
  {
    printf("MTestBasis.CompareInts: DblA (=%.7f) == DblB (=%.7f) (EXPECTED)\n\n", DblA, DblB);
  }
  else
  {
    printf("MTestBasis.CompareInts: DblA (=%.7f) != DblB (=%.7f) (UNEXPECTED)\n", DblA, DblB);
    return 0;
  }
  return 1;
}

/**
 * Test Procedure MBasis.StrTrimChar
 **
int TestStrTrimChar()
{
  char* P_CompleteString = (char*)malloc(sizeof(char) * 255);
  char* P_NewString;
  char* P_Expected = (char*)malloc(sizeof(char) * 255);

  printf("MTestBasis.TestStrTrimChar: Procedure started\n");

  strcpy(P_CompleteString, " \' 2345678 \' ");

  strcpy(P_Expected, " \' 2345678 \' ");
  printf("MTestBasis.TestStrTrimChar: P_Expected = <%s>\n", P_Expected);
  P_NewString = StrTrimChar( P_CompleteString, '\'', 0);
  if (CompareStrings(P_NewString, P_Expected) == 0)
    return 0;

  strcpy(P_Expected, " \' 2345678 \' ");
  printf("MTestBasis.TestStrTrimChar: P_Expected = <%s>\n", P_Expected);
  P_NewString = StrTrimChar( P_CompleteString, '\'', 1);
  if (CompareStrings(P_NewString, P_Expected) == 0)
    return 0;

  strcpy(P_Expected, " \' 2345678 \' ");
  printf("MTestBasis.TestStrTrimChar: P_Expected = <%s>\n", P_Expected);
  P_NewString = StrTrimChar( P_CompleteString, '\'', 2);
  if (CompareStrings(P_NewString, P_Expected) == 0)
    return 0;

  strcpy(P_Expected, "\' 2345678 \'");
  printf("MTestBasis.TestStrTrimChar: P_Expected = <%s>\n", P_Expected);
  P_NewString = StrTrimChar( P_CompleteString, ' ', 2);
  if (CompareStrings(P_NewString, P_Expected) == 0)
    return 0;

  strcpy(P_CompleteString, "\' 2345678 \'");
  strcpy(P_Expected, " 2345678 \'");
  printf("MTestBasis.TestStrTrimChar: P_Expected = <%s>\n", P_Expected);
  P_NewString = StrTrimChar( P_CompleteString, '\'', 0);
  if (CompareStrings(P_NewString, P_Expected) == 0)
    return 0;

  strcpy(P_Expected, "\' 2345678 ");
  printf("MTestBasis.TestStrTrimChar: P_Expected = <%s>\n", P_Expected);
  P_NewString = StrTrimChar( P_CompleteString, '\'', 1);
  if (CompareStrings(P_NewString, P_Expected) == 0)
    return 0;

  strcpy(P_Expected, " 2345678 ");
  printf("MTestBasis.TestStrTrimChar: P_Expected = <%s>\n", P_Expected);
  P_NewString = StrTrimChar( P_CompleteString, '\'', 2);
  if (CompareStrings(P_NewString, P_Expected) == 0)
    return 0;

  return 1;
}

/**
 * Test Procedure MBasis.StrTrim
 **
int TestStrTrim()
{
  char* P_CompleteString = (char*)malloc(sizeof(char) * 255);
  char* P_NewString;
  char* P_Expected = (char*)malloc(sizeof(char) * 255);

  printf("MTestBasis.TestStrTrim: Procedure started\n");

  strcpy(P_CompleteString, "  2345678  ");
  strcpy(P_Expected, "2345678  ");
  printf("MTestBasis.TestStrTrim: P_Expected = <%s>\n", P_Expected);

  P_NewString = StrTrim( P_CompleteString, 0);
  if (CompareStrings(P_NewString, P_Expected) == 0)
    return 0;

  strcpy(P_Expected, "2345678");
  printf("MTestBasis.TestStrTrim: P_Expected = <%s>\n", P_Expected);
  free(P_NewString);
  P_NewString = StrTrim( P_CompleteString, 2);
  if (CompareStrings(P_NewString, P_Expected) == 0)
    return 0;

  strcpy(P_Expected, "  2345678");
  printf("MTestBasis.TestStrTrim: P_Expected = <%s>\n", P_Expected);
  free(P_NewString);
  P_NewString = StrTrim( P_CompleteString, 1);
  if (CompareStrings(P_NewString, P_Expected) == 0)
    return 0;

  return 1;
}

/**
 * Test Procedure MBasis.SubString
 **
int TestSubString()
{
  char* P_CompleteString = (char*)malloc(sizeof(char) * 255);
  char* P_SubString;
  char* P_TempStr = (char*)malloc(sizeof(char) * 255);
  char* P_Expected = (char*)malloc(sizeof(char) * 255);

  printf("MTestBasis.TestSubString: Procedure started\n");

  strcpy(P_CompleteString, "01234567890abcdefghijklmnopqrstuvwxyz");
  strcpy(P_Expected, "3456");

  P_SubString = SubString(P_CompleteString, 3, 6);
  if (CompareStrings(P_SubString, P_Expected) == 0)
    return 0;

  free(P_SubString);
  P_SubString = SubString(P_CompleteString, 3, 6);
  if (CompareStrings(P_SubString, P_Expected) == 0)
    return 0;

  strcpy(P_Expected, "90abcdefghijklmnopqrstuvwxyz");
  P_SubString = SubString(P_CompleteString, 9, 200);
  if (CompareStrings(P_SubString, P_Expected) == 0)
    return 0;

  strcpy(P_CompleteString, "COMMENT and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H\n");
  strcpy(P_TempStr, "COMMENT");
  strcpy(P_Expected, "and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H\n");
  P_SubString = SubString(P_CompleteString, strlen(P_TempStr) + 1, strlen(P_CompleteString));
  if (CompareStrings(P_SubString, P_Expected) == 0)
    return 0;

  return 1;
}

/**
 * Test Procedure MBasis.TestStringMethods
 **
int TestStringMethods()
{
  char *P_Original = (char*)malloc(sizeof(char) * 251);
  char *P_TestStr = (char*)malloc(sizeof(char) * 250);
  char *P_Expected = (char*)malloc(sizeof(char) * 249);
  int Expected;

  printf("MTestBasis.TestStringMethods: Procedure started\n");

  strcpy(P_Original, "01234567890abcdefghijklmnopqrstuvwxyz");
  strcpy(P_TestStr, "01234567890abcdefghijklmnopqrstuvwxyz");

  strcpy(P_Expected, "01234567890abcdefghijklmnopqrstuvwxyz");

  Expected = 0;
  printf("MTestBasis.TestStringMethods: starting CharPosInStr(P_Original = %s, '0')\n", P_Original);
  if (!CompareInts(CharPosInStr(P_Original, '0'), Expected))
    return 0;

  Expected = 10;
  printf("MTestBasis.TestStringMethods: starting LastCharPosInStr(P_Original = %s, '0')\n", P_Original);
  if (!CompareInts(LastCharPosInStr(P_Original, '0'), Expected))
    return 0;

  Expected = 36;
  printf("MTestBasis.TestStringMethods: starting CharPosInStr(P_Original = %s, 'z')\n", P_Original);
  if (!CompareInts(LastCharPosInStr(P_Original, 'z'), Expected))
    return 0;

  Expected = 0;
  printf("MTestBasis.TestStringMethods: starting CharPosInStr(P_Original = %s, '0')\n", P_Original);
  if (!CompareInts(StrPosInStr(P_Original, "0"), Expected))
    return 0;

  printf("MTestBasis.TestStringMethods: starting CharPosInStr(P_Original = %s, '0') == StrPosInStrFrom(P_Original, '0', 0)\n", P_Original);
  if (!CompareInts(StrPosInStr(P_Original, "0"), StrPosInStrFrom(P_Original, "0", 0)))
    return 0;

  Expected = 10;
  printf("MTestBasis.TestStringMethods: starting StrPosInStrFrom(P_Original = %s, '0', 1)\n", P_Original);
  if (!CompareInts(StrPosInStrFrom(P_Original, "0", 1), Expected))
    return 0;

  Expected = 36;
  printf("MTestBasis.TestStringMethods: starting StrPosInStr(P_Original = %s, 'z') == StrPosInStrFrom(P_Original, 'z', 20)\n", P_Original);
  if (!CompareInts(StrPosInStr(P_Original, "z"), StrPosInStrFrom(P_Original, "z", 20)))
    return 0;

  Expected = -1;
  printf("MTestBasis.TestStringMethods: starting StrPosInStr(P_Original = %s, 'Z')\n", P_Original);
  if (!CompareInts(StrPosInStr(P_Original, "Z"), Expected))
    return 0;

  return 1;
}

/**
 * Test Procedure MBasis.IsNumber
 **
int TestIsNumber()
{
  char *P_TestStr = (char*)malloc(sizeof(char) * 255);
  P_TestStr = "2352242352.232262dsg";
  int Expected = 0;

  printf("\nMTestBasis.TestIsNumber: Testing IsNumber(P_TestStr = <%s>)\n", P_TestStr);
  if (CompareInts(IsNumber(P_TestStr), Expected) == 0)
    return 0;

  P_TestStr = SubString(P_TestStr, 0, CharPosInStr(P_TestStr, 'd') - 1);
  printf("\nMTestBasis.TestIsNumber: Testing IsNumber(P_TestStr = <%s>)\n", P_TestStr);
  Expected = 1;
  if (CompareInts(IsNumber(P_TestStr), Expected) == 0)
    return 0;

  printf("\nMTestBasis.TestIsNumber: Testing IsDouble(P_TestStr = <%s>)\n", P_TestStr);
  Expected = 1;
  if (CompareInts(IsDouble(P_TestStr), Expected) == 0)
    return 0;

  printf("\nMTestBasis.TestIsNumber: Testing IsInt(P_TestStr = <%s>)\n", P_TestStr);
  Expected = 0;
  if (CompareInts(IsInt(P_TestStr), Expected) == 0)
    return 0;

  P_TestStr = SubString(P_TestStr, 0, CharPosInStr(P_TestStr, '.') - 1);
  printf("\nMTestBasis.TestIsNumber: Testing IsDouble(P_TestStr = <%s>)\n", P_TestStr);
  Expected = 0;
  if (CompareInts(IsDouble(P_TestStr), Expected) == 0)
    return 0;

  printf("\nMTestBasis.TestIsNumber: Testing IsInt(P_TestStr = <%s>)\n", P_TestStr);
  Expected = 1;
  if (CompareInts(IsInt(P_TestStr), Expected) == 0)
    return 0;

  return 1;
}

/**
 * Test Procedure MBasis.Round
 **
int TestRound()
{
  double ExpectedDbl, ToRound;
  long ExpectedLong;

  ToRound = 1234567890.123456789012345;

  ExpectedDbl = 1234567890.123457;
  printf("\nMTestBasis.TestIsNumber: Testing Round(ToRound = <%.7f>, 6)\n", ToRound);
  if (CompareDoubles(Round(ToRound, 6), ExpectedDbl) == 0)
    return 0;

  ExpectedDbl = 1234567890.1;
  printf("\nMTestBasis.TestIsNumber: Testing Round(ToRound = <%.7f>, 6)\n", ToRound);
  if (CompareDoubles(Round(ToRound, 1), ExpectedDbl) == 0)
    return 0;

  ExpectedDbl = 1234567890.1;
  printf("\nMTestBasis.TestIsNumber: Testing Round(ToRound = <%.7f>, 1)\n", ToRound);
  if (CompareDoubles(Round(ToRound, 1), ExpectedDbl) == 0)
    return 0;

  ExpectedDbl = 1234567890.;
  printf("\nMTestBasis.TestIsNumber: Testing Round(ToRound = <%.7f>, 0)\n", ToRound);
  if (CompareDoubles(Round(ToRound, 0), ExpectedDbl) == 0)
    return 0;

  ExpectedLong = 1234567890;
  printf("\nMTestBasis.TestIsNumber: Testing Round(ToRound = <%.7f>, 0)\n", ToRound);
  if (CompareDoubles(RoundToLong(ToRound), ExpectedDbl) == 0)
    return 0;


  return 1;
}

/**
 * Test Procedure MBasis.Where...
 **
int TestWhere()
{
  double *P_DblArr = (double*)malloc(sizeof(double) * 10);
  int *P_IndexArr, i;
  int IndexArrSize;
  int DblArrSize = 10;

  for (i = 0; i < DblArrSize; i++)
  {
    P_DblArr[i] = i;
    printf("MTestBasis.TestWhere: P_DblArr[i=%d] = %.7f\n", i, P_DblArr[i]);
  }

  printf("MTestBasis.TestWhere: Testing WhereEqual(P_DblArr, DblArrSize = %d, 5., IndexArrSize)\n", DblArrSize);
  P_IndexArr = WhereEqual(P_DblArr, DblArrSize, 5., &IndexArrSize);
  if (IndexArrSize != 1)
  {
    printf("MTestBasis.TestWhere: WhereEqual(P_DblArr, DblArrSize = %d, 5., IndexArrSize = %d): IndexArrSize UNEXPECTED!!!\n", DblArrSize, IndexArrSize);
    return 0;
  }
  for (i = 0; i < IndexArrSize; i++)
  {
    if (CompareDoubles(P_DblArr[P_IndexArr[i]], P_DblArr[5]) == 0)
      return 0;
  }

  P_DblArr[2] = 5.;
  printf("MTestBasis.TestWhere: Testing WhereEqual(P_DblArr, DblArrSize = %d, 5., IndexArrSize)\n", DblArrSize);
  P_IndexArr = WhereEqual(P_DblArr, DblArrSize, 5., &IndexArrSize);
  if (IndexArrSize != 2)
  {
    printf("MTestBasis.TestWhere: WhereEqual(P_DblArr, DblArrSize = %d, 5., IndexArrSize = %d): IndexArrSize UNEXPECTED!!!\n", DblArrSize, IndexArrSize);
    return 0;
  }
//  for (i = 0; i < IndexArrSize; i++)
//  {
  if (CompareDoubles(P_DblArr[P_IndexArr[0]], P_DblArr[2]) == 0)
    return 0;
  else
    printf("MTestBasis.TestWhere: P_DblArr[P_IndexArr[0] = %d] = %.7f == %.7f = P_DblArr[2] (EXPECTED)\n", P_IndexArr[0], P_DblArr[P_IndexArr[0]], P_DblArr[2]);
  if (CompareDoubles(P_DblArr[P_IndexArr[1]], P_DblArr[5]) == 0)
    return 0;
  else
    printf("MTestBasis.TestWhere: P_DblArr[P_IndexArr[1] = %d] = %.7f == %.7f = P_DblArr[5] (EXPECTED)\n", P_IndexArr[1], P_DblArr[P_IndexArr[1]], P_DblArr[5]);
//  }

  printf("MTestBasis.TestWhere: Testing WhereEqual(P_DblArr, DblArrSize = %d, 15., IndexArrSize)\n", DblArrSize);
  P_IndexArr = WhereEqual(P_DblArr, DblArrSize, 15., &IndexArrSize);
  if (IndexArrSize > 0)
  {
    printf("MTestBasis.TestWhere: WhereEqual(P_DblArr, DblArrSize = %d, 5., IndexArrSize = %d): IndexArrSize UNEXPECTED!!!\n", DblArrSize, IndexArrSize);
    return 0;
  }
  if (P_IndexArr != NULL)
  {
    printf("MTestBasis.TestWhere: WhereEqual(P_DblArr, DblArrSize = %d, 15., IndexArrSize = %d) returned not NULL UNEXPECTED!!!\n", DblArrSize, IndexArrSize);
    return 0;
  }

  printf("MTestBasis.TestWhere: Testing WhereGreater(P_DblArr, DblArrSize = %d, 5., IndexArrSize)\n", DblArrSize);
  P_IndexArr = WhereGreater(P_DblArr, DblArrSize, 5., &IndexArrSize);
  if (IndexArrSize != 4)
  {
    printf("MTestBasis.TestWhere: WhereGreater(P_DblArr, DblArrSize = %d, 5., IndexArrSize = %d): IndexArrSize UNEXPECTED!!!\n", DblArrSize, IndexArrSize);
    return 0;
  }
  for (i = 0; i < IndexArrSize; i++)
  {
    if (CompareDoubles(P_DblArr[P_IndexArr[i]], P_DblArr[6+i]) == 0)
      return 0;
  }

  P_DblArr[2] = 2.;
  printf("MTestBasis.TestWhere: Testing WhereLower(P_DblArr, DblArrSize = %d, 5., IndexArrSize)\n", DblArrSize);
  P_IndexArr = WhereLower(P_DblArr, DblArrSize, 5., &IndexArrSize);
  if (IndexArrSize != 5)
  {
    printf("MTestBasis.TestWhere: WhereLower(P_DblArr, DblArrSize = %d, 5., IndexArrSize = %d): IndexArrSize UNEXPECTED!!!\n", DblArrSize, IndexArrSize);
    return 0;
  }
  for (i = 0; i < IndexArrSize; i++)
  {
    if (CompareDoubles(P_DblArr[P_IndexArr[i]], P_DblArr[i]) == 0)
      return 0;
  }

  return 1;
}

/**
 * Test Procedure MBasis.Total
 **
int TestTotal()
{
  double ***PPP_DblArr = (double***)malloc(sizeof(double**) * 5);
  double **PP_Total;
  int i, j, k;
  int *P_Dim = (int*)malloc(sizeof(int) * 3);

  for (i = 0; i < 5; i++)
  {
    PPP_DblArr[i] = (double**)malloc(sizeof(double*) * 7);
    for (j = 0; j < 7; j++)
    {
      PPP_DblArr[i][j] = (double*)malloc(sizeof(double) * 10);
      for (k = 0; k < 10; k++)
      {
        PPP_DblArr[i][j][k] = i + j + k;//(double)malloc(sizeof(double));
      }
    }
  }
  P_Dim[0] = 5;
  P_Dim[1] = 7;
  P_Dim[2] = 10;

/*  PP_Total = Total(PPP_DblArr, 3, P_Dim, 1);
  for (i = 0; i < 5; i++)
  {
  for (j = 0; j < 10; j++)
  {
  printf("MTestBasis.TestTotal: PPP_Total[i=%d][j=%d] = %.7f\n", i, j, PP_Total[i][j]);
}
}
*
}

/**
 * Test Module MBasis
 **
int TestBasis()
{
  DblArrSize = 10;
  P_DblArr = (double*)malloc(sizeof(double) * DblArrSize);
  if (P_DblArr == NULL)
  {
    printf("MTestBasis.TestBasis: ERROR: Cannot allocate memory for P_DblArr!\n");
    return 0;
  }

  // Test Mean
  printf("\n");
  printf("MTestBasis.TestBasis: Testing Mean(P_DblArr)\n");
  // Expected Result: 1.0
  Expected = 1.;
  for (i = 0; i < DblArrSize; i++)
  {
    P_DblArr[i] = Expected;
  }
  TempDbl = Mean(P_DblArr, 0, 9);
  if (TempDbl - Expected < 0.00000001)
  {
    printf("MTestBasis.TestBasis: Mean(P_DblArr) returned %.7f (EXPECTED)\n", TempDbl);
  }
  else
  {
    printf("MTestBasis.TestBasis: Mean(P_DblArr) returned %.7f (UNEXPECTED!!!)\n", TempDbl);
    return 0;
  }

  // Expected Result: 5.5
  Expected = 5.5;
  for (i = 0; i < DblArrSize; i++)
  {
    P_DblArr[i] = i;
  }
  TempDbl = Mean(P_DblArr, 0, 9);
  if (TempDbl - Expected < 0.00000001)
  {
    printf("MTestBasis.TestBasis: Mean(P_DblArr) returned %.7f (EXPECTED)\n", TempDbl);
  }
  else
  {
    printf("MTestBasis.TestBasis: Mean(P_DblArr) returned %.7f (UNEXPECTED!!!)\n", TempDbl);
    return 0;
  }

  // Test Median
  printf("\n");
  printf("MTestBasis.TestBasis: Testing Median(P_DblArr)\n");
  // Expected Result: 1.0
  Expected = 1.;
  for (i = 0; i < DblArrSize; i++)
  {
    P_DblArr[i] = Expected;
  }
  TempDbl = Median(P_DblArr, 0, 9);
  if (TempDbl - Expected < 0.00000001)
  {
    printf("MTestBasis.TestBasis: Median(P_DblArr) returned %.7f (EXPECTED)\n", TempDbl);
  }
  else
  {
    printf("MTestBasis.TestBasis: Median(P_DblArr) returned %.7f (UNEXPECTED!!!)\n", TempDbl);
    return 0;
  }

  // Expected Result: 4.5
  Expected = 4.5;
  for (i = 0; i < DblArrSize; i++)
  {
    P_DblArr[i] = i;
  }
  TempDbl = Median(P_DblArr, 0, 9);
  if (fabs(TempDbl - Expected) < 0.00000001)
  {
    printf("MTestBasis.TestBasis: Median(P_DblArr) returned %.7f (EXPECTED)\n", TempDbl);
  }
  else
  {
    printf("MTestBasis.TestBasis: Median(P_DblArr) returned %.7f (UNEXPECTED!!!)\n", TempDbl);
    return 0;
  }

  // Expected Result: 3.5
  Expected = 4.0;
  for (i = 0; i < DblArrSize; i++)
  {
    P_DblArr[i] = i;
  }
  TempDbl = Median(P_DblArr, 0, 8);
  if (fabs(TempDbl - Expected) < 0.00000001)
  {
    printf("MTestBasis.TestBasis: Median(P_DblArr) returned %.7f (EXPECTED)\n", TempDbl);
  }
  else
  {
    printf("MTestBasis.TestBasis: Median(P_DblArr) returned %.7f (UNEXPECTED!!!)\n", TempDbl);
    return 0;
  }

/*  TempDbl = mean(P_DblArr);
  if (TempDbl - Expected < 0.00000001)
  {
  printf("MTestBasis.TestBasis: Mean(P_DblArr) returned %.7f (EXPECTED)\n", TempDbl);
}
  else
  {
  printf("MTestBasis.TestBasis: Mean(P_DblArr) returned %.7f (UNEXPECTED!!!)\n", TempDbl);
  return 0;
}*

  // --- Test BubbleSort
  printf("\n");
  printf("MTestBasis.TestBasis: Testing BubbleSort(P_DblArr)\n");
  for (i = 0; i < DblArrSize; i++)
  {
    P_DblArr[i] = DblArrSize - i - 1;
    printf("MTestBasis.TestBasis: P_DblArr[%d] = %.7f\n", i, P_DblArr[i]);
  }
  BubbleSort(P_DblArr, DblArrSize);
  for (i = 0; i < DblArrSize; i++)
  {
    if (fabs(P_DblArr[i] - i) < 0.00000001)
      printf("MTestBasis.TestBasis: P_DblArr[%d] = %.7f (EXPECTED)\n", i, P_DblArr[i]);
    else
    {
      printf("MTestBasis.TestBasis: P_DblArr[%d] = %.7f != %d (UNEXPECTED!!!)\n", i, P_DblArr[i]), i + 1;
      return 0;
    }
  }

  // --- Test ShellSort
  printf("\n");
  printf("MTestBasis.TestBasis: Testing ShellSort(P_DblArr)\n");
  for (i = 0; i < DblArrSize; i++)
  {
    P_DblArr[i] = DblArrSize - i - 1;
    printf("MTestBasis.TestBasis: P_DblArr[%d] = %.7f\n", i, P_DblArr[i]);
  }
  ShellSort(P_DblArr, DblArrSize);
  for (i = 0; i < DblArrSize; i++)
  {
    printf("MTestBasis.TestBasis: fabs(P_DblArr[%d] (= %.7f) - i (= %d)) returned %.7f\n", i, P_DblArr[i], i, fabs(P_DblArr[i] - i));
    if (fabs(P_DblArr[i] - i) < 0.00000001)
    {
      printf("MTestBasis.TestBasis: P_DblArr[%d] = %.7f (EXPECTED)\n", i, P_DblArr[i]);
    }
    else
    {
      printf("MTestBasis.TestBasis: P_DblArr[%d] = %.7f != %d (UNEXPECTED!!!)\n", i, P_DblArr[i], i);
      return 0;
    }
  }

  if (TestSubString() == 0)
    return 0;
  if (TestStrTrimChar() == 0)
    return 0;
  if (TestStrTrim() == 0)
    return 0;
  if (TestIsNumber() == 0)
    return 0;
  if (TestStringMethods() == 0)
    return 0;
  if (TestRound() == 0)
    return 0;
  if (TestWhere() == 0)
    return 0;
  if (TestTotal() == 0)
    return 0;
  return 1;
}
**/
