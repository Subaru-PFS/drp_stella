/*
author: Andreas Ritter
created: 01/12/2007
last edited: 01/12/2007
compiler: g++ 4.0
basis machine: Ubuntu Linux 6.06
*/

#include "MTestApp.h"

/*int DateInput(CDate &dat){
  int tempInt;
  bool t7;
  cout << "PLEASE TYPE THE DAY: ";
  cin >> tempInt;
  cout << tempInt;
  t7 = dat.SetDay(tempInt);
  if (t7)
    cout << "MTestApp::DateInput:                  t7(Set...)     = \"TRUE\" " << endl;
  else
    cout << "MTestApp::DateInput:                  t7(Set...)     = \"FALSE\" " << endl;
  cout  << "PLEASE TYPE THE MONTH: ";
  cin >> tempInt;
  cout << tempInt;
  t7 = dat.SetMonth(tempInt);
  if (t7)
    cout << "MTestApp::DateInput:                  t7(Set...)     = \"TRUE\" " << endl;
  else
    cout << "MTestApp::DateInput:                  t7(Set...)     = \"FALSE\" " << endl;
  cout << "PLEASE TYPE THE YEAR: ";
  cin >> tempInt;
  cout << tempInt;
  t7 = dat.SetYear(tempInt);
  if (t7)
    cout << "MTestApp::DateInput:                  t7(Set...)     = \"TRUE\" " << endl;
  else
    cout << "MTestApp::DateInput:                  t7(Set...)     = \"FALSE\" " << endl;
  return tempInt;
}

/*******************************************************************/

bool TestCStringConstructors()
{
  //  return TestConstructors("CString");
  bool t1, t2, t3, t4, t5, t6, t7, t8, t9, t10;
  char *tempStr = new char[255];
  CString *P_TestString = new CString();
  CString *P_TestString1 = new CString();
  CString *P_TestString2 = new CString();

  cout << "MTestApp::TestCStringConstructors: P_TestString = <" << *P_TestString << ">" << endl;
  cout << "MTestApp::TestCStringConstructors: P_TestString1 = <" << *P_TestString1 << ">" << endl;
  cout << "MTestApp::TestCStringConstructors: P_TestString2 = <" << *P_TestString2 << ">" << endl;
  CString *p_testString = new CString("0123456789 Hello World!");
  // Test: CString::Constructors
  // Tests included: CString::CalcString, CString::Dbl2ushort,
  //                 CString::operator=, CString::Copy,
  //                 CString::operator==, CString::EqualValue,
  //                 CString::operator<<, CString::Show
  // require : nothing
  // ensure  : t1 is "FALSE", t2, t3, t4, t5, t6 and t7 are "TRUE"

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCStringConstructors: starting standard constructor" << endl;
#endif
  if (P_TestString != NULL)
    delete(P_TestString);
  P_TestString = new CString();

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCStringConstructors: starting init constructor('P_TestString1')" << endl;
#endif
  if (P_TestString1 != NULL)
    delete(P_TestString1);
  P_TestString1 = new CString("P_TestString1");
#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCStringConstructors:           *P_TestString : " << *P_TestString;
  cout << "                                         *P_TestString1: " << *P_TestString1;
#endif

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCStringConstructors: starting P_TestString->ClassInvariant()" << endl;
#endif

  t1 = P_TestString->ClassInvariant();
  //  cout << *P_TestString;
  if (t1)
    cout << "MTestApp::TestCStringConstructors:                    t1(ClassInvariant) = \"TRUE\" (expected)" << endl;
  else
    cout << "MTestApp::TestCStringConstructors:                    t1(ClassInvariant) = \"FALSE\" (UNEXPECTED) " << endl;

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCStringConstructors: starting Copy Constructor" << endl;
#endif
  if (P_TestString != NULL)
    delete(P_TestString);
  P_TestString = new CString(*p_testString);

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCStringConstructors: starting P_TestString->ClassInvariant()" << endl;
#endif
  t2 = P_TestString->ClassInvariant();

  if (t2)
    cout << "MTestApp::TestCStringConstructors:                    t2(ClassInvariant) = \"TRUE\" (expected) " << endl;
  else
    cout << "MTestApp::TestCStringConstructors:                    t2(ClassInvariant) = \"FALSE\" (UNEXPECTED)" << endl;

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCStringConstructors: comparing result of init constructor with expected one" << endl;
#endif

  t3 = (P_TestString->EqualValue(*p_testString));
  if (t3)
    cout << "MTestApp::TestCStringConstructors:                    t3(CompareResult)  = \"TRUE\" (expected) " << endl;
  else
    cout << "MTestApp::TestCStringConstructors:                    t3(CompareResult)  = \"FALSE\" (UNEXPECTED)" << endl;

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCStringConstructors: starting init constructor(string)" << endl;
#endif
  tempStr = strdup("hello World!");
  if (P_TestString2 != NULL)
    delete(P_TestString2);
  tempStr = strdup("My Hello World!!!");
  P_TestString2 = new CString(tempStr);

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCStringConstructors: starting P_TestString2->ClassInvariant()" << endl;
#endif
  t4 = P_TestString2->ClassInvariant();

  if (t4)
    cout << "MTestApp::TestCStringConstructors:                    t4(ClassInvariant) = \"TRUE\" (expected) " << endl;
  else
    cout << "MTestApp::TestCStringConstructors:                    t4(ClassInvariant) = \"FALSE\" (UNEXPECTED)" << endl;

//#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCStringConstructors: starting P_TestString(=" << *P_TestString << ")->GetString() == tempStr" << endl;
//#endif
  if (strcmp(P_TestString2->StringToPChar(P_TestString2->GetString()), P_TestString2->StringToPChar(tempStr)) == 0)
    t5 = true;
  else
    t5 = false;
  if (t5)
    cout << "MTestApp::TestCStringConstructors:                    t5(P_TestString2->GetString(=" << P_TestString2->GetString() << ") == tempStr = <" << tempStr << ">     = \"TRUE\" (expected) " << endl;
  else
    cout << "MTestApp::TestCStringConstructors:                    t5(P_TestString2->GetString(=" << P_TestString2->GetString() << ") == tempStr = <" << tempStr << ">     = \"FALSE\" (UNEXPECTED)" << endl;

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCStringConstructors: starting P_TestString2->ClassInvariant()" << endl;
#endif
  t6 = P_TestString2->ClassInvariant();
  if (t6)
    cout << "MTestApp::TestCStringConstructors:                    t6(ClassInvariant)     = \"TRUE\" (expected) " << endl;
  else
    cout << "MTestApp::TestCStringConstructors:                    t6(ClassInvariant)     = \"FALSE\" (UNEXPECTED)" << endl;



#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCStringConstructors: starting init constructor(char*)" << endl;
#endif
  char p_MyChar[] = "hello World!";
  if (P_TestString2 != NULL)
    delete(P_TestString2);
  P_TestString2 = new CString(p_MyChar);

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCStringConstructors: starting P_TestString2->ClassInvariant()" << endl;
#endif
  t7 = P_TestString2->ClassInvariant();

  if (t7)
    cout << "MTestApp::TestCStringConstructors:                    t7(ClassInvariant) = \"TRUE\" (expected) " << endl;
  else
    cout << "MTestApp::TestCStringConstructors:                    t7(ClassInvariant) = \"FALSE\" (UNEXPECTED)" << endl;

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCStringConstructors: starting P_TestString2->GetString() == tempStr" << endl;
#endif
  if (strcmp(P_TestString2->Get(),p_MyChar) == 0)
    t8 = true;
  else
    t8 = false;
  if (t8)
    cout << "MTestApp::TestCStringConstructors:                    t8(P_TestString2->Get(=" << P_TestString2->Get() << ") == p_MyChar = <" << p_MyChar << ">     = \"TRUE\" (expected) " << endl;
  else
    cout << "MTestApp::TestCStringConstructors:                    t8(P_TestString2->Get(=" << P_TestString2->Get() << ") == p_MyChar = <" << p_MyChar << ">     = \"FALSE\" (UNEXPECTED)" << endl;

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCStringConstructors: starting P_TestString2->ClassInvariant()" << endl;
#endif
  t9 = P_TestString2->ClassInvariant();
  if (t9)
    cout << "MTestApp::TestCStringConstructors:                    t9(ClassInvariant)     = \"TRUE\" (expected) " << endl;
  else
    cout << "MTestApp::TestCStringConstructors:                    t9(ClassInvariant)     = \"FALSE\" (UNEXPECTED)" << endl;

  cout << "MTestApp::TestCStringConstructors: *P_TestString = " << *P_TestString << endl;;

//  delete tempStr;
  delete P_TestString;
  delete P_TestString1;
  delete P_TestString2;


  if (t1 && t2 && t3 && t4 && t5 && t6 && t7 && t8 && t9)
  {
    cout << "MTestApp::TestCStringConstructors:                       Test PASSED" << endl;
    cout << "================================================================" << endl;
    return true;
  }
//  else
//  {
    cout << "MTestApp::TestCStringConstructors:                       Test FAILED" << endl;
    cout << "================================================================" << endl;
/*  }
#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCStringConstructors: starting user defined init constructor test" << endl;
#endif
  char* p_tempStr = (char*)malloc(sizeof(char) * 255);
  cout << "PLEASE TYPE A STRING TO CONVERT: <x> to stop";
  cin >> p_tempStr;
  cout << endl << p_tempStr << endl;
  while (strcmp(p_tempStr, "x") != 0)
  {
    if (P_TestString != NULL)
      delete(P_TestString);
    P_TestString = new CString(p_tempStr);
    cout << "String = <" << *P_TestString << ">" << endl;
    t10 = P_TestString->ClassInvariant();
    if (t10)
      cout << "MTestApp::TestCStringConstructors:                    t10(ClassInvariant())     = \"TRUE\" " << endl;
    else
      cout << "MTestApp::TestCStringConstructors:                    t10(ClassInvariant())     = \"FALSE\" " << endl;
    cout << "PLEASE TYPE A STRING TO CONVERT: ";
    cin >> p_tempStr;
    cout << p_tempStr << endl;
  }
  cout << " " << endl;

  cout << "TEST SUCCESSFULL (y/n)";
  cin >> tempStr;
  cout << endl;
  if (*(new CString(tempStr)) == (*(new CString("y"))))
  {
    t10 = true;
    cout << "MTestApp::TestCStringConstructors:                             USERDEFINED TEST PASSED" << endl;
    cout << "==================================================================================" << endl;
  }
  else
  {
    t10 = false;
    cout << "MTestApp::TestCStringConstructors:                             USERDEFINED TEST FAILED" << endl;
    cout << "==================================================================================" << endl;
  }
  if (P_TestString != NULL)
    delete(P_TestString);
  if (P_TestString1 != NULL)
    delete(P_TestString1);
  if (P_TestString2 != NULL)
    delete(P_TestString2);

  if (t1 && t2 && t3 && t4 && t5 && t6 && t7 && t8 && t9 && t10)
  {
    cout << "MTestApp::TestCStringConstructors:                       Test PASSED" << endl;
    cout << "================================================================" << endl;
    cout << "TYPE any key TO CONTINUE" << endl;
    cin >> tempStr;

    return true;
  }
  cout << "MTestApp::TestCStringConstructors:                       Test FAILED" << endl;
  cout << "================================================================" << endl;
  cout << "TYPE any key TO CONTINUE" << endl;
  cin >> tempStr;
*/
  return false;

}

/*******************************************************************/

bool TestCStringClassInvAndCopyAndEVal()
{
  //  return TestConstructors("CString");
  bool t1, t2, t3, t4, t5, t6, t7, t8, t9, t10;
//  string tempStr = (char*)malloc(sizeof(char) * 255);
  CString *P_TestString = new CString();
  CString *P_TestString1 = new CString();
  CString *P_TestString2 = new CString();
  char* P_Char = new char[255];
  strcpy(P_Char, "My P_Char");

  cout << "MTestApp::TestCStringClassInvAndCopyAndEVal : P_TestString = <" << *P_TestString << ">" << endl;
  cout << "MTestApp::TestCStringClassInvAndCopyAndEVal: P_TestString1 = <" << *P_TestString1 << ">" << endl;
  cout << "MTestApp::TestCStringClassInvAndCopyAndEVal: P_TestString2 = <" << *P_TestString2 << ">" << endl;
  CString *p_testString = new CString("0123456789 Hello World!");

  t1 = P_TestString->EqualValue(*P_TestString1);
  if (t1)
    cout << "MTestApp::TestCStringClassInvAndCopyAndEVal:                    t1(P_TestString=" << *P_TestString << ")->EqualValue(P_TestString1=" << *P_TestString1 << ")     = \"TRUE\" (expected) " << endl;
  else
    cout << "MTestApp::TestCStringClassInvAndCopyAndEVal:                    t1(P_TestString="<< *P_TestString << ")->EqualValue(P_TestString1=" << *P_TestString1 << ")    = \"FALSE\" (UNEXPECTED)" << endl;

  cout << "MTestApp::TestCStringClassInvAndCopyAndEVal: Starting Inti Constructor" << endl;
  delete(P_TestString1);
  P_TestString1 = new CString(P_Char);

  if (strcmp(P_TestString1->Get(), P_Char) == 0)
    t2 = true;
  else
    t2 = false;
  if (t2)
    cout << "MTestApp::TestCStringClassInvAndCopyAndEVal:                    t2(strcmp(P_TestString1->Get()=" << *P_TestString1 << "), P_Char = " << P_Char << ")     = \"TRUE\" (expected) " << endl;
  else
    cout << "MTestApp::TestCStringClassInvAndCopyAndEVal:                    t2(strcmp(P_TestString1->Get()=" << *P_TestString1 << "), P_Char = " << P_Char << ")    = \"FALSE\" (UNEXPECTED)" << endl;

  cout << "MTestApp::TestCStringClassInvAndCopyAndEVal: Starting P_TestString->Copy(P_TestString1 = " << *P_TestString1 << ")" << endl;
  P_TestString2->Copy(*P_TestString1);
  t3 = P_TestString2->EqualValue(*P_TestString1);
  if (t3)
    cout << "MTestApp::TestCStringClassInvAndCopyAndEVal:                    t3(P_TestString2(=" << *P_TestString2 << ")->EqualValue(P_TestString1=" << *P_TestString1 << ")     = \"TRUE\" (expected) " << endl;
  else
    cout << "MTestApp::TestCStringClassInvAndCopyAndEVal:                    t3(P_TestString2(=" << *P_TestString2 << ")->EqualValue(P_TestString1=" << *P_TestString1 << ")     = \"FALSE\" (UNEXPECTED)" << endl;

  delete P_TestString;
  delete P_TestString1;
  delete P_TestString2;
  delete[] P_Char;

  return (t1 && t2 && t3);

}

/*******************************************************************/

bool TestCStringOperators()
{
  bool t1, t2, t3, t4, t5, t6, t7, t8, t9, t10;
  CString *P_TestString = new CString();
  CString *P_TestString1 = new CString();
  CString *P_TestString2 = new CString();
  CString *P_TestString3 = new CString();
  char* P_Char = new char[255];
  strcpy(P_Char, "My P_Char");

  cout << "MTestApp::TestCStringOperators : P_TestString = <" << *P_TestString << ">" << endl;
  cout << "MTestApp::TestCStringOperators : P_TestString1 = <" << *P_TestString1 << ">" << endl;
  cout << "MTestApp::TestCStringOperators : P_TestString2 = <" << *P_TestString2 << ">" << endl;

  cout << "MTestApp::TestCStringOperators: Starting test of operator=" << endl;
  *P_TestString = P_Char;
  cout << "MTestApp::TestCStringOperators: P_TestString = <" << *P_TestString << ">" << endl;

  delete(P_TestString1);
  P_TestString1 = new CString(P_Char);
  t1 = P_TestString->EqualValue(*P_TestString1);
  if (t1)
    cout << "MTestApp::TestCStringOperators:                    t1(P_TestString=" << *P_TestString << ")->EqualValue(P_TestString1=" << *P_TestString1 << ")     = \"TRUE\" (expected) " << endl;
  else
    cout << "MTestApp::TestCStringOperators:                    t1(P_TestString="<< *P_TestString << ")->EqualValue(P_TestString1=" << *P_TestString1 << ")    = \"FALSE\" (UNEXPECTED)" << endl;



  cout << "MTestApp::TestCStringOperators: Starting test of operator+" << endl;
//  char* p_tempstr = (char*)malloc(sizeof(char) * 255);
//  *P_TestString = ((*P_TestString) + (*P_Char));
  P_TestString2->Copy((*P_TestString2) + P_Char);
  cout << "MTestApp::TestCStringOperators: P_TestString2 = <" << *P_TestString2 << ">" << endl;
  t2 = P_TestString2->EqualValue(*P_TestString1);
  if (t2)
    cout << "MTestApp::TestCStringOperators:                    t2(P_TestString2=" << *P_TestString2 << ")->EqualValue(P_TestString1=" << *P_TestString1 << ")     = \"TRUE\" (expected) " << endl;
  else
    cout << "MTestApp::TestCStringOperators:                    t2(P_TestString2="<< *P_TestString2 << ")->EqualValue(P_TestString1=" << *P_TestString1 << ")    = \"FALSE\" (UNEXPECTED)" << endl;

  for (int i = 0; i < strlen(P_Char); i++)
  {
    *P_TestString3 = (*(P_TestString3)) + P_Char[i];
    cout << "MTestApp::TestCStringOperators: P_TestString3 = <" << *P_TestString3 << ">" << endl;
  }
  t3 = P_TestString3->EqualValue(*P_TestString1);
  if (t3)
    cout << "MTestApp::TestCStringOperators:                    t3(P_TestString2=" << *P_TestString2 << ")->EqualValue(P_TestString1=" << *P_TestString1 << ")     = \"TRUE\" (expected) " << endl;
  else
    cout << "MTestApp::TestCStringOperators:                    t3(P_TestString2="<< *P_TestString2 << ")->EqualValue(P_TestString1=" << *P_TestString1 << ")    = \"FALSE\" (UNEXPECTED)" << endl;




  delete(P_TestString1);
  P_TestString1 = new CString();
  (*(P_TestString1)) += (*(P_TestString3));
  t4 = P_TestString1->EqualValue(*P_TestString3);
  if (t4)
    cout << "MTestApp::TestCStringOperators:                    t4(P_TestString1=" << *P_TestString1 << ")->EqualValue(P_TestString3=" << *P_TestString3 << ")     = \"TRUE\" (expected) " << endl;
  else
    cout << "MTestApp::TestCStringOperators:                    t4(P_TestString1="<< *P_TestString1 << ")->EqualValue(P_TestString3=" << *P_TestString3 << ")    = \"FALSE\" (UNEXPECTED)" << endl;



  delete(P_TestString1);
  P_TestString1 = new CString();
  for (int i = 0; i < strlen(P_Char); i++)
  {
    (*(P_TestString1)) += P_Char[i];
  }
  t5 = P_TestString1->EqualValue(*P_TestString3);
  if (t5)
    cout << "MTestApp::TestCStringOperators:                    t5(P_TestString1=" << *P_TestString1 << ")->EqualValue(P_TestString3=" << *P_TestString3 << ")     = \"TRUE\" (expected) " << endl;
  else
    cout << "MTestApp::TestCStringOperators:                    t5(P_TestString1="<< *P_TestString1 << ")->EqualValue(P_TestString3=" << *P_TestString3 << ")    = \"FALSE\" (UNEXPECTED)" << endl;

  delete P_TestString;
  delete P_TestString1;
  delete P_TestString2;
  delete P_TestString3;
  delete[] P_Char;


  return (t1 && t2 && t3 && t4 && t5);
}
/*******************************************************************/

bool TestCString()
{
  return (TestCStringConstructors());// && TestCStringClassInvAndCopyAndEVal() && TestCStringOperators());
}

/*********************************************************************/

bool TestCFSConstructors()
{
  //  return TestConstructors("CFormatedString");
  bool t1, t2, t3, t4, t5, t6, t7;
  string tempstr;
  CString *p_testString = new CString("0123456789");
  CFormatedString *testfs = new CFormatedString();
  CFormatedString *testfs1 = new CFormatedString();
  CFormatedString *testfs2 = new CFormatedString();
  // Test: CFormatedString::Constructors
  // Tests included: CFormatedString::CalcFormatedString, CFormatedString::Dbl2ushort,
  //                 CFormatedString::operator=, CFormatedString::Copy,
  //                 CFormatedString::operator==, CFormatedString::EqualValue,
  //                 CFormatedString::operator<<, CFormatedString::Show
  // require : nothing
  // ensure  : t1 is "FALSE", t2, t3, t4, t5, t6 and t7 are "TRUE"

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFSConstructors: starting standard constructor" << endl;
#endif
  if (testfs != NULL)
    delete(testfs);
  testfs = new CFormatedString();
#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFSConstructors: standard constructor ready: testfs = <" << *testfs << ">" << endl;
#endif

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFSConstructors: starting init constructor(12345678,5)" << endl;
#endif
  if (testfs1 != NULL)
    delete(testfs1);
  testfs1 = new CFormatedString(12345678, 5); // 'Length' lower than length of 'Number'
#ifdef _DEBUG_TESTAPP_
  cout << "                                         *testfs1: " << *testfs1;
  //  cout << "                                         *testfs2: " << *testfs2;
#endif

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFSConstructors: starting testfs->ClassInvariant()" << endl;
#endif

  t1 = testfs->ClassInvariant();
  //  cout << *testfs;
  if (t1)
    cout << "MTestApp::TestCFSConstructors:                    t1(ClassInvariant) = \"TRUE\" " << endl;
  else
    cout << "MTestApp::TestCFSConstructors:                    t1(ClassInvariant) = \"FALSE\" (exp) " << endl;

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFSConstructors: starting initialization constructor" << endl;
#endif
  if (testfs != NULL)
    delete(testfs);
  testfs = new CFormatedString(123456789,10);

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFSConstructors: starting testfs->ClassInvariant()" << endl;
#endif
  t2 = testfs->ClassInvariant();

  if (t2)
    cout << "MTestApp::TestCFSConstructors:                    t2(ClassInvariant) = \"TRUE\" (exp) " << endl;
  else
    cout << "MTestApp::TestCFSConstructors:                    t2(ClassInvariant) = \"FALSE\" " << endl;

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFSConstructors: comparing result of init constructor with expected one" << endl;
#endif

  if (strcmp(testfs->GetString()->Get(), p_testString->Get()) == 0)
    t3 = true;
  else
    t3 = false;
  if (t3)
    cout << "MTestApp::TestCFSConstructors:                    t3(CompareResult)  = \"TRUE\" (exp) " << endl;
  else
    cout << "MTestApp::TestCFSConstructors:                    t3(CompareResult)  = \"FALSE\" " << endl;

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFSConstructors: starting copy constructor" << endl;
#endif
  if (testfs2 != NULL)
    delete(testfs2);
  testfs2 = new CFormatedString(*testfs);

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFSConstructors: starting testfs2(=" << *testfs2 << ")->ClassInvariant()" << endl;
#endif
  t4 = testfs2->ClassInvariant();

  if (t4)
    cout << "MTestApp::TestCFSConstructors:                    t4(ClassInvariant) = \"TRUE\" (exp) " << endl;
  else
    cout << "MTestApp::TestCFSConstructors:                    t4(ClassInvariant) = \"FALSE\" " << endl;

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFSConstructors: starting testfs(=" << *testfs << ")->EqualValue(*testfs2=" << *testfs2 << ")" << endl;
  cout << "MTestApp::TestCFSConstructors: testfs->GetLength(=" << testfs->GetLength() << "), testfs2->GetLength=" << testfs2->GetLength() << ")" << endl;
  cout << "MTestApp::TestCFSConstructors: testfs->GetNumber(=" << testfs->GetNumber() << "), testfs2->GetNumber=" << testfs2->GetNumber() << ")" << endl;
#endif
  t5 = testfs->EqualValue(*testfs2);
  if (t5)
    cout << "MTestApp::TestCFSConstructors:                    t5(EqualValue)     = \"TRUE\" (exp) " << endl;
  else
    cout << "MTestApp::TestCFSConstructors:                    t5(EqualValue)     = \"FALSE\" " << endl;

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFSConstructors: starting testfs1->ClassInvariant()" << endl;
#endif
  t6 = testfs1->ClassInvariant();
  if (t6)
    cout << "MTestApp::TestCFSConstructors:                    t6(ClassInvariant)     = \"TRUE\" (exp) " << endl;
  else
    cout << "MTestApp::TestCFSConstructors:                    t6(ClassInvariant)     = \"FALSE\" " << endl;


  cout << "MTestApp::TestCFSConstructors: *testfs = " << *testfs << endl;;
  if (!t1 && t2 && t3 && t4 && t5 && t6)
  {
    cout << "MTestApp::TestCFSConstructors:                       Test PASSED" << endl;
    cout << "================================================================" << endl;
  }
  else
  {
    cout << "MTestApp::TestCFSConstructors:                       Test FAILED" << endl;
    cout << "================================================================" << endl;
  }
#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFSConstructors: starting user defined init constructor test" << endl;
#endif
  unsigned int len;
  unsigned int num;
  cout << "PLEASE TYPE A NUMBER TO CONVERT: ";
  cin >> num;
  cout << num << endl;
  cout << "PLEASE TYPE THE LENGTH (0 FOR STOP): ";
  cin >> len;
  cout << len << endl;
  while (len > 0)
  {
    if (testfs != NULL)
      delete(testfs);
    testfs = new CFormatedString(num,len);
    cout << "FormatedString = " << *testfs << ", Number = " << testfs->GetNumber() << ", Length = " << testfs->GetLength() << endl;
    t7 = testfs->ClassInvariant();
    if (t7)
      cout << "MTestApp::TestCFSConstructors:                    t7(ClassInvariant())     = \"TRUE\" " << endl;
    else
      cout << "MTestApp::TestCFSConstructors:                    t7(ClassInvariant())     = \"FALSE\" " << endl;

    cout << "PLEASE TYPE A NUMBER TO CONVERT: ";
    cin >> num;
    cout << endl << num << endl;
    cout << "PLEASE TYPE THE LENGTH (0 FOR STOP): ";
    cin >> len;
    cout << endl << len << endl;
  }
  cout << " " << endl;

  cout << "TEST SUCCESSFULL (y/n)";
  cin >> tempstr;
  cout << endl;
  if (*(new CString(tempstr)) == (*(new CString("y"))))
  {
    t7 = true;
    cout << "MTestApp::TestCFSConstructors:                             USERDEFINED TEST PASSED" << endl;
    cout << "==================================================================================" << endl;
  }
  else
  {
    t7 = false;
    cout << "MTestApp::TestCFSConstructors:                             USERDEFINED TEST FAILED" << endl;
    cout << "==================================================================================" << endl;
  }
  if (testfs != NULL)
    delete(testfs);
  if (testfs1 != NULL)
    delete(testfs1);
  if (testfs2 != NULL)
    delete(testfs2);

  delete p_testString;
/*  delete testfs;
  delete testfs1;
  delete testfs2;
*/

  if (!t1 && t2 && t3 && t4 && t5 && t6 && t7)
  {
    cout << "MTestApp::TestCFSConstructors:                       Test PASSED" << endl;
    cout << "================================================================" << endl;
    cout << "TYPE any key TO CONTINUE" << endl;
    cin >> tempstr;

    return true;
  }
  cout << "MTestApp::TestCFSConstructors:                       Test FAILED" << endl;
  cout << "================================================================" << endl;
  cout << "TYPE any key TO CONTINUE" << endl;
  cin >> tempstr;
  return false;

  //return (t1 && t2 && t3);
}

/**********************************************************************/

bool TestCFSCalcFSAndEqualOpAndEVal()
{
  bool t1, t2, t3, t4, t5;
  string tempstr;// = new CString();

  CFormatedString *fs1 = new CFormatedString(123456789, 10);
  CFormatedString *fs2 = new CFormatedString(987654321, 10);
  CFormatedString *fs3 = new CFormatedString();

  // Test: CFormatedString::CalcFormatedString, CFormatedString::operator=,
  //       CFormatedString::operator==
  // Tests included: Standard Constructor, initialization constructor,
  //                 CFormatedString::EqualValue,
  //                 CFormatedString::Dbl2ushort, CFormatedString::IntToChar,
  //                 CFormatedString::operator<<, CFormatedString::Show

  // require : fs1 is a 'ClassInvariant()' instance of 'CFormatedString'.
  //           fs2 is an instance of 'CFormatedString'.
  // ensure  : t1 is "FALSE", t2, t3, t4 and t5 are "TRUE"

  string expectedString = "0987654321";

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFSCalcFSAndEqualOpAndEVal:       *fs1: <" << *fs1 << ">" << endl;
  cout << "                                                *fs2: <" << *fs2 << ">" << endl;
  cout << "                                                *fs3: <" << *fs3 << ">" << endl;
#endif

  t1 = (fs1->GetNumber() == fs3->GetNumber()) &&
       (fs1->GetLength() == fs3->GetLength()) &&
       ((*(fs1->GetString())) == (*(fs3->GetString())));
  if (t1)
    cout << "MTestApp::TestCFSCalcFSAndEqualOpAndEVal:         t1(Comparison 1 and 3) = \"TRUE\" " << endl;
  else
    cout << "MTestApp::TestCFSCalcFSAndEqualOpAndEVal:         t1(Comparison 1 and 3) = \"FALSE\" (exp)" << endl;

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFSCalcFSAndEqualOpAndEVal: starting operator=(3,1)" << endl;
#endif
  *fs3 = *fs1;
#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFSCalcFSAndEqualOpAndEVal: fs1->GetString() = " << *(fs1->GetString()) << " fs3->GetString() " << *(fs3->GetString()) << endl;
#endif

  t2 = (fs1->GetNumber() == fs3->GetNumber()) &&
       (fs1->GetLength() == fs3->GetLength()) &&
       ((*(fs1->GetString())) == (*(fs3->GetString())));
  if (t2)
    cout << "MTestApp::TestCFSCalcFSAndEqualOpAndEVal:         t2(Comparison 1 and 3) = \"TRUE\" (expected)" << endl;
  else
    cout << "MTestApp::TestCFSCalcFSAndEqualOpAndEVal:         t2(Comparison 1 and 3) = \"FALSE\" (UNEXPECTED) " << endl;

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFSCalcFSAndEqualOpAndEVal: starting CalcFormatedString" << endl;
#endif
  fs3->CalcFormatedString();
#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFSCalcFSAndEqualOpAndEVal: fs1->GetString() = " << *(fs1->GetString()) << " fs3->GetString() " << *(fs3->GetString()) << endl;
#endif

  t3 = (fs1->GetNumber() == fs3->GetNumber()) &&
       (fs1->GetLength() == fs3->GetLength()) &&
       ((*(fs1->GetString())) == (*(fs3->GetString())));
  if (t3)
    cout << "MTestApp::TestCFSCalcFSAndEqualOpAndEVal:         t3(Comparison 1 and 3) = \"TRUE\" (expected)" << endl;
  else
    cout << "MTestApp::TestCFSCalcFSAndEqualOpAndEVal:         t3(Comparison 1 and 3) = \"FALSE\" (UNEXPECTED)" << endl;

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFSCalcFSAndEqualOpAndEVal: starting EqualValue(3,1)" << endl;
#endif
  t4 = fs3->EqualValue(*fs1);

  if (t4)
    cout << "MTestApp::TestCFSCalcFSAndEqualOpAndEVal:         t4(EqualValue)         = \"TRUE\" (exp)" << endl;
  else
    cout << "MTestApp::TestCFSCalcFSAndEqualOpAndEVal:         t4(EqualValue)         = \"FALSE\" " << endl;

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFSCalcFSAndEqualOpAndEVal: starting operator==" << endl;
#endif
  if (*fs1 == *fs3)
    t5 = true;
  else
    t5 = false;

  if (t5)
    cout << "MTestApp::TestCFSCalcFSAndEqualOpAndEVal:         t5(*fs1 == *fs3)       = \"TRUE\" (exp)" << endl;
  else
    cout << "MTestApp::TestCFSCalcFSAndEqualOpAndEVal:         t5(*fs1 == *fs3)       = \"FALSE\" " << endl;

  cout << " " << endl;
  if (!t1 && t2 && t3 && t4 && t5)
  {
    cout << "MTestApp::TestCFSCalcFSAndEqualOpAndEVal:            Test PASSED" << endl;
    cout << "================================================================" << endl;
    cout << "TYPE any key TO CONTINUE" << endl;
    cin >> tempstr;
    return true;
  }
  cout << "MTestApp::TestCFSCalcFSAndEqualOpAndEVal:            Test FAILED" << endl;
  cout << "================================================================" << endl;
  cout << "TYPE any key TO CONTINUE" << endl;
  cin >> tempstr;
  if (fs1 != NULL)
    delete(fs1);
  if (fs2 != NULL)
    delete(fs2);
  if (fs3 != NULL)
    delete(fs3);
  return false;
}

/**********************************************************************/

bool TestCFSClassInvAndCopyAndEVal()
{
  bool t1, t2, t3, t4, t5;
  string tempstr;// = new CString();

  CFormatedString *fs1 = new CFormatedString(123456789, 10);
  CFormatedString *fs2 = new CFormatedString();

  // Test: CFormatedString::ClassInvariant, CFormatedString::Copy and
  //       CFormatedString::EqualValue
  // Tests included: Standard Constructor, CFormatedString::CalcFormatedString,
  //                 CFormatedString::Dbl2ushort, CFormatedString::IntToChar,
  //                 CFormatedString::operator=, CFormatedString::operator==,
  //                 CFormatedString::operator<<, CFormatedString::Show

  // require : fs1 is a 'ClassInvariant()' instance of 'CFormatedString'.
  //           fs2 is an instance of 'CFormatedString'.
  // ensure  : t2 is "FALSE", t1, t3, t4 and t5 are "TRUE"

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFSClassInvAndCopyAndEVal: fs1: " << *fs1 << endl;
  cout << "                                         fs2: " << *fs2 << endl;
#endif

  t1 = fs1->ClassInvariant();
  if (t1)
    cout << "MTestApp::TestCFSClassInvAndCopyAndEVal:          t1(ClassInvariant) = \"TRUE\" (exp) " << endl;
  else
    cout << "MTestApp::TestCFSClassInvAndCopyAndEVal:          t1(ClassInvariant) = \"FALSE\" " << endl;

  t2 = fs1->EqualValue(*fs2);
  if (t2)
    cout << "MTestApp::TestCFSClassInvAndCopyAndEVal:          t2(EqualValue)     = \"TRUE\" " << endl;
  else
    cout << "MTestApp::TestCFSClassInvAndCopyAndEVal:          t2(EqualValue)     = \"FALSE\" (exp) " << endl;

  t3 = fs2->Copy(*fs1);
#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFSClassInvAndCopyAndEVal: fs2->Copy(*fs1) liefert *fs2 = " << *fs2 << endl;
#endif
  if (t3)
    cout << "MTestApp::TestCFSClassInvAndCopyAndEVal:          t3(Copy)           = \"TRUE\" (exp) " << endl;
  else
    cout << "MTestApp::TestCFSClassInvAndCopyAndEVal:          t3(Copy)           = \"FALSE\" " << endl;

  t4 = fs2->ClassInvariant();
  if (t4)
    cout << "MTestApp::TestCFSClassInvAndCopyAndEVal:          t4(ClassInvariant) = \"TRUE\" (exp) " << endl;
  else
    cout << "MTestApp::TestCFSClassInvAndCopyAndEVal:          t4(ClassInvariant) = \"FALSE\" " << endl;

  t5 = fs1->EqualValue(*fs2);
  if (t5)
    cout << "MTestApp::TestCFSClassInvAndCopyAndEVal:          t5(EqualValue)     = \"TRUE\" (exp) " << endl;
  else
    cout << "MTestApp::TestCFSClassInvAndCopyAndEVal:          t5(EqualValue)     = \"FALSE\" " << endl;

  cout << " " << endl;

  if (t1 && !t2 && t3 && t4 && t5)
  {
    cout << "MTestApp::TestCFSClassInvAndCopyAndEVal:             Test PASSED" << endl;
    cout << "================================================================" << endl;
    cout << "TYPE any key TO CONTINUE" << endl;
    cin >> tempstr;
    return true;
  }
  cout << "MTestApp::TestCFSClassInvAndCopyAndEVal:             Test FAILED" << endl;
  cout << "================================================================" << endl;
  cout << "TYPE any key TO CONTINUE" << endl;
  cin >> tempstr;
  if (fs1 != NULL)
    delete(fs1);
  if (fs2 != NULL)
    delete(fs2);
  return false;
}

/**********************************************************************/

bool TestCFormatedString()
{
  return (TestCFSConstructors() && TestCFSCalcFSAndEqualOpAndEVal() && TestCFSClassInvAndCopyAndEVal());
}

/*********************************************************************/

bool TestCFitsConstructors()
{
  //  return TestConstructors("CFormatedString");
  bool t1, t2, t3, t4, t5, t6, t7;
  string tempstr;// = new CString();
  int NRows = 13;
  int NCols = 7;
  CString *p_testString = new CString();
  CFits *testfits = new CFits();
  CFits *testfits1 = new CFits();
  CFits *testfits2 = new CFits();
  // Test: CFormatedString::Constructors
  // Tests included: CFormatedString::operator=, CFormatedString::Copy,
  //                 CFormatedString::operator==, CFormatedString::EqualValue,
  //                 CFormatedString::operator<<, CFormatedString::Show
  // require : nothing
  // ensure  : t1, t2 and t7 are "FALSE", t3, t4, t5 and t6 are "TRUE"

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsConstructors: starting standard constructor" << endl;
#endif
  if (testfits != NULL)
    delete(testfits);
  testfits = new CFits();
  t1 = (testfits == NULL);
  if (t1)
  {
    cout << "MTestApp::TestCFitsConstructors:                                   t1(== NULL) = \"TRUE\" (UNEXPECTED)" << endl;
    return false;
  }
  else
    cout << "MTestApp::TestCFitsConstructors:                               t1(== NULL) = \"FALSE\" (expected) " << endl;

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsConstructors: starting init constructor(p_testString, NCols, NRows)" << endl;
#endif
  if (testfits1 != NULL)
    delete(testfits1);
  testfits1 = new CFits(*p_testString, NCols, NRows);
#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsConstructors:           *testfits : " << *testfits;
  cout << "                                         *testfits1: " << *testfits1;
  //  cout << "                                         *testfits2: " << *testfits2;
#endif

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsConstructors: starting testfits->ClassInvariant()" << endl;
#endif

  t2 = testfits->ClassInvariant();
  //  cout << *testfits;
  if (t2)
  {
    cout << "MTestApp::TestCFitsConstructors:                    t2(ClassInvariant) = \"TRUE\" (UNEXPECTED)" << endl;
    return false;
  }
  else
    cout << "MTestApp::TestCFitsConstructors:                    t2(ClassInvariant) = \"FALSE\" (expected) " << endl;

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsConstructors: starting initialization constructor" << endl;
#endif
  if (testfits != NULL)
    delete(testfits);
  p_testString->Copy(CString("test/test_small.fits"));
  testfits = new CFits(*p_testString, NCols, NRows);

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsConstructors: starting testfits->ClassInvariant()" << endl;
#endif
  t3 = testfits->ClassInvariant();

  if (t3)
    cout << "MTestApp::TestCFitsConstructors:                    t3(ClassInvariant) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsConstructors:                    t3(ClassInvariant) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsConstructors: comparing result of init constructor with expected one" << endl;
#endif

  CString tempFileName = testfits->GetFileName();
  t4 = (tempFileName.EqualValue(*p_testString));
  if (t4)
    cout << "MTestApp::TestCFitsConstructors:                    t4(CompareResult)  = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsConstructors:                    t4(CompareResult)  = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsConstructors: starting copy constructor" << endl;
#endif
  if (testfits2 != NULL)
    delete(testfits2);
  testfits2 = new CFits(*testfits);

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsConstructors: starting testfits2->ClassInvariant()" << endl;
#endif
  t5 = testfits2->ClassInvariant();

  if (t5)
    cout << "MTestApp::TestCFitsConstructors:                    t5(ClassInvariant) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsConstructors:                    t5(ClassInvariant) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsConstructors: starting testfits->EqualValue(*testfits2)" << endl;
#endif
  t6 = testfits->EqualValue(*testfits2);
  if (t6)
    cout << "MTestApp::TestCFitsConstructors:                      t6(EqualValue)     = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsConstructors:                      t6(EqualValue)     = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsConstructors: starting testfits1->ClassInvariant()" << endl;
#endif
  t7 = testfits1->ClassInvariant();
  if (t7)
  {
    cout << "MTestApp::TestCFitsConstructors:                    t7(ClassInvariant)     = \"TRUE\" (UNEXPECTED) " << endl;
    return false;
  }
  else
    cout << "MTestApp::TestCFitsConstructors:                    t7(ClassInvariant)     = \"FALSE\" (expected)" << endl;

  cout << "MTestApp::TestCFitsConstructors: *testfits = " << *testfits << endl;;
  if (!t1 && !t2 && t3 && t4 && t5 && t6)
  {
    cout << "MTestApp::TestCFitsConstructors:                       Test PASSED" << endl;
    cout << "================================================================" << endl;
  }
  else
  {
    cout << "MTestApp::TestCFitsConstructors:                       Test FAILED" << endl;
    cout << "================================================================" << endl;
    if (testfits != NULL)
      delete(testfits);
    if (testfits1 != NULL)
      delete(testfits1);
    if (testfits2 != NULL)
      delete(testfits2);
    return false;
  }

  delete p_testString;
  if (testfits != NULL)
    delete(testfits);
  if (testfits1 != NULL)
    delete(testfits1);
  if (testfits2 != NULL)
    delete(testfits2);

  return true;

}

/*********************************************************************/

bool TestCFitsClassInvAndCopyAndEVal()
{
  //  return TestConstructors("CFormatedString");
  bool t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14;
  string tempstr;// = new CString();
  int NCols = 9;
  int NRows = 17;
  CString *p_testString = new CString("test/test_small.fits");
  CFits *testfits = new CFits();
  CFits *testfits1 = new CFits();
  CFits *testfits2 = new CFits();
  // Test: CFormatedString::Constructors
  // Tests included: CFits::operator=, CFits::Copy,
  //                 CFits::operator==, CFits::EqualValue,
  //                 CFits::operator<<, CFits::Show, CFits::ReadArray
  // require : nothing
  // ensure  : t2, t4, t6, and t9 are "FALSE", t1, t3, t5, t6, t7, t8, t10, t11, t12, t13, and t14 are "TRUE"

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsClassInvAndCopyAndEval: starting init constructor" << endl;
#endif
  if (testfits != NULL)
    delete(testfits);
  testfits = new CFits(*p_testString, NCols, NRows);
#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsClassInvAndCopyAndEval:           *testfits : " << *testfits;
#endif

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsClassInvAndCopyAndEval: starting testfits->ClassInvariant()" << endl;
#endif

  t1 = testfits->ClassInvariant();
  //  cout << *testfits;
  if (t1)
    cout << "MTestApp::TestCFitsClassInvAndCopyAndEval:                    t1(ClassInvariant) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsClassInvAndCopyAndEval:                    t1(ClassInvariant) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsClassInvAndCopyAndEval: starting standard constructor" << endl;
#endif
  if (testfits1 != NULL)
    delete(testfits1);
  testfits1 = new CFits();

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsClassInvAndCopyAndEval: starting testfits1->ClassInvariant()" << endl;
#endif
  t2 = testfits1->ClassInvariant();

  if (t2)
  {
    cout << "MTestApp::TestCFitsClassInvAndCopyAndEval:                    t2(ClassInvariant) = \"TRUE\" (UNEXPECTED)" << endl;
    return false;
  }
  else
    cout << "MTestApp::TestCFitsClassInvAndCopyAndEval:                    t2(ClassInvariant) = \"FALSE\" (expected) " << endl;

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsClassInvAndCopyAndEval: starting standard constructor" << endl;
#endif
  if (testfits2 != NULL)
    delete(testfits2);
  testfits2 = new CFits();

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsClassInvAndCopyAndEval: starting testfits->ClassInvariant()" << endl;
#endif
  t3 = testfits->ClassInvariant();

  if (t3)
    cout << "MTestApp::TestCFitsClassInvAndCopyAndEval:                    t3(ClassInvariant) = \"TRUE\" (expected)" << endl;
  else
  {
    cout << "MTestApp::TestCFitsClassInvAndCopyAndEval:                    t3(ClassInvariant) = \"FALSE\" (UNEXPECTED) " << endl;
    return false;
  }

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsClassInvAndCopyAndEval: starting testfits1->Copy(testfits2)" << endl;
#endif
  t4 = testfits1->Copy( *testfits2 );

  if (t4)
  {
    cout << "MTestApp::TestCFitsClassInvAndCopyAndEval:                    t4(Copy) = \"TRUE\" (UNEXPECTED)" << endl;
    return false;
  }
  else
    cout << "MTestApp::TestCFitsClassInvAndCopyAndEval:                    t4(Copy) = \"FALSE\" (expected) " << endl;

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsClassInvAndCopyAndEval: starting testfits1->Copy(testfits)" << endl;
#endif
  t5 = testfits1->Copy( *testfits );

  if (t5)
    cout << "MTestApp::TestCFitsClassInvAndCopyAndEval:                    t5(Copy) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsClassInvAndCopyAndEval:                    t5(Copy) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsClassInvAndCopyAndEval: starting testfits1->Equal( (testfits)" << endl;
#endif
  t6 = testfits1->Equal( *testfits );

  if (t6)
  {
    cout << "MTestApp::TestCFitsClassInvAndCopyAndEval:                    t6(Equal) = \"TRUE\" (UNEXPECTED)" << endl;
    return false;
  }
  else
    cout << "MTestApp::TestCFitsClassInvAndCopyAndEval:                    t6(Equal) = \"FALSE\" (expected) " << endl;

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsClassInvAndCopyAndEval: starting testfits1->EqualValue( (testfits)" << endl;
#endif
  t7 = testfits1->EqualValue( *testfits );

  if (t7)
    cout << "MTestApp::TestCFitsClassInvAndCopyAndEval:                    t7(EqualValue) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsClassInvAndCopyAndEval:                    t7(EqualValue) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsClassInvAndCopyAndEval: starting testfits->ReadArray ()" << endl;
#endif
  t8 = testfits->ReadArray(  );

  if (t8)
    cout << "MTestApp::TestCFitsClassInvAndCopyAndEval:                    t8(ReadArray) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsClassInvAndCopyAndEval:                    t8(ReadArray) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }
  cout << "MTestApp::TestCFitsClassInvAndCopyAndEval: testfits = " << *testfits << endl;

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsClassInvAndCopyAndEval: starting testfits1(=" << *testfits1 << ")->EqualValue(testfits=" << *testfits << ")" << endl;
#endif
  t9 = testfits1->EqualValue( *testfits );

  if (t9)
  {
    cout << "MTestApp::TestCFitsClassInvAndCopyAndEval:                    t9(EqualValue) = \"TRUE\" (UNEXPECTED) " << endl;
    return false;
  }
  else
    cout << "MTestApp::TestCFitsClassInvAndCopyAndEval:                    t9(EqualValue) = \"FALSE\" (expected)" << endl;

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsClassInvAndCopyAndEval: starting testfits1(=" << *testfits1 << ")->ReadArray ()" << endl;
#endif
  t10 = testfits1->ReadArray(  );

  if (t10)
    cout << "MTestApp::TestCFitsClassInvAndCopyAndEval:                    t10(testfits1->ReadArray) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsClassInvAndCopyAndEval:                    t10(ReadArray) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }
  cout << "MTestApp::TestCFitsClassInvAndCopyAndEval: testfits1 = " << *testfits1 << endl;

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsClassInvAndCopyAndEval: starting testfits1(=" << *testfits1 << ")->EqualValue( (testfits=" << *testfits << ")" << endl;
#endif
  t11 = testfits1->EqualValue( *testfits );

  if (t11)
    cout << "MTestApp::TestCFitsClassInvAndCopyAndEval:                    t11(EqualValue) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsClassInvAndCopyAndEval:                    t11(EqualValue) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsClassInvAndCopyAndEval: starting testfits1->SetFileName ()" << endl;
#endif
  t12 = testfits1->SetFileName( (*(new CString("test/test_small_new.fits") )));

  if (t12)
    cout << "MTestApp::TestCFitsClassInvAndCopyAndEval:                    t12(SetFileName) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsClassInvAndCopyAndEval:                    t12(SetFileName) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsClassInvAndCopyAndEval: starting testfits1->WriteArray() " << endl;
#endif
  t13 = testfits1->WriteArray(  );

  if (t13)
    cout << "MTestApp::TestCFitsClassInvAndCopyAndEval:                    t13(WriteArray) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsClassInvAndCopyAndEval:                    t13(WriteArray) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsClassInvAndCopyAndEval: starting testfits1->ClassInvariant()" << endl;
#endif
  t14 = testfits1->ClassInvariant();

  if (t14)
    cout << "MTestApp::TestCFitsClassInvAndCopyAndEval:                    t14(ClassInvariant) = \"TRUE\" (expected)" << endl;
  else
  {
    cout << "MTestApp::TestCFitsClassInvAndCopyAndEval:                    t14(ClassInvariant) = \"FALSE\" (UNEXPECTED) " << endl;
    return false;
  }


  cout << "MTestApp::TestCFitsClassInvAndCopyAndEval: *testfits = " << *testfits << endl;;
  if (t1 && !t2 && t3 && !t4 && t5 && !t6 && t7 && t8 && !t9 && t10 && t11 && t12 && t13 && t14)
  {
    cout << "MTestApp::TestCFitsClassInvAndCopyAndEval:                       Test PASSED" << endl;
    cout << "================================================================" << endl;
  }
  else
  {
    cout << "MTestApp::TestCFitsClassInvAndCopyAndEval:                       Test FAILED" << endl;
    cout << "================================================================" << endl;
    if (testfits != NULL)
      delete(testfits);
    if (testfits1 != NULL)
      delete(testfits1);
    if (testfits2 != NULL)
      delete(testfits2);
    return false;
  }
  if (testfits != NULL)
    delete(testfits);
  if (testfits1 != NULL)
    delete(testfits1);
  if (testfits2 != NULL)
    delete(testfits2);

  return true;

}

/** ***********************************************************/

bool TestCFitsMedian()
{
  //  return TestConstructors("CFormatedString");
  bool t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14;
  string tempstr;// = new CString();
  int NCols = 9;
  int NRows = 17;
  int m;
  CString *p_testString = new CString("test/test_small.fits");
  CFits *testfits = new CFits();
  CFits *testfits1 = new CFits();
  CFits *testfits2 = new CFits();
  firstIndex i;
  double Result;
  Array<double, 1> *P_OddArray = new Array<double, 1>(11);
  (*P_OddArray) = i;
  Array<double, 1> *P_TempOddArray = new Array<double, 1>(11);
  (*P_TempOddArray) = i;
  Array<double, 1> *P_EvenArray = new Array<double, 1>(10);
  (*P_EvenArray) = i;
  Array<double, 1> *P_TempEvenArray;
  Array<double, 1> *P_TestEvenArray = new Array<double, 1>(10);
  (*P_TestEvenArray) = i;
  Array<double, 1> *P_TestOddArray = new Array<double, 1>(10);
  (*P_TestOddArray) = i;
  // Test: CFormatedString::Median
  // Tests included: CFits::operator=, CFits::Copy,
  //                 CFits::operator==, CFits::EqualValue,
  //                 CFits::operator<<, CFits::Show, CFits::ReadArray
  // require : nothing
  // ensure  : t2, t4, t6, and t9 are "FALSE", t1, t3, t5, t6, t7, t8, t10, t11, t12, t13, and t14 are "TRUE"

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsMedian: starting init constructor" << endl;
#endif
  if (testfits != NULL)
    delete(testfits);
  testfits = new CFits(*p_testString, NCols, NRows);
#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsMedian:           *testfits : " << *testfits;
#endif

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsMedian: P_OddArray = " << *P_OddArray << endl;
  cout << "MTestApp::TestCFitsMedian: P_EvenArray = " << *P_EvenArray << endl;
  cout << "MTestApp::TestCFitsMedian: starting testfits->Swap((*P_TempOddArray)(4), (*P_TempOddArray(5)))" << endl;
#endif
  testfits->Swap((*P_TempOddArray)(4), (*P_TempOddArray)(5));
  cout << "MTestApp::TestCFitsMedian: P_TempOddArray = " << *P_TempOddArray << endl;
  if ((*P_TempOddArray)(4) == (*P_OddArray)(5))
    t1 = true;
  else
    t1 = false;
  if (t1)
    cout << "MTestApp::TestCFitsMedian:                    t1(Swap) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsMedian:                    t1(Swap) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsMedian: starting testfits->Swap((*P_TempOddArray)(0), (*P_TempOddArray(7)))" << endl;
#endif
  testfits->Swap((*P_TempOddArray)(0), (*P_TempOddArray)(7));
  cout << "MTestApp::TestCFitsMedian: P_TempOddArray = " << *P_TempOddArray << endl;
  if ((*P_TempOddArray)(0) == (*P_OddArray)(7))
    t2 = true;
  else
    t2 = false;
  if (t2)
    cout << "MTestApp::TestCFitsMedian:                    t2(Swap) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsMedian:                    t2(Swap) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsMedian: starting testfits->Select(P_OddArray)" << endl;
#endif
  m = 0;
  t3 = true;
  while (m < P_OddArray->size())
  {
    Result = testfits->Select(*P_OddArray, m+1);
    cout << "MTestApp::TestCFitsMedian: Result of Select(P_OddArray, m+1=" << m+1 << ") = " << Result << endl;
    if (abs(Result - (double)m) > 0.00001)
    {
      t3 = false;
      break;
    }
    m++;
  }
  if (t3)
    cout << "MTestApp::TestCFitsMedian:                    t3(Select(OddArray)) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsMedian:                    t3(Select(OddArray)) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsMedian: starting testfits->Median(P_OddArray)" << endl;
#endif
  Result = testfits->Median(*P_OddArray);
  cout << "MTestApp::TestCFitsMedian: Result of Median(P_OddArray) = " << Result << endl;
  if (abs(Result - 5.) < 0.00001)
    t4 = true;
  else
    t4 = false;
  if (t4)
    cout << "MTestApp::TestCFitsMedian:                    t4(Median(OddArray)) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsMedian:                    t4(Median(OddArray)) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsMedian: starting testfits->Median(P_EvenArray)" << endl;
#endif
  Result = testfits->Median(*P_EvenArray);
  cout << "MTestApp::TestCFitsMedian: Result of Median(P_EvenArray) = " << Result << endl;
  if (abs(Result - 4.5) < 0.00001)
    t5 = true;
  else
    t5 = false;
  if (t5)
    cout << "MTestApp::TestCFitsMedian:                    t5(Median(EvenArray)) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsMedian:                    t5(Median(EvenArray)) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsMedian: starting testfits->Median(P_EvenArray, Even)" << endl;
#endif
  Array<CString, 1> CS_A1_Args_Median(10);
  CS_A1_Args_Median = CString(" ");
  CS_A1_Args_Median(0) = CString("EVEN");
  void **PP_Args_Median;
  PP_Args_Median = (void**)malloc(sizeof(void*) * 10);

  Result = testfits->Median(*P_EvenArray, CS_A1_Args_Median, PP_Args_Median);
  cout << "MTestApp::TestCFitsMedian: Result of Median(P_EvenArray, EVEN) = " << Result << endl;
  if (abs(Result - 4.) < 0.00001)
    t6 = true;
  else
    t6 = false;
  if (t6)
    cout << "MTestApp::TestCFitsMedian:                    t6(Median(EvenArray, EVEN)) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsMedian:                    t6(Median(EvenArray, EVEN)) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsMedian: starting testfits->MedianVec(P_EvenArray, 1)" << endl;
#endif
  P_TempEvenArray = testfits->MedianVec(*P_EvenArray, 1);
  cout << "MTestApp::TestCFitsMedian: Result of MedianVec(P_EvenArray, 1) = " << *P_TempEvenArray << endl;
  if (max(where((*P_TempEvenArray) == (*P_EvenArray), 0, 1)) < 0.1)
    t7 = true;
  else
    t7 = false;
  if (t7)
    cout << "MTestApp::TestCFitsMedian:                    t7(MedianVec(EvenArray), 1) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsMedian:                    t7(MedianVec(EvenArray), 1) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsMedian: starting testfits->MedianVec(P_EvenArray, 3, NORMAL)" << endl;
#endif
//  (*P_TestEvenArray)(i) = i;
  delete P_TempEvenArray;
  P_TempEvenArray = testfits->MedianVec(*P_EvenArray, 3, (*(new CString("NORMAL"))));
  cout << "MTestApp::TestCFitsMedian: Result of MedianVec(P_EvenArray, 3, NORMAL) = " << *P_TempEvenArray << endl;
  (*P_TestEvenArray) = 0., 1., 2., 3., 4., 5., 6., 7., 8., 9.;
  if (max(where((*P_TempEvenArray) == (*P_TestEvenArray), 0, 1)) < 0.001)
    t8 = true;
  else
    t8 = false;
  if (t8)
    cout << "MTestApp::TestCFitsMedian:                    t8(MedianVec(EvenArray), 3, NORMAL) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsMedian:                    t8(MedianVec(EvenArray), 3, NORMAL) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsMedian: starting testfits->MedianVec(P_EvenArray, 3, EVEN)" << endl;
#endif
//  (*P_TestEvenArray)(i) = i;
  delete P_TempEvenArray;
  P_TempEvenArray = testfits->MedianVec(*P_EvenArray, 3, (*(new CString("EVEN"))));
  cout << "MTestApp::TestCFitsMedian: Result of MedianVec(P_EvenArray, 3, EVEN) = " << *P_TempEvenArray << endl;
  (*P_TestEvenArray)(0) = 0.;
  (*P_TestEvenArray)(9) = 9.;
  cout << "MTestApp::TestCFitsMedian: Comparing Result with P_TestEvenArray = " << *P_TestEvenArray << endl;
  if (max(where((*P_TempEvenArray) == (*P_TestEvenArray), 0, 1)) < 0.1)
    t9 = true;
  else
    t9 = false;
  if (t9)
    cout << "MTestApp::TestCFitsMedian:                    t9(MedianVec(EvenArray), 3, EVEN) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsMedian:                    t9(MedianVec(EvenArray), 1, EVEN) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  Array<double, 1> D_A1_TTemp(51);
  D_A1_TTemp = 845612, 1.59998e+06, 2.36258e+06, 2.40367e+06, 1.64219e+06,    492533,     68242,
      32288.8,   29362.8,   27416.8,   25175.1,   24449.2,   24700.6,   24543.6,
      23817.7,     22994,   22677.2,   22658.6,   22742.4,   23445.4,   23971.3,
      24262.3,   24183.2,     24449,   26262.6,     30457,   39641.6,    107857,
      841007, 2.2282e+06, 2.83111e+06, 2.68893e+06, 1.65766e+06,    939603, 1.80772e+06,
      2.46659e+06, 2.32071e+06, 1.32943e+06,    292920,     47181,   28378.2,   26199.1,
      25598.4,   25837.1,   29630.6,     86642,    628046, 1.67953e+06, 2.13906e+06,
      1.878e+06, 1.02451e+06;

  Array<double, 1> D_A1_TMedian(51);
  Array<double, 1> *p_D_A1_TMedian = testfits->MedianVec(D_A1_TTemp, 5);
  D_A1_TMedian = (*p_D_A1_TMedian);
  delete p_D_A1_TMedian;
  Array<double, 1> D_A1_TTest(51);
  D_A1_TTest = 845612.,  1.59998e+06,  1.64218e+06,  1.64218e+06,  1.64218e+06,      492533.,
      68242.0,      32288.8,     29362.8,      27416.8,      25175.1,      24700.6,
      24543.6,      24449.2,      23817.7,      22994.0,      22742.4,      22742.4,
      22742.4,      23445.4,      23971.3,      24183.2,      24262.3,      24449.0,
      26262.6,      30457.0,      39641.6,      107857.,      841008.,  2.22820e+06,
      2.22820e+06, 2.22820e+06,  1.80772e+06,  1.80772e+06,  1.80772e+06,  1.80772e+06,
      1.80772e+06,  1.32943e+06,      292920.,      47181.0,      28378.2,      26199.1,
      26199.1,      26199.1,      29630.6,      86642.0,      628046.,  1.67953e+06,
      1.67953e+06,  1.87800e+06,  1.02451e+06;

      Array<int,1> I_A1_TestInd(D_A1_TTest.size());
      I_A1_TestInd = where(fabs(D_A1_TTest - D_A1_TMedian) > D_A1_TTest / 10000., 0, 1);
      for (int mm = 0; mm < I_A1_TestInd.size(); mm++)
      {
        if (I_A1_TestInd(mm) == 0)
        {
          cout << "MTestApp.TestCFitsMedian: D_A1_TTest(mm=" << mm << ")=" << D_A1_TTest(mm) << " != D_A1_TMedian(mm)=" << D_A1_TMedian(mm) << " => Returning FALSE!" << endl;
          return false;
        }
      }
      t10 = true;
  if (t1 && t2 && t3 && t4 && t5 && t6 && t7 && t8 && t9 && t10)
  {
    cout << "MTestApp::TestCFitsMedian:                       Test PASSED" << endl;
    cout << "=========================================" << endl;
  }
  else
  {
    cout << "MTestApp::TestCFitsMedian:                       Test FAILED" << endl;
    cout << "=========================================" << endl;
    if (testfits != NULL)
      delete(testfits);
    if (testfits1 != NULL)
      delete(testfits1);
    if (testfits2 != NULL)
      delete(testfits2);
    return false;

  }
  if (testfits != NULL)
    delete(testfits);
  if (testfits1 != NULL)
    delete(testfits1);
  if (testfits2 != NULL)
    delete(testfits2);
  delete P_OddArray;
  delete P_TempOddArray;
  delete P_EvenArray;
  delete P_TempEvenArray;
  delete P_TestEvenArray;
  delete P_TestOddArray;

  return true;

}

/** ***********************************************************/

bool TestCFitsLinFit()
{
  //  return TestConstructors("CFormatedString");
  bool t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34;
  string tempstr;// = new CString();
  int NCols = 9;
  int NRows = 17;
  int m;
  CString *p_testString = new CString("test/test_small.fits");
  CFits *testfits = new CFits();
  firstIndex i;
  double Result;

  Array<double, 1> D_A1_SP(1);
  Array<double, 1> D_A1_Sky(1);

  Array<double, 1> D_A1_X(10);
  D_A1_X(0) = 1;
  D_A1_X(1) = 2;
  D_A1_X(2) = 3;
  D_A1_X(3) = 4;
  D_A1_X(4) = 5;
  D_A1_X(5) = 6;
  D_A1_X(6) = 7;
  D_A1_X(7) = 8;
  D_A1_X(8) = 9;
  D_A1_X(9) = 10;

  Array<double, 1> D_A1_SF(10);
  D_A1_SF(0) = 0.;
  D_A1_SF(1) = 0.7;
  D_A1_SF(2) = 1.5;
  D_A1_SF(3) = 4.5;
  D_A1_SF(4) = 10.9;
  D_A1_SF(5) = 5.7;
  D_A1_SF(6) = 3.;
  D_A1_SF(7) = 1.;
  D_A1_SF(8) = 0.2;
  D_A1_SF(9) = 0.;
  D_A1_SF = D_A1_SF / sum(D_A1_SF);
  Array<double, 2> D_A2_SF(10,10);

  Array<double, 2> D_A2_Y(10,10);
  Array<double, 1> D_A1_SP_In(10);
  Array<double, 1> D_A1_Sky_In(10);
  for (int i=0; i<10; i++){
    D_A1_SP_In(i) = double(1500*i)+1.5;
    D_A1_Sky_In(i) = double(1500*i)+1.5;
    D_A2_Y(i, Range::all()) = D_A1_SP_In(i) * D_A1_SF;
    D_A2_Y(i, Range::all()) += D_A1_Sky_In(i);
    D_A2_SF(i, Range::all()) = D_A1_SF;
  }

  Array<double, 1> D_A1_Y(10);
  D_A1_Y = D_A2_Y(3,Range::all());

  #ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsLinFit: starting LinFit_1D" << endl;
  #endif
  if (testfits != NULL)
    delete(testfits);
  testfits = new CFits(*p_testString, NCols, NRows);

  Array<CString,1> CS_A1_Args_LinFit(10);
  CS_A1_Args_LinFit = CString(" ");
  void **PP_Args_LinFit = (void**)malloc(sizeof(void*) * 10);
  int pos = 0;

  double D_SP = 0.;
  double D_Sky = 0.;

  cout << "MTestApp::TestCFitsLinFit: D_A1_Y = " << D_A1_Y << endl;
  cout << "MTestApp::TestCFitsLinFit: D_A1_SF = " << D_A1_SF << endl;
  cout << "MTestApp::TestCFitsLinFit: D_SP = " << D_SP << endl;
  cout << "MTestApp::TestCFitsLinFit: D_Sky = " << D_Sky << endl;
  cout << "MTestApp::TestCFitsLinFit: CS_A1_Args_LinFit = " << CS_A1_Args_LinFit<< endl;

  if (testfits->LinFit(D_A1_Y, D_A1_SF, D_SP, D_Sky, CS_A1_Args_LinFit, PP_Args_LinFit))
    t1 = true;
  else
    t1 = false;
  if (t1)
    cout << "MTestApp::TestCFitsLinFit:                   t1: LinFit returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsLinFit:                   t1: LinFit returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  if (fabs(D_SP - D_A1_SP_In(3)) < 0.001)
    t2 = true;
  else
    t2 = false;
  if (t2)
    cout << "MTestApp::TestCFitsLinFit:                   t2: D_SP = " << D_SP << "  (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsLinFit:                   t2: D_SP = " << D_SP << " (UNEXPECTED)" << endl;
    return false;
  }

  if (fabs(D_Sky - D_A1_Sky_In(3)) < 0.001)
    t3 = true;
  else
    t3 = false;
  if (t3)
    cout << "MTestApp::TestCFitsLinFit:                   t3: D_Sky = " << D_Sky << "  (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsLinFit:                   t3: D_Sky = " << D_Sky << " (UNEXPECTED)" << endl;
    return false;
  }

  cout << "MTestApp::TestCFitsLinFit: D_A2_Y = " << D_A2_Y << endl;
  cout << "MTestApp::TestCFitsLinFit: D_A2_SF = " << D_A2_SF << endl;

  if (testfits->LinFit(D_A2_Y, D_A2_SF, D_A1_SP, D_A1_Sky, CS_A1_Args_LinFit, PP_Args_LinFit))
    t4 = true;
  else
    t4 = false;
  if (t4)
    cout << "MTestApp::TestCFitsLinFit:                   t4: LinFit returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsLinFit:                   t4: LinFit returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  cout << "MTestApp::TestCFitsLinFit: D_A1_SP = " << D_A1_SP << endl;
  cout << "MTestApp::TestCFitsLinFit: D_A1_Sky = " << D_A1_Sky << endl;

  for (int i=0; i<10; i++){
    if (fabs(D_A1_SP(i) - D_A1_SP_In(i)) < 0.001)
      t5 = true;
    else
      t5 = false;
    if (t5)
      cout << "MTestApp::TestCFitsLinFit: i = " << i << ":     t5: D_A1_SP(i) = " << D_A1_SP(i) << "  (expected) " << endl;
    else
    {
      cout << "MTestApp::TestCFitsLinFit: i = " << i << ":     t5: D_A1_SP(i) = " << D_A1_SP(i) << " (UNEXPECTED)" << endl;
      return false;
    }

    if (fabs(D_A1_Sky(i) - D_A1_Sky_In(i)) < 0.001)
      t6 = true;
    else
      t6 = false;
    if (t6)
      cout << "MTestApp::TestCFitsLinFit: i = " << i << ":     t6: D_A1_Sky(i) = " << D_A1_Sky(i) << "  (expected) " << endl;
    else
    {
      cout << "MTestApp::TestCFitsLinFit: i = " << i << ":     t6: D_A1_Sky(i) = " << D_A1_Sky(i) << " (UNEXPECTED)" << endl;
      return false;
    }
  }

  CS_A1_Args_LinFit(0) = CString("COVAR");
  Array<double, 3> D_A3_CoVar(D_A2_Y.rows(),D_A2_Y.cols(),2);
  PP_Args_LinFit[0] = &D_A3_CoVar;

  if (testfits->LinFit(D_A2_Y, D_A2_SF, D_A1_SP, D_A1_Sky, CS_A1_Args_LinFit, PP_Args_LinFit))
    t7 = true;
  else
    t7 = false;
  if (t7)
    cout << "MTestApp::TestCFitsLinFit:                   t7: LinFit returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsLinFit:                   t7: LinFit returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }
  cout << "MTestApp::TestCFitsLinFit:              t7: D_A3_CoVar = " << D_A3_CoVar << endl;

  CS_A1_Args_LinFit(0) = CString("MEASURE_ERRORS");
  Array<double , 2> D_A2_MeasureErrors(D_A2_Y.rows(), D_A2_Y.cols());
  D_A2_MeasureErrors = sqrt(D_A2_Y);
  PP_Args_LinFit[0] = &D_A2_MeasureErrors;
  t8 = testfits->LinFit(D_A2_Y, D_A2_SF, D_A1_SP, D_A1_Sky, CS_A1_Args_LinFit, PP_Args_LinFit);
  if (t8)
    cout << "MTestApp::TestCFitsLinFit:                   t8: LinFit returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsLinFit:                   t8: LinFit returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }
  cout << "MTestApp::TestCFitsLinFit:              t8: D_A2_MeasureErrors = " << D_A2_MeasureErrors << endl;
  for (int i=0; i<10; i++){
    if (fabs(D_A1_SP(i) - D_A1_SP_In(i)) < 0.001)
      t8 = true;
    else
      t8 = false;
    if (t8)
      cout << "MTestApp::TestCFitsLinFit: i = " << i << ":     t8: D_A1_SP(i) = " << D_A1_SP(i) << "  (expected) " << endl;
    else
    {
      cout << "MTestApp::TestCFitsLinFit: i = " << i << ":     t8: D_A1_SP(i) = " << D_A1_SP(i) << " (UNEXPECTED)" << endl;
      return false;
    }

    if (fabs(D_A1_Sky(i) - D_A1_Sky_In(i)) < 0.001)
      t9 = true;
    else
      t9 = false;
    if (t9)
      cout << "MTestApp::TestCFitsLinFit: i = " << i << ":     t9: D_A1_Sky(i) = " << D_A1_Sky(i) << "  (expected) " << endl;
    else
    {
      cout << "MTestApp::TestCFitsLinFit: i = " << i << ":     t9: D_A1_Sky(i) = " << D_A1_Sky(i) << " (UNEXPECTED)" << endl;
      return false;
    }
  }




  CS_A1_Args_LinFit(0) = CString("SDEV_IN");
  t10 = testfits->LinFit(D_A2_Y, D_A2_SF, D_A1_SP, D_A1_Sky, CS_A1_Args_LinFit, PP_Args_LinFit);
  if (t10)
    cout << "MTestApp::TestCFitsLinFit:                   t10: LinFit returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsLinFit:                   t10: LinFit returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }
  cout << "MTestApp::TestCFitsLinFit:              t10: D_A2_MeasureErrors = " << D_A2_MeasureErrors << endl;
  for (int i=0; i<10; i++){
    if (fabs(D_A1_SP(i) - D_A1_SP_In(i)) < 0.001)
      t11 = true;
    else
      t11 = false;
    if (t11)
      cout << "MTestApp::TestCFitsLinFit: i = " << i << ":     t11: D_A1_SP(i) = " << D_A1_SP(i) << "  (expected) " << endl;
    else
    {
      cout << "MTestApp::TestCFitsLinFit: i = " << i << ":     t11: D_A1_SP(i) = " << D_A1_SP(i) << " (UNEXPECTED)" << endl;
      return false;
    }

    if (fabs(D_A1_Sky(i) - D_A1_Sky_In(i)) < 0.001)
      t12 = true;
    else
      t12 = false;
    if (t12)
      cout << "MTestApp::TestCFitsLinFit: i = " << i << ":     t12: D_A1_Sky(i) = " << D_A1_Sky(i) << "  (expected) " << endl;
    else
    {
      cout << "MTestApp::TestCFitsLinFit: i = " << i << ":     t12: D_A1_Sky(i) = " << D_A1_Sky(i) << " (UNEXPECTED)" << endl;
      return false;
    }
  }

  CS_A1_Args_LinFit(0) = CString("MEASURE_ERRORS");
  CS_A1_Args_LinFit(1) = CString("SDEV_IN");
  PP_Args_LinFit[1] = &D_A2_MeasureErrors;
  t13 = testfits->LinFit(D_A2_Y, D_A2_SF, D_A1_SP, D_A1_Sky, CS_A1_Args_LinFit, PP_Args_LinFit);
  if (!t13)
    cout << "MTestApp::TestCFitsLinFit:                   t13: LinFit returned \"FALSE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsLinFit:                   t13: LinFit returned \"TRUE\" (UNEXPECTED)" << endl;
    return false;
  }

  CS_A1_Args_LinFit(1) = CString("PROB");
  Array<double, 1> D_A1_Prob(1);
  PP_Args_LinFit[1] = &D_A1_Prob;

  CS_A1_Args_LinFit(2) = CString("COVAR");
  PP_Args_LinFit[2] = &D_A3_CoVar;

  CS_A1_Args_LinFit(3) = CString("YFIT");
  Array<double, 2> D_A2_YFit(1);
  PP_Args_LinFit[3] = &D_A2_YFit;

  CS_A1_Args_LinFit(4) = CString("SIGMA");
  Array<double, 2> D_A2_Sigma(1,1);
  PP_Args_LinFit[4] = &D_A2_Sigma;

  CS_A1_Args_LinFit(5) = CString("CHISQ");
  Array<double, 1> D_A1_ChiSq(1);
  PP_Args_LinFit[5] = &D_A1_ChiSq;

  t14 = testfits->LinFit(D_A2_Y, D_A2_SF, D_A1_SP, D_A1_Sky, CS_A1_Args_LinFit, PP_Args_LinFit);
  if (t14)
    cout << "MTestApp::TestCFitsLinFit:                   t14: LinFit returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsLinFit:                   t14: LinFit returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  cout << "MTestApp::TestCFitsLinFit: D_A2_YFit = " << D_A2_YFit << endl;
  if (sum(fabs(D_A2_Y - D_A2_YFit)) < 0.0001)
    t15 = true;
  else
    t15 = false;
  if (t15)
    cout << "MTestApp::TestCFitsLinFit:                   t15: Y == YFit returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsLinFit:                   t15: Y == YFit returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  t16 = true;
/**  cout << "MTestApp::TestCFitsLinFit: D_A1_Prob = " << D_A1_Prob << endl;
  if (fabs((sum(D_A1_Prob) / D_A1_Prob.size()) - 1.) < 0.00001)
    t16 = true;
  else
    t16 = false;
  if (t16)
    cout << "MTestApp::TestCFitsLinFit:                   t16: Prob == 1 returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsLinFit:                   t16: Prob == 1 returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }**/

  cout << "MTestApp::TestCFitsLinFit: D_A3_CoVar = " << D_A3_CoVar << endl;

  cout << "MTestApp::TestCFitsLinFit: D_A2_Sigma = " << D_A2_Sigma << endl;

  CS_A1_Args_LinFit(6) = CString("MASK");
  Array<int, 2> I_A2_Mask(D_A2_Y.rows(), D_A2_Y.cols());
  I_A2_Mask = 1;
  PP_Args_LinFit[6] = &I_A2_Mask;

  t17 = testfits->LinFit(D_A2_Y, D_A2_SF, D_A1_SP, D_A1_Sky, CS_A1_Args_LinFit, PP_Args_LinFit);
  if (t17)
    cout << "MTestApp::TestCFitsLinFit:                   t17: LinFit returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsLinFit:                   t17: LinFit returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  cout << "MTestApp::TestCFitsLinFit: D_A2_YFit = " << D_A2_YFit << endl;
  if (sum(fabs(D_A2_Y - D_A2_YFit)) < 0.0001)
    t18 = true;
  else
    t18 = false;
  if (t18)
    cout << "MTestApp::TestCFitsLinFit:                   t18: Y == YFit returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsLinFit:                   t18: Y == YFit returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  I_A2_Mask(3,4) = 0;
  I_A2_Mask(3,5) = 0;
  t19 = testfits->LinFit(D_A2_Y, D_A2_SF, D_A1_SP, D_A1_Sky, CS_A1_Args_LinFit, PP_Args_LinFit);
  if (t19)
    cout << "MTestApp::TestCFitsLinFit:                   t19: LinFit returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsLinFit:                   t19: LinFit returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  cout << "MTestApp::TestCFitsLinFit: D_A2_YFit = " << D_A2_YFit << endl;
  if (sum(fabs(D_A2_Y - D_A2_YFit)) < 0.0001)
    t20 = true;
  else
    t20 = false;
  if (t20)
    cout << "MTestApp::TestCFitsLinFit:                   t20: Y == YFit returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsLinFit:                   t20: Y == YFit returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  Array<double, 2> D_A2_Y_Orig(D_A2_Y.rows(), D_A2_Y.cols());
  D_A2_Y_Orig = D_A2_Y;
  D_A2_Y(6,4) = 65000.;
  D_A2_MeasureErrors(6,4) = sqrt(65000.);
  Array<int, 2> I_A2_Mask_Orig(I_A2_Mask.rows(), I_A2_Mask.cols());
  I_A2_Mask_Orig = I_A2_Mask;

  t21 = testfits->LinFit(D_A2_Y, D_A2_SF, D_A1_SP, D_A1_Sky, CS_A1_Args_LinFit, PP_Args_LinFit);
  if (t21)
    cout << "MTestApp::TestCFitsLinFit:                   t21: LinFit returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsLinFit:                   t21: LinFit returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  cout << "MTestApp::TestCFitsLinFit: D_A2_YFit = " << D_A2_YFit << endl;
  if (sum(fabs(D_A2_Y - D_A2_YFit)) <= 0.0001)
    t22 = true;
  else
    t22 = false;
  if (!t22)
    cout << "MTestApp::TestCFitsLinFit:                   t22: Y == YFit returned \"FALSE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsLinFit:                   t22: Y == YFit returned \"TRUE\" (UNEXPECTED)" << endl;
    return false;
  }

  if (sum(fabs(I_A2_Mask - I_A2_Mask_Orig)) < 1)
    t23 = true;
  else
    t23 = false;
  if (t23)
    cout << "MTestApp::TestCFitsLinFit:                   t23: Mask == Mask_Orig returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsLinFit:                   t23: Mask == Mask_Orig returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  Array<double, 2> D_A2_Y_Diff(D_A2_Y.rows(), D_A2_Y.cols());
  Array<double, 2> D_A2_Y_WithNoise(D_A2_Y.rows(), D_A2_Y.cols());
  /// Add noise
  double d_temp;
  for (int i_row = 0; i_row < D_A2_Y.rows(); i_row++){
    for (int i_col = 0; i_col < D_A2_Y.cols(); i_col++){
      Normal<double> normalGen(D_A2_Y(i_row,i_col),sqrt(D_A2_Y(i_row,i_col)));
      d_temp = normalGen.random();
      D_A2_Y(i_row,i_col) = d_temp;
    }
  }
  D_A2_Y_WithNoise = D_A2_Y;
  cout << "MTestApp::TestCFitsLinFit: D_A2_Y set to " << D_A2_Y << endl;
  D_A2_Y_Diff = D_A2_Y_Orig - D_A2_Y;
  cout << "MTestApp::TestCFitsLinFit: D_A2_Y_Diff set to " << D_A2_Y_Diff << endl;
//  return false;

  CS_A1_Args_LinFit(7) = CString("REJECT");
  double D_Reject = 2.5;
  PP_Args_LinFit[7] = &D_Reject;

  t24 = testfits->LinFit(D_A2_Y, D_A2_SF, D_A1_SP, D_A1_Sky, CS_A1_Args_LinFit, PP_Args_LinFit);
  if (t24)
    cout << "MTestApp::TestCFitsLinFit:                   t24: LinFit returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsLinFit:                   t24: LinFit returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  cout << "MTestApp::TestCFitsLinFit: D_A1_SP = " << D_A1_SP << endl;
  cout << "MTestApp::TestCFitsLinFit: D_A1_Sky = " << D_A1_Sky << endl;
  cout << "MTestApp::TestCFitsLinFit: D_A2_YFit = " << D_A2_YFit << endl;
  cout << "MTestApp::TestCFitsLinFit: I_A2_Mask = " << I_A2_Mask << endl;
  for (int i=0; i<D_A2_Y.rows(); i++){
    if ((D_A1_SP_In(i) < 50.) || (fabs(D_A1_SP(i) - D_A1_SP_In(i)) < sqrt(D_A1_SP_In(i))))
      t25 = true;
    else
      t25 = false;
    if (t25)
      cout << "MTestApp::TestCFitsLinFit:                   t25: D_A1_SP(" << i << ")=" << D_A1_SP(i) << " == D_A1_SP_In(" << i << ")=" << D_A1_SP_In(i) << " returned \"TRUE\" (EXPECTED)" << endl;
    else
    {
      cout << "MTestApp::TestCFitsLinFit:                   t25: D_A1_SP(" << i << ")=" << D_A1_SP(i) << " == D_A1_SP_In(" << i << ")=" << D_A1_SP_In(i) << " returned \"FALSE\" (UNEXPECTED)" << endl;
      //return false;
    }

    if ((D_A1_Sky_In(i) < 50.) || (fabs(D_A1_Sky(i) - D_A1_Sky_In(i)) < sqrt(D_A1_Sky_In(i))))
      t25 = true;
    else
      t25 = false;
    if (t25)
      cout << "MTestApp::TestCFitsLinFit:                   t25: D_A1_Sky(" << i << ")=" << D_A1_Sky(i) << " == D_A1_Sky_In(" << i << ")=" << D_A1_Sky_In(i) << " returned \"TRUE\" (EXPECTED)" << endl;
    else
    {
      cout << "MTestApp::TestCFitsLinFit:                   t25: D_A1_Sky(" << i << ")=" << D_A1_Sky(i) << " == D_A1_Sky_In(" << i << ")=" << D_A1_Sky_In(i) << " returned \"FALSE\" (UNEXPECTED)" << endl;
      //return false;
    }
  }

  if (I_A2_Mask(6,4) == 0)
    t26 = true;
  else
    t26 = false;
  if (t26)
    cout << "MTestApp::TestCFitsLinFit:                   t26: I_A2_Mask(6,4) == 0 returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsLinFit:                   t26: I_A2_Mask(6,4) == 0 returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }










  /// TEST FIT
  CS_A1_Args_LinFit = CString(" ");
  D_A2_Y = D_A2_Y_Orig;
  D_Sky = 1.;
  D_A1_Sky = 1.;
  Array<double, 1> D_A1_Y_Orig(10);
  D_A1_Y_Orig = D_A1_Y;
  Array<double, 1> D_A1_SF_Orig(10);
  D_A1_SF_Orig = D_A1_SF;
  Array<double, 1> D_A1_MeasureErrors(10);
  D_A1_MeasureErrors = sqrt(D_A1_Y);
  Array<double, 1> D_A1_MeasureErrors_Orig(10);
  D_A1_MeasureErrors_Orig = D_A1_MeasureErrors;

  if (testfits->Fit(D_A1_Y, D_A1_SF, D_SP, D_Sky, CS_A1_Args_LinFit, PP_Args_LinFit))
    t1 = true;
  else
    t1 = false;
  if (t1)
    cout << "MTestApp::TestCFitsFit:                   t1: Fit returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t1: Fit returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  if (fabs(D_SP - D_A1_SP_In(3)) < 0.001)
    t2 = true;
  else
    t2 = false;
  if (t2)
    cout << "MTestApp::TestCFitsFit:                   t2: D_SP = " << D_SP << "  (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t2: D_SP = " << D_SP << " (UNEXPECTED)" << endl;
    return false;
  }

  if (fabs(D_Sky - D_A1_Sky_In(3)) < 0.001)
    t3 = true;
  else
    t3 = false;
  if (t3)
    cout << "MTestApp::TestCFitsFit:                   t3: D_Sky = " << D_Sky << "  (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t3: D_Sky = " << D_Sky << " (UNEXPECTED)" << endl;
    return false;
  }

  D_A1_Y(4) = 65000.;
  Array<int, 1> I_A1_Mask(10);
  I_A1_Mask = 1;
  Array<double, 1> D_A1_YFit(10);
  CS_A1_Args_LinFit(0) = CString("MEASURE_ERRORS_IN");
  PP_Args_LinFit[0] = &D_A1_MeasureErrors;
  CS_A1_Args_LinFit(1) = CString("MASK_INOUT");
  PP_Args_LinFit[1] = &I_A1_Mask;
  CS_A1_Args_LinFit(2) = CString("YFIT_OUT");
  PP_Args_LinFit[2] = &D_A1_YFit;
  CS_A1_Args_LinFit(3) = CString("CHISQ_OUT");
  double D_ChiSqOut;
  PP_Args_LinFit[3] = &D_ChiSqOut;
  CS_A1_Args_LinFit(4) = CString("SIGMA_OUT");
  Array<double, 1> D_A1_SigmaOut(1);
  PP_Args_LinFit[4] = &D_A1_SigmaOut;
  CS_A1_Args_LinFit(5) = CString(" ");
  CS_A1_Args_LinFit(6) = CString(" ");
  cout << "MTestApp::TestCFitsFit: D_A1_Y = " << D_A1_Y << endl;
  cout << "MTestApp::TestCFitsFit: D_A1_SF = " << D_A1_SF << endl;
  if (testfits->Fit(D_A1_Y, D_A1_SF, D_SP, D_Sky, CS_A1_Args_LinFit, PP_Args_LinFit))
    t1 = true;
  else
    t1 = false;
  if (t1)
    cout << "MTestApp::TestCFitsFit:                   t1 with cosmic: Fit returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t1 with cosmic: Fit returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  cout << "MTestApp::TestCFitsFit: D_A1_YFit = " << D_A1_YFit << endl;
  cout << "MTestApp::TestCFitsFit: D_A1_SigmaOut = " << D_A1_SigmaOut << endl;
  cout << "MTestApp::TestCFitsFit: I_A1_Mask = " << I_A1_Mask << endl;
  cout << "MTestApp::TestCFitsFit: D_ChiSqOut = " << D_ChiSqOut << endl;

  if (fabs(D_SP - D_A1_SP_In(3)) < 0.001)
    t2 = false;
  else
    t2 = true;
  if (t2)
    cout << "MTestApp::TestCFitsFit:                   t2 with cosmic: D_SP = " << D_SP << "  (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t2 with cosmic: D_SP = " << D_SP << " (UNEXPECTED)" << endl;
    return false;
  }

  if (fabs(D_Sky - D_A1_Sky_In(3)) < 0.001)
    t3 = false;
  else
    t3 = true;
  if (t3)
    cout << "MTestApp::TestCFitsFit:                   t3 with cosmic: D_Sky = " << D_Sky << "  (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t3 with cosmic: D_Sky = " << D_Sky << " (UNEXPECTED)" << endl;
    return false;
  }
  CS_A1_Args_LinFit(5) = CString("REJECT_IN");
  D_Reject = 2.5;
  PP_Args_LinFit[5] = &D_Reject;
  if (testfits->Fit(D_A1_Y, D_A1_SF, D_SP, D_Sky, CS_A1_Args_LinFit, PP_Args_LinFit))
    t1 = true;
  else
    t1 = false;
  if (t1)
    cout << "MTestApp::TestCFitsFit:                   t1 with cosmic and reject: Fit returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t1 with cosmic and reject: Fit returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  cout << "MTestApp::TestCFitsFit: D_A1_YFit = " << D_A1_YFit << endl;
  cout << "MTestApp::TestCFitsFit: D_A1_SigmaOut = " << D_A1_SigmaOut << endl;
  cout << "MTestApp::TestCFitsFit: I_A1_Mask = " << I_A1_Mask << endl;
  cout << "MTestApp::TestCFitsFit: D_ChiSqOut = " << D_ChiSqOut << endl;

  if (fabs(D_SP - D_A1_SP_In(3)) < 0.001)
    t2 = true;
  else
    t2 = false;
  if (t2)
    cout << "MTestApp::TestCFitsFit:                   t2 with cosmic and reject: D_SP = " << D_SP << "  (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t2 with cosmic and reject: D_SP = " << D_SP << " (UNEXPECTED)" << endl;
//    return false;
  }

  if (fabs(D_Sky - D_A1_Sky_In(3)) < 0.001)
    t3 = true;
  else
    t3 = false;
  if (t3)
    cout << "MTestApp::TestCFitsFit:                   t3 with cosmic and reject: D_Sky = " << D_Sky << "  (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t3 with cosmic and reject: D_Sky = " << D_Sky << " (UNEXPECTED)" << endl;
//    return false;
  }



  Array<double, 1> D_A1_YNine(9);
  Array<int, 1> I_A1_MaskNine(9);
  Array<double, 1> D_A1_SFNine(9);
  Array<double, 1> D_A1_MeasureErrorsNine(9);
  I_A1_Mask = 1;
  pos = 0;
  for (int i=0; i<4; i++){
    I_A1_MaskNine(pos) = I_A1_Mask(i);
    D_A1_YNine(pos) = D_A1_Y(i);
    D_A1_SFNine(pos) = D_A1_SF(i);
    D_A1_MeasureErrorsNine(pos) = D_A1_MeasureErrors(i);
    pos++;
  }
  for (int i=5; i<10; i++){
    I_A1_MaskNine(pos) = I_A1_Mask(i);
    D_A1_YNine(pos) = D_A1_Y(i);
    D_A1_SFNine(pos) = D_A1_SF(i);
    D_A1_MeasureErrorsNine(pos) = D_A1_MeasureErrors(i);
    pos++;
  }
  CS_A1_Args_LinFit = CString(" ");
  PP_Args_LinFit[0] = &D_A1_MeasureErrorsNine;
  PP_Args_LinFit[1] = &I_A1_MaskNine;
  if (testfits->Fit(D_A1_YNine, D_A1_SFNine, D_SP, D_Sky, CS_A1_Args_LinFit, PP_Args_LinFit))
    t1 = true;
  else
    t1 = false;
  if (t1)
    cout << "MTestApp::TestCFitsFit:                   t1 Nine: Fit returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t1 Nine: Fit returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  if (fabs(D_SP - D_A1_SP_In(3)) < 0.001)
    t2 = true;
  else
    t2 = false;
  if (t2)
    cout << "MTestApp::TestCFitsFit:                   t2 Nine: D_SP = " << D_SP << "  (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t2 Nine: D_SP = " << D_SP << " (UNEXPECTED)" << endl;
    return false;
  }

  if (fabs(D_Sky - D_A1_Sky_In(3)) < 0.001)
    t3 = true;
  else
    t3 = false;
  if (t3)
    cout << "MTestApp::TestCFitsFit:                   t3 Nine: D_Sky = " << D_Sky << "  (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t3 Nine: D_Sky = " << D_Sky << " (UNEXPECTED)" << endl;
    return false;
  }

  CS_A1_Args_LinFit(0) = CString("MEASURE_ERRORS_IN");
  CS_A1_Args_LinFit(1) = CString("MASK_INOUT");
  CS_A1_Args_LinFit(2) = CString("YFIT_OUT");
  CS_A1_Args_LinFit(3) = CString("CHISQ_OUT");
  CS_A1_Args_LinFit(4) = CString("SIGMA_OUT");
  CS_A1_Args_LinFit(5) = CString(" ");
  CS_A1_Args_LinFit(6) = CString(" ");
  if (testfits->Fit(D_A1_YNine, D_A1_SFNine, D_SP, D_Sky, CS_A1_Args_LinFit, PP_Args_LinFit))
    t1 = true;
  else
    t1 = false;
  if (t1)
    cout << "MTestApp::TestCFitsFit:                   t1 Nine with errors: Fit returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t1 Nine with errors: Fit returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  cout << "MTestApp::TestCFitsFit: D_A1_YFit = " << D_A1_YFit << endl;
  cout << "MTestApp::TestCFitsFit: D_A1_SigmaOut = " << D_A1_SigmaOut << endl;
  cout << "MTestApp::TestCFitsFit: I_A1_Mask = " << I_A1_Mask << endl;
  cout << "MTestApp::TestCFitsFit: D_ChiSqOut = " << D_ChiSqOut << endl;

  if (fabs(D_SP - D_A1_SP_In(3)) < 0.001)
    t2 = true;
  else
    t2 = false;
  if (t2)
    cout << "MTestApp::TestCFitsFit:                   t2 Nine with errors: D_SP = " << D_SP << "  (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t2 Nine with errors: D_SP = " << D_SP << " (UNEXPECTED)" << endl;
    return false;
  }

  if (fabs(D_Sky - D_A1_Sky_In(3)) < 0.001)
    t3 = true;
  else
    t3 = false;
  if (t3)
    cout << "MTestApp::TestCFitsFit:                   t3 Nine with errors: D_Sky = " << D_Sky << "  (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t3 Nine with errors: D_Sky = " << D_Sky << " (UNEXPECTED)" << endl;
    return false;
  }






  I_A1_Mask = 1;
  pos = 0;
  D_A1_Y = D_A2_Y_WithNoise(3,Range::all());
  for (int i=0; i<4; i++){
    I_A1_MaskNine(pos) = I_A1_Mask(i);
    D_A1_YNine(pos) = D_A1_Y(i);
    pos++;
  }
  for (int i=5; i<10; i++){
    I_A1_MaskNine(pos) = I_A1_Mask(i);
    D_A1_YNine(pos) = D_A1_Y(i);
    pos++;
  }
  D_A1_MeasureErrorsNine = sqrt(D_A1_YNine);
  PP_Args_LinFit[0] = &D_A1_MeasureErrorsNine;
  PP_Args_LinFit[1] = &I_A1_MaskNine;
  if (testfits->Fit(D_A1_YNine, D_A1_SFNine, D_SP, D_Sky, CS_A1_Args_LinFit, PP_Args_LinFit))
    t1 = true;
  else
    t1 = false;
  if (t1)
    cout << "MTestApp::TestCFitsFit:                   t1 Nine with noise and errors: Fit returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t1 Nine with noise and errors: Fit returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  if (fabs(D_SP - D_A1_SP_In(3)) < 3. * sqrt(D_A1_SP_In(3)))
    t2 = true;
  else
    t2 = false;
  if (t2)
    cout << "MTestApp::TestCFitsFit:                   t2 Nine with noise and errors: D_SP = " << D_SP << "  (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t2 Nine with noise and errors: D_SP = " << D_SP << " (UNEXPECTED)" << endl;
    return false;
  }

  if (fabs(D_Sky - D_A1_Sky_In(3)) < 3. * sqrt(D_A1_Sky_In(3)))
    t3 = true;
  else
    t3 = false;
  if (t3)
    cout << "MTestApp::TestCFitsFit:                   t3 Nine with noise and errors: D_Sky = " << D_Sky << "  (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t3 Nine with noise and errors: D_Sky = " << D_Sky << " (UNEXPECTED)" << endl;
    return false;
  }




//return false;




  CS_A1_Args_LinFit = CString(" ");
  D_A1_Y = D_A2_Y_WithNoise(3,Range::all());
  D_A1_MeasureErrors = sqrt(D_A1_Y);
  PP_Args_LinFit[0] = &D_A1_MeasureErrors;
  D_Sky = 1.;
  if (testfits->Fit(D_A1_Y, D_A1_SF, D_SP, D_Sky, CS_A1_Args_LinFit, PP_Args_LinFit))
    t1 = true;
  else
    t1 = false;
  if (t1)
    cout << "MTestApp::TestCFitsFit:                   t1 with noise: Fit returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t1 with noise: Fit returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  if (fabs(D_SP - D_A1_SP_In(3)) < 5. * sqrt(D_A1_SP_In(3)))
    t2 = true;
  else
    t2 = false;
  if (t2)
    cout << "MTestApp::TestCFitsFit:                   t2 with noise: D_SP = " << D_SP << "  (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t2 with noise: D_SP = " << D_SP << " (UNEXPECTED)" << endl;
    return false;
  }

  if (fabs(D_Sky - D_A1_Sky_In(3)) < 3. * sqrt(D_A1_SP_In(3)))
    t3 = true;
  else
    t3 = false;
  if (t3)
    cout << "MTestApp::TestCFitsFit:                   t3 with noise: D_Sky = " << D_Sky << "  (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t3 with noise: D_Sky = " << D_Sky << " (UNEXPECTED)" << endl;
    return false;
  }

  D_A1_Y(4) = 65000.;
  I_A1_Mask = 1;
  D_A1_MeasureErrors = sqrt(D_A1_Y);
  CS_A1_Args_LinFit(0) = CString("MEASURE_ERRORS_IN");
  PP_Args_LinFit[0] = &D_A1_MeasureErrors;
  CS_A1_Args_LinFit(1) = CString("MASK_INOUT");
  PP_Args_LinFit[1] = &I_A1_Mask;
  CS_A1_Args_LinFit(2) = CString("YFIT_OUT");
  PP_Args_LinFit[2] = &D_A1_YFit;
  CS_A1_Args_LinFit(3) = CString("CHISQ_OUT");
  PP_Args_LinFit[3] = &D_ChiSqOut;
  CS_A1_Args_LinFit(4) = CString("SIGMA_OUT");
  PP_Args_LinFit[4] = &D_A1_SigmaOut;
  CS_A1_Args_LinFit(5) = CString(" ");
  CS_A1_Args_LinFit(6) = CString(" ");
  if (testfits->Fit(D_A1_Y, D_A1_SF, D_SP, D_Sky, CS_A1_Args_LinFit, PP_Args_LinFit))
    t1 = true;
  else
    t1 = false;
  if (t1)
    cout << "MTestApp::TestCFitsFit:                   t1 with noise and cosmic: Fit returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t1 with noise and cosmic: Fit returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  cout << "MTestApp::TestCFitsFit: D_A1_YFit = " << D_A1_YFit << endl;
  cout << "MTestApp::TestCFitsFit: D_A1_SigmaOut = " << D_A1_SigmaOut << endl;
  cout << "MTestApp::TestCFitsFit: I_A1_Mask = " << I_A1_Mask << endl;
  cout << "MTestApp::TestCFitsFit: D_ChiSqOut = " << D_ChiSqOut << endl;

  if (fabs(D_SP - D_A1_SP_In(3)) < 0.001)
    t2 = false;
  else
    t2 = true;
  if (t2)
    cout << "MTestApp::TestCFitsFit:                   t2 with noise and cosmic: D_SP = " << D_SP << "  (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t2 with noise and cosmic: D_SP = " << D_SP << " (UNEXPECTED)" << endl;
    return false;
  }

  if (fabs(D_Sky - D_A1_Sky_In(3)) < 0.001)
    t3 = false;
  else
    t3 = true;
  if (t3)
    cout << "MTestApp::TestCFitsFit:                   t3 with noise and cosmic: D_Sky = " << D_Sky << "  (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t3 with noise and cosmic: D_Sky = " << D_Sky << " (UNEXPECTED)" << endl;
    return false;
  }

  CS_A1_Args_LinFit(5) = CString("REJECT_IN");
  D_Reject = 2.5;
  PP_Args_LinFit[5] = &D_Reject;
  if (testfits->Fit(D_A1_Y, D_A1_SF, D_SP, D_Sky, CS_A1_Args_LinFit, PP_Args_LinFit))
    t1 = true;
  else
    t1 = false;
  if (t1)
    cout << "MTestApp::TestCFitsFit:                   t1 noise and with cosmic and reject: Fit returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t1 noise and with cosmic and reject: Fit returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  cout << "MTestApp::TestCFitsFit: D_A1_YFit = " << D_A1_YFit << endl;
  cout << "MTestApp::TestCFitsFit: D_A1_SigmaOut = " << D_A1_SigmaOut << endl;
  cout << "MTestApp::TestCFitsFit: I_A1_Mask = " << I_A1_Mask << endl;
  cout << "MTestApp::TestCFitsFit: D_ChiSqOut = " << D_ChiSqOut << endl;

  if (fabs(D_SP - D_A1_SP_In(3)) < 3. * sqrt(D_A1_SP_In(3)))
    t2 = true;
  else
    t2 = false;
  if (t2)
    cout << "MTestApp::TestCFitsFit:                   t2 with noise and cosmic and reject: D_SP = " << D_SP << "  (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t2 with noise and cosmic and reject: D_SP = " << D_SP << " (UNEXPECTED)" << endl;
    return false;
  }

  if (fabs(D_Sky - D_A1_Sky_In(3)) < 3. * sqrt(D_A1_Sky_In(3)))
    t3 = true;
  else
    t3 = false;
  if (t3)
    cout << "MTestApp::TestCFitsFit:                   t3 with noise and cosmic and reject: D_Sky = " << D_Sky << "  (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t3 with noise and cosmic and reject: D_Sky = " << D_Sky << " (UNEXPECTED)" << endl;
    return false;
  }




//return false;
  cout << "MTestApp::TestCFitsFit: D_A2_Y = " << D_A2_Y << endl;
  cout << "MTestApp::TestCFitsFit: D_A2_SF = " << D_A2_SF << endl;
  CS_A1_Args_LinFit = CString(" ");
  if (testfits->Fit(D_A2_Y, D_A2_SF, D_A1_SP, D_A1_Sky, CS_A1_Args_LinFit, PP_Args_LinFit))
    t4 = true;
  else
    t4 = false;
  if (t4)
    cout << "MTestApp::TestCFitsFit:                   t4: Fit returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t4: Fit returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  cout << "MTestApp::TestCFitsFit: D_A1_SP = " << D_A1_SP << endl;
  cout << "MTestApp::TestCFitsFit: D_A1_Sky = " << D_A1_Sky << endl;

  for (int i=0; i<10; i++){
    if (fabs(D_A1_SP(i) - D_A1_SP_In(i)) < 0.001)
      t5 = true;
    else
      t5 = false;
    if (t5)
      cout << "MTestApp::TestCFitsFit: i = " << i << ":     t5: D_A1_SP(i) = " << D_A1_SP(i) << "  (expected) " << endl;
    else
    {
      cout << "MTestApp::TestCFitsFit: i = " << i << ":     t5: D_A1_SP(i) = " << D_A1_SP(i) << " (UNEXPECTED)" << endl;
      return false;
    }

    if (fabs(D_A1_Sky(i) - D_A1_Sky_In(i)) < 0.001)
      t6 = true;
    else
      t6 = false;
    if (t6)
      cout << "MTestApp::TestCFitsFit: i = " << i << ":     t6: D_A1_Sky(i) = " << D_A1_Sky(i) << "  (expected) " << endl;
    else
    {
      cout << "MTestApp::TestCFitsFit: i = " << i << ":     t6: D_A1_Sky(i) = " << D_A1_Sky(i) << " (UNEXPECTED)" << endl;
      return false;
    }
  }

  CS_A1_Args_LinFit(0) = CString("Q_OUT");
  Array<double,1> D_A1_Q;
  PP_Args_LinFit[0] = &D_A1_Q;

  if (testfits->Fit(D_A2_Y, D_A2_SF, D_A1_SP, D_A1_Sky, CS_A1_Args_LinFit, PP_Args_LinFit))
    t7 = true;
  else
    t7 = false;
  if (t7)
    cout << "MTestApp::TestCFitsFit:                   t7: Fit returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t7: Fit returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }
  cout << "MTestApp::TestCFitsFit:              t7: D_A1_Q = " << D_A1_Q << endl;

  CS_A1_Args_LinFit(0) = CString("MEASURE_ERRORS_IN");
  D_A2_MeasureErrors = sqrt(D_A2_Y);
  PP_Args_LinFit[0] = &D_A2_MeasureErrors;
  t8 = testfits->Fit(D_A2_Y, D_A2_SF, D_A1_SP, D_A1_Sky, CS_A1_Args_LinFit, PP_Args_LinFit);
  if (t8)
    cout << "MTestApp::TestCFitsFit:                   t8: Fit returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t8: Fit returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }
  cout << "MTestApp::TestCFitsFit:              t8: D_A2_MeasureErrors = " << D_A2_MeasureErrors << endl;
  for (int i=0; i<10; i++){
    if (fabs(D_A1_SP(i) - D_A1_SP_In(i)) < 0.001)
      t8 = true;
    else
      t8 = false;
    if (t8)
      cout << "MTestApp::TestCFitsFit: i = " << i << ":     t8: D_A1_SP(i) = " << D_A1_SP(i) << "  (expected) " << endl;
    else
    {
      cout << "MTestApp::TestCFitsFit: i = " << i << ":     t8: D_A1_SP(i) = " << D_A1_SP(i) << " (UNEXPECTED)" << endl;
      return false;
    }

    if (fabs(D_A1_Sky(i) - D_A1_Sky_In(i)) < 0.001)
      t9 = true;
    else
      t9 = false;
    if (t9)
      cout << "MTestApp::TestCFitsFit: i = " << i << ":     t9: D_A1_Sky(i) = " << D_A1_Sky(i) << "  (expected) " << endl;
    else
    {
      cout << "MTestApp::TestCFitsFit: i = " << i << ":     t9: D_A1_Sky(i) = " << D_A1_Sky(i) << " (UNEXPECTED)" << endl;
      return false;
    }
  }

  CS_A1_Args_LinFit(1) = CString("CHISQ_OUT");
  PP_Args_LinFit[1] = &D_A1_ChiSq;

  CS_A1_Args_LinFit(2) = CString("SIGMA_OUT");
  PP_Args_LinFit[2] = &D_A2_Sigma;

  CS_A1_Args_LinFit(3) = CString("YFIT_OUT");
  PP_Args_LinFit[3] = &D_A2_YFit;

  CS_A1_Args_LinFit(4) = CString("Q_OUT");
  PP_Args_LinFit[4] = &D_A1_Q;

  t14 = testfits->Fit(D_A2_Y, D_A2_SF, D_A1_SP, D_A1_Sky, CS_A1_Args_LinFit, PP_Args_LinFit);
  if (t14)
    cout << "MTestApp::TestCFitsFit:                   t14: Fit returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t14: Fit returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  cout << "MTestApp::TestCFitsFit: D_A2_YFit = " << D_A2_YFit << endl;
  if (sum(fabs(D_A2_Y - D_A2_YFit)) < 0.0001)
    t15 = true;
  else
    t15 = false;
  if (t15)
    cout << "MTestApp::TestCFitsFit:                   t15: Y == YFit returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t15: Y == YFit returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  t16 = true;
  /**  cout << "MTestApp::TestCFitsFit: D_A1_Prob = " << D_A1_Prob << endl;
  if (fabs((sum(D_A1_Prob) / D_A1_Prob.size()) - 1.) < 0.00001)
    t16 = true;
  else
    t16 = false;
  if (t16)
    cout << "MTestApp::TestCFitsFit:                   t16: Prob == 1 returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t16: Prob == 1 returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
}**/

  cout << "MTestApp::TestCFitsFit: D_A1_ChiSq= " << D_A1_Q << endl;
  cout << "MTestApp::TestCFitsFit: D_A1_Q = " << D_A1_Q << endl;
  cout << "MTestApp::TestCFitsFit: D_A2_Sigma = " << D_A2_Sigma << endl;

  CS_A1_Args_LinFit(6) = CString("MASK_INOUT");
  I_A2_Mask = 1;
  PP_Args_LinFit[6] = &I_A2_Mask;

  t17 = testfits->Fit(D_A2_Y, D_A2_SF, D_A1_SP, D_A1_Sky, CS_A1_Args_LinFit, PP_Args_LinFit);
  if (t17)
    cout << "MTestApp::TestCFitsFit:                   t17: Fit returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t17: Fit returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  cout << "MTestApp::TestCFitsFit: D_A2_YFit = " << D_A2_YFit << endl;
  if (sum(fabs(D_A2_Y - D_A2_YFit)) < 0.0001)
    t18 = true;
  else
    t18 = false;
  if (t18)
    cout << "MTestApp::TestCFitsFit:                   t18: Y == YFit returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t18: Y == YFit returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  I_A2_Mask(3,4) = 0;
  I_A2_Mask(3,5) = 0;
  t19 = testfits->Fit(D_A2_Y, D_A2_SF, D_A1_SP, D_A1_Sky, CS_A1_Args_LinFit, PP_Args_LinFit);
  if (t19)
    cout << "MTestApp::TestCFitsFit:                   t19: Fit returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t19: Fit returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  cout << "MTestApp::TestCFitsFit: D_A2_YFit = " << D_A2_YFit << endl;
  if (sum(fabs(D_A2_Y - D_A2_YFit)) < 0.0001)
    t20 = true;
  else
    t20 = false;
  if (t20)
    cout << "MTestApp::TestCFitsFit:                   t20: Y == YFit returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t20: Y == YFit returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  D_A2_Y(6,4) = 65000.;
  D_A2_MeasureErrors(6,4) = sqrt(65000.);
  I_A2_Mask_Orig = I_A2_Mask;

  t21 = testfits->Fit(D_A2_Y, D_A2_SF, D_A1_SP, D_A1_Sky, CS_A1_Args_LinFit, PP_Args_LinFit);
  if (t21)
    cout << "MTestApp::TestCFitsFit:                   t21: Fit returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t21: Fit returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  cout << "MTestApp::TestCFitsFit: D_A2_YFit = " << D_A2_YFit << endl;
  if (sum(fabs(D_A2_Y - D_A2_YFit)) <= 0.0001)
    t22 = true;
  else
    t22 = false;
  if (!t22)
    cout << "MTestApp::TestCFitsFit:                   t22: Y == YFit returned \"FALSE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t22: Y == YFit returned \"TRUE\" (UNEXPECTED)" << endl;
    return false;
  }

  if (sum(fabs(I_A2_Mask - I_A2_Mask_Orig)) < 1)
    t23 = true;
  else
    t23 = false;
  if (t23)
    cout << "MTestApp::TestCFitsFit:                   t23: Mask == Mask_Orig returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t23: Mask == Mask_Orig returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  /// Add noise
  for (int i_row = 0; i_row < D_A2_Y.rows(); i_row++){
    for (int i_col = 0; i_col < D_A2_Y.cols(); i_col++){
      Normal<double> normalGen(D_A2_Y(i_row,i_col),sqrt(D_A2_Y(i_row,i_col)));
      d_temp = normalGen.random();
      D_A2_Y(i_row,i_col) = d_temp;
    }
  }
  cout << "MTestApp::TestCFitsFit: D_A2_Y set to " << D_A2_Y << endl;
  D_A2_Y_Diff = D_A2_Y_Orig - D_A2_Y;
  cout << "MTestApp::TestCFitsFit: D_A2_Y_Diff set to " << D_A2_Y_Diff << endl;
  //  return false;

  CS_A1_Args_LinFit(7) = CString("REJECT_IN");
  D_Reject = 2.5;
  PP_Args_LinFit[7] = &D_Reject;

  t24 = testfits->Fit(D_A2_Y, D_A2_SF, D_A1_SP, D_A1_Sky, CS_A1_Args_LinFit, PP_Args_LinFit);
  if (t24)
    cout << "MTestApp::TestCFitsFit:                   t24: Fit returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t24: Fit returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  cout << "MTestApp::TestCFitsFit: D_A1_SP = " << D_A1_SP << endl;
  cout << "MTestApp::TestCFitsFit: D_A1_Sky = " << D_A1_Sky << endl;
  cout << "MTestApp::TestCFitsFit: D_A2_YFit = " << D_A2_YFit << endl;
  cout << "MTestApp::TestCFitsFit: I_A2_Mask = " << I_A2_Mask << endl;
  for (int i=0; i<D_A2_Y.rows(); i++){
    if ((D_A1_SP_In(i) < 50.) || (fabs(D_A1_SP(i) - D_A1_SP_In(i)) < 3. * sqrt(D_A1_SP_In(i))))
      t25 = true;
    else
      t25 = false;
    if (t25)
      cout << "MTestApp::TestCFitsFit:                   t25: D_A1_SP(" << i << ")=" << D_A1_SP(i) << " == D_A1_SP_In(" << i << ")=" << D_A1_SP_In(i) << " returned \"TRUE\" (EXPECTED)" << endl;
    else
    {
      cout << "MTestApp::TestCFitsFit:                   t25: D_A1_SP(" << i << ")=" << D_A1_SP(i) << " == D_A1_SP_In(" << i << ")=" << D_A1_SP_In(i) << " returned \"FALSE\" (UNEXPECTED)" << endl;
      //return false;
    }

    if ((D_A1_Sky_In(i) < 50.) || (fabs(D_A1_Sky(i) - D_A1_Sky_In(i)) < 3. * sqrt(D_A1_Sky_In(i))))
      t25 = true;
    else
      t25 = false;
    if (t25)
      cout << "MTestApp::TestCFitsFit:                   t25: D_A1_Sky(" << i << ")=" << D_A1_Sky(i) << " == D_A1_Sky_In(" << i << ")=" << D_A1_Sky_In(i) << " returned \"TRUE\" (EXPECTED)" << endl;
    else
    {
      cout << "MTestApp::TestCFitsFit:                   t25: D_A1_Sky(" << i << ")=" << D_A1_Sky(i) << " == D_A1_Sky_In(" << i << ")=" << D_A1_Sky_In(i) << " returned \"FALSE\" (UNEXPECTED)" << endl;
      //return false;
    }
  }

  if (I_A2_Mask(6,4) == 0)
    t26 = true;
  else
    t26 = false;
  if (t26)
    cout << "MTestApp::TestCFitsFit:                   t26: I_A2_Mask(6,4) == 0 returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsFit:                   t26: I_A2_Mask(6,4) == 0 returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }





















  if (t1 && t2 && t3 && t4 && t5 && t6 && t7 && t8 && t9 && t10 && t11 && t12 && !t13 && t14 && t15 && t16 && t17 && t18 && t19 && t20 && t21 && !t22 && t23 && t24 && t25 && t26 && t27 && t28 && t29)
  {
    cout << "MTestApp::TestCFitsLinFit:                       Test PASSED" << endl;
    cout << "=========================================" << endl;
  }
  else
  {
    cout << "MTestApp::TestCFitsLinFit:                       Test FAILED" << endl;
    cout << "=========================================" << endl;
    if (testfits != NULL)
      delete(testfits);
    return false;

  }
  return false;
  return true;
}

/*************************************************************/

bool TestCFitsReformMultInvert()
{
  //  return TestConstructors("CFormatedString");
  bool t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14;
  string tempstr;// = new CString();
  int NCols = 3;
  int NRows = 7;
  int m;
  CString *p_testString = new CString("test/test_small.fits");
  CFits *testfits = new CFits();
  firstIndex i;
  secondIndex j;
  double Result;
  Array<double, 1> *P_VecArrayA = new Array<double, 1>(20);
  (*P_VecArrayA) = i;
  Array<double, 1> *P_VecArrayB = new Array<double, 1>(20);
  (*P_VecArrayB) = i * 2.;
  Array<double, 2> *P_Array45 = new Array<double, 2>(4,5);
  (*P_Array45) = i * 5 + j;
  Array<double, 2> *P_Array54 = new Array<double, 2>(5,4);
  (*P_Array54) = i * 4 + j;
  Array<double, 2> *P_Array = new Array<double, 2>(6,6);
  (*P_Array) = i * 6 + j;
  Array<double, 2> *P_TempArray = new Array<double, 2>(6,6);
  (*P_TempArray) = i * 6 + j;
  Array<double, 2> *P_ProductArr = new Array<double, 2>(6,6);
  (*P_ProductArr) = i * 6 + j;
  Array<double, 1> *P_ProductVecArr = new Array<double, 1>(4);
  (*P_ProductVecArr) = i;

  // Test: CFormatedString::ReformMultInvert
  // Tests included: CFits::operator=, CFits::Copy,
  //                 CFits::operator==, CFits::EqualValue,
  //                 CFits::operator<<, CFits::Show, CFits::ReadArray
  // require : nothing
  // ensure  : t2, t4, t6, and t9 are "FALSE", t1, t3, t5, t6, t7, t8, t10, t11, t12, t13, and t14 are "TRUE"

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert: starting init constructor" << endl;
#endif
  if (testfits != NULL)
    delete(testfits);
  testfits = new CFits(*p_testString, NCols, NRows);
#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert:           *testfits : " << *testfits;
#endif

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert: P_VecArrayA = " << *P_VecArrayA << endl;
  cout << "MTestApp::TestCFitsReformMultInvert: P_VecArrayB = " << *P_VecArrayB << endl;
  cout << "MTestApp::TestCFitsReformMultInvert: P_Array45 = " << *P_Array45 << endl;
  cout << "MTestApp::TestCFitsReformMultInvert: P_Array54 = " << *P_Array54 << endl;
  cout << "MTestApp::TestCFitsReformMultInvert: P_Array = " << *P_Array << endl;
#endif

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert: starting testfits->Reform(*P_VecArrayA, 5, 4)" << endl;
#endif
  delete P_Array54;
  P_Array54 = testfits->Reform(*P_VecArrayA, 5, 4);
#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert: P_Array54 = " << *P_Array54 << endl;
  cout << "MTestApp::TestCFitsReformMultInvert: P_Array54(1, 0) = " << (*P_Array54)(1, 0) << endl;
  cout << "MTestApp::TestCFitsReformMultInvert: P_VecArrayA(4) = " << (*P_VecArrayA)(4) << endl;
#endif
  if ((*P_Array54)(1, 0) == (*P_VecArrayA)(4))
    t1 = true;
  else
    t1 = false;
  if (t1)
    cout << "MTestApp::TestCFitsReformMultInvert:                    t1(Reform(..., 5, 4) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsReformMultInvert:                    t1(Reform(..., 5, 4) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert: starting testfits->ReformMultInvert(*P_VecArrayA, 4, 5)" << endl;
#endif

  delete P_Array45;
  P_Array45 = testfits->Reform(*P_VecArrayA, 4, 5);

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert: P_Array45 = " << *P_Array45 << endl;
  cout << "MTestApp::TestCFitsReformMultInvert: P_Array45(1, 0) = " << (*P_Array45)(1, 0) << endl;
  cout << "MTestApp::TestCFitsReformMultInvert: P_VecArrayA(5) = " << (*P_VecArrayA)(5) << endl;
#endif
  if ((*P_Array45)(1, 0) == (*P_VecArrayA)(5))
    t2 = true;
  else
    t2 = false;
  if (t2)
    cout << "MTestApp::TestCFitsReform:                    t2(Reform(..., 4, 5) = \"TRUE\" (expected) " << endl;
  else
    cout << "MTestApp::TestCFitsReform:                    t2(Reform(..., 4, 5) = \"FALSE\" (UNEXPECTED)" << endl;

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert: starting testfits->VecACrossB(*P_VecArrayA, *P_VecArrayB)" << endl;
#endif
  P_VecArrayA->resizeAndPreserve(6);
  P_VecArrayB->resizeAndPreserve(6);

  delete P_Array;
  P_Array = testfits->VecArrACrossB(*P_VecArrayA, *P_VecArrayB);
#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert: P_Array = " << *P_Array << endl;
  cout << "MTestApp::TestCFitsReformMultInvert: P_Array(2, 5) = " << (*P_Array)(2, 5) << endl;
  cout << "MTestApp::TestCFitsReformMultInvert: P_VecArrayA(5) = " << (*P_VecArrayA)(5) << endl;
  cout << "MTestApp::TestCFitsReformMultInvert: P_VecArrayB(2) = " << (*P_VecArrayB)(2) << endl;
#endif
  if ((*P_Array)(2, 5) == (*P_VecArrayA)(5) * (*P_VecArrayB)(2))
    t3 = true;
  else
    t3 = false;
  if (t3)
    cout << "MTestApp::TestCFitsReformMultInvert:                    t3(VecArrACrossB(VecArrayA, VecArrayB) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsReformMultInvert:                    t3(VecArrACrossB(VecArrayA, VecArrayB) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert: starting testfits->VecArrAScalarB(*P_VecArrayA, *P_VecArrayB)" << endl;
#endif
//  P_Array->resize(6, 6);

  double ScalarProduct = testfits->VecArrAScalarB(*P_VecArrayA, *P_VecArrayB);
#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert: ScalarProduct = " << ScalarProduct << endl;
#endif
  if (ScalarProduct == sum((*P_VecArrayA) * (*P_VecArrayB)))
    t4 = true;
  else
    t4 = false;
  if (t4)
    cout << "MTestApp::TestCFitsReformMultInvert:                    t4(VecArrAScalarB(VecArrayA, VecArrayB) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsReformMultInvert:                    t4(VecArrAScalarB(VecArrayA, VecArrayB) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert: starting testfits->MatrixTimesVecArr(*P_Array45, *P_VecArrayB)" << endl;
#endif
  P_VecArrayA->resize(4);
  P_VecArrayB->resizeAndPreserve(5);

  delete P_VecArrayA;
  P_VecArrayA = testfits->MatrixTimesVecArr((*P_Array45), (*P_VecArrayB));
#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert: P_Array45 = " << *P_Array45 << endl;
  cout << "MTestApp::TestCFitsReformMultInvert: P_VecArrayB = " << *P_VecArrayB << endl;
  cout << "MTestApp::TestCFitsReformMultInvert: P_VecArrayA = " << *P_VecArrayA << endl;
  cout << "MTestApp::TestCFitsReformMultInvert: P_VecArrayA(3) = " << (*P_VecArrayA)(3) << endl;
#endif
  P_ProductVecArr->resize(P_Array45->extent(secondDim));
  cout << "MTestApp::TestCFitsReformMultInvert: P_VecArrayB = " << (*P_VecArrayB) << endl;
  cout << "MTestApp::TestCFitsReformMultInvert: P_Array45(3, Range::all()) = " << (*P_Array45)(3, Range::all()) << endl;

  (*P_ProductVecArr) = (*P_VecArrayB) * (*P_Array45)(3, Range::all());
  cout << "MTestApp::TestCFitsReformMultInvert: P_ProductVecArr = " << (*P_ProductVecArr) << endl;
  double Sum = sum((*P_ProductVecArr));
#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert: Sum = " << Sum << endl;
#endif
  if ((*(P_VecArrayA))(3) == Sum)
    t5 = true;
  else
    t5 = false;
  if (t5)
    cout << "MTestApp::TestCFitsReformMultInvert:                    t5(MatrixTimesVecArr(Array45, VecArrayB) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsReformMultInvert:                    t5(MatrixTimesVecArr(Array54, VecArrayB) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert: starting testfits->VecArrTimesMatrix(*P_VecArrayB, *P_Array54)" << endl;
#endif
  P_VecArrayA->resize(4);
  P_VecArrayB->resizeAndPreserve(5);

  delete P_VecArrayA;
  P_VecArrayA = testfits->VecArrTimesMatrix((*P_VecArrayB), (*P_Array54));
#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert: P_Array54 = " << *P_Array54 << endl;
  cout << "MTestApp::TestCFitsReformMultInvert: P_VecArrayB = " << *P_VecArrayB << endl;
  cout << "MTestApp::TestCFitsReformMultInvert: P_VecArrayA = " << *P_VecArrayA << endl;
  cout << "MTestApp::TestCFitsReformMultInvert: P_VecArrayA(3) = " << (*P_VecArrayA)(3) << endl;
#endif
  P_ProductVecArr->resize(P_Array54->extent(firstDim));
  cout << "MTestApp::TestCFitsReformMultInvert: P_VecArrayB = " << (*P_VecArrayB) << endl;
  cout << "MTestApp::TestCFitsReformMultInvert: P_Array54(3, Range::all()) = " << (*P_Array54)(Range::all(), 3) << endl;

  (*P_ProductVecArr) = (*P_VecArrayB) * (*P_Array54)(Range::all(), 3);
  cout << "MTestApp::TestCFitsReformMultInvert: P_ProductVecArr = " << (*P_ProductVecArr) << endl;
  Sum = sum((*P_ProductVecArr));
#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert: Sum = " << Sum << endl;
#endif
  if ((*(P_VecArrayA))(3) == Sum)
    t6 = true;
  else
    t6 = false;
  if (t6)
    cout << "MTestApp::TestCFitsReformMultInvert:                    t6(VecArrTimesMatrix(VecArrayB, Array54) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsReformMultInvert:                    t6(VecArrTimesMatrix(VecArrayB, Array54) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert: starting testfits->MatrixATimesB(*P_Array54, *P_Array45)" << endl;
#endif
  P_ProductArr->resize(P_Array54->rows(), P_Array45->cols());

  delete P_ProductArr;
  P_ProductArr = testfits->MatrixATimesB((*P_Array54), (*P_Array45));
#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert: P_Array54 = " << *P_Array54 << endl;
  cout << "MTestApp::TestCFitsReformMultInvert: P_Array45 = " << *P_Array45 << endl;
  cout << "MTestApp::TestCFitsReformMultInvert: P_ProductArr = " << *P_ProductArr << endl;
#endif
  P_TempArray->resize(P_Array54->rows(), P_Array45->cols());

  (*P_TempArray) =   70., 76., 82., 88., 94.,
                    190., 212., 234., 256., 278.,
                    310., 348., 386., 424., 462.,
                    430., 484., 538., 592., 646.,
                    550., 620., 690., 760., 830.;
#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert: P_TempArray = " << (*P_TempArray) << endl;
#endif
  t7 = true;
  if (P_TempArray->cols() != P_ProductArr->cols())
    t7 = false;
  if (P_TempArray->rows() != P_ProductArr->rows())
    t7 = false;
  if (t7)
  {
    for (int m = 0; m < P_TempArray->cols(); m++)
    {
      for (int n = 0; n < P_TempArray->rows(); n++)
      {
        if ((*P_TempArray)(m, n) != (*P_ProductArr)(m, n))
          t7 = false;
      }
    }
  }
  if (t7)
    cout << "MTestApp::TestCFitsReformMultInvert:                    t7(MatrixATimesB(Array54, Array45) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsReformMultInvert:                    t7(MatrixATimesB(Array54, Array45) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert: starting testfits->MatrixBTimesA(*P_Array54, *P_Array45)" << endl;
#endif
  delete P_ProductArr;//->resize(P_Array54->cols(), P_Array45->rows());

  P_ProductArr = testfits->MatrixBTimesA((*P_Array54), (*P_Array45));
#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert: P_Array54 = " << *P_Array54 << endl;
  cout << "MTestApp::TestCFitsReformMultInvert: P_Array45 = " << *P_Array45 << endl;
  cout << "MTestApp::TestCFitsReformMultInvert: P_ProductArr = " << *P_ProductArr << endl;
#endif
  P_TempArray->resize(P_Array54->cols(), P_Array45->rows());

  (*P_TempArray) =   120., 130., 140., 150.,
                     320., 355., 390., 425.,
                     520., 580., 640., 700.,
                     720., 805., 890., 975.;
#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert: P_TempArray = " << (*P_TempArray) << endl;
#endif
  t8 = true;
  if (P_TempArray->cols() != P_ProductArr->cols())
    t8 = false;
  if (P_TempArray->rows() != P_ProductArr->rows())
    t8 = false;
  if (t8)
  {
    for (int m = 0; m < P_TempArray->cols(); m++)
    {
      for (int n = 0; n < P_TempArray->rows(); n++)
      {
        if ((*P_TempArray)(m, n) != (*P_ProductArr)(m, n))
          t8 = false;
      }
    }
  }
  if (t8)
    cout << "MTestApp::TestCFitsReformMultInvert:                    t8(MatrixBTimesA(Array54, Array45) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsReformMultInvert:                    t8(MatrixBTimesA(Array54, Array45) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  Array<double, 2> tmpArr(1, 4);
  tmpArr(0, Range(0, 3)) = ((*P_Array54)(0, Range(0, 3)) * (*P_Array45)(0, Range(0, 3)));
  cout << "MTestApp::TestCFitsReformMultInvert: tmpArr(Array54(0, Range(0,3))[=" << (*P_Array54)(0, Range(0,3)) << "] * Array45(0, Range(0, 3))[=" << (*P_Array45)(0, Range(0, 3)) << "]) = " << tmpArr << endl;

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert: starting testfits->ValueLocate(IndDVecArr, TestIndDVecArr)" << endl;
#endif
  Array<double, 1> IndDVecArr = Array<double, 1>(10);
  for (int m = 0; m < 10; m++)
    IndDVecArr(m) = (double)m;
  Array<double, 1> TestIndDVecArr(11);
  for (int m = 0; m < 11; m++)
    TestIndDVecArr(m) = (double)m;
  Array<int, 1> SVecArr(11);
  SVecArr = 0;
#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert: IndDVecArr = " << IndDVecArr << endl;
  cout << "MTestApp::TestCFitsReformMultInvert: TestIndDVecArr = " << TestIndDVecArr << endl;
  cout << "MTestApp::TestCFitsReformMultInvert: SVecArr = " << SVecArr << endl;
#endif
  Array<int, 1> *p_SVecArr = testfits->ValueLocate(IndDVecArr, TestIndDVecArr);
  SVecArr.resize(p_SVecArr->size());
  SVecArr = (*p_SVecArr);
  delete p_SVecArr;
  t9 = true;
  TestIndDVecArr(10) = 9;
  for (int m = 0; m < 10; m++)
  {
    if (SVecArr(m) != TestIndDVecArr(m))
      t9 = false;
  }
  if (t9)
    cout << "MTestApp::TestCFitsReformMultInvert:                    t9(ValueLocate(IndDVecArr=" << IndDVecArr << ", TestIndDVecArr=" << TestIndDVecArr << ")=SVecArr=" << SVecArr << ") = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsReformMultInvert:                    t9(ValueLocate(IndDVecArr=" << IndDVecArr << ", TestIndDVecArr=" << TestIndDVecArr << ")=SVecArr=" << SVecArr << ") = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert: starting SVecArr = where(SVecArr < 4, 4, SVecArr)" << endl;
  cout << "MTestApp::TestCFitsReformMultInvert: SVecArr = " << SVecArr << endl;
#endif
  SVecArr = where(SVecArr < 4, 4, SVecArr);
  t10 = true;
  for (int m = 0; m < 11; m++)
  {
    if (m < 4)
    {
      if (SVecArr(m) != 4)
        t10 = false;
    }
    else if (m < 10)
    {
      if (SVecArr(m) != m)
        t10 = false;
    }
    else
    {
      if (SVecArr(m) != 9)
        t10 = false;
    }
  }
  if (t10)
    cout << "MTestApp::TestCFitsReformMultInvert:                    t10(SVecArr(=" << SVecArr << ") = where(SVecArr < 4, 4, SVecArr) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsReformMultInvert:                    t10(SVecArr(=" << SVecArr << ") = where(SVecArr < 4, 4, SVecArr) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert: starting InvertGaussJ(P_ProductArr, P_Array)" << endl;
#endif
  P_ProductArr->resize(5, 5);
  (*P_ProductArr) = 3., 1., 2., 3., 4.,
                    1., 3., 7., 8., 9.,
                    2., 7., 3., 13., 14.,
                    3., 8., 13., 3., 19.,
                    4., 9., 14., 19., 3.;
  P_TempArray->resize(5,5);
  (*P_TempArray) = P_ProductArr->copy();
  P_Array->resize(P_TempArray->rows(), P_TempArray->rows());
  (*P_Array) = 0.;
  for (int m = 0; m < P_TempArray->rows(); m++)
    (*P_Array)(m, m) = 1.;
  Array<double, 2> TestArray;
  TestArray.resize(P_TempArray->rows(), P_TempArray->cols());
#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert: P_ProductArr = " << *P_ProductArr << endl;
  cout << "MTestApp::TestCFitsReformMultInvert: P_Array = " << *P_Array << endl;
#endif

  testfits->InvertGaussJ(*P_ProductArr, *P_Array);
#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert: P_ProductArr = " << *P_ProductArr << endl;
  cout << "MTestApp::TestCFitsReformMultInvert: P_Array = " << *P_Array << endl;
#endif
  Array<double, 2> *p_TestArray = testfits->MatrixATimesB(*P_ProductArr, *P_TempArray);
  TestArray.resize(p_TestArray->rows(), p_TestArray->cols());
  TestArray = (*p_TestArray);
  delete p_TestArray;
#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert: TestArray = " << TestArray << endl;
#endif
  t11 = true;
  for (int m = 0; m < P_ProductArr->rows(); m++)
  {
    for (int n = 0; n < P_ProductArr->cols(); n++)
    {
      if (m == n)
      {
        if (abs(TestArray(m, n) - 1.0) > 0.00001)
          t11 = false;
      }
      else
      {
        if (abs(TestArray(m, n)) > 0.00001)
          t11 = false;
      }
    }
  }
  if (t11)
    cout << "MTestApp::TestCFitsReformMultInvert:                    t11(InvertGaussJ(P_ProductArr=" << *P_ProductArr << ", P_Array=" << *P_Array << ") = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsReformMultInvert:                    t11(InvertGaussJ(P_ProductArr=" << *P_ProductArr << ", P_Array=" << *P_Array << ") = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

//  (*P_ProductArr) = P_TempArray->copy();
  p_TestArray = testfits->MatrixBTimesA(*P_ProductArr, *P_TempArray);
  TestArray.resize(p_TestArray->rows(), p_TestArray->cols());
  delete p_TestArray;
#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert: TestArray = " << TestArray << endl;
#endif
  t12 = true;
  for (int m = 0; m < P_ProductArr->rows(); m++)
  {
    for (int n = 0; n < P_ProductArr->cols(); n++)
    {
      if (m == n)
      {
        if (abs(TestArray(m, n) - 1.0) > 0.00001)
          t12 = false;
      }
      else
      {
        if (abs(TestArray(m, n)) > 0.00001)
          t12 = false;
      }
    }
  }
  if (t12)
    cout << "MTestApp::TestCFitsReformMultInvert:                    t12(InvertGaussJ(P_ProductArr=" << *P_TempArray << ", P_Array=" << *P_Array << ") = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsReformMultInvert:                    t12(InvertGaussJ(P_ProductArr=" << *P_TempArray << ", P_Array=" << *P_Array << ") = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert: starting InvertGaussJ(P_ProductArr)" << endl;
#endif
  (*P_ProductArr) = (*P_TempArray);
  TestArray.resize(P_TempArray->rows(), P_TempArray->cols());
#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert: P_ProductArr = " << *P_ProductArr << endl;
#endif

  testfits->InvertGaussJ(*P_ProductArr);
#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert: P_ProductArr = " << *P_ProductArr << endl;
  cout << "MTestApp::TestCFitsReformMultInvert: P_Array = " << *P_Array << endl;
#endif
  p_TestArray = testfits->MatrixATimesB(*P_ProductArr, *P_TempArray);
  TestArray.resize(p_TestArray->rows(), p_TestArray->cols());
  TestArray = (*p_TestArray);
  delete p_TestArray;
#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsReformMultInvert: TestArray = " << TestArray << endl;
#endif
  t13 = true;
  for (int m = 0; m < P_ProductArr->rows(); m++)
  {
    for (int n = 0; n < P_ProductArr->cols(); n++)
    {
      if (m == n)
      {
        if (abs(TestArray(m, n) - 1.0) > 0.00001)
          t13 = false;
      }
      else
      {
        if (abs(TestArray(m, n)) > 0.00001)
          t13 = false;
      }
    }
  }
  if (t13)
    cout << "MTestApp::TestCFitsReformMultInvert:                    t13(InvertGaussJ(P_ProductArr=" << *P_TempArray << ", P_Array=" << *P_Array << ") = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsReformMultInvert:                    t13(InvertGaussJ(P_ProductArr=" << *P_TempArray << ", P_Array=" << *P_Array << ") = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  if (t1 && t2 && t3 && t4 && t5 && t6 && t7 && t8 && t9 && t10 && t11 && t12 && t13)
  {
    cout << "MTestApp::TestCFitsReformMultInvert:                       Test PASSED" << endl;
    cout << "=========================================" << endl;
  }
  else
  {
    cout << "MTestApp::TestCFitsReformMultInvert:                       Test FAILED" << endl;
    cout << "=========================================" << endl;
    if (testfits != NULL)
      delete(testfits);
    return false;
  }
  if (testfits != NULL)
    delete(testfits);

  delete p_testString;
  delete P_VecArrayA;
  delete P_VecArrayB;
  delete P_Array45;
  delete P_Array54;
  delete P_Array;
  delete P_TempArray;
  delete P_ProductArr;
  delete P_ProductVecArr;
  return true;

}

/*************************************************************/

bool TestCFitsInterPolUniqCeil()
{
  // Test: CFits::InterPolUniq
  // Tests included: CFits::GetSubArrCopy
  // require : nothing
  // ensure  : t1, t2, t3, t4, t5 are "TRUE"
  bool t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14;
  string tempstr;
  int NCols = 3;
  int NRows = 7;
  int m;
  CString *p_testString = new CString("test/test_small.fits");
  CFits *p_testfits = new CFits();
  firstIndex i;
  secondIndex j;
  double Result;
  Array<double, 1> XDArr1(61);
  Array<double, 1> VDArr1(61);
  Array<double, 1> UDArr1(18);
  Array<double, 1> SearchDArr1(1);
  Array<int, 1> IA1_Result;
  Array<int, 1> IA1_ResultA;
  Array<int, 1> IA1_ToSearch;
  Array<int, 1> IA1_Test;
  Array<TinyVector<int, 1>, 1> IndIArr1;

  XDArr1 = i;
  XDArr1 /= 10.;
  XDArr1 -= 3.;
  cout << "MTestApp::TestCFitsInterPolUniqCeil: XDArr1 = " << XDArr1 << endl;

  VDArr1 = sin(XDArr1);
  cout << "MTestApp::TestCFitsInterPolUniqCeil: VDArr1 = " << VDArr1 << endl;

  UDArr1 = -2.5, -2.25, -1.85, -1.55, -1.20, -0.85, -0.5, -0.1, 0.3, 0.4, 0.75, 0.85, 1.05, 1.45, 1.85, 2.0, 2.25, 2.75;
  cout << "MTestApp::TestCFitsInterPolUniqCeil: UDArr1 = " << UDArr1 << endl;

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsInterPolUniqCeil: starting InterPol(VDArr1, XDArr1, UDArr1)" << endl;
#endif
  Array<double, 1> *p_ResultDArr1;
  if (!p_testfits->InterPol(VDArr1, XDArr1, UDArr1, p_ResultDArr1))
    return false;
  cout << "MTestApp::TestCFitsInterPolUniqCeil: ResultDArr1 = " << *p_ResultDArr1 << endl;

  t1 = true;
  double x, x1, x2, v, v1, v2, weight;
  TinyVector<int, 1> IndTVec;
  int ind;
  for (int m = 0; m < UDArr1.size() - 1; m++)
  {
    x = UDArr1(m);
//    find(IndIArr1, XDArr1(i) <= x && XDArr1(i+1) > x);
    ind = 0;
    while(true)
    {
      if (XDArr1(ind) <= x && XDArr1(ind+1) > x)
        break;
      if (ind == XDArr1.size() - 2)
      {
        ind ++;
        break;
      }
      ind++;
    }
//    cout << "MTestApp::TestCFitsInterPolUniqCeil: x = " << x << ": UDArr1(ind = " << ind << ") = " << UDArr1(ind) << ", UDArr1(ind+1 = " << ind+1 << ") = " << UDArr1(ind+1) << endl;
    x1 = XDArr1(ind);
    x2 = XDArr1(ind + 1);
    v1 = VDArr1(ind);
    v2 = VDArr1(ind + 1);
    cout << "MTestApp::TestCFitsInterPolUniqCeil: x = " << x << ", x1 = " << x1 << ", x2 = " << x2 << ", v1 = " << v1 << ", v2 = " << v2 << endl;

    weight = (x - x1) / (x2 - x1);
    v = v1 + (v2 - v1) * weight;
    cout << "MTestApp::TestCFitsInterPolUniqCeil: weight = " << weight << ", v = " << v << endl;

    cout << "MTestApp::TestCFitsInterPolUniqCeil: ResultDArr1(" << m << ") = " << (*p_ResultDArr1)(m) << endl;
    if (fabs(v - (*p_ResultDArr1)(m)) > 1.5e-3)
    {
      t1 = false;
      cout << "MTestApp::TestCFitsInterPolUniqCeil:  v(=" << v << ") != ResultDArr1(m=" << m << ")=" << (*p_ResultDArr1)(m) << "   Setting t1 to \"FALSE\" (UNEXPECTED)" << endl;
    }
  }
  delete p_ResultDArr1;
  if (t1)
    cout << "MTestApp::TestCFitsInterPolUniqCeil:                    t1(InterPol) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsInterPolUniqCeil:                    t1(InterPol) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  IA1_ToSearch.resize(20);
  IA1_ToSearch = 0, 0, 0, 1, 1, 3, 3, 4, 5, 5, 6, 8, 8, 8, 10, 14, 14, 16, 16, 19;
#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsInterPolUniqCeil: starting Uniq(IA1_ToSearch=" << IA1_ToSearch << ")" << endl;
#endif
  p_testfits->Uniq(IA1_ToSearch, IA1_Result);
  IA1_Test.resize(11);
  IA1_Test = 2, 4, 6, 7, 9, 10, 13, 14, 16, 18, 19;
#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsInterPolUniqCeil: Result = " << IA1_Result << endl;
  cout << "MTestApp::TestCFitsInterPolUniqCeil: Test   = " << IA1_Test << endl;
#endif
  if (IA1_Result.size() == IA1_Test.size())
    t2 = true;
  else
    t2 = false;
  if (t2)
    cout << "MTestApp::TestCFitsInterPolUniqCeil: Result.size(=" << IA1_Result.size() << ") == Test.size(=" << IA1_Test.size() << ") t2(Uniq) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsInterPolUniqCeil: Result.size(=" << IA1_Result.size() << ") == Test.size(=" << IA1_Test.size() << ") t2(Uniq) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  t3 = true;
  for (int m = 0; m < IA1_Test.size(); m++)
  {
    if (IA1_Result(m) != IA1_Test(m))
    {
#ifdef _DEBUG_TESTAPP_
      cout << "MTestApp::TestCFitsInterPolUniqCeil: Result(" << m << ")=" << IA1_Result(m) << " != Test(" << m << ")=" << IA1_Test(m) << " =! Setting t3 to FALSE" << endl;
#endif
      t3 = false;
    }
  }
  if (t3)
    cout << "MTestApp::TestCFitsInterPolUniqCeil: Result(=" << IA1_Result << ") == Test(=" << IA1_Test << ") t3(Uniq) = \"TRUE\" (expected) " << endl;
  else{
    cout << "MTestApp::TestCFitsInterPolUniqCeil: Result(=" << IA1_Result << ") == Test(=" << IA1_Test << ") t3(Uniq) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  IA1_Test = 0, 1, 3, 4, 5, 6, 8, 10, 14, 16, 19;
  p_testfits->GetSubArrCopy(IA1_ToSearch, IA1_Result, IA1_ResultA);
#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsInterPolUniqCeil: Range = " << Range(*(IA1_Result.data())) << endl;
  cout << "MTestApp::TestCFitsInterPolUniqCeil: Result = " << IA1_ResultA << endl;
  cout << "MTestApp::TestCFitsInterPolUniqCeil: Test   = " << IA1_Test << endl;
#endif
  if (IA1_ResultA.size() == IA1_Test.size())
    t4 = true;
  else
    t4 = false;
  if (t4)
    cout << "MTestApp::TestCFitsInterPolUniqCeil: ResultA.size(=" << IA1_ResultA.size() << ") == Test.size(=" << IA1_Test.size() << ") t4(Uniq) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsInterPolUniqCeil: ResultA.size(=" << IA1_ResultA.size() << ") == Test.size(=" << IA1_Test.size() << ") t4(Uniq) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  t5 = true;
  for (int m = 0; m < IA1_Test.size(); m++)
  {
    if (IA1_ResultA(m) != IA1_Test(m))
    {
#ifdef _DEBUG_TESTAPP_
      cout << "MTestApp::TestCFitsInterPolUniqCeil: ResultA(" << m << ")=" << IA1_ResultA(m) << " != Test(" << m << ")=" << IA1_Test(m) << " =! Setting t5 to FALSE" << endl;
#endif
      t5 = false;
    }
  }
  if (t5)
    cout << "MTestApp::TestCFitsInterPolUniqCeil: ResultA(=" << IA1_ResultA << ") == Test(=" << IA1_Test << ") t5(Uniq) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsInterPolUniqCeil: ResultA(=" << IA1_ResultA << ") == Test(=" << IA1_Test << ") t5(Uniq) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsInterPolUniqCeil: starting Ceil()" << endl;
#endif
  double D_Temp = 1.0;
  long L_Ceil = p_testfits->Ceil(D_Temp);
  if (L_Ceil == 1)
    t6 = true;
  else
    t6 = false;
  if (t6)
    cout << "MTestApp::TestCFitsInterPolUniqCeil: t6(L_Ceil(=Ceil(" << D_Temp << ")=" << L_Ceil << ") == 1) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsInterPolUniqCeil: t6(L_Ceil(=Ceil(" << D_Temp << ")=" << L_Ceil << ") == 1) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  D_Temp = 1.01;
  L_Ceil = p_testfits->Ceil(D_Temp);
  if (L_Ceil == 2)
    t7 = true;
  else
    t7 = false;
  if (t7)
    cout << "MTestApp::TestCFitsInterPolUniqCeil: t7(L_Ceil(=Ceil(" << D_Temp << ")=" << L_Ceil << ") == 2) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsInterPolUniqCeil: t7(L_Ceil(=Ceil(" << D_Temp << ")=" << L_Ceil << ") == 2) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  int I_L = 0;
  int I_Ir = 0;
  int I_K = (I_L + I_Ir);
  cout << "I_K = (I_L(=" << I_L << ") + I_Ir(=" << I_Ir << ")) = " << I_K << endl;

  I_K = (I_L + I_Ir) >> 1;
  cout << "I_K = (I_L(=" << I_L << ") + I_Ir(=" << I_Ir << ")) >> 1 = " << I_K << endl;

  I_L = 1;
  I_K = (I_L + I_Ir);
  cout << "I_K = (I_L(=" << I_L << ") + I_Ir(=" << I_Ir << ")) = " << I_K << endl;
  I_K = (I_L + I_Ir) >> 1;
  cout << "I_K = (I_L(=" << I_L << ") + I_Ir(=" << I_Ir << ")) >> 1 = " << I_K << endl;


  I_Ir = 1;
  I_K = (I_L + I_Ir);
  cout << "I_K = (I_L(=" << I_L << ") + I_Ir(=" << I_Ir << ")) = " << I_K << endl;
  I_K = (I_L + I_Ir) >> 1;
  cout << "I_K = (I_L(=" << I_L << ") + I_Ir(=" << I_Ir << ")) >> 1 = " << I_K << endl;

  I_Ir = 3;
  I_K = (I_L + I_Ir);
  cout << "I_K = (I_L(=" << I_L << ") + I_Ir(=" << I_Ir << ")) = " << I_K << endl;
  I_K = (I_L + I_Ir) >> 1;
  cout << "I_K = (I_L(=" << I_L << ") + I_Ir(=" << I_Ir << ")) >> 1 = " << I_K << endl;

  I_L = 7;
  I_K = (I_L + I_Ir);
  cout << "I_K = (I_L(=" << I_L << ") + I_Ir(=" << I_Ir << ")) = " << I_K << endl;
  I_K = (I_L + I_Ir) >> 1;
  cout << "I_K = (I_L(=" << I_L << ") + I_Ir(=" << I_Ir << ")) >> 1 = " << I_K << endl;

  for (I_K = -1; I_K < 30; I_K++)
  {
    int I_KTemp = I_K;
    cout << "Start: I_K = " << I_KTemp;
    I_KTemp = (I_K) >> 1;
    cout << "End:   I_K = " << I_K << endl;
    cout << "End:   I_KTemp = " << I_KTemp << endl;
  }

  /*
  unsigned long UL_Temp = 7;
  while(UL_Temp > 0)
  {
    UL_Temp -= 2;
    cout << "UL_Temp = " << UL_Temp << endl;
}

  int *istack;
  istack = ivector(1, 50);
  cout << "istack = " << istack << endl;

  return false;
  */
#ifdef _DEBUG_TESTAPP_
  cout << "MTestApp::TestCFitsInterPolUniqCeil: starting SortIndices()" << endl;
#endif

  IA1_ToSearch = 4, 45, 123, 435, 132, 564, 57, 875, 54, 56, 635, 8456, 38, 6458, 710, 5414, 14, 1546, 1446, 149;
  Array<double, 1> *p_D_A1_Result = p_testfits->FixD(IA1_ToSearch);
  Array<int, 1> *p_IA1_ResultB = p_testfits->SortIndices(*p_D_A1_Result);
  delete p_D_A1_Result;

  t8 = true;
  for (int m = 0; m < 19; m++)
  {
    if (IA1_ToSearch((*p_IA1_ResultB)(m)) > IA1_ToSearch((*p_IA1_ResultB)(m+1)))
    {
      cout << " MTestApp::TestCFitsInterPolUniqCeil: IA1_ToSearch(IA1_ResultB(m=" << m << ")=" << (*p_IA1_ResultB)(m) << ") = " << IA1_ToSearch((*p_IA1_ResultB)(m+1)) << " > IA1_ToSearch(IA1_ResultB(m+1=" << m+1 << ")=" << (*p_IA1_ResultB)(m+1) << " => Setting t8 to false" << endl;
      t8 = false;
    }
    else
    {
      cout << " MTestApp::TestCFitsInterPolUniqCeil: IA1_ToSearch(IA1_ResultB(m=" << m << ")=" << (*p_IA1_ResultB)(m) << ") = " << IA1_ToSearch((*p_IA1_ResultB)(m+1)) << " <= IA1_ToSearch(IA1_ResultB(m+1=" << m+1 << ")=" << (*p_IA1_ResultB)(m+1) << " (expected)" << endl;
    }
  }
  if (t8)
    cout << "MTestApp::TestCFitsInterPolUniqCeil: t8(SortIndices(" << IA1_ToSearch << ")=" << *p_IA1_ResultB << ") = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsInterPolUniqCeil: t8(SortIndices(" << IA1_ToSearch << ")=" << *p_IA1_ResultB << ") = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  delete p_IA1_ResultB;
  /*
  Array<double, 1> D_A1_Temp(10);
  cout << "D_A1_Temp = " << D_A1_Temp << endl;
  D_A1_Temp = i;
  cout << "D_A1_Temp = i = " << D_A1_Temp << endl;
  for (int m = 0; m < 10; m++)
    D_A1_Temp(m) = (double)m;
  cout << "D_A1_Temp = (double)m = " << D_A1_Temp << endl;

  Array<int, 1> A(5);
  Array<int, 1> B(5);
  Array<float, 1> C(5);
  cout << "A<int, 1>   = " << A << endl;
  cout << "B<int, 1>   = " << B << endl;
  cout << "C<float, 1> = " << C << endl;
  A = 1, 2, 3, 5, 7;
  B = 2, 2, 2, 7, 9;
  C = A / B;
  cout << "A<int, 1>   set to " << A << endl;
  cout << "B<int, 1>   set to " << B << endl;
  cout << "C<float, 1> = A / B = " << C << endl;
  C = A / p_testfits->FixD(B);
  cout << "C<float, 1> = A / casted B" << C << endl;
  C = C / B;
  cout << "C<float, 1> / B = " << C << endl;

  Array<int, 2> Mat(6, 3);
  Mat = i * 3 + j;
  cout << "Mat = " << Mat << endl;
  Array<int, 2> TMat;
  TMat.resize(3, 6);
  TMat = Mat.transpose(secondDim, firstDim);
  cout << "Result from transpose: TMat = " << TMat << endl;

  Array<int, 2> Vec(1, 6);
  Vec = j;
  cout << "Vec = " << Vec << endl;
  TMat.resize(6, 1);
  TMat = Vec.transpose(secondDim, firstDim);
  cout << "Result from transpose: TMat = " << TMat << endl;
  cout << "Mat = " << Mat << endl;
*/
  return (t1 && t2 && t3 && t4 && t5 && t6 && t7 && t8);
}

/*************************************************************/

bool TestCFitsPiskunov()
{
  // Test: CFits::Methods of Piskunov and Valenti
  // Tests included: CFits::GetSubArrCopy
  // require : nothing
  // ensure  : t1, t2, t3, t4, t5 are "TRUE"
  bool t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t35, t36, t37, t38, t39, t40;
  string tempstr;
  int NCols = 3;
  int NRows = 7;
  int m;
  CString *p_testString = new CString("test/combinedFlat_s.fits");
  CString *p_CS_DatabaseString = new CString("test/database/apcombinedFlat_s");
  CFits *p_testfits = new CFits();
  CFits *p_testfitsP = new CFits();
  firstIndex i;
  secondIndex j;

  t1 = p_testfits->SetFileName(*p_testString);
  if (t1)
    cout << "MTestApp::TestCFitsPiskunov: t1(p_testfits->SetFileName) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t1(p_testfits->SetFileName) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  t2 = p_testfits->ReadArray();
  if (t2)
    cout << "MTestApp::TestCFitsPiskunov: t2(p_testfits->ReadArray) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t2(p_testfits->ReadArray) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  t3 = p_testfits->ClassInvariant();
  if (t3)
    cout << "MTestApp::TestCFitsPiskunov: t3(p_testfits->ClassInvariant) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t3(p_testfits(=" << *p_testfits << ")->ClassInvariant) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  t4 = p_testfits->SetDatabaseFileName(*p_CS_DatabaseString);
  if (t4)
    cout << "MTestApp::TestCFitsPiskunov: t4(p_testfits->SetDatabaseFileName) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t4(p_testfits(=" << *p_testfits << ")->SetDatabaseFileName) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  t5 = p_testfits->ClassInvariant();
  if (t5)
    cout << "MTestApp::TestCFitsPiskunov: t5(p_testfits->ClassInvariant) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t5(p_testfits(=" << *p_testfits << ")->ClassInvariant) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  t6 = p_testfits->Set_NApertures(2);
  if (t6)
    cout << "MTestApp::TestCFitsPiskunov: t6(p_testfits->Set_NApertures) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t6(p_testfits(=" << *p_testfits << ")->Set_NApertures) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  t7 = p_testfits->ReadDatabaseEntry();
  if (t7)
    cout << "MTestApp::TestCFitsPiskunov: t7(p_testfits->ReadDatabaseEntry) = \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t7(p_testfits(=" << *p_testfits << ")->ReadDatabaseEntry) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  p_testfitsP->Copy( *p_testfits );
//  p_testfits->ReadDatabaseEntry();
  cout << "MTestApp::TestCFitsPiskunov: Database Entry for p_testfits read" << endl << " p_testfits = " << *p_testfits << endl;

  (*MTestApp::P_Log) << "MTestApp::TestCFitsPiskunov: before t8: p_testfitsP->GetFileName() = " << p_testfitsP->GetFileName() << endl;

  Array<double, 2> *P_D_A2_Coeffs = p_testfits->Get_Coeffs();
  if (abs((*P_D_A2_Coeffs)(p_testfits->Get_NApertures() - 1, 2) - 6.25942) < 1.1e-5)
    t8 = true;
  else
    t8 = false;
  if (t8)
    cout << "MTestApp::TestCFitsPiskunov: t8(fabs(p_testfits->Get_Coeffs()(p_testfits->Get_NApertures(=" << p_testfits->Get_NApertures() << ")=" << (*P_D_A2_Coeffs)(p_testfits->Get_NApertures()-1, 2) << " - 6.25942) = " << abs((*P_D_A2_Coeffs)(p_testfits->Get_NApertures() - 1, 2) - 6.25942) << " < 1.1e-5  =  \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t8(fabs(p_testfits->Get_Coeffs()(p_testfits->Get_NApertures(=" << p_testfits->Get_NApertures() << ")=" << (*P_D_A2_Coeffs)(p_testfits->Get_NApertures()-1, 2) << " - 6.25942) = " << abs((*P_D_A2_Coeffs)(p_testfits->Get_NApertures() - 1, 2) - 6.25942) << " < 1.1e-5  =  \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }
  delete(P_D_A2_Coeffs);

  (*MTestApp::P_Log) << "MTestApp::TestCFitsPiskunov: after t8: p_testfitsP->GetFileName() = " << p_testfitsP->GetFileName() << endl;

  Array<double, 1> D_A1_Temp(10);
  D_A1_Temp = i;

  Array<double, 1> D_A1_Test = D_A1_Temp(Range(2, 6));

  if (D_A1_Test(2) == D_A1_Temp(4))
    t9 = true;
  else
    t9 = false;
  if (t9)
    cout << "MTestApp::TestCFitsPiskunov: t9(D_A1_Test(2)=" << D_A1_Test(2) << " == D_A1_Temp(4)=" << D_A1_Temp(4) << ") returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t9(D_A1_Test(2)=" << D_A1_Test(2) << " == D_A1_Temp(4)=" << D_A1_Temp(4) << ") = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  D_A1_Test(2) = 6;
  if (D_A1_Test(2) == D_A1_Temp(4))
    t10 = true;
  else
    t10 = false;
  if (t10)
    cout << "MTestApp::TestCFitsPiskunov: t10(D_A1_Test(2)=" << D_A1_Test(2) << " == D_A1_Temp(4)=" << D_A1_Temp(4) << ") returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t10(D_A1_Test(2)=" << D_A1_Test(2) << " == D_A1_Temp(4)=" << D_A1_Temp(4) << ") = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  t11 = p_testfits->CalcTraceFunctions();
  if (t11)
    cout << "MTestApp::TestCFitsPiskunov: t11(p_testfits->CalcTraceFunctions()) returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t11(p_testfits->CalcTraceFunctions()) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  cout << "MTestApp::TestCFitsPiskunov: Trace functions for p_testfits calculated" << endl << " p_testfits->Get_XCenters() = " << p_testfits->Get_XCenters() << endl;

  Array<double, 2> D_A2_Temp;
  D_A2_Temp.resize(p_testfits->Get_NApertures(), p_testfits->GetNRows());
  D_A2_Temp = 0.;

  cout << "MTestApp::TestCFitsPiskunov: Starting Copy Constructor" << endl;
  CFits *P_TestFitsA = new CFits(*p_testfits);
  t12 = P_TestFitsA->EqualValue(*p_testfits);
  if (t12)
    cout << "MTestApp::TestCFitsPiskunov: t12(P_TestFitsA->EqualValue(p_testfits) returned \"TRUE\" (expected) " << endl;
  else
  {
//    cout << "MTestApp::TestCFitsPiskunov: t12(P_TestFitsA(=" << *P_TestFitsA << ")->EqualValue(p_testfits=" << *p_testfits << ")) = \"FALSE\" (UNEXPECTED)" << endl;
    cout << "MTestApp::TestCFitsPiskunov: t12(P_TestFitsA)->EqualValue(p_testfits)) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  int i_nargs = 2;
  CString CS_TempFileName("test/SlitFunc.dat");

  t13 = P_TestFitsA->SetFileName(*(new CString("test/combinedFlat_test.fits")));
  if (t13)
    cout << "MTestApp::TestCFitsPiskunov: t13(P_TestFitsA->SetFileName()) returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t13(P_TestFitsA->SetFileName()) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  t14 = P_TestFitsA->MarkCenters();
  if (t14)
    cout << "MTestApp::TestCFitsPiskunov: t14(P_TestFitsA->MarkCenters()) returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t14(P_TestFitsA->MarkCenters()) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  t15 = P_TestFitsA->SetFileName(CString("test/MarkedCenters.fits"));
  if (t15)
    cout << "MTestApp::TestCFitsPiskunov: t15(P_TestFitsA->SetFileName) returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t15(P_TestFitsA->SetFileName) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

//  (*MTestApp::P_Log) << "MTestApp::TestCFitsPiskunov: after t15: p_testfitsP->GetFileName() = " << p_testfitsP->GetFileName() << endl;

  t16 = P_TestFitsA->WriteArray();
  if (t16)
    cout << "MTestApp::TestCFitsPiskunov: t16(P_TestFitsA->WriteArray()) returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t16(P_TestFitsA->WriteArray()) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  Array<double, 1> D_A1_TestB(10);
  D_A1_TestB = i;
  Array<int, 1> I_A1_Indices(5);
  I_A1_Indices = 3, 1, 6, 4, 9;
  Array<double, 1> D_A1_ExpResult(5);
  D_A1_ExpResult = P_TestFitsA->GetSubArr(D_A1_TestB, I_A1_Indices);
  D_A1_ExpResult -= 3.;
  t17 = P_TestFitsA->Set_SubArray(D_A1_TestB,
                                  I_A1_Indices,
                                  D_A1_ExpResult);
  if (t17)
    cout << "MTestApp::TestCFitsPiskunov: t17(P_TestFitsA->SetSubArr()) returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t17(P_TestFitsA->SetSubArr()) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  t18 = true;
  for (int m = 0; m < I_A1_Indices.size(); m++)
  {
    if (D_A1_TestB(I_A1_Indices(m)) != I_A1_Indices(m) - 3.)
    {
      cout << "MTestApp::TestCFitsPiskunov: t18: D_A1_TestB(I_A1_Indices(" << m << ")=" << I_A1_Indices(m) << ")=" << D_A1_TestB(I_A1_Indices(m)) << " != I_A1_Indices(m)-3 => Returning FALSE" << endl;
      t18 = false;
    }
  }
  if (t18)
    cout << "MTestApp::TestCFitsPiskunov: t18(Comparison P_TestFitsA->SetSubArr()) returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t18(Comparison P_TestFitsA->SetSubArr()) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  CString fntest("test/aperture_14.dat");
  p_testfits->WriteCenters(14, fntest);
  t19 = p_testfits->FileAccess(fntest);
  if (t19)
    cout << "MTestApp::TestCFitsPiskunov: t19(Access(" << fntest << ")) returned \"TRUE\" (EXPECTED)" << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t19(Access(" << fntest << ")) returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  fntest.Set("test/2D_PixArray.dat");
  p_testfits->WriteArrayToFile(P_TestFitsA->GetPixArray(), fntest, CString("ascii"));
  t20 = p_testfits->FileAccess(fntest);
  if (t20)
    cout << "MTestApp::TestCFitsPiskunov: t20(Access(" << fntest << ")) returned \"TRUE\" (EXPECTED)" << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t20(Access(" << fntest << ")) returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  t21 = P_TestFitsA->SetFileName(*(new CString("test/test_FEROS_301x51.fits")));
  if (t21)
    cout << "MTestApp::TestCFitsPiskunov: t21(P_TestFitsA->SetFileName()) returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t21(P_TestFitsA->SetFileName()) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  Array<double, 2> D_A2_Im;
  D_A2_Im.resize(51,301);

  ifstream ifs_im("test/im.dat");
  ifs_im >> D_A2_Im;
  Array<double, 1> D_A1_YCen;
  D_A1_YCen.resize(301);
  ifstream ifs_yc("test/ycen.dat");
  ifs_yc >> D_A1_YCen;
  Array<double, 1> D_A1_SF;
  D_A1_SF.resize(51);
  ifstream ifs_sf("test/sf.dat");
  ifs_sf >> D_A1_SF;
  Array<double, 1> D_A1_SP;
  D_A1_SP.resize(301);
  ifstream ifs_sp("test/sp.dat");
  ifs_sp >> D_A1_SP;

  Array<double, 2> D_A2_TIm(D_A2_Im.transpose(secondDim, firstDim));
  ofstream ofs_tim("test/tim.dat");
  ofs_tim << D_A2_TIm;

  cout << "MTestApp.TestCFitsPiskunov: Starting SlitFunc" << endl;
//  cout << "MTestApp.TestCFitsPiskunov: D_A2_Im = " << D_A2_Im << endl;
//  cout << "MTestApp.TestCFitsPiskunov: D_A1_YCen = " << D_A1_YCen << endl;
  (*MTestApp::P_Log) << "MTestApp.TestCFitsPiskunov: Starting SlitFunc" << endl;
//  (*MTestApp::P_Log) << "MTestApp.TestCFitsPiskunov: D_A2_Im = " << D_A2_Im << endl;
//  (*MTestApp::P_Log) << "MTestApp.TestCFitsPiskunov: D_A1_YCen = " << D_A1_YCen << endl;
  Array<double, 1> D_A1_SF_Out;
  Array<double, 1> D_A1_SP_Out;
  Array<CString, 1> CS_A1_Args(10);
  CS_A1_Args = CString("");
  void **PP_Args = (void**)malloc(sizeof(void*) * 10);

//  PP_CS[0] = new CString("NOISE");
//  double noise = 10;
//  PP_Args[0] = &noise;

  CS_A1_Args(0).Set("OVERSAMPLE");
  int OverSample = 10;
  PP_Args[0] = &OverSample;

  CS_A1_Args(1).Set("LAMBDA_SF");
  double Lambda_SF = 40.;
  PP_Args[1] = &Lambda_SF;

  CS_A1_Args(2).Set("LAMBDA_SP");
  double Lambda_SP = 400.;
  PP_Args[2] = &Lambda_SP;

  Array<double, 2> D_A2_ImOut;
  D_A2_ImOut.resize(D_A2_TIm.rows(), D_A2_TIm.cols());
  CS_A1_Args(3).Set("IM_OUT");
  PP_Args[3] = &D_A2_ImOut;

  t22 = P_TestFitsA->SlitFunc(D_A2_TIm,
                              0,
                              D_A1_YCen,
                              D_A1_SP_Out,
                              D_A1_SF_Out,
                              *(const_cast<const Array<CString, 1>*>(&CS_A1_Args)),
                              PP_Args);
  if (t22)
    cout << "MTestApp::TestCFitsPiskunov: t22(SlitFunc) returned \"TRUE\" (EXPECTED)" << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t22(SlitFunc) returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  ofstream *P_OFS_SF = new ofstream("test/SlitFunc_SF_Out.text");
  for (int III=0; III < D_A1_SF_Out.size(); III++)
    (*P_OFS_SF) << D_A1_SF_Out(III) << endl;
  delete(P_OFS_SF);
  (*MTestApp::P_Log) << "MTestApp.TestCFitsPiskunov: SlitFunc ready: D_A1_SF_Out = " << D_A1_SF_Out << endl;
  (*MTestApp::P_Log) << "MTestApp.TestCFitsPiskunov: SlitFunc ready: D_A1_SP_Out = " << D_A1_SP_Out << endl;
  (*MTestApp::P_Log) << "MTestApp.TestCFitsPiskunov: SlitFunc ready: D_A2_ImOut = " << D_A2_ImOut << endl;

  /// Compare to results of IDL
  /// ImOut
  ifstream *P_IFS = new ifstream("test/slitfunc_im_out.dat");
  Array<double, 2> D_A2_Pis_Im_Out;
  (*P_IFS) >> D_A2_Pis_Im_Out;
  //cout << "MTestApp::TestCFitsPiskunov: D_A2_Pis_Im_Out read: " << D_A2_Pis_Im_Out << endl;
  //(*MTestApp::P_Log) << "MTestApp.TestCFitsPiskunov: D_A2_Pis_Im_Out read: " << D_A2_Pis_Im_Out << endl;
  t23 = true;
  for (int m = 0; m < D_A2_ImOut.rows(); m++)
  {
    for (int n = 0; n < D_A2_ImOut.cols(); n++)
    {
      if (fabs(D_A2_Pis_Im_Out(n,m) - D_A2_ImOut(m,n)) / D_A2_ImOut(m,n) > 0.00001)
      {
        cout << "CFits::TestCFitsPiskunov: t23: ERROR: fabs(=" << fabs(D_A2_Pis_Im_Out(n,m) - D_A2_ImOut(m,n)) << "): D_A2_ImOut(m=" << m << ",n=" << n << ")=" << D_A2_ImOut(m,n) << " != D_A2_Pis_Im_Out(n,m)=" << D_A2_Pis_Im_Out(n,m) << endl;
        (*MTestApp::P_Log) << "CFits::TestCFitsPiskunov: t23: ERROR: fabs(=" << fabs(D_A2_Pis_Im_Out(n,m) - D_A2_ImOut(m,n)) << "): D_A2_ImOut(m=" << m << ",n=" << n << ")=" << D_A2_ImOut(m,n) << " != D_A2_Pis_Im_Out(n,m)=" << D_A2_Pis_Im_Out(n,m) << endl;
        t23 = false;
        return false;
      }
    }
  }
  cout << "CFits::TestCFitsPiskunov: t23: D_A2_ImOut == D_A2_Pis_Im_Out (EXPECTED)" << endl;
  (*MTestApp::P_Log) << "CFits::TestCFitsPiskunov: t23: D_A2_ImOut == D_A2_Pis_Im_Out (EXPECTED)" << endl;

  /// SFOut
  delete(P_IFS);
  P_IFS = new ifstream("test/slitfunc_sf_out.dat");
  Array<double, 1> D_A1_Pis_SFOut;
  (*P_IFS) >> D_A1_Pis_SFOut;
  //cout << "MTestApp::TestCFitsPiskunov: D_A1_Pis_SFOut read: " << D_A1_Pis_SFOut << endl;
  //(*MTestApp::P_Log) << "MTestApp.TestCFitsPiskunov: D_A1_Pis_SFOut read: " << D_A1_Pis_SFOut << endl;
  t24 = true;
  for (int m = 0; m < D_A1_SF_Out.size(); m++)
  {
    if (fabs(D_A1_Pis_SFOut(m) - D_A1_SF_Out(m)) / D_A1_SF_Out(m) > 0.00001)
    {
      cout << "CFits::TestCFitsPiskunov: t24: ERROR: fabs(=" << fabs(D_A1_Pis_SFOut(m) - D_A1_SF_Out(m)) << "): D_A1_SF_Out(m=" << m << ")=" << D_A1_SF_Out(m) << " != D_A1_Pis_SFOut(m)=" << D_A1_Pis_SFOut(m) << endl;
      (*MTestApp::P_Log) << "CFits::TestCFitsPiskunov: t25: ERROR: fabs(=" << fabs(D_A1_Pis_SFOut(m) - D_A1_SF_Out(m)) << "): D_A1_SF_Out(m=" << m << ")=" << D_A1_SF_Out(m) << " != D_A1_Pis_SFOut(m)=" << D_A1_Pis_SFOut(m) << endl;
      t24 = false;
      return false;
    }
  }
  cout << "CFits::TestCFitsPiskunov: t24: D_A1_SF_Out == D_A1_Pis_SFOut (EXPECTED)" << endl;
  (*MTestApp::P_Log) << "CFits::TestCFitsPiskunov: t24: D_A1_SF_Out == D_A1_Pis_SFOut (EXPECTED)" << endl;

  /// SPOut
  delete(P_IFS);
  P_IFS = new ifstream("test/slitfunc_sp_out.dat");
  Array<double, 1> D_A1_Pis_SPOut;
  (*P_IFS) >> D_A1_Pis_SPOut;
  //cout << "MTestApp::TestCFitsPiskunov: D_A1_Pis_SPOut read: " << D_A1_Pis_SPOut << endl;
  //(*MTestApp::P_Log) << "MTestApp.TestCFitsPiskunov: D_A1_Pis_SPOut read: " << D_A1_Pis_SPOut << endl;
  t25 = true;
  for (int m = 0; m < D_A1_SP_Out.size(); m++)
  {
    if (fabs(D_A1_Pis_SPOut(m) - D_A1_SP_Out(m)) / D_A1_SP_Out(m) > 0.00001)
    {
      cout << "CFits::TestCFitsPiskunov: t25: ERROR: fabs(=" << fabs(D_A1_Pis_SPOut(m) - D_A1_SP_Out(m)) << "): D_A1_SP_Out(m=" << m << ")=" << D_A1_SP_Out(m) << " != D_A1_Pis_SPOut(m)=" << D_A1_Pis_SPOut(m) << endl;
      (*MTestApp::P_Log) << "CFits::TestCFitsPiskunov: t25: ERROR: fabs(=" << fabs(D_A1_Pis_SPOut(m) - D_A1_SP_Out(m)) << "): D_A1_SP_Out(m=" << m << ")=" << D_A1_SP_Out(m) << " != D_A1_Pis_SPOut(m)=" << D_A1_Pis_SPOut(m) << endl;
      t25 = false;
      return false;
    }
  }
  cout << "CFits::TestCFitsPiskunov: t25: D_A1_SP_Out == D_A1_Pis_SPOut (EXPECTED)" << endl;
  (*MTestApp::P_Log) << "CFits::TestCFitsPiskunov: t25: D_A1_SP_Out == D_A1_Pis_SPOut (EXPECTED)" << endl;


  Array<double, 2> trans(5,7);
  trans = 10*i+j;
  cout << "MTestApp.TestCFitsPiskunov: trans = " << trans << endl;
  double *d_trans = trans.data();
  for (int mmm = 0; mmm < 35; mmm++)
  {
    cout << "MTestApp.TestCFitsPiskunov: trans.data[" << mmm << "] = " << d_trans[mmm] << endl;
  }

  Array<double, 2> ttrans(7,5);
  ttrans = trans.transpose(secondDim, firstDim);
  cout << "MTestApp.TestCFitsPiskunov: ttrans = " << ttrans << endl;
  d_trans = ttrans.data();
  for (int mmm = 0; mmm < 35; mmm++)
  {
    cout << "MTestApp.TestCFitsPiskunov: ttrans.data[" << mmm << "] = " << d_trans[mmm] << endl;
  }

  Array<double, 1> A(4);
  Array<double, 1> B(4);
  Array<double, 1> C(4);
  Array<double, 1> R(4);
  Array<double, 1> Expec(4);
  Array<double, 1> TriDagRes(4);
  A = 0., 2., 2., 2.;
  B = -4., -4., -4., -4.;
  C = 1.0, 1.0, 1.0, 0.;
  R = 6., -8., -5., 8.;
  Expec = -1., 2., 2., -1.;
  p_testfits->TriDag(A, B, C, R, TriDagRes);
  t26 = true;
  for (int m = 0; m < 4; m++)
  {
    if (Expec(m) != TriDagRes(m))
    {
      cout << "MTestApp::TestCFitsPiskunov: t26: Expec(m=" << m << ")=" << Expec(m) << " != TriDagRes(m)=" << TriDagRes(m) << "! UNEXPECTED => Returning FALSE" << endl;
      t26 = false;
      return false;
    }
  }
  cout << "MTestApp::TestCFitsPiskunov: t26: Expec=" << Expec << " = TriDagRes(m)=" << TriDagRes << " (EXPECTED)" << endl;

  /*
  GeneralArrayStorage<2> storage;
  storage.ordering() = secondDim, firstDim;
  storage.base() = 0, 0;
  storage.ascendingFlag() = true, true;

  Array<double, 2> D_A2_Tmp(3,5);
  D_A2_Tmp = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14;
  Array<double, 2> D_A2_IDLArray(5, 3, storage);
  D_A2_IDLArray = 0;//, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14;


  cout << "MTestApp.TestCFitsPiskunov: D_A2_Tmp = " << D_A2_Tmp << endl;
  cout << "MTestApp.TestCFitsPiskunov: D_A2_IDLArray = " << D_A2_IDLArray << endl;
  //  for (int mm = 0; mm < D_A2_IDLArray.rows(); m++)
  int cols = D_A2_IDLArray.cols();
  int rows = D_A2_IDLArray.rows();
  cout << "MTestApp.TestCFitsPiskunov: cols = " << cols << ", rows = " << rows << endl;
  D_A2_IDLArray = D_A2_Tmp.transpose(secondDim, firstDim);
  cout << "MTestApp.TestCFitsPiskunov: D_A2_Tmp = " << D_A2_Tmp << endl;
  cout << "MTestApp.TestCFitsPiskunov: D_A2_IDLArray = " << D_A2_IDLArray << endl;
  t21 = (D_A2_Tmp(2, 3) == D_A2_IDLArray(3, 2));
  if (t21)
    cout << "MTestApp::TestCFitsPiskunov: t21(D_A2_Tmp(2,3)(=" << D_A2_Tmp(2, 3) << ") == D_A2_IDLArray(3,2)(=" << D_A2_IDLArray(3, 2) << ")) returned \"TRUE\" (EXPECTED)" << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t21(D_A2_Tmp(2,3)(=" << D_A2_Tmp(2, 3) << ") == D_A2_IDLArray(3,2)(=" << D_A2_IDLArray(3, 2) << ")) returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  D_A2_IDLArray(1, 2) = 100;
  cout << "MTestApp.TestCFitsPiskunov: t22: D_A2_Tmp = " << D_A2_Tmp << endl;
  cout << "MTestApp.TestCFitsPiskunov: t22: D_A2_IDLArray = " << D_A2_IDLArray << endl;
  t22 = (D_A2_Tmp(2, 1) != D_A2_IDLArray(1, 2));
  if (t22)
    cout << "MTestApp::TestCFitsPiskunov: t22(D_A2_Tmp(2,1)(=" << D_A2_Tmp(2, 1) << ") != D_A2_IDLArray(1,2)(=" << D_A2_IDLArray(1, 2) << ")) returned \"TRUE\" (EXPECTED)" << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t22(D_A2_Tmp(2,1)(=" << D_A2_Tmp(2, 1) << ") != D_A2_IDLArray(1,2)(=" << D_A2_IDLArray(1, 2) << ")) returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  D_A2_Tmp = 10. * i + j;
  D_A2_IDLArray = 10. * i + j;
  cout << "MTestApp.TestCFitsPiskunov: t23: D_A2_Tmp = " << D_A2_Tmp << endl;
  cout << "MTestApp.TestCFitsPiskunov: t23: D_A2_IDLArray = " << D_A2_IDLArray << endl;
  t23 = (D_A2_Tmp(2, 1) == D_A2_IDLArray(1, 2));
  if (t23)
    cout << "MTestApp::TestCFitsPiskunov: t23(D_A2_Tmp(2,1)(=" << D_A2_Tmp(2,1) << ") != D_A2_IDLArray(1,2)(=" << D_A2_IDLArray(1, 2) << ")) returned \"TRUE\" (EXPECTED)" << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t23(D_A2_Tmp(2,1)(=" << D_A2_Tmp(2, 1) << ") != D_A2_IDLArray(1,2)(=" << D_A2_IDLArray(1, 2) << ")) returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }
  */
  ///MkSlitFunc
  double D_XLeftLim = 23;
  double D_XRightLim = 23;
  int I_YUpperLim = 4089;
  int I_YLowerLim = 1136;
  int I_SwathWidth = 300;
  int I_OverSample = 10;
  int I_LamSP = 20;
  double D_LamSF = 40.;
  double D_Noise;
  double D_Gain = 1.25;
  double D_ReadN = 17.2177;

  delete(P_IFS);
  P_IFS = new ifstream("test/mkslitf_yscatter_above_in.dat");
  Array<double, 1> D_A1_YScatterAbove;
  (*P_IFS) >> D_A1_YScatterAbove;
  //cout << "MTestApp::TestCFitsPiskunov: D_A1_YScatterAbove read: " << D_A1_YScatterAbove << endl;
  //(*MTestApp::P_Log) << "MTestApp.TestCFitsPiskunov: D_A1_YScatterAbove read: " << D_A1_YScatterAbove << endl;

  delete(P_IFS);
  P_IFS = new ifstream("test/mkslitf_yscatter_below_in.dat");
  Array<double, 1> D_A1_YScatterBelow;
  (*P_IFS) >> D_A1_YScatterBelow;
  //cout << "MTestApp::TestCFitsPiskunov: D_A1_YScatterBelow read: " << D_A1_YScatterBelow << endl;
  //(*MTestApp::P_Log) << "MTestApp.TestCFitsPiskunov: D_A1_YScatterBelow read: " << D_A1_YScatterBelow << endl;

  delete(P_IFS);
  P_IFS = new ifstream("test/mkslitf_ycen_in.dat");
  Array<double, 1> D_A1_YCenters;
  (*P_IFS) >> D_A1_YCenters;
  //cout << "MTestApp::TestCFitsPiskunov: D_A1_YCenters read: " << D_A1_YCenters << endl;
  //(*MTestApp::P_Log) << "MTestApp.TestCFitsPiskunov: D_A1_YCenters read: " << D_A1_YCenters << endl;

  delete(P_IFS);
  P_IFS = new ifstream("test/mkslitf_scatter_below_in.dat");
  Array<double, 1> D_A1_ScatterBelow;
  (*P_IFS) >> D_A1_ScatterBelow;
  //cout << "MTestApp::TestCFitsPiskunov: D_A1_ScatterBelow read: " << D_A1_ScatterBelow << endl;
  //(*MTestApp::P_Log) << "MTestApp.TestCFitsPiskunov: D_A1_ScatterBelow read: " << D_A1_ScatterBelow << endl;

  delete(P_IFS);
  P_IFS = new ifstream("test/mkslitf_scatter_above_in.dat");
  Array<double, 1> D_A1_ScatterAbove;
  (*P_IFS) >> D_A1_ScatterAbove;
  //cout << "MTestApp::TestCFitsPiskunov: D_A1_ScatterAbove read: " << D_A1_ScatterAbove << endl;
  //(*MTestApp::P_Log) << "MTestApp.TestCFitsPiskunov: D_A1_ScatterAbove read: " << D_A1_ScatterAbove << endl;

  delete(P_IFS);
  P_IFS = new ifstream("test/mkslitf_im_in.dat");
  Array<double, 2> D_A2_Im_InT;
  (*P_IFS) >> D_A2_Im_InT;
//  cout << "MTestApp::TestCFitsPiskunov: D_A2_Im_InT read: " << D_A2_Im_InT.transpose(secondDim, firstDim) << endl;
  //(*MTestApp::P_Log) << "MTestApp.TestCFitsPiskunov: D_A2_Im_InT read: " << D_A2_Im_InT << endl;
  Array<double, 2> D_A2_Im_In(D_A2_Im_InT.transpose(secondDim, firstDim));
  //cout << "MTestApp::TestCFitsPiskunov: D_A2_Im_In read: " << D_A2_Im_In.transpose(secondDim, firstDim) << endl;
  //(*MTestApp::P_Log) << "MTestApp.TestCFitsPiskunov: D_A2_Im_In read: " << D_A2_Im_In.transpose(secondDim, firstDim) << endl;

//  for (int m = 0; m < 6; m++)
//    free(PP_CS[m]);
//  PP_CS = (CString**)malloc(sizeof(CString*) * 6);
//  free(PP_Args);
//  PP_Args = (void**)malloc(sizeof(void*) * 6);

  CS_A1_Args(0).Set("BLZ");
  Array<double, 1> D_A1_BLZ(1);
  D_A1_BLZ = 0.;
  PP_Args[0] = &D_A1_BLZ;

  CS_A1_Args(1).Set("OVERSAMPLE");
  PP_Args[1] = &I_OverSample;

  CS_A1_Args(2).Set("LAMBDA_SF");
  PP_Args[2] = &D_LamSF;

  CS_A1_Args(3).Set("LAMBDA_SP");
  PP_Args[3] = &I_LamSP;

//  PP_CS[4] = new CString("MASK");
//  Array<double, 2> D_A2_Mask;
//  D_A2_Mask.resize(D_A2_Im.rows(), D_A2_Im.cols());
//  D_A2_Mask = 0.;
//  PP_Args[4] = &D_A2_Mask;

  CS_A1_Args(4).Set("SWATH_WIDTH");
  PP_Args[4] = &I_SwathWidth;

  CS_A1_Args(5).Set("Y_LOWER_LIM");
  PP_Args[5] = &I_YLowerLim;

  CS_A1_Args(6).Set("Y_UPPER_LIM");
  PP_Args[6] = &I_YUpperLim;

  CS_A1_Args(7).Set("CCD_GAIN");
  PP_Args[7] = &D_Gain;

  CS_A1_Args(8).Set("CCD_READN");
  PP_Args[8] = &D_ReadN;

  Array<double, 2> D_A2_ImOutTemp(P_TestFitsA->GetNRows(), P_TestFitsA->GetNCols());
  CS_A1_Args(9).Set("IM_OUT");
  PP_Args[9] = &D_A2_ImOutTemp;

  Array<double, 1> D_A1_XSlitF(D_A2_Im.rows());
  Array<double, 1> D_A1_BinCen(D_A2_Im.rows());
  Array<double, 2> D_A2_SlitF(D_A2_Im.rows(), D_A2_Im.cols());
  t27 = P_TestFitsA->SetNRows(D_A2_Im_In.rows());
  if (t27)
    cout << "MTestApp::TestCFitsPiskunov: t27(SetNRows) returned \"TRUE\" (EXPECTED)" << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t27(SetNRows) returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }
  t28 = P_TestFitsA->SetNCols(D_A2_Im_In.cols());
  if (t28)
    cout << "MTestApp::TestCFitsPiskunov: t28(SetNCols) returned \"TRUE\" (EXPECTED)" << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t28(SetNCols) returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  (*MTestApp::P_Log) << "MTestApp::TestCFitsPiskunov: after t28: p_testfitsP->GetFileName() = " << p_testfitsP->GetFileName() << endl;

  t29 = P_TestFitsA->SetFileName(*(new CString("test/test_FEROS_401x4090.fits")));
  if (t29)
    cout << "MTestApp::TestCFitsPiskunov: t29(SetFileName) returned \"TRUE\" (EXPECTED)" << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t29(SetFileName) returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  t30 = P_TestFitsA->GetFileName().EqualValue( p_testfitsP->GetFileName());
  if (t30)
  {
    cout << "MTestApp::TestCFitsPiskunov: t30(FileNames->EqualValue) returned \"TRUE\" (UNEXPECTED)" << endl;
    return false;
  }
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t30(FileNames->EqualValue) returned \"FALSE\" (EXPECTED)" << endl;
  }

/*  t35 = P_TestFitsA->WriteArrayToFile(D_A2_Im_In, (*new CString("test/MkSlitFunc_Im_in.arr")));
  if (t35)
    cout << "MTestApp::TestCFitsPiskunov: t35(WriteArrayToFile(D_A2_Im_In, MkSlitFunc_Im_in.arr) returned \"TRUE\" (EXPECTED)" << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t35(WriteArrayToFile(D_A2_Im_In, MkSlitFunc_Im_in.arr) returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  (*MTestApp::P_Log) << "MTestApp::TestCFitsPiskunov: P_TestFitsA::PixArray.size = " << P_TestFitsA->GetNRows() << " x " << P_TestFitsA->GetNCols() << endl;
  (*MTestApp::P_Log) << "MTestApp::TestCFitsPiskunov: D_A2_Im_In.size = " << P_TestFitsA->GetNRows() << " x " << P_TestFitsA->GetNCols() << endl;
*/
  P_TestFitsA->GetPixArray() = D_A2_Im_In;
  (*MTestApp::P_Log) << "MTestApp::TestCFitsPiskunov: P_TestFitsA::PixArray.size = " << P_TestFitsA->GetNRows() << " x " << P_TestFitsA->GetNCols() << endl;

  t31 = P_TestFitsA->ClassInvariant();
  if (t31)
    cout << "MTestApp::TestCFitsPiskunov: t31(P_TestFitsA->ClassInvariant) returned \"TRUE\" (EXPECTED)" << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t31(P_TestFitsA(=" << *P_TestFitsA << ")->ClassInvariant) returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  t32 = P_TestFitsA->WriteArray();
  if (t32)
    cout << "MTestApp::TestCFitsPiskunov: t32(WriteArray) returned \"TRUE\" (EXPECTED)" << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t32(WriteArray) returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

/*  t33 = P_TestFitsA->MkSlitFunc(D_A1_ScatterBelow,  //: in
                                D_A1_YScatterBelow, //: in
                                D_A1_ScatterAbove,  //: in
                                D_A1_YScatterAbove, //: in
                                D_A1_YCenters,      //: in
                                D_XLeftLim,         //: in
                                D_XRightLim,        //: in
                                D_A1_XSlitF,        //: out
                                D_A2_SlitF,         //: out
                                D_A1_BinCen,        //: out
                                0,                  //: in
                                (*const_cast<const Array<CString, 1>*>(&CS_A1_Args)),           //: in
                                PP_Args);

  if (t33)
    cout << "MTestApp::TestCFitsPiskunov: t33(MkSlitFunc) returned \"TRUE\" (EXPECTED)" << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t33(MkSlitFunc) returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  FILE *f_xsf;
  f_xsf = fopen("test/MKSlitFunc_XSlitF_out.dat", "w");
  for (int m = 0; m < D_A1_XSlitF.size(); m++)
    fprintf(f_xsf,"%.9f\n",D_A1_XSlitF(m));
  fclose(f_xsf);

  ofstream OFS_SlitF("test/MKSlitFunc_SlitF_out.dat");
  OFS_SlitF << D_A2_SlitF << endl;

  ofstream OFS_ImOut("test/MKSlitFunc_Im_out.dat");
  OFS_ImOut << D_A2_ImOutTemp << endl;

  FILE *f_binc;
  f_binc = fopen("test/MKSlitFunc_BinCen_out.dat", "w");
  for (int m = 0; m < D_A1_BinCen.size(); m++)
    fprintf(f_binc,"%.9f\n",D_A1_BinCen(m));
  fclose(f_binc);

  t34 = P_TestFitsA->SetFileName(*(new CString("test/MKSlitFunc_SF_out.fits")));
  if (t34)
    cout << "MTestApp::TestCFitsPiskunov: t35(P_TestFitsA->SetFileName()) returned \"TRUE\" (expected) " << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t35(P_TestFitsA->SetFileName()) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }
  t35 = P_TestFitsA->SetNRows(D_A2_SlitF.rows());
  if (t35)
    cout << "MTestApp::TestCFitsPiskunov: t35(SetNRows) returned \"TRUE\" (EXPECTED)" << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t35(SetNRows) returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }
  t36 = P_TestFitsA->SetNCols(D_A2_SlitF.cols());
  if (t36)
    cout << "MTestApp::TestCFitsPiskunov: t36(SetNCols) returned \"TRUE\" (EXPECTED)" << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t36(SetNCols) returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  P_TestFitsA->GetPixArray() = D_A2_SlitF;
  t37 = P_TestFitsA->WriteArray();
  if (t37)
    cout << "MTestApp::TestCFitsPiskunov: t37(WriteArray) returned \"TRUE\" (EXPECTED)" << endl;
  else
  {
    cout << "MTestApp::TestCFitsPiskunov: t37(WriteArray) returned \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }
*/
  return (t1 && t2 && t3 && t4 && t5 && t6 && t7 && t8 && t9 && t10 && t11 && t12 && t13 && t14 && t15 && t16 && t17 && t18 && t19 && t20 && t21 && t22 && t23 && t24 && t25 && t26 && t27 && t28 && t29 && !t30 && t31 && t32);// && t33);// && t34 && t35 && t36 && t37);
}
/************************************************************/

/*************************************************************/

bool TestCFitsMkProf()
{
  // Test: CFits::Methods of Piskunov and Valenti
  // Tests included: CFits::GetSubArrCopy
  // require : nothing
  // ensure  : t1, t2, t3, t4, t5 are "TRUE"
  bool t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t35, t36, t37, t38, t39, t40;
  string tempstr;
  int NCols = 3;
  int NRows = 7;
  int m;
  CString *p_testString = new CString("test/combinedFlat_s.fits");
  CString *p_CS_DatabaseString = new CString("test/database/apcombinedFlat_s");
  CFits *p_testfitsP = new CFits();
  firstIndex i;
  secondIndex j;

  /// MkProfIm
  t1 = p_testfitsP->SetFileName( *p_testString );
  if (t1)
  {
    cout << "MTestApp::TestCFitsMkProf: t1(p_testfitsP->SetFileName()) returned \"TRUE\" (expected) " << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkProf: t1(p_testfitsP->SetFileName()) returned \"TRUE\" (expected) " << endl;
  }
  else
  {
    cout << "MTestApp::TestCFitsMkProf: t1(p_testfitsP->SetFileName()) = \"FALSE\" (UNEXPECTED)" << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkProf: t1(p_testfitsP->SetFileName()) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  t2 = p_testfitsP->Set_Gain(1.03);
  if (t2)
  {
    cout << "MTestApp::TestCFitsMkProf: t2(p_testfitsP->SetGain()) returned \"TRUE\" (expected) " << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkProf: t2(p_testfitsP->SetGain()) returned \"TRUE\" (expected) " << endl;
  }
  else
  {
    cout << "MTestApp::TestCFitsMkProf: t2(p_testfitsP->SetGain()) = \"FALSE\" (UNEXPECTED)" << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkProf: t2(p_testfitsP->SetGain()) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  t3 = p_testfitsP->Set_ReadOutNoise( 3.48 );
  if (t3)
  {
    cout << "MTestApp::TestCFitsMkProf: t3(p_testfitsP->Set_ReadOutNoise()) returned \"TRUE\" (expected) " << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkProf: t3(p_testfitsP->Set_ReadOutNoise()) returned \"TRUE\" (expected) " << endl;
  }
  else
  {
    cout << "MTestApp::TestCFitsMkProf: t3(p_testfitsP->Set_ReadOutNoise()) = \"FALSE\" (UNEXPECTED)" << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkProf: t3(p_testfitsP->Set_ReadOutNoise()) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  t4 = p_testfitsP->Set_OverSample( 10 );
  if (t4)
  {
    cout << "MTestApp::TestCFitsMkProf: t4(p_testfitsP->Set_OverSample()) returned \"TRUE\" (expected) " << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkProf: t4(p_testfitsP->Set_OverSample()) returned \"TRUE\" (expected) " << endl;
  }
  else
  {
    cout << "MTestApp::TestCFitsMkProf: t4(p_testfitsP->Set_OverSample()) = \"FALSE\" (UNEXPECTED)" << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkProf: t4(p_testfitsP->Set_OverSample()) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  t5 = p_testfitsP->SetDatabaseFileName(*p_CS_DatabaseString);
  if (t5)
  {
    cout << "MTestApp::TestCFitsMkProf: t5(p_testfitsP->SetDatabaseFileName()) returned \"TRUE\" (expected) " << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkProf: t5(p_testfitsP->SetDatabaseFileName()) returned \"TRUE\" (expected) " << endl;
  }
  else
  {
    cout << "MTestApp::TestCFitsMkProf: t5(p_testfitsP->SetDatabaseFileName()) = \"FALSE\" (UNEXPECTED)" << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkProf: t5(p_testfitsP->SetDatabaseFileName()) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  t6 = p_testfitsP->ReadArray();
  if (t6)
  {
    cout << "MTestApp::TestCFitsMkProf: t6(p_testfitsP->ReadArray()) returned \"TRUE\" (expected) " << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkProf: t6(p_testfitsP->ReadArray()) returned \"TRUE\" (expected) " << endl;
  }
  else
  {
    cout << "MTestApp::TestCFitsMkProf: t6(p_testfitsP->ReadArray()) = \"FALSE\" (UNEXPECTED)" << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkProf: t6(p_testfitsP->ReadArray()) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  t7 = p_testfitsP->ReadDatabaseEntry();
  if (t7)
  {
    cout << "MTestApp::TestCFitsMkProf: t7(p_testfitsP->ReadDatabaseEntry()) returned \"TRUE\" (expected) " << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkProf: t7(p_testfitsP->ReadDatabaseEntry()) returned \"TRUE\" (expected) " << endl;
  }
  else
  {
    cout << "MTestApp::TestCFitsMkProf: t7(p_testfitsP->ReadDatabaseEntry()) = \"FALSE\" (UNEXPECTED)" << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkProf: t7(p_testfitsP->ReadDatabaseEntry()) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  t8 = p_testfitsP->CalcTraceFunctions();
  if (t8)
  {
    cout << "MTestApp::TestCFitsMkProf: t8(p_testfitsP->CalcTraceFunctions()) returned \"TRUE\" (expected) " << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkProf: t8(p_testfitsP->CalcTraceFunctions()) returned \"TRUE\" (expected) " << endl;
  }
  else
  {
    cout << "MTestApp::TestCFitsMkProf: t8(p_testfitsP->CalcTraceFunctions()) = \"FALSE\" (UNEXPECTED)" << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkProf: t8(p_testfitsP->CalcTraceFunctions()) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  cout << "MTestApp::TestCFitsMkProf: starting t9(p_testfitsP(=" << *p_testfitsP << ")->MkProfIm())" << endl;
  (*MTestApp::P_Log) << "MTestApp::TestCFitsMkProf: starting t1(p_testfitsP(=" << *p_testfitsP << ")->MkProfIm())" << endl;

  t9 = p_testfitsP->MkProfIm();
  if (t9)
  {
    cout << "MTestApp::TestCFitsMkProf: t9(p_testfitsP->MkProfIm()) returned \"TRUE\" (expected) " << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkProf: t9(p_testfitsP->MkProfIm()) returned \"TRUE\" (expected) " << endl;
  }
  else
  {
    cout << "MTestApp::TestCFitsMkProf: t9(p_testfitsP->MkProfIm()) = \"FALSE\" (UNEXPECTED)" << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkProf: t9(p_testfitsP->MkProfIm()) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }
  return (t1 && t2 && t3 && t4 && t5 && t6 && t7 && t8 && t9);
}

/*************************************************************/

bool TestCFitsMkScatter()
{
  // Test: CFits::Methods of Piskunov and Valenti
  // Tests included: CFits::GetSubArrCopy
  // require : nothing
  // ensure  : t1, t2, t3, t4, t5 are "TRUE"
  bool t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t35, t36, t37, t38, t39, t40;
  string tempstr;
  int NCols = 3;
  int NRows = 7;
  int m;
  CString *p_testString = new CString("test/lq_hya5_2006-04-12T00-21-05.674_840s_b_600-900_3-4092sn_transp.fits");///my_last_run.flat_transp6.fits");
  CString *p_CS_DatabaseString = new CString("test/database/apmy_last_run.flat.mkslitf");
  CFits *p_testfitsP = new CFits();
  Array<CString, 1> CS_A1_Args(10);
  CS_A1_Args = CString("");
  void **PP_Args = (void**)malloc(sizeof(void*) * 10);

//  CS_A1_Args(0).Set("GAIN");
//  double gain = 1.25;
//  PP_Args[0] = &gain;

  Array<double,2> A(3,3);
  A(0,0) = 5.;
  A(0,1) = -1.;
  A(0,2) = 3.;
  A(1,0) = 2.;
  A(1,1) = 0.;
  A(1,2) = 1.;
  A(2,0) = 3.;
  A(2,1) = 2.;
  A(2,2) = 1.;
  Array<double,2> B(3,3);
  B = 0.;
  B(0,0) = 1.;
  B(1,1) = 1.;
  B(2,2) = 1.;
  Array<double,2> C(3,3);
  C = A;
  if (!p_testfitsP->InvertGaussJ(A)){
    cout << "MTestApp::TestCFitsMkScatter: ERROR! InvertGaussJ(A,B) returned false!" << endl;
    return false;
  }

  Array<double,2> *D;
  D = p_testfitsP->MatrixBTimesA(A,C);
  for (int i=0; i<3; i++){
    cout << "MTestApp::TestCFitsMkScatter: D(i,*) = " << (*D)(i,0) << ", " << (*D)(i,1) << ", " << (*D)(i,2) << endl;
  }
  if (abs(sum(*D)-3.) > 0.000000001){
    cout << "MTestApp::TestCFitsMkScatter: ERROR! sum(D)(=" << sum(*D) << ") != 3.!" << endl;
    return false;
  }

  CS_A1_Args(0).Set("SWATH_WIDTH");
  int swidth = 300;
  PP_Args[0] = &swidth;

  CS_A1_Args(1).Set("POLYTRACE");
  int poly = 1;
  PP_Args[1] = &poly;

  CS_A1_Args(2).Set("LAMBDA_SF");
  double D_Lambda_SF = 100.;
  PP_Args[2] = &D_Lambda_SF;

  CS_A1_Args(3).Set("LAMBDA_SP");
  double D_Lambda_SP = 100.;
  PP_Args[3] = &D_Lambda_SP;

  CS_A1_Args(4).Set("SUBTRACT");
  PP_Args[4] = &D_Lambda_SP;

  Array<int,1> DD(4);
  DD(0) = 1;
  DD(1) = 2;
  DD(2) = 3;
  DD(3) = 4;
  int len;
  Array<int,1> I_A1_Where(DD.size());
  I_A1_Where = where(DD > 2,1,0);
  cout << "MTestApp::TestCFitsMkScatter: I_A1_Where set to " << I_A1_Where << endl;
  Array<int,1> *P_Ind = p_testfitsP->GetIndex(I_A1_Where,len);
  if (P_Ind->size() == 2)
    cout << "MTestApp::TestCFitsMkScatter: P_Ind(=" << *P_Ind << ").size() == 2 (EXPECTED)" << endl;
  else{
    cout << "MTestApp::TestCFitsMkScatter: P_Ind(=" << *P_Ind << ").size()=" << P_Ind->size() << " != 2 (UNEXPECTED)" << endl;
    return false;
  }

  Array<int,1> I_A1_WherePlus1(DD.size()+100);
  I_A1_WherePlus1 += 1;
  cout << "MTestApp::TestCFitsMkScatter: I_A1_WherePlus1 set to " << I_A1_WherePlus1 << endl;
  I_A1_WherePlus1 = I_A1_Where;
  cout << "MTestApp::TestCFitsMkScatter: I_A1_WherePlus1 set to " << I_A1_WherePlus1 << endl;

  Array<double,2> AA(2,3);
  AA(0,0) = 1.;
  AA(0,1) = 2.;
  AA(0,2) = 1.;
  AA(1,0) = 2.;
  AA(1,1) = -1.;
  AA(1,2) = 2.;

  Array<double,1> BB(3);
  BB(0) =1.;
  BB(1) =2.;
  BB(2) =1.;

  Array<double,1> *CC = p_testfitsP->MatrixTimesVecArr(AA,BB);
  cout << "MTestApp::TestCFitsMkScatter: *CC set to " << *CC << endl;
//  return false;

//  return false;

  /// MkScatter
  t1 = p_testfitsP->SetFileName( *p_testString );
  if (t1)
  {
    cout << "MTestApp::TestCFitsMkScatter: t1(p_testfitsP->SetFileName()) returned \"TRUE\" (expected) " << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkScatter: t1(p_testfitsP->SetFileName()) returned \"TRUE\" (expected) " << endl;
  }
  else
  {
    cout << "MTestApp::TestCFitsMkScatter: t1(p_testfitsP->SetFileName()) = \"FALSE\" (UNEXPECTED)" << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkScatter: t1(p_testfitsP->SetFileName()) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  t2 = p_testfitsP->Set_Gain(1.25);
  if (t2)
  {
    cout << "MTestApp::TestCFitsMkScatter: t2(p_testfitsP->SetGain()) returned \"TRUE\" (expected) " << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkScatter: t2(p_testfitsP->SetGain()) returned \"TRUE\" (expected) " << endl;
  }
  else
  {
    cout << "MTestApp::TestCFitsMkScatter: t2(p_testfitsP->SetGain()) = \"FALSE\" (UNEXPECTED)" << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkScatter: t2(p_testfitsP->SetGain()) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  t3 = p_testfitsP->Set_ReadOutNoise( 7.700 );
  if (t3)
  {
    cout << "MTestApp::TestCFitsMkScatter: t3(p_testfitsP->Set_ReadOutNoise()) returned \"TRUE\" (expected) " << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkScatter: t3(p_testfitsP->Set_ReadOutNoise()) returned \"TRUE\" (expected) " << endl;
  }
  else
  {
    cout << "MTestApp::TestCFitsMkScatter: t3(p_testfitsP->Set_ReadOutNoise()) = \"FALSE\" (UNEXPECTED)" << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkScatter: t3(p_testfitsP->Set_ReadOutNoise()) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  t4 = p_testfitsP->Set_OverSample( 10 );
  if (t4)
  {
    cout << "MTestApp::TestCFitsMkScatter: t4(p_testfitsP->Set_OverSample()) returned \"TRUE\" (expected) " << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkScatter: t4(p_testfitsP->Set_OverSample()) returned \"TRUE\" (expected) " << endl;
  }
  else
  {
    cout << "MTestApp::TestCFitsMkScatter: t4(p_testfitsP->Set_OverSample()) = \"FALSE\" (UNEXPECTED)" << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkScatter: t4(p_testfitsP->Set_OverSample()) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  t5 = p_testfitsP->SetDatabaseFileName(*p_CS_DatabaseString);
  if (t5)
  {
    cout << "MTestApp::TestCFitsMkScatter: t5(p_testfitsP->SetDatabaseFileName()) returned \"TRUE\" (expected) " << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkScatter: t5(p_testfitsP->SetDatabaseFileName()) returned \"TRUE\" (expected) " << endl;
  }
  else
  {
    cout << "MTestApp::TestCFitsMkScatter: t5(p_testfitsP->SetDatabaseFileName()) = \"FALSE\" (UNEXPECTED)" << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkScatter: t5(p_testfitsP->SetDatabaseFileName()) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  t6 = p_testfitsP->ReadArray();
  if (t6)
  {
    cout << "MTestApp::TestCFitsMkScatter: t6(p_testfitsP->ReadArray()) returned \"TRUE\" (expected) " << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkScatter: t6(p_testfitsP->ReadArray()) returned \"TRUE\" (expected) " << endl;
  }
  else
  {
    cout << "MTestApp::TestCFitsMkScatter: t6(p_testfitsP->ReadArray()) = \"FALSE\" (UNEXPECTED)" << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkScatter: t6(p_testfitsP->ReadArray()) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  t7 = p_testfitsP->ReadDatabaseEntry();
  if (t7)
  {
    cout << "MTestApp::TestCFitsMkScatter: t7(p_testfitsP->ReadDatabaseEntry()) returned \"TRUE\" (expected) " << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkScatter: t7(p_testfitsP->ReadDatabaseEntry()) returned \"TRUE\" (expected) " << endl;
  }
  else
  {
    cout << "MTestApp::TestCFitsMkScatter: t7(p_testfitsP->ReadDatabaseEntry()) = \"FALSE\" (UNEXPECTED)" << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkScatter: t7(p_testfitsP->ReadDatabaseEntry()) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  t8 = p_testfitsP->CalcTraceFunctions();
  if (t8)
  {
    cout << "MTestApp::TestCFitsMkScatter: t8(p_testfitsP->CalcTraceFunctions()) returned \"TRUE\" (expected) " << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkScatter: t8(p_testfitsP->CalcTraceFunctions()) returned \"TRUE\" (expected) " << endl;
  }
  else
  {
    cout << "MTestApp::TestCFitsMkScatter: t8(p_testfitsP->CalcTraceFunctions()) = \"FALSE\" (UNEXPECTED)" << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkScatter: t8(p_testfitsP->CalcTraceFunctions()) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  cout << "PP_Args[0] = " << *(int*)PP_Args[0] << endl;
  cout << "PP_Args[1] = " << *(int*)PP_Args[1] << endl;
  cout << "PP_Args[2] = " << *(double*)PP_Args[2] << endl;
  cout << "PP_Args[3] = " << *(double*)PP_Args[3] << endl;

  cout << "MTestApp::TestCFitsMkScatter: starting t9(p_testfitsP(=" << *p_testfitsP << ")->MkScatter())" << endl;
  (*MTestApp::P_Log) << "MTestApp::TestCFitsMkScatter: starting t1(p_testfitsP(=" << *p_testfitsP << ")->MkScatter())" << endl;
  t9 = p_testfitsP->MkScatter((*const_cast<const Array<CString, 1>*>(&CS_A1_Args)),           //: in
                              PP_Args);
  if (t9)
  {
    cout << "MTestApp::TestCFitsMkScatter: t9(p_testfitsP->MkScatter()) returned \"TRUE\" (expected) " << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkScatter: t9(p_testfitsP->MkScatter()) returned \"TRUE\" (expected) " << endl;
  }
  else
  {
    cout << "MTestApp::TestCFitsMkScatter: t9(p_testfitsP->MkScatter()) = \"FALSE\" (UNEXPECTED)" << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkScatter: t9(p_testfitsP->MkScatter()) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  p_testString->Set("test/lq_hya5_2006-04-12T00-21-05.674_840s_b_600-900_3-4092_z_transp_scattered-light-subtracted.fits");
  t10 = p_testfitsP->SetFileName( *p_testString );
  if (t10)
  {
    cout << "MTestApp::TestCFitsMkScatter: t10(p_testfitsP->SetFileName()) returned \"TRUE\" (expected) " << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkScatter: t10(p_testfitsP->SetFileName()) returned \"TRUE\" (expected) " << endl;
  }
  else
  {
    cout << "MTestApp::TestCFitsMkScatter: t10(p_testfitsP->SetFileName()) = \"FALSE\" (UNEXPECTED)" << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkScatter: t10(p_testfitsP->SetFileName()) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }

  cout << "MTestApp::TestCFitsMkScatter: starting t11(p_testfitsP(=" << *p_testfitsP << ")->WriteArray())" << endl;
  (*MTestApp::P_Log) << "MTestApp::TestCFitsMkScatter: starting t11(p_testfitsP(=" << *p_testfitsP << ")->WriteArray())" << endl;
  t11 = p_testfitsP->WriteArray();
  if (t11)
  {
    cout << "MTestApp::TestCFitsMkScatter: t11(p_testfitsP->WriteArray()) returned \"TRUE\" (expected) " << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkScatter: t11(p_testfitsP->WriteArray()) returned \"TRUE\" (expected) " << endl;
  }
  else
  {
    cout << "MTestApp::TestCFitsMkScatter: t11(p_testfitsP->WriteArray()) = \"FALSE\" (UNEXPECTED)" << endl;
    (*MTestApp::P_Log) << "MTestApp::TestCFitsMkScatter: t11(p_testfitsP->WriteArray()) = \"FALSE\" (UNEXPECTED)" << endl;
    return false;
  }
  return (t1 && t2 && t3 && t4 && t5 && t6 && t7 && t8 && t9 && t10 && t11);
}

/** ***********************************************************/

bool TestCFitsTelluric()
{
  // Test: CFits::Methods of Piskunov and Valenti
  // Tests included: CFits::GetSubArrCopy
  // require : nothing
  // ensure  : t1, t2, t3, t4, t5 are "TRUE"
  bool t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t35, t36, t37, t38, t39, t40;
  string tempstr;
  int NCols = 3;
  int NRows = 7;
  int m;
  CString *p_testString = new CString("/home/azuri/entwicklung/stella/ses-pipeline/c/cfits/test/sf_x_sp+bias+sky_with_errors.fits");// /home/azuri/spectra/elaina/eso_archive/blue_390/LMC-X1_b_2000-01-12T01-37-42.000_390_1200s_botzfxs.fits");///my_last_run.flat_transp6.fits");
  CString *p_outString = new CString("/home/azuri/entwicklung/stella/ses-pipeline/c/cfits/test/sf_x_sp+bias+sky_with_errors.fits");///my_last_run.flat_transp6.fits");
  CString *p_CS_DatabaseString = new CString("/home/azuri/entwicklung/stella/ses-pipeline/c/cfits/test/database/apsf_x_sp+bias+sky_with_errors");//spectra/elaina/eso_archive/blue_390/database/apLMC-X1_b_2000-01-12T01-37-42.000_390_1200s_botzfxs_18");
  CString *p_CS_outDatabaseString = new CString("/home/azuri/entwicklung/stella/ses-pipeline/c/cfits/test/database/apsf_x_sp+bias+sky_with_errors");
  CFits *p_testfitsP = new CFits();
  CFits *p_outfitsP = new CFits();
  Array<CString, 1> CS_A1_Args(8);
  CS_A1_Args = CString("");
  void **PP_Args = (void**)malloc(sizeof(void*) * 8);

  Array<double,2> D_A2_CCD(30,9);
  Array<double,2> D_A2_Bias(30,9);
  Array<double,2> D_A2_Error(30,9);
  Array<double,1> D_A1_SF(9);
  Array<double,1> D_A1_SP(30);
  Array<double,1> D_A1_Sky(30);

  Array<double,1> D_A1_SP_Out(5);
  Array<double,1> D_A1_Sky_Out(5);
  Array<double,1> D_A1_STDDEV_Out(5);
  Array<double,1> D_A1_Covariance_Out(5);
  Array<int, 1> I_A1_IntArr(5);
  D_A1_Covariance_Out(0) = 0.5;
  D_A1_Covariance_Out(1) = 1.5;
  D_A1_Covariance_Out(2) = 2.5;
  D_A1_Covariance_Out(3) = 3.5;
  D_A1_Covariance_Out(4) = 4.5;
  I_A1_IntArr(0) = 0;
  I_A1_IntArr(1) = 1;
  I_A1_IntArr(2) = 2;
  I_A1_IntArr(3) = 3;
  I_A1_IntArr(4) = 4;
  Array<double, 1> D_A1_DblArr(5);
  D_A1_DblArr = D_A1_Covariance_Out * I_A1_IntArr;
  cout << "TestCFitsTelluric: D_A1_DblArr set to " << D_A1_DblArr << endl;
//  return false;

//  cout << "pow(5.1) = " << pow(5.1) << endl;
  cout << "TestCFitsTelluric: pow(5.1,2) = " << pow(5.1,2) << endl;
  cout << "TestCFitsTelluric: pow2(5.1) = " << pow2(5.1) << endl;
  cout << "TestCFitsTelluric: SQR(5.1) = " << SQR(5.1) << endl;
  cout << "TestCFitsTelluric: SQR(-5.1) = " << SQR(-5.1) << endl;
//return false;

  D_A2_CCD.resize(5,7);
  D_A2_CCD(0,0) = 0.;
  D_A2_CCD(0,1) = 1.;
  D_A2_CCD(0,2) = 3.;
  D_A2_CCD(0,3) = 6.;
  D_A2_CCD(0,4) = 3.2;
  D_A2_CCD(0,5) = 1.2;
  D_A2_CCD(0,6) = 0.1;

  D_A2_CCD(1,0) = 0.;
  D_A2_CCD(1,1) = 1.1;
  D_A2_CCD(1,2) = 3.;
  D_A2_CCD(1,3) = 5.9;
  D_A2_CCD(1,4) = 3.1;
  D_A2_CCD(1,5) = 1.3;
  D_A2_CCD(1,6) = 0.1;

  D_A2_CCD(2,0) = 0.;
  D_A2_CCD(2,1) = 0.5;
  D_A2_CCD(2,2) = 1.6;
  D_A2_CCD(2,3) = 3.1;
  D_A2_CCD(2,4) = 1.5;
  D_A2_CCD(2,5) = 0.7;
  D_A2_CCD(2,6) = 0.06;

  D_A2_CCD(3,0) = 3.;
  D_A2_CCD(3,1) = 4.1;
  D_A2_CCD(3,2) = 6.;
  D_A2_CCD(3,3) = 9.1;
  D_A2_CCD(3,4) = 6.2;
  D_A2_CCD(3,5) = 4.2;
  D_A2_CCD(3,6) = 3.2;

  D_A2_CCD(4,0) = 2.;
  D_A2_CCD(4,1) = 4.1;
  D_A2_CCD(4,2) = 5.1;
  D_A2_CCD(4,3) = 8.;
  D_A2_CCD(4,4) = 5.;
  D_A2_CCD(4,5) = 3.4;
  D_A2_CCD(4,6) = 2.1;

  Array<double,2> D_A2_SF(5,7);
  D_A2_SF(0,0) = 0.;
  D_A2_SF(0,1) = 1.;
  D_A2_SF(0,2) = 3.;
  D_A2_SF(0,3) = 6.;
  D_A2_SF(0,4) = 3.2;
  D_A2_SF(0,5) = 1.2;
  D_A2_SF(0,6) = 0.1;

  D_A2_SF(1,0) = 0.;
  D_A2_SF(1,1) = 1.;
  D_A2_SF(1,2) = 3.;
  D_A2_SF(1,3) = 6.;
  D_A2_SF(1,4) = 3.2;
  D_A2_SF(1,5) = 1.2;
  D_A2_SF(1,6) = 0.1;

  D_A2_SF(2,0) = 0.;
  D_A2_SF(2,1) = 1.;
  D_A2_SF(2,2) = 3.;
  D_A2_SF(2,3) = 6.;
  D_A2_SF(2,4) = 3.2;
  D_A2_SF(2,5) = 1.2;
  D_A2_SF(2,6) = 0.1;

  D_A2_SF(3,0) = 0.;
  D_A2_SF(3,1) = 1.;
  D_A2_SF(3,2) = 3.;
  D_A2_SF(3,3) = 6.;
  D_A2_SF(3,4) = 3.2;
  D_A2_SF(3,5) = 1.2;
  D_A2_SF(3,6) = 0.1;

  D_A2_SF(4,0) = 0.;
  D_A2_SF(4,1) = 1.;
  D_A2_SF(4,2) = 3.;
  D_A2_SF(4,3) = 6.;
  D_A2_SF(4,4) = 3.2;
  D_A2_SF(4,5) = 1.2;
  D_A2_SF(4,6) = 0.1;

  Array<double, 1> D_A1_TempA(4);
  D_A1_TempA(0) = 4.;
  D_A1_TempA(1) = 3.;
  D_A1_TempA(2) = 2.;
  D_A1_TempA(3) = 1.;

  Array<int, 1> I_A1_TempB(D_A2_SF.size());
  for (int m = 0; m < D_A2_SF.size(); m++)
    I_A1_TempB(m) = m;
  cout << "TestCFitsTelluric: I_A1_TempB set to " << I_A1_TempB << endl;

  cout << "TestCFitsTelluric: D_A2_SF = " << D_A2_SF << endl;
  Array<double, 2> D_A2_TempA(1,1);
  if (p_testfitsP->GetSubArrCopy(D_A2_SF, I_A1_TempB, 2, D_A2_TempA))
    return false;
  cout << "TestCFitsTelluric: D_A2_TempA set to " << D_A2_TempA << endl;
  cout << "TestCFitsTelluric: D_A2_SF = " << D_A2_SF << endl;

  if (p_testfitsP->GetSubArrCopy(D_A2_SF, I_A1_TempB, 3, D_A2_TempA))
    return false;
  cout << "TestCFitsTelluric: D_A2_TempA set to " << D_A2_TempA << endl;
  cout << "TestCFitsTelluric: D_A2_SF = " << D_A2_SF << endl;

  Array<double, 1> *P_D_A1_Temp = p_outfitsP->VecArrTimesMatrix(D_A1_TempA, D_A2_SF);
  cout << "TestCFitsTelluric: VecArrTimesMatrix(D_A1_TempA(=" << D_A1_TempA << "), D_A2_SF) = " << *P_D_A1_Temp << endl;
  ///return false;

  D_A1_SP_Out.resize(5);
  D_A1_Sky_Out.resize(5);
  D_A1_STDDEV_Out.resize(5);
  D_A1_Covariance_Out.resize(5);

  if (!p_testfitsP->LinearRegression(D_A2_CCD,D_A2_SF,D_A1_SP_Out,D_A1_Sky_Out,D_A1_STDDEV_Out,D_A1_Covariance_Out))
    return false;

  cout << "TestCFitsTelluric: D_A2_CCD = " << D_A2_CCD << endl;
  cout << "TestCFitsTelluric: D_A2_SF = " << D_A2_SF << endl;
  cout << "TestCFitsTelluric: D_A1_SP_Out = " << D_A1_SP_Out << endl;
  cout << "TestCFitsTelluric: D_A1_Sky_Out = " << D_A1_Sky_Out << endl;
  cout << "TestCFitsTelluric: D_A1_STDDEV_Out = " << D_A1_STDDEV_Out << endl;
  cout << "TestCFitsTelluric: D_A1_Covariance_Out = " << D_A1_Covariance_Out << endl;




  D_A2_CCD.resize(30,11);
  D_A2_Bias.resize(30,11);
  D_A2_Error.resize(30,11);
  D_A1_SF.resize(11);
  D_A1_SP.resize(30);
  D_A1_Sky.resize(30);

  D_A1_SF(0) = 0.;
  D_A1_SF(1) = 0.;
  D_A1_SF(2) = 1.;
  D_A1_SF(3) = 3.;
  D_A1_SF(4) = 7.;
  D_A1_SF(5) = 4.;
  D_A1_SF(6) = 2.;
  D_A1_SF(7) = 1.;
  D_A1_SF(8) = 0.;
  D_A1_SF(9) = 0.;
  D_A1_SF(10) = 0.;
  D_A1_SF /= sum(D_A1_SF);
  cout << "TestCFitsTelluric: D_A1_SF = " << D_A1_SF << endl;
  ///[         0 0.0555556  0.166667  0.388889  0.222222  0.111111 0.0555556       0         0  ]

  D_A1_SP(0) = 100.;
  D_A1_SP(1) = 1000.;
  D_A1_SP(2) = 10000.;
  D_A1_SP(3) = 30000.;
  D_A1_SP(4) = 75000.;
  D_A1_SP(5) = 100000.;
  D_A1_SP(6) = 30000.;
  D_A1_SP(7) = 30000.;
  D_A1_SP(8) = 30000.;
  D_A1_SP(9) = 30000.;
  D_A1_SP(10) = 30000.;
  D_A1_SP(11) = 30000.;
  D_A1_SP(12) = 30000.;
  D_A1_SP(13) = 30000.;
  D_A1_SP(14) = 30000.;
  D_A1_SP(15) = 30000.;
  D_A1_SP(16) = 30000.;
  D_A1_SP(17) = 30000.;
  D_A1_SP(18) = 30000.;
  D_A1_SP(19) = 30000.;
  D_A1_SP(20) = 30000.;
  D_A1_SP(21) = 30000.;
  D_A1_SP(22) = 30000.;
  D_A1_SP(23) = 30000.;
  D_A1_SP(24) = 30000.;
  D_A1_SP(25) = 30000.;
  D_A1_SP(26) = 30000.;
  D_A1_SP(27) = 30000.;
  D_A1_SP(28) = 30000.;
  D_A1_SP(29) = 30000.;
  cout << "TestCFitsTelluric: D_A1_SP = " << D_A1_SP << endl;

  D_A1_Sky(0) = 300.;
  D_A1_Sky(1) = 3000.;
  D_A1_Sky(2) = 0.;
  D_A1_Sky(3) = 0.;
  D_A1_Sky(4) = 0.;
  D_A1_Sky(5) = 30000.;
  D_A1_Sky(6) = 1000.;
  D_A1_Sky(7) = 0.;
  D_A1_Sky(8) = 0.;
  D_A1_Sky(9) = 100000.;
  D_A1_Sky(10) = 0.;
  D_A1_Sky(11) = 0.;
  D_A1_Sky(12) = 30000.;
  D_A1_Sky(13) = 0.;
  D_A1_Sky(14) = 100.;
  D_A1_Sky(15) = 200.;
  D_A1_Sky(16) = 0.;
  D_A1_Sky(17) = 0.;
  D_A1_Sky(18) = 20000.;
  D_A1_Sky(19) = 28000.;
  D_A1_Sky(20) = 30000.;
  D_A1_Sky(21) = 25000.;
  D_A1_Sky(22) = 28000.;
  D_A1_Sky(23) = 32000.;
  D_A1_Sky(24) = 30000.;
  D_A1_Sky(25) = 28000.;
  D_A1_Sky(26) = 5000.;
  D_A1_Sky(27) = 0.;
  D_A1_Sky(28) = 0.;
  D_A1_Sky(29) = 0.;
  cout << "TestCFitsTelluric: D_A1_Sky = " << D_A1_Sky << endl;

  Array<double,2> *P_TempArr = p_testfitsP->VecArrACrossB(D_A1_SP,D_A1_SF);
  D_A2_CCD.resize(P_TempArr->rows(),P_TempArr->cols());
  D_A2_CCD = (*P_TempArr);
  D_A2_CCD(8,3) = 200000.;
  D_A2_CCD(10,5) = 200000.;
  delete(P_TempArr);
  cout << "TestCFitsTelluric: D_A2_CCD = " << D_A2_CCD << endl;

  for (int m=0;m < D_A2_CCD.rows(); m++){
    for (int n=0;n < D_A2_CCD.cols(); n++){
      Normal<double> normalGen(D_A2_CCD(m,n),sqrt(D_A2_CCD(m,n)));
      Normal<double> normalGenA(D_A1_Sky(m),sqrt(D_A1_Sky(m)));
      double normalCCD = normalGen.random();
      double normalSky = normalGenA.random();
      cout << "TestCFitsTelluric: m=" << m << ", n = " << n << ": normalCCD = " << normalCCD << ", normalSky = " << normalSky << endl;
///      D_A2_CCD(m,n) = normalCCD + normalSky;
    }
  }
  cout << "TestCFitsTelluric: D_A2_CCD = " << D_A2_CCD << endl;

  double Gain = 0.49;
  double RdNoise = 3.9;

  double bias_mean = 194.4;
  double bias_stddev = 2.2;
  double d_temp;

  D_A2_Bias = bias_mean;

  D_A2_Error = 0.;

  Normal<double> normalGen(bias_mean,bias_stddev);
  for (int m=0;m < D_A2_Bias.rows(); m++){
    for (int n=0;n < D_A2_Bias.cols(); n++){
      d_temp = normalGen.random();
      //cout << "m = " << m << ", n = " << n << ": d_temp = " << d_temp << endl;
      D_A2_Bias(m,n) = d_temp;
    }
  }
  cout << "TestCFitsTelluric: normalGen.random = " << normalGen.random() << endl;
  cout << "TestCFitsTelluric: D_A2_Bias = " << D_A2_Bias << endl;

///  D_A2_CCD += D_A2_Bias - mean(D_A2_Bias);
  cout << "TestCFitsTelluric: D_A2_CCD += Bias = " << D_A2_CCD << endl;

  D_A2_Error = sqrt(abs(D_A2_CCD));
  D_A2_Error += bias_stddev;
  cout << "TestCFitsTelluric: D_A2_Error = " << D_A2_Error << endl;
  CString CS_ErrorFileName = CString("/home/azuri/entwicklung/stella/ses-pipeline/c/cfits/test/sf_x_sp+bias+sky_with_errors_err.fits");

  if (!p_outfitsP->SetNRows(D_A2_CCD.rows()))
    return false;
  if (!p_outfitsP->SetNCols(D_A2_CCD.cols()))
    return false;

  ///Write Error fits file
  if (!p_outfitsP->SetFileName( CS_ErrorFileName ))
    return false;
  p_outfitsP->GetPixArray() = D_A2_Error;
  if (!p_outfitsP->WriteArray())
    return false;

  ///Write CCD fits file minus sky
  if (!p_outfitsP->SetFileName( CString("/home/azuri/entwicklung/stella/ses-pipeline/c/cfits/test/sf_x_sp+bias+sky_with_errors_minus_sky.fits") ))
    return false;
  Array<double, 1> *p_rep = p_outfitsP->Replicate(1., D_A2_CCD.cols());
  Array<double, 2> *p_skyxrep = p_outfitsP->VecArrACrossB(D_A1_Sky, *p_rep);
  p_outfitsP->GetPixArray() = D_A2_CCD - (*p_skyxrep);
  if (!p_outfitsP->WriteArray())
    return false;


  ///Write CCD fits file
  if (!p_outfitsP->SetFileName( *p_outString ))
    return false;
  p_outfitsP->GetPixArray() = D_A2_CCD;
  if (!p_outfitsP->WriteArray())
    return false;

  if (!p_outfitsP->SetErrFileName(CS_ErrorFileName))
    return false;

  if (!p_outfitsP->ReadErrArray())
    return false;


  /// Calculate profile and extract spectra
  if (!p_outfitsP->Set_Gain(1.))
    return false;

  if (!p_outfitsP->Set_ReadOutNoise( 0. ))
    return false;

  if (!p_outfitsP->Set_OverSample( 10 ))
    return false;

  if (!p_outfitsP->SetDatabaseFileName(*p_CS_outDatabaseString))
    return false;

  if (!p_outfitsP->ReadDatabaseEntry())
    return false;
  cout << "TestCFitsTelluric: p_outfitsP->Get_XMin() = " << p_outfitsP->Get_XMin() << endl;

  if (!p_outfitsP->CalcTraceFunctions())
    return false;
  cout << "TestCFitsTelluric: p_outfitsP->Get_XMin() = " << p_outfitsP->Get_XMin() << endl;

//  return false;

  CS_A1_Args(0).Set("SWATH_WIDTH");
  int swidth = 300;
  PP_Args[0] = &swidth;

  CS_A1_Args(1).Set(" ");//"TELLURIC");
  int I_Telluric = 2;
  PP_Args[1] = &I_Telluric;

  CS_A1_Args(2).Set("STOP");
  int I_Stop = 1;
  PP_Args[2] = &I_Stop;

  CS_A1_Args(3).Set(" ");
  CS_A1_Args(4).Set(" ");

  cout << "TestCFitsTelluric: D_A2_CCD = " << D_A2_CCD << endl;
//  return false;

  if (!p_outfitsP->MkProfIm(CS_A1_Args, PP_Args))
    return false;

  p_outString->InsertAt(CString("_out"),p_outString->GetLength()-5);
  p_outfitsP->SetFileName(*p_outString);
  p_outfitsP->WriteArray();

  cout << "TestCFitsTelluric: MkProfIm ready" << endl;

//  return false;
  /// create sf_x_sp+bias+sky_with_errors_norm.fits
  Array<double,1> D_A1_Sum(D_A2_CCD.rows());
  for (int oo=0; oo < D_A2_CCD.rows(); oo++){
//    cout << "D_A2_CCD(oo=" << oo << ",Range::all()) = " << D_A2_CCD(oo,Range::all()) << endl;
    D_A1_Sum(oo) = sum(D_A2_CCD(oo,Range::all()));
    D_A2_CCD(oo,Range::all()) /= D_A1_Sum(oo);
    cout << "TestCFitsTelluric: D_A1_Sum(oo=" << oo << ") set to " << D_A1_Sum(oo) << endl;
//    cout << "D_A2_CCD(oo=" << oo << ",Range::all()) set to " << D_A2_CCD(oo,Range::all()) << endl;
  }
  if (!p_outfitsP->SetFileName( CString("/home/azuri/entwicklung/stella/ses-pipeline/c/cfits/test/sf_x_sp+bias+sky_with_errors_norm.fits") ))
    return false;

  p_outfitsP->GetPixArray() = D_A2_CCD;
  if (!p_outfitsP->WriteArray())
    return false;

  /// recalc original D_A2_CCD
  for (int oo=0; oo < D_A2_CCD.rows(); oo++){
    D_A2_CCD(oo,Range::all()) *= D_A1_Sum(oo);
  }

  /// create profile image
  if (!p_outfitsP->SetFileName( CString("/home/azuri/entwicklung/stella/ses-pipeline/c/cfits/test/sf_x_sp+bias+sky_with_errors_profile_telluric_mine.fits") ))
    return false;

  p_outfitsP->GetPixArray() = p_outfitsP->GetProfArray();
  if (!p_outfitsP->WriteArray())
    return false;

  /// create reconstructed object image
  if (!p_outfitsP->SetFileName( CString("/home/azuri/entwicklung/stella/ses-pipeline/c/cfits/test/sf_x_sp+bias+sky_with_errors_RecObj_telluric_mine.fits") ))
    return false;

  p_outfitsP->GetPixArray() = p_outfitsP->GetRecArray();
  if (!p_outfitsP->WriteArray())
    return false;


  /// create reconstructed sky image
  if (!p_outfitsP->SetFileName( CString("/home/azuri/entwicklung/stella/ses-pipeline/c/cfits/test/sf_x_sp+bias+sky_with_errors_RecSky_telluric_mine.fits") ))
    return false;

  p_outfitsP->GetPixArray() = p_outfitsP->GetRecSkyArray();
  if (!p_outfitsP->WriteArray())
    return false;

  p_outfitsP->GetPixArray() = D_A2_CCD;

  Array<double,1> D_A1_SF_calc(D_A1_SF.size());
  Array<double,1> D_A1_SP_calc(D_A1_SP.size());

  firstIndex i;
  secondIndex j;

  cout << "TestCFitsTelluric: D_A2_CCD = " << D_A2_CCD << endl;

  /// calculate SF and SP from summing over different dimensions of D_A2_CCD and takes Median(width=5)
  D_A1_SF_calc = sum(D_A2_CCD(j,i),j);
  cout << "TestCFitsTelluric: sum: D_A1_SF_calc set to " << D_A1_SF_calc << endl;

  Array<double,1> *p_a1_temp = p_outfitsP->MedianVec(D_A1_SF_calc,5);
  D_A1_SF_calc = (*p_a1_temp);
  cout << "TestCFitsTelluric: Median: D_A1_SF_calc set to " << D_A1_SF_calc << endl;
  delete(p_a1_temp);

//  cout << "D_A2_CCD = " << D_A2_CCD << endl;

  cout << "TestCFitsTelluric: D_A1_SP_calc set to " << D_A1_SP_calc << endl;
  D_A1_SP_calc = sum(D_A2_CCD,j);
//  cout << "sum(D_A2_CCD,j) = " << sum(D_A2_CCD,j) << endl;
  cout << "TestCFitsTelluric: D_A1_SP_calc set to " << D_A1_SP_calc << endl;

  p_a1_temp = p_outfitsP->MedianVec(D_A1_SP_calc,5);
  D_A1_SP_calc = (*p_a1_temp);
  cout << "TestCFitsTelluric: Median: D_A1_SP_calc set to " << D_A1_SP_calc << endl;
  delete(p_a1_temp);


  /// create sf_x_sp+bias+sky_with_errors_Ec_telluric_mine.fits

  cout << "TestCFitsTelluric: p_outfitsP->GetSpec() = " << p_outfitsP->GetSpec() << endl;

//  return false;

  CFits F_OutImage;
  CString CS_FitsFileName_Out("/home/azuri/entwicklung/stella/ses-pipeline/c/cfits/test/sf_x_sp+bias+sky_with_errors_Ec_telluric_mine.fits");//spectra/elaina/eso_archive/blue_390/LMC-X1_b_2000-01-12T01-37-42.000_390_1200s_botzfxs_Ec_telluric_mine.fits");
  if (!F_OutImage.SetFileName(CS_FitsFileName_Out))
    return false;


  if (!F_OutImage.SetNCols(p_outfitsP->GetNRows()))
  {
    cout << "TestCFitsTelluric: ERROR: p_outfitsP->SetNCols(" << p_outfitsP->GetNRows() << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  if (!F_OutImage.SetNRows(p_outfitsP->Get_NApertures()))
  {
    cout << "TestCFitsTelluric: ERROR: F_OutImage.SetNRows(" << p_outfitsP->Get_NApertures() << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  F_OutImage.GetPixArray() = p_outfitsP->GetSpec();

  //cout << "MExctract: F_Image.GetSpec = " << F_Image.GetSpec() << endl;

  /// Write spectrum Image
/*  if (!F_OutImage.SetFileName(CS_FitsFileName_Out))
  {
  cout << "MExtract::main: ERROR: F_OutImage.SetFileName(" << CS_FitsFileName_Out << ") returned FALSE!" << endl;
  exit(EXIT_FAILURE);
}*/
  cout << "TestCFitsTelluric: Starting F_OutImage.WriteArray()" << endl;
  if (!F_OutImage.WriteArray())
  {
    cout << "TestCFitsTelluric: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  /// creating sf_x_sp+bias+sky_with_errors_err_Ec_piskunov.fits

  F_OutImage.GetPixArray() = sqrt(F_OutImage.GetPixArray());

  CString CS_ErrFileName_Out("/home/azuri/entwicklung/stella/ses-pipeline/c/cfits/test/sf_x_sp+bias+sky_with_errors_err_Ec_telluric_mine.fits");//telluric_mine.fits");
  cout << "TestCFitsTelluric: Starting F_OutImage.SetErrFileName(" << CS_ErrFileName_Out << ")" << endl;
  if (!F_OutImage.SetFileName(CS_ErrFileName_Out))
  {
    cout << "TestCFitsTelluric: ERROR: F_OutImage.SetFileName(" << CS_ErrFileName_Out << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  cout << "TestCFitsTelluric: Starting F_OutImage.WriteArray()" << endl;
  if (!F_OutImage.WriteArray())
  {
    cout << "TestCFitsTelluric: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  cout << "TestCFitsTelluric: p_outfitsP->GetSky() returns " << p_outfitsP->GetSky() << endl;
//  return false;
  F_OutImage.GetPixArray() = p_outfitsP->GetSky();

  CString CS_SkyFileName_Out("/home/azuri/entwicklung/stella/ses-pipeline/c/cfits/test/sf_x_sp+bias+sky_with_errors_EcSky_telluric_mine.fits");//telluric_mine.fits");
  cout << "TestCFitsTelluric: Starting F_OutImage.SetFileName(" << CS_SkyFileName_Out << ")" << endl;
  if (!F_OutImage.SetFileName(CS_SkyFileName_Out))
  {
    cout << "TestCFitsTelluric: ERROR: F_OutImage.SetFileName(" << CS_SkyFileName_Out << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

  cout << "TestCFitsTelluric: Starting F_OutImage.WriteArray()" << endl;
  if (!F_OutImage.WriteArray())
  {
    cout << "TestCFitsTelluric: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }

/**  if (!p_outfitsP->ExtractErrors())
    return false;

  CS_ErrFileName_Out.Set("/home/azuri/entwicklung/stella/ses-pipeline/c/cfits/test/sf_x_sp+bias+sky_with_errors_err_Ec_telluric_mine.fits");//telluric_mine.fits");
  cout << "MExtract::main: Starting F_OutImage.SetErrFileName(" << CS_ErrFileName_Out << ")" << endl;
  if (!F_OutImage.SetFileName(CS_ErrFileName_Out))
  {
    cout << "MExtract::main: ERROR: F_OutImage.SetFileName(" << CS_ErrFileName_Out << ") returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }


  F_OutImage.GetPixArray() = p_outfitsP->GetErrorExt();

  /// Write Image containing the extracted errors
  cout << "MExtract::main: Starting F_OutImage.WriteArray()" << endl;
  if (!F_OutImage.WriteArray())
  {
    cout << "MExtract::main: ERROR: F_OutImage.WriteArray() returned FALSE!" << endl;
    exit(EXIT_FAILURE);
  }


  /// Write Image containing the errors x profile
  Array<double,2> D_A2_CCDOut(p_outfitsP->GetNRows(), p_outfitsP->GetNCols());
  D_A2_CCDOut = p_outfitsP->GetPixArray();
  Array<double,2> D_A2_ErrorOut(p_outfitsP->GetNRows(),p_outfitsP->GetNCols());
  D_A2_ErrorOut = p_outfitsP->GetErrArray();
  Array<double,2> D_A2_Profile(p_outfitsP->GetNRows(),p_outfitsP->GetNCols());
  D_A2_Profile = p_outfitsP->GetProfArray();
  p_outfitsP->GetPixArray() = D_A2_ErrorOut * D_A2_Profile;

  CS_ErrFileName_Out.Set("/home/azuri/entwicklung/stella/ses-pipeline/c/cfits/test/sf_x_sp+bias+sky_with_errors_err_x_prof_telluric_mine.fits");//telluric_mine.fits");
  p_outfitsP->SetFileName(CS_ErrFileName_Out);
  p_outfitsP->WriteArray();

  /// Write image * profile to fits file
  D_A2_CCDOut *= D_A2_Profile;
  p_outfitsP->GetPixArray() = D_A2_CCDOut;
  CS_ErrFileName_Out.Set("/home/azuri/entwicklung/stella/ses-pipeline/c/cfits/test/sf_x_sp+bias+sky_with_errors_im_x_prof_telluric_mine.fits");//telluric_mine.fits");
  p_outfitsP->SetFileName(CS_ErrFileName_Out);
  p_outfitsP->WriteArray();
**/
  return (t1 && t2 && t3 && t4 && t5 && t6 && t7 && t8 && t9 && t10 && t11);
}

/** ***********************************************************/

bool TestCFitsCrossCorrelate()
{
  // Test: CFits::CrossCorrelate
  // Tests included: CFits::GetSubArrCopy
  // require : nothing
  // ensure  : t1, t2, t3, t4, t5 are "TRUE"
  bool t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t35, t36, t37, t38, t39, t40;

  CFits *P_MyCFits = new CFits();
  CString CS_FitsFileName("/home/azuri/spectra/WiFes/Red/NeArarc_2009-06-30T08-44-17.5_14s_r11.fits");
  CString CS_DatabaseFileName("/home/azuri/spectra/WiFes/Red/database/apQtzflat_2009-06-30T19-54-11.5_4s_r11");
  P_MyCFits->SetFileName(CS_FitsFileName);
  P_MyCFits->SetDatabaseFileName(CS_DatabaseFileName);
  P_MyCFits->ReadArray();
  P_MyCFits->ReadDatabaseEntry();
  P_MyCFits->CalcTraceFunctions();
  int I_NOrders = P_MyCFits->Get_NApertures();
  Array<double, 2> *P_DA2_Centers = P_MyCFits->Get_XCenters();
  Array<double, 1> *P_DA1_Low = P_MyCFits->Get_XLow();
  Array<double, 1> *P_DA1_High = P_MyCFits->Get_XHigh();

//  P_MyCFits->SetFileName(
  Array<double, 1> DA1_Static(20);
  Array<double, 1> DA1_Moving(20);
  DA1_Static(0) = 0.;
  DA1_Static(1) = 1.;
  DA1_Static(2) = 2.;
  DA1_Static(3) = 3.;
  DA1_Static(4) = 4.;
  DA1_Static(5) = 5.;
  DA1_Static(6) = 6.;
  DA1_Static(7) = 7.;
  DA1_Static(8) = 8.;
  DA1_Static(9) = 9.;
  DA1_Static(10) = 8.;
  DA1_Static(11) = 7.;
  DA1_Static(12) = 6.;
  DA1_Static(13) = 5.;
  DA1_Static(14) = 4.;
  DA1_Static(15) = 3.;
  DA1_Static(16) = 2.;
  DA1_Static(17) = 1.;
  DA1_Static(18) = 0.;
  DA1_Static(19) = 0.;
  DA1_Moving(0) = 0.;
  DA1_Moving(1) = 0.;
  DA1_Moving(2) = 0.;
  DA1_Moving(3) = 0.;
  DA1_Moving(4) = 1.;
  DA1_Moving(5) = 2.;
  DA1_Moving(6) = 3.;
  DA1_Moving(7) = 4.;
  DA1_Moving(8) = 5.;
  DA1_Moving(9) = 6.;
  DA1_Moving(10) = 7.;
  DA1_Moving(11) = 8.;
  DA1_Moving(12) = 9.;
  DA1_Moving(13) = 8.;
  DA1_Moving(14) = 7.;
  DA1_Moving(15) = 6.;
  DA1_Moving(16) = 5.;
  DA1_Moving(17) = 4.;
  DA1_Moving(18) = 3.;
  DA1_Moving(19) = 2.;
  int I_Result;
  P_MyCFits->CrossCorrelate(DA1_Static, DA1_Moving, 10, 10, I_Result);
  cout << "TestCFitsCrossCorrelate: I_Result = " << I_Result << endl;
  if (I_Result == -3)
    t1 = true;
  else
    t1 = false;

//  Array<int, 1> IA1_Result(1);
//  if (P_MyCFits->CrossCorrelateAllApertureColsToColNo(2070, 200, 5, IA1_Result))
  t2 = true;
//  else
//    t2 = false;
//  cout << "TestCFitsCrossCorrelate: IA1_Result = " << IA1_Result << endl;

//  CString CS_CenterFile("/home/azuri/spectra/WiFes/Red/Centers_ap1.dat");
//  P_MyCFits->WriteCenters(0,CS_CenterFile);
//  return false;

  Array<double, 1> D_A1_Result(1);
  Array<int, 1> I_A1_X(1);
  Array<int, 2> I_A2_ColsMinMax(2,2);
  t3 = P_MyCFits->CalcFeatureOffsets(2070, 200, 5, 2, I_A1_X, I_A2_ColsMinMax, D_A1_Result, CString("Poly"));
  cout << "I_A2_ColsMinMax = " << I_A2_ColsMinMax << endl;

/*  if (P_MyCFits->CalcFeatureOffsets(2070, 200, 5, 3, I_A1_X, D_A1_Result, CString("Poly")))
    t4 = true;
  else
    t4 = false;

  if (P_MyCFits->CalcFeatureOffsets(2070, 200, 5, 4, I_A1_X, D_A1_Result, CString("Poly")))
    t5 = true;
  else
    t5 = false;

  if (P_MyCFits->CalcFeatureOffsets(2070, 200, 5, 5, I_A1_X, D_A1_Result, CString("Poly")))
    t6 = true;
  else
    t6 = false;

  if (P_MyCFits->CalcFeatureOffsets(2070, 200, 5, 6, I_A1_X, D_A1_Result, CString("Poly")))
    t7 = true;
  else
    t7 = false;
*/
  CFits CF_Temp(*P_MyCFits);
  CF_Temp.ForceCopy(*P_MyCFits);

  Array<double, 2> D_A2_Temp = CF_Temp.GetPixArray();
  Array<CString, 1> CS_A1_Temp(1);
  CS_A1_Temp(0) = CString(" ");
  t4 = CF_Temp.ShiftColumns(I_A1_X, D_A1_Result, CS_A1_Temp);
  CF_Temp.SetFileName(CString("/home/azuri/spectra/WiFes/Red/NeArarc_2009-06-30T08-44-17.5_14s_r11_shifted_lin.fits"));
  CF_Temp.WriteArray();

/*  CF_Temp.ForceCopy(*P_MyCFits);
  CS_A1_Temp(0) = CString("QUADRATIC");
  t4 = CF_Temp.ShiftColumns(I_A1_X, D_A1_Result, CS_A1_Temp);
  CF_Temp.SetFileName(CString("/home/azuri/spectra/WiFes/Red/NeArarc_2009-06-30T08-44-17.5_14s_r_shifted_lin_quad.fits"));
  CF_Temp.WriteArray();

  CF_Temp.ForceCopy(*P_MyCFits);
  CS_A1_Temp(0) = CString("LSQUADRATIC");
  t4 = CF_Temp.ShiftColumns(I_A1_X, D_A1_Result, CS_A1_Temp);
  CF_Temp.SetFileName(CString("/home/azuri/spectra/WiFes/Red/NeArarc_2009-06-30T08-44-17.5_14s_r_shifted_lin_lsquad.fits"));
  CF_Temp.WriteArray();

  CF_Temp.ForceCopy(*P_MyCFits);
  CS_A1_Temp(0) = CString("SPLINE");
  t4 = CF_Temp.ShiftColumns(I_A1_X, D_A1_Result, CS_A1_Temp);
  CF_Temp.SetFileName(CString("/home/azuri/spectra/WiFes/Red/NeArarc_2009-06-30T08-44-17.5_14s_r_shifted_lin_spline.fits"));
  CF_Temp.WriteArray();
*/
  Array<int, 1> I_A1_Where(I_A1_X.size());
  Array<int, 1> I_A1_Indices(1);
  double D_MinShift, D_MaxShift, D_Shift, D_ShiftOffset;
  int I_NInd, I_MinInd, I_MaxInd;
  Array<double, 1> D_A1_Temp(2);
  for (int i=0; i<P_MyCFits->Get_NApertures(); i++){
    cout << "i = " << i << ": I_A1_X = " << I_A1_X << endl;
    cout << "i = " << i << ": I_A2_ColsMinMax(i,0) = " << I_A2_ColsMinMax(i,0) << endl;
    cout << "i = " << i << ": I_A2_ColsMinMax(i,1) = " << I_A2_ColsMinMax(i,1) << endl;

    I_A1_Where = where(I_A1_X == I_A2_ColsMinMax(i,0),1,0);
    cout << "i = " << i << ": I_A1_Where = " << I_A1_Where << endl;
    P_MyCFits->GetIndex(I_A1_Where, I_NInd, I_A1_Indices);
    cout << "i = " << i << ": I_NInd = " << I_NInd << ", I_A1_Indices = " << I_A1_Indices << endl;
    I_MinInd = I_A1_Indices(0);
    cout << "i = " << i << ": I_MinInd = " << I_MinInd << endl;
    D_MinShift = D_A1_Result(I_MinInd);
    cout << "i =" << i << ": D_MinShift = " << D_MinShift << endl;

    I_A1_Where = where(I_A1_X == I_A2_ColsMinMax(i,1),1,0);
    P_MyCFits->GetIndex(I_A1_Where, I_NInd, I_A1_Indices);
    cout << "i = " << i << ": I_NInd = " << I_NInd << ", I_A1_Indices = " << I_A1_Indices << endl;
    I_MaxInd = I_A1_Indices(0);
    cout << "i = " << i << ": I_MaxInd = " << I_MaxInd << endl;
    D_MaxShift = D_A1_Result(I_MaxInd);
    cout << "i =" << i << ": D_MaxShift = " << D_MaxShift << endl;

    if (D_MinShift > D_MaxShift)
      P_MyCFits->Swap(D_MinShift, D_MaxShift);

/**    if (P_MyCFits->Signum(D_MinShift) == P_MyCFits->Signum(D_MaxShift)){
      D_A1_Temp(0) = D_MinShift;
      D_A1_Temp(1) = D_MaxShift;
      D_Shift = min(fabs(D_A1_Temp));

      if (abs(D_Shift - abs(D_MinShift)) < 0.000000001){/// min(D_MinShift, D_MaxShift) = D_MinShift
        if (D_MinShift < 0.)
          D_Shift = 0. - D_Shift;
      }
      else if (abs(D_Shift - abs(D_MaxShift)) < 0.000000001){/// min(D_MinShift, D_MaxShift) = D_MaxShift
        if (D_MaxShift < 0.)
          D_Shift = 0. - D_Shift;
      }
      cout << "i=" << i << ": D_Shift = " << D_Shift << endl;

    } else{
      D_Shift = 0.;
    }
    D_ShiftOffset = (D_MaxShift - D_MinShift) / 2.;
    if (D_MaxShift > 0.){

    }**/

    D_Shift = ((D_MaxShift - D_MinShift) / 2.);// + D_MinShift;
    cout << "i = " << i << ": D_Shift = " << D_Shift << endl;

    D_A1_Result(Range(I_MinInd, I_MaxInd)) = D_A1_Result(Range(I_MinInd, I_MaxInd))
                                             - D_MinShift
                                             - D_Shift;
    cout << "i = " << i << ": D_A1_Result(Range(I_MinInd=" << I_MinInd << ", I_MaxInd=" << I_MaxInd << ") = " << D_A1_Result(Range(I_MinInd, I_MaxInd)) << endl;
  }

  t5 = P_MyCFits->ShiftColumns(I_A1_X, D_A1_Result, CS_A1_Temp);
  P_MyCFits->SetFileName(CString("/home/azuri/spectra/WiFes/Red/NeArarc_2009-06-30T08-44-17.5_14s_r11_shifted.fits"));
  P_MyCFits->WriteArray();

  t6 = P_MyCFits->CalculateFeatureOffsetAndShiftAllImages(CString("/home/azuri/spectra/WiFes/Red/NeArarc_2009-06-30T08-44-17.5_14s_r11.fits"), CString("/home/azuri/spectra/WiFes/Red/database/apQtzflat_2009-06-30T19-54-11.5_4s_r11"), CString("/home/azuri/spectra/WiFes/Red/to_shift.list"), CString("/home/azuri/spectra/WiFes/Red/straightened.list"), 55, 5, 5, 2, CString("Poly"));

  delete(P_DA2_Centers);
  delete(P_DA1_Low);
  delete(P_DA1_High);

  return (t1 && t2 && t3 && t4 && t5);
}

/** ***********************************************************/

bool TestCFitsFindAndTrace()
{
  // Test: CFits::CrossCorrelate
  // Tests included: CFits::GetSubArrCopy
  // require : nothing
  // ensure  : t1, t2, t3, t4, t5 are "TRUE"
  bool t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t35, t36, t37, t38, t39, t40;

  CFits *P_MyCFits = new CFits();
///  CString CS_FitsFileName("/home/azuri/spectra/WiFes/Red/NeArarc_2009-06-30T08-44-17.5_14s_r11.fits");
///  CString CS_DatabaseFileName("/home/azuri/spectra/WiFes/Red/database/apQtzflat_2009-06-30T19-54-11.5_4s_r11");
///  P_MyCFits->SetFileName(CS_FitsFileName);
///  P_MyCFits->SetDatabaseFileName(CS_DatabaseFileName);
///  P_MyCFits->ReadArray();
///  P_MyCFits->ReadDatabaseEntry();
///  P_MyCFits->CalcTraceFunctions();
///  int I_NOrders = P_MyCFits->Get_NApertures();
///  Array<double, 2> DA2_Centers(I_NOrders, P_MyCFits->GetNRows());
///  DA2_Centers = P_MyCFits->Get_XCenters();
///  Array<double, 1> DA1_Low(I_NOrders);
///  Array<double, 1> DA1_High(I_NOrders);
///  DA1_Low = P_MyCFits->Get_XLow();
///  DA1_High = P_MyCFits->Get_XHigh();

  Array<double, 1> D_A1_Test(20);
  D_A1_Test(0) = 0.;
  D_A1_Test(1) = 1.;
  D_A1_Test(2) = 2.;
  D_A1_Test(3) = 3.;
  D_A1_Test(4) = 4.;
  D_A1_Test(5) = 3.;
  D_A1_Test(6) = 2.;
  D_A1_Test(7) = 1.;
  D_A1_Test(8) = 0.;
  D_A1_Test(9) = 0.;
  D_A1_Test(10) = 1.;
  D_A1_Test(11) = 2.;
  D_A1_Test(12) = 3.;
  D_A1_Test(13) = 4.;
  D_A1_Test(14) = 5.;
  D_A1_Test(15) = 3.;
  D_A1_Test(16) = 2.;
  D_A1_Test(17) = 1.;
  D_A1_Test(18) = 0.;
  D_A1_Test(19) = 1.;

  double D_Threshold = 1.5;
  double D_FWHM = 1.3;

  Array<int, 1> I_A1_Test(D_A1_Test.size());
  I_A1_Test = where(D_A1_Test < D_Threshold, 0, 1);

  Array<int, 1> I_A1_TestCopy(I_A1_Test.size());
  I_A1_TestCopy = I_A1_Test;
  cout << "TestCFitsFindAndTrace: I_A1_Test = " << I_A1_Test << endl;
  cout << "TestCFitsFindAndTrace: I_A1_TestCopy = " << I_A1_TestCopy << endl;

  Array<int, 1> I_A1_Expected(I_A1_Test.size());
  I_A1_Expected(0) = 0;
  I_A1_Expected(1) = 0;
  I_A1_Expected(2) = 1;
  I_A1_Expected(3) = 2;
  I_A1_Expected(4) = 3;
  I_A1_Expected(5) = 4;
  I_A1_Expected(6) = 5;
  I_A1_Expected(7) = 0;
  I_A1_Expected(8) = 0;
  I_A1_Expected(9) = 0;
  I_A1_Expected(10) = 0;
  I_A1_Expected(11) = 1;
  I_A1_Expected(12) = 2;
  I_A1_Expected(13) = 3;
  I_A1_Expected(14) = 4;
  I_A1_Expected(15) = 5;
  I_A1_Expected(16) = 6;
  I_A1_Expected(17) = 0;
  I_A1_Expected(18) = 0;
  I_A1_Expected(19) = 0;

  int I_Result;
  if (P_MyCFits->CountPixGTZero(I_A1_TestCopy)){
    t1 = true;
    cout << "TestCFitsFindAndTrace: t1: CountPixGTZero returned true (expected)" << endl;
  }
  else{
    t1 = false;
    cout << "TestCFitsFindAndTrace: t1: CountPixGTZero returned false (UNEXPECTED)" << endl;
  }

  t2 = true;
  for (int i=0; i < I_A1_Test.size(); i++){
    if (I_A1_TestCopy(i) != I_A1_Expected(i)){
    cout << "TestCFitsFindAndTrace: t2: I_A1_TestCopy(i=" << i << ") = " << I_A1_TestCopy(i) << " != I_A1_Expected(i) = " << I_A1_Expected(i) << " (UNEXPECTED)" << endl;
      t2 = false;
    }
  }

  int I_FirstIndGE = P_MyCFits->FirstIndexWithValueGE(I_A1_TestCopy, (int)(2. * D_FWHM));
  t3 = false;
  if (I_FirstIndGE == 3){
    cout << "TestCFitsFindAndTrace: t3: I_FirstIndGE = " << I_FirstIndGE << " (expected)" << endl;
    t3 = true;
  }
  else{
    cout << "TestCFitsFindAndTrace: t3: I_FirstIndGE = " << I_FirstIndGE << " (UNEXPECTED)" << endl;
  }

  int I_FirstIndGT = P_MyCFits->FirstIndexWithValueGT(I_A1_TestCopy, (int)(2. * D_FWHM));
  t4 = false;
  if (I_FirstIndGT == 4){
    cout << "TestCFitsFindAndTrace: t4: I_FirstIndGT = " << I_FirstIndGT << " (expected)" << endl;
    t4 = true;
  }
  else{
    cout << "TestCFitsFindAndTrace: t4: I_FirstIndGT = " << I_FirstIndGT << " (UNEXPECTED)" << endl;
  }

  int I_FirstZero;

  I_FirstZero = P_MyCFits->FirstIndexWithZeroValueFrom(I_A1_TestCopy, I_A1_TestCopy.size()-1);
  if (I_FirstZero == I_A1_TestCopy.size()-1){
    cout << "TestCFitsFindAndTrace: t5: I_FirstZero = " << I_FirstZero << " (expected)" << endl;
    t5 = true;
  }
  else{
    cout << "TestCFitsFindAndTrace: t5: I_FirstZero = " << I_FirstZero << " (UNEXPECTED)" << endl;
    t5 = false;
  }

  I_FirstZero = P_MyCFits->FirstIndexWithZeroValueFrom(I_A1_TestCopy, I_A1_TestCopy.size());
  if (I_FirstZero == -1){
    cout << "TestCFitsFindAndTrace: t6: I_FirstZero = " << I_FirstZero << " (expected)" << endl;
    t6 = true;
  }
  else{
    cout << "TestCFitsFindAndTrace: t6: I_FirstZero = " << I_FirstZero << " (UNEXPECTED)" << endl;
    t6 = false;
  }

  I_FirstZero = P_MyCFits->FirstIndexWithZeroValueFrom(I_A1_TestCopy, I_FirstIndGT);
  t7 = false;
  if (I_FirstZero == 7){
    cout << "TestCFitsFindAndTrace: t7: I_FirstZero = " << I_FirstZero << " (expected)" << endl;
    t7 = true;
  }
  else{
    cout << "TestCFitsFindAndTrace: t7: I_FirstZero = " << I_FirstZero << " (UNEXPECTED)" << endl;
  }

  I_A1_TestCopy(19) = 1;
  I_FirstZero = P_MyCFits->FirstIndexWithZeroValueFrom(I_A1_TestCopy, I_A1_TestCopy.size()-1);
  if (I_FirstZero == -1){
    cout << "TestCFitsFindAndTrace: t8: I_FirstZero = " << I_FirstZero << " (expected)" << endl;
    t8 = true;
  }
  else{
    cout << "TestCFitsFindAndTrace: t8: I_FirstZero = " << I_FirstZero << " (UNEXPECTED)" << endl;
    t8 = false;
  }

  I_A1_TestCopy(0) = 1;
  I_FirstZero = P_MyCFits->FirstIndexWithZeroValueFrom(I_A1_TestCopy, 0);
  if (I_FirstZero == 1){
    cout << "TestCFitsFindAndTrace: t9: I_FirstZero = " << I_FirstZero << " (expected)" << endl;
    t9 = true;
  }
  else{
    cout << "TestCFitsFindAndTrace: t9: I_FirstZero = " << I_FirstZero << " (UNEXPECTED)" << endl;
    t9 = false;
  }

  int I_LastZero;
  I_LastZero = P_MyCFits->LastIndexWithZeroValueBefore(I_A1_TestCopy, 5);
  if (I_LastZero == 1){
    cout << "TestCFitsFindAndTrace: t10: I_LastZero = " << I_LastZero << " (expected)" << endl;
    t10 = true;
  }
  else{
    cout << "TestCFitsFindAndTrace: t10: I_LastZero = " << I_LastZero << " (UNEXPECTED)" << endl;
    t10 = false;
  }

  I_LastZero = P_MyCFits->LastIndexWithZeroValueBefore(I_A1_TestCopy, 0);
  if (I_LastZero == -1){
    cout << "TestCFitsFindAndTrace: t11: I_LastZero = " << I_LastZero << " (expected)" << endl;
    t11 = true;
  }
  else{
    cout << "TestCFitsFindAndTrace: t11: I_LastZero = " << I_LastZero << " (UNEXPECTED)" << endl;
    t11 = false;
  }
  CString CS_FileName("/home/azuri/spectra/SEDIFU/SEDM-sim-cal-2012-05-14_s.fits");

  P_MyCFits->SetFileName(CS_FileName);
  P_MyCFits->ReadArray();
  Array<int, 2> I_A2_Where(P_MyCFits->GetNRows(), P_MyCFits->GetNCols());
  I_A2_Where = where(P_MyCFits->GetPixArray()<=0,1,0);
  Array<int, 2> I_A2_Ind(2,2);
  int I_NInds;
  P_MyCFits->GetIndex(I_A2_Where, I_NInds, I_A2_Ind);
  Array<double,2> D_A2_TempA(P_MyCFits->GetNRows(), P_MyCFits->GetNCols());
  D_A2_TempA = P_MyCFits->GetPixArray();
  P_MyCFits->GetPixArray() = sqrt(P_MyCFits->GetPixArray());
  for (int ii=0; ii< I_NInds; ii++){
    (P_MyCFits->GetPixArray())(I_A2_Ind(ii,0), I_A2_Ind(ii,1)) = 0. - D_A2_TempA(I_A2_Ind(ii,0), I_A2_Ind(ii,1));
  }
  P_MyCFits->SetFileName(CString("/home/azuri/spectra/SEDIFU/SEDM-sim-cal-2012-05-14_s_err.fits"));
  P_MyCFits->WriteArray();
  delete(P_MyCFits);
  P_MyCFits = NULL;
  return false;

  CS_FileName.Set("/home/azuri/spectra/SEDIFU/SEDM-deep-sim-flat-2012-05-14_Flat.fits");

  CFits *P_CF_Bias = new CFits();
  P_CF_Bias->SetFileName(CString("/home/azuri/spectra/SEDIFU/SEDM-deep-sim-flat-2012-05-14_Read.fits"));
  P_CF_Bias->ReadArray();
  cout << "MTestApp: mean(P_CF_Bias) = " << mean(P_CF_Bias->GetPixArray()) << endl;


  P_MyCFits->SetFileName(CS_FileName);
  P_MyCFits->ReadArray();
  *P_MyCFits -= mean(P_CF_Bias->GetPixArray());

  CFits *P_CF_Scatter = new CFits();
  P_CF_Scatter->SetFileName(CString("/home/azuri/spectra/SEDIFU/SEDM-deep-sim-flat-2012-05-14_ScatterFit.fits"));
  P_CF_Scatter->ReadArray();

  Array<double, 2> D_A2_Flat = P_MyCFits->GetPixArray();
  cout << "TestCFitsFindAndTrace: D_A2_Flat.rows() = " << D_A2_Flat.rows() << endl;
  cout << "TestCFitsFindAndTrace: D_A2_Flat.cols() = " << D_A2_Flat.cols() << endl;
  Array<double, 2> D_A2_FlatCrop(D_A2_Flat.rows()-1700, D_A2_Flat.cols() - 1700);
  D_A2_FlatCrop = D_A2_Flat(Range(850, D_A2_Flat.rows()-850-1), Range(850, D_A2_Flat.cols()-850-1));

  P_CF_Scatter->SetNRows(P_CF_Scatter->GetNRows()-1);
  P_CF_Scatter->SetNCols(P_CF_Scatter->GetNCols()-1);
  P_CF_Scatter->SetFileName(CString("/home/azuri/spectra/SEDIFU/SEDM-deep-sim-flat-2012-05-14_ScatterFit_2048x2048.fits"));
  P_CF_Scatter->WriteArray();
  Array<double, 2> D_A2_Scatter = P_CF_Scatter->GetPixArray();
  cout << "TestCFitsFindAndTrace: D_A2_Scatter.rows() = " << D_A2_Scatter.rows() << endl;
  cout << "TestCFitsFindAndTrace: D_A2_Scatter.cols() = " << D_A2_Scatter.cols() << endl;
  return false;
  Array<double, 2> D_A2_ScatterCrop(D_A2_Scatter.rows()-1700, D_A2_Scatter.cols() - 1700);
  D_A2_ScatterCrop = D_A2_Scatter(Range(850, D_A2_Scatter.rows()-850-1), Range(850, D_A2_Scatter.cols()-850-1));

  double D_Fac;
  if (!P_CF_Scatter->ScaleImageToFitBackground(D_A2_FlatCrop, D_A2_ScatterCrop, D_Fac))
    return false;
  cout << "TestCFitsFindAndTrace: D_Fac = " << D_Fac << endl;

  Array<double, 2> D_A2_Temp(D_A2_Scatter.rows(), D_A2_Scatter.cols());
  D_A2_Temp = D_A2_Scatter * D_Fac;
  D_A2_Temp.resizeAndPreserve(D_A2_Flat.rows(), D_A2_Flat.cols());
  *P_MyCFits -= D_A2_Temp;
  P_MyCFits->WriteFits(&(P_MyCFits->GetPixArray()), CString("/home/azuri/spectra/SEDIFU/SEDM-deep-sim-flat-2012-05-14_Flat-back.fits"));

//  delete(P_MyCFits);
//  P_MyCFits = new CFits();
//  P_MyCFits->SetFileName(CString("/home/azuri/spectra/SEDIFU/SEDM-deep-sim-flat-2012-05-14_Flat-back.fits"));
//  P_MyCFits->ReadArray();

  P_MyCFits->Set_SaturationLevel(65000.);
  P_MyCFits->Set_SignalThreshold(80.);
  P_MyCFits->Set_ApertureFWHM(1.16);
  cout << "TestCFitsFindAndTrace: P_MyCFits->Get_ApertureFWHM = " << P_MyCFits->Get_ApertureFWHM() << endl;
  t12 = P_MyCFits->FindAndTraceApertures(3,
                                         2,
//                                         170,
//                                         280,
                                         3);
  if (t12){
    cout << "TestCFitsFindAndTrace: FindAndTraceApertures returned TRUE (expected)" << endl;
  }
  else{
    cout << "TestCFitsFindAndTrace: FindAndTraceApertures returned FALSE (UNEXPECTED)" << endl;
  }
  P_MyCFits->SetDatabaseFileName(CString("/home/azuri/spectra/SEDIFU/database/apSEDM-deep-sim-flat-2012-05-14_Flat-back"));
  Array<double, 1> D_A1_XLow(P_MyCFits->Get_NApertures());
  D_A1_XLow = -4.;

  P_MyCFits->Set_XLow(D_A1_XLow);
  D_A1_XLow = -4.;
  P_MyCFits->Set_XMin(D_A1_XLow);
  Array<double, 1> D_A1_XHigh(P_MyCFits->Get_NApertures());
  D_A1_XHigh = 4.;
  P_MyCFits->Set_XHigh(D_A1_XHigh);
  D_A1_XHigh = 4.;
  P_MyCFits->Set_XMax(D_A1_XHigh);
  P_MyCFits->WriteDatabaseEntry();

  cout << "TestCFitsFindAndTrace: P_MyCFits->Get_NApertures returns " << P_MyCFits->Get_NApertures() << endl;
  cout << "TestCFitsFindAndTrace: D_A1_XHigh = " << D_A1_XHigh << endl;
  cout << "TestCFitsFindAndTrace: P_MyCFits->Get_XMax returns " << P_MyCFits->Get_XMax() << endl;
//  return false;

  ///read aperture definition file
  cout << "TestCFitsFindAndTrace: deleting P_MyCFits" << endl;
  P_MyCFits = NULL;
  cout << "TestCFitsFindAndTrace: P_MyCFits deleted" << endl;
  P_MyCFits = new CFits();
  CS_FileName.Set("/home/azuri/spectra/SEDIFU/SEDM-deep-sim-flat-2012-05-14_Flat-back.fits");
  cout << "TestCFitsFindAndTrace: Setting filename" << endl;
  if (!P_MyCFits->SetFileName(CS_FileName))
    return false;
  cout << "TestCFitsFindAndTrace: Filename " << CS_FileName << " set" << endl;
  if (!P_MyCFits->ReadArray())
    return false;
  cout << "TestCFitsFindAndTrace: Array read" << endl;
  if (!P_MyCFits->SetDatabaseFileName(CString("/home/azuri/spectra/SEDIFU/database/apSEDM-deep-sim-flat-2012-05-14_Flat-back")))
    return false;
  cout << "TestCFitsFindAndTrace: Database Filename set to " << P_MyCFits->GetDatabaseFileName() << endl;
  if (!P_MyCFits->ReadDatabaseEntry())
    return false;
  cout << "TestCFitsFindAndTrace: Database entry read" << endl;
  if (!P_MyCFits->CalcTraceFunctions())
    return false;
  cout << "TestCFitsFindAndTrace: Trace functions calculated" << endl;
  if (!P_MyCFits->MarkCenters())
    return false;
  cout << "TestCFitsFindAndTrace: centers marked" << endl;
  CS_FileName.Set("/home/azuri/spectra/SEDIFU/SEDM-deep-sim-flat-2012-05-14_Flat-back_marked.fits");
  if (!P_MyCFits->SetFileName(CS_FileName))
    return false;
  cout << "TestCFitsFindAndTrace: Filename " << CS_FileName << " set" << endl;
  if (!P_MyCFits->WriteArray())
    return false;
  cout << "TestCFitsFindAndTrace: Array written" << endl;

  return false;

  int i_temp = 5;
  double d_temp = 5.;
  int i_temp_half = int(i_temp/2.);
  int i_d_temp_half = int(d_temp/2.);
  int i_i_temp_half = int(i_temp/2);
  cout << "i_temp_half = " << i_temp_half << endl;
  cout << "i_d_temp_half = " << i_d_temp_half << endl;
  cout << "i_i_temp_half = " << i_i_temp_half << endl;

  //  return false;
  /// Add scattered light
  delete P_MyCFits;
  P_MyCFits = new CFits();
  CS_FileName.Set("/home/azuri/spectra/SEDIFU/SEDM-deep-sim-flat-2012-05-14_Flat.fits");
  if (!P_MyCFits->SetFileName(CS_FileName))
    return false;
  if (!P_MyCFits->ReadArray())
    return false;
  if (!P_MyCFits->SetDatabaseFileName(CString("/home/azuri/spectra/SEDIFU/database/apSEDM-deep-sim-flat-2012-05-14_Flat")))
    return false;
  if (!P_MyCFits->ReadDatabaseEntry())
    return false;
  if (!P_MyCFits->CalcTraceFunctions())
    return false;
//  Array<double, 2> D_A2_ScatteredLight(P_MyCFits->GetNRows(), P_MyCFits->GetNCols());
//  D_A2_ScatteredLight = 20.;
//  P_MyCFits->GetPixArray() = P_MyCFits->GetPixArray() + D_A2_ScatteredLight;
//  CS_FileName.Set("/home/azuri/spectra/SEDIFU/SEDM-deep-sim-flat-2012-05-14_Flat.fits");
//  if (!P_MyCFits->SetFileName(CS_FileName))
//    return false;
//  if (!P_MyCFits->WriteArray())
//    return false;
//  Array<CString, 1> CS_A1_Args_MkScatter(10);
//  CS_A1_Args_MkScatter = CString(" ");
//  void **PP_Args_MkScatter;
//  PP_Args_MkScatter = (void**)malloc(sizeof(void*) * 10);
  /** KeyWords and Values: COLRANGE=colrange
  LAMBDA_SF=lam_sf...double
  LAMBDA_SP=lam_sp...double
  SWATH_WIDTH=swath_width...int
  OSAMPLE=osample...int
  MASK=mask
  SUBTRACT=subtract...bool
  POLY=pol
  DATA=back_data
  ORDER_WIDTH=order_width
  **/
  int I_OSample = 10;
  if (!P_MyCFits->Set_OverSample(I_OSample))
    return false;
//  CS_A1_Args_MkScatter(0) = CString("SUBTRACT");
//  CS_A1_Args_MkScatter(1) = CString("DATA");
//  Array<double, 2> D_A2_BackData(2,2);
//  PP_Args_MkScatter[1] = &D_A2_BackData;
//  CS_A1_Args_MkScatter(2) = CString("OSAMPLE");
//  PP_Args_MkScatter[2] = &I_OSample;
  //CS_A1_Args_MkScatter(1) = CString("ORDER_WIDTH");


  if (!P_MyCFits->CalculateScatteredLight(3,10))
    return false;
//  cout << "D_A2_BackData = " << D_A2_BackData << endl;
  CS_FileName.Set("/home/azuri/spectra/SEDIFU/SEDM-deep-sim-flat-2012-05-14_Flat_s.fits");
  P_MyCFits->SetFileName(CS_FileName);
  P_MyCFits->WriteArray();

  return false;

  delete P_MyCFits;
  P_MyCFits = new CFits();
  CS_FileName.Set("/home/azuri/spectra/rave/aaron_6df/6df/combinedFlat.fits");
  P_MyCFits->SetFileName(CS_FileName);
  P_MyCFits->ReadArray();
  P_MyCFits->Set_SaturationLevel(65000.);
  P_MyCFits->Set_SignalThreshold(10000.);
  P_MyCFits->Set_ApertureFWHM(2.);
  cout << "TestCFitsFindAndTrace: P_MyCFits->Get_ApertureFWHM = " << P_MyCFits->Get_ApertureFWHM() << endl;
  t12 = P_MyCFits->FindAndTraceApertures(3,
                                         3,
//                                         1000,
//                                         1200,
                                         1);
  if (t12){
    cout << "TestCFitsFindAndTrace: FindAndTraceApertures returned TRUE (expected)" << endl;
  }
  else{
    cout << "TestCFitsFindAndTrace: FindAndTraceApertures returned FALSE (UNEXPECTED)" << endl;
  }

  return (t1 && t2 && t3 && t4 && t5 && t6 && t7 && t8 && t9 && t10 && t11 && t12);
}

/** ***********************************************************/

bool TestCFitsPixInFigure()
{
  // Test: CFits::CrossCorrelate
  // Tests included: CFits::GetSubArrCopy
  // require : nothing
  // ensure  : t1, t2, t3, t4, t5 are "TRUE"
  bool t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t35, t36, t37, t38, t39, t40;
  CFits *P_MyCFits = new CFits();

  Array<double, 2> D_A2_LineA(2,2);
  Array<double, 2> D_A2_LineB(2,2);
  Array<double, 1> D_A1_Point(2);
  Array<double, 1> D_A1_Point_Ref(2);

  /// y=x
  D_A2_LineA(0,0) = 0.;
  D_A2_LineA(0,1) = 0.;
  D_A2_LineA(1,0) = 1.;
  D_A2_LineA(1,1) = 1.;

  /// horizontal line y=0.5
  D_A2_LineB(0,0) = 0.;
  D_A2_LineB(0,1) = 0.5;
  D_A2_LineB(1,0) = 1.;
  D_A2_LineB(1,1) = 0.5;

  D_A1_Point_Ref(0) = 0.5;
  D_A1_Point_Ref(1) = 0.5;
  t1=P_MyCFits->FindCrossPoint(D_A2_LineA, D_A2_LineB, D_A1_Point);
  if (t1){
    cout << "MTestApp::TestCFitsPixInFigure: t1 returned TRUE (EXPECTED)" << endl;
  }
  else{
    cout << "MTestApp::TestCFitsPixInFigure: t1 returned FALSE (UNEXPECTED)" << endl;
    return false;
  }
  if (max(fabs(D_A1_Point_Ref - D_A1_Point)) < 0.0000000001){
    cout << "MTestApp::TestCFitsPixInFigure: D_A1_Point(=" << D_A1_Point << ") == D_A1_Point_Ref(=" << D_A1_Point_Ref << ") returned TRUE (EXPECTED)" << endl;
  }
  else{
    cout << "MTestApp::TestCFitsPixInFigure: D_A1_Point(=" << D_A1_Point << ") == D_A1_Point_Ref(=" << D_A1_Point_Ref << ") returned FALSE (UNEXPECTED)" << endl;
    return false;
  }
  cout << "t1 PASSED" << endl << endl;

  t2=P_MyCFits->FindCrossPoint(D_A2_LineB, D_A2_LineA, D_A1_Point);
  if (t2){
    cout << "MTestApp::TestCFitsPixInFigure: t2 returned TRUE (EXPECTED)" << endl;
  }
  else{
    cout << "MTestApp::TestCFitsPixInFigure: t2 returned FALSE (UNEXPECTED)" << endl;
    return false;
  }
  if (max(fabs(D_A1_Point_Ref - D_A1_Point)) < 0.0000000001){
    cout << "MTestApp::TestCFitsPixInFigure: D_A1_Point(=" << D_A1_Point << ") == D_A1_Point_Ref(=" << D_A1_Point_Ref << ") returned TRUE (EXPECTED)" << endl;
  }
  else{
    cout << "MTestApp::TestCFitsPixInFigure: D_A1_Point(=" << D_A1_Point << ") == D_A1_Point_Ref(=" << D_A1_Point_Ref << ") returned FALSE (UNEXPECTED)" << endl;
    return false;
  }
  cout << "t2 PASSED" << endl << endl;

  /// Crossing point is end of line B
  D_A2_LineB(0,0) = 0.;
  D_A2_LineB(0,1) = 1.5;
  D_A2_LineB(1,0) = 1.5;
  D_A2_LineB(1,1) = 0.;

  D_A1_Point_Ref(0) = 0.75;
  D_A1_Point_Ref(1) = 0.75;
  t3=P_MyCFits->FindCrossPoint(D_A2_LineB, D_A2_LineA, D_A1_Point);
  if (t3){
    cout << "MTestApp::TestCFitsPixInFigure: t3 returned TRUE (EXPECTED)" << endl;
  }
  else{
    cout << "MTestApp::TestCFitsPixInFigure: t3 returned FALSE (UNEXPECTED)" << endl;
    return false;
  }
  if (max(fabs(D_A1_Point_Ref - D_A1_Point)) < 0.0000000001){
    cout << "MTestApp::TestCFitsPixInFigure: D_A1_Point(=" << D_A1_Point << ") == D_A1_Point_Ref(=" << D_A1_Point_Ref << ") returned TRUE (EXPECTED)" << endl;
  }
  else{
    cout << "MTestApp::TestCFitsPixInFigure: D_A1_Point(=" << D_A1_Point << ") == D_A1_Point_Ref(=" << D_A1_Point_Ref << ") returned FALSE (UNEXPECTED)" << endl;
    return false;
  }
  cout << "t3 PASSED" << endl << endl;

  /// LineA vertical
  D_A2_LineA(0,0) = 0.5;
  D_A2_LineA(0,1) = 1.;
  D_A2_LineA(1,0) = 0.5;
  D_A2_LineA(1,1) = 0.;

  D_A1_Point_Ref(0) = 0.5;
  D_A1_Point_Ref(1) = 1.;

  t4=P_MyCFits->FindCrossPoint(D_A2_LineB, D_A2_LineA, D_A1_Point);
  if (t4){
    cout << "MTestApp::TestCFitsPixInFigure: t4 returned TRUE (EXPECTED)" << endl;
  }
  else{
    cout << "MTestApp::TestCFitsPixInFigure: t4 returned FALSE (UNEXPECTED)" << endl;
    return false;
  }
  if (max(fabs(D_A1_Point_Ref - D_A1_Point)) < 0.0000000001){
    cout << "MTestApp::TestCFitsPixInFigure: D_A1_Point(=" << D_A1_Point << ") == D_A1_Point_Ref(=" << D_A1_Point_Ref << ") returned TRUE (EXPECTED)" << endl;
  }
  else{
    cout << "MTestApp::TestCFitsPixInFigure: D_A1_Point(=" << D_A1_Point << ") == D_A1_Point_Ref(=" << D_A1_Point_Ref << ") returned FALSE (UNEXPECTED)" << endl;
    return false;
  }
  cout << "t4 PASSED" << endl << endl;

  /// normal lines, no crosspoint
  D_A2_LineA(0,0) = 0.;
  D_A2_LineA(0,1) = 0.5;
  D_A2_LineA(1,0) = 0.;
  D_A2_LineA(1,1) = 0.5;
  t5=P_MyCFits->FindCrossPoint(D_A2_LineB, D_A2_LineA, D_A1_Point);
  if (!t5){
    cout << "MTestApp::TestCFitsPixInFigure: t5 returned FALSE (EXPECTED)" << endl;
  }
  else{
    cout << "MTestApp::TestCFitsPixInFigure: t5 returned TRUE (UNEXPECTED)" << endl;
    return false;
  }
  cout << "t5 PASSED" << endl << endl;

  /// Vertical line, no crosspoint
  D_A2_LineA(0,0) = 0.;
  D_A2_LineA(0,1) = 1.;
  D_A2_LineA(1,0) = 0.;
  D_A2_LineA(1,1) = 0.;
  t6=P_MyCFits->FindCrossPoint(D_A2_LineB, D_A2_LineA, D_A1_Point);
  if (!t6){
    cout << "MTestApp::TestCFitsPixInFigure: t6 returned FALSE (EXPECTED)" << endl;
  }
  else{
    cout << "MTestApp::TestCFitsPixInFigure: t6 returned TRUE (UNEXPECTED)" << endl;
    return false;
  }
  cout << "t6 PASSED" << endl << endl;

  /// SortCoordinatesCounterClockWise
  Array<double, 2> D_A2_Fig(3,2);
  Array<double, 2> D_A2_FigSorted(2,2);
  Array<double, 2> D_A2_FigSorted_Test(3,2);
  Array<double, 2> D_A2_Test(1,2);
  D_A2_Test = 1.;
  Array<double, 2> D_A2_TestResult(2,2);
  t7 = P_MyCFits->SortCoordinatesCounterClockWise(D_A2_Test, D_A2_TestResult);
  if (!t7){
    cout << "MTestApp::TestCFitsPixInFigure: t7 returned FALSE (EXPECTED)" << endl;
  }
  else{
    cout << "MTestApp::TestCFitsPixInFigure: t7 returned TRUE (UNEXPECTED)" << endl;
    return false;
  }
  cout << "t7 PASSED" << endl << endl;

  D_A2_FigSorted_Test.resize(2,2);
  D_A2_FigSorted_Test(0,0) = 0.;
  D_A2_FigSorted_Test(0,1) = 0.;
  D_A2_FigSorted_Test(1,0) = 0.;
  D_A2_FigSorted_Test(1,1) = 1.;
  t8 = P_MyCFits->SortCoordinatesCounterClockWise(D_A2_LineA, D_A2_LineB);
  if (!t8){
    cout << "MTestApp::TestCFitsPixInFigure: t8 returned FALSE (EXPECTED)" << endl;
  }
  else{
    cout << "MTestApp::TestCFitsPixInFigure: t8 returned TRUE (UNEXPECTED)" << endl;
    return false;
  }
/**  if (max(fabs(D_A2_LineB - D_A2_FigSorted_Test)) < 0.00000001){
    cout << "MTestApp::TestCFitsPixInFigure: t8: D_A2_FigSorted(=" << D_A2_LineB << ") == D_A2_FigSorted_Test(=" << D_A2_FigSorted_Test << ") (EXPECTED)" << endl;
  }
  else{
    cout << "MTestApp::TestCFitsPixInFigure: t8: D_A2_FigSorted(=" << D_A2_FigSorted << ") != D_A2_FigSorted_Test(=" << D_A2_FigSorted_Test << ") (UNEXPECTED)" << endl;
    return false;
  }**/
  cout << "t8 PASSED" << endl << endl;

  D_A2_Fig(0,0) = 1.;
  D_A2_Fig(0,1) = 1.;
  D_A2_Fig(1,0) = 0.;
  D_A2_Fig(1,1) = 0.;
  D_A2_Fig(2,0) = 0.;
  D_A2_Fig(2,1) = 1.;
  D_A2_FigSorted_Test.resize(3,2);
  D_A2_FigSorted_Test(0,0) = 0.;
  D_A2_FigSorted_Test(0,1) = 0.;
  D_A2_FigSorted_Test(1,0) = 0.;
  D_A2_FigSorted_Test(1,1) = 1.;
  D_A2_FigSorted_Test(2,0) = 1.;
  D_A2_FigSorted_Test(2,1) = 1.;
  t9 = P_MyCFits->SortCoordinatesCounterClockWise(D_A2_Fig, D_A2_FigSorted);
  if (t9){
    cout << "MTestApp::TestCFitsPixInFigure: t9 returned TRUE (EXPECTED)" << endl;
  }
  else{
    cout << "MTestApp::TestCFitsPixInFigure: t9 returned FALSE (UNEXPECTED)" << endl;
    return false;
  }
  if (max(fabs(D_A2_FigSorted - D_A2_FigSorted_Test)) < 0.000001){
    cout << "MTestApp::TestCFitsPixInFigure: t9: D_A2_FigSorted(=" << D_A2_FigSorted << ") == D_A2_FigSorted_Test(=" << D_A2_FigSorted_Test << ") (EXPECTED)" << endl;
  }
  else{
    cout << "MTestApp::TestCFitsPixInFigure: t9: D_A2_FigSorted(=" << D_A2_FigSorted << ") != D_A2_FigSorted_Test(=" << D_A2_FigSorted_Test << ") (UNEXPECTED)" << endl;
    return false;
  }
  cout << "t9 PASSED" << endl << endl;

  /// 4 corners
  D_A2_Fig.resize(4,2);
  D_A2_Fig(0,0) = 0.75;
  D_A2_Fig(0,1) = 0.75;
  D_A2_Fig(1,0) = 0.;
  D_A2_Fig(1,1) = 0.;
  D_A2_Fig(2,0) = 0.;
  D_A2_Fig(2,1) = 1.;
  D_A2_Fig(3,0) = 1.;
  D_A2_Fig(3,1) = 0.;

  D_A2_FigSorted_Test.resize(4,2);
  D_A2_FigSorted_Test(0,0) = 0.;
  D_A2_FigSorted_Test(0,1) = 0.;
  D_A2_FigSorted_Test(1,0) = 1.;
  D_A2_FigSorted_Test(1,1) = 0.;
  D_A2_FigSorted_Test(2,0) = 0.75;
  D_A2_FigSorted_Test(2,1) = 0.75;
  D_A2_FigSorted_Test(3,0) = 0.;
  D_A2_FigSorted_Test(3,1) = 1.;
  t10 = P_MyCFits->SortCoordinatesCounterClockWise(D_A2_Fig, D_A2_FigSorted);
  if (t10){
    cout << "MTestApp::TestCFitsPixInFigure: t10 returned TRUE (EXPECTED)" << endl;
  }
  else{
    cout << "MTestApp::TestCFitsPixInFigure: t10 returned FALSE (UNEXPECTED)" << endl;
    return false;
  }
  if (max(fabs(D_A2_FigSorted - D_A2_FigSorted_Test)) < 0.000001){
    cout << "MTestApp::TestCFitsPixInFigure: t10: D_A2_FigSorted(=" << D_A2_FigSorted << ") == D_A2_FigSorted_Test(=" << D_A2_FigSorted_Test << ") (EXPECTED)" << endl;
  }
  else{
    cout << "MTestApp::TestCFitsPixInFigure: t10: D_A2_FigSorted(=" << D_A2_FigSorted << ") != D_A2_FigSorted_Test(=" << D_A2_FigSorted_Test << ") (UNEXPECTED)" << endl;
    return false;
  }
  cout << "t10 PASSED" << endl << endl;

  /// 6 corners
  D_A2_Fig.resize(6,2);
  D_A2_Fig(0,0) = 0.75;
  D_A2_Fig(0,1) = 0.75;
  D_A2_Fig(1,0) = 0.;
  D_A2_Fig(1,1) = 0.;
  D_A2_Fig(2,0) = 0.;
  D_A2_Fig(2,1) = 1.;
  D_A2_Fig(3,0) = 1.;
  D_A2_Fig(3,1) = 0.;
  D_A2_Fig(4,0) = 0.5;
  D_A2_Fig(4,1) = -0.2;
  D_A2_Fig(5,0) = -0.2;
  D_A2_Fig(5,1) = 0.5;

  D_A2_FigSorted_Test.resize(6,2);
  D_A2_FigSorted_Test(0,0) = -0.2;
  D_A2_FigSorted_Test(0,1) = 0.5;
  D_A2_FigSorted_Test(1,0) = 0.;
  D_A2_FigSorted_Test(1,1) = 0.;
  D_A2_FigSorted_Test(2,0) = 0.5;
  D_A2_FigSorted_Test(2,1) = -0.2;
  D_A2_FigSorted_Test(3,0) = 1.;
  D_A2_FigSorted_Test(3,1) = 0.;
  D_A2_FigSorted_Test(4,0) = 0.75;
  D_A2_FigSorted_Test(4,1) = 0.75;
  D_A2_FigSorted_Test(5,0) = 0.;
  D_A2_FigSorted_Test(5,1) = 1.;
  t11 = P_MyCFits->SortCoordinatesCounterClockWise(D_A2_Fig, D_A2_FigSorted);
  if (t11){
    cout << "MTestApp::TestCFitsPixInFigure: t11 returned TRUE (EXPECTED)" << endl;
  }
  else{
    cout << "MTestApp::TestCFitsPixInFigure: t11 returned FALSE (UNEXPECTED)" << endl;
    return false;
  }
  if (max(fabs(D_A2_FigSorted - D_A2_FigSorted_Test)) < 0.000001){
    cout << "MTestApp::TestCFitsPixInFigure: t11: D_A2_FigSorted(=" << D_A2_FigSorted << ") == D_A2_FigSorted_Test(=" << D_A2_FigSorted_Test << ") (EXPECTED)" << endl;
  }
  else{
    cout << "MTestApp::TestCFitsPixInFigure: t11: D_A2_FigSorted(=" << D_A2_FigSorted << ") != D_A2_FigSorted_Test(=" << D_A2_FigSorted_Test << ") (UNEXPECTED)" << endl;
    return false;
  }
  cout << "t11 PASSED" << endl << endl;

  D_A2_Fig(0,0) = 621.262;
  D_A2_Fig(0,1) = 565.649;
  D_A2_Fig(1,0) = 602.647;
  D_A2_Fig(1,1) = 587.883;
  D_A2_Fig(2,0) = 613.157;
  D_A2_Fig(2,1) = 619.401;
  D_A2_Fig(3,0) = 641.499;
  D_A2_Fig(3,1) = 625.607;
  D_A2_Fig(4,0) = 660.525;
  D_A2_Fig(4,1) = 602.785;
  D_A2_Fig(5,0) = 649.784;
  D_A2_Fig(5,1) = 571.768;

  D_A2_FigSorted_Test(0,0) = 602.647;
  D_A2_FigSorted_Test(0,1) = 587.883;
  D_A2_FigSorted_Test(1,0) = 621.262;
  D_A2_FigSorted_Test(1,1) = 565.649;
  D_A2_FigSorted_Test(2,0) = 649.784;
  D_A2_FigSorted_Test(2,1) = 571.768;
  D_A2_FigSorted_Test(3,0) = 660.525;
  D_A2_FigSorted_Test(3,1) = 602.785;
  D_A2_FigSorted_Test(4,0) = 641.499;
  D_A2_FigSorted_Test(4,1) = 625.607;
  D_A2_FigSorted_Test(5,0) = 613.157;
  D_A2_FigSorted_Test(5,1) = 619.401;
  t12 = P_MyCFits->SortCoordinatesCounterClockWise(D_A2_Fig, D_A2_FigSorted);
  if (t12){
    cout << "MTestApp::TestCFitsPixInFigure: t12 returned TRUE (EXPECTED)" << endl;
  }
  else{
    cout << "MTestApp::TestCFitsPixInFigure: t12 returned FALSE (UNEXPECTED)" << endl;
    return false;
  }
  if (max(fabs(D_A2_FigSorted - D_A2_FigSorted_Test)) < 0.000001){
    cout << "MTestApp::TestCFitsPixInFigure: t12: D_A2_FigSorted(=" << D_A2_FigSorted << ") == D_A2_FigSorted_Test(=" << D_A2_FigSorted_Test << ") (EXPECTED)" << endl;
  }
  else{
    cout << "MTestApp::TestCFitsPixInFigure: t12: D_A2_FigSorted(=" << D_A2_FigSorted << ") != D_A2_FigSorted_Test(=" << D_A2_FigSorted_Test << ") (UNEXPECTED)" << endl;
    return false;
  }
  cout << "t12 PASSED" << endl << endl;

  /// Test SortCoordinatesToCreateHexagon
  t13 = P_MyCFits->SortCoordinatesToCreateHexagon(D_A2_Fig, D_A2_FigSorted);
  if (t13){
    cout << "MTestApp::TestCFitsPixInFigure: t13 returned TRUE (EXPECTED)" << endl;
  }
  else{
    cout << "MTestApp::TestCFitsPixInFigure: t13 returned FALSE (UNEXPECTED)" << endl;
    return false;
  }
  if (max(fabs(D_A2_FigSorted - D_A2_FigSorted_Test)) < 0.000001){
    cout << "MTestApp::TestCFitsPixInFigure: t13: D_A2_FigSorted(=" << D_A2_FigSorted << ") == D_A2_FigSorted_Test(=" << D_A2_FigSorted_Test << ") (EXPECTED)" << endl;
  }
  else{
    cout << "MTestApp::TestCFitsPixInFigure: t13: D_A2_FigSorted(=" << D_A2_FigSorted << ") != D_A2_FigSorted_Test(=" << D_A2_FigSorted_Test << ") (UNEXPECTED)" << endl;
    return false;
  }
  cout << "t13 PASSED" << endl << endl;

  /// Test IntegralInClosedStructure
  D_A2_Fig.resize(4,2);
  D_A2_Fig(0,0) = 0.;
  D_A2_Fig(0,1) = 0.;
  D_A2_Fig(1,0) = 1.;
  D_A2_Fig(1,1) = 0.;
  D_A2_Fig(2,0) = 1.;
  D_A2_Fig(2,1) = 1.;
  D_A2_Fig(3,0) = 0.;
  D_A2_Fig(3,1) = 1.;
  double D_Check = 1.;
  double D_Result;
  t14 = P_MyCFits->IntegralInClosedStructure(D_A2_Fig, D_Result);
  if (t14){
    cout << "MTestApp::TestCFitsPixInFigure: t14 returned TRUE (EXPECTED)" << endl;
  }
  else{
    cout << "MTestApp::TestCFitsPixInFigure: t14 returned FALSE (UNEXPECTED)" << endl;
    return false;
  }
  if (fabs(D_Check - D_Result) < 0.000001){
    cout << "MTestApp::TestCFitsPixInFigure: t14: D_Result(=" << D_Result << ") == D_Check(=" << D_Check << ") (EXPECTED)" << endl;
  }
  else{
    cout << "MTestApp::TestCFitsPixInFigure: t14: D_Result(=" << D_Result << ") != D_Check(=" << D_Check << ") (UNEXPECTED)" << endl;
    return false;
  }
  cout << "t14 PASSED" << endl << endl;

  /// Test PixelIsInFigure
  Array<double, 1> D_A1_Pix(2);
  D_A1_Pix(0) = 0.;
  D_A1_Pix(1) = 0.;
  t15 = P_MyCFits->PixelIsInFigure(D_A1_Pix, D_A2_Fig);
  if (t15){
    cout << "MTestApp::TestCFitsPixInFigure: t15 (PixelIsInFigure) returned TRUE (EXPECTED)" << endl;
  }
  else{
    cout << "MTestApp::TestCFitsPixInFigure: t15 (PixelIsInFigure) returned FALSE (UNEXPECTED)" << endl;
    return false;
  }

  D_A1_Pix(0) = 0.2;
  D_A1_Pix(1) = 0.2;
  t16 = P_MyCFits->PixelIsInFigure(D_A1_Pix, D_A2_Fig);
  if (t16){
    cout << "MTestApp::TestCFitsPixInFigure: t16 (PixelIsInFigure) returned TRUE (EXPECTED)" << endl;
  }
  else{
    cout << "MTestApp::TestCFitsPixInFigure: t16 (PixelIsInFigure) returned FALSE (UNEXPECTED)" << endl;
    return false;
  }

  D_A1_Pix(0) = 1.;
  D_A1_Pix(1) = 1.;
  t17 = P_MyCFits->PixelIsInFigure(D_A1_Pix, D_A2_Fig);
  if (t17){
    cout << "MTestApp::TestCFitsPixInFigure: t17 (PixelIsInFigure) returned TRUE (EXPECTED)" << endl;
  }
  else{
    cout << "MTestApp::TestCFitsPixInFigure: t17 (PixelIsInFigure) returned FALSE (UNEXPECTED)" << endl;
    return false;
  }

  D_A1_Pix(0) = 1.1;
  D_A1_Pix(1) = 0.;
  t18 = P_MyCFits->PixelIsInFigure(D_A1_Pix, D_A2_Fig);
  if (!t18){
    cout << "MTestApp::TestCFitsPixInFigure: t18 (PixelIsInFigure) returned FALSE (EXPECTED)" << endl;
  }
  else{
    cout << "MTestApp::TestCFitsPixInFigure: t18 (PixelIsInFigure) returned TRUE (UNEXPECTED)" << endl;
    return false;
  }

  D_A1_Pix(0) = 0.;
  D_A1_Pix(1) = 0.5;
  t19 = P_MyCFits->PixelIsInFigure(D_A1_Pix, D_A2_Fig);
  if (t19){
    cout << "MTestApp::TestCFitsPixInFigure: t19 (PixelIsInFigure) returned TRUE (EXPECTED)" << endl;
  }
  else{
    cout << "MTestApp::TestCFitsPixInFigure: t19 (PixelIsInFigure) returned FALSE (UNEXPECTED)" << endl;
    return false;
  }

  /// Test CalcOverlapFig
  Array<double, 2> D_A2_Rect(4,2);
  Array<double, 2> D_A2_Hex(6,2);
  Array<double, 2> D_A2_OverlapFig(2,2);

  D_A2_Rect(0,0) = 580.;
  D_A2_Rect(0,1) = 560.;
  D_A2_Rect(1,0) = 620.;
  D_A2_Rect(1,1) = 560.;
  D_A2_Rect(2,0) = 620.;
  D_A2_Rect(2,1) = 600.;
  D_A2_Rect(3,0) = 580.;
  D_A2_Rect(3,1) = 600.;

  D_A2_Hex(0,0) = 602.647;
  D_A2_Hex(0,1) = 587.883;
  D_A2_Hex(1,0) = 621.262;
  D_A2_Hex(1,1) = 565.649;
  D_A2_Hex(2,0) = 649.784;
  D_A2_Hex(2,1) = 571.768;
  D_A2_Hex(3,0) = 660.525;
  D_A2_Hex(3,1) = 602.785;
  D_A2_Hex(4,0) = 641.499;
  D_A2_Hex(4,1) = 625.607;
  D_A2_Hex(5,0) = 613.157;
  D_A2_Hex(5,1) = 619.401;
  t20 = P_MyCFits->CalcOverlapFig(D_A2_Rect, D_A2_Hex, D_A2_OverlapFig);
  if (t20){
    cout << "MTestApp::TestCFitsPixInFigure: t20 (CalcOverlapFig) returned TRUE (EXPECTED)" << endl;
  }
  else{
    cout << "MTestApp::TestCFitsPixInFigure: t20 (CalcOverlapFig) returned FALSE (UNEXPECTED)" << endl;
    return false;
  }
  cout << "MTestApp::TextCFitsPixInFigure: D_A2_OverlapFig = " << D_A2_OverlapFig << endl;

  D_A2_Rect(0,0) = 580.;
  D_A2_Rect(0,1) = 800.;
  D_A2_Rect(1,0) = 612.963;
  D_A2_Rect(1,1) = 800.;
  D_A2_Rect(2,0) = 612.963;
  D_A2_Rect(2,1) = 850.;
  D_A2_Rect(3,0) = 580.;
  D_A2_Rect(3,1) = 850.;

  D_A2_Hex(0,0) = 524.678;
  D_A2_Hex(0,1) = 817.27;
  D_A2_Hex(1,0) = 543.067;
  D_A2_Hex(1,1) = 795.181;
  D_A2_Hex(2,0) = 572.01;
  D_A2_Hex(2,1) = 801.268;
  D_A2_Hex(3,0) = 582.033;
  D_A2_Hex(3,1) = 831.005;
  D_A2_Hex(4,0) = 563.908;
  D_A2_Hex(4,1) = 852.233;
  D_A2_Hex(5,0) = 535.591;
  D_A2_Hex(5,1) = 846.77;
  t21 = P_MyCFits->CalcOverlapFig(D_A2_Rect, D_A2_Hex, D_A2_OverlapFig);
  if (t21){
    cout << "MTestApp::TestCFitsPixInFigure: t21 (CalcOverlapFig) returned TRUE (EXPECTED)" << endl;
  }
  else{
    cout << "MTestApp::TestCFitsPixInFigure: t21 (CalcOverlapFig) returned FALSE (UNEXPECTED)" << endl;
    return false;
  }
  cout << "MTestApp::TextCFitsPixInFigure: D_A2_OverlapFig = " << D_A2_OverlapFig << endl;

  D_A2_Rect(0,0) = 612.963;
  D_A2_Rect(0,1) = 600;
  D_A2_Rect(1,0) = 645.926;
  D_A2_Rect(1,1) = 600;
  D_A2_Rect(2,0) = 645.926;
  D_A2_Rect(2,1) = 650;
  D_A2_Rect(3,0) = 612.963;
  D_A2_Rect(3,1) = 650;

  D_A2_Hex(0,0) = 602.647;
  D_A2_Hex(0,1) = 587.883;
  D_A2_Hex(1,0) = 621.262;
  D_A2_Hex(1,1) = 565.649;
  D_A2_Hex(2,0) = 649.784;
  D_A2_Hex(2,1) = 571.768;
  D_A2_Hex(3,0) = 660.525;
  D_A2_Hex(3,1) = 602.785;
  D_A2_Hex(4,0) = 641.499;
  D_A2_Hex(4,1) = 625.607;
  D_A2_Hex(5,0) = 613.157;
  D_A2_Hex(5,1) = 619.401;
  t22 = P_MyCFits->CalcOverlapFig(D_A2_Rect, D_A2_Hex, D_A2_OverlapFig);
  if (t22){
    cout << "MTestApp::TestCFitsPixInFigure: t22 (CalcOverlapFig) returned TRUE (EXPECTED)" << endl;
  }
  else{
    cout << "MTestApp::TestCFitsPixInFigure: t22 (CalcOverlapFig) returned FALSE (UNEXPECTED)" << endl;
    return false;
  }
  cout << "MTestApp::TextCFitsPixInFigure: D_A2_OverlapFig = " << D_A2_OverlapFig << endl;


  return true;
}

/** *************************************************/

bool TestCFits()
{
  return (//TestCFitsConstructors() &&
//  TestCFitsClassInvAndCopyAndEVal() &&
//  TestCFitsMedian() &&
//  TestCFitsLinFit());// &&
//  TestCFitsReformMultInvert() &&
//  TestCFitsInterPolUniqCeil() &&
//  TestCFitsPiskunov() &&
//  TestCFitsMkProf() &&
//  TestCFitsMkScatter() &&
//  TestCFitsTelluric() &&
//  TestCFitsCrossCorrelate() &&
  TestCFitsPixInFigure());// &&
//  TestCFitsFindAndTrace());
}

/************************************************************/

bool Run(CString &cn)
{
  cout << "MTestApp::Run(" << cn << ") started" << endl;
  if (cn.EqualValue(CString("CString")))
  {
    if (TestCString())
    {
      cout << "MTestApp::Run(" << cn << "): Test, ,  PASSED" << endl;
      return true;
    }
    cout << "MTestApp::Run(" << cn << "): Test, ,  FAILED" << endl;
    return false;
  }
  else if (cn.EqualValue(CString("CFormatedString")))
  {
    if (TestCFormatedString())
    {
      cout << "MTestApp::Run(" << cn << "): Test, ,  PASSED" << endl;
      return true;
    }
    cout << "MTestApp::Run(" << cn << "): Test, ,  FAILED" << endl;
    return false;
  }
  else if (cn == CString("CFits")){
    if (TestCFits()){
      cout << "MTestApp::Run(" << cn << "): Test, , , , PASSED" << endl;
      return true;
    }
    cout << "MTestApp::Run(" << cn << "): Test, , , , FAILED" << endl;
    return false;
  }
/*  else if (cn == "CDate"){
    if (TestCDate()){
, cout << "MTestApp::Run(" << cn << "): Test, , , , PASSED" << endl;
, return true;
    }
    cout << "MTestApp::Run(" << cn << "): Test, , , , FAILED" << endl;
    return false;
}*/
  return true;
}

/************************************************************************/


int main(int argc, char *argv[])
{
  printf("CFits.main: STARTED\n");
  cout << "Hello, world!" << endl;

  CString testString("test/combinedFlat.fits\0");
//  CString *P_CS_DatabaseString = new CString("/home/azuri/entwicklung/idl/REDUCE/REDUCE/database/apcombinedFlat\0");
/*  CFits *p_testfits = new CFits();

  p_testfits->SetFileName(*p_testString);
  p_testfits->SetDatabaseFileName(*p_CS_DatabaseString);
  p_testfits->ReadArray();
  p_testfits->ReadDatabaseEntry();
  if (!p_testfits->CalcTraceFunctions())
  {
    cout << "MTestApp: p_testfits(=" << *p_testfits << ")->CalcTraceFunctions() returned FALSE => Returning FALSE";
    return false;
  }

  p_testString->Set("test/orderdef.fits");
  CFits *p_testfitsA = new CFits();
  p_testfitsA->SetFileName(*p_testString);
  p_testfitsA->ReadArray();

  for (int o = 14; o < 15; o++)
  {
//  int xmin, xmax;
    Array<double, 2> D_A2_PixArray(p_testfits->GetPixArray());
    Array<double, 2> D_A2_PixArrayA(p_testfitsA->GetPixArray());
    if (p_testfits->GetNRows() != p_testfitsA->GetNRows() || p_testfits->GetNCols() != p_testfitsA->GetNCols())
    {
      cout << "MTestApp: ERROR: size of testfits != size of testfitsA => Returning FALSE" << endl;
      return false;
    }
//  for (int n = 3800; n < p_testfits->Get_XCenters().cols(); n++)
//  {
//    cout << "MTestApp: row = n = " << n << endl;
    Array<int, 1> xmin(p_testfits->Get_XCenters().cols());
    xmin = p_testfits->Get_XCenters()(o,Range::all()) + p_testfits->Get_Low()(o)+8;
//    cout << "MTestApp: xmin = " << xmin << endl;
    Array<int, 1> xmax(p_testfits->Get_XCenters().cols());
    xmax = p_testfits->Get_XCenters()(o,Range::all()) + p_testfits->Get_High()(o) + 15;
    cout << "MTestApp: xmin = " << xmin << ", xmax = " << xmax << endl;
    for (int m = 0; m < xmax.size(); m++)
    {
//    D_A2_PixArrayA(m, Range(xmin(m)+8, xmax(m)+8)) = D_A2_PixArrayA(m, Range(xmin(m)+15, xmax(m)+15));
      D_A2_PixArrayA(m, Range(xmin(m), xmax(m))) = D_A2_PixArray(m, Range(xmin(m), xmax(m)));
    }
  }
  for (int o = 15; o < 16; o++)
  {
//  int xmin, xmax;
    Array<double, 2> D_A2_PixArray(p_testfits->GetPixArray());
    Array<double, 2> D_A2_PixArrayA(p_testfitsA->GetPixArray());
    if (p_testfits->GetNRows() != p_testfitsA->GetNRows() || p_testfits->GetNCols() != p_testfitsA->GetNCols())
    {
      cout << "MTestApp: ERROR: size of testfits != size of testfitsA => Returning FALSE" << endl;
      return false;
    }
//  for (int n = 3800; n < p_testfits->Get_XCenters().cols(); n++)
//  {
//    cout << "MTestApp: row = n = " << n << endl;
    Array<int, 1> xmin(p_testfits->Get_XCenters().cols());
    xmin = p_testfits->Get_XCenters()(o,Range::all()) + p_testfits->Get_Low()(o)+3;
//    cout << "MTestApp: xmin = " << xmin << endl;
    Array<int, 1> xmax(p_testfits->Get_XCenters().cols());
    xmax = p_testfits->Get_XCenters()(o,Range::all()) + p_testfits->Get_High()(o) + 12;
    cout << "MTestApp: xmin = " << xmin << ", xmax = " << xmax << endl;
    for (int m = 0; m < xmax.size(); m++)
    {
//    D_A2_PixArrayA(m, Range(xmin(m)+8, xmax(m)+8)) = D_A2_PixArrayA(m, Range(xmin(m)+15, xmax(m)+15));
      D_A2_PixArrayA(m, Range(xmin(m), xmax(m))) = D_A2_PixArray(m, Range(xmin(m), xmax(m)));
    }
  }
  for (int o = 16; o < 17; o++)
  {
//  int xmin, xmax;
    Array<double, 2> D_A2_PixArray(p_testfits->GetPixArray());
    Array<double, 2> D_A2_PixArrayA(p_testfitsA->GetPixArray());
    if (p_testfits->GetNRows() != p_testfitsA->GetNRows() || p_testfits->GetNCols() != p_testfitsA->GetNCols())
    {
      cout << "MTestApp: ERROR: size of testfits != size of testfitsA => Returning FALSE" << endl;
      return false;
    }
//  for (int n = 3800; n < p_testfits->Get_XCenters().cols(); n++)
//  {
//    cout << "MTestApp: row = n = " << n << endl;
    Array<int, 1> xmin(p_testfits->Get_XCenters().cols());
    xmin = p_testfits->Get_XCenters()(o,Range::all()) + p_testfits->Get_Low()(o)+5;
//    cout << "MTestApp: xmin = " << xmin << endl;
    Array<int, 1> xmax(p_testfits->Get_XCenters().cols());
    xmax = p_testfits->Get_XCenters()(o,Range::all()) + p_testfits->Get_High()(o) + 10;
    cout << "MTestApp: xmin = " << xmin << ", xmax = " << xmax << endl;
    for (int m = 0; m < xmax.size(); m++)
    {
//    D_A2_PixArrayA(m, Range(xmin(m)+8, xmax(m)+8)) = D_A2_PixArrayA(m, Range(xmin(m)+15, xmax(m)+15));
      D_A2_PixArrayA(m, Range(xmin(m), xmax(m))) = D_A2_PixArray(m, Range(xmin(m), xmax(m)));
    }
  }
  for (int o = 17; o < 18; o++)
  {
//  int xmin, xmax;
    Array<double, 2> D_A2_PixArray(p_testfits->GetPixArray());
    Array<double, 2> D_A2_PixArrayA(p_testfitsA->GetPixArray());
    if (p_testfits->GetNRows() != p_testfitsA->GetNRows() || p_testfits->GetNCols() != p_testfitsA->GetNCols())
    {
      cout << "MTestApp: ERROR: size of testfits != size of testfitsA => Returning FALSE" << endl;
      return false;
    }
//  for (int n = 3800; n < p_testfits->Get_XCenters().cols(); n++)
//  {
//    cout << "MTestApp: row = n = " << n << endl;
    Array<int, 1> xmin(p_testfits->Get_XCenters().cols());
    xmin = p_testfits->Get_XCenters()(o,Range::all()) + p_testfits->Get_Low()(o)+4;
//    cout << "MTestApp: xmin = " << xmin << endl;
    Array<int, 1> xmax(p_testfits->Get_XCenters().cols());
    xmax = p_testfits->Get_XCenters()(o,Range::all()) + p_testfits->Get_High()(o) + 9;
    cout << "MTestApp: xmin = " << xmin << ", xmax = " << xmax << endl;
    for (int m = 0; m < xmax.size(); m++)
    {
//    D_A2_PixArrayA(m, Range(xmin(m)+8, xmax(m)+8)) = D_A2_PixArrayA(m, Range(xmin(m)+15, xmax(m)+15));
      D_A2_PixArrayA(m, Range(xmin(m), xmax(m))) = D_A2_PixArray(m, Range(xmin(m), xmax(m)));
    }
  }
  for (int o = 18; o < 19; o++)
  {
//  int xmin, xmax;
    Array<double, 2> D_A2_PixArray(p_testfits->GetPixArray());
    Array<double, 2> D_A2_PixArrayA(p_testfitsA->GetPixArray());
    if (p_testfits->GetNRows() != p_testfitsA->GetNRows() || p_testfits->GetNCols() != p_testfitsA->GetNCols())
    {
      cout << "MTestApp: ERROR: size of testfits != size of testfitsA => Returning FALSE" << endl;
      return false;
    }
//  for (int n = 3800; n < p_testfits->Get_XCenters().cols(); n++)
//  {
//    cout << "MTestApp: row = n = " << n << endl;
    Array<int, 1> xmin(p_testfits->Get_XCenters().cols());
    xmin = p_testfits->Get_XCenters()(o,Range::all()) + p_testfits->Get_Low()(o)+3;
//    cout << "MTestApp: xmin = " << xmin << endl;
    Array<int, 1> xmax(p_testfits->Get_XCenters().cols());
    xmax = p_testfits->Get_XCenters()(o,Range::all()) + p_testfits->Get_High()(o) + 9;
    cout << "MTestApp: xmin = " << xmin << ", xmax = " << xmax << endl;
    for (int m = 0; m < xmax.size(); m++)
    {
//    D_A2_PixArrayA(m, Range(xmin(m)+8, xmax(m)+8)) = D_A2_PixArrayA(m, Range(xmin(m)+15, xmax(m)+15));
      D_A2_PixArrayA(m, Range(xmin(m), xmax(m))) = D_A2_PixArray(m, Range(xmin(m), xmax(m)));
    }
  }
  for (int o = 19; o < 20; o++)
  {
//  int xmin, xmax;
    Array<double, 2> D_A2_PixArray(p_testfits->GetPixArray());
    Array<double, 2> D_A2_PixArrayA(p_testfitsA->GetPixArray());
    if (p_testfits->GetNRows() != p_testfitsA->GetNRows() || p_testfits->GetNCols() != p_testfitsA->GetNCols())
    {
      cout << "MTestApp: ERROR: size of testfits != size of testfitsA => Returning FALSE" << endl;
      return false;
    }
//  for (int n = 3800; n < p_testfits->Get_XCenters().cols(); n++)
//  {
//    cout << "MTestApp: row = n = " << n << endl;
    Array<int, 1> xmin(p_testfits->Get_XCenters().cols());
    xmin = p_testfits->Get_XCenters()(o,Range::all()) + p_testfits->Get_Low()(o)+2;
//    cout << "MTestApp: xmin = " << xmin << endl;
    Array<int, 1> xmax(p_testfits->Get_XCenters().cols());
    xmax = p_testfits->Get_XCenters()(o,Range::all()) + p_testfits->Get_High()(o) + 9;
    cout << "MTestApp: xmin = " << xmin << ", xmax = " << xmax << endl;
    for (int m = 0; m < xmax.size(); m++)
    {
//    D_A2_PixArrayA(m, Range(xmin(m)+8, xmax(m)+8)) = D_A2_PixArrayA(m, Range(xmin(m)+15, xmax(m)+15));
      D_A2_PixArrayA(m, Range(xmin(m), xmax(m))) = D_A2_PixArray(m, Range(xmin(m), xmax(m)));
    }
  }
  for (int o = 20; o < 21; o++)
  {
//  int xmin, xmax;
    Array<double, 2> D_A2_PixArray(p_testfits->GetPixArray());
    Array<double, 2> D_A2_PixArrayA(p_testfitsA->GetPixArray());
    if (p_testfits->GetNRows() != p_testfitsA->GetNRows() || p_testfits->GetNCols() != p_testfitsA->GetNCols())
    {
      cout << "MTestApp: ERROR: size of testfits != size of testfitsA => Returning FALSE" << endl;
      return false;
    }
//  for (int n = 3800; n < p_testfits->Get_XCenters().cols(); n++)
//  {
//    cout << "MTestApp: row = n = " << n << endl;
    Array<int, 1> xmin(p_testfits->Get_XCenters().cols());
    xmin = p_testfits->Get_XCenters()(o,Range::all()) + p_testfits->Get_Low()(o) - 2;
//    cout << "MTestApp: xmin = " << xmin << endl;
    Array<int, 1> xmax(p_testfits->Get_XCenters().cols());
    xmax = p_testfits->Get_XCenters()(o,Range::all()) + p_testfits->Get_High()(o) + 3;
    cout << "MTestApp: xmin = " << xmin << ", xmax = " << xmax << endl;
    for (int m = 0; m < xmax.size(); m++)
    {
//    D_A2_PixArrayA(m, Range(xmin(m)+8, xmax(m)+8)) = D_A2_PixArrayA(m, Range(xmin(m)+15, xmax(m)+15));
      D_A2_PixArrayA(m, Range(xmin(m), xmax(m))) = D_A2_PixArray(m, Range(xmin(m), xmax(m)));
    }
  }
  for (int o = 21; o < 22; o++)
  {
//  int xmin, xmax;
    Array<double, 2> D_A2_PixArray(p_testfits->GetPixArray());
    Array<double, 2> D_A2_PixArrayA(p_testfitsA->GetPixArray());
    if (p_testfits->GetNRows() != p_testfitsA->GetNRows() || p_testfits->GetNCols() != p_testfitsA->GetNCols())
    {
      cout << "MTestApp: ERROR: size of testfits != size of testfitsA => Returning FALSE" << endl;
      return false;
    }
//  for (int n = 3800; n < p_testfits->Get_XCenters().cols(); n++)
//  {
//    cout << "MTestApp: row = n = " << n << endl;
    Array<int, 1> xmin(p_testfits->Get_XCenters().cols());
    xmin = p_testfits->Get_XCenters()(o,Range::all()) + p_testfits->Get_Low()(o)-2;
//    cout << "MTestApp: xmin = " << xmin << endl;
    Array<int, 1> xmax(p_testfits->Get_XCenters().cols());
    xmax = p_testfits->Get_XCenters()(o,Range::all()) + p_testfits->Get_High()(o) + 4;
    cout << "MTestApp: xmin = " << xmin << ", xmax = " << xmax << endl;
    for (int m = 0; m < xmax.size(); m++)
    {
//    D_A2_PixArrayA(m, Range(xmin(m)+8, xmax(m)+8)) = D_A2_PixArrayA(m, Range(xmin(m)+15, xmax(m)+15));
      D_A2_PixArrayA(m, Range(xmin(m), xmax(m))) = D_A2_PixArray(m, Range(xmin(m), xmax(m)));
    }
  }
  for (int o = 22; o < 23; o++)
  {
//  int xmin, xmax;
    Array<double, 2> D_A2_PixArray(p_testfits->GetPixArray());
    Array<double, 2> D_A2_PixArrayA(p_testfitsA->GetPixArray());
    if (p_testfits->GetNRows() != p_testfitsA->GetNRows() || p_testfits->GetNCols() != p_testfitsA->GetNCols())
    {
      cout << "MTestApp: ERROR: size of testfits != size of testfitsA => Returning FALSE" << endl;
      return false;
    }
//  for (int n = 3800; n < p_testfits->Get_XCenters().cols(); n++)
//  {
//    cout << "MTestApp: row = n = " << n << endl;
    Array<int, 1> xmin(p_testfits->Get_XCenters().cols());
    xmin = p_testfits->Get_XCenters()(o,Range::all()) + p_testfits->Get_Low()(o)-4;
//    cout << "MTestApp: xmin = " << xmin << endl;
    Array<int, 1> xmax(p_testfits->Get_XCenters().cols());
    xmax = p_testfits->Get_XCenters()(o,Range::all()) + p_testfits->Get_High()(o) + 3;
    cout << "MTestApp: xmin = " << xmin << ", xmax = " << xmax << endl;
    for (int m = 0; m < xmax.size(); m++)
    {
//    D_A2_PixArrayA(m, Range(xmin(m)+8, xmax(m)+8)) = D_A2_PixArrayA(m, Range(xmin(m)+15, xmax(m)+15));
      D_A2_PixArrayA(m, Range(xmin(m), xmax(m))) = D_A2_PixArray(m, Range(xmin(m), xmax(m)));
    }
  }
  if (!p_testfitsA->WriteArray())
  {
    cout << "MTestApp: p_testfitsA->WriteArray() returned FALSE => Returning FALSE" << endl;
    return false;
  }
*/
  MTestApp::P_Log = new ofstream("test/logfile_TestApp.log");


  CString ClassName;
  ClassName.Set("all");
  printf("MTestApp::main: ClassName = <%s>\n", ClassName.Get());
  /*  CFormatedString *P_MyString = new CFormatedString(2,2);


    cout << "CFits.main: MyString = <";
    P_MyString->Show(cout);
    cout << ">" << endl;
  */
  //  MTestBasis::TestBasis();
  //  MTestApp::Test();

  //  printf("CFits.main: Hello, world!\n");// << endl;

  /*  string className;

    cout << " " << endl;
    cout << " TestApplication" << endl;
    cout << " ===============" << endl;
    cout << " " << endl;
    cout << " Bitte geben Sie den Namen der Klasse, welche Sie testen" << endl;
    cout << " moechten, ein (z.B. CDate, all fuer alle Klassen): " << endl;
    cin >> className;
    cout << " " << endl;
    cout << " Sie haben " << className << " gewaehlt." << endl;
    cout << " " << endl;
  */
  if(ClassName.EqualValue(CString("CFormatedString")) ||
     ClassName.EqualValue(CString("CFits")) ||
     ClassName.EqualValue(CString("CString")))
  {/* ||
, , , className == "CDate" ||
, , , className == "CPerson" ||
, , , className == "CAmount" ||
, , , className == "CAccountType" ||
, , , className == "CAccount" ||
, , , className == "CGiro" ||
, , , className == "CStudentGiro" ||
, , , className == "CSavings" ||
, , , className == "CRegister"){
,   */
    Run(ClassName);

  }
  else if (ClassName.EqualValue(CString("all")))
  {
    CString tmpstr("CString");
    Run(tmpstr);
    cout << " " << endl;
//    Run(*(new CString("CFormatedString")));
//    cout << " " << endl;
    tmpstr.Set("CFits");
    Run(tmpstr);
    cout << " " << endl;
    /*, Run("CDate");
, , , cout << " " << endl;
, , , Run("CPerson");
, , , cout << " " << endl;
, , , Run("CAmount");
, , , cout << " " << endl;
, , , Run("CAccountType");
, , , cout << " " << endl;
, , , Run("CAccount");
, , , cout << " " << endl;
, , , Run("CGiro");
, , , cout << " " << endl;
, , , Run("CStudentGiro");
, , , cout << " " << endl;
, , , Run("CSavings");
, , , cout << " " << endl;
, , Run("CRegister");*/
  }
  else
    cout << " FEHLER: " << ClassName <<  " ist kein gueltiger Klassenname!!!" << endl;

//  cout << "MTestApp::main( ): Starting CVektor::Run()" << endl;
//  RunV();
  delete MTestApp::P_Log;
  return 0;
}

/************************************************************************/
