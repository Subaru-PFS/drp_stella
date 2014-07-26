/*
author: Andreas Ritter
created: 01/12/2007
last edited: 01/12/2007
compiler: g++ 4.0
basis machine: Ubuntu Linux 6.06
*/

#include "CString.h"

CString::CString()
    : CAny()
{
//#ifdef __DEBUG_STRING__
//  cout << "CString::CString(): Started" << endl;
//#endif
  this->P_String = new char[STRLEN];
  this->P_String[0] = '\0';
//#ifdef __DEBUG_STRING__
//  cout << "CString::CString(): Memory for P_String allocated" << endl;
//#endif
  //this->ClassName = new char[STRLEN];
  this->ClassName = strdup("CString");
//#ifdef __DEBUG_STRING__
//  cout << "CString::CString(): ClassName set to <" << this->ClassName << ">" << endl;
//#endif
}

/***********************************************************************/

CString::CString(const string str)
  : CAny()
{
#ifdef __DEBUG_STRING__
    cout << "CString::CString(const string str) Started" << endl;
#endif
/*  if (&str == NULL)
  {
#ifdef __DEBUG_STRING__
  cout << "CString::CString(const string str) str == NULL => Starting CString()" << endl;
#endif
  CString();
}*/
  cout << "CString::CString: str = " << str << endl;
  if (this->LastCharPosInString(str, '\0') < 1)
  {
#ifdef __DEBUG_STRING__
    cout << "CString::CString(const string str=" << str << ") LastCharPosInString( str, 0) returned " << this->LastCharPosInString(str, '\0') << " => Starting CString()" << endl;
#endif
    CString();
  }

  int i = 0;
  bool EndFound = false;
  this->P_String = new char[STRLEN];
  this->P_String[0] = '\0';
#ifdef __DEBUG_STRING__
  cout << "CString::CString(const string str=" << str << ") Memory for P_String allocated" << endl;
#endif
  //this->ClassName = new char[STRLEN];
  this->ClassName = strdup("CString");
#ifdef __DEBUG_STRING__
  cout << "CString::CString(const string str=" << str << ") ClassName set to <" << this->ClassName << ">" << endl;
#endif
  while (!EndFound && i < STRLEN)
  {
#ifdef __DEBUG_STRING__
    cout << "CString::CString(const string str=" << str << ") while: str[i = " << i << "] = <" << str[i] << ">" << endl;
#endif
    this->P_String[i] = str[i];
#ifdef __DEBUG_STRING__
    cout << "CString::CString(const string str=" << str << ") while: P_String = <" << P_String << ">" << endl;
#endif
    if (P_String[i] == '\0')
    {
      EndFound = true;
#ifdef __DEBUG_STRING__
      cout << "CString::CString(const string str=" << str << ") End Found" << endl;
#endif
    }
    i++;
  }
}

/***********************************************************************/

CString::CString(const char* p_char)
    : CAny()
{
//#ifdef __DEBUG_STRING__
//  cout << "CString::CString(const char* p_char=" << p_char << ") started" << endl;
//#endif
  int i = 0;
  bool EndFound = false;
  if (p_char == NULL)
  {
#ifdef __DEBUG_STRING__
    cout << "CString::CString(const char* p_char) p_char == NULL => Starting CString::CString()" << endl;
#endif
    CString();
  }
  if (strlen(p_char) < 1)
  {
//#ifdef __DEBUG_STRING__
//    cout << "CString::CString(const char* p_char=" << p_char << ") strlen(p_char)=" << strlen(p_char) << " => Starting CString::CString()" << endl;
//#endif
    CString();
  }
  this->P_String = new char[STRLEN];
  this->P_String[0] = '\0';
//#ifdef __DEBUG_STRING__
//  cout << "CString::CString(const char* p_char=" << p_char << ") Memory for P_String allocated" << endl;
//#endif

  while (!EndFound && i < STRLEN)
  {
//#ifdef __DEBUG_STRING__
//    cout << "CString::CString(const string str=" << str << ") while: str[i = " << i << "] = <" << str[i] << ">" << endl;
//#endif
    this->P_String[i] = p_char[i];
//#ifdef __DEBUG_STRING__
//    cout << "CString::CString(const string str=" << str << ") while: P_String = <" << P_String << ">" << endl;
//#endif
    if (P_String[i] == '\0')
    {
      EndFound = true;
//#ifdef __DEBUG_STRING__
//      cout << "CString::CString(const string str=" << str << ") End Found" << endl;
//#endif
    }
    i++;
  }
  /*do
  {
    this->P_String[i] = p_char[i];
//#ifdef __DEBUG_STRING__
//    cout << "CString::CString(const char* p_char=" << p_char << ") P_String = <" << P_String << ">" << endl;
//#endif
    i++;
} while(P_String[i-1] != '\0');*/
//#ifdef __DEBUG_STRING__
//  cout << "CString::CString(const char* p_char=" << p_char << ") P_String = <" << P_String << ">" << endl;
//#endif
  //this->ClassName = new char[STRLEN];
  this->ClassName = strdup("CString");
//#ifdef __DEBUG_STRING__
//  cout << "CString::CString(const char* p_char=" << p_char << ") ClassName = <" << this->ClassName << ">" << endl;
  //#endif
}

/***********************************************************************

CString::CString(const char[STRLEN] charr)
{
  if (charr == NULL)
    CString();

  bool EndFound = false;
  P_String = (char*)malloc(sizeof(char) * STRLEN);
  ClassName = "CString";

  for (int i = 0; i < STRLEN; i++)
  {
    P_String[i] = charr[i];
    if (P_String[i] == '\0')
      EndFound = true;
  }
  if (!EndFound)
    P_String[STRLEN-1] = '\0';
}

/***********************************************************************/

CString::CString(const CString &cstr)
    : CAny()
{
#ifdef __DEBUG_STRING__
    cout << "CString::CString(const CString &cstr) Started" << endl;
#endif
/*  if ((&cstr) == NULL)
  {
#ifdef __DEBUG_STRING__
    cout << "CString::CString(const CString &cstr) &cstr == NULL => Starting CString()" << endl;
#endif
    CString();
}*/
//#ifdef __DEBUG_STRING__
//    cout << "CString::CString(const CString &cstr) starting cstr->ClassInvariant()" << endl;
//#endif
  if (!(cstr.ClassInvariant()))
  {
//#ifdef __DEBUG_STRING__
//    cout << "CString::CString(const CString &cstr) cstr->ClassInvariant() returned FALSE => Starting CString()" << endl;
//#endif
    CString();
  }
  this->P_String = new char[STRLEN];
  this->P_String[0] = '\0';
//#ifdef __DEBUG_STRING__
//  cout << "CString::CString(const CString &cstr=" << cstr << ") Memory for P_String allocated" << endl;
//#endif
  int i = 0;
  bool EndFound = false;
  while (!EndFound && i < STRLEN)
  {
    this->P_String[i] = *((*(const_cast<CString*>(&cstr)))[i]);
//#ifdef __DEBUG_STRING__
//    cout << "CString::CString(const CString &cstr=" << cstr << ") P_String = <" << this->P_String << ">" << endl;
//#endif
    if (this->P_String[i] == '\0')
    {
      EndFound = true;
//#ifdef __DEBUG_STRING__
//      cout << "CString::CString(const CString &cstr=" << cstr <<") End Found" << endl;
//#endif
    }
    i++;
  }
  //this->ClassName = new char[STRLEN];
  this->ClassName = strdup("CString");
//#ifdef __DEBUG_STRING__
//  cout << "CString::CString(const CString &cstr=" << cstr << ") ClassName set to <" << this->ClassName << ">" << endl;
//#endif
}

/***********************************************************************/

CString::~CString()
{
  delete[] this->P_String;
//  if (this->ClassName != NULL)
  free(this->ClassName);
}

/***********************************************************************/

bool CString::ClassInvariant() const
{
#ifdef __DEBUG_STRING__
    cout << "CString::ClassInvariant(this->P_String = <" << this->P_String << ">) Started" << endl;
#endif
  if (this->CharPos('\0') < 0)
  {
#ifdef __DEBUG_STRING__
    cout << "CString::ClassInvariant(this->P_String = <" << this->P_String << ">): CharPos(\\0) returned -1 => Returning FALSE!" << endl;
#endif
    return false;
  }
  if (strlen(this->P_String) < 0 || strlen(this->P_String) > STRLEN-1)
  {
#ifdef __DEBUG_STRING__
    cout << "CString::ClassInvariant(this->P_String = <" << this->P_String << ">): strlen(this->P_String=" << P_String << ") returned " << strlen(this->P_String) << " => Returning FALSE!" << endl;
#endif
    return false;
  }
#ifdef __DEBUG_STRING__
    cout << "CString::ClassInvariant(this->P_String = <" << this->P_String << ">): Returning TRUE!" << endl;
#endif
  return true;
}

/***********************************************************************/

bool CString::EqualValue(const CAny &any) const
{
#ifdef __DEBUG_STRING__
    cout << "CString::EqualValue( any = <" << any << "> ): this->P_String = <" << this->P_String << ">): Started" << endl;
#endif
  if ((&any) == NULL)
  {
#ifdef __DEBUG_STRING__
    cout << "CString::EqualValue( any ): this->P_String = <" << this->P_String << ">): any == NULL => Returning FALSE" << endl;
#endif
    return false;
  }
  if (this->Equal(any))
    return false;
  const CString *p_cstr = dynamic_cast<const CString*>(&any);
  if (!ClassInvariant() || !(p_cstr->ClassInvariant()))
  {
#ifdef __DEBUG_STRING__
    cout << "CString::EqualValue( any = <" << any << ">): this->P_String = <" << this->P_String << ">): this->ClassInvariant() or p_cstr(=" << p_cstr << ") returned FALSE => Returning FALSE" << endl;
#endif
    return false;
  }
  bool EndFound = false;
  int i = 0;

  while(!EndFound && i < STRLEN)
  {
    if (P_String[i] != *((*(const_cast<CString*>(p_cstr)))[i]))
    {
#ifdef __DEBUG_STRING__
    cout << "CString::EqualValue( any ): this->P_String = <" << this->P_String << ">): P_String[i=" << i << "] = " << P_String[i] << " != p_cstr[i]=" << *((*(const_cast<CString*>(p_cstr)))[i]) << " => Returning FALSE" << endl;
#endif
      return false;
    }
    if (P_String[i] == '\0')
    {
#ifdef __DEBUG_STRING__
      cout << "CString::EqualValue( any ): this->P_String = <" << this->P_String << ">): EndFound == true" << endl;
#endif
      EndFound = true;
    }
    i++;
  }
#ifdef __DEBUG_STRING__
    cout << "CString::EqualValue( any ): this->P_String = <" << this->P_String << ">): Returning TRUE" << endl;
#endif

  return true;
}

/***********************************************************************/

bool CString::Copy(const CAny &any)
{
  const CString *cstr = dynamic_cast<const CString*>(&any);
  if (cstr == NULL)
    return false;
  if (Equal(*cstr))
    return false;
  if (!(cstr->ClassInvariant()))
    return false;
  this->ClassName = strdup("CString");
  *this = *cstr;
  return true;
}

/***********************************************************************/

CString& CString::operator=(const CString &cstr)
{
  int i = 0;
  bool EndFound = false;
  char* p_TempStr = cstr.Get();
#ifdef __DEBUG_STRING__
  cout << "CString::operator =( cstr = <" << cstr << ">): Start: this = <" << *this << ">" << endl;
#endif
  while(!EndFound && i < STRLEN && i < cstr.GetLength())
  {
#ifdef __DEBUG_STRING__
    cout << "CString::operator =( cstr = <" << cstr << ">): p_TempStr[i=" << i << "] = <" << p_TempStr[i] << ">" << endl;
#endif
    this->P_String[i] = p_TempStr[i];
    if (this->P_String[i] == '\0' || this->P_String[i] == '\n')
    {
#ifdef __DEBUG_STRING__
      cout << "CString::operator =( cstr = <" << cstr << ">): EndFound" << endl;
#endif
      EndFound = true;
    }
    i++;
  }
  if (!EndFound && i < STRLEN)
    this->P_String[i] = '\0';
  if (!EndFound && i >= STRLEN)
    this->P_String[STRLEN-1] = '\0';
#ifdef __DEBUG_STRING__
  cout << "CString::operator =( cstr = <" << cstr << ">): Returning this = <" << *this << ">" << endl;
#endif
  return *this;
}

/***********************************************************************/

bool CString::operator==(const CAny &any) const
{
  return (this->EqualValue( any ));
}

/***********************************************************************/

bool CString::operator!=(const CAny &any) const
{
  return (!this->EqualValue( any ));
}

/********************************************************************/

CString& CString::operator+(const CString &cstr)
{
#ifdef __DEBUG_STRING__
  cout << "CString::this(=" << *this << ")->operator+(cstr = " << cstr << ") started" << endl;
#endif
  int i;
  int oldlength = this->GetLength();
  CString CS(*this);
#ifdef __DEBUG_STRING__
  cout << "CString::this(=" << *this << ")->operator+(cstr = " << cstr << ") oldlength = " << oldlength << endl;
  cout << "CString::this(=" << *this << ")->operator+(cstr = " << cstr << ") cstr.GetLength() = " << cstr.GetLength() << endl;
#endif
  for (i = 0; i < cstr.GetLength(); i++)
  {
#ifdef __DEBUG_STRING__
    cout << "CString::this(=" << *this << ")->operator+(cstr = " << cstr << ") cstr[i=" << i << "] = <" << *((*(const_cast<CString*>(&cstr)))[i]) << ">, P_String = <" << *(this->P_String) << ">" << endl;
#endif
    this->P_String[oldlength + i] = (*((*(const_cast<CString*>(&cstr)))[i]));
    this->P_String[oldlength + i + 1] = '\0';
#ifdef __DEBUG_STRING__
    cout << "CString::this(=" << *this << ")->operator+(cstr = " << cstr << ") P_String = <" << *(this->P_String) << ">" << endl;
#endif
    if ( i == STRLEN-2 )
    {
#ifdef __DEBUG_STRING__
      cout << "CString::this(=" << *this << ")->operator+(cstr = " << cstr << ") i == STRLEN-1, P_String = <" << *(this->P_String) << ">" << endl;
#endif
      i = strlen(this->P_String) + strlen(cstr.Get());
      this->P_String[STRLEN-1] = '\0';
    }
  }
#ifdef __DEBUG_STRING__
  cout << "CString::this(=" << *this << ")->operator+(cstr = " << cstr << ") Returning <" << *this << ">" << endl;
#endif
  CString *P_CS_Out = new CString(*this);
  *this = CS;
  return (*P_CS_Out);
}

/********************************************************************/

CString& CString::operator+(const char cstr)
{
#ifdef __DEBUG_STRING__
  cout << "CString::this(=" << *this << ")->operator+(char cstr = " << cstr << ") started" << endl;
#endif
  CString CS(*this);
  int oldlength = this->GetLength();
#ifdef __DEBUG_STRING__
  cout << "CString::this(=" << *this << ")->operator+(char cstr = " << cstr << "), this = <" << *this << ">" << endl;
#endif
  this->P_String[oldlength] = cstr;
  this->P_String[oldlength + 1] = '\0';
#ifdef __DEBUG_STRING__
  cout << "CString::this(=" << *this << ")->operator+(char cstr = " << cstr << ") returning <" << *this << ">" << endl;
#endif
  CString *P_CS_Out = new CString(*this);
  *this = CS;
  return (*P_CS_Out);
}

/** Operator "+="
      task   : Concatenates 'this' and 'CS_Str'.
      require: .
      ensure : .
**/
CString& CString::operator+=(const CString &CS_Str)
{
  this->Add(CS_Str);
  return (*this);
}

/** Operator "+="
      task   : Concatenates 'this' and 'cstr'.
      require: .
      ensure : .
 **/
CString& CString::operator+=(const char cstr)
{
  this->Add(CString(&cstr));
  return (*this);
}

/********************************************************************/

char* CString::operator[](int i)
{
  return (&(this->P_String[i]));
}

/***********************************************************************/

bool CString::Set( const char* p_str)
{
#ifdef __DEBUG_STRING_SET__
  cout << "CString::Set: p_str = <" << p_str << ">" << endl;
#endif
  if (p_str == NULL)
    return false;
  if (strlen(p_str) < 1 || strlen(p_str) > STRLEN){
    cout << "CString::Set: ERROR: strlen(p_str=" << strlen(p_str) << ") < 1 || strlen(p_str) > STRLEN" << endl;
    return false;
  }
  for (int i = 0; i < strlen(p_str); i++){
#ifdef __DEBUG_STRING_SET__
    cout << "CString::Set: setting P_String(i=" << i << ") to <" << p_str[i] << ">" << endl;
#endif
    this->P_String[i] = p_str[i];
#ifdef __DEBUG_STRING_SET__
    cout << "CString::Set: P_String(i=" << i << ") set to <" << P_String[i] << ">" << endl;
#endif
  }
  this->P_String[strlen(p_str)] = '\0';
  return (strcmp(this->P_String, p_str) == 0);
}

/***********************************************************************/

bool CString::Set( const string str)
{
  int i;
  if (&str == NULL)
    return false;
  if (strlen(StringToPChar(str)) < 1 || strlen(StringToPChar(str)) > STRLEN-1)
    return false;
  for (i = 0; i < strlen(StringToPChar(str)); i++)
    this->P_String[i] = str[i];
  P_String[i+1] = '\0';
  return (strcmp(this->P_String, StringToPChar(str)) == 0);
}

bool CString::Set(const CString &C_Str){
  return this->Set(C_Str.Get());
}

/***********************************************************************

bool CString::Set( const char charr[STRLEN])
{
  int i;
  if (charposincharr( charr, '\0') < 0)
    return false;
  for (i = 0; i < STRLEN; i++)
    this->P_String[i] = p_str[i];
  return (strcmp(this->P_String, p_str) == 0);
}

/***********************************************************************/

char* CString::Get() const
{
  return(this->P_String);
}

/***********************************************************************/

std::string CString::GetString() const
{
  CString *p_tempStr = new CString(*this);
  return(this->PCharToString(p_tempStr->Get()));
}

/***********************************************************************/

char* CString::GetPChar() const
{
  return(this->Get());
}

/**
  Returns number of characters in this->P_String before ending '\0'.
**/

int CString::GetLength() const
{
  int i;
  for (i = 0; i < STRLEN; i++)
  {
    if (this->P_String[i] == '\0')
      return i;
  }
  return(0);
}

/** *********************************************************************/

/**
function char* StrTrim(const char Char: in, int Mode: in)
Mode == 0: Remove starting occurences of Char from this
Mode == 1: Remove trailing occurences of Char from this
Mode == 2: Remove starting and trailing occurences of Char from this
Removes empty spaces of <this.P_String>.
 **/
void CString::TrimChar(const char Char, int Mode)
{
  int i;
  int NewLength = 0;
  int Before = 1;
  //  int Behind = 1;
  CString *P_NewStr = new CString(" ");
  int Size = this->GetLength();

#ifdef __DEBUG_STRING_TRIMCHAR__
  cout << "CString::TrimChar: this->P_String = " << P_String << ", this->GetLength() = " << this->GetLength() << ", Size = " << Size << ", Mode = " << Mode << endl;
#endif
  if (Mode == 0 || Mode == 2)
  {
#ifdef __DEBUG_STRING_TRIMCHAR__
    cout << "CString::TrimChar: Mode == 0 || 2, P_NewStr(=" << *P_NewStr << ")->GetLength() = " << P_NewStr->GetLength() << endl;
#endif
    for (i = 0; i < Size; i++)
    {
#ifdef __DEBUG_STRING_TRIMCHAR__
      cout << "CString::TrimChar: i = " << i << ": this->P_String[i] = <" << P_String[i] << ">" << endl;
#endif
      if (Before == 1 && this->P_String[i] == Char)// || (Before == 0 && Mode < 2))
      {
#ifdef __DEBUG_STRING_TRIMCHAR__
        cout << "CString::TrimChar: Skipping Character this->P_String[i=" << i <<"] = <" << P_String[i] << ">" << endl;
#endif
      }
      else
      {
        Before = 0;
//        if (NewLength == 0)
//          P_NewStr->Set(this->P_String[i]);
//        else
        *((*P_NewStr)[NewLength]) = this->P_String[i];
        NewLength++;
        *((*P_NewStr)[NewLength]) = '\0';
#ifdef __DEBUG_STRING_TRIMCHAR__
        cout << "CString::TrimChar: NewLength = " << NewLength << ": P_NewStr = <" << *P_NewStr << ">" << endl;
#endif

      }
      if (this->P_String[i] == '\0')
        i = Size;
    }
  }
  else
  {
    P_NewStr->Copy( *this );
  }
  if (Mode == 1 || Mode == 2)
  {
#ifdef __DEBUG_STRING_TRIMCHAR__
    cout << "CString::TrimChar: Mode == 1 || 2, P_NewStr(=" << *P_NewStr << ")->GetLength() = " << P_NewStr->GetLength() << endl;
#endif
    for (i = P_NewStr->GetLength() - 1; i > 0; i--)
    {
#ifdef __DEBUG_STRING_TRIMCHAR__
      cout << "CString::TrimChar: P_NewStr[i=" << i << "] = <" << *((*P_NewStr)[i]) << ">" << endl;
#endif
      if (*((*P_NewStr)[i]) == Char)
      {
#ifdef __DEBUG_STRING_TRIMCHAR__
        cout << "CString::TrimChar: Removing Character this->P_String[i=" << i << "] = <" << P_String[i] << ">" << endl;
#endif
        *((*P_NewStr)[i]) = '\0';
        i = 0;
      }
      else if (*((*P_NewStr)[i]) == '\0' || *((*P_NewStr)[i]) == '\n')
      {
        /// do nothing
      }
      else
      {
        i = 0;
      }
#ifdef __DEBUG_STRING_TRIMCHAR__
      cout << "CString::TrimChar: i = " << i << ": P_NewStr = <" << P_NewStr->Get() << ">" << endl;
#endif

    }
  }

  this->Copy(*P_NewStr);
  delete P_NewStr;
  return;
}

/** *********************************************************************/

/**
function char* StrTrim(const char Char: in, int Mode: in)
Mode == 0: Remove starting occurences of Char from this
Mode == 1: Remove trailing occurences of Char from this
Mode == 2: Remove starting and trailing occurences of Char from this
Removes empty spaces of <this.P_String>.
 **/
void CString::TrimChar(const CString &CS_Char, int Mode)
{
  string str = CS_Char.Get();
  char chr = str[0];
#ifdef __DEBUG_STRING_TRIMCHAR__
  cout << "CString::TrimChar(CString, int): chr set to '" << chr << "'" << endl;
#endif
  return this->TrimChar(chr, Mode);
}

/**
function int charrcat{CHar Array CAT}(char inoutarr[STRLEN]: inout, char *strtoapp: in, int OLDLENGTHofinoutarr: in)
Appends strtoapp at inoutarr[oldlength] and a '\0' behind and returns position of '\0'.
 **/
void CString::Trim(int Mode)
{
  this->TrimChar(' ', Mode);
  return;
}

/**
function int charrcat{CHar Array CAT}(char inoutarr[STRLEN]: inout, char *strtoapp: in, int OLDLENGTHofinoutarr: in)
Appends strtoapp at inoutarr[oldlength] and a '\0' behind and returns position of '\0'.
 **/
CString* CString::SubString(int Start, int End) const
{
  int i, Pos, Length;
  char* P_OutStr;

  if (Start < 0)
    Start = 0;

  P_OutStr = new char[STRLEN];//End - Start + 2];

  Pos = 0;
  for (i = Start; i <= End; i++)
  {
    P_OutStr[Pos] = this->P_String[i];
    Pos++;
    if (this->P_String[i] == '\0')
      i = End;
  }
  P_OutStr[Pos] = '\0';
  CString *P_CS_Out = new CString(P_OutStr);
  delete[] P_OutStr;
  return P_CS_Out;
}

/**
function int charrcat{CHar Array CAT}(char inoutarr[STRLEN]: inout, char *strtoapp: in, int OLDLENGTHofinoutarr: in)
Appends strtoapp at inoutarr[oldlength] and a '\0' behind and returns position of '\0'.
 **/
CString* CString::SubString(int Start) const
{
  return (this->SubString(Start, this->GetLength()-1));
}

/**
function int charposincharrfrom{CHARacter POSition IN CHar ARR}(char inarr[STRLEN]: in, char lookfor: in, int start: in)
Returns first position of character 'in' in 'inarr', beginning at position 'start', if found, else '-1'
 **/
int CString::CharPosInCharArrFrom(const char inarr[STRLEN], const char lookfor, int start) const
{
  int pos = start;
  do
  {
    if (inarr[pos] == lookfor)
    {
#ifdef __DEBUG_STRING__
      cout << "CString::CharPosInCharArrFrom: inarr[pos=" << pos << "] == lookfor(=" << lookfor << ") returned TRUE" << endl;
#endif
      break;
    }
    if (inarr[pos] == '\0')
      return -1;
    pos++;
  } while (pos < STRLEN);
  if (pos == STRLEN)
    pos = -1;
#ifdef __DEBUG_STRING__
  cout << "CString::CharPosInCharArrFrom: returning pos = " <<  pos << endl;
#endif
  return pos;
}

/**
function int charposincharr{CHARacter POSition IN CHar ARR}(char inarr[STRLEN]: in, char lookfor: in, int start: in)
Returns first position of character 'in' in 'inarr', if found, else '-1'
 **/
int CString::CharPosInCharArr(const char inarr[STRLEN], const char lookfor) const
{
  return CharPosInCharArrFrom(inarr, lookfor, 0);
}

/**
function int lastcharposincharr{LAST CHARacter POSition IN CHar ARR}(char inarr[STRLEN]: in, char lookfor: in)
Returns last position of character 'in' in 'inarr', if found, else '-1'
 **/
int CString::LastCharPosInCharArr(const char inarr[STRLEN], const char lookfor) const
{
  int pos = 0;
  int lastpos = -1;
#ifdef __DEBUG_STRING__
  cout << "CString::LastCharPosInCharArr: function started" << endl;
#endif
  do
  {
    pos = this->CharPosInCharArrFrom(inarr, lookfor, pos+1);
#ifdef __DEBUG_STRING__
    cout << "CString::LastCharPosInCharArr: CharPosInCharArrFrom(..) returned pos=" << pos << endl;
#endif
    if (pos >= 0)
      lastpos = pos;
  }
  while (pos >= 0);
#ifdef __DEBUG_STRING__
  cout << "CString::LastCharPosInCharArr: returning lastpos = " << lastpos << endl;
#endif
  return lastpos;
}

/**
  function int CharrPosInStr(char *P_InStr: in, const char LookFor: in)
  Returns first position of String 'LookFor' in 'P_InStr', looking from position Start, if found, else '-1'
 **/
int CString::CharPosFrom(const char LookFor, int Start) const
{
  char StrArr[STRLEN];
  strcpy(StrArr, this->P_String);
  return (this->CharPosInCharArrFrom(StrArr, LookFor, Start));
}

/**
  function int CharrPosInStr(char *P_InStr: in, const char LookFor: in)
  Returns first position of String 'P_LookFor' in 'P_InStr', if found, else '-1'
 **/
int CString::CharPos(const char LookFor) const
{
  char StrArr[STRLEN];
  strcpy(StrArr, this->P_String);
  return (this->CharPosInCharArr(StrArr, LookFor));
}

/**
function int strposincharrfrom{LAST STRing POSition IN CHar ARR}(char inarr[STRLEN]: in, char* lookfor: in, int startpos)
Returns last position of character 'in' in 'inarr', if found, else '-1'
 **/
int CString::StrPosInCharArrFrom(const char inarr[STRLEN], const char* lookfor, int startpos) const
{
  int i;
  int pos = startpos;
#ifdef __DEBUG_STRING__
  cout << "CString::StrPosInCharArrFrom: STARTED: inarr = <" << inarr << ">, lookfor = <" << lookfor << ">, startpos = " << startpos << endl;
#endif
  if (pos >= strlen(inarr) - strlen(lookfor) || strlen(lookfor) > strlen(inarr))
    return -1;
  do
  {
    if (inarr[pos] == '\0')
      return -1;
#ifdef __DEBUG_STRING__
    cout << "CString::StrPosInCharArrFrom: inarr[pos=" << pos << "] = " << inarr[pos] << endl;
#endif
    for (i = 0; i < strlen(lookfor); i++)
    {
      if (inarr[pos + i] == lookfor[i])
      {
#ifdef __DEBUG_STRING__
        cout << "CString::StrPosInCharArrFrom: inarr[pos=" << pos << " + i=" << i << "](=" << inarr[pos+i] << ") == lookfor[i=" << i << "](=" << lookfor[i] << ") returned TRUE" << endl;
        //      cout << "mbasis.strposincharrfrom: inarr[pos(=%d) + strlen(lookfor=<%s>(=%d)) - 1 = %d] = %s\n", pos, lookfor, strlen(lookfor), pos + strlen(lookfor) - 1, inarr[pos + strlen(lookfor) - 1]);
#endif
        if (i == strlen(lookfor) - 1)
        {
#ifdef __DEBUG_STRING__
          cout << "CString::StrPosInCharArrFrom: i = " << i << ", strlen(lookfor=" << lookfor << ") = " << strlen(lookfor) << ": Returning pos = " << pos << endl;
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
#ifdef __DEBUG_STRING__
  cout << "CString::StrPosInCharArrFrom: returning pos = " << pos << endl;
#endif
  return pos;
}

/**
function int strposincharr{STRing POSition IN CHar ARR}(char inarr[STRLEN]: in, char* lookfor: in)
Returns first position of string 'in' in 'inarr', if found, else '-1'
 **/
int CString::StrPosInCharArr(const char inarr[STRLEN], const char* lookfor) const
{
  return this->StrPosInCharArrFrom(inarr, lookfor, 0);
}

/**
  function int StrPosInStr(char *P_InStr: in, const char* P_LookFor: in)
  Returns first position of String 'P_LookFor' in 'P_InStr', if found, else '-1'
 **/
int CString::StrPos(const char* P_LookFor) const
{
  CString CS_In(P_LookFor);
  return this->StrPos(CS_In);
/*  char PointChar = '.';
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

  for (Pos = 0; Pos < strlen(this->P_String); Pos++)
  {
    if (this->CharPosInCharArr(NumberCharArr, this->P_String[Pos]) < 0)
      return 0;
  }

  if (this->CharPos(PointChar) < 0)
    return 0;

  return 1;
  */
}

/**
  function int StrPosInStr(char *P_InStr: in, const char* P_LookFor: in)
  Returns first position of String 'P_LookFor' in 'P_InStr', if found, else '-1'
 **/
int CString::StrPos(const CString &LookFor) const
{
  int I_LengthIn = LookFor.GetLength();
  CString *P_SubString;
  for (int m=0; m<=this->GetLength() - I_LengthIn; m++){
    P_SubString = this->SubString(m,m+I_LengthIn-1);
    if (P_SubString->EqualValue(LookFor)){
      delete(P_SubString);
      return m;
    }
    delete(P_SubString);
  }
  return -1;
}

/**
  function int LastCharPosInStr(char *P_InStr: in, const char LookFor: in)
  Returns last position of character 'P_LookFor' in 'Str', if found, else '-1'
**/
int CString::LastCharPosInString(const string InStr, const char LookFor) const
{
  char* p_char = StringToPChar(InStr);
  CString *P_CS_Temp = new CString(p_char);
  int pos = P_CS_Temp->LastCharPos(LookFor);
  delete(P_CS_Temp);
  free(p_char);
  return pos;
}

/**
  function int LastCharPosInStr(char *P_InStr: in, const char LookFor: in)
  Returns last position of character 'P_LookFor' in 'P_InStr', if found, else '-1'
**/
int CString::LastCharPos(const char LookFor) const
{
  char StrArr[STRLEN];
  strcpy(StrArr, this->P_String);
  return (this->LastCharPosInCharArr(StrArr, LookFor));
}

/**
  function int StrPosInStrFrom(char *P_InStr: in, char* P_LookFor: in, int Start)
  Returns first position of String 'P_LookFor' in 'P_InStr', looking from position Start, if found, else '-1'
**/
int CString::StrPosFrom(const char* P_LookFor, int Start) const
{
  char StrArr[STRLEN];
  strcpy(StrArr, this->P_String);
#ifdef __DEBUG_STRING__
  cout << "CString::StrPosInStrFrom(this->P_String = <" << P_String << ">, P_LookFor = <" << P_LookFor << ">, Start = " << Start << ") started" << endl;
#endif
  return (StrPosInCharArrFrom(StrArr, P_LookFor, Start));
}


/**
  function int LastStrPosInStr(const char* p_LookFor: in)
  Returns last position of string 'p_LookFor' in 'this->P_String', if found, else '-1'
 **/
int CString::LastStrPos(const char* p_LookFor) const
{
  int Pos = -1;
  int LastPos;
  bool found = true;
  while ( found )
  {
    LastPos = Pos;
//    cout << "CString::LastStrPos: LastPos = " << LastPos << endl;
    Pos = this->StrPosFrom(p_LookFor, Pos + 1);
//    cout << "CString::LastStrPos: Pos = " << Pos << endl;
    if (Pos < 0)
      found = false;
  }
//  cout << "CString::LastStrPos: LastPos = " << LastPos << endl;
  return (LastPos);
}

/**
  function int LastStrPos(const CString &LookFor: in)
  Returns last position of string 'LookFor' in 'this->P_String', if found, else '-1'
 **/
int CString::LastStrPos(const CString &LookFor) const
{
  return (LastStrPos(LookFor.Get()));
}

char* CString::StringToPChar(const string &str) const
{
  char *p_char = (char*)malloc(sizeof(char) * STRLEN);
  int i = 0;
  bool EndFound = false;
#ifdef __DEBUG_STRING__
    cout << "CString::StringToPChar( str = " << str << "): Started" << endl;
#endif
#ifdef __DEBUG_STRING__
    cout << "CString::StringToPChar( str = " << str << "): str[0] = <" << str[0] << ">" << endl;
#endif
  while (!EndFound && i < STRLEN)
  {
#ifdef __DEBUG_STRING__
    cout << "CString::StringToPChar( str = " << str << "): while: str[i=" << i << "] = <" << str[i] << ">, p_char = <" << p_char << ">" << endl;
#endif
    p_char[i] = str[i];
    if (p_char[i] == '\0')
    {
#ifdef __DEBUG_STRING__
      cout << "CString::StringToPChar( str = " << str << "): EndFound" << endl;
#endif
      EndFound = true;
    }
    i++;
  }
#ifdef __DEBUG_STRING__
  cout << "CString::StringToPChar( str = " << str << "): Returning p_char = <" << p_char << ">" << endl;
#endif
  return p_char;
}

string CString::PCharToString(const char* p_str) const
{
#ifdef __DEBUG_STRING__
  cout << "CString::PCharToString( p_str = <" << p_str << ">) Started" << endl;
#endif
  string *str = new string("\0");
  (*str) = (*p_str);//(char*)malloc(sizeof(char) * STRLEN);
  return *str;
  /*



  int i = 0;
  bool EndFound = false;
  while (!EndFound && i < STRLEN)
  {
#ifdef __DEBUG_STRING__
    cout << "CString::PCharToString( p_str = <" << p_str << ">) p_str[i=" << i << "] = <" << p_str[i] << ">, str = <" << str << ">" << endl;
#endif
    str = str + p_str[i];
    if (str[i] == '\0')
      EndFound = true;
    i++;
  }
#ifdef __DEBUG_STRING__
  cout << "CString::PCharToString( p_str = <" << p_str << ">) Returning str = <" << str << ">" << endl;
#endif
  return str;*/
}

/**
    function bool InsertAt(const CString &cs: in, int Pos: in)
    Inserts CString 'cs' in 'P_String' at position Pos and returns TRUE, if successfull, else returns FALSE.
 **/
bool CString::InsertAt(const CString &CS, int Pos)
{
  int i;
  if (Pos < 0 || Pos > this->GetLength())
    return false;
  if (CS.GetLength() + this->GetLength() > STRLEN)
    return false;
  CString *P_TempString = new CString();
  for (i = 0; i < Pos; i++)
    (*P_TempString) += this->GetPChar()[i];
  for (i = 0; i < CS.GetLength(); i++)
    (*P_TempString) += CS.GetPChar()[i];
  for (i = Pos; i < this->GetLength(); i++)
    (*P_TempString) += this->GetPChar()[i];
  (*P_TempString) += '\0';
  return this->Copy(CS);
}

/**
function int IsDouble() const
Returns 1, if P_Str is a double, else 0.
 **/
int CString::IsDouble() const
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

  for (Pos = 0; Pos < strlen(this->P_String); Pos++)
  {
    if (this->CharPosInCharArr(NumberCharArr, this->P_String[Pos]) < 0)
      return 0;
  }

  if (this->CharPos(PointChar) < 0)
    return 0;

  return 1;
}

/**
function int IsInt()
Returns 1, if this->P_String is a integer, else 0.
 **/
int CString::IsInt() const
{
  int Pos;
  int IntArrPos;
  char NumberCharArr[11];
  char P_NumberString[STRLEN];

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

  for (Pos = 0; Pos < strlen(this->P_String); Pos++)
  {
    if (this->CharPosInCharArr(NumberCharArr, this->P_String[Pos]) < 0)
      return 0;
  }
  return 1;
}

/**
function int IsNumber()
Returns 1, if this->P_String is a integer or a double, else 0.
 **/
int CString::IsNumber() const
{
  if (this->IsDouble() == 1 || this->IsInt() == 1)
    return 1;
  return 0;
}

/** ********************************************************************/

CString* CString::IToA(int I_In){
  int i_temp = I_In;
  int length = 0;
  if (i_temp == 0){
    CString *P_CS_Out = new CString("0");
    return P_CS_Out;
  }
  while (i_temp > 0){
    length++;
    i_temp /= 10;
  }
  std::stringstream ss;
  ss << I_In;
  char *tempStr = new char[STRLEN];
  string str = ss.str();
#ifdef __DEBUG_STRING_ITOA__
  cout << "CString::IToA: str set to " << str << endl;
#endif
  for (int m=0; m < length; m++)
    tempStr[m] = str[m];
  tempStr[length] = '\0';
#ifdef __DEBUG_STRING_ITOA__
  cout << "CString::IToA: tempStr set to " << tempStr << endl;
#endif
  CString *P_CS_Out = new CString(tempStr);
#ifdef __DEBUG_STRING_ITOA__
  cout << "CString::IToA: P_CS_Out set to " << *P_CS_Out << endl;
#endif
  delete[](tempStr);
  return P_CS_Out;
}

/** ********************************************************************/

CString* CString::DToA(double D_In, int I_Precision){
  int i_temp = int(D_In);
  int length = 0;
  while (i_temp > 0){
    length++;
    i_temp /= 10;
  }
  std::stringstream ss;
  ss << D_In;
  char *tempStr = new char[STRLEN];
  string str = ss.str();
#ifdef __DEBUG_STRING_DTOA__
  cout << "CString::DToA: str set to " << str << endl;
  cout << "CString::DToA: str.length() = " << str.length() << endl;
#endif
  if (I_Precision == 0)
    I_Precision = 0-1;
  for (int m=0; m < length+I_Precision+1; m++){
    if (m >= str.length())
      tempStr[m] = '0';
    else
      tempStr[m] = str[m];
#ifdef __DEBUG_STRING_DTOA__
    cout << "CString::DToA: tempStr[m=" << m << "] set to " << tempStr[m] << endl;
#endif
  }
  tempStr[length+1+I_Precision] = '\0';
#ifdef __DEBUG_STRING_DTOA__
  cout << "CString::DToA: tempStr set to " << tempStr << endl;
#endif
  CString *P_CS_Out = new CString(tempStr);
#ifdef __DEBUG_STRING_DTOA__
  cout << "CString::DToA: P_CS_Out set to " << *P_CS_Out << endl;
#endif
  delete[](tempStr);
  return P_CS_Out;
}

/** ********************************************************************/

int CString::AToI() const{
  int retVal = int(atoi(this->Get()));
  return retVal;
}

/** ********************************************************************/

bool CString::AToI(int &I_Out) const{
  I_Out = 0;
  for (int i=0; i<this->GetLength(); i++){
    if ((this->P_String[i] != '0') &&
        (this->P_String[i] != '1') &&
        (this->P_String[i] != '2') &&
        (this->P_String[i] != '3') &&
        (this->P_String[i] != '4') &&
        (this->P_String[i] != '5') &&
        (this->P_String[i] != '6') &&
        (this->P_String[i] != '7') &&
        (this->P_String[i] != '8') &&
        (this->P_String[i] != '9') &&
        (this->P_String[i] != '-')){
      cout << "CString::AToI: ERROR: this->P_String[i=" << i << "] = <" << P_String[i] << "> is not a number => Returning FALSE" << endl;
//      return false;
    }
  }
  I_Out = int(atoi(this->Get()));
  return true;
}

/** ********************************************************************/

double CString::AToD() const{
  double retVal = double(atof(this->Get()));
  return retVal;
  /**
  CString CS_Temp = *this;
  CS_Temp += CString("\0");
  char *a = CS_Temp.Get();
  double retVal = atoi(a); /// start off getting the number, assuming it is a valid string to use atoi on.
#ifdef __DEBUG_STRING_ATOD__
  cout << "CString::AToD: retVal set to " << retVal << endl;
#endif
  int start = 0;
  int end = CS_Temp.GetLength();
#ifdef __DEBUG_STRING_ATOD__
  cout << "CString::AToD: end set to " << end << endl;
#endif
  for(int i=0; i < CS_Temp.GetLength(); i++){ /// loop through the string to find the positions of the decimal portion, if there is one
#ifdef __DEBUG_STRING_ATOD__
    cout << "CString::AToD: a[i="<< i << "] = <" << a[i] << ">" << endl;
#endif
    if(a[i] == '.' && start == 0){
      start = i+1; /// set the start position to 1 more than the current (avoids a char as the first character - we want a digit)
#ifdef __DEBUG_STRING_ATOD__
      cout << "CString::AToD: start set to " << start << endl;
#endif
//      break;
    }
    else if(start != 0 &&  (a[i] < '0' || a[i] > '9')){ /// make sure that start is set and that we aren't looking at digits
      end = i; /// if so, set the end location for the substring
#ifdef __DEBUG_STRING_ATOD__
      cout << "CString::AToD: end set to " << end << endl;
#endif
      break; /// we don't need to continue anymore - break out of the loop
    }
  }
  if(end > start){ /// avoids substring problems.
    CString* P_CS_Temp = CS_Temp.SubString(start, end);
#ifdef __DEBUG_STRING_ATOD__
    cout << "CString::AToD: P_CS_Temp set to " << *P_CS_Temp << endl;
#endif
    char* decimal = P_CS_Temp->Get(); /// get the string that is after the decimal point
#ifdef __DEBUG_STRING_ATOD__
    cout << "CString::AToD: decimal set to " << decimal << endl;
#endif
    long dec = atol(decimal); /// make it an integer
    delete(P_CS_Temp);
#ifdef __DEBUG_STRING_ATOD__
    cout << "CString::AToD: dec set to " << dec << endl;
#endif
    int power = end-start; /// find the power of 10 we need to divide by
#ifdef __DEBUG_STRING_ATOD__
    cout << "CString::AToD: power set to " << power << endl;
#endif
    retVal += ((double)dec)/(pow(10.0, (double)power)); /// divide and add to the return value (thus making it a true double)
  }
  return retVal; /// return - simple enough*/
}

/** ********************************************************************/

bool CString::AToD(double &D_Out) const{
  D_Out = 0;
  for (int i=0; i<this->GetLength(); i++){
    if ((this->P_String[i] != '0') &&
        (this->P_String[i] != '1') &&
        (this->P_String[i] != '2') &&
        (this->P_String[i] != '3') &&
        (this->P_String[i] != '4') &&
        (this->P_String[i] != '5') &&
        (this->P_String[i] != '6') &&
        (this->P_String[i] != '7') &&
        (this->P_String[i] != '8') &&
        (this->P_String[i] != '9') &&
        (this->P_String[i] != '-') &&
        (this->P_String[i] != '+') &&
        (this->P_String[i] != 'e') &&
        (this->P_String[i] != 'E') &&
        (this->P_String[i] != '.')){
      cout << "CString::AToI: ERROR: this->P_String[i=" << i << "] = <" << P_String[i] << "> is not a number => Returning FALSE" << endl;
      return false;
    }
  }
  D_Out = double(atof(this->Get()));
  return true;
}

/** ********************************************************************/

bool CString::AToD(const Array<CString, 1> &CS_A1_In, Array<double, 1> &D_A1_Out) const{
  int I_NElements = CS_A1_In.size();
  D_A1_Out.resize(I_NElements);
  for (int i=0; i < I_NElements; i++){
    D_A1_Out(i) = (CS_A1_In(i)).AToD();
  }
  return true;
}

/** ********************************************************************/

/// Replaces all occurences CString <CS_ToReplace> in this with <CS_Replace>
/// Returns a copy of itself if CS_ToReplace_In is not found in this
CString* CString::StrReplace(const CString& CS_ToReplace_In, const CString& CS_Replace_In){
  CString *P_CS_Out;
  CString *P_CS_Temp;
  int LastPos;
  int Pos = this->StrPos(CS_ToReplace_In);
  if (Pos < 0){
    P_CS_Out = new CString(*this);
    return P_CS_Out;
  }
  P_CS_Out = this->SubString(0,Pos-1);
  LastPos = Pos;
  while (Pos >= 0){
    LastPos = Pos;
    *P_CS_Out += CS_Replace_In;
    Pos = this->StrPosFrom(CS_ToReplace_In.Get(),Pos+1);
    if (Pos > 0){
      P_CS_Temp = this->SubString(LastPos+CS_ToReplace_In.GetLength(),Pos-1);
      *P_CS_Out += *P_CS_Temp;
      delete(P_CS_Temp);
    }
  }
  P_CS_Temp = this->SubString(LastPos+CS_ToReplace_In.GetLength());
  *P_CS_Out += *P_CS_Temp;
  delete(P_CS_Temp);
  return P_CS_Out;
}

/// Replaces all occurences CString <CS_ToReplace> in this with <CS_Replace> and writes result to 
/// CS_Out
/// Returns false if CS_ToReplace_In is not found in this
bool CString::StrReplace(const CString& CS_ToReplace_In, const CString& CS_Replace_In, CString &CS_Out){
  int LastPos;
  int Pos = this->StrPos(CS_ToReplace_In);
  if (Pos < 0)
    return false;
  CString *P_CS_Out = this->SubString(0,Pos-1);
  CString *P_CS_Temp;
  while (Pos >= 0){
    LastPos = Pos;
    *P_CS_Out += CS_Replace_In;
    Pos = this->StrPosFrom(CS_ToReplace_In.Get(),Pos+1);
    if (Pos > 0){
      P_CS_Temp = this->SubString(LastPos+CS_ToReplace_In.GetLength(),Pos-1);
      *P_CS_Out += *P_CS_Temp;
      delete(P_CS_Temp);
    }
  }
  P_CS_Temp = this->SubString(LastPos+CS_ToReplace_In.GetLength());
  *P_CS_Out += *P_CS_Temp;
  delete(P_CS_Temp);
  CS_Out.Set(*P_CS_Out);
  delete(P_CS_Out);
  return true;
}
/** ********************************************************************/

void CString::Show (ostream &os) const
{
/*#ifdef __DEBUG_STRING__
  cout << "CString::Show(ostream &os) started" << endl;
#endif
*/
  os << this->P_String;
}

/** *********************************************************************/

CString* CString::IntToCString(int I_In){
  return this->IToA(I_In);
//  std::ostringstream out;
//  out << I_In;
//  CString *CS_Out = new CString(out.str());
//  return CS_Out;
}

/** *********************************************************************/

long CString::CountLines(const CString &fnc) const{
#ifdef __DEBUG__
  //  fprintf(logfile,"CString::CountLines: method started\n");
#endif
  FILE *ffile;
  long nelements;
  char oneword[STRLEN];
  char fname[STRLEN];
  char *line;
  strcpy(fname, fnc.Get());

#ifdef __DEBUG__
  printf("CString::CountLines: function started\n");
#endif
  ffile = fopen(fname, "r");
  if (ffile == NULL)
  {
    cout << "CString::CountLines: Failed to open file fname (=<" << fname << ">)" << endl;
//    (*P_OFS_Log) << "CString::CountLines: Failed to open file fname (=<" << fname << ">)" << endl;
    exit (EXIT_FAILURE);
    //return 0;
  }
#ifdef __DEBUG__
  printf("CString::CountLines: File fname(=<%s>) opened\n", fname);
#endif

  nelements = 0;
  // --- read file <fname> named <ffile>
  do
  {
    line = fgets(oneword, STRLEN, ffile);
    if (line != NULL)
    {
#ifdef __DEBUG__
    //      printf("CString::CountLines: oneword = <%s>\n", oneword);
#endif
    // --- count lines
      nelements++;
    }
  }
  while (line != NULL);
#ifdef __DEBUG__
  printf("CString::CountLines: File fname(=<%s>) contains %d data lines\n",fname,nelements);
#endif
  // --- close input file
  fclose(ffile);
#ifdef __DEBUG__
  printf("CString::CountLines: File fname (=<%s>) closed\n", fname);
#endif
  return nelements;
}

bool CString::Add(const CString &CS_ToAdd){
  int I_Len = this->GetLength();
  if (I_Len + CS_ToAdd.GetLength() + 1 >= STRLEN){
    cout << "CString::Add: ERROR: this->GetLength(=" << I_Len << ") + CS_ToAdd.GetLength(=" << CS_ToAdd.GetLength() << ") > " << STRLEN << endl;
    return false;
  }
  char *p_char = CS_ToAdd.GetPChar();
  for (int i=0; i<=CS_ToAdd.GetLength(); i++)
    this->P_String[I_Len+i] = p_char[i];
  this->P_String[I_Len + CS_ToAdd.GetLength() + 1] = '\0';
//  free(p_char);
  return true;
}

/// Replaces all occurences of CString <CS_ToReplace_In> in textfile <CS_TextFile_In> with
/// <CS_Replace_In> and writes result to <CS_TextFile_Out>
bool CString::StrReplaceInList(const CString &CS_TextFile_In,
		      	       const CString &CS_ToReplace_In,
		   	       const CString &CS_Replace_In,
		   	       const CString &CS_TextFile_Out) const{
  Array<CString, 1> CS_A1_TextLines_In(1);
  Array<CString, 1> CS_A1_TextLines_Out(1);
  if (!this->ReadFileLinesToStrArr(CS_TextFile_In, CS_A1_TextLines_In)){
    cout << "CString::StrReplaceInList: ERROR: ReadFileLinesToStrArr(" << CS_TextFile_In << ") returned FALSE" << endl;
    return false;
  }
  if (!this->StrReplaceInList(CS_A1_TextLines_In, 
                              CS_ToReplace_In, 
                              CS_Replace_In, 
                              CS_A1_TextLines_Out)){
    cout << "CString:StrReplaceInList: ERROR: StrReplaceInList(CS_A1) returned FALSE" << endl;
    return false;
  }
  if (!this->WriteStrListToFile(CS_A1_TextLines_Out, CS_TextFile_Out)){
    cout << "CString::StrReplaceInList: ERROR: WriteStrListToFile(" << CS_TextFile_In << ") returned FALSE" << endl;
    return false;
  }
  return true;
}

/// Replaces all occurences of CString <CS_ToReplace_In> in textfile <CS_TextFile_In> with
/// <CS_Replace_In> and writes result to <CS_TextFile_Out>
bool CString::StrReplaceInList(const Array<CString, 1> &CS_A1_In,
                               const CString &CS_ToReplace_In,
                               const CString &CS_Replace_In,
                               Array<CString, 1> &CS_A1_Out) const{
  CString CS_LineNew(" ");
  CS_A1_Out.resize(CS_A1_In.size());
  CString CS_Line(" ");
  for (int i_line=0; i_line < CS_A1_In.size(); i_line++){
    CS_Line.Set(CS_A1_In(i_line));
    if (!CS_Line.StrReplace(CS_ToReplace_In, CS_Replace_In, CS_LineNew)){
      cout << "CString::StrReplaceInList: ERROR: StrReplace(" << CS_ToReplace_In << ", " << CS_Replace_In << ") returned FALSE" << endl;
      return false;
    }
    CS_A1_Out(i_line).Set(CS_LineNew);
  }
  #ifdef __DEBUG_CSTRING_STRREPLACEINLIST__
    cout << "StrReplaceInList: CS_A1_In = " << CS_A1_In << endl;
    cout << "StrReplaceInList: CS_A1_Out = " << CS_A1_Out << endl;
  #endif
  return true;
}

bool CString::ReadFileToStrArr(const CString &CS_FileName_In,
                               Array<CString, 2> &CS_A2_Out,
                               const CString &CS_Delimiter) const{
  if (!this->FileAccess(CS_FileName_In)){
    cout << "CString::ReadFileToStrArr: ERROR: Access(CS_FileName_In) returned FALSE" << endl;
    return false;
  }
  int I_NLines = this->CountLines(CS_FileName_In);
  int I_NCols = this->CountCols(CS_FileName_In, CS_Delimiter);
  cout << "CString::ReadFileToStrArr: " << CS_FileName_In << ": I_NLines = " << I_NLines << ", I_NCols = " << I_NCols << endl;
  CString templine(" ");
  FILE *ffile;
  long nelements;
  char oneword[255];
  char fname[255];
  char *line;
  strcpy(fname, CS_FileName_In.Get());
  CString CS_Line;

  CS_A2_Out.resize(I_NLines, I_NCols);

#ifdef __DEBUG_FITS_READFILETOSTRARR__
  printf("CString::ReadFileToStrArr: function started\n");
#endif
  ffile = fopen(fname, "r");
  if (ffile == NULL)
  {
    cout << "CString::ReadFileToStrArr: Failed to open file fname (=<" << fname << ">)" << endl;
    return false;
  }
#ifdef __DEBUG_FITS_READFILETOSTRARR__
  printf("CString::ReadFileToStrArr: File fname(=<%s>) opened\n", fname);
#endif

  int I_Row = 0;
  int I_Col = 0;
  int I_Pos;
  CString *P_CS_Temp;
  char *chr_cstring;
  do
  {
    line = fgets(oneword, 255, ffile);
    if (line != NULL)
    {
      if (!CS_Line.Set(line)){
        cout << "CString::ReadFileToStrArr: ERROR: CS_Line.Set(line) returned FALSE." << endl;
        return false;
      }
      I_Pos = CS_Line.CharPos('\n');
#ifdef __DEBUG_FITS_READFILETOSTRARR__
      cout << "CString::ReadFileToStrArr: I_Row = " << I_Row << ": I_Pos(tempchar) set to " << I_Pos << endl;
#endif
      if (I_Pos >= 0){
        chr_cstring = CS_Line.Get();
        chr_cstring[I_Pos] = '\0';
        CS_Line = CString(chr_cstring);
#ifdef __DEBUG_FITS_READFILETOSTRARR__
        cout << "CString::ReadFileToStrArr: I_Row = " << I_Row << ": CS_Line set to " << CS_Line << endl;
#endif
      }
      I_Pos = CS_Line.StrPos(CString("#"));
      if (I_Pos != 0){
	CS_Line.Trim(2);
        I_Col = 0;
        while (I_Pos = CS_Line.StrPos(CS_Delimiter) >= 0){
          I_Pos = CS_Line.StrPos(CS_Delimiter);
#ifdef __DEBUG_FITS_READFILETOSTRARR__
          cout << "CString::ReadFileToStrArr: while: I_Pos set to " << I_Pos << endl;
#endif
          P_CS_Temp = CS_Line.SubString(0,I_Pos-1);
#ifdef __DEBUG_FITS_READFILETOSTRARR__
          cout << "CString::ReadFileToStrArr: while: P_CS_Temp set to " << *P_CS_Temp << endl;
#endif
          CS_A2_Out(I_Row, I_Col) = (*P_CS_Temp);//)){
#ifdef __DEBUG_FITS_READFILETOSTRARR__
          cout << "CString::ReadFileToStrArr: CS_A2_Out(I_Row, I_Col) set to '" << *P_CS_Temp << "'" << endl;
#endif
          delete(P_CS_Temp);

          P_CS_Temp = CS_Line.SubString(CS_Line.StrPos(CS_Delimiter)+1);
#ifdef __DEBUG_FITS_READFILETOSTRARR__
          cout << "CString::ReadFileToStrArr: while: P_CS_Temp set to " << *P_CS_Temp << endl;
#endif
          P_CS_Temp->TrimChar(CS_Delimiter.Get(),2);
#ifdef __DEBUG_FITS_READFILETOSTRARR__
          cout << "CString::ReadFileToStrArr: while: P_CS_Temp set to " << *P_CS_Temp << endl;
#endif
          CS_Line = *P_CS_Temp;
#ifdef __DEBUG_FITS_READFILETOSTRARR__
          cout << "CString::ReadFileToStrArr: while: CS_Line set to " << CS_Line << endl;
#endif
          delete(P_CS_Temp);
          I_Col++;
        }
#ifdef __DEBUG_FITS_READFILETOSTRARR__
        cout << "CString::ReadFileToStrArr: end of while: I_Row = " << I_Row << ", I_Col = " << I_Col << ", CS_Line set to " << CS_Line << endl;
#endif
        CS_A2_Out(I_Row, I_Col) = CS_Line;//))){
//          cout << "CString::ReadFileToStrArr: ERROR: CS_A2_Out.(I_Row, I_Col).Set(" << *P_CS_Temp << ") returned FALSE." << endl;
//          return false;
//        }
      }
      I_Row++;
    }
  }
  while (line != NULL);
#ifdef __DEBUG_FITS_READFILETOSTRARR__
  printf("CString::ReadFileToStrArr: File fname(=<%s>) contains %d data lines\n",fname,I_Row);
#endif
  // --- close input file
  fclose(ffile);
#ifdef __DEBUG_FITS_READFILETOSTRARR__
  printf("CString::ReadFileToStrArr: File fname (=<%s>) closed\n", fname);
#endif
  return true;
}

bool CString::ReadFileToDblArr(const CString &CS_FileName_In,
                             Array<double, 2> &D_A2_Out,
                             const CString &CS_Delimiter) const{
  Array<CString, 2> CS_A2_Arr(2,2);
  if (!this->ReadFileToStrArr(CS_FileName_In, CS_A2_Arr, CS_Delimiter)){
    cout << "CString::ReadFileToDblArr: ERROR: ReadFileToStrArr returned FALSE" << endl;
    return false;
  }
  D_A2_Out.resize(CS_A2_Arr.rows(), CS_A2_Arr.cols());
  for (int i_row=0; i_row<CS_A2_Arr.rows(); i_row++){
    for (int i_col=0; i_col<CS_A2_Arr.cols(); i_col++){
      D_A2_Out(i_row, i_col) = (CS_A2_Arr(i_row, i_col)).AToD();
    }
  }
  return true;
}

/** *******************************************************/

bool CString::ReadFileLinesToStrArr(const CString &CS_FileName_In,
                                  Array<CString, 1> &CS_A1_Out) const{
  CString *P_CS_FirstChar = CS_FileName_In.SubString(0,0);
  CString CS_FileName(CS_FileName_In);
  if (P_CS_FirstChar->EqualValue(CString("@"))){
    delete(P_CS_FirstChar);
    P_CS_FirstChar = CS_FileName_In.SubString(1);
    CS_FileName.Set(*P_CS_FirstChar);
  }
  delete(P_CS_FirstChar);

  if (!this->FileAccess(CS_FileName)){
    cout << "CString::ReadFileLinesToStrArr: ERROR: Access(" << CS_FileName << ") returned FALSE" << endl;
    return false;
  }
  int I_NLines = this->CountLines(CS_FileName);
  cout << "CString::ReadFileLinesToStrArr: I_NLines = " << I_NLines << endl;
  CString templine(" ");
  FILE *ffile;
  long nelements;
  char oneword[255];
  char fname[255];
  char *line;
  strcpy(fname, CS_FileName.Get());
  CString CS_Line;

  CS_A1_Out.resize(I_NLines);

  #ifdef __DEBUG_FITS_READFILELINESTOSTRARR__
  cout << "CString::ReadFileLinesToStrArr: function started" << endl;;
  #endif
  ffile = fopen(fname, "r");
  if (ffile == NULL)
  {
    cout << "CString::ReadFileLinesToStrArr: Failed to open file fname (=<" << fname << ">)" << endl;
    return false;
  }
  #ifdef __DEBUG_FITS_READFILELINESTOSTRARR__
    cout << "CString::ReadFileLinesToStrArr: File fname(=<" << fname << ">) opened" << endl;
    cout << "CString::ReadFileLinesToStrArr: 1. Starting do loop" << endl;
  #endif

  int I_Row = 0;
  int I_Pos;
  CString *P_CS_Temp;
  char *chr_cstring;
  cout << "CString::ReadFileLinesToStrArr: 2. Starting do loop" << endl;
  do
  {
    line = fgets(oneword, 255, ffile);
    if (line != NULL)
    {
      #ifdef __DEBUG_FITS_READFILELINESTOSTRARR__
      cout << "CString::ReadFileLinesToStrArr: I_Row = " << I_Row << ": line set to " << line << endl;
      #endif
      if (!CS_Line.Set(line)){
        cout << "CString::ReadFileLinesToStrArr: ERROR: CS_Line.Set(line) returned FALSE." << endl;
        return false;
      }
      I_Pos = CS_Line.CharPos('\n');
      #ifdef __DEBUG_FITS_READFILELINESTOSTRARR__
        cout << "CString::ReadFileLinesToStrArr: I_Row = " << I_Row << ": CS_Line=" << CS_Line << ": I_Pos('\n') set to " << I_Pos << endl;
      #endif
      if (I_Pos >= 0){
        chr_cstring = CS_Line.Get();
        chr_cstring[I_Pos] = '\0';
        CS_Line = CString(chr_cstring);
        #ifdef __DEBUG_FITS_READFILELINESTOSTRARR__
          cout << "CString::ReadFileLinesToStrArr: I_Row = " << I_Row << ": CS_Line set to " << CS_Line << endl;
        #endif
      }
      I_Pos = CS_Line.StrPos(CString("#"));
      if (I_Pos != 0){
        CS_A1_Out(I_Row) = CS_Line;//)){
        #ifdef __DEBUG_FITS_READFILELINESTOSTRARR__
          cout << "CString::ReadFileLinesToStrArr: CS_A1_Out(I_Row=" << I_Row << ") set to '" << CS_A1_Out(I_Row) << "'" << endl;
        #endif
      }
      I_Row++;
    }
  }while (line != NULL);
  #ifdef __DEBUG_FITS_READFILELINESTOSTRARR__
    cout << "CString::ReadFileLinesToStrArr: File fname(=<" << fname << ">) contains " << I_Row << " data lines" << endl;
  #endif
  // --- close input file
  fclose(ffile);
  #ifdef __DEBUG_FITS_READFILELINESTOSTRARR__
    cout << "CString::ReadFileLinesToStrArr: File fname (=<" << fname << ">) closed";
  #endif
  CS_A1_Out.resizeAndPreserve(I_Row);
  return true;
}

/**
 *
 **/
bool CString::FileAccess(const CString &fn) const
{
  FILE *ffile;
  #ifdef __DEBUG_CSTRING_FILEACCESS__
    cout << "CString::FileAccess: Checking for file " << fn << endl;
  #endif
  ffile = fopen(fn.Get(), "r");
  if (ffile == NULL)
  {
    cout << "CString::FileAccess: Failed to open file fname (" << fn.Get() << ")" << endl;
    return false;
  }
  fclose(ffile);
  return true;
}

/**
function int CountDataLines(const CString &fnc: inout)
Returns number of lines which do not start with '#' of file <fnc>.
 **/
long CString::CountDataLines(const CString &fnc) const
{
#ifdef __DEBUG_FITS__
  //  fprintf(logfile,"CString::CountDataLines: method started\n");
#endif
  FILE *ffile;
  long nelements;
  char oneword[255];
  char fname[255];
  char *line;
  strcpy(fname, fnc.Get());

#ifdef __DEBUG_FITS__
  printf("CString::CountDataLines: function started\n");
#endif
  ffile = fopen(fname, "r");
  if (ffile == NULL)
  {
    cout << "CString::CountDataLines: Failed to open file fname (=<" << fname << ">)" << endl;
#ifdef __DEBUG_FITS_PISKUNOV__
    (*P_OFS_Log) << "CString::CountDataLines: Failed to open file fname (=<" << fname << ">)" << endl;
    exit (EXIT_FAILURE);
#endif
    //return 0;
  }
#ifdef __DEBUG_FITS__
  printf("CString::CountDataLines: File fname(=<%s>) opened\n", fname);
#endif

  nelements = 0;
  // --- read file <fname> named <ffile>
  do
  {
    line = fgets(oneword, 255, ffile);
    if (line != NULL && line[0] != '#')
    {
#ifdef __DEBUG_FITS__
      //      printf("CString::CountDataLines: oneword = <%s>\n", oneword);
#endif
      // --- count lines
      nelements++;
    }
  }
  while (line != NULL);
#ifdef __DEBUG_FITS__
  printf("CString::CountDataLines: File fname(=<%s>) contains %d data lines\n",fname,nelements);
#endif
  // --- close input file
  fclose(ffile);
#ifdef __DEBUG_FITS__
  printf("CString::CountDataLines: File fname (=<%s>) closed\n", fname);
#endif
  return nelements;
}

/**
 function int CountCols(const CString &fnc: inout)
 Returns number of columns of file <fnc>.
 **/
long CString::CountCols(const CString &CS_FileName_In, const CString &CS_Delimiter) const{

  long L_Cols = 0;
  long L_OldCols = 0;
  CString templine(" ");
  int I_NLines = this->CountLines(CS_FileName_In);
//    openr,lun,filename,/get_lun
  FILE *ffile;
  long I_Row;
  char oneword[255];
  char fname[255];
  char *line;
  char tempchar[255];
  tempchar[0] = '\n';
  tempchar[1] = '\0';
  strcpy(fname, CS_FileName_In.Get());
  CString CS_Line;
  char *chr_cstring;

#ifdef __DEBUG_STRING_COUNTCOLS__
  printf("CString::CountCols: function started\n");
#endif
  ffile = fopen(fname, "r");
  if (ffile == NULL)
  {
    cout << "CString::CountCols: Failed to open file fname (=<" << fname << ">)" << endl;
    exit (EXIT_FAILURE);
    //return 0;
  }
#ifdef __DEBUG_STRING_COUNTCOLS__
  printf("CString::CountCols: File fname(=<%s>) opened\n", fname);
#endif

  I_Row = 0;
  // --- read file <fname> named <ffile>
  int I_Pos;
  CString *P_CS_Temp;
  do
  {
    line = fgets(oneword, 255, ffile);
    if (line != NULL)
    {
      // --- count lines
      if (!CS_Line.Set(line)){
        cout << "CString::CountCols: ERROR: CS_Line.Set(line) returned FALSE." << endl;
        return false;
      }
      CS_Line.Trim(2);
      I_Pos = CS_Line.CharPos('\n');
#ifdef __DEBUG_STRING_COUNTCOLS__
      cout << "CString::CountCols: I_Row = " << I_Row << ": I_Pos(tempchar) set to " << I_Pos << endl;
#endif
      if (I_Pos >= 0){
        chr_cstring = CS_Line.Get();
        chr_cstring[I_Pos] = '\0';
//        P_CS_Temp = CS_Line.SubString(0,CS_Line.StrPos(tempchar)-1);
        CS_Line = CString(chr_cstring);
#ifdef __DEBUG_STRING_COUNTCOLS__
        cout << "CString::CountCols: I_Row = " << I_Row << ": CS_Line set to <" << CS_Line << ">" << endl;
#endif
//        delete(P_CS_Temp);
      }
      I_Pos = CS_Line.StrPos(CString("#"));
      if (I_Pos != 0){
        L_Cols = 0;
        while (CS_Line.StrPos(CS_Delimiter) >= 0){
//          while (CS_Line.StrPos(CS_Delimiter) == 0){
//            CString *P_CS_Line = CS_Line.SubString(1);
//            CS_Line.Set(*P_CS_Line);
//            delete(P_CS_Line);
//          }
          L_Cols++;
#ifdef __DEBUG_STRING_COUNTCOLS__
          cout << "CString::CountCols: I_Row = " << I_Row << ": L_Cols set to " << L_Cols << endl;
#endif
          I_Pos = CS_Line.StrPos(CS_Delimiter);
#ifdef __DEBUG_STRING_COUNTCOLS__
          cout << "CString::CountCols: I_Row = " << I_Row << ": I_Pos set to " << I_Pos << endl;
#endif
          P_CS_Temp = CS_Line.SubString(I_Pos+1);
#ifdef __DEBUG_STRING_COUNTCOLS__
          cout << "CString::CountCols: I_Row = " << I_Row << ": P_CS_Temp set to <" << *P_CS_Temp << ">" << endl;
#endif
          P_CS_Temp->TrimChar(CS_Delimiter,2);
#ifdef __DEBUG_STRING_COUNTCOLS__
          cout << "CString::CountCols: I_Row = " << I_Row << ": P_CS_Temp set to <" << *P_CS_Temp << ">" << endl;
#endif
          CS_Line = *P_CS_Temp;
#ifdef __DEBUG_STRING_COUNTCOLS__
          cout << "CString::CountCols: I_Row = " << I_Row << ": CS_Line set to <" << CS_Line << ">" << endl;
#endif
          delete(P_CS_Temp);
        }
        L_Cols++;
#ifdef __DEBUG_STRING_COUNTCOLS__
        cout << "CString::CountCols: I_Row = " << I_Row << ": L_Cols set to " << L_Cols << endl;
#endif
      }
      if (L_Cols > L_OldCols){
          L_OldCols = L_Cols;
#ifdef __DEBUG_STRING_COUNTCOLS__
          cout << "CString::CountCols: L_Cols = " << L_Cols << endl;
#endif
      }
      I_Row++;
    }
  }
  while (line != NULL);
#ifdef __DEBUG_FITS__
  printf("CString::CountCols: File fname(=<%s>) contains %d data lines\n",fname,I_Row);
#endif
  // --- close input file
  fclose(ffile);
#ifdef __DEBUG_FITS__
  printf("CString::CountCols: File fname (=<%s>) closed\n", fname);
#endif
  return L_OldCols;

}

bool CString::WriteStrListToFile(const Array<CString, 1> &CS_A1_In, const CString &CS_FileNameOut_In) const{
  FILE *p_file;
  p_file = fopen(CS_FileNameOut_In.Get(), "w");

  for (int m = 0; m < CS_A1_In.size(); m++)
    fprintf(p_file, "%s\n", (CS_A1_In)(m).Get());

  fclose(p_file);
  return true;

}

bool CString::WriteStrListToFile(const Array<CString, 2> &CS_A2_In, const CString &CS_Delimiter, const CString &CS_FileNameOut_In) const{
  Array<CString, 1> CS_A1_TextLines(CS_A2_In.rows());
  for (int i_row=0; i_row<CS_A2_In.rows(); i_row++){
    CS_A1_TextLines(i_row).Set(CS_A2_In(i_row, 0));
    for (int i_col=1; i_col<CS_A2_In.cols(); i_col++){
      CS_A1_TextLines(i_row).Add(CS_Delimiter);
      CS_A1_TextLines(i_row).Add(CS_A2_In(i_row, i_col));
    }
  }
  return this->WriteStrListToFile(CS_A1_TextLines, CS_FileNameOut_In);
}

bool CString::IsList() const{
  CString *P_CS_FirstChar = this->SubString(0,0);
  cout << "CString::IsList: First character = " << *P_CS_FirstChar << endl;
  CString CS_Temp("@");
  if (CS_Temp.EqualValue(*P_CS_FirstChar)){
    delete(P_CS_FirstChar);
    return true;
  }
  delete(P_CS_FirstChar);
  return false;
}

bool CString::MkDir(const CString &CS_PathName) const{
  int I_Status;
  bool B_Success = false;
  I_Status = mkdir(CS_PathName.GetPChar(), S_IRWXU);
  cout << "CString::MkDir: I_Status = " << I_Status << endl;
  if (I_Status == 0)
    B_Success = true;
  else if( I_Status == -1 )
  {
    switch( errno )
    {
//      case ENOENT:
//        //parent didn't exist, try to create it
//        if( MkDir( path.substr(0, path.find_last_of('/')) ) )
//            //Now, try to create again.
//          bSuccess = 0 == ::mkdir( path.c_str(), 0775 );
//        else
//                    bSuccess = false;
//                break;
      case EEXIST:
        //Done!
        B_Success = true;
        break;
      default:
        B_Success = false;
        break;
    }
  }
  return B_Success;
}

bool CString::AddFirstPartAsDir(const Array<CString, 1> &CS_A1_In, Array<CString, 1> &CS_A1_Out) const{
  CS_A1_Out.resize(CS_A1_In.size());

  CString *P_CS_Temp;
  int I_Pos = 0;
  for (int i_row=0; i_row<CS_A1_In.size(); i_row++){
    I_Pos = CS_A1_In(i_row).CharPos('_') - 1;
    if (I_Pos < 1){
      cout << "CString::AddFirstPartAsDir: ERROR: '_' not found in CS_A1_In(i_row=" << i_row << ") = <" << CS_A1_In(i_row) << ">" << endl;
      return false;
    }
    P_CS_Temp = CS_A1_In(i_row).SubString(0,I_Pos);
    CS_A1_Out(i_row).Set(*P_CS_Temp);
    CS_A1_Out(i_row).Add("/");
    CS_A1_Out(i_row).Add(CS_A1_In(i_row));
    delete(P_CS_Temp);
  }
  return true;
}

bool CString::AddNameAsDir(const Array<CString, 1> &CS_A1_In, Array<CString, 1> &CS_A1_Out) const{
  CS_A1_Out.resize(CS_A1_In.size());

  CString *P_CS_Temp;
  int I_Pos = 0;
  for (int i_row=0; i_row<CS_A1_In.size(); i_row++){
    I_Pos = CS_A1_In(i_row).LastCharPos('.') - 1;
    if (I_Pos < 1){
      cout << "CString::AddFirstPartAsDir: ERROR: '_' not found in CS_A1_In(i_row=" << i_row << ") = <" << CS_A1_In(i_row) << ">" << endl;
      return false;
    }
    P_CS_Temp = CS_A1_In(i_row).SubString(0,I_Pos);
    CS_A1_Out(i_row).Set(*P_CS_Temp);
    CS_A1_Out(i_row).Add("/");
    CS_A1_Out(i_row).Add(CS_A1_In(i_row));
    delete(P_CS_Temp);
  }
  return true;
}


/// Checks CS_A1_In and returns an integer array with 1 where CS_A1_In(i).EqualValue(CS_Comp),
/// otherwise 0
Array<int, 1>* CString::Where(const Array<CString, 1> &CS_A1_In,
                              const CString &CS_Comp) const{
  Array<int, 1> *P_I_A1_Where = new Array<int, 1>(CS_A1_In.size());
  (*P_I_A1_Where) = 0;
  for (int i_el=0; i_el<CS_A1_In.size(); i_el++){
    if (CS_A1_In(i_el).EqualValue(CS_Comp)){
      (*P_I_A1_Where)(i_el) = 1;
    }
  }
  return P_I_A1_Where;
}


/** Removes element at position I_Pos from CS_A1_InOut
 * */
bool CString::RemoveElementFromArray(Array<CString, 1> &CS_A1_InOut, int I_Pos) const{
  if ((I_Pos < 0) || (I_Pos >= CS_A1_InOut.size())){
    cout << "CString::RemoveElementFromArray: ERROR: I_Pos=" << I_Pos << " < 0 or >= CS_A1_InOut.size()=" << CS_A1_InOut.size() << endl;
    return false;
  }
  Array<CString, 1> CS_A1_Temp(CS_A1_InOut.size()-1);
  int I_El = 0;
  for (int i_el=0; i_el<CS_A1_InOut.size(); i_el++){
    if (i_el != I_Pos){
      CS_A1_Temp(I_El).Set(CS_A1_InOut(i_el));
      I_El++;
    }
  }
  CS_A1_InOut.resize(CS_A1_Temp.size());
  CS_A1_InOut = CS_A1_Temp;
  return true;
}

/** Removes elements listed in I_A1_ElementsToRemove_In from CS_A1_InOut
 * */
bool CString::RemoveElementsFromArray(Array<CString, 1> &CS_A1_InOut, 
                                      const Array<int, 1> &I_A1_ElementsToRemove_In) const{
  if (I_A1_ElementsToRemove_In.size() >= CS_A1_InOut.size()){
    cout << "CString::RemoveElementFromArray: ERROR: I_A1_ElementsToRemove_In.size()=" << I_A1_ElementsToRemove_In.size() << " >= CS_A1_InOut.size()=" << CS_A1_InOut.size() << endl;
    return false;
  }
  Array<CString, 1> CS_A1_Temp(CS_A1_InOut.size()-I_A1_ElementsToRemove_In.size());
  int I_El = 0;
  bool B_Found = false;
  for (int i_el=0; i_el<CS_A1_InOut.size(); i_el++){
    B_Found = false;
    for (int i_elrem=0; i_elrem<I_A1_ElementsToRemove_In.size(); i_elrem++){
      if (i_el == I_A1_ElementsToRemove_In(i_elrem)){
        B_Found = true;
      }
    }
    if (!B_Found){
      CS_A1_Temp(I_El).Set(CS_A1_InOut(i_el));
      I_El++;
    }
  }
  CS_A1_InOut.resize(CS_A1_Temp.size());
  CS_A1_InOut = CS_A1_Temp;
  return true;
}
