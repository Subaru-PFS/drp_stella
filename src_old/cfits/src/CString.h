/*
author: Andreas Ritter
created: 01/12/2007
last edited: 01/12/2007
compiler: g++ 4.0
basis machine: Ubuntu Linux 6.06

TODO: Memory leaks in + operators

*/

#ifndef __CSTRING_H__
#define __CSTRING_H__

//#define __DEBUG_STRING__
//#define __DEBUG_STRING_SET__
//#define __DEBUG_STRING_COUNTCOLS__
//#define __DEBUG_STRING_TRIMCHAR__
//#define __DEBUG_STRING_ATOD__
//#define __DEBUG_STRING_ITOA__
//#define __DEBUG_STRING_DTOA__

#include <iostream>
#include <sstream>
#include <cstdlib>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <blitz/array.h>
#include <blitz/vector.h>
#include <blitz/vector-et.h>
#include <blitz/vecwhere.h>
#include <blitz/tinyvec.h>
#include <blitz/tinyvec-et.h>
#include "CAny.h"

#define STRLEN 255

  using namespace std;

  class CString: public CAny {
    private:
      char* P_String;

    protected:

    public:
    /** Constructors:
        =============

        Standard constructor
        Constructs an object of 'CString' and sets <P_String[0]> to '\0'.
    **/
      CString();

    /** Constructs an object of type 'CString' and sets <P_String> to <p_char>.
    **/
      CString(const char* p_char);

    /** Constructs an object of type 'CString' and sets <P_String> to <str>.
     **/
      CString(const string str);

    /** Copy constructor
        Constructs an object of 'CString' with the values of <cstr>, if <cstr> is 'ClassInvariant()', else starts standard constructor.
    **/
      CString(const CString &cstr);

    /** Destructor:
        ===========

        Destructs the object.
    **/
      virtual ~CString();

    /** Methods:
        ========**/

        /** task   : clones the object
            require: none
            ensure : 'EqualValue(...)' is TRUE**/
      //virtual CAny* Generate() const;

        /**
        task   : Proves the invariance.
        require: none
        ensure : Returns "TRUE", if the invariance is valid (1900<'Year'<2200,
                'Month', 'Day', 'Hour', 'Minute' and 'Second' valid),
                else returns "FALSE".
    **/
      virtual bool ClassInvariant() const;

    /** task   : Compares the object's attributes.
        require: Both objects are 'ClassInvariant()'.
        ensure : Returns "TRUE", if the objects' attributes are equal,
                 else returns "FALSE".
    **/
      virtual bool EqualValue(const CAny &any) const;

    /** task   : Copies the object 'any', if it is possible.
        require: Objects are not 'Equal()' and 'any' is a 'ClassInvariant()'
                 instance of 'CString'.
        ensure : Returns "FALSE", if 'any' is not 'ClassInvariant()' or
                 the objects are 'Equal()', else returns "TRUE".
    **/
      virtual bool Copy(const CAny &any);

    /** Operators:
        ==========

        task   : Like 'Copy' but 'dat' doesn't have to be 'ClassInvariant()'
        require: none
        ensure : works
    **/
      CString& operator=(const CString &cstr);

    /** Operator "==".
        task   : Compares the values of two objects of the class 'CString'.
        require: Both objects are 'ClassInvariant()' instances of 'CString'.
        ensure : Returns "TRUE", if 'this' is earlier than or equal to 'dat',
                 else returns "FALSE".
    **/
      bool operator==(const CAny &any) const;

    /** Operator "!=".
      task   : Compares the values of two objects of the class 'CString'.
      require: Both objects are 'ClassInvariant()' instances of 'CString'.
      ensure : Returns "TRUE", if 'this' is earlier than or equal to 'dat',
      else returns "FALSE".
    **/
      bool operator!=(const CAny &any) const;

    /** Operator "+"
      task   : Concatenates two instances of 'CString'.
      require: .
      ensure : .
     **/
      CString& operator+(const CString &cstr);

    /** Operator "+"
      task   : Concatenates 'CString' and 'cstr'.
      require: .
      ensure : .
     **/
      CString& operator+(const char cha);

    /** Operator "+="
      task   : Concatenates 'this' and 'cstr'.
      require: .
      ensure : .
     **/
      CString& operator+=(const CString &p_cstr);

    /** Operator "+="
      task   : Concatenates 'this' and 'cstr'.
      require: .
      ensure : .
     **/
      CString& operator+=(const char cstr);

    /** Operator "[]"
        task   : .
        require: .
        ensure : .
    **/
      char* operator[](int i);

    /** Set methods
        ===========
        task   : Set the specified value.
        require: Values are valid, NOT "", str.length() must be > 0.
        ensure : Return "TRUE", if the specified values are valid, else set the
                 value of the specified attribute to the nearest valid one and
                 return "FALSE".
    **/
      bool Set(const char* p_char);
      bool Set(const string str);
      bool Set(const CString &C_Str);
      bool SetNum(double dbl);

    // Get methods
    // ===========
    // task   : Return the specified value.
    // require: none
    // ensure : work
      char* Get() const;
      std::string GetString() const;
      char* GetPChar() const;

      /**
      Returns number of characters in this->P_String before ending '\0'.
      **/
      int GetLength() const;

      /**
      String manipulation methods
      **/
      /// Trim: Mode == 0: Remove leading spaces or Char's
      ///       Mode == 1: Remove trailing spaces or Char's
      ///       Mode == 2: Remove leading and trailing spaces or Char's
      void TrimChar(const char Char, int Mode);
      void TrimChar(const CString &CS_Char, int Mode);
      void Trim(int Mode);

      CString* SubString(int Start, int End) const;
      CString* SubString(int Start) const;

      bool Add(const CString &CS_ToAdd);

      /**
      function int CharrPosInStr(char *P_InStr: in, const char LookFor: in)
      Returns first position of String 'LookFor' in 'P_InStr', looking from position Start, if found, else '-1'
      **/
      int CharPosFrom(const char LookFor, int Start) const;

      int StrPosInCharArr(const char inarr[255], const char* lookfor) const;
      int StrPosInCharArrFrom(const char inarr[255], const char* lookfor, int startpos) const;

      /**
      function int CharrPos(const char LookFor: in)
      Returns first position of String 'P_LookFor' in 'this', if
      found, else '-1'
      **/
      int CharPos(const char LookFor) const;

      int CharPosInCharArrFrom(const char inarr[255], const char lookfor, int start) const;

      int CharPosInCharArr(const char inarr[255], const char lookfor) const;
      int LastCharPosInCharArr(const char inarr[255], const char lookfor) const;

      /**
       function int StrPos(const char* P_LookFor: in)
       Returns first position of String 'P_LookFor' in 'P_InStr', if
       found, else '-1'
      **/
      int StrPos(const char* P_LookFor) const;

      /**
       function int StrPosInStr(char *P_InStr: in, const char* P_LookFor: in)
       Returns first position of String 'P_LookFor' in 'P_InStr', if
       found, else '-1'
      **/
      int StrPos(const CString &LookFor) const;

      /**
       function int LastCharPosInStr(char *P_InStr: in, const char LookFor: in)
       Returns last position of character 'P_LookFor' in 'Str', if
       found, else '-1'
      **/
      int LastCharPosInString(const string InStr, const char LookFor) const;

      /**
       function int LastCharPosInStr(char *P_InStr: in, const char LookFor: in)
       Returns last position of character 'P_LookFor' in 'P_InStr',
       if found, else '-1'
      **/
      int LastCharPos(const char LookFor) const;

      /**
       function int LastStrPosInStr(const char* p_LookFor: in)
       Returns last position of character 'p_LookFor' in
       'this->P_String', if found, else '-1'
      **/
      int LastStrPos(const char* p_LookFor) const;

      /**
       function int LastStrPosInStr(const CString &LookFor: in)
       Returns last position of character 'LookFor' in
       'this->P_String', if found, else '-1'
      **/
      int LastStrPos(const CString &LookFor) const;

      /**
       function int StrPosInStrFrom(char *P_InStr: in, char* P_LookFor: in, int Start)
       Returns first position of String 'P_LookFor' in 'P_InStr',
       looking from position Start, if found, else '-1'
      **/
      int StrPosFrom(const char* P_LookFor, int Start) const;

      /**
       function bool InsertAt(const CString &cs: in, int Pos: in)
       Inserts CString 'cs' in 'P_String' at position Pos and returns
       TRUE, if successfull, else returns FALSE.
      **/
      bool InsertAt(const CString &CS, int Pos);

      char* StringToPChar(const string &str) const;
      string PCharToString(const char* p_str) const;
      int IsDouble() const;
      int IsInt() const;
      int IsNumber() const;
      CString* IToA(int I_In);
      CString* DToA(double D_In, int I_Precision);
      double AToD() const;
      bool AToD(double &D_Out) const;
      int AToI() const;
      bool AToI(int &I_Out) const;
      bool AToD(const blitz::Array<CString, 1> &CS_A1_In, blitz::Array<double, 1> &D_A1_Out) const;
      CString* IntToCString(int I_In);

      /**
       function int CountLines(const CString &fnc: inout)
       Returns number of lines of file <fnc>.
      **/
      long CountLines(const CString &fnc) const;

      /**
       function int CountDataLines(const CString &fnc: inout)
       Returns number of lines which do not start with '#' of file <fnc>.
       **/
      long CountDataLines(const CString &fnc) const;

      /**
      function int CountCols(const CString &fnc: inout)
      Returns number of columns of file <fnc>.
       **/
      long CountCols(const CString &CS_FileName_In, const CString &CS_Delimiter) const;

      /// Replaces all occurences CString <CS_ToReplace> in this with <CS_Replace>
      /// Returns a copy of itself if CS_ToReplace_In is not found in this
      CString* StrReplace(const CString& CS_ToReplace, const CString& CS_Replace);
      
      /// Replaces all occurences CString <CS_ToReplace> in this with <CS_Replace> and writes result to 
      /// CS_Out
      /// Returns false if CS_ToReplace_In is not found in this
      bool StrReplace(const CString& CS_ToReplace_In, const CString& CS_Replace_In, CString &CS_Out);
        
      /// Replaces all occurences of CString <CS_ToReplace_In> in textfile <CS_TextFile_In> with
      /// <CS_Replace_In> and writes result to <CS_TextFile_Out>
      bool StrReplaceInList(const CString &CS_TextFile_In,
			    const CString &CS_ToReplace_In,
			    const CString &CS_Replace_In,
			    const CString &CS_TextFile_Out) const;

      bool StrReplaceInList(const blitz::Array<CString, 1> &CS_A1_In,
                            const CString &CS_ToReplace_In,
                            const CString &CS_Replace_In,
                            blitz::Array<CString, 1> &CS_A1_Out) const;

      bool ReadFileToStrArr(const CString &CS_FileName_In,
                            blitz::Array<CString, 2> &CS_A2_Out,
                            const CString &CS_Delimiter) const;

      bool ReadFileToDblArr(const CString &CS_FileName_In,
                            blitz::Array<double, 2> &D_A2_Out,
                            const CString &CS_Delimiter) const;

      bool ReadFileLinesToStrArr(const CString &CS_FileName_In,
                                 blitz::Array<CString, 1> &CS_A1_Out) const;

      bool FileAccess(const CString &fn) const;

      bool IsList() const;
      
      /// Checks CS_A1_In and returns an integer array with 1 where CS_A1_In(i).EqualValue(CS_Comp),
      /// otherwise 0
      blitz::Array<int, 1>* Where(const blitz::Array<CString, 1> &CS_A1_In,
                           const CString &CS_Comp) const;
                           
      /** Removes element at position I_Pos from CS_A1_InOut
       * */
      bool RemoveElementFromArray(blitz::Array<CString, 1> &CS_A1_InOut, int I_Pos) const;
      
      /** Removes elements listed in I_A1_ElementsToRemove_In from CS_A1_InOut
       * */
      bool RemoveElementsFromArray(blitz::Array<CString, 1> &CS_A1_InOut, const blitz::Array<int, 1> &I_A1_ElementsToRemove_In) const;
      
      bool WriteStrListToFile(const blitz::Array<CString, 1> &CS_A1_In, const CString &CS_FileNameOut_In) const;
      bool WriteStrListToFile(const blitz::Array<CString, 2> &CS_A2_In, const CString &CS_Delimiter, const CString &CS_FileNameOut_In) const;

      bool AddFirstPartAsDir(const blitz::Array<CString, 1> &CS_A1_In, blitz::Array<CString, 1> &CS_A1_Out) const;

      bool AddNameAsDir(const blitz::Array<CString, 1> &CS_A1_In, blitz::Array<CString, 1> &CS_A1_Out) const;

      bool MkDir(const CString &CS_PathName) const;

      /// task   : returns 'ClassName'
      //virtual char* GetClassName() const;

      /// task   : returns the Key of the object
      //virtual char* KeyOf() const;

      /// Show methods:
      /// =============

      /// task   : Shows the object's attributes at 'os'.
      /// require: none
      /// ensure : works
      virtual void Show(std::ostream &os) const;

      /// Persistence methods:
      /// ====================

      /// task   : write to a stream
      //virtual void WriteData(std::iostream &stream) const;

      /// task   : read from a stream
      //virtual bool ReadData(std::ostream &stream);

  };
#endif
