/*
author: Andreas Ritter
created: 01/12/2007
last edited: 01/12/2007
compiler: g++ 4.0
basis machine: Ubuntu Linux 6.06
*/

#ifndef __CANY_H__
#define __CANY_H__

  #include <iostream>
  #include <string>
  #include <stdio.h>
  #include <cstdlib>
  #include <c++/4/bits/stdc++.h>

using namespace std;

class CAny{
  private:

  protected:
    char* ClassName;

    // Constructors:
    // =============

    // Standard constructor
    // Constructs on object of 'CAny' and sets the values of the attributes
    // to standard values.
    // Note: Some of the instances constructed with the standard constructor
    //       are not ClassInvariant()!
    CAny();

  public:
    // Destructor:
    // ===========
    // Destructs the object.
    virtual ~CAny();

    // Methods:
    // ========

    // task   : Proves the invariance.
    // require: none
    // ensure : Returns "TRUE", if the invariance is valid (strings are
    //          not empty, ...), else returns "FALSE".
    virtual bool ClassInvariant() const = 0;

    // task   : Compares the memory addresses of two objects
    // require: none
    // ensure : Returns "TRUE", if the objects' mem addresses are equal,
    //          else returns "FALSE".
    bool Equal(const CAny &any) const;

    // task   : Compares the object's attributes.
    // require: Both objects are 'ClassInvariant()'.
    // ensure : Returns "TRUE", if the objects' attributes are equal,
    //          else returns "FALSE".
    virtual bool EqualValue(const CAny &any) const = 0;

    // task   : Copies the object 'any', if it is possible.
    // require: Objects are not 'Equal()' and 'any' is 'ClassInvariant()'.
    // ensure : Returns "FALSE", if 'any' is not 'ClassInvariant()' or
    //          the objects are 'Equal()', else returns "TRUE".
    virtual bool Copy(const CAny &any) = 0;

    // Operators:
    // ==========

    // task   : Comparison operator (another 'EqualValue()').
    // require: Both objects are 'ClassInvariant()'.
    // ensure : Returns "TRUE", if the objects' attributes are equal,
    //          else returns "FALSE".
    bool operator==(const CAny &any) const;

    // task   : Neqative comparison operator.
    // require: Both objects are 'ClassInvariant()'.
    // ensure : Returns "TRUE", if the objects' attributes are not equal,
    //          else returns "FALSE".
    bool operator!=(const CAny &any) const;

    // Get-/Set-methods:
    // =================

    // task   : Returns the 'ClassName' of the object.
    // require: none
    // ensure : works
    void GetClassName(char *cn) const;

    // task   : Returns the 'ClassName' of the object.
    // require: none
    // ensure : works
    char* GetClassName() const;

    // Show methods:
    // =============

    // task   : Shows the object's attributes at 'cout'
    // require: none
    // ensure : works
    void Show() const;

    // task   : Shows the object's attributes at 'os'.
    // require: none
    // ensure : works
    virtual void Show(std::ostream &os) const = 0;
};

// task   : Output operator.
// require: none
// ensure : works
std::ostream& operator<<(std::ostream &os, const CAny &any);

#endif
