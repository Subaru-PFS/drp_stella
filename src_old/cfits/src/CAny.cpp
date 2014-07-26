/*
author: Andreas Ritter
created: 01/12/2007
last edited: 01/12/2007
compiler: g++ 4.0
basis machine: Ubuntu Linux 6.06
*/
#include "CAny.h"

CAny::CAny(){
}

/*************************************************************/

CAny::~CAny(){
}

/*************************************************************/

bool CAny::Equal(const CAny &any) const{
  return this == &any;
}

/*************************************************************/

bool CAny::operator==(const CAny &any) const{
  return EqualValue(any);
}

/*************************************************************/

bool CAny::operator!=(const CAny &any) const{
  return !EqualValue(any);
}

/*************************************************************/

void CAny::GetClassName(char *cn) const{
  int pos = 0;
  do
  {
    cn[pos] = ClassName[pos];
    pos++;
  } while (cn[pos-1] != '\0');
}

/*************************************************************/

char* CAny::GetClassName() const{
  char *tempStr = new char[255];
  this->GetClassName(tempStr);
  return tempStr;
}

/*************************************************************/

void CAny::Show() const{
  std::cout << *this;
}

/*************************************************************/

std::ostream& operator<<(std::ostream &os, const CAny &any){
  any.Show(os);
  return os;
}
