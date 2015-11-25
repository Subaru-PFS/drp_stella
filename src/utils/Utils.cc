#include "pfs/drp/stella/utils/Utils.h"
namespace pfs { namespace drp { namespace stella { namespace utils{
  int KeyWord_Set(vector<string> const& keyWords_In,
                  string const& str_In){
    for (int m = 0; m < int(keyWords_In.size()); ++m){
      if (keyWords_In[m].compare(str_In) == 0)
        return m;
    }
    return -1;
  }
  
  bool trimString(string &str, const int mode){
    return trimString(str, ' ', mode);
  }

  bool trimString(string &str, const char chr, const int mode){
    if ((mode == 0) || (mode == 2)){
      while (str.find(chr) == 0){
        str.erase(0,1);
      }
    }
    if ((mode == 1) || (mode == 2)){
      while (str.rfind(chr) == str.length()-1){
        str.erase(str.length()-1);
      }
    }
    return true;
  }

  bool sToD(const string &str, double &D_Out){
    D_Out = 0;
    for (unsigned int i=0; i<str.length(); i++){
      if ((str[i] != '0') &&
        (str[i] != '1') &&
        (str[i] != '2') &&
        (str[i] != '3') &&
        (str[i] != '4') &&
        (str[i] != '5') &&
        (str[i] != '6') &&
        (str[i] != '7') &&
        (str[i] != '8') &&
        (str[i] != '9') &&
        (str[i] != '-') &&
        (str[i] != '+') &&
        (str[i] != 'e') &&
        (str[i] != 'E') &&
        (str[i] != '.')){
        cout << "CFits::AToI: ERROR: str[i=" << i << "] = <" << str[i] << "> is not a number => Returning FALSE" << endl;
        return false;
        }
    }
    //  D_Out = stod(str);
    D_Out = double(atof(str.c_str()));
    return true;
  }

  /** ********************************************************************/

  bool sToI(const string &str, int &I_Out){
    I_Out = 0;
    for (unsigned int i=0; i<str.length(); i++){
      if ((str[i] != '0') &&
          (str[i] != '1') &&
          (str[i] != '2') &&
          (str[i] != '3') &&
          (str[i] != '4') &&
          (str[i] != '5') &&
          (str[i] != '6') &&
          (str[i] != '7') &&
          (str[i] != '8') &&
          (str[i] != '9') &&
          (str[i] != '-') &&
          (str[i] != '+') &&
          (str[i] != 'e') &&
          (str[i] != 'E') &&
          (str[i] != '.')){
        cout << "sToI: ERROR: str[i=" << i << "] = <" << str[i] << "> is not a number => Returning FALSE" << endl;
        return false;
      }
    }
    I_Out = int(atoi(str.c_str()));
    return true;
  }

  bool FileAccess(const string &fn){
    FILE *ffile;

    ffile = fopen(fn.c_str(), "r");
    if (ffile == NULL)
    {
      cout << "FileAccess: Failed to open file fname (" << fn << ")" << endl;
      return false;
    }
    fclose(ffile);
    return true;
  }

  long countLines(const string &fnc){
    if (fnc.length() > 255){
      cout << "countLines: ERROR: input file name = <" << fnc << "> too long => Returning -1" << endl;
      return -1;
    }
    FILE *ffile;
    long nelements;
    char oneword[255];
    char fname[255];
    char *line;
    strcpy(fname, fnc.c_str());

    #ifdef __DEBUG_FITS__
      printf("countLines: function started\n");
    #endif
    ffile = fopen(fname, "r");
    if (ffile == NULL){
      cout << "countLines: Failed to open file fname (=<" << fname << ">)" << endl;
      return -1;
    }
    #ifdef __DEBUG_FITS__
      printf("countLines: File fname(=<%s>) opened\n", fname);
    #endif

    nelements = 0;
    // --- read file <fname> named <ffile>
    do{
      line = fgets(oneword, 255, ffile);
      if (line != NULL){
        // --- count lines
        nelements++;
      }
    } while (line != NULL);
    #ifdef __DEBUG_FITS__
      printf("countLines: File fname(=<%s>) contains %d data lines\n",fname,nelements);
    #endif
    // --- close input file
    fclose(ffile);
    #ifdef __DEBUG_FITS__
      printf("countLines: File fname (=<%s>) closed\n", fname);
    #endif
    return nelements;
  }

  /**
    * function int CountDataLines(const string &fnc: in)
    * Returns number of lines which do not start with '#' of file <fnc>.
    **/
  long countDataLines(const string &fnc){
    FILE *ffile;
    long nelements;
    char oneword[255];
    char fname[255];
    char *line;
    strcpy(fname, fnc.c_str());

    #ifdef __DEBUG_FITS__
      printf("countDataLines: function started\n");
    #endif
    ffile = fopen(fname, "r");
    if (ffile == NULL)
    {
      cout << "countDataLines: Failed to open file fname (=<" << fname << ">)" << endl;
      return -1;
    }
    #ifdef __DEBUG_FITS__
      printf("CountDataLines: File fname(=<%s>) opened\n", fname);
    #endif

    nelements = 0;
    // --- read file <fname> named <ffile>
    do{
      line = fgets(oneword, 255, ffile);
      if (line != NULL && line[0] != '#'){
        // --- count lines
        nelements++;
      }
    }
    while (line != NULL);
    #ifdef __DEBUG_FITS__
      printf("countDataLines: File fname(=<%s>) contains %d data lines\n",fname,nelements);
    #endif
    // --- close input file
    fclose(ffile);
    #ifdef __DEBUG_FITS__
      printf("countDataLines: File fname (=<%s>) closed\n", fname);
    #endif
    return nelements;
  }

  /**
    * function int CountCols(const string &fnc: in, const string delimiter: in)
    * Returns number of columns of file <fnc>.
    **/
  long countCols(const string &fileName_In, const string &delimiter){
    long L_Cols = 0;
    long L_OldCols = 0;
    string tempLine = " ";
    FILE *ffile;
    long I_Row;
    char oneword[255];
    char fname[255];
    char *line;
    char tempchar[255];
    tempchar[0] = '\n';
    tempchar[1] = '\0';
    strcpy(fname, fileName_In.c_str());
    string sLine;

    #ifdef __DEBUG_FITS_COUNTCOLS__
      printf("countCols: function started\n");
    #endif
    ffile = fopen(fname, "r");
    if (ffile == NULL){
      cout << "countCols: Failed to open file fname (=<" << fname << ">)" << endl;
      return -1;
    }
    #ifdef __DEBUG_FITS_COUNTCOLS__
      printf("countCols: File fname(=<%s>) opened\n", fname);
    #endif

    I_Row = 0;
    // --- read file <fname> named <ffile>
    size_t I_Pos;
    string sTemp;
    do{
      line = fgets(oneword, 255, ffile);
      if (line != NULL){
        // --- count lines
        sLine = line;
        I_Pos = sLine.find('\n');
        #ifdef __DEBUG_FITS_COUNTCOLS__
          cout << "countCols: I_Row = " << I_Row << ": I_Pos(tempchar) set to " << I_Pos << endl;
        #endif
        if (I_Pos != string::npos){
          sLine.copy(tempchar,sLine.length(),0);
          tempchar[I_Pos] = '\0';
          sLine = tempchar;
          #ifdef __DEBUG_FITS_COUNTCOLS__
            cout << "countCols: I_Row = " << I_Row << ": sLine set to <" << sLine << ">" << endl;
          #endif
        }
        I_Pos = sLine.find("#");
        if (I_Pos == string::npos){
          L_Cols = 0;
          sTemp = sLine.substr(I_Pos+1);
          while(sTemp.substr(0,1).compare(delimiter) == 0)
            sTemp.erase(0,1);
          while(sTemp.substr(sTemp.length()-1,1).compare(delimiter) == 0)
            sTemp.erase(sTemp.length()-1,1);
          sLine = sTemp;
          while (sLine.find(delimiter) != string::npos){
            L_Cols++;
            #ifdef __DEBUG_FITS_COUNTCOLS__
              cout << "countCols: I_Row = " << I_Row << ": L_Cols set to " << L_Cols << endl;
            #endif
            I_Pos = sLine.find(delimiter);
            #ifdef __DEBUG_FITS_COUNTCOLS__
              cout << "countCols: I_Row = " << I_Row << ": I_Pos set to " << I_Pos << endl;
            #endif
            sTemp = sLine.substr(I_Pos+1);
            #ifdef __DEBUG_FITS_COUNTCOLS__
              cout << "countCols: I_Row = " << I_Row << ": sTemp set to <" << sTemp << ">" << endl;
            #endif
            while(sTemp.substr(0,1).compare(delimiter) == 0)
              sTemp.erase(0,1);
            while(sTemp.substr(sTemp.length()-1,1).compare(delimiter) == 0)
              sTemp.erase(sTemp.length()-1,1);
            #ifdef __DEBUG_FITS_COUNTCOLS__
              cout << "countCols: I_Row = " << I_Row << ": sTemp set to <" << sTemp << ">" << endl;
            #endif
            sLine = sTemp;
            #ifdef __DEBUG_FITS_COUNTCOLS__
              cout << "countCols: I_Row = " << I_Row << ": CS_Line set to <" << sLine << ">" << endl;
            #endif
          }
          L_Cols++;
          #ifdef __DEBUG_FITS_COUNTCOLS__
            cout << "countCols: I_Row = " << I_Row << ": L_Cols set to " << L_Cols << endl;
          #endif
        }
        if (L_Cols > L_OldCols){
          L_OldCols = L_Cols;
          #ifdef __DEBUG_FITS_COUNTCOLS__
            cout << "countCols: L_Cols = " << L_Cols << endl;
          #endif
        }
        I_Row++;
      }
    } while (line != NULL);
    #ifdef __DEBUG_FITS__
      printf("countCols: File fname(=<%s>) contains %d data lines\n",fname,I_Row);
    #endif
    // --- close input file
    fclose(ffile);
    #ifdef __DEBUG_FITS__
      printf("countCols: File fname (=<%s>) closed\n", fname);
    #endif
    return L_OldCols;
  }

  template<typename T>
  ndarray::Array<T, 2, 1> get2DndArray(T nRows, T nCols){
    ndarray::Array<T, 2, 1> out = ndarray::allocate(nRows, nCols);
    out[ndarray::view()()] = 0;
    return out;
  }

  template<typename T>
  ndarray::Array<T, 1, 1> get1DndArray(T size){
    ndarray::Array<T, 1, 1> out = ndarray::allocate(size);
    out[ndarray::view()] = 0;
    return out;
  }

  template<typename T>
  std::vector<T> copy(const std::vector<T> &vecIn){
    std::vector<T> vecOut = vecIn;
    return vecOut;
  }

  template< typename T > 
  std::string numberToString_dotToUnderscore( T number, int accuracy ){
    string out;
    if ( ( number > 1.e5 ) || ( ( number < 1.e-5 ) && ( number > 1.e-15 ) ) ){
      char buffer[ 255 ];
      sprintf( buffer, "%.10e", number );
      string str( buffer );
      out = str;
    }
    else{
      string str = std::to_string( number );
      out = str;
    }
    cout << "out = <" << out << ">" << endl;
    int dotPos = out.find( "." );
    cout << "dotPos = " << dotPos << endl;
    if ( dotPos != std::string::npos ){
      out.replace( dotPos, 1, "_" );
      size_t pos = out.find("e");
      string ex;
      if ( pos != std::string::npos ){
        ex = out.substr(pos);
        cout << "ex = <" << ex << ">" << endl;
      }
      if ( ( dotPos + accuracy + 1 ) < out.length() ){
        if ( accuracy == 0 )
          out.resize( dotPos );
        else
          out.resize( dotPos + accuracy + 1 );
        cout << "out = <" << out << ">" << endl;
      }
      if ( pos != std::string::npos ){
        out.append( ex );
        cout << "out = <" << out << ">" << endl;
      }
    }
    return out;
  }

  std::string dotToUnderscore( std::string number, int accuracy ){
    string out(number);
    int dotPos = out.find( "." );
    cout << "dotPos = " << dotPos << endl;
    if ( dotPos != std::string::npos ){
      out.replace( dotPos, 1, "_" );
      size_t pos = out.find("e");
      if (pos == std::string::npos )
        pos = out.find("E");
      string ex;
      if ( pos != std::string::npos ){
        ex = out.substr(pos);
        cout << "ex = <" << ex << ">" << endl;
      }
      if ( ( dotPos + accuracy + 1 ) < out.length() ){
        if ( accuracy == 0 )
          out.resize( dotPos );
        else
          out.resize( dotPos + accuracy + 1 );
        cout << "out = <" << out << ">" << endl;
      }
      if ( pos != std::string::npos ){
        out.append( ex );
        cout << "out = <" << out << ">" << endl;
      }
    }
    return out;
  }
  
  template std::string numberToString_dotToUnderscore( float, int );
  template std::string numberToString_dotToUnderscore( double, int );
  
//    template<typename ImageT, typename MaskT, typename VarianceT>
//    PTR(afwImage::MaskedImage<ImageT, MaskT, VarianceT>) getShared(afwImage::MaskedImage<ImageT, MaskT, VarianceT> const &maskedImage){
//      PTR(afwImage::MaskedImage<ImageT, MaskT, VarianceT>) ptr = PTR(const new afwImage::MaskedImage<ImageT, MaskT, VarianceT>(maskedImage));
//      return ptr;
//    }
  template std::vector<int> copy(const std::vector<int>&);
  template std::vector<float> copy(const std::vector<float>&);
  template std::vector<double> copy(const std::vector<double>&);

  template ndarray::Array<size_t, 1, 1> get1DndArray(size_t);
  template ndarray::Array<unsigned short, 1, 1> get1DndArray(unsigned short);
  template ndarray::Array<int, 1, 1> get1DndArray(int);
  template ndarray::Array<float, 1, 1> get1DndArray(float);
  template ndarray::Array<double, 1, 1> get1DndArray(double);
  template ndarray::Array<size_t, 2, 1> get2DndArray(size_t, size_t);
  template ndarray::Array<unsigned short, 2, 1> get2DndArray(unsigned short, unsigned short);
  template ndarray::Array<int, 2, 1> get2DndArray(int, int);
  template ndarray::Array<float, 2, 1> get2DndArray(float, float);
  template ndarray::Array<double, 2, 1> get2DndArray(double, double);
}
}}}
