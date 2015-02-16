#include "pfs/drp/stella/utils/Utils.h"
namespace pfs { namespace drp { namespace stella { namespace utils{

  /**
    *       Returns Position of <str_In> in Array of strings <keyWords_In>, if <keyWords_In> contains string <str_In>, else returns -1.
    **/
  int KeyWord_Set(const blitz::Array<string, 1> &keyWords_In,
                  const string &str_In){
    for (int m = 0; m < static_cast<int>(keyWords_In.size()); m++){
      if (keyWords_In(m).compare(str_In) == 0)
        return m;
    }
    return -1;
  }
  int KeyWord_Set(vector<string> const& keyWords_In,
                  string const& str_In){
    for (int m = 0; m < int(keyWords_In.size()); ++m){
      if (keyWords_In[m].compare(str_In) == 0)
        return m;
    }
    return -1;
  }

  template<typename T>
  bool WriteFits(const blitz::Array<T,1>* image_In, const string &fileName_In){
    blitz::Array<T, 2> image(image_In->size(), 1);
    image(blitz::Range::all(), 0) = (*image_In);
    return WriteFits(&image, fileName_In);
  }

  template<typename T>
  bool WriteFits(const blitz::Array<T,2>* image_In, const string &fileName_In){
    fitsfile *P_Fits;
    int Status;
    long fpixel, nelements;
    void *p_void;

    Status=0;
    remove(fileName_In.c_str());
    fits_create_file(&P_Fits, fileName_In.c_str(), &Status);//{
    if (Status !=0){
      cout << "CFits::WriteFits: Error <" << Status << "> while creating file " << fileName_In << endl;
      char* P_ErrMsg = new char[255];
      ffgerr(Status, P_ErrMsg);
      cout << "CFits::WriteFits: <" << P_ErrMsg << "> => Returning FALSE" << endl;
      delete[] P_ErrMsg;
      return false;
    }

    ///  fits_write_img(P_FitsFile, TDOUBLE, fpixel, nelements,
    ///    p_void, &Status);
    long naxes[2] = {image_In->cols(), image_In->rows()};
    int naxis = 2;
    fits_create_img(P_Fits, DOUBLE_IMG, naxis, naxes, &Status);
    if (Status !=0){
      cout << "CFits::WriteFits: Error <" << Status << "> while creating image " << fileName_In << endl;
      char* P_ErrMsg = new char[255];
      ffgerr(Status, P_ErrMsg);
      cout << "CFits::WriteFits: <" << P_ErrMsg << "> => Returning FALSE" << endl;
      delete[] P_ErrMsg;
      return false;
    }
    #ifdef __DEBUG_WRITEFITS__
      cout << "CFits::WriteFits: size of image_In = <" << image_In->rows() << "x" << image_In->cols() << ">" << endl;
      cout << "CFits::WriteFits: size of image_In = <" << image_In->rows() << "x" << image_In->cols() << ">" << endl;
    #endif

    fpixel = 1;
    //nullval = 0.;
    nelements = image_In->cols() * image_In->rows();

    p_void = (const_cast<blitz::Array<T, 2>*>(image_In))->data();// = new blitz::Array<double,2>(p_Array, blitz::shape(naxes[0], naxes[1]),
    #ifdef __DEBUG_WRITEFITS__
      cout << "CFits::WriteFits: p_void = <" << (*((double*)p_void)) << ">" << endl;
    #endif

    int nbits = TDOUBLE;
    if (typeid(T) == typeid(short))
        nbits = TSHORT;
    else if (typeid(T) == typeid(unsigned short))
        nbits = TUSHORT;
    else if (typeid(T) == typeid(int))
        nbits = TINT;
    else if (typeid(T) == typeid(unsigned int))
        nbits = TUINT;
    else if (typeid(T) == typeid(long))
        nbits = TLONG;
    else if (typeid(T) == typeid(unsigned long))
        nbits = TULONG;
    else if (typeid(T) == typeid(float))
        nbits = TFLOAT;
    else
        nbits = TDOUBLE;
    fits_write_img(P_Fits, nbits, fpixel, nelements, p_void, &Status);

    if (Status !=0){
      cout << "CFits::WriteFits: Error " << Status << " while writing file " << fileName_In << endl;
      char* P_ErrMsg = new char[255];
      ffgerr(Status, P_ErrMsg);
      cout << "CFits::WriteFits: <" << P_ErrMsg << "> => Returning FALSE" << endl;
      delete[] P_ErrMsg;
      return false;
    }

    fits_close_file(P_Fits, &Status);
    cout << "CFits::WriteFits: FitsFileName <" << fileName_In << "> closed" << endl;
    if (Status !=0){
      cout << "CFits::WriteFits: Error " << Status << " while closing file " << fileName_In << endl;
      char* P_ErrMsg = new char[255];
      ffgerr(Status, P_ErrMsg);
      cout << "CFits::WriteFits: <" << P_ErrMsg << "> => Returning FALSE" << endl;
      delete[] P_ErrMsg;
      return false;
    }
    return true;
  }

/*  template<typename T>
  bool WriteFits(ndarray::Array<T, 2, 2> const& image_In, const string &fileName_In){
    fitsfile *P_Fits;
    int Status;
    long fpixel, nelements;
    void *p_void;

    Status=0;
    remove(fileName_In.c_str());
    fits_create_file(&P_Fits, fileName_In.c_str(), &Status);//{
    if (Status !=0){
      cout << "CFits::WriteFits: Error <" << Status << "> while creating file " << fileName_In << endl;
      char* P_ErrMsg = new char[255];
      ffgerr(Status, P_ErrMsg);
      cout << "CFits::WriteFits: <" << P_ErrMsg << "> => Returning FALSE" << endl;
      delete[] P_ErrMsg;
      return false;
    }

    ///  fits_write_img(P_FitsFile, TDOUBLE, fpixel, nelements,
    ///    p_void, &Status);
    long naxes[2] = {image_In.getShape()[1], image_In.getShape()[0]};
    int naxis = 2;
    fits_create_img(P_Fits, DOUBLE_IMG, naxis, naxes, &Status);
    if (Status !=0){
      cout << "CFits::WriteFits: Error <" << Status << "> while creating image " << fileName_In << endl;
      char* P_ErrMsg = new char[255];
      ffgerr(Status, P_ErrMsg);
      cout << "CFits::WriteFits: <" << P_ErrMsg << "> => Returning FALSE" << endl;
      delete[] P_ErrMsg;
      return false;
    }
    #ifdef __DEBUG_WRITEFITS__
      cout << "CFits::WriteFits: size of image_In = <" << image_In->rows() << "x" << image_In->cols() << ">" << endl;
      cout << "CFits::WriteFits: size of image_In = <" << image_In->rows() << "x" << image_In->cols() << ">" << endl;
    #endif

    fpixel = 1;
    //nullval = 0.;
    nelements = image_In.getShape()[0] * image_In.getShape()[1];

    p_void = (const_cast<ndarray::Array<T, 2, 2>*>(&image_In))->get_data();// = new blitz::Array<double,2>(p_Array, blitz::shape(naxes[0], naxes[1]),
    #ifdef __DEBUG_WRITEFITS__
      cout << "CFits::WriteFits: p_void = <" << (*((double*)p_void)) << ">" << endl;
    #endif

    int nbits = TDOUBLE;
    if (typeid(T) == typeid(short))
        nbits = TSHORT;
    else if (typeid(T) == typeid(unsigned short))
        nbits = TUSHORT;
    else if (typeid(T) == typeid(int))
        nbits = TINT;
    else if (typeid(T) == typeid(unsigned int))
        nbits = TUINT;
    else if (typeid(T) == typeid(long))
        nbits = TLONG;
    else if (typeid(T) == typeid(unsigned long))
        nbits = TULONG;
    else if (typeid(T) == typeid(float))
        nbits = TFLOAT;
    else
        nbits = TDOUBLE;
    fits_write_img(P_Fits, nbits, fpixel, nelements, p_void, &Status);

    if (Status !=0){
      cout << "CFits::WriteFits: Error " << Status << " while writing file " << fileName_In << endl;
      char* P_ErrMsg = new char[255];
      ffgerr(Status, P_ErrMsg);
      cout << "CFits::WriteFits: <" << P_ErrMsg << "> => Returning FALSE" << endl;
      delete[] P_ErrMsg;
      return false;
    }

    fits_close_file(P_Fits, &Status);
    cout << "CFits::WriteFits: FitsFileName <" << fileName_In << "> closed" << endl;
    if (Status !=0){
      cout << "CFits::WriteFits: Error " << Status << " while closing file " << fileName_In << endl;
      char* P_ErrMsg = new char[255];
      ffgerr(Status, P_ErrMsg);
      cout << "CFits::WriteFits: <" << P_ErrMsg << "> => Returning FALSE" << endl;
      delete[] P_ErrMsg;
      return false;
    }
    return true;
  }*/

  /**
    *  task: Writes Array <Array_In> to file <CS_FileName_In>
    **/
  template<typename T, int N>
  bool WriteArrayToFile(const blitz::Array<T, N> &Array_In,
                        const string &S_FileName_In,
                        const string &S_Mode)
  {
    int m, n;
//      ofstream ofs(S_FileName_In.c_str());
    //  FILE *p_file;
    FILE *p_file;
    p_file = fopen(S_FileName_In.c_str(), "w");

    if (S_Mode.compare(string("binary")) == 0){
      if (N == 1){
        fwrite(Array_In.data(), sizeof(T), Array_In.size(), p_file);
      }
      else if (N == 2){
        for (m = 0; m < Array_In.rows(); m++)
        {
          fwrite(Array_In(m, blitz::Range::all()).data(), sizeof(T), Array_In.cols(), p_file);
//          for (n = 0; n < D_A2_In.cols(); n++)
//            ofs << D_A2_In(m, n) << " ";
//          ofs << endl;
        }
      }
    }
    else{
      blitz::Array<bool, 1> B_A1_Exp(1);
      bool B_Exp = false;
      if (N == 1){
        if (max(Array_In) < 1e-7)
          B_Exp = true;
      }
      else{
        B_A1_Exp.resize(Array_In.cols());
        B_A1_Exp = false;
        for (m = 0; m < Array_In.cols(); m++){
          if (max(Array_In(blitz::Range::all(), m)) < 1e-7)
            B_A1_Exp(m) = true;
        }
      }
      for (m = 0; m < Array_In.rows(); m++)
      {
        if (N == 1){
          if (!B_Exp){
            if (typeid(T) == typeid(short))
              fprintf(p_file, "%hd\n", static_cast<short>(Array_In(m)));
            else if (typeid(T) == typeid(unsigned short))
              fprintf(p_file, "%u\n", static_cast<unsigned short>(Array_In(m)));
            else if (typeid(T) == typeid(int))
              fprintf(p_file, "%d\n", static_cast<int>(Array_In(m)));
            else if (typeid(T) == typeid(unsigned int))
              fprintf(p_file, "%u\n", static_cast<unsigned int>(Array_In(m)));
            else if (typeid(T) == typeid(long))
              fprintf(p_file, "%ld\n", static_cast<long>(Array_In(m)));
            else if (typeid(T) == typeid(unsigned long))
              fprintf(p_file, "%lu\n", static_cast<unsigned long>(Array_In(m)));
            else if (typeid(T) == typeid(float))
              fprintf(p_file, "%.17f\n", static_cast<float>(Array_In(m)));
            else
              fprintf(p_file, "%.17f\n", static_cast<double>(Array_In(m)));
          }
          else{
            fprintf(p_file, "%.8e\n", static_cast<double>(Array_In(m)));
          }
        }
        else{/// N == 2
          for (n = 0; n < Array_In.cols(); n++){
            if (B_A1_Exp(n))
              fprintf(p_file, " %.8e", double(Array_In(m,n)));
            else{
              if (typeid(T) == typeid(short))
                fprintf(p_file, " %hd", static_cast<short>(Array_In(m,n)));
              else if (typeid(T) == typeid(unsigned short))
                fprintf(p_file, " %u", static_cast<unsigned short>(Array_In(m,n)));
              else if (typeid(T) == typeid(int))
                fprintf(p_file, " %d", static_cast<int>(Array_In(m,n)));
              else if (typeid(T) == typeid(unsigned int))
                fprintf(p_file, " %u", static_cast<unsigned int>(Array_In(m,n)));
              else if (typeid(T) == typeid(long))
                fprintf(p_file, " %ld", static_cast<long>(Array_In(m,n)));
              else if (typeid(T) == typeid(unsigned long))
                fprintf(p_file, " %lu", static_cast<unsigned long>(Array_In(m,n)));
              else if (typeid(T) == typeid(float))
                fprintf(p_file, " %.10f", static_cast<float>(Array_In(m,n)));
              else
                fprintf(p_file, " %.10f", static_cast<double>(Array_In(m,n)));
            }
          }
          fprintf(p_file, "\n");
        }
      }
    }
    fclose(p_file);
    return true;
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

  bool sToD(const blitz::Array<string, 1> &S_A1_In, blitz::Array<double, 1> &D_A1_Out){
    int I_NElements = S_A1_In.size();
    D_A1_Out.resize(I_NElements);
    for (int i=0; i < I_NElements; i++){
      if (!sToD(S_A1_In(i), D_A1_Out(i)))
        return false;
    }
    return true;
  }

  //int sToI(const string &str){
  //  int retVal = int(atoi(str.c_str()));
  //  return retVal;
  //}

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

  bool readFileToStrArr(const string &S_FileName_In,
                        blitz::Array<string, 2> &S_A2_Out,
                        const string &S_Delimiter){
    if (!pfs::drp::stella::utils::FileAccess(S_FileName_In)){
      cout << "readFileToStrArr: ERROR: Access(" << S_FileName_In << ") returned FALSE" << endl;
      return false;
    }

    int I_NLines = pfs::drp::stella::utils::countDataLines(S_FileName_In);
    int I_NCols = pfs::drp::stella::utils::countCols(S_FileName_In, S_Delimiter);
    cout << "readFileToStrArr: " << S_FileName_In << ": I_NLines = " << I_NLines << ", I_NCols = " << I_NCols << endl;
    string templine = " ";
    FILE *ffile;
    char oneword[255];
    char fname[255];
    char *line;
    strcpy(fname, S_FileName_In.c_str());
    string S_Line;

    S_A2_Out.resize(I_NLines, I_NCols);

    #ifdef __DEBUG_FITS_READFILETOSTRARR__
      cout << "readFileToStrArr: function started" << endl;
    #endif
    ffile = fopen(fname, "r");
    if (ffile == NULL){
      cout << "readFileToStrArr: Failed to open file fname (=<" << fname << ">)" << endl;
      return false;
    }
    #ifdef __DEBUG_FITS_READFILETOSTRARR__
      cout << "readFileToStrArr: File fname(=<" << fname << ">) opened" << endl;
    #endif

    int I_Row = 0;
    int I_Col = 0;
    size_t pos;
    string S_Temp;
    do{
      line = fgets(oneword, 255, ffile);
      #ifdef __DEBUG_FITS_READFILETOSTRARR__
        cout << "readFileToStrArr: I_Row = " << I_Row << ": S_Line set to <" << S_Line << ">" << endl;
      #endif

      if (line != NULL){
        S_Line = line;
        #ifdef __DEBUG_FITS_READFILETOSTRARR__
          cout << "readFileToStrArr: 0. S_Line = <" << S_Line << ">" << endl;
        #endif
        if (S_Line.find('\n') == S_Line.length()-1)
          S_Line.erase(S_Line.length()-1);
        #ifdef __DEBUG_FITS_READFILETOSTRARR__
          cout << "readFileToStrArr: 1. S_Line = <" << S_Line << ">" << endl;
        #endif

        pos = S_Line.find("#");
        if (pos != 0){
          #ifdef __DEBUG_FITS_READFILETOSTRARR__
            cout << "readFileToStrArr: 2. S_Line = <" << S_Line << ">" << endl;
          #endif
          trimString(S_Line, 2);
          #ifdef __DEBUG_FITS_READFILETOSTRARR__
            cout << "readFileToStrArr: 3. S_Line = <" << S_Line << ">" << endl;
          #endif
          I_Col = 0;
          while ((pos = S_Line.find(S_Delimiter)) != string::npos){
            pos = S_Line.find(S_Delimiter);
            #ifdef __DEBUG_FITS_READFILETOSTRARR__
              cout << "readFileToStrArr: while: pos set to " << pos << endl;
            #endif
            S_Temp = S_Line.substr(0,pos);
            #ifdef __DEBUG_FITS_READFILETOSTRARR__
              cout << "readFileToStrArr: while: S_Temp set to <" << S_Temp << ">" << endl;
            #endif
            S_A2_Out(I_Row, I_Col) = S_Temp;//)){
            #ifdef __DEBUG_FITS_READFILETOSTRARR__
              cout << "readFileToStrArr: S_A2_Out(" << I_Row << ", " << I_Col << ") set to <" << S_Temp << ">" << endl;
            #endif

            S_Line = S_Line.substr(pos+1);
            #ifdef __DEBUG_FITS_READFILETOSTRARR__
              cout << "readFileToStrArr: while: new S_Line set to " << S_Line << endl;
            #endif
            trimString(S_Line, 2);
            #ifdef __DEBUG_FITS_READFILETOSTRARR__
              cout << "readFileToStrArr: while: after trimming: S_Line set to " << S_Line << endl;
            #endif
            I_Col++;
          }
          #ifdef __DEBUG_FITS_READFILETOSTRARR__
            cout << "readFileToStrArr: end of while: I_Row = " << I_Row << ", I_Col = " << I_Col << ", S_Line set to " << S_Line << endl;
          #endif
          S_A2_Out(I_Row, I_Col) = S_Line;//))){
        }
        I_Row++;
      }
    } while (line != NULL);
    #ifdef __DEBUG_FITS_READFILETOSTRARR__
      cout << "readFileToStrArr: File fname(=<" << fname << ">) contains " << I_Row << " data lines" << endl;
    #endif
    // --- close input file
    fclose(ffile);
    #ifdef __DEBUG_FITS_READFILETOSTRARR__
      cout << "readFileToStrArr: File fname (=<" << fname << ">) closed" << endl;
    #endif
    return true;
  }

  bool readFileToDblArr(const string &S_FileName_In,
                        blitz::Array<double, 2> &D_A2_Out,
                        const string &S_Delimiter){
    blitz::Array<string, 2> S_A2_Arr(2,2);
    if (!readFileToStrArr(S_FileName_In, S_A2_Arr, S_Delimiter)){
      cout << "readFileToDblArr: ERROR: readFileToStrArr returned FALSE" << endl;
      return false;
    }
    D_A2_Out.resize(S_A2_Arr.rows(), S_A2_Arr.cols());
    for (int i_row=0; i_row<S_A2_Arr.rows(); i_row++){
      for (int i_col=0; i_col<S_A2_Arr.cols(); i_col++){
        if (!sToD(S_A2_Arr(i_row, i_col), D_A2_Out(i_row, i_col))){
          cout << "readFileToDblArr: ERROR: could not convert S_A2_Arr(i_row=" << i_row << ", i_col=" << i_col << ")=" << S_A2_Arr(i_row, i_col) << " to double value" << endl;
        }
      }
    }
    return true;
  }

  /** *******************************************************/

  bool readFileLinesToStrArr(const string &S_FileName_In,
                              blitz::Array<string, 1> &S_A1_Out){
    if (!FileAccess(S_FileName_In)){
      cout << "readFileLinesToStrArr: ERROR: Access(" << S_FileName_In << ") returned FALSE" << endl;
      return false;
    }
    int I_NLines = countDataLines(S_FileName_In);
    cout << "readFileLinesToStrArr: " << S_FileName_In << ": I_NLines = " << I_NLines << endl;
    string templine = " ";
    FILE *ffile;
  //  long nelements;
    char oneword[255];
    char fname[255];
    char *line;
    strcpy(fname, S_FileName_In.c_str());
    string S_Line;

    S_A1_Out.resize(I_NLines);

    #ifdef __DEBUG_FITS_READFILELINESTOSTRARR__
      cout << "readFileLinesToStrArr: function started" << endl;
    #endif
    ffile = fopen(fname, "r");
    if (ffile == NULL){
      cout << "readFileLinesToStrArr: Failed to open file fname (=<" << fname << ">)" << endl;
      return false;
    }
    #ifdef __DEBUG_FITS_READFILELINESTOSTRARR__
      cout << "readFileLinesToStrArr: File fname(=<" << fname << ">) opened" << endl;
    #endif

    int I_Row = 0;
    size_t pos;
    string S_Temp;
    do{
      line = fgets(oneword, 255, ffile);
      #ifdef __DEBUG_FITS_READFILELINESTOSTRARR__
        cout << "readFileLinesToStrArr: I_Row = " << I_Row << ": S_Line set to <" << S_Line << ">" << endl;
      #endif

      if (line != NULL){
        S_Line = line;
        #ifdef __DEBUG_FITS_READFILELINESTOSTRARR__
          cout << "readFileLinesToStrArr: 0. S_Line = <" << S_Line << ">" << endl;
        #endif
        if (S_Line.find('\n') == S_Line.length()-1)
          S_Line.erase(S_Line.length()-1);
        #ifdef __DEBUG_FITS_READFILELINESTOSTRARR__
          cout << "readFileLinesToStrArr: 1. S_Line = <" << S_Line << ">" << endl;
        #endif

        pos = S_Line.find("#");
        if (pos != 0){
          #ifdef __DEBUG_FITS_READFILELINESTOSTRARR__
            cout << "readFileLinesToStrArr: 2. S_Line = <" << S_Line << ">" << endl;
          #endif
          S_A1_Out(I_Row) = S_Line;//))){
        }
        I_Row++;
      }
    }
    while (line != NULL);
    #ifdef __DEBUG_FITS_READFILELINESTOSTRARR__
      cout << "readFileLinesToStrArr: File fname(=<" << fname << ">) contains " << I_Row << " data lines" << endl;
    #endif
    // --- close input file
    fclose(ffile);
    #ifdef __DEBUG_FITS_READFILELINESTOSTRARR__
      cout << "readFileLinesToStrArr: File fname (=<" << fname << ">) closed" << endl;
    #endif
    return true;
  }

  template<typename T>
  blitz::Array<T, 2> get2DBlitzArray(T nRows, T nCols){
    blitz::Array<T, 2> out(2,2);
    out.resize(int(nRows), int(nCols));
    out = 0;
    return out;
  }

  template<typename T>
  blitz::Array<T, 1> get1DBlitzArray(T size){
    blitz::Array<T, 1> out(2);
    out.resize(int(size));
    out = 0;
    return out;
  }

  template<typename T>
  ndarray::Array<T, 2, 2> get2DndArray(T nRows, T nCols){
    ndarray::Array<T, 2, 2> out = ndarray::allocate(nRows, nCols);
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
  
//    template<typename ImageT, typename MaskT, typename VarianceT>
//    PTR(afwImage::MaskedImage<ImageT, MaskT, VarianceT>) getShared(afwImage::MaskedImage<ImageT, MaskT, VarianceT> const &maskedImage){
//      PTR(afwImage::MaskedImage<ImageT, MaskT, VarianceT>) ptr = PTR(const new afwImage::MaskedImage<ImageT, MaskT, VarianceT>(maskedImage));
//      return ptr;
//    }
}
template std::vector<int> utils::copy(const std::vector<int>&);
template std::vector<float> utils::copy(const std::vector<float>&);
template std::vector<double> utils::copy(const std::vector<double>&);

template bool utils::WriteFits(const blitz::Array<unsigned short, 2>* image_In, const string &fileName_In);
template bool utils::WriteFits(const blitz::Array<int, 2>* image_In, const string &fileName_In);
template bool utils::WriteFits(const blitz::Array<long, 2>* image_In, const string &fileName_In);
template bool utils::WriteFits(const blitz::Array<float, 2>* image_In, const string &fileName_In);
template bool utils::WriteFits(const blitz::Array<double, 2>* image_In, const string &fileName_In);

//template bool utils::WriteFits(ndarray::Array<unsigned short, 2, 2> const& image_In, const string &fileName_In);
//template bool utils::WriteFits(ndarray::Array<int, 2, 2> const& image_In, const string &fileName_In);
//template bool utils::WriteFits(ndarray::Array<long, 2, 2> const& image_In, const string &fileName_In);
//template bool utils::WriteFits(ndarray::Array<float, 2, 2> const& image_In, const string &fileName_In);
//template bool utils::WriteFits(ndarray::Array<double, 2, 2> const& image_In, const string &fileName_In);

template bool utils::WriteFits(const blitz::Array<unsigned short, 1>* image_In, const string &fileName_In);
template bool utils::WriteFits(const blitz::Array<int, 1>* image_In, const string &fileName_In);
template bool utils::WriteFits(const blitz::Array<long, 1>* image_In, const string &fileName_In);
template bool utils::WriteFits(const blitz::Array<float, 1>* image_In, const string &fileName_In);
template bool utils::WriteFits(const blitz::Array<double, 1>* image_In, const string &fileName_In);

template bool utils::WriteArrayToFile(const blitz::Array<unsigned short, 1> &I_A1_In, const string &S_FileName_In, const string &S_Mode);
template bool utils::WriteArrayToFile(const blitz::Array<unsigned long, 1> &I_A1_In, const string &S_FileName_In, const string &S_Mode);
template bool utils::WriteArrayToFile(const blitz::Array<int, 1> &I_A1_In, const string &S_FileName_In, const string &S_Mode);
template bool utils::WriteArrayToFile(const blitz::Array<long, 1> &I_A1_In, const string &S_FileName_In, const string &S_Mode);
template bool utils::WriteArrayToFile(const blitz::Array<float, 1> &I_A1_In, const string &S_FileName_In, const string &S_Mode);
template bool utils::WriteArrayToFile(const blitz::Array<double, 1> &I_A1_In, const string &S_FileName_In, const string &S_Mode);

template bool utils::WriteArrayToFile(const blitz::Array<unsigned short, 2> &D_A2_In, const string &S_FileName_In, const string &S_Mode);
template bool utils::WriteArrayToFile(const blitz::Array<unsigned long, 2> &D_A2_In, const string &S_FileName_In, const string &S_Mode);
template bool utils::WriteArrayToFile(const blitz::Array<int, 2> &D_A2_In, const string &S_FileName_In, const string &S_Mode);
template bool utils::WriteArrayToFile(const blitz::Array<long, 2> &D_A2_In, const string &S_FileName_In, const string &S_Mode);
template bool utils::WriteArrayToFile(const blitz::Array<float, 2> &D_A2_In, const string &S_FileName_In, const string &S_Mode);
template bool utils::WriteArrayToFile(const blitz::Array<double, 2> &D_A2_In, const string &S_FileName_In, const string &S_Mode);
  
template blitz::Array<unsigned short, 1> utils::get1DBlitzArray(unsigned short);
template blitz::Array<int, 1> utils::get1DBlitzArray(int);
template blitz::Array<float, 1> utils::get1DBlitzArray(float);
template blitz::Array<double, 1> utils::get1DBlitzArray(double);
template blitz::Array<unsigned short, 2> utils::get2DBlitzArray(unsigned short, unsigned short);
template blitz::Array<int, 2> utils::get2DBlitzArray(int, int);
template blitz::Array<float, 2> utils::get2DBlitzArray(float, float);
template blitz::Array<double, 2> utils::get2DBlitzArray(double, double);
  
template ndarray::Array<size_t, 1, 1> utils::get1DndArray(size_t);
template ndarray::Array<unsigned short, 1, 1> utils::get1DndArray(unsigned short);
template ndarray::Array<int, 1, 1> utils::get1DndArray(int);
template ndarray::Array<float, 1, 1> utils::get1DndArray(float);
template ndarray::Array<double, 1, 1> utils::get1DndArray(double);
template ndarray::Array<size_t, 2, 2> utils::get2DndArray(size_t, size_t);
template ndarray::Array<unsigned short, 2, 2> utils::get2DndArray(unsigned short, unsigned short);
template ndarray::Array<int, 2, 2> utils::get2DndArray(int, int);
template ndarray::Array<float, 2, 2> utils::get2DndArray(float, float);
template ndarray::Array<double, 2, 2> utils::get2DndArray(double, double);


}}}
