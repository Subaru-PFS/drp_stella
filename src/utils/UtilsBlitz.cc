#include "pfs/drp/stella/utils/UtilsBlitz.h"
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
  
  bool sToD(const blitz::Array<string, 1> &S_A1_In, blitz::Array<double, 1> &D_A1_Out){
    int I_NElements = S_A1_In.size();
    D_A1_Out.resize(I_NElements);
    for (int i=0; i < I_NElements; i++){
      if (!sToD(S_A1_In(i), D_A1_Out(i)))
        return false;
    }
    return true;
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

  template bool WriteFits(const blitz::Array<unsigned short, 2>* image_In, const string &fileName_In);
  template bool WriteFits(const blitz::Array<int, 2>* image_In, const string &fileName_In);
  template bool WriteFits(const blitz::Array<long, 2>* image_In, const string &fileName_In);
  template bool WriteFits(const blitz::Array<float, 2>* image_In, const string &fileName_In);
  template bool WriteFits(const blitz::Array<double, 2>* image_In, const string &fileName_In);

  template bool WriteFits(const blitz::Array<unsigned short, 1>* image_In, const string &fileName_In);
  template bool WriteFits(const blitz::Array<int, 1>* image_In, const string &fileName_In);
  template bool WriteFits(const blitz::Array<long, 1>* image_In, const string &fileName_In);
  template bool WriteFits(const blitz::Array<float, 1>* image_In, const string &fileName_In);
  template bool WriteFits(const blitz::Array<double, 1>* image_In, const string &fileName_In);

  template bool WriteArrayToFile(const blitz::Array<unsigned short, 1> &I_A1_In, const string &S_FileName_In, const string &S_Mode);
  template bool WriteArrayToFile(const blitz::Array<unsigned long, 1> &I_A1_In, const string &S_FileName_In, const string &S_Mode);
  template bool WriteArrayToFile(const blitz::Array<int, 1> &I_A1_In, const string &S_FileName_In, const string &S_Mode);
  template bool WriteArrayToFile(const blitz::Array<long, 1> &I_A1_In, const string &S_FileName_In, const string &S_Mode);
  template bool WriteArrayToFile(const blitz::Array<float, 1> &I_A1_In, const string &S_FileName_In, const string &S_Mode);
  template bool WriteArrayToFile(const blitz::Array<double, 1> &I_A1_In, const string &S_FileName_In, const string &S_Mode);

  template bool WriteArrayToFile(const blitz::Array<unsigned short, 2> &D_A2_In, const string &S_FileName_In, const string &S_Mode);
  template bool WriteArrayToFile(const blitz::Array<unsigned long, 2> &D_A2_In, const string &S_FileName_In, const string &S_Mode);
  template bool WriteArrayToFile(const blitz::Array<int, 2> &D_A2_In, const string &S_FileName_In, const string &S_Mode);
  template bool WriteArrayToFile(const blitz::Array<long, 2> &D_A2_In, const string &S_FileName_In, const string &S_Mode);
  template bool WriteArrayToFile(const blitz::Array<float, 2> &D_A2_In, const string &S_FileName_In, const string &S_Mode);
  template bool WriteArrayToFile(const blitz::Array<double, 2> &D_A2_In, const string &S_FileName_In, const string &S_Mode);

  template blitz::Array<unsigned short, 1> get1DBlitzArray(unsigned short);
  template blitz::Array<int, 1> get1DBlitzArray(int);
  template blitz::Array<float, 1> get1DBlitzArray(float);
  template blitz::Array<double, 1> get1DBlitzArray(double);
  template blitz::Array<unsigned short, 2> get2DBlitzArray(unsigned short, unsigned short);
  template blitz::Array<int, 2> get2DBlitzArray(int, int);
  template blitz::Array<float, 2> get2DBlitzArray(float, float);
  template blitz::Array<double, 2> get2DBlitzArray(double, double);
  

}}}}