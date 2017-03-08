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

  /**************************************************************************/

  bool trimString(string &str, const int mode){
    return trimString(str, ' ', mode);
  }

  /**************************************************************************/

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

  /**************************************************************************/

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

  /**************************************************************************/

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

  /**************************************************************************/

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

  /**************************************************************************/

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

  /**************************************************************************/

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

  /**************************************************************************/

  template<typename T>
  ndarray::Array<T, 2, 1> get2DndArray(T nRows, T nCols){
    ndarray::Array<T, 2, 1> out = ndarray::allocate(nRows, nCols);
    out[ndarray::view()()] = 0;
    return out;
  }

  /**************************************************************************/

  template<typename T>
  ndarray::Array<T, 1, 1> get1DndArray(T size){
    ndarray::Array<T, 1, 1> out = ndarray::allocate(size);
    out[ndarray::view()] = 0;
    return out;
  }

  /**************************************************************************/

  template<typename T>
  std::vector<T> copy(const std::vector<T> &vecIn){
    std::vector<T> vecOut = vecIn;
    return vecOut;
  }

  /**************************************************************************/

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

  /**************************************************************************/

  std::string dotToUnderscore( std::string number, int accuracy ){
    string out(number);
    int dotPos = out.find( "." );
    cout << "dotPos = " << dotPos << endl;
    if ( dotPos != int( std::string::npos ) ){
      out.replace( dotPos, 1, "_" );
      size_t pos = out.find("e");
      if (pos == std::string::npos )
        pos = out.find("E");
      string ex;
      if ( pos != std::string::npos ){
        ex = out.substr(pos);
        cout << "ex = <" << ex << ">" << endl;
      }
      if ( ( dotPos + accuracy + 1 ) < int( out.length() ) ){
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

  /**************************************************************************/

  template<typename T>
  PTR(T) getPointer(T &obj){
    PTR(T) pointer(new T(obj));
    return pointer;
  }

  void testPolyFit(){
    size_t nX = 10;
    ndarray::Array<double, 1, 1> xArr = ndarray::allocate(nX);
    size_t nDeg = 1;
    size_t badPos = 6;
    double badVal = -10.;
    ndarray::Array<double, 1, 1> coeffsIn = ndarray::allocate(nDeg+1);
    for (size_t i = 0; i <= nDeg; ++i)
      coeffsIn[i] = i + 1.;
    for (size_t pos = 0; pos < xArr.getShape()[0]; ++pos){
      xArr[pos] = double(pos);
    }
    ndarray::Array<double, 1, 1> yArr = pfs::drp::stella::math::Poly(xArr, coeffsIn);
    double goodVal = yArr[badPos];
    yArr[badPos] = badVal;

    cout << "Test PolyFit(xArr, yArr, nDeg)" << endl;
    #ifdef __DEBUG_POLYFIT__
      cout << "testPolyFit: Testing PolyFit(xArr=" << xArr << ", yArr=" << yArr << ", nDeg=" << nDeg << ")" << endl;
    #endif
    ndarray::Array<double, 1, 1> coeffs = pfs::drp::stella::math::PolyFit(xArr, yArr, nDeg);
    #ifdef __DEBUG_POLYFIT__
      cout << "testPolyFit: coeffs = " << coeffs << endl;
    #endif
    ndarray::Array<double, 1, 1> yFit = pfs::drp::stella::math::Poly(xArr, coeffs);
    #ifdef __DEBUG_POLYFIT__
      cout << "testPolyFit: yFit = " << yFit << endl;
    #endif
    for (size_t i = 0; i < xArr.getShape()[0]; ++i){
      if (i != badPos){
        if (fabs(yArr[i] - yFit[i]) < 0.01){
          std::string message("error: fabs(yArr[");
          message += to_string(i) + "] - yFit[" + to_string(i) + "]) = " + to_string(fabs(yArr[i]-yFit[i])) + " < 0.01";
          throw std::runtime_error(message);
        }
      }
    }

    cout << "Test PolyFit(xArr, yArr, nDeg, lSig, uSig, nIter)" << endl;
    double lSig = -2.;
    double uSig = 2.;
    size_t nIter = 2;
    coeffs = pfs::drp::stella::math::PolyFit(xArr, yArr, nDeg, lSig, uSig, nIter);
    #ifdef __DEBUG_POLYFIT__
      cout << "testPolyFit:coeffs = " << coeffs << endl;
    #endif
    yFit = pfs::drp::stella::math::Poly(xArr, coeffs);
    #ifdef __DEBUG_POLYFIT__
      cout << "testPolyFit: yFit = " << yFit << endl;
    #endif
    for (size_t i = 0; i < xArr.getShape()[0]; ++i){
      if (i != badPos){
        if (fabs(yArr[i] - yFit[i]) > 0.01){
          std::string message("error: fabs(yArr[");
          message += to_string(i) + "] - yFit[" + to_string(i) + "]) = " + to_string(fabs(yArr[i]-yFit[i])) + " > 0.01";
          throw std::runtime_error(message);
        }
      }
    }
    for (size_t i = 0; i <= nDeg; ++i){
      if (fabs(coeffs[i] - coeffsIn[i]) > 0.0000001){
        std::string message("error: fabs(coeffs[i](=");
        message += to_string(coeffs[i]) + ") - coeffsIn[i](=" + to_string(coeffsIn[i]) + ")) > 0.0000001";
        throw std::runtime_error(message);
      }
    }

    ndarray::Array<double, 1, 1> xRange = ndarray::allocate(2);
    xRange[0] = 0.;
    xRange[1] = xArr[xArr.getShape()[0]-1];

    ndarray::Array<double, 1, 1> xNorm = pfs::drp::stella::math::convertRangeToUnity(xArr, xRange);
    #ifdef __DEBUG_POLYFIT__
      cout << "testPolyFit: xNorm = " << xNorm << endl;
    #endif

    int nArgs = 8;
    std::vector< std::string > keyWords(nArgs);
    keyWords[0] = std::string("REJECTED");
    keyWords[1] = std::string("NOT_REJECTED");
    keyWords[2] = std::string("N_REJECTED");
    PTR(std::vector<size_t>) P_I_A1_Rejected(new std::vector<size_t>());
    PTR(std::vector<size_t>) P_I_A1_NotRejected(new std::vector<size_t>());
    int I_NRejected = 3;
    PTR(int) P_I_NRejected(new int(I_NRejected));
    PTR(ndarray::Array<double, 1, 1>) pXRange(new ndarray::Array<double, 1, 1>(xRange));
    ndarray::Array<double, 1, 1> xRangeBak = ndarray::allocate(xRange.getShape()[0]);
    xRangeBak.deep() = xRange;
    std::vector<void*> args(nArgs);
    args[0] = &P_I_A1_Rejected;
    args[1] = &P_I_A1_NotRejected;
    args[2] = &P_I_NRejected;
    ndarray::Array<double, 1, 1> sigma = ndarray::allocate(nDeg+1);
    PTR(ndarray::Array<double, 1, 1>) pSigma(new ndarray::Array<double, 1, 1>(sigma));
    keyWords[6] = std::string("SIGMA");
    args[6] = &pSigma;
    ndarray::Array<double, 2, 2> covar = ndarray::allocate(ndarray::makeVector(nDeg+1,nDeg+1));
    PTR(ndarray::Array<double, 2, 2>) pCovar(new ndarray::Array<double, 2, 2>(covar));
    keyWords[7] = std::string("COVAR");
    args[7] = &pCovar;
    #ifdef __DEBUG_POLYFIT__
      cout << "=================================================================" << endl;
      cout << "testPolyFit: Testing PolyFit(xNorm=" << xNorm << ", yArr=" << yArr << ", nDeg=" << nDeg << ", lSig=" << lSig << ", uSig=" << uSig << ", nIter=" << nIter << ", keyWords, args)" << endl;
    #endif

    cout << "Test PolyFit without MeasureErrors and without re-scaling the xRange, using the already re-scaled xRange 'xNorm'" << endl;
    coeffs = pfs::drp::stella::math::PolyFit(xNorm, yArr, nDeg, lSig, uSig, nIter, keyWords, args);
    #ifdef __DEBUG_POLYFIT__
      cout << "testPolyFit: xRange = " << xRange << endl;
      cout << "testPolyFit: xRangeBak = " << xRangeBak << endl;
      cout << "testPolyFit: coeffs = " << coeffs << endl;
      cout << "testPolyFit: P_I_A1_Rejected = ";
      for (size_t pos = 0; pos < P_I_A1_Rejected->size(); ++pos)
        cout << (*P_I_A1_Rejected)[pos] << ", ";
      cout << endl;
      cout << "testPolyFit: not rejected = ";
      for (size_t pos = 0; pos < P_I_A1_NotRejected->size(); ++pos)
        cout << (*P_I_A1_NotRejected)[pos] << ", ";
      cout << endl;
      cout << "testPolyFit: n rejected = " << *P_I_NRejected << endl;
    #endif
    if (*P_I_NRejected != 1){
      std::string message("testPolyFit: ERROR: *P_I_NRejected(=");
      message += to_string(*P_I_NRejected) + " != 1";
      throw std::runtime_error(message);
    }
    if (P_I_A1_Rejected->size() != 1){
      std::string message("testPolyFit: ERROR: P_I_A1_Rejected->size()=");
      message += to_string(P_I_A1_Rejected->size()) + " != 1";
      throw std::runtime_error(message);
    }
    if ((*P_I_A1_Rejected)[0] != badPos){
      std::string message("testPolyFit: ERROR: P_I_A1_Rejected[0]=");
      message += to_string((*P_I_A1_Rejected)[0]) + "!= badPos=" + to_string(badPos);
      throw std::runtime_error(message);
    }
    if (P_I_A1_NotRejected->size() != (nX - 1)){
      std::string message("testPolyFit: ERROR: P_I_A1_NotRejected-size(=");
      message += to_string(P_I_A1_NotRejected->size()) + ") != " + to_string(nX-1);
      throw std::runtime_error(message);
    }
    if (pfs::drp::stella::math::find(pfs::drp::stella::math::vectorToNdArray(*P_I_A1_NotRejected, false), (*P_I_A1_Rejected)[0]) >= 0){
      std::string message("testPolyFit: can still find (*P_I_A1_Rejected)[0] in P_I_A1_NotRejected");
      throw std::runtime_error(message);
    }
    if (pSigma->getShape()[0] != (nDeg + 1)){
      std::string message("testPolyFit: ERROR: pSigma->getShape()[0] != nDeg+1(=");
      message += to_string(nDeg+1);
      throw std::runtime_error(message);
    }
    if ((pCovar->getShape()[0] != (nDeg + 1))
        || (pCovar->getShape()[1] != (nDeg + 1))){
      std::string message("testPolyFit: ERROR: pCovar not (nDeg+1)x(nDeg+1)( = ");
      message += to_string(nDeg+1) + "x" + to_string(nDeg+1);
      throw std::runtime_error(message);
    }

    yFit = pfs::drp::stella::math::Poly(xNorm, coeffs, -1., 1.);
    for (size_t i = 0; i < yFit.getShape()[0]; ++i){
      #ifdef __DEBUG_POLYFIT__
        cout << "testPolyFit: yArr[" << i << "] = " << yArr[i] << ", yFit[" << i << "] = " << yFit[i] << endl;
      #endif
      if ((i != badPos) && (fabs(yFit[i] - yArr[i]) > 0.1)){
        std::string message("testPolyFit: ERROR1: fabs(yFit[");
        message += to_string(i) + "] - yArr[" + to_string(i) + "])=" + to_string(fabs(yFit[i]-yArr[i])) + " > 0.1";
        throw std::runtime_error(message);
      }
      if ((i == badPos) && (fabs(yFit[i] - goodVal) > 0.1)){
        std::string message("testPolyFit: ERROR1: fabs(yFit[");
        message += to_string(i) + "] - goodVal=" + to_string(goodVal) + " = " + to_string(fabs(yFit[i]-goodVal)) + " > 0.1";
        throw std::runtime_error(message);
      }
    }

    cout << "Test with Measure Errors (wrong length) and with re-scaling the xRange to [-1,1]" << endl;
    keyWords[3] = std::string("XRANGE");
    args[3] = &pXRange;
    keyWords[4] = std::string("MEASURE_ERRORS");
    ndarray::Array<double, 1, 1> measureErrorsWrongSize = ndarray::allocate(xArr.getShape()[0]-1);
    for (size_t pos = 0; pos < measureErrorsWrongSize.getShape()[0]; ++pos)
      measureErrorsWrongSize[pos] = sqrt(fabs(yArr[pos]));
    PTR(ndarray::Array<double, 1, 1>) pMeasureErrorsWrongSize(new ndarray::Array<double, 1, 1>(measureErrorsWrongSize));
    args[4] = &pMeasureErrorsWrongSize;
    #ifdef __DEBUG_POLYFIT__
      cout << "=================================================================" << endl;
      cout << "testPolyFit: Testing PolyFit(xArr=" << xArr << ", yArr=" << yArr << ", nDeg=" << nDeg << ", lSig=" << lSig << ", uSig=" << uSig << ", nIter=" << nIter << ", keyWords, args)" << endl;
    #endif
    try{
      coeffs = pfs::drp::stella::math::PolyFit(xArr, yArr, nDeg, lSig, uSig, nIter, keyWords, args);
    }
    catch (const std::exception& e) {
      std::string errorMessage = e.what();
      std::string testMessage("pfs::drp::stella::math::CurveFitting::PolyFit: Error:");
      testMessage += " P_D_A1_MeasureErrors->getShape()[0](=" + to_string(nX-1);
      testMessage += ") != D_A1_X_In.getShape()[0](=" + to_string(nX) + ")";
      if (errorMessage.compare(testMessage) != 0)
        throw;
    }
    for (size_t i = 0; i < pXRange->getShape()[0]; ++i){
      if (fabs((*pXRange)[i] - xRangeBak[i]) > 0.0000001){
        string message("error: fabs((*pXRange)[");
        message += to_string(i) +"](=" + to_string((*pXRange)[i]) + ") - xRangeBak[" + to_string(i) + "](=" + to_string(xRangeBak[i]) + ")) = ";
        message += to_string(fabs((*pXRange)[i] - xRangeBak[i])) + " > 0.0000001";
        throw std::runtime_error(message);
      }
    }

    cout << "Test with Measure Errors (correct length) and with re-scaling the xRange to [-1,1]" << endl;
    ndarray::Array<double, 1, 1> measureErrors = ndarray::allocate(xArr.getShape()[0]);
    for (size_t pos=0; pos<xArr.getShape()[0]; ++pos)
      measureErrors[pos] = sqrt(fabs(yArr[pos]));
    PTR(ndarray::Array<double, 1, 1>) pMeasureErrors(new ndarray::Array<double, 1, 1>(measureErrors));
    args[4] = &pMeasureErrors;
    ndarray::Array<double, 1, 1> yFitCheck = ndarray::allocate(xArr.getShape()[0]);
    PTR(ndarray::Array<double, 1, 1>) pYFitCheck(new ndarray::Array<double, 1, 1>(yFitCheck));
    keyWords[5] = std::string("YFIT");
    args[5] = &pYFitCheck;
    #ifdef __DEBUG_POLYFIT__
      cout << "=================================================================" << endl;
      cout << "testPolyFit: Testing PolyFit(xArr=" << xArr << ", yArr=" << yArr << ", nDeg=" << nDeg << ", lSig=" << lSig << ", uSig=" << uSig << ", nIter=" << nIter << ", keyWords, args)" << endl;
    #endif
    coeffs = pfs::drp::stella::math::PolyFit(xArr, yArr, nDeg, lSig, uSig, nIter, keyWords, args);
    for (size_t i = 0; i < pXRange->getShape()[0]; ++i){
      if (fabs((*pXRange)[i] - xRangeBak[i]) > 0.0000001){
        string message("error: 2. fabs((*pXRange)[");
        message += to_string(i) +"](=" + to_string((*pXRange)[i]) + ") - xRangeBak[" + to_string(i) + "](=" + to_string(xRangeBak[i]) + ")) = ";
        message += to_string(fabs((*pXRange)[i] - xRangeBak[i])) + " > 0.0000001";
        throw std::runtime_error(message);
      }
    }
    #ifdef __DEBUG_POLYFIT__
      cout << "testPolyFit: coeffs = " << coeffs << endl;
      cout << "testPolyFit: P_I_A1_Rejected = ";
      for (size_t pos = 0; pos < P_I_A1_Rejected->size(); ++pos)
        cout << (*P_I_A1_Rejected)[pos] << ", ";
      cout << endl;
      cout << "testPolyFit: not rejected = ";
      for (size_t pos = 0; pos < P_I_A1_NotRejected->size(); ++pos)
        cout << (*P_I_A1_NotRejected)[pos] << ", ";
      cout << endl;
      cout << "testPolyFit: n rejected = " << *P_I_NRejected << endl;
    #endif
    if (pMeasureErrors->getShape()[0] != xArr.getShape()[0]){
      std::string message("error: pMeasureErrors->getShape()[0](=");
      message += to_string(pMeasureErrors->getShape()[0]) + ") != xArr.getShape()[0](=" + to_string(xArr.getShape()[0]) + ")";
      throw std::runtime_error(message);
    }
    for (size_t pos = 0; pos < P_I_A1_NotRejected->size(); ++pos){
      if ((*P_I_A1_NotRejected)[pos] > xArr.getShape()[0]){
        std::string message("testPolyFit: ERROR: (*P_I_A1_NotRejected)[pos] = ");
        message += to_string((*P_I_A1_NotRejected)[pos]) + " outside limits";
        throw std::runtime_error(message);
      }
    }
    if (*P_I_NRejected != 1){
      std::string message("testPolyFit: error: *P_I_NRejected=");
      message += to_string(*P_I_NRejected) + " != 1";
      throw std::runtime_error(message);
    }
    yFit = pfs::drp::stella::math::Poly(xArr, coeffs, xRange[0], xRange[1]);
    #ifdef __DEBUG_POLYFIT__
      cout << "testPolyFit: yFit = " << yFit << endl;
      cout << "testPolyFit: yFitCheck = " << yFitCheck << endl;
    #endif
    for (size_t i = 0; i < yFit.getShape()[0]; ++i){
      #ifdef __DEBUG_POLYFIT__
        cout << "testPolyFit: yArr[" << i << "] = " << yArr[i] << ", yFit[" << i << "] = " << yFit[i] << endl;
      #endif
      if ((i != badPos) && (fabs(yArr[i] - yFit[i]) > 0.0001)){
        std::string message("testPolyFit: ERROR: fabs(yArr[");
        message += to_string(i) + "]=" + to_string(yArr[i]) + ") - yFit[" + to_string(i) + "]=" + to_string(yFit[i]);
        message += ") = " + to_string(fabs(yArr[i] - yFit[i])) + " > 0.0001";
        throw std::runtime_error(message);
      }
      if ((i == badPos) && (fabs(yFit[i] - goodVal) > 0.0001)){
        std::string message("testPolyFit: ERROR2: fabs(yFit[");
        message += to_string(i) + "]=" + to_string(yFit[i]) + " - goodVal=" + to_string(goodVal);
        message += " = " + to_string(fabs(yFit[i] - goodVal)) + " > 0.0001";
        throw std::runtime_error(message);
      }
      if (fabs(yFit[i] - yFitCheck[i]) > 0.0001){
        std::string message("testPolyFit: ERROR2: fabs(yFit[");
        message += to_string(i) + "]=" + to_string(yFit[i]) + ") - yFitCheck[" + to_string(i) + "]=" + to_string(yFitCheck[i]);
        message += ") = " + to_string(fabs(yFit[i] - yFitCheck[i])) + " > 0.0001";
        throw std::runtime_error(message);
      }
    }
    if (pSigma->getShape()[0] != nDeg+1){
        std::string message("testPolyFit: ERROR: pSigma->getShape()[0](=");
        message += to_string(pSigma->getShape()[0]) + ") != nDeg+1(=" + to_string(nDeg+1);
        throw std::runtime_error(message);
    }
    if (pCovar->getShape()[0] != nDeg+1){
        std::string message("testPolyFit: ERROR: pCovar->getShape()[0](=");
        message += to_string(pCovar->getShape()[0]) + ") != nDeg+1(=" + to_string(nDeg+1);
        throw std::runtime_error(message);
    }
    if (pCovar->getShape()[1] != nDeg+1){
        std::string message("testPolyFit: ERROR: pCovar->getShape()[0](=");
        message += to_string(pCovar->getShape()[1]) + ") != nDeg+1(=" + to_string(nDeg+1);
        throw std::runtime_error(message);
    }
    return;
  }

  template<typename T>
  ndarray::Array<T, 1, 1> vectorToNdArray(std::vector<T> & vec){
    ndarray::Array<T, 1, 1> array = external(vec.data(),
                                             ndarray::makeVector(int(vec.size())),
                                             ndarray::makeVector(1));
    return array;
  }

  template<typename T>
  ndarray::Array<T const, 1, 1> vectorToNdArray(std::vector<T> const& vec){
    const ndarray::Array<T const, 1, 1> array = external(vec.data(),
                                                         ndarray::makeVector(int(vec.size())),
                                                         ndarray::makeVector(1));
    return array;
  }

  template<typename T, typename U>
  ndarray::Array<U, 1, 1> typeCastNdArray(ndarray::Array<T const, 1, 1> const& arr, U const& newType){
    ndarray::Array<U, 1, 1> out = ndarray::allocate(arr.getShape()[0]);
    auto itOut = out.begin();
    for (auto itIn = arr.begin(); itIn != arr.end(); ++itIn, ++itOut){
      *itOut = U(*itIn);
    }
    return out;
  }

  template std::string numberToString_dotToUnderscore( float, int );
  template std::string numberToString_dotToUnderscore( double, int );

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

  template ndarray::Array<size_t const, 1, 1> vectorToNdArray(std::vector<size_t> const&);
  template ndarray::Array<int const, 1, 1> vectorToNdArray(std::vector<int> const&);
  template ndarray::Array<long const, 1, 1> vectorToNdArray(std::vector<long> const&);
  template ndarray::Array<float const, 1, 1> vectorToNdArray(std::vector<float> const&);
  template ndarray::Array<double const, 1, 1> vectorToNdArray(std::vector<double> const&);

  template ndarray::Array<size_t, 1, 1> vectorToNdArray(std::vector<size_t> &);
  template ndarray::Array<int, 1, 1> vectorToNdArray(std::vector<int> &);
  template ndarray::Array<long, 1, 1> vectorToNdArray(std::vector<long> &);
  template ndarray::Array<float, 1, 1> vectorToNdArray(std::vector<float> &);
  template ndarray::Array<double, 1, 1> vectorToNdArray(std::vector<double> &);

  template ndarray::Array<double, 1, 1> typeCastNdArray(ndarray::Array<float const, 1, 1> const&, double const&);
  template ndarray::Array<float, 1, 1> typeCastNdArray(ndarray::Array<double const, 1, 1> const&, float const&);
  template ndarray::Array<int, 1, 1> typeCastNdArray(ndarray::Array<size_t const, 1, 1> const&, int const&);
  template ndarray::Array<float, 1, 1> typeCastNdArray(ndarray::Array<int const, 1, 1> const&, float const&);
  template ndarray::Array<double, 1, 1> typeCastNdArray(ndarray::Array<int const, 1, 1> const&, double const&);
}
}}}
