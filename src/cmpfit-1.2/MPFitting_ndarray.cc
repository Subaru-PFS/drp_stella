#include <cstring>

#include "lsst/pex/exceptions.h"
#include "pfs/drp/stella/cmpfit-1.2/MPFitting_ndarray.h"

namespace pexExcept = lsst::pex::exceptions;

/* This is the private data structure which contains the data points
   and their uncertainties */
struct vars_struct {
  double *x;
  double *y;
  double *ey;
};

namespace {
#ifdef __DEBUG_GAUSSFITLIM__
/* Simple routine to print the fit results */
void PrintResult(double *x, mp_result *result){
  int i;

  if ((x == 0) || (result == 0)) return;
  cout << "MPFitting_ndarray:  CHI-SQUARE = " << result->bestnorm << "    (" << result->nfunc-result->nfree << " DOF)" << endl;
  cout << "MPFitting_ndarray:        NPAR = " << result->npar << endl;
  cout << "MPFitting_ndarray:       NFREE = " << result->nfree << endl;
  cout << "MPFitting_ndarray:     NPEGGED = " << result->npegged << endl;
  cout << "MPFitting_ndarray:     NITER = " << result->niter << endl;
  cout << "MPFitting_ndarray:      NFEV = " << result->nfev << endl << endl;
  for (i=0; i<result->npar; i++) {
    cout << "MPFitting_ndarray:  P[" << i << "] = " << x[i] << " +/- " << result->xerror[i] << endl;
  }
}
#endif

int MPFitGaussFuncCB(int m, int n, double *p, double *dy, double **dvec, void *vars)
{
  int i;
  struct vars_struct *v = (struct vars_struct *) vars;
  double *x, *y, *ey;
  double xc, sig2;

  x = v->x;
  y = v->y;
  ey = v->ey;

  sig2 = p[2]*p[2];

  for (i=0; i<m; i++) {
    xc = x[i]-p[1];
    dy[i] = (y[i] - p[0]*exp(-0.5*xc*xc/sig2) - p[3])/ey[i];
  }

  return 0;
}

int MPFitGaussFuncNB(int m, int n, double *p, double *dy, double **dvec, void *vars)
{
  int i;
  struct vars_struct *v = (struct vars_struct *) vars;
  double *x, *y, *ey;
  double xc, sig2;

  x = v->x;
  y = v->y;
  ey = v->ey;

  sig2 = p[2]*p[2];

  for (i=0; i<m; i++) {
    xc = x[i]-p[1];
    dy[i] = (y[i] - p[0]*exp(-0.5*xc*xc/sig2))/ey[i];
  }

  return 0;
}

int MPFitGaussFuncACB(int m, int n, double *p, double *dy, double **dvec, void *vars){
  int i;
  struct vars_struct *v = (struct vars_struct *) vars;
  double *x, *y, *ey;
  double xc, sig2;

  x = v->x;
  y = v->y;
  ey = v->ey;

  sig2 = p[2]*p[2];

  for (i=0; i<m; i++) {
    xc = x[i]-p[1];
    dy[i] = (y[i] - (p[0]*exp(-0.5*xc*xc/sig2)/(sqrt(2. * D_PI) * p[2])) - p[3])/ey[i];
  }

  return 0;

}

int MPFitGaussFuncANB(int m, int n, double *p, double *dy, double **dvec, void *vars){
  int i;
  struct vars_struct *v = (struct vars_struct *) vars;
  double *x, *y, *ey;
  double xc, sig2;

  x = v->x;
  y = v->y;
  ey = v->ey;

  sig2 = p[2]*p[2];

  for (i=0; i<m; i++) {
    xc = x[i]-p[1];
    dy[i] = (y[i] - (p[0]*exp(-0.5*xc*xc/sig2)/(sqrt(2. * D_PI) * p[2])))/ey[i];
  }

  return 0;
}

int MPFitGaussFuncLB(int m, int n, double *p, double *dy, double **dvec, void *vars){
  int i;
  struct vars_struct *v = (struct vars_struct *) vars;
  double *x, *y, *ey;
  double xc, sig2;

  x = v->x;
  y = v->y;
  ey = v->ey;

  sig2 = p[2]*p[2];

  for (i=0; i<m; i++) {
    xc = x[i]-p[1];
    dy[i] = (y[i] - p[0]*exp(-0.5*xc*xc/sig2) - p[3] - p[4]*(x[i]-x[0]))/ey[i];
  }

  return 0;
}

int MPFitGaussFuncALB(int m, int n, double *p, double *dy, double **dvec, void *vars){
  int i;
  struct vars_struct *v = (struct vars_struct *) vars;
  double *x, *y, *ey;
  double xc, sig2;

  x = v->x;
  y = v->y;
  ey = v->ey;

  sig2 = p[2]*p[2];

  for (i=0; i<m; i++) {
    xc = x[i]-p[1];
    dy[i] = (y[i] - (p[0]*exp(-0.5*xc*xc/sig2)/(sqrt(2. * D_PI) * p[2])) - p[3] - p[4]*(x[i]-x[0]))/ey[i];
  }

  return 0;
}
}

template<typename T>
bool MPFitGauss(const ndarray::Array< T, 1, 1 >& D_A1_X_In,
                const ndarray::Array< T, 1, 1 >& D_A1_Y_In,
                const ndarray::Array< T, 1, 1 >& D_A1_EY_In,
                const ndarray::Array< T, 1, 1 >& D_A1_Guess_In,
                const bool B_WithConstantBackground,
                const bool B_FitArea,
                ndarray::Array< T, 1, 1 >& D_A1_Coeffs_Out,
                ndarray::Array< T, 1, 1 >& D_A1_ECoeffs_Out){
  int I_NParams = 4;
  if (!B_WithConstantBackground)
    I_NParams = 3;
  unsigned int I_NPts = D_A1_X_In.getShape()[0];
  if (D_A1_Y_In.getShape()[0] != I_NPts){
    cout << "MPFitting_ndarray::MPFitGauss: ERROR: D_A1_X_In and D_A1_Y_In must have same size" << endl;
    return false;
  }
  if (D_A1_EY_In.getShape()[0] != I_NPts){
    cout << "MPFitting_ndarray::MPFitGauss: ERROR: D_A1_X_In and D_A1_EY_In must have same size" << endl;
    return false;
  }
  if (D_A1_Guess_In.getShape()[0] != I_NParams){
    cout << "MPFitting_ndarray::MPFitGauss: ERROR: D_A1_Guess_In must have " << I_NParams << " elements" << endl;
    return false;
  }
  double x[I_NPts];
  double y[I_NPts];
  double ey[I_NPts];
  double p[I_NParams];
  for (int i_pt=0; i_pt<I_NPts; i_pt++){
    x[i_pt] = D_A1_X_In[i_pt];
    y[i_pt] = D_A1_Y_In[i_pt];
    ey[i_pt] = D_A1_EY_In[i_pt];
  }
  for (int i_par=0; i_par<I_NParams; i_par++){
    p[i_par] = D_A1_Guess_In[i_par];       /* Initial conditions */
  }
  double perror[I_NParams];			   /* Returned parameter errors */
  mp_par pars[I_NParams];			   /* Parameter constraints */
  if (D_A1_Coeffs_Out.getShape()[0] != I_NParams){
    string message("MPFitting_ndarray::MPFitGauss: ERROR: D_A1_Coeffs_Out.getShape()[0]=");
    message += to_string(D_A1_Coeffs_Out.getShape()[0]) + " != I_NParams=" + to_string(I_NParams);
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  if (D_A1_ECoeffs_Out.getShape()[0] != I_NParams){
    string message("MPFitting_ndarray::MPFitGauss: ERROR: D_A1_ECoeffs_Out.getShape()[0]=");
    message += to_string(D_A1_ECoeffs_Out.getShape()[0]) + " != I_NParams=" + to_string(I_NParams);
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  struct vars_struct v;
  mp_result result;

  memset(&result,0,sizeof(result));      /* Zero results structure */
  result.xerror = perror;

  memset(pars,0,sizeof(pars));        /* Initialize constraint structure */
  /* No constraints */

  v.x = x;
  v.y = y;
  v.ey = ey;

  /* Call fitting function for 10 data points and 4 parameters (no
     parameters fixed) */
  if (B_WithConstantBackground){
    if (B_FitArea)
      result.status = mpfit(MPFitGaussFuncACB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
    else
      result.status = mpfit(MPFitGaussFuncCB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
  }
  else{
    if (B_FitArea)
      result.status = mpfit(MPFitGaussFuncANB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
    else
      result.status = mpfit(MPFitGaussFuncNB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
  }

  for (int i_par=0; i_par<I_NParams; i_par++){
    D_A1_Coeffs_Out[i_par] = p[i_par];
    D_A1_ECoeffs_Out[i_par] = result.xerror[i_par];
  }

  return true;
}

template<typename T>
bool MPFitGaussLim(const ndarray::Array< T, 1, 1 >& D_A1_X_In,
                   const ndarray::Array< T, 1, 1 >& D_A1_Y_In,
                   const ndarray::Array< T, 1, 1 >& D_A1_EY_In,
                   const ndarray::Array< T, 1, 1 >& D_A1_Guess_In,
                   const ndarray::Array< int, 2, 1 > &I_A2_Limited,
                   const ndarray::Array< T, 2, 1 > &D_A2_Limits,
                   const int Background,
                   const bool B_FitArea,
                   ndarray::Array< T, 1, 1 >& D_A1_Coeffs_Out,
                   ndarray::Array< T, 1, 1 >& D_A1_ECoeffs_Out,
                   bool Debug){

  int I_NParams = 3;
  if (Background == 1)
    I_NParams = 4;
  else if (Background == 2)
    I_NParams = 5;

  ///Check Limits
  for (int i_par=0; i_par<I_NParams; i_par++){
    if ((I_A2_Limited(i_par, 0) == 1) && (I_A2_Limited(i_par, 1) == 1)){
      if (D_A2_Limits(i_par, 0) > D_A2_Limits(i_par, 1)){
        cout << "MPFitting_ndarray::MPFitGaussLim: ERROR: D_A2_Limits(" << i_par << ", 0)(=" << D_A2_Limits(i_par, 0) <<") > D_A2_Limits(" << i_par << ", 1)(=" << D_A2_Limits(i_par, 1) << ")" << endl;
        return false;
      }
    }
  }
  unsigned int I_NPts = D_A1_X_In.getShape()[0];
  if (D_A1_Y_In.getShape()[0] != I_NPts){
    cout << "MPFitting_ndarray::MPFitGaussLim: ERROR: D_A1_X_In and D_A1_Y_In must have same size" << endl;
    return false;
  }
  if (D_A1_EY_In.getShape()[0] != I_NPts){
    cout << "MPFitting_ndarray::MPFitGaussLim: ERROR: D_A1_X_In and D_A1_EY_In must have same size" << endl;
    return false;
  }
  if (D_A1_Guess_In.getShape()[0] != I_NParams){
    cout << "MPFitting_ndarray::MPFitGaussLim: ERROR: D_A1_Guess_In must have " << I_NParams << " elements" << endl;
    return false;
  }
  double x[I_NPts];
  double y[I_NPts];
  double ey[I_NPts];
  double p[I_NParams];
  for (int i_pt=0; i_pt<I_NPts; i_pt++){
    x[i_pt] = double(D_A1_X_In[i_pt]);
    y[i_pt] = double(D_A1_Y_In[i_pt]);
    ey[i_pt] = double(D_A1_EY_In[i_pt]);
    #ifdef __DEBUG_GAUSSFITLIM__
      if (Debug)
        cout << "MPFitting_ndarray::MPFitGaussLim: x[" << i_pt << "] = " << x[i_pt] << ", y[" << i_pt << "] = " << y[i_pt] << ", ey[" << i_pt << "] = " << ey[i_pt] << endl;
    #endif
  }
  for (int i_par=0; i_par<I_NParams; i_par++){
    p[i_par] = double(D_A1_Guess_In[i_par]);       /* Initial conditions */
    #ifdef __DEBUG_GAUSSFITLIM__
      if (Debug)
        cout << "MPFitting_ndarray::MPFitGaussLim: p[" << i_par << "] = " << p[i_par] << endl;
    #endif
  }
  double perror[I_NParams];                        /* Returned parameter errors */
  mp_par pars[I_NParams];                          /* Parameter constraints */
  if (D_A1_Coeffs_Out.getShape()[0] != I_NParams){
    string message("MPFitting_ndarray::MPFitGaussLim: ERROR: D_A1_Coeffs_Out.getShape()[0]=");
    message += to_string(D_A1_Coeffs_Out.getShape()[0]) + " != I_NParams=" + to_string(I_NParams);
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  if (D_A1_ECoeffs_Out.getShape()[0] != I_NParams){
    string message("MPFitting_ndarray::MPFitGaussLim: ERROR: D_A1_ECoeffs_Out.getShape()[0]=");
    message += to_string(D_A1_ECoeffs_Out.getShape()[0]) + " != I_NParams=" + to_string(I_NParams);
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  struct vars_struct v;
  mp_result result;

  memset(&result,0,sizeof(result));      /* Zero results structure */
  result.xerror = perror;

  memset(pars,0,sizeof(pars));        /* Initialize constraint structure */
  for (int i_par=0; i_par<I_NParams; i_par++){
    pars[i_par].limited[0] = I_A2_Limited[ ndarray::makeVector( i_par, 0 ) ];
    pars[i_par].limited[1] = I_A2_Limited[ ndarray::makeVector( i_par, 1 ) ];
    pars[i_par].limits[0] = double(D_A2_Limits[ ndarray::makeVector( i_par, 0 ) ] );              /* lower parameter limit */
    pars[i_par].limits[1] = double(D_A2_Limits[ ndarray::makeVector(i_par, 1 ) ] );              /* upper parameter limit */
    #ifdef __DEBUG_GAUSSFITLIM__
      if (Debug){
        cout << "MPFitting_ndarray::MPFitGaussLim: pars[" << i_par << "].limited[0] = " << pars[i_par].limited[0] << ", pars[" << i_par << "].limited[1] = " << pars[i_par].limited[1] << endl;
        cout << "MPFitting_ndarray::MPFitGaussLim: pars[" << i_par << "].limits[0] = " << pars[i_par].limits[0] << ", pars[" << i_par << "].limits[1] = " << pars[i_par].limits[1] << endl;
      }
    #endif
  }

  v.x = x;
  v.y = y;
  v.ey = ey;

  if (Background == 0){
    #ifdef __DEBUG_GAUSSFITLIM__
      if (Debug)
        cout << "MPFitting_ndarray::MPFitGaussLim: Background == 0: starting mpfit" << endl;
    #endif
    if (B_FitArea){
      result.status = mpfit(MPFitGaussFuncANB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
    }
    else{
      result.status = mpfit(MPFitGaussFuncNB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
    }
  }
  else if (Background == 1){
    #ifdef __DEBUG_GAUSSFITLIM__
      if (Debug)
        cout << "MPFitting_ndarray::MPFitGaussLim: Background == 1: starting mpfit" << endl;
    #endif
    if (B_FitArea)
      result.status = mpfit(MPFitGaussFuncACB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
    else
      result.status = mpfit(MPFitGaussFuncCB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
  }
  else{/// Background == 2
    #ifdef __DEBUG_GAUSSFITLIM__
      if (Debug)
        cout << "MPFitting_ndarray::MPFitGaussLim: Background == 2: starting mpfit" << endl;
    #endif
    if (B_FitArea)
      result.status = mpfit(MPFitGaussFuncALB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
    else
      result.status = mpfit(MPFitGaussFuncLB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
  }
  #ifdef __DEBUG_GAUSSFITLIM__
    if (Debug)
      cout << "MPFitting_ndarray::MPFitGaussLim: mpfit status = " << result.status << endl;
  #endif

  for (int i_par=0; i_par<I_NParams; i_par++){
    D_A1_Coeffs_Out[i_par] = p[i_par];
    D_A1_ECoeffs_Out[i_par] = result.xerror[i_par];
  }

  #ifdef __DEBUG_GAUSSFITLIM__
    if (Debug)
      PrintResult(p, &result);
  #endif
  if (result.status < 1)
    return false;
  return true;
}

template bool MPFitGauss(const ndarray::Array<float, 1, 1> &D_A1_X_In,
                         const ndarray::Array<float, 1, 1> &D_A1_Y_In,
                         const ndarray::Array<float, 1, 1> &D_A1_EY_In,
                         const ndarray::Array<float, 1, 1> &D_A1_Guess_In,
                         const bool B_WithConstantBackground,
                         const bool B_FitArea,
                         ndarray::Array<float, 1, 1> &D_A1_Coeffs_Out,
                         ndarray::Array<float, 1, 1>& D_A1_ECoeffs_Out);
template bool MPFitGaussLim(const ndarray::Array<float, 1, 1> &D_A1_X_In,
                            const ndarray::Array<float, 1, 1> &D_A1_Y_In,
                            const ndarray::Array<float, 1, 1> &D_A1_EY_In,
                            const ndarray::Array<float, 1, 1> &D_A1_Guess_In,
                            const ndarray::Array<int, 2, 1> &I_A2_Limited,
                            const ndarray::Array<float, 2, 1> &D_A2_Limits,
                            const int Background,
                            const bool B_FitArea,
                            ndarray::Array<float, 1, 1> &D_A1_Coeffs_Out,
                            ndarray::Array<float, 1, 1>& D_A1_ECoeffs_Out,
                            bool);
