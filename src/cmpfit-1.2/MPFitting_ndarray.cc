/*
 * MINPACK-1 Least Squares Fitting Library
 *
 * Test routines
 *
 * These test routines provide examples for users to familiarize
 * themselves with the mpfit library.  They also provide a baseline
 * test data set for users to be sure that the library is functioning
 * properly on their platform.
 *
 * By default, testmpfit is built by the distribution Makefile.
 *
 * To test the function of the mpfit library,
 *   1. Build testmpfit   ("make testmpfit")
 *   2. Run testmpfit     ("./mpfit")
 *   3. Compare results of your run with the distributed file testmpfit.log
 *
 * This file contains several test user functions:
 *   1. linfunc() linear fit function, y = f(x) = a - b*x
 *      - Driver is testlinfit()
 *   2. quadfunc() quadratic polynomial function, y = f(x) = a + b*x + c*x^2
 *      - Driver is testquadfit() - all parameters free
 *      - Driver is testquadfix() - linear parameter fixed
 *   3. gaussfunc() gaussian peak
 *      - Driver is testgaussfit() - all parameters free
 *      - Driver is testgaussfix() - constant & centroid fixed
 *           (this routine demonstrates in comments how to impose parameter limits)
 *   4. main() routine calls all five driver functions
 *
 * Copyright (C) 2003,2006,2009,2010, Craig Markwardt
 *
 */

/* Test routines for mpfit library
   $Id: testmpfit.c,v 1.6 2010/11/13 08:18:02 craigm Exp $
*/

#include "pfs/drp/stella/cmpfit-1.2/MPFitting_ndarray.h"

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

/*
 * linear fit function
 *
 * m - number of data points
 * n - number of parameters (2)
 * p - array of fit parameters
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 *
int linfunc(int m, int n, double *p, double *dy, double **dvec, void *vars)
{
  int i;
  struct vars_struct *v = (struct vars_struct *) vars;
  double *x, *y, *ey, f;

  x = v->x;
  y = v->y;
  ey = v->ey;

  for (i=0; i<m; i++) {
    f = p[0] - p[1]*x[i];     /* Linear fit function; note f = a - b*x */
/*    dy[i] = (y[i] - f)/ey[i];
  }

  return 0;
}

/* Test harness routine, which contains test data, invokes mpfit() *
int testlinfit()
{
  double x[] = {-1.7237128E+00,1.8712276E+00,-9.6608055E-01,
		-2.8394297E-01,1.3416969E+00,1.3757038E+00,
		-1.3703436E+00,4.2581975E-02,-1.4970151E-01,
		8.2065094E-01};
  double y[] = {1.9000429E-01,6.5807428E+00,1.4582725E+00,
		2.7270851E+00,5.5969253E+00,5.6249280E+00,
		0.787615,3.2599759E+00,2.9771762E+00,
		4.5936475E+00};
  double ey[10];
  /*      y = a - b*x    */
  /*              a    b */
//  double p[2] = {1.0, 1.0};           /* Parameter initial conditions */
//  double pactual[2] = {3.20, 1.78};   /* Actual values used to make data */
//  double perror[2];                   /* Returned parameter errors */
/*  int i;
  struct vars_struct v;
  int status;
  mp_result result;

  memset(&result,0,sizeof(result));       /* Zero results structure */
/*  result.xerror = perror;
  for (i=0; i<10; i++) ey[i] = 0.07;   /* Data errors */
/*
  v.x = x;
  v.y = y;
  v.ey = ey;

  /* Call fitting function for 10 data points and 2 parameters */
/*  status = mpfit(linfunc, 10, 2, p, 0, 0, (void *) &v, &result);

  printf("*** testlinfit status = %d\n", status);
  printresult(p, pactual, &result);

  return 0;
}

/*
 * quadratic fit function
 *
 * m - number of data points
 * n - number of parameters (2)
 * p - array of fit parameters
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 *
int quadfunc(int m, int n, double *p, double *dy, double **dvec, void *vars)
{
  int i;
  struct vars_struct *v = (struct vars_struct *) vars;
  double *x, *y, *ey;

  x = v->x;
  y = v->y;
  ey = v->ey;

  /* printf ("quadfunc %f %f %f\n", p[0], p[1], p[2]); */
/*
  for (i=0; i<m; i++) {
    dy[i] = (y[i] - p[0] - p[1]*x[i] - p[2]*x[i]*x[i])/ey[i];
  }

  return 0;
}

/* Test harness routine, which contains test quadratic data, invokes
   mpfit() *
int testquadfit()
{
  double x[] = {-1.7237128E+00,1.8712276E+00,-9.6608055E-01,
		-2.8394297E-01,1.3416969E+00,1.3757038E+00,
		-1.3703436E+00,4.2581975E-02,-1.4970151E-01,
		8.2065094E-01};
  double y[] = {2.3095947E+01,2.6449392E+01,1.0204468E+01,
		5.40507,1.5787588E+01,1.6520903E+01,
		1.5971818E+01,4.7668524E+00,4.9337711E+00,
		8.7348375E+00};
  double ey[10];
  double p[] = {1.0, 1.0, 1.0};        /* Initial conditions */
//  double pactual[] = {4.7, 0.0, 6.2};  /* Actual values used to make data */
//  double perror[3];		       /* Returned parameter errors */
/*  int i;
  struct vars_struct v;
  int status;
  mp_result result;

  memset(&result,0,sizeof(result));          /* Zero results structure */
/*  result.xerror = perror;
  for (i=0; i<10; i++) ey[i] = 0.2;       /* Data errors */
/*
  v.x = x;
  v.y = y;
  v.ey = ey;

  /* Call fitting function for 10 data points and 3 parameters */
/*  status = mpfit(quadfunc, 10, 3, p, 0, 0, (void *) &v, &result);

  printf("*** testquadfit status = %d\n", status);
  printresult(p, pactual, &result);

  return 0;
}

/* Test harness routine, which contains test quadratic data;

   Example of how to fix a parameter
*
int testquadfix()
{
  double x[] = {-1.7237128E+00,1.8712276E+00,-9.6608055E-01,
		-2.8394297E-01,1.3416969E+00,1.3757038E+00,
		-1.3703436E+00,4.2581975E-02,-1.4970151E-01,
		8.2065094E-01};
  double y[] = {2.3095947E+01,2.6449392E+01,1.0204468E+01,
		5.40507,1.5787588E+01,1.6520903E+01,
		1.5971818E+01,4.7668524E+00,4.9337711E+00,
		8.7348375E+00};

  double ey[10];
  double p[] = {1.0, 0.0, 1.0};        /* Initial conditions */
//  double pactual[] = {4.7, 0.0, 6.2};  /* Actual values used to make data */
//  double perror[3];		       /* Returned parameter errors */
//  mp_par pars[3];                      /* Parameter constraints */
/*  int i;
  struct vars_struct v;
  int status;
  mp_result result;

  memset(&result,0,sizeof(result));       /* Zero results structure */
/*  result.xerror = perror;

  memset(pars, 0, sizeof(pars));       /* Initialize constraint structure */
/*  pars[1].fixed = 1;                   /* Fix parameter 1 */
/*
  for (i=0; i<10; i++) ey[i] = 0.2;

  v.x = x;
  v.y = y;
  v.ey = ey;

  /* Call fitting function for 10 data points and 3 parameters (1
     parameter fixed) */
/*  status = mpfit(quadfunc, 10, 3, p, pars, 0, (void *) &v, &result);

  printf("*** testquadfix status = %d\n", status);
  printresult(p, pactual, &result);

  return 0;
}

/*
 * gaussian fit function
 *
 * m - number of data points
 * n - number of parameters (4)
 * p - array of fit parameters
 *     p[0] = peak y value
 *     p[1] = x centroid position
 *     p[2] = gaussian sigma width
 *     p[3] = constant offset
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 */
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

/*
 * gaussian fit function
 *
 * m - number of data points
 * n - number of parameters (3)
 * p - array of fit parameters
 *     p[0] = peak y value
 *     p[1] = x centroid position
 *     p[2] = gaussian sigma width
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 */
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

/*
 * gaussian fit function
 *
 * m - number of data points
 * n - number of parameters (4)
 * p - array of fit parameters
 *     p[0] = area under curve value
 *     p[1] = x centroid position
 *     p[2] = gaussian sigma width
 *     p[3] = constant offset
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 */
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
/*
 * gaussian fit function
 *
 * m - number of data points
 * n - number of parameters (3)
 * p - array of fit parameters
 *     p[0] = area under curve value
 *     p[1] = x centroid position
 *     p[2] = gaussian sigma width
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 */
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

/*
 * gaussian fit function
 *
 * m - number of data points
 * n - number of parameters (5)
 * p - array of fit parameters
 *     p[0] = peak y value
 *     p[1] = x centroid position
 *     p[2] = gaussian sigma width
 *     p[3] = constant offset
 *     p[4] = linear offset
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 */
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

/*
 * gaussian fit function
 *
 * m - number of data points
 * n - number of parameters (5)
 * p - array of fit parameters
 *     p[0] = area under curve value
 *     p[1] = x centroid position
 *     p[2] = gaussian sigma width
 *     p[3] = constant offset
 *     p[4] = linear offset
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 */
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

/*
 * gaussian fit function
 *
 * m - number of data points
 * n - number of parameters (6)
 * p - array of fit parameters
 *     p[0] = peak y value 1st Gauss
 *     p[1] = x centroid position 1st Gauss
 *     p[2] = gaussian sigma width
 *     p[3] = peak y value 2nd Gauss
 *     p[4] = x centroid position 2nd Gauss
 *     p[5] = constant offset
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 */
int MPFitTwoGaussFuncCB(int m, int n, double *p, double *dy, double **dvec, void *vars){
  int i;
  struct vars_struct *v = (struct vars_struct *) vars;
  double *x, *y, *ey;
  double xc_a, xc_b, sig2;//, sig2_b;

  x = v->x;
  y = v->y;
  ey = v->ey;

  sig2 = p[2]*p[2];
//  sig2_b = p[6]*p[6];

  for (i=0; i<m; i++) {
    xc_a = x[i]-p[1];
    xc_b = x[i]-p[4];
    dy[i] = (y[i] - p[0]*exp(-0.5*xc_a*xc_a/sig2) - p[3]*exp(-0.5*xc_b*xc_b/sig2) - p[5])/ey[i];
  }

  return 0;
}

/*
 * gaussian fit function
 *
 * m - number of data points
 * n - number of parameters (5)
 * p - array of fit parameters
 *     p[0] = peak y value 1st Gauss
 *     p[1] = x centroid position 1st Gauss
 *     p[2] = gaussian sigma width
 *     p[3] = peak y value 2nd Gauss
 *     p[4] = x centroid position 2nd Gauss
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 */
int MPFitTwoGaussFuncNB(int m, int n, double *p, double *dy, double **dvec, void *vars){
  int i;
  struct vars_struct *v = (struct vars_struct *) vars;
  double *x, *y, *ey;
  double xc_a, xc_b, sig2;//_a, sig2_b;

  x = v->x;
  y = v->y;
  ey = v->ey;

  sig2 = p[2]*p[2];
//  sig2_b = p[5]*p[5];

  for (i=0; i<m; i++) {
    xc_a = x[i]-p[1];
    xc_b = x[i]-p[4];
    dy[i] = (y[i] - p[0]*exp(-0.5*xc_a*xc_a/sig2) - p[3]*exp(-0.5*xc_b*xc_b/sig2))/ey[i];
  }

  return 0;
}

/*
 * gaussian fit function
 *
 * m - number of data points
 * n - number of parameters (6)
 * p - array of fit parameters
 *     p[0] = area under curve value 1st Gauss
 *     p[1] = x centroid position 1st Gauss
 *     p[2] = gaussian sigma width
 *     p[3] = area under curve value 2nd Gauss
 *     p[4] = x centroid position 2nd Gauss
 *     p[5] = constant offset
* dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 */
int MPFitTwoGaussFuncACB(int m, int n, double *p, double *dy, double **dvec, void *vars){
  int i;
  struct vars_struct *v = (struct vars_struct *) vars;
  double *x, *y, *ey;
  double xc_a, xc_b, sig2;//_a, sig2_b;

  x = v->x;
  y = v->y;
  ey = v->ey;

  sig2 = p[2]*p[2];
//  sig2_b = p[6]*p[6];

  for (i=0; i<m; i++) {
    xc_a = x[i]-p[1];
    xc_b = x[i]-p[4];
    dy[i] = (y[i] - (p[0]*exp(-0.5*xc_a*xc_a/sig2)/(sqrt(2. * D_PI) * p[2])) - (p[3]*exp(-0.5*xc_b*xc_b/sig2)/(sqrt(2. * D_PI) * p[2])) - p[5])/ey[i];
  }

  return 0;
}

/*
 * gaussian fit function
 *
 * m - number of data points
 * n - number of parameters (5)
 * p - array of fit parameters
 *     p[0] = area under curve value 1st Gauss
 *     p[1] = x centroid position 1st Gauss
 *     p[2] = gaussian sigma width
 *     p[3] = area under curve value 2nd Gauss
 *     p[4] = x centroid position 2nd Gauss
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 */
int MPFitTwoGaussFuncANB(int m, int n, double *p, double *dy, double **dvec, void *vars){
  int i;
  struct vars_struct *v = (struct vars_struct *) vars;
  double *x, *y, *ey;
  double xc_a, xc_b, sig2;//_a, sig2_b;

  x = v->x;
  y = v->y;
  ey = v->ey;

  sig2 = p[2]*p[2];
//  sig2_b = p[5]*p[5];

  for (i=0; i<m; i++) {
    xc_a = x[i]-p[1];
    xc_b = x[i]-p[4];
    dy[i] = (y[i] - (p[0]*exp(-0.5*xc_a*xc_a/sig2)/(sqrt(2. * D_PI) * p[2])) - (p[3]*exp(-0.5*xc_b*xc_b/sig2)/(sqrt(2. * D_PI) * p[2])))/ey[i];
  }

  return 0;
}


/*
 * gaussian fit function
 *
 * m - number of data points
 * n - number of parameters (8)
 * p - array of fit parameters
 *     p[0] = peak y value 1st Gauss
 *     p[1] = x centroid position 1st Gauss
 *     p[2] = gaussian sigma width
 *     p[3] = peak y value 2nd Gauss
 *     p[4] = x centroid position 2nd Gauss
 *     p[5] = peak y value 3rd Gauss
 *     p[6] = x centroid position 3rd Gauss
 *     p[7] = constant offset
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 */
int MPFitThreeGaussFuncCB(int m, int n, double *p, double *dy, double **dvec, void *vars){
  int i;
  struct vars_struct *v = (struct vars_struct *) vars;
  double *x, *y, *ey;
  double xc_a, xc_b, xc_c, sig2;//, sig2_b;

  x = v->x;
  y = v->y;
  ey = v->ey;

  sig2 = p[2]*p[2];
  //  sig2_b = p[6]*p[6];

  for (i=0; i<m; i++) {
    xc_a = x[i]-p[1];
    xc_b = x[i]-p[4];
    xc_c = x[i]-p[6];
    dy[i] = (y[i] - p[0]*exp(-0.5*xc_a*xc_a/sig2) - p[3]*exp(-0.5*xc_b*xc_b/sig2) - p[5]*exp(-0.5*xc_c*xc_c/sig2) - p[7])/ey[i];
  }

  return 0;
}

/*
 * gaussian fit function
 *
 * m - number of data points
 * n - number of parameters (7)
 * p - array of fit parameters
 *     p[0] = peak y value 1st Gauss
 *     p[1] = x centroid position 1st Gauss
 *     p[2] = gaussian sigma width 1st Gauss
 *     p[3] = peak y value 2nd Gauss
 *     p[4] = x centroid position 2nd Gauss
 *     p[5] = peak y value 3rd Gauss
 *     p[6] = x centroid position 3rd Gauss
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 */
int MPFitThreeGaussFuncNB(int m, int n, double *p, double *dy, double **dvec, void *vars){
  int i;
  struct vars_struct *v = (struct vars_struct *) vars;
  double *x, *y, *ey;
  double xc_a, xc_b, xc_c, sig2;//_a, sig2_b;

  x = v->x;
  y = v->y;
  ey = v->ey;

  sig2 = p[2]*p[2];
  //  sig2_b = p[5]*p[5];

  for (i=0; i<m; i++) {
    xc_a = x[i]-p[1];
    xc_b = x[i]-p[4];
    xc_c = x[i]-p[6];
    dy[i] = (y[i] - p[0]*exp(-0.5*xc_a*xc_a/sig2) - p[3]*exp(-0.5*xc_b*xc_b/sig2) - p[5]*exp(-0.5*xc_c*xc_c/sig2))/ey[i];
  }

  return 0;
}

/*
 * gaussian fit function
 *
 * m - number of data points
 * n - number of parameters (8)
 * p - array of fit parameters
 *     p[0] = area under curve value 1st Gauss
 *     p[1] = x centroid position 1st Gauss
 *     p[2] = gaussian sigma width 1st Gauss
 *     p[3] = area under curve value 2nd Gauss
 *     p[4] = x centroid position 2nd Gauss
 *     p[5] = area under curve value 3rd Gauss
 *     p[6] = x centroid position 3rd Gauss
 *     p[7] = constant offset
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 */
int MPFitThreeGaussFuncACB(int m, int n, double *p, double *dy, double **dvec, void *vars){
  int i;
  struct vars_struct *v = (struct vars_struct *) vars;
  double *x, *y, *ey;
  double xc_a, xc_b, xc_c, sig2;//_a, sig2_b;

  x = v->x;
  y = v->y;
  ey = v->ey;

  sig2 = p[2]*p[2];
  //  sig2_b = p[6]*p[6];

  for (i=0; i<m; i++) {
    xc_a = x[i]-p[1];
    xc_b = x[i]-p[4];
    xc_c = x[i]-p[6];
    dy[i] = (y[i] - (p[0]*exp(-0.5*xc_a*xc_a/sig2)/(sqrt(2. * D_PI) * p[2])) - (p[3]*exp(-0.5*xc_b*xc_b/sig2)/(sqrt(2. * D_PI) * p[2])) - (p[5]*exp(-0.5*xc_c*xc_c/sig2)/(sqrt(2. * D_PI) * p[2])) - p[7])/ey[i];
  }

  return 0;
}

/*
 * gaussian fit function
 *
 * m - number of data points
 * n - number of parameters (7)
 * p - array of fit parameters
 *     p[0] = area under curve value 1st Gauss
 *     p[1] = x centroid position 1st Gauss
 *     p[2] = gaussian sigma width 1st Gauss
 *     p[3] = area under curve value 2nd Gauss
 *     p[4] = x centroid position 2nd Gauss
 *     p[5] = area under curve value 3rd Gauss
 *     p[6] = x centroid position 2nd Gauss
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 */
int MPFitThreeGaussFuncANB(int m, int n, double *p, double *dy, double **dvec, void *vars){
  int i;
  struct vars_struct *v = (struct vars_struct *) vars;
  double *x, *y, *ey;
  double xc_a, xc_b, xc_c, sig2;//_a, sig2_b;

  x = v->x;
  y = v->y;
  ey = v->ey;

  sig2 = p[2]*p[2];
  //  sig2_b = p[5]*p[5];

  for (i=0; i<m; i++) {
    xc_a = x[i]-p[1];
    xc_b = x[i]-p[4];
    xc_c = x[i]-p[6];
    dy[i] = (y[i] - (p[0]*exp(-0.5*xc_a*xc_a/sig2)/(sqrt(2. * D_PI) * p[2])) - (p[3]*exp(-0.5*xc_b*xc_b/sig2)/(sqrt(2. * D_PI) * p[2])) - (p[5]*exp(-0.5*xc_c*xc_c/sig2)/(sqrt(2. * D_PI) * p[2])))/ey[i];
  }

  return 0;
}

int Chebyshev1stKind(int m, int n, double *p, double *dy, double **dvec, void *vars){
  int i, j;
  struct vars_struct *v = (struct vars_struct *) vars;
  double *x, *y, *ey;
  double yCalc;
  ndarray::Array<double, 1, 1> Tn = ndarray::allocate(n);

  x = v->x;
  y = v->y;
  ey = v->ey;

  for (i=0; i<m; i++) {
    Tn[0] = 1;
    if (n > 1)
      Tn[1] = x[i];
    for (j = 2; j < n; ++j)
      Tn[j] = 2. * x[i] * Tn[j-1] - Tn[j-2];
    yCalc = p[0];
    for (j = 1; j < n; ++j)
      yCalc += p[j] * Tn[j];
    dy[i] = (y[i] - yCalc)/ey[i];
  }

  return 0;
}

template<typename T> 
bool MPFitChebyshev1stKind(const ndarray::Array<T, 1, 1> & D_A1_X_In,
                           const ndarray::Array<T, 1, 1> & D_A1_Y_In,
                           const ndarray::Array<T, 1, 1> & D_A1_EY_In,
                           ndarray::Array<T, 1, 1> & D_A1_Coeffs_Out,
                           ndarray::Array<T, 1, 1> & D_A1_ECoeffs_Out){
  int I_NParams = D_A1_Coeffs_Out.getShape()[0];
  ndarray::Array<T, 1, 1> D_A1_Guess = ndarray::allocate(I_NParams);
  ndarray::Array<T, 1, 1> moments = pfs::drp::stella::math::moment(D_A1_Y_In, 1);
  D_A1_Guess[0] = moments[0];
  for (int i_par=1; i_par<I_NParams; i_par++){
    D_A1_Guess[i_par] = 1.0*pow(10, -7.);//*double(i_par));//D_A1_Guess_In[i_par];       /* Initial conditions */
  }
  return MPFitChebyshev1stKind(D_A1_X_In,
                               D_A1_Y_In,
                               D_A1_EY_In,
                               D_A1_Guess,
                               D_A1_Coeffs_Out,
                               D_A1_ECoeffs_Out);
}

template<typename T> 
bool MPFitChebyshev1stKind(const ndarray::Array<T, 1, 1> & D_A1_X_In,
                           const ndarray::Array<T, 1, 1> & D_A1_Y_In,
                           const ndarray::Array<T, 1, 1> & D_A1_EY_In,
                           const ndarray::Array<T, 1, 1> & D_A1_Guess_In,
                           ndarray::Array<T, 1, 1> & D_A1_Coeffs_Out,
                           ndarray::Array<T, 1, 1> & D_A1_ECoeffs_Out){
  int I_NParams = D_A1_Coeffs_Out.getShape()[0];
  if (I_NParams != D_A1_ECoeffs_Out.getShape()[0]){
    cout << "MPFitting_ndarray::MPFitChebyshev1stKind: ERROR: D_A1_Coeffs_Out and D_A1_ECoeffs_Out must have same size" << endl;
    return false;
  }
  unsigned int I_NPts = D_A1_X_In.getShape()[0];
  if (D_A1_Y_In.getShape()[0] != I_NPts){
    cout << "MPFitting_ndarray::MPFitChebyshev1stKind: ERROR: D_A1_X_In and D_A1_Y_In must have same size" << endl;
    return false;
  }
  if (D_A1_EY_In.getShape()[0] != I_NPts){
    cout << "MPFitting_ndarray::MPFitChebyshev1stKind: ERROR: D_A1_X_In and D_A1_EY_In must have same size" << endl;
    return false;
  }
  if (D_A1_Guess_In.getShape()[0] != I_NParams){
    cout << "MPFitting_ndarray::MPFitChebyshev1stKind: ERROR: D_A1_Guess_In and D_A1_Coeffs_Out must have same size" << endl;
    return false;
  }
  
  double x[I_NPts];
  double y[I_NPts];
  double ey[I_NPts];
  double p[I_NParams];
  double xMin, xMax;
  xMin = pfs::drp::stella::math::min(D_A1_X_In);
  xMax = pfs::drp::stella::math::max(D_A1_X_In);
  ndarray::Array<double, 1, 1> xRange = ndarray::allocate(2);
  xRange[0] = xMin;
  xRange[1] = xMax;
  ndarray::Array<T, 1, 1> xNew = pfs::drp::stella::math::convertRangeToUnity(D_A1_X_In, xRange);
  
  for (int i_pt=0; i_pt<I_NPts; i_pt++){
    x[i_pt] = double(xNew[i_pt]);
    y[i_pt] = D_A1_Y_In[i_pt];
    ey[i_pt] = D_A1_EY_In[i_pt];
  }
  for (int i_par=0; i_par<I_NParams; i_par++){
    p[i_par] = D_A1_Guess_In[i_par];//*double(i_par));//D_A1_Guess_In[i_par];       /* Initial conditions */
  }
  double perror[I_NParams];			   /* Returned parameter errors */
  mp_par pars[I_NParams];			   /* Parameter constraints */
  struct vars_struct v;
  int status;
  mp_result result;

  memset(&result,0,sizeof(result));      /* Zero results structure */
  result.xerror = perror;

  memset(pars,0,sizeof(pars));        /* Initialize constraint structure */

  v.x = x;
  v.y = y;
  v.ey = ey;

  status = mpfit(Chebyshev1stKind, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);

  for (int i_par=0; i_par<I_NParams; i_par++){
    D_A1_Coeffs_Out[i_par] = p[i_par];
    D_A1_ECoeffs_Out[i_par] = result.xerror[i_par];
  }

  return true;
}

/* Test harness routine, which contains test gaussian-peak data */
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
//  double pactual[] = {0.0, 4.70, 0.0, 0.5};/* Actual values used to make data*/
  double perror[I_NParams];			   /* Returned parameter errors */
  mp_par pars[I_NParams];			   /* Parameter constraints */
  if (D_A1_Coeffs_Out.getShape()[0] != I_NParams){
    string message("MPFitting_ndarray::MPFitGauss: ERROR: D_A1_Coeffs_Out.getShape()[0]=");
    message += to_string(D_A1_Coeffs_Out.getShape()[0]) + " != I_NParams=" + to_string(I_NParams);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  if (D_A1_ECoeffs_Out.getShape()[0] != I_NParams){
    string message("MPFitting_ndarray::MPFitGauss: ERROR: D_A1_ECoeffs_Out.getShape()[0]=");
    message += to_string(D_A1_ECoeffs_Out.getShape()[0]) + " != I_NParams=" + to_string(I_NParams);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
//  int i;
  struct vars_struct v;
  int status;
  mp_result result;

  memset(&result,0,sizeof(result));      /* Zero results structure */
  result.xerror = perror;

  memset(pars,0,sizeof(pars));        /* Initialize constraint structure */
  /* No constraints */

//  for (i=0; i<10; i++) ey[i] = 0.5;

  v.x = x;
  v.y = y;
  v.ey = ey;

  /* Call fitting function for 10 data points and 4 parameters (no
     parameters fixed) */
  if (B_WithConstantBackground){
    if (B_FitArea)
      status = mpfit(MPFitGaussFuncACB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
    else
      status = mpfit(MPFitGaussFuncCB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
  }
  else{
    if (B_FitArea)
      status = mpfit(MPFitGaussFuncANB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
    else
      status = mpfit(MPFitGaussFuncNB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
  }

//  cout << "MPFitting_ndarray::MPFitGauss: *** testgaussfit status = " << status << endl;
//  PrintResult(p, &result);

  for (int i_par=0; i_par<I_NParams; i_par++){
    D_A1_Coeffs_Out[i_par] = p[i_par];
    D_A1_ECoeffs_Out[i_par] = result.xerror[i_par];
  }

  return true;
}


/* Test harness routine, which contains test gaussian-peak data

   Example of fixing two parameter

   Commented example of how to put boundary constraints
*/
template<typename T>
bool MPFitGaussFix(const ndarray::Array< T, 1, 1 >& D_A1_X_In,
                   const ndarray::Array< T, 1, 1 >& D_A1_Y_In,
                   const ndarray::Array< T, 1, 1 >& D_A1_EY_In,
                   const ndarray::Array< T, 1, 1 >& D_A1_Guess_In,
                   const ndarray::Array<int, 1, 1>& I_A1_Fix,
                   const bool B_WithConstantBackground,
                   const bool B_FitArea,
                   ndarray::Array<T, 1, 1> & D_A1_Coeffs_Out,
                   ndarray::Array<T, 1, 1> & D_A1_ECoeffs_Out){
  int I_NParams = 4;
  if (!B_WithConstantBackground)
    I_NParams = 3;
  unsigned int I_NPts = D_A1_X_In.getShape()[0];
  if (D_A1_Y_In.getShape()[0] != I_NPts){
    cout << "MPFitting_ndarray::MPFitGaussFix: ERROR: D_A1_X_In and D_A1_Y_In must have same size" << endl;
    return false;
  }
  if (D_A1_EY_In.getShape()[0] != I_NPts){
    cout << "MPFitting_ndarray::MPFitGaussFix: ERROR: D_A1_X_In and D_A1_EY_In must have same size" << endl;
    return false;
  }
  if (D_A1_Guess_In.getShape()[0] != I_NParams){
    cout << "MPFitting_ndarray::MPFitGaussFix: ERROR: D_A1_Guess_In must have " << I_NParams << " elements" << endl;
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
  //  double pactual[] = {0.0, 4.70, 0.0, 0.5};/* Actual values used to make data*/
  double perror[I_NParams];                        /* Returned parameter errors */
  mp_par pars[I_NParams];                          /* Parameter constraints */
  if (D_A1_Coeffs_Out.getShape()[0] != I_NParams){
    string message("MPFitting_ndarray::MPFitGaussFix: ERROR: D_A1_Coeffs_Out.getShape()[0]=");
    message += to_string(D_A1_Coeffs_Out.getShape()[0]) + " != I_NParams=" + to_string(I_NParams);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  if (D_A1_ECoeffs_Out.getShape()[0] != I_NParams){
    string message("MPFitting_ndarray::MPFitGaussFix: ERROR: D_A1_ECoeffs_Out.getShape()[0]=");
    message += to_string(D_A1_ECoeffs_Out.getShape()[0]) + " != I_NParams=" + to_string(I_NParams);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  struct vars_struct v;
  int status;
  mp_result result;

  memset(&result,0,sizeof(result));      /* Zero results structure */
  result.xerror = perror;

  memset(pars,0,sizeof(pars));        /* Initialize constraint structure */
  for (int i_par=0; i_par<I_NParams; i_par++){
    pars[i_par].fixed = I_A1_Fix[i_par];              /* Fix parameters 0 and 2 */
  }

  /* How to put limits on a parameter.  In this case, parameter 3 is
     limited to be between -0.3 and +0.2.
  pars[3].limited[0] = 0;
  pars[3].limited[1] = 1;
  pars[3].limits[0] = -0.3;
  pars[3].limits[1] = +0.2;
  */

  v.x = x;
  v.y = y;
  v.ey = ey;

  /* Call fitting function for 10 data points and 4 parameters (2
     parameters fixed) */
  if (B_WithConstantBackground){
    if (B_FitArea)
      status = mpfit(MPFitGaussFuncACB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
    else
      status = mpfit(MPFitGaussFuncCB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
  }
  else{
    if (B_FitArea)
      status = mpfit(MPFitGaussFuncANB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
    else
      status = mpfit(MPFitGaussFuncNB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
  }

//  cout << "MPFitting_ndarray:MPFitGaussFix: *** testgaussfix status = " << status << endl;
//  PrintResult(p, &result);

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
                   const int Background,///0-none, 1-constant, 2-linear
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
  //  double pactual[] = {0.0, 4.70, 0.0, 0.5};/* Actual values used to make data*/
  double perror[I_NParams];                        /* Returned parameter errors */
  mp_par pars[I_NParams];                          /* Parameter constraints */
  if (D_A1_Coeffs_Out.getShape()[0] != I_NParams){
    string message("MPFitting_ndarray::MPFitGaussLim: ERROR: D_A1_Coeffs_Out.getShape()[0]=");
    message += to_string(D_A1_Coeffs_Out.getShape()[0]) + " != I_NParams=" + to_string(I_NParams);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  if (D_A1_ECoeffs_Out.getShape()[0] != I_NParams){
    string message("MPFitting_ndarray::MPFitGaussLim: ERROR: D_A1_ECoeffs_Out.getShape()[0]=");
    message += to_string(D_A1_ECoeffs_Out.getShape()[0]) + " != I_NParams=" + to_string(I_NParams);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  struct vars_struct v;
  int status;
  mp_result result;

  memset(&result,0,sizeof(result));      /* Zero results structure */
  result.xerror = perror;

  memset(pars,0,sizeof(pars));        /* Initialize constraint structure */
  for (int i_par=0; i_par<I_NParams; i_par++){
    pars[i_par].limited[0] = I_A2_Limited[i_par][0];
    pars[i_par].limited[1] = I_A2_Limited[i_par][1];
    pars[i_par].limits[0] = double(D_A2_Limits[i_par][0]);              /* lower parameter limit */
    pars[i_par].limits[1] = double(D_A2_Limits[i_par][1]);              /* upper parameter limit */
    #ifdef __DEBUG_GAUSSFITLIM__
      if (Debug){
        cout << "MPFitting_ndarray::MPFitGaussLim: pars[" << i_par << "].limited[0] = " << pars[i_par].limited[0] << ", pars[" << i_par << "].limited[1] = " << pars[i_par].limited[1] << endl;
        cout << "MPFitting_ndarray::MPFitGaussLim: pars[" << i_par << "].limits[0] = " << pars[i_par].limits[0] << ", pars[" << i_par << "].limits[1] = " << pars[i_par].limits[1] << endl;
      }
    #endif
  }

  /* How to put limits on a parameter.  In this case, parameter 3 is
     limited to be between -0.3 and +0.2.
  pars[3].limited[0] = 0;
  pars[3].limited[1] = 1;
  pars[3].limits[0] = -0.3;
  pars[3].limits[1] = +0.2;
  */

  v.x = x;
  v.y = y;
  v.ey = ey;

  /* Call fitting function for 10 data points and 4 parameters (2
     parameters fixed) */
  if (Background == 0){
    #ifdef __DEBUG_GAUSSFITLIM__
      if (Debug)
        cout << "MPFitting_ndarray::MPFitGaussLim: Background == 0: starting mpfit" << endl;
    #endif
    if (B_FitArea)
      status = mpfit(MPFitGaussFuncANB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
    else
      status = mpfit(MPFitGaussFuncNB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
  }
  else if (Background == 1){
    #ifdef __DEBUG_GAUSSFITLIM__
      if (Debug)
        cout << "MPFitting_ndarray::MPFitGaussLim: Background == 1: starting mpfit" << endl;
    #endif
    if (B_FitArea)
      status = mpfit(MPFitGaussFuncACB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
    else
      status = mpfit(MPFitGaussFuncCB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
  }
  else{/// Background == 2
    #ifdef __DEBUG_GAUSSFITLIM__
      if (Debug)
        cout << "MPFitting_ndarray::MPFitGaussLim: Background == 2: starting mpfit" << endl;
    #endif
    if (B_FitArea)
      status = mpfit(MPFitGaussFuncALB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
    else
      status = mpfit(MPFitGaussFuncLB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
  }
  #ifdef __DEBUG_GAUSSFITLIM__
    if (Debug)
      cout << "MPFitting_ndarray::MPFitGaussLim: mpfit status = " << status << endl;
  #endif

  for (int i_par=0; i_par<I_NParams; i_par++){
    D_A1_Coeffs_Out[i_par] = p[i_par];
    D_A1_ECoeffs_Out[i_par] = result.xerror[i_par];
  }

//  cout << "MPFitting_ndarray:MPFitGaussLim: *** testgaussfix status = " << status << endl;
  #ifdef __DEBUG_GAUSSFITLIM__
    if (Debug)
      PrintResult(p, &result);
  #endif
  if (status < 1)
    return false;
  return true;
}

/* Test harness routine, which contains test gaussian-peak data */
template<typename T>
bool MPFitTwoGauss(const ndarray::Array< T, 1, 1 > &D_A1_X_In,
                   const ndarray::Array< T, 1, 1 > &D_A1_Y_In,
                   const ndarray::Array< T, 1, 1 > &D_A1_EY_In,
                   const ndarray::Array< T, 1, 1 > &D_A1_Guess_In,
                   const bool B_WithConstantBackground,
                   const bool B_FitArea,
                   ndarray::Array< T, 1, 1 > &D_A1_Coeffs_Out,
                   ndarray::Array< T, 1, 1 >& D_A1_ECoeffs_Out){
  int I_NParams = 6;
  if (!B_WithConstantBackground)
    I_NParams = 5;
  unsigned int I_NPts = D_A1_X_In.getShape()[0];
  if (D_A1_Y_In.getShape()[0] != I_NPts){
    cout << "MPFitting_ndarray::MPFitTwoGauss: ERROR: D_A1_X_In and D_A1_Y_In must have same size" << endl;
    return false;
  }
  if (D_A1_EY_In.getShape()[0] != I_NPts){
    cout << "MPFitting_ndarray::MPFitTwoGauss: ERROR: D_A1_X_In and D_A1_EY_In must have same size" << endl;
    return false;
  }
  if (D_A1_Guess_In.getShape()[0] != I_NParams){
    cout << "MPFitting_ndarray::MPFitTwoGauss: ERROR: D_A1_Guess_In must have " << I_NParams << " elements" << endl;
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
//  double pactual[] = {0.0, 4.70, 0.0, 0.5};/* Actual values used to make data*/
  double perror[I_NParams];                        /* Returned parameter errors */
  mp_par pars[I_NParams];                          /* Parameter constraints */
  if (D_A1_Coeffs_Out.getShape()[0] != I_NParams){
    string message("MPFitting_ndarray::MPFitTwoGauss: ERROR: D_A1_Coeffs_Out.getShape()[0]=");
    message += to_string(D_A1_Coeffs_Out.getShape()[0]) + " != I_NParams=" + to_string(I_NParams);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  if (D_A1_ECoeffs_Out.getShape()[0] != I_NParams){
    string message("MPFitting_ndarray::MPFitTwoGauss: ERROR: D_A1_ECoeffs_Out.getShape()[0]=");
    message += to_string(D_A1_ECoeffs_Out.getShape()[0]) + " != I_NParams=" + to_string(I_NParams);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  struct vars_struct v;
  int status;
  mp_result result;

  memset(&result,0,sizeof(result));      /* Zero results structure */
  result.xerror = perror;

  memset(pars,0,sizeof(pars));        /* Initialize constraint structure */
  /* No constraints */

//  for (i=0; i<10; i++) ey[i] = 0.5;

  v.x = x;
  v.y = y;
  v.ey = ey;

  /* Call fitting function for 10 data points and 4 parameters (no
     parameters fixed) */
  if (B_WithConstantBackground){
    if (B_FitArea)
      status = mpfit(MPFitTwoGaussFuncACB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
    else
      status = mpfit(MPFitTwoGaussFuncCB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
  }
  else{
    if (B_FitArea)
      status = mpfit(MPFitTwoGaussFuncANB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
    else
      status = mpfit(MPFitTwoGaussFuncNB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
  }

//  cout << "MPFitting_ndarray::MPFitTwoGauss: *** testgaussfit status = " << status << endl;
//  PrintResult(p, &result);

  for (int i_par=0; i_par<I_NParams; i_par++){
    D_A1_Coeffs_Out[i_par] = p[i_par];
    D_A1_ECoeffs_Out[i_par] = result.xerror[i_par];
  }

  return true;
}


/* Test harness routine, which contains test gaussian-peak data
 *
 *   Example of fixing two parameter
 *
 *   Commented example of how to put boundary constraints
 */
template<typename T>
bool MPFitTwoGaussFix(const ndarray::Array< T, 1, 1 > &D_A1_X_In,
                      const ndarray::Array< T, 1, 1 > &D_A1_Y_In,
                      const ndarray::Array< T, 1, 1 > &D_A1_EY_In,
                      const ndarray::Array< T, 1, 1 > &D_A1_Guess_In,
                      const ndarray::Array< int, 1, 1 > &I_A1_Fix,
                      const bool B_WithConstantBackground,
                      const bool B_FitArea,
                      ndarray::Array< T, 1, 1 > &D_A1_Coeffs_Out,
                      ndarray::Array< T, 1, 1 >& D_A1_ECoeffs_Out){
  int I_NParams = 6;
  if (!B_WithConstantBackground)
    I_NParams = 5;
  unsigned int I_NPts = D_A1_X_In.getShape()[0];
  if (D_A1_Y_In.getShape()[0] != I_NPts){
    cout << "MPFitting_ndarray::MPFitTwoGaussFix: ERROR: D_A1_X_In and D_A1_Y_In must have same size" << endl;
    return false;
  }
  if (D_A1_EY_In.getShape()[0] != I_NPts){
    cout << "MPFitting_ndarray::MPFitTwoGaussFix: ERROR: D_A1_X_In and D_A1_EY_In must have same size" << endl;
    return false;
  }
  if (D_A1_Guess_In.getShape()[0] != I_NParams){
    cout << "MPFitting_ndarray::MPFitTwoGaussFix: ERROR: D_A1_Guess_In must have " << I_NParams << " elements" << endl;
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
  //  double pactual[] = {0.0, 4.70, 0.0, 0.5};/* Actual values used to make data*/
  double perror[I_NParams];                        /* Returned parameter errors */
  mp_par pars[I_NParams];                          /* Parameter constraints */
  if (D_A1_Coeffs_Out.getShape()[0] != I_NParams){
    string message("MPFitting_ndarray::MPFitTwoGaussFix: ERROR: D_A1_Coeffs_Out.getShape()[0]=");
    message += to_string(D_A1_Coeffs_Out.getShape()[0]) + " != I_NParams=" + to_string(I_NParams);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  if (D_A1_ECoeffs_Out.getShape()[0] != I_NParams){
    string message("MPFitting_ndarray::MPFitTwoGaussFix: ERROR: D_A1_ECoeffs_Out.getShape()[0]=");
    message += to_string(D_A1_ECoeffs_Out.getShape()[0]) + " != I_NParams=" + to_string(I_NParams);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  struct vars_struct v;
  int status;
  mp_result result;

  memset(&result,0,sizeof(result));      /* Zero results structure */
  result.xerror = perror;

  memset(pars,0,sizeof(pars));        /* Initialize constraint structure */
  for (int i_par=0; i_par<I_NParams; i_par++){
    pars[i_par].fixed = I_A1_Fix[i_par];              /* Fix parameters 0 and 2 */
  }

  /* How to put limits on a parameter.  In this case, parameter 3 is
     limited to be between -0.3 and +0.2.
  pars[3].limited[0] = 0;
  pars[3].limited[1] = 1;
  pars[3].limits[0] = -0.3;
  pars[3].limits[1] = +0.2;
  */

  v.x = x;
  v.y = y;
  v.ey = ey;

  /* Call fitting function for 10 data points and 4 parameters (2
     parameters fixed) */
  if (B_WithConstantBackground){
    if (B_FitArea)
      status = mpfit(MPFitTwoGaussFuncACB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
    else
      status = mpfit(MPFitTwoGaussFuncCB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
  }
  else{
    if (B_FitArea)
      status = mpfit(MPFitTwoGaussFuncANB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
    else
      status = mpfit(MPFitTwoGaussFuncNB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
  }

//  cout << "MPFitting_ndarray:MPFitTwoGaussFix: *** testgaussfix status = " << status << endl;
//  PrintResult(p, &result);

  for (int i_par=0; i_par<I_NParams; i_par++){
    D_A1_Coeffs_Out[i_par] = p[i_par];
    D_A1_ECoeffs_Out[i_par] = result.xerror[i_par];
  }

  return true;
}

template<typename T>
bool MPFitTwoGaussLim(const ndarray::Array< T, 1, 1 > &D_A1_X_In,
                      const ndarray::Array< T, 1, 1 > &D_A1_Y_In,
                      const ndarray::Array< T, 1, 1 > &D_A1_EY_In,
                      const ndarray::Array< T, 1, 1 > &D_A1_Guess_In,
                      const ndarray::Array< int, 2, 1 > &I_A2_Limited,
                      const ndarray::Array< T, 2, 1 > &D_A2_Limits,
                      const bool B_WithConstantBackground,
                      const bool B_FitArea,
                      ndarray::Array< T, 1, 1 > &D_A1_Coeffs_Out,
                      ndarray::Array< T, 1, 1 >& D_A1_ECoeffs_Out){
  int I_NParams = 6;
  if (!B_WithConstantBackground)
    I_NParams = 5;
  unsigned int I_NPts = D_A1_X_In.getShape()[0];
  if (D_A1_Y_In.getShape()[0] != I_NPts){
    cout << "MPFitting_ndarray::MPFitTwoGaussLim: ERROR: D_A1_X_In and D_A1_Y_In must have same size" << endl;
    return false;
  }
  if (D_A1_EY_In.getShape()[0] != I_NPts){
    cout << "MPFitting_ndarray::MPFitTwoGaussLim: ERROR: D_A1_X_In and D_A1_EY_In must have same size" << endl;
    return false;
  }
  if (D_A1_Guess_In.getShape()[0] != I_NParams){
    cout << "MPFitting_ndarray::MPFitTwoGaussLim: ERROR: D_A1_Guess_In must have " << I_NParams << " elements" << endl;
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
  //  double pactual[] = {0.0, 4.70, 0.0, 0.5};/* Actual values used to make data*/
  double perror[I_NParams];                        /* Returned parameter errors */
  mp_par pars[I_NParams];                          /* Parameter constraints */
  if (D_A1_Coeffs_Out.getShape()[0] != I_NParams){
    string message("MPFitting_ndarray::MPFitTwoGaussLim: ERROR: D_A1_Coeffs_Out.getShape()[0]=");
    message += to_string(D_A1_Coeffs_Out.getShape()[0]) + " != I_NParams=" + to_string(I_NParams);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  if (D_A1_ECoeffs_Out.getShape()[0] != I_NParams){
    string message("MPFitting_ndarray::MPFitTwoGaussLim: ERROR: D_A1_ECoeffs_Out.getShape()[0]=");
    message += to_string(D_A1_ECoeffs_Out.getShape()[0]) + " != I_NParams=" + to_string(I_NParams);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  struct vars_struct v;
  int status;
  mp_result result;

  memset(&result,0,sizeof(result));      /* Zero results structure */
  result.xerror = perror;

  memset(pars,0,sizeof(pars));        /* Initialize constraint structure */
  for (int i_par=0; i_par<I_NParams; i_par++){
    pars[i_par].limited[0] = I_A2_Limited[i_par][0];
    pars[i_par].limited[1] = I_A2_Limited[i_par][1];
    pars[i_par].limits[0] = D_A2_Limits[i_par][0];              /* lower parameter limit */
    pars[i_par].limits[1] = D_A2_Limits[i_par][1];              /* upper parameter limit */
  }

  /* How to put limits on a parameter.  In this case, parameter 3 is
     limited to be between -0.3 and +0.2.
  pars[3].limited[0] = 0;
  pars[3].limited[1] = 1;
  pars[3].limits[0] = -0.3;
  pars[3].limits[1] = +0.2;
  */

  v.x = x;
  v.y = y;
  v.ey = ey;

  /* Call fitting function for 10 data points and 4 parameters (2
     parameters fixed) */
  if (B_WithConstantBackground){
    if (B_FitArea)
      status = mpfit(MPFitTwoGaussFuncACB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
    else
      status = mpfit(MPFitTwoGaussFuncCB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
  }
  else{
    if (B_FitArea)
      status = mpfit(MPFitTwoGaussFuncANB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
    else
      status = mpfit(MPFitTwoGaussFuncNB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
  }

  for (int i_par=0; i_par<I_NParams; i_par++){
    D_A1_Coeffs_Out[i_par] = p[i_par];
    D_A1_ECoeffs_Out[i_par] = result.xerror[i_par];
  }

//  cout << "MPFitting_ndarray:MPFitTwoGaussLim: *** testgaussfix status = " << status << endl;
//  PrintResult(p, &result);

  return true;
}

template<typename T>
bool MPFitThreeGauss(const ndarray::Array< T, 1, 1 > &D_A1_X_In,
                     const ndarray::Array< T, 1, 1 > &D_A1_Y_In,
                     const ndarray::Array< T, 1, 1 > &D_A1_EY_In,
                     const ndarray::Array< T, 1, 1 > &D_A1_Guess_In,
                     const bool B_WithConstantBackground,
                     const bool B_FitArea,
                     ndarray::Array< T, 1, 1 > &D_A1_Coeffs_Out,
                     ndarray::Array< T, 1, 1 >& D_A1_ECoeffs_Out){
  int I_NParams = 8;
  if (!B_WithConstantBackground)
    I_NParams = 7;
  unsigned int I_NPts = D_A1_X_In.getShape()[0];
  if (D_A1_Y_In.getShape()[0] != I_NPts){
    cout << "MPFitting_ndarray::MPFitThreeGauss: ERROR: D_A1_X_In and D_A1_Y_In must have same size" << endl;
    return false;
  }
  if (D_A1_EY_In.getShape()[0] != I_NPts){
    cout << "MPFitting_ndarray::MPFitThreeGauss: ERROR: D_A1_X_In and D_A1_EY_In must have same size" << endl;
    return false;
  }
  if (D_A1_Guess_In.getShape()[0] != I_NParams){
    cout << "MPFitting_ndarray::MPFitThreeGauss: ERROR: D_A1_Guess_In must have " << I_NParams << " elements" << endl;
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
//  double pactual[] = {0.0, 4.70, 0.0, 0.5};/* Actual values used to make data*/
  double perror[I_NParams];                        /* Returned parameter errors */
  mp_par pars[I_NParams];                          /* Parameter constraints */
  if (D_A1_Coeffs_Out.getShape()[0] != I_NParams){
    string message("MPFitting_ndarray::MPFitThreeGauss: ERROR: D_A1_Coeffs_Out.getShape()[0]=");
    message += to_string(D_A1_Coeffs_Out.getShape()[0]) + " != I_NParams=" + to_string(I_NParams);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  if (D_A1_ECoeffs_Out.getShape()[0] != I_NParams){
    string message("MPFitting_ndarray::MPFitThreeGauss: ERROR: D_A1_ECoeffs_Out.getShape()[0]=");
    message += to_string(D_A1_ECoeffs_Out.getShape()[0]) + " != I_NParams=" + to_string(I_NParams);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  struct vars_struct v;
  int status;
  mp_result result;

  memset(&result,0,sizeof(result));      /* Zero results structure */
  result.xerror = perror;

  memset(pars,0,sizeof(pars));        /* Initialize constraint structure */
  /* No constraints */

//  for (i=0; i<10; i++) ey[i] = 0.5;

  v.x = x;
  v.y = y;
  v.ey = ey;

  /* Call fitting function for 10 data points and 4 parameters (no
     parameters fixed) */
  if (B_WithConstantBackground){
    if (B_FitArea)
      status = mpfit(MPFitThreeGaussFuncACB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
    else
      status = mpfit(MPFitThreeGaussFuncCB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
  }
  else{
    if (B_FitArea)
      status = mpfit(MPFitThreeGaussFuncANB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
    else
      status = mpfit(MPFitThreeGaussFuncNB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
  }

//  cout << "MPFitting_ndarray::MPFitThreeGauss: *** testgaussfit status = " << status << endl;
//  PrintResult(p, &result);

  for (int i_par=0; i_par<I_NParams; i_par++){
    D_A1_Coeffs_Out[i_par] = p[i_par];
    D_A1_ECoeffs_Out[i_par] = result.xerror[i_par];
  }

  return true;
}

template<typename T>
bool MPFitThreeGaussFix(const ndarray::Array< T, 1, 1 > &D_A1_X_In,
                        const ndarray::Array< T, 1, 1 > &D_A1_Y_In,
                        const ndarray::Array< T, 1, 1 > &D_A1_EY_In,
                        const ndarray::Array< T, 1, 1 > &D_A1_Guess_In,
                        const ndarray::Array< int, 1, 1 > &I_A1_Fix,
                        const bool B_WithConstantBackground,
                        const bool B_FitArea,
                        ndarray::Array< T, 1, 1 > &D_A1_Coeffs_Out,
                        ndarray::Array< T, 1, 1 >& D_A1_ECoeffs_Out){
  int I_NParams = 8;
  if (!B_WithConstantBackground)
    I_NParams = 7;
  unsigned int I_NPts = D_A1_X_In.getShape()[0];
  if (D_A1_Y_In.getShape()[0] != I_NPts){
    cout << "MPFitting_ndarray::MPFitThreeGaussFix: ERROR: D_A1_X_In and D_A1_Y_In must have same size" << endl;
    return false;
  }
  if (D_A1_EY_In.getShape()[0] != I_NPts){
    cout << "MPFitting_ndarray::MPFitThreeGaussFix: ERROR: D_A1_X_In and D_A1_EY_In must have same size" << endl;
    return false;
  }
  if (D_A1_Guess_In.getShape()[0] != I_NParams){
    cout << "MPFitting_ndarray::MPFitThreeGaussFix: ERROR: D_A1_Guess_In must have " << I_NParams << " elements" << endl;
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
  //  double pactual[] = {0.0, 4.70, 0.0, 0.5};/* Actual values used to make data*/
  double perror[I_NParams];                        /* Returned parameter errors */
  mp_par pars[I_NParams];                          /* Parameter constraints */
  if (D_A1_Coeffs_Out.getShape()[0] != I_NParams){
    string message("MPFitting_ndarray::MPFitThreeGaussFix: ERROR: D_A1_Coeffs_Out.getShape()[0]=");
    message += to_string(D_A1_Coeffs_Out.getShape()[0]) + " != I_NParams=" + to_string(I_NParams);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  if (D_A1_ECoeffs_Out.getShape()[0] != I_NParams){
    string message("MPFitting_ndarray::MPFitThreeGaussFix: ERROR: D_A1_ECoeffs_Out.getShape()[0]=");
    message += to_string(D_A1_ECoeffs_Out.getShape()[0]) + " != I_NParams=" + to_string(I_NParams);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  struct vars_struct v;
  int status;
  mp_result result;

  memset(&result,0,sizeof(result));      /* Zero results structure */
  result.xerror = perror;

  memset(pars,0,sizeof(pars));        /* Initialize constraint structure */
  for (int i_par=0; i_par<I_NParams; i_par++){
    pars[i_par].fixed = I_A1_Fix[i_par];              /* Fix parameters 0 and 2 */
  }

  /* How to put limits on a parameter.  In this case, parameter 3 is
     limited to be between -0.3 and +0.2.
  pars[3].limited[0] = 0;
  pars[3].limited[1] = 1;
  pars[3].limits[0] = -0.3;
  pars[3].limits[1] = +0.2;
  */

  v.x = x;
  v.y = y;
  v.ey = ey;

  /* Call fitting function for 10 data points and 4 parameters (2
     parameters fixed) */
  if (B_WithConstantBackground){
    if (B_FitArea)
      status = mpfit(MPFitThreeGaussFuncACB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
    else
      status = mpfit(MPFitThreeGaussFuncCB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
  }
  else{
    if (B_FitArea)
      status = mpfit(MPFitThreeGaussFuncANB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
    else
      status = mpfit(MPFitThreeGaussFuncNB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
  }

//  cout << "MPFitting_ndarray:MPFitThreeGaussFix: *** testgaussfix status = " << status << endl;
//  PrintResult(p, &result);

  for (int i_par=0; i_par<I_NParams; i_par++){
    D_A1_Coeffs_Out[i_par] = p[i_par];
    D_A1_ECoeffs_Out[i_par] = result.xerror[i_par];
  }

  return true;
}

template<typename T>
bool MPFitThreeGaussLim(const ndarray::Array< T, 1, 1 > &D_A1_X_In,
                        const ndarray::Array< T, 1, 1 > &D_A1_Y_In,
                        const ndarray::Array< T, 1, 1 > &D_A1_EY_In,
                        const ndarray::Array< T, 1, 1 > &D_A1_Guess_In,
                        const ndarray::Array< int, 2, 1 > &I_A2_Limited,
                        const ndarray::Array< T, 2, 1 > &D_A2_Limits,
                        const bool B_WithConstantBackground,
                        const bool B_FitArea,
                        ndarray::Array< T, 1, 1 > &D_A1_Coeffs_Out,
                        ndarray::Array< T, 1, 1 >& D_A1_ECoeffs_Out){
  int I_NParams = 8;
  if (!B_WithConstantBackground)
    I_NParams = 7;
  unsigned int I_NPts = D_A1_X_In.getShape()[0];
  if (D_A1_Y_In.getShape()[0] != I_NPts){
    cout << "MPFitting_ndarray::MPFitThreeGaussLim: ERROR: D_A1_X_In and D_A1_Y_In must have same size" << endl;
    return false;
  }
  if (D_A1_EY_In.getShape()[0] != I_NPts){
    cout << "MPFitting_ndarray::MPFitThreeGaussLim: ERROR: D_A1_X_In and D_A1_EY_In must have same size" << endl;
    return false;
  }
  if (D_A1_Guess_In.getShape()[0] != I_NParams){
    cout << "MPFitting_ndarray::MPFitThreeGaussLim: ERROR: D_A1_Guess_In must have " << I_NParams << " elements" << endl;
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
  //  double pactual[] = {0.0, 4.70, 0.0, 0.5};/* Actual values used to make data*/
  double perror[I_NParams];                        /* Returned parameter errors */
  mp_par pars[I_NParams];                          /* Parameter constraints */
  if (D_A1_Coeffs_Out.getShape()[0] != I_NParams){
    string message("MPFitting_ndarray::MPFitThreeGaussLim: ERROR: D_A1_Coeffs_Out.getShape()[0]=");
    message += to_string(D_A1_Coeffs_Out.getShape()[0]) + " != I_NParams=" + to_string(I_NParams);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  if (D_A1_ECoeffs_Out.getShape()[0] != I_NParams){
    string message("MPFitting_ndarray::MPFitThreeGaussLim: ERROR: D_A1_ECoeffs_Out.getShape()[0]=");
    message += to_string(D_A1_ECoeffs_Out.getShape()[0]) + " != I_NParams=" + to_string(I_NParams);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
//  int i;
  struct vars_struct v;
  int status;
  mp_result result;

  memset(&result,0,sizeof(result));      /* Zero results structure */
  result.xerror = perror;

  memset(pars,0,sizeof(pars));        /* Initialize constraint structure */
  for (int i_par=0; i_par<I_NParams; i_par++){
    pars[i_par].limited[0] = I_A2_Limited[i_par][0];
    pars[i_par].limited[1] = I_A2_Limited[i_par][1];
    pars[i_par].limits[0] = D_A2_Limits[i_par][0];              /* lower parameter limit */
    pars[i_par].limits[1] = D_A2_Limits[i_par][1];              /* upper parameter limit */
  }

  /* How to put limits on a parameter.  In this case, parameter 3 is
     limited to be between -0.3 and +0.2.
  pars[3].limited[0] = 0;
  pars[3].limited[1] = 1;
  pars[3].limits[0] = -0.3;
  pars[3].limits[1] = +0.2;
  */

  v.x = x;
  v.y = y;
  v.ey = ey;

  /* Call fitting function for 10 data points and 4 parameters (2
     parameters fixed) */
  if (B_WithConstantBackground){
    if (B_FitArea)
      status = mpfit(MPFitThreeGaussFuncACB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
    else
      status = mpfit(MPFitThreeGaussFuncCB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
  }
  else{
    if (B_FitArea)
      status = mpfit(MPFitThreeGaussFuncANB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
    else
      status = mpfit(MPFitThreeGaussFuncNB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);
  }

  for (int i_par=0; i_par<I_NParams; i_par++){
    D_A1_Coeffs_Out[i_par] = p[i_par];
    D_A1_ECoeffs_Out[i_par] = result.xerror[i_par];
  }

//  cout << "MPFitting_ndarray:MPFitThreeGaussLim: *** testgaussfix status = " << status << endl;
//  PrintResult(p, &result);

  return true;
}


/*
 * gaussian fit function
 *
 * m - number of data points
 * n - number of parameters (5)
 * p - array of fit parameters
 *     p[0] = peak z value Gauss
 *     p[1] = x centroid position Gauss
 *     p[2] = y centroid position Gauss
 *     p[3] = gaussian sigma width
 *     p[4] = constant offset
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 */
int MPFit2DGaussFuncCB(int m, int n, double *p, double *dz, double **dvec, void *vars){
  int i;
  struct vars_struct *v = (struct vars_struct *) vars;
  double *x, *y, *z;
  double xc, yc, sig2;//, sig2_b;

  x = v->x;
  y = v->y;
  z = v->ey;

  sig2 = p[3]*p[3];
  //  sig2_b = p[6]*p[6];

  for (i=0; i<m; i++) {
    xc = x[i]-p[1];
    yc = y[i]-p[2];
    dz[i] = (z[i] - p[0]*exp(-0.5*((xc*xc) + (yc * yc))/sig2) - p[4]) / sqrt(z[i]);
  }

  return 0;

}

template<typename T>
bool MPFit2DGaussLim(const ndarray::Array< T, 1, 1 >& D_A1_X_In,
                     const ndarray::Array< T, 1, 1 >& D_A1_Y_In,
                     const ndarray::Array< T, 1, 1 >& D_A1_Z_In,
                     const ndarray::Array< T, 1, 1 >& D_A1_Guess_In,
                     const ndarray::Array< int, 2, 1 > &I_A2_Limited,
                     const ndarray::Array< T, 2, 1 > &D_A2_Limits,
//                     const bool B_WithConstantBackground,
//                     const bool B_FitArea,
                     ndarray::Array< T, 1, 1 >& D_A1_Coeffs_Out,
                     ndarray::Array< T, 1, 1 >& D_A1_ECoeffs_Out){
  int I_NParams = 5;
//  if (!B_WithConstantBackground)
//    I_NParams = 3;
  unsigned int I_NPts = D_A1_X_In.getShape()[0];
  if (D_A1_Y_In.getShape()[0] != I_NPts){
    cout << "MPFitting_ndarray::MPFitGaussLim: ERROR: D_A1_X_In and D_A1_Y_In must have same size" << endl;
    return false;
  }
  if (D_A1_Z_In.getShape()[0] != I_NPts){
    cout << "MPFitting_ndarray::MPFitGaussLim: ERROR: D_A1_X_In and D_A1_Z_In must have same size" << endl;
    return false;
  }
  if (D_A1_Guess_In.getShape()[0] != I_NParams){
    cout << "MPFitting_ndarray::MPFitGaussLim: ERROR: D_A1_Guess_In must have " << I_NParams << " elements" << endl;
    return false;
  }
  double x[I_NPts];
  double y[I_NPts];
  double z[I_NPts];
  double p[I_NParams];
  for (int i_pt=0; i_pt<I_NPts; i_pt++){
    x[i_pt] = D_A1_X_In[i_pt];
    y[i_pt] = D_A1_Y_In[i_pt];
    z[i_pt] = D_A1_Z_In[i_pt];
  }
  for (int i_par=0; i_par<I_NParams; i_par++){
    p[i_par] = D_A1_Guess_In[i_par];       /* Initial conditions */
  }
  //  double pactual[] = {0.0, 4.70, 0.0, 0.5};/* Actual values used to make data*/
  double perror[I_NParams];                        /* Returned parameter errors */
  mp_par pars[I_NParams];                          /* Parameter constraints */
  if (D_A1_Coeffs_Out.getShape()[0] != I_NParams){
    string message("MPFitting_ndarray::MPFitGaussLim: ERROR: D_A1_Coeffs_Out.getShape()[0]=");
    message += to_string(D_A1_Coeffs_Out.getShape()[0]) + " != I_NParams=" + to_string(I_NParams);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  if (D_A1_ECoeffs_Out.getShape()[0] != I_NParams){
    string message("MPFitting_ndarray::MPFitGaussLim: ERROR: D_A1_ECoeffs_Out.getShape()[0]=");
    message += to_string(D_A1_ECoeffs_Out.getShape()[0]) + " != I_NParams=" + to_string(I_NParams);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  struct vars_struct v;
  int status;
  mp_result result;

  memset(&result,0,sizeof(result));      /* Zero results structure */
  result.xerror = perror;

  memset(pars,0,sizeof(pars));        /* Initialize constraint structure */
  for (int i_par=0; i_par<I_NParams; i_par++){
    pars[i_par].limited[0] = I_A2_Limited[i_par][0];
    pars[i_par].limited[1] = I_A2_Limited[i_par][1];
    pars[i_par].limits[0] = D_A2_Limits[i_par][0];              /* lower parameter limit */
    pars[i_par].limits[1] = D_A2_Limits[i_par][1];              /* upper parameter limit */
  }

  /* How to put limits on a parameter.  In this case, parameter 3 is
     limited to be between -0.3 and +0.2.
  pars[3].limited[0] = 0;
  pars[3].limited[1] = 1;
  pars[3].limits[0] = -0.3;
  pars[3].limits[1] = +0.2;
  */

  v.x = x;
  v.y = y;
  v.ey = z;

  /* Call fitting function for 10 data points and 4 parameters (2
     parameters fixed) */
//  cout << "MPFitting_ndarray::MPFit2DGaussLim: Starting mpfit: I_NPts = " << I_NPts << ", I_NParams = " << I_NParams << endl;
  status = mpfit(MPFit2DGaussFuncCB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);

  for (int i_par=0; i_par<I_NParams; i_par++){
    D_A1_Coeffs_Out[i_par] = p[i_par];
    D_A1_ECoeffs_Out[i_par] = result.xerror[i_par];
  }

//  cout << "MPFitting_ndarray:MPFit2DGaussLim: *** testgaussfix status = " << status << endl;
//  PrintResult(p, &result);

  return true;
}

int MPFitAnyFunc(int m, int n, double *p, double *dz, double **dvec, void *vars){
  int i;
  struct vars_struct *v = (struct vars_struct *) vars;
  double *x, *y, *z;
  double xc, yc, sig2;//, sig2_b;

  x = v->x;
  y = v->y;
  z = v->ey;

  sig2 = p[4]*p[4];
  //  sig2_b = p[6]*p[6];

  for (i=0; i<m; i++) {
    dz[i] = z[i] - p[0] * yValues[i];
  }

  return 0;

}

template bool MPFitGauss(const ndarray::Array<double, 1, 1> &D_A1_X_In,
                         const ndarray::Array<double, 1, 1> &D_A1_Y_In,
                         const ndarray::Array<double, 1, 1> &D_A1_EY_In,
                         const ndarray::Array<double, 1, 1> &D_A1_Guess_In,
                         const bool B_WithConstantBackground,
                         const bool B_FitArea,
                         ndarray::Array<double, 1, 1> &D_A1_Coeffs_Out,
                         ndarray::Array<double, 1, 1>& D_A1_ECoeffs_Out);
template bool MPFitGauss(const ndarray::Array<float, 1, 1> &D_A1_X_In,
                         const ndarray::Array<float, 1, 1> &D_A1_Y_In,
                         const ndarray::Array<float, 1, 1> &D_A1_EY_In,
                         const ndarray::Array<float, 1, 1> &D_A1_Guess_In,
                         const bool B_WithConstantBackground,
                         const bool B_FitArea,
                         ndarray::Array<float, 1, 1> &D_A1_Coeffs_Out,
                         ndarray::Array<float, 1, 1>& D_A1_ECoeffs_Out);

template bool MPFitGaussFix(const ndarray::Array<double, 1, 1> &D_A1_X_In,
                            const ndarray::Array<double, 1, 1> &D_A1_Y_In,
                            const ndarray::Array<double, 1, 1> &D_A1_EY_In,
                            const ndarray::Array<double, 1, 1> &D_A1_Guess_In,
                            const ndarray::Array< int, 1, 1 > &I_A1_Fix,
                            const bool B_WithConstantBackground,
                            const bool B_FitArea,
                            ndarray::Array<double, 1, 1> & D_A1_Coeffs_Out,
                            ndarray::Array<double, 1, 1> & D_A1_ECoeffs_Out);
template bool MPFitGaussFix(const ndarray::Array<float, 1, 1> &D_A1_X_In,
                            const ndarray::Array<float, 1, 1> &D_A1_Y_In,
                            const ndarray::Array<float, 1, 1> &D_A1_EY_In,
                            const ndarray::Array<float, 1, 1> &D_A1_Guess_In,
                            const ndarray::Array< int, 1, 1 > &I_A1_Fix,
                            const bool B_WithConstantBackground,
                            const bool B_FitArea,
                            ndarray::Array<float, 1, 1> &D_A1_Coeffs_Out,
                            ndarray::Array<float, 1, 1>& D_A1_ECoeffs_Out);

template bool MPFitGaussLim(const ndarray::Array<double, 1, 1> &D_A1_X_In,
                            const ndarray::Array<double, 1, 1> &D_A1_Y_In,
                            const ndarray::Array<double, 1, 1> &D_A1_EY_In,
                            const ndarray::Array<double, 1, 1> &D_A1_Guess_In,
                            const ndarray::Array<int, 2, 1> &I_A2_Limited,
                            const ndarray::Array<double, 2, 1> &D_A2_Limits,
                            const int Background,
                            const bool B_FitArea,
                            ndarray::Array<double, 1, 1> &D_A1_Coeffs_Out,
                            ndarray::Array<double, 1, 1>& D_A1_ECoeffs_Out,
                            bool);
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

template bool MPFitTwoGauss(const ndarray::Array<double, 1, 1> &D_A1_X_In,
                            const ndarray::Array<double, 1, 1> &D_A1_Y_In,
                            const ndarray::Array<double, 1, 1> &D_A1_EY_In,
                            const ndarray::Array<double, 1, 1> &D_A1_Guess_In,
                            const bool B_WithConstantBackground,
                            const bool B_FitArea,
                            ndarray::Array<double, 1, 1> &D_A1_Coeffs_Out,
                            ndarray::Array<double, 1, 1>& D_A1_ECoeffs_Out);
template bool MPFitTwoGauss(const ndarray::Array<float, 1, 1> &D_A1_X_In,
                            const ndarray::Array<float, 1, 1> &D_A1_Y_In,
                            const ndarray::Array<float, 1, 1> &D_A1_EY_In,
                            const ndarray::Array<float, 1, 1> &D_A1_Guess_In,
                            const bool B_WithConstantBackground,
                            const bool B_FitArea,
                            ndarray::Array<float, 1, 1> &D_A1_Coeffs_Out,
                            ndarray::Array<float, 1, 1>& D_A1_ECoeffs_Out);

template bool MPFitTwoGaussFix(const ndarray::Array<double, 1, 1> &D_A1_X_In,
                               const ndarray::Array<double, 1, 1> &D_A1_Y_In,
                               const ndarray::Array<double, 1, 1> &D_A1_EY_In,
                               const ndarray::Array<double, 1, 1> &D_A1_Guess_In,
                               const ndarray::Array< int, 1, 1 > &I_A1_Fix,
                               const bool B_WithConstantBackground,
                               const bool B_FitArea,
                               ndarray::Array<double, 1, 1> &D_A1_Coeffs_Out,
                               ndarray::Array<double, 1, 1>& D_A1_ECoeffs_Out);
template bool MPFitTwoGaussFix(const ndarray::Array<float, 1, 1> &D_A1_X_In,
                               const ndarray::Array<float, 1, 1> &D_A1_Y_In,
                               const ndarray::Array<float, 1, 1> &D_A1_EY_In,
                               const ndarray::Array<float, 1, 1> &D_A1_Guess_In,
                               const ndarray::Array< int, 1, 1 > &I_A1_Fix,
                               const bool B_WithConstantBackground,
                               const bool B_FitArea,
                               ndarray::Array<float, 1, 1> &D_A1_Coeffs_Out,
                               ndarray::Array<float, 1, 1>& D_A1_ECoeffs_Out);

template bool MPFitTwoGaussLim(const ndarray::Array<double, 1, 1> &D_A1_X_In,
                               const ndarray::Array<double, 1, 1> &D_A1_Y_In,
                               const ndarray::Array<double, 1, 1> &D_A1_EY_In,
                               const ndarray::Array<double, 1, 1> &D_A1_Guess_In,
                               const ndarray::Array<int, 2, 1> &I_A2_Limited,
                               const ndarray::Array<double, 2, 1> &D_A2_Limits,
                               const bool B_WithConstantBackground,
                               const bool B_FitArea,
                               ndarray::Array<double, 1, 1> &D_A1_Coeffs_Out,
                               ndarray::Array<double, 1, 1>& D_A1_ECoeffs_Out);
template bool MPFitTwoGaussLim(const ndarray::Array<float, 1, 1> &D_A1_X_In,
                               const ndarray::Array<float, 1, 1> &D_A1_Y_In,
                               const ndarray::Array<float, 1, 1> &D_A1_EY_In,
                               const ndarray::Array<float, 1, 1> &D_A1_Guess_In,
                               const ndarray::Array<int, 2, 1> &I_A2_Limited,
                               const ndarray::Array<float, 2, 1> &D_A2_Limits,
                               const bool B_WithConstantBackground,
                               const bool B_FitArea,
                               ndarray::Array<float, 1, 1> &D_A1_Coeffs_Out,
                               ndarray::Array<float, 1, 1>& D_A1_ECoeffs_Out);

template bool MPFitThreeGauss(const ndarray::Array<double, 1, 1> &D_A1_X_In,
                              const ndarray::Array<double, 1, 1> &D_A1_Y_In,
                              const ndarray::Array<double, 1, 1> &D_A1_EY_In,
                              const ndarray::Array<double, 1, 1> &D_A1_Guess_In,
                              const bool B_WithConstantBackground,
                              const bool B_FitArea,
                              ndarray::Array<double, 1, 1> &D_A1_Coeffs_Out,
                              ndarray::Array<double, 1, 1>& D_A1_ECoeffs_Out);
template bool MPFitThreeGauss(const ndarray::Array<float, 1, 1> &D_A1_X_In,
                              const ndarray::Array<float, 1, 1> &D_A1_Y_In,
                              const ndarray::Array<float, 1, 1> &D_A1_EY_In,
                              const ndarray::Array<float, 1, 1> &D_A1_Guess_In,
                              const bool B_WithConstantBackground,
                              const bool B_FitArea,
                              ndarray::Array<float, 1, 1> &D_A1_Coeffs_Out,
                              ndarray::Array<float, 1, 1>& D_A1_ECoeffs_Out);

template bool MPFitThreeGaussFix(const ndarray::Array<double, 1, 1> &D_A1_X_In,
                                 const ndarray::Array<double, 1, 1> &D_A1_Y_In,
                                 const ndarray::Array<double, 1, 1> &D_A1_EY_In,
                                 const ndarray::Array<double, 1, 1> &D_A1_Guess_In,
                                 const ndarray::Array< int, 1, 1 > &I_A1_Fix,
                                 const bool B_WithConstantBackground,
                                 const bool B_FitArea,
                                 ndarray::Array<double, 1, 1> &D_A1_Coeffs_Out,
                                 ndarray::Array<double, 1, 1>& D_A1_ECoeffs_Out);
template bool MPFitThreeGaussFix(const ndarray::Array<float, 1, 1> &D_A1_X_In,
                                 const ndarray::Array<float, 1, 1> &D_A1_Y_In,
                                 const ndarray::Array<float, 1, 1> &D_A1_EY_In,
                                 const ndarray::Array<float, 1, 1> &D_A1_Guess_In,
                                 const ndarray::Array< int, 1, 1 > &I_A1_Fix,
                                 const bool B_WithConstantBackground,
                                 const bool B_FitArea,
                                 ndarray::Array<float, 1, 1> &D_A1_Coeffs_Out,
                                 ndarray::Array<float, 1, 1>& D_A1_ECoeffs_Out);

template bool MPFitThreeGaussLim(const ndarray::Array<double, 1, 1> &D_A1_X_In,
                                 const ndarray::Array<double, 1, 1> &D_A1_Y_In,
                                 const ndarray::Array<double, 1, 1> &D_A1_EY_In,
                                 const ndarray::Array<double, 1, 1> &D_A1_Guess_In,
                                 const ndarray::Array<int, 2, 1> &I_A2_Limited,
                                 const ndarray::Array<double, 2, 1> &D_A2_Limits,
                                 const bool B_WithConstantBackground,
                                 const bool B_FitArea,
                                 ndarray::Array<double, 1, 1> &D_A1_Coeffs_Out,
                                 ndarray::Array<double, 1, 1>& D_A1_ECoeffs_Out);
template bool MPFitThreeGaussLim(const ndarray::Array<float, 1, 1> &D_A1_X_In,
                                 const ndarray::Array<float, 1, 1> &D_A1_Y_In,
                                 const ndarray::Array<float, 1, 1> &D_A1_EY_In,
                                 const ndarray::Array<float, 1, 1> &D_A1_Guess_In,
                                 const ndarray::Array<int, 2, 1> &I_A2_Limited,
                                 const ndarray::Array<float, 2, 1> &D_A2_Limits,
                                 const bool B_WithConstantBackground,
                                 const bool B_FitArea,
                                 ndarray::Array<float, 1, 1> &D_A1_Coeffs_Out,
                                 ndarray::Array<float, 1, 1>& D_A1_ECoeffs_Out);

template bool MPFit2DGaussLim(const ndarray::Array<double, 1, 1>& D_A1_X_In,
                              const ndarray::Array<double, 1, 1>& D_A1_Y_In,
                              const ndarray::Array<double, 1, 1>& D_A1_Z_In,
                              const ndarray::Array<double, 1, 1>& D_A1_Guess_In,
                              const ndarray::Array<int, 2, 1> &I_A2_Limited,
                              const ndarray::Array<double, 2, 1> &D_A2_Limits,
                              ndarray::Array<double, 1, 1>& D_A1_Coeffs_Out,
                              ndarray::Array<double, 1, 1>& D_A1_ECoeffs_Out);
template bool MPFit2DGaussLim(const ndarray::Array<float, 1, 1>& D_A1_X_In,
                              const ndarray::Array<float, 1, 1>& D_A1_Y_In,
                              const ndarray::Array<float, 1, 1>& D_A1_Z_In,
                              const ndarray::Array<float, 1, 1>& D_A1_Guess_In,
                              const ndarray::Array<int, 2, 1> &I_A2_Limited,
                              const ndarray::Array<float, 2, 1> &D_A2_Limits,
                              ndarray::Array<float, 1, 1>& D_A1_Coeffs_Out,
                              ndarray::Array<float, 1, 1>& D_A1_ECoeffs_Out);

template bool MPFitChebyshev1stKind(const ndarray::Array<float, 1, 1> & D_A1_X_In,
                                    const ndarray::Array<float, 1, 1> & D_A1_Y_In,
                                    const ndarray::Array<float, 1, 1> & D_A1_EY_In,
                                    ndarray::Array<float, 1, 1> & D_A1_Coeffs_Out,
                                    ndarray::Array<float, 1, 1> & D_A1_ECoeffs_Out);
template bool MPFitChebyshev1stKind(const ndarray::Array<double, 1, 1> & D_A1_X_In,
                                    const ndarray::Array<double, 1, 1> & D_A1_Y_In,
                                    const ndarray::Array<double, 1, 1> & D_A1_EY_In,
                                    ndarray::Array<double, 1, 1> & D_A1_Coeffs_Out,
                                    ndarray::Array<double, 1, 1> & D_A1_ECoeffs_Out);

template bool MPFitChebyshev1stKind(const ndarray::Array<float, 1, 1> & D_A1_X_In,
                                    const ndarray::Array<float, 1, 1> & D_A1_Y_In,
                                    const ndarray::Array<float, 1, 1> & D_A1_EY_In,
                                    const ndarray::Array<float, 1, 1> & D_A1_Guess_In,
                                    ndarray::Array<float, 1, 1> & D_A1_Coeffs_Out,
                                    ndarray::Array<float, 1, 1> & D_A1_ECoeffs_Out);
template bool MPFitChebyshev1stKind(const ndarray::Array<double, 1, 1> & D_A1_X_In,
                                    const ndarray::Array<double, 1, 1> & D_A1_Y_In,
                                    const ndarray::Array<double, 1, 1> & D_A1_EY_In,
                                    const ndarray::Array<double, 1, 1> & D_A1_Guess_In,
                                    ndarray::Array<double, 1, 1> & D_A1_Coeffs_Out,
                                    ndarray::Array<double, 1, 1> & D_A1_ECoeffs_Out);
