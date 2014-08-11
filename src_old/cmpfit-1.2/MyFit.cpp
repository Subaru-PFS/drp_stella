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

#include "MyFit.h"

/* Simple routine to print the fit results */
void PrintResult(double *x, mp_result *result){
  int i;

  if ((x == 0) || (result == 0)) return;
  cout << "MyFit:  CHI-SQUARE = " << result->bestnorm << "    (" << result->nfunc-result->nfree << " DOF)" << endl;
  cout << "MyFit:        NPAR = " << result->npar << endl;
  cout << "MyFit:       NFREE = " << result->nfree << endl;
  cout << "MyFit:     NPEGGED = " << result->npegged << endl;
  cout << "MyFit:     NITER = " << result->niter << endl;
  cout << "MyFit:      NFEV = " << result->nfev << endl << endl;
  for (i=0; i<result->npar; i++) {
    cout << "MyFit:  P[" << i << "] = " << x[i] << " +/- " << result->xerror[i] << endl;
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
 *     p[0] = constant offset
 *     p[1] = peak y value
 *     p[2] = x centroid position
 *     p[3] = gaussian sigma width
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

  sig2 = p[3]*p[3];

  for (i=0; i<m; i++) {
    xc = x[i]-p[2];
    dy[i] = (y[i] - p[1]*exp(-0.5*xc*xc/sig2) - p[0])/ey[i];
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
 *     p[0] = constant offset
 *     p[1] = area under curve value
 *     p[2] = x centroid position
 *     p[3] = gaussian sigma width
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

  sig2 = p[3]*p[3];

  for (i=0; i<m; i++) {
    xc = x[i]-p[2];
    dy[i] = (y[i] - (p[1]*exp(-0.5*xc*xc/sig2)/(sqrt(2. * D_PI) * p[3])) - p[0])/ey[i];
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
 * n - number of parameters (6)
 * p - array of fit parameters
 *     p[0] = constant offset
 *     p[1] = peak y value 1st Gauss
 *     p[2] = x centroid position 1st Gauss
 *     p[3] = gaussian sigma width
 *     p[4] = peak y value 2nd Gauss
 *     p[5] = x centroid position 2nd Gauss
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

  sig2 = p[3]*p[3];
//  sig2_b = p[6]*p[6];

  for (i=0; i<m; i++) {
    xc_a = x[i]-p[2];
    xc_b = x[i]-p[5];
    dy[i] = (y[i] - p[1]*exp(-0.5*xc_a*xc_a/sig2) - p[4]*exp(-0.5*xc_b*xc_b/sig2) - p[0])/ey[i];
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
 *     p[0] = constant offset
 *     p[1] = area under curve value 1st Gauss
 *     p[2] = x centroid position 1st Gauss
 *     p[3] = gaussian sigma width
 *     p[4] = area under curve value 2nd Gauss
 *     p[5] = x centroid position 2nd Gauss
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

  sig2 = p[3]*p[3];
//  sig2_b = p[6]*p[6];

  for (i=0; i<m; i++) {
    xc_a = x[i]-p[2];
    xc_b = x[i]-p[5];
    dy[i] = (y[i] - (p[1]*exp(-0.5*xc_a*xc_a/sig2)/(sqrt(2. * D_PI) * p[3])) - (p[4]*exp(-0.5*xc_b*xc_b/sig2)/(sqrt(2. * D_PI) * p[3])) - p[0])/ey[i];
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
 *     p[0] = constant offset
 *     p[1] = peak y value 1st Gauss
 *     p[2] = x centroid position 1st Gauss
 *     p[3] = gaussian sigma width
 *     p[4] = peak y value 2nd Gauss
 *     p[5] = x centroid position 2nd Gauss
 *     p[6] = peak y value 3rd Gauss
 *     p[7] = x centroid position 3rd Gauss
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

  sig2 = p[3]*p[3];
  //  sig2_b = p[6]*p[6];

  for (i=0; i<m; i++) {
    xc_a = x[i]-p[2];
    xc_b = x[i]-p[5];
    xc_c = x[i]-p[7];
    dy[i] = (y[i] - p[1]*exp(-0.5*xc_a*xc_a/sig2) - p[4]*exp(-0.5*xc_b*xc_b/sig2) - p[6]*exp(-0.5*xc_c*xc_c/sig2) - p[0])/ey[i];
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
 *     p[0] = constant offset
 *     p[1] = area under curve value 1st Gauss
 *     p[2] = x centroid position 1st Gauss
 *     p[3] = gaussian sigma width 1st Gauss
 *     p[4] = area under curve value 2nd Gauss
 *     p[5] = x centroid position 2nd Gauss
 *     p[6] = area under curve value 3rd Gauss
 *     p[7] = x centroid position 3rd Gauss
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

  sig2 = p[3]*p[3];
  //  sig2_b = p[6]*p[6];

  for (i=0; i<m; i++) {
    xc_a = x[i]-p[2];
    xc_b = x[i]-p[5];
    xc_c = x[i]-p[7];
    dy[i] = (y[i] - (p[1]*exp(-0.5*xc_a*xc_a/sig2)/(sqrt(2. * D_PI) * p[3])) - (p[4]*exp(-0.5*xc_b*xc_b/sig2)/(sqrt(2. * D_PI) * p[3])) - (p[6]*exp(-0.5*xc_c*xc_c/sig2)/(sqrt(2. * D_PI) * p[3])) - p[0])/ey[i];
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

/* Test harness routine, which contains test gaussian-peak data */
bool MPFitGauss(const blitz::Array< double, 1 >& D_A1_X_In,
                const blitz::Array< double, 1 >& D_A1_Y_In,
                const blitz::Array< double, 1 >& D_A1_EY_In,
                const blitz::Array< double, 1 >& D_A1_Guess_In,
                const bool B_WithConstantBackground,
                const bool B_FitArea,
                blitz::Array< double, 1 >& D_A1_Coeffs_Out,
                blitz::Array< double, 1 >& D_A1_ECoeffs_Out){
  int I_NParams = 4;
  if (!B_WithConstantBackground)
    I_NParams = 3;
  int I_NPts = D_A1_X_In.size();
  if (D_A1_Y_In.size() != I_NPts){
    cout << "MyFit::MPFitGauss: ERROR: D_A1_X_In and D_A1_Y_In must have same size" << endl;
    return false;
  }
  if (D_A1_EY_In.size() != I_NPts){
    cout << "MyFit::MPFitGauss: ERROR: D_A1_X_In and D_A1_EY_In must have same size" << endl;
    return false;
  }
  if (D_A1_Guess_In.size() != I_NParams){
    cout << "MyFit::MPFitGauss: ERROR: D_A1_Guess_In must have " << I_NParams << " elements" << endl;
    return false;
  }
  double x[I_NPts];
  double y[I_NPts];
  double ey[I_NPts];
  double p[I_NParams];
  for (int i_pt=0; i_pt<I_NPts; i_pt++){
    x[i_pt] = D_A1_X_In(i_pt);
    y[i_pt] = D_A1_Y_In(i_pt);
    ey[i_pt] = D_A1_EY_In(i_pt);
  }
  for (int i_par=0; i_par<I_NParams; i_par++){
    p[i_par] = D_A1_Guess_In(i_par);       /* Initial conditions */
  }
//  double pactual[] = {0.0, 4.70, 0.0, 0.5};/* Actual values used to make data*/
  double perror[I_NParams];			   /* Returned parameter errors */
  mp_par pars[I_NParams];			   /* Parameter constraints */
  D_A1_Coeffs_Out.resize(I_NParams);
  D_A1_ECoeffs_Out.resize(I_NParams);
  int i;
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

//  cout << "MyFit::MPFitGauss: *** testgaussfit status = " << status << endl;
//  PrintResult(p, &result);

  for (int i_par=0; i_par<I_NParams; i_par++){
    D_A1_Coeffs_Out(i_par) = p[i_par];
    D_A1_ECoeffs_Out(i_par) = result.xerror[i_par];
  }

  return true;
}


/* Test harness routine, which contains test gaussian-peak data

   Example of fixing two parameter

   Commented example of how to put boundary constraints
*/
bool MPFitGaussFix(const blitz::Array< double, 1 >& D_A1_X_In,
                   const blitz::Array< double, 1 >& D_A1_Y_In,
                   const blitz::Array< double, 1 >& D_A1_EY_In,
                   const blitz::Array< double, 1 >& D_A1_Guess_In,
                   const blitz::Array< int, 1 >& I_A1_Fix,
                   const bool B_WithConstantBackground,
                   const bool B_FitArea,
                   blitz::Array< double, 1 >& D_A1_Coeffs_Out,
                   blitz::Array< double, 1 >& D_A1_ECoeffs_Out){
  int I_NParams = 4;
  if (!B_WithConstantBackground)
    I_NParams = 3;
  int I_NPts = D_A1_X_In.size();
  if (D_A1_Y_In.size() != I_NPts){
    cout << "MyFit::MPFitGaussFix: ERROR: D_A1_X_In and D_A1_Y_In must have same size" << endl;
    return false;
  }
  if (D_A1_EY_In.size() != I_NPts){
    cout << "MyFit::MPFitGaussFix: ERROR: D_A1_X_In and D_A1_EY_In must have same size" << endl;
    return false;
  }
  if (D_A1_Guess_In.size() != I_NParams){
    cout << "MyFit::MPFitGaussFix: ERROR: D_A1_Guess_In must have " << I_NParams << " elements" << endl;
    return false;
  }
  double x[I_NPts];
  double y[I_NPts];
  double ey[I_NPts];
  double p[I_NParams];
  for (int i_pt=0; i_pt<I_NPts; i_pt++){
    x[i_pt] = D_A1_X_In(i_pt);
    y[i_pt] = D_A1_Y_In(i_pt);
    ey[i_pt] = D_A1_EY_In(i_pt);
  }
  for (int i_par=0; i_par<I_NParams; i_par++){
    p[i_par] = D_A1_Guess_In(i_par);       /* Initial conditions */
  }
  //  double pactual[] = {0.0, 4.70, 0.0, 0.5};/* Actual values used to make data*/
  double perror[I_NParams];                        /* Returned parameter errors */
  mp_par pars[I_NParams];                          /* Parameter constraints */
  D_A1_Coeffs_Out.resize(I_NParams);
  D_A1_ECoeffs_Out.resize(I_NParams);
  int i;
  struct vars_struct v;
  int status;
  mp_result result;

  memset(&result,0,sizeof(result));      /* Zero results structure */
  result.xerror = perror;

  memset(pars,0,sizeof(pars));        /* Initialize constraint structure */
  for (int i_par=0; i_par<I_NParams; i_par++){
    pars[i_par].fixed = I_A1_Fix(i_par);              /* Fix parameters 0 and 2 */
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

//  cout << "MyFit:MPFitGaussFix: *** testgaussfix status = " << status << endl;
//  PrintResult(p, &result);

  for (int i_par=0; i_par<I_NParams; i_par++){
    D_A1_Coeffs_Out(i_par) = p[i_par];
    D_A1_ECoeffs_Out(i_par) = result.xerror[i_par];
  }

  return true;
}

bool MPFitGaussLim(const blitz::Array< double, 1 >& D_A1_X_In,
                   const blitz::Array< double, 1 >& D_A1_Y_In,
                   const blitz::Array< double, 1 >& D_A1_EY_In,
                   const blitz::Array< double, 1 >& D_A1_Guess_In,
                   const blitz::Array<int, 2> &I_A2_Limited,
                   const blitz::Array<double, 2> &D_A2_Limits,
                   const bool B_WithConstantBackground,
                   const bool B_FitArea,
                   blitz::Array< double, 1 >& D_A1_Coeffs_Out,
                   blitz::Array< double, 1 >& D_A1_ECoeffs_Out){
  int I_NParams = 4;
  if (!B_WithConstantBackground)
    I_NParams = 3;
  ///Check Limits
  for (int i_par=0; i_par<I_NParams; i_par++){
    if ((I_A2_Limited(i_par, 0) == 1) && (I_A2_Limited(i_par, 1) == 1)){
      if (D_A2_Limits(i_par, 0) > D_A2_Limits(i_par, 1)){
        cout << "MyFit::MPFitGaussLim: ERROR: D_A2_Limits(" << i_par << ", 0)(=" << D_A2_Limits(i_par, 0) <<") > D_A2_Limits(" << i_par << ", 1)(=" << D_A2_Limits(i_par, 1) << ")" << endl;
        return false;
      }
    }
  }
  int I_NPts = D_A1_X_In.size();
  if (D_A1_Y_In.size() != I_NPts){
    cout << "MyFit::MPFitGaussLim: ERROR: D_A1_X_In and D_A1_Y_In must have same size" << endl;
    return false;
  }
  if (D_A1_EY_In.size() != I_NPts){
    cout << "MyFit::MPFitGaussLim: ERROR: D_A1_X_In and D_A1_EY_In must have same size" << endl;
    return false;
  }
  if (D_A1_Guess_In.size() != I_NParams){
    cout << "MyFit::MPFitGaussLim: ERROR: D_A1_Guess_In must have " << I_NParams << " elements" << endl;
    return false;
  }
  double x[I_NPts];
  double y[I_NPts];
  double ey[I_NPts];
  double p[I_NParams];
  for (int i_pt=0; i_pt<I_NPts; i_pt++){
    x[i_pt] = D_A1_X_In(i_pt);
    y[i_pt] = D_A1_Y_In(i_pt);
    ey[i_pt] = D_A1_EY_In(i_pt);
  }
  for (int i_par=0; i_par<I_NParams; i_par++){
    p[i_par] = D_A1_Guess_In(i_par);       /* Initial conditions */
  }
  //  double pactual[] = {0.0, 4.70, 0.0, 0.5};/* Actual values used to make data*/
  double perror[I_NParams];                        /* Returned parameter errors */
  mp_par pars[I_NParams];                          /* Parameter constraints */
  D_A1_Coeffs_Out.resize(I_NParams);
  D_A1_ECoeffs_Out.resize(I_NParams);
  int i;
  struct vars_struct v;
  int status;
  mp_result result;

  memset(&result,0,sizeof(result));      /* Zero results structure */
  result.xerror = perror;

  memset(pars,0,sizeof(pars));        /* Initialize constraint structure */
  for (int i_par=0; i_par<I_NParams; i_par++){
    pars[i_par].limited[0] = I_A2_Limited(i_par,0);
    pars[i_par].limited[1] = I_A2_Limited(i_par,1);
    pars[i_par].limits[0] = D_A2_Limits(i_par,0);              /* lower parameter limit */
    pars[i_par].limits[1] = D_A2_Limits(i_par,1);              /* upper parameter limit */
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

  for (int i_par=0; i_par<I_NParams; i_par++){
    D_A1_Coeffs_Out(i_par) = p[i_par];
    D_A1_ECoeffs_Out(i_par) = result.xerror[i_par];
  }

//  cout << "MyFit:MPFitGaussLim: *** testgaussfix status = " << status << endl;
//  PrintResult(p, &result);

  return true;
}

/* Test harness routine, which contains test gaussian-peak data */
bool MPFitTwoGauss(const blitz::Array<double, 1> &D_A1_X_In,
                   const blitz::Array<double, 1> &D_A1_Y_In,
                   const blitz::Array<double, 1> &D_A1_EY_In,
                   const blitz::Array<double, 1> &D_A1_Guess_In,
                   const bool B_WithConstantBackground,
                   const bool B_FitArea,
                   blitz::Array<double, 1> &D_A1_Coeffs_Out,
                   blitz::Array< double, 1 >& D_A1_ECoeffs_Out){
  int I_NParams = 6;
  if (!B_WithConstantBackground)
    I_NParams = 5;
  int I_NPts = D_A1_X_In.size();
  if (D_A1_Y_In.size() != I_NPts){
    cout << "MyFit::MPFitTwoGauss: ERROR: D_A1_X_In and D_A1_Y_In must have same size" << endl;
    return false;
  }
  if (D_A1_EY_In.size() != I_NPts){
    cout << "MyFit::MPFitTwoGauss: ERROR: D_A1_X_In and D_A1_EY_In must have same size" << endl;
    return false;
  }
  if (D_A1_Guess_In.size() != I_NParams){
    cout << "MyFit::MPFitTwoGauss: ERROR: D_A1_Guess_In must have " << I_NParams << " elements" << endl;
    return false;
  }
  double x[I_NPts];
  double y[I_NPts];
  double ey[I_NPts];
  double p[I_NParams];
  for (int i_pt=0; i_pt<I_NPts; i_pt++){
    x[i_pt] = D_A1_X_In(i_pt);
    y[i_pt] = D_A1_Y_In(i_pt);
    ey[i_pt] = D_A1_EY_In(i_pt);
  }
  for (int i_par=0; i_par<I_NParams; i_par++){
    p[i_par] = D_A1_Guess_In(i_par);       /* Initial conditions */
  }
//  double pactual[] = {0.0, 4.70, 0.0, 0.5};/* Actual values used to make data*/
  double perror[I_NParams];                        /* Returned parameter errors */
  mp_par pars[I_NParams];                          /* Parameter constraints */
  D_A1_Coeffs_Out.resize(I_NParams);
  D_A1_ECoeffs_Out.resize(I_NParams);
  int i;
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

//  cout << "MyFit::MPFitTwoGauss: *** testgaussfit status = " << status << endl;
//  PrintResult(p, &result);

  for (int i_par=0; i_par<I_NParams; i_par++){
    D_A1_Coeffs_Out(i_par) = p[i_par];
    D_A1_ECoeffs_Out(i_par) = result.xerror[i_par];
  }

  return true;
}


/* Test harness routine, which contains test gaussian-peak data
 *
 *   Example of fixing two parameter
 *
 *   Commented example of how to put boundary constraints
 */
bool MPFitTwoGaussFix(const blitz::Array<double, 1> &D_A1_X_In,
                      const blitz::Array<double, 1> &D_A1_Y_In,
                      const blitz::Array<double, 1> &D_A1_EY_In,
                      const blitz::Array<double, 1> &D_A1_Guess_In,
                      const blitz::Array<int, 1> &I_A1_Fix,
                      const bool B_WithConstantBackground,
                      const bool B_FitArea,
                      blitz::Array<double, 1> &D_A1_Coeffs_Out,
                      blitz::Array< double, 1 >& D_A1_ECoeffs_Out){
  int I_NParams = 6;
  if (!B_WithConstantBackground)
    I_NParams = 5;
  int I_NPts = D_A1_X_In.size();
  if (D_A1_Y_In.size() != I_NPts){
    cout << "MyFit::MPFitTwoGaussFix: ERROR: D_A1_X_In and D_A1_Y_In must have same size" << endl;
    return false;
  }
  if (D_A1_EY_In.size() != I_NPts){
    cout << "MyFit::MPFitTwoGaussFix: ERROR: D_A1_X_In and D_A1_EY_In must have same size" << endl;
    return false;
  }
  if (D_A1_Guess_In.size() != I_NParams){
    cout << "MyFit::MPFitTwoGaussFix: ERROR: D_A1_Guess_In must have " << I_NParams << " elements" << endl;
    return false;
  }
  double x[I_NPts];
  double y[I_NPts];
  double ey[I_NPts];
  double p[I_NParams];
  for (int i_pt=0; i_pt<I_NPts; i_pt++){
    x[i_pt] = D_A1_X_In(i_pt);
    y[i_pt] = D_A1_Y_In(i_pt);
    ey[i_pt] = D_A1_EY_In(i_pt);
  }
  for (int i_par=0; i_par<I_NParams; i_par++){
    p[i_par] = D_A1_Guess_In(i_par);       /* Initial conditions */
  }
  //  double pactual[] = {0.0, 4.70, 0.0, 0.5};/* Actual values used to make data*/
  double perror[I_NParams];                        /* Returned parameter errors */
  mp_par pars[I_NParams];                          /* Parameter constraints */
  D_A1_Coeffs_Out.resize(I_NParams);
  D_A1_ECoeffs_Out.resize(I_NParams);
  int i;
  struct vars_struct v;
  int status;
  mp_result result;

  memset(&result,0,sizeof(result));      /* Zero results structure */
  result.xerror = perror;

  memset(pars,0,sizeof(pars));        /* Initialize constraint structure */
  for (int i_par=0; i_par<I_NParams; i_par++){
    pars[i_par].fixed = I_A1_Fix(i_par);              /* Fix parameters 0 and 2 */
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

//  cout << "MyFit:MPFitTwoGaussFix: *** testgaussfix status = " << status << endl;
//  PrintResult(p, &result);

  for (int i_par=0; i_par<I_NParams; i_par++){
    D_A1_Coeffs_Out(i_par) = p[i_par];
    D_A1_ECoeffs_Out(i_par) = result.xerror[i_par];
  }

  return true;
}

bool MPFitTwoGaussLim(const blitz::Array<double, 1> &D_A1_X_In,
                      const blitz::Array<double, 1> &D_A1_Y_In,
                      const blitz::Array<double, 1> &D_A1_EY_In,
                      const blitz::Array<double, 1> &D_A1_Guess_In,
                      const blitz::Array<int, 2> &I_A2_Limited,
                      const blitz::Array<double, 2> &D_A2_Limits,
                      const bool B_WithConstantBackground,
                      const bool B_FitArea,
                      blitz::Array<double, 1> &D_A1_Coeffs_Out,
                      blitz::Array< double, 1 >& D_A1_ECoeffs_Out){
  int I_NParams = 6;
  if (!B_WithConstantBackground)
    I_NParams = 5;
  int I_NPts = D_A1_X_In.size();
  if (D_A1_Y_In.size() != I_NPts){
    cout << "MyFit::MPFitTwoGaussLim: ERROR: D_A1_X_In and D_A1_Y_In must have same size" << endl;
    return false;
  }
  if (D_A1_EY_In.size() != I_NPts){
    cout << "MyFit::MPFitTwoGaussLim: ERROR: D_A1_X_In and D_A1_EY_In must have same size" << endl;
    return false;
  }
  if (D_A1_Guess_In.size() != I_NParams){
    cout << "MyFit::MPFitTwoGaussLim: ERROR: D_A1_Guess_In must have " << I_NParams << " elements" << endl;
    return false;
  }
  double x[I_NPts];
  double y[I_NPts];
  double ey[I_NPts];
  double p[I_NParams];
  for (int i_pt=0; i_pt<I_NPts; i_pt++){
    x[i_pt] = D_A1_X_In(i_pt);
    y[i_pt] = D_A1_Y_In(i_pt);
    ey[i_pt] = D_A1_EY_In(i_pt);
  }
  for (int i_par=0; i_par<I_NParams; i_par++){
    p[i_par] = D_A1_Guess_In(i_par);       /* Initial conditions */
  }
  //  double pactual[] = {0.0, 4.70, 0.0, 0.5};/* Actual values used to make data*/
  double perror[I_NParams];                        /* Returned parameter errors */
  mp_par pars[I_NParams];                          /* Parameter constraints */
  D_A1_Coeffs_Out.resize(I_NParams);
  D_A1_ECoeffs_Out.resize(I_NParams);
  int i;
  struct vars_struct v;
  int status;
  mp_result result;

  memset(&result,0,sizeof(result));      /* Zero results structure */
  result.xerror = perror;

  memset(pars,0,sizeof(pars));        /* Initialize constraint structure */
  for (int i_par=0; i_par<I_NParams; i_par++){
    pars[i_par].limited[0] = I_A2_Limited(i_par,0);
    pars[i_par].limited[1] = I_A2_Limited(i_par,1);
    pars[i_par].limits[0] = D_A2_Limits(i_par,0);              /* lower parameter limit */
    pars[i_par].limits[1] = D_A2_Limits(i_par,1);              /* upper parameter limit */
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
    D_A1_Coeffs_Out(i_par) = p[i_par];
    D_A1_ECoeffs_Out(i_par) = result.xerror[i_par];
  }

//  cout << "MyFit:MPFitTwoGaussLim: *** testgaussfix status = " << status << endl;
//  PrintResult(p, &result);

  return true;
}

bool MPFitThreeGauss(const blitz::Array<double, 1> &D_A1_X_In,
                     const blitz::Array<double, 1> &D_A1_Y_In,
                     const blitz::Array<double, 1> &D_A1_EY_In,
                     const blitz::Array<double, 1> &D_A1_Guess_In,
                     const bool B_WithConstantBackground,
                     const bool B_FitArea,
                     blitz::Array<double, 1> &D_A1_Coeffs_Out,
                     blitz::Array< double, 1 >& D_A1_ECoeffs_Out){
  int I_NParams = 8;
  if (!B_WithConstantBackground)
    I_NParams = 7;
  int I_NPts = D_A1_X_In.size();
  if (D_A1_Y_In.size() != I_NPts){
    cout << "MyFit::MPFitThreeGauss: ERROR: D_A1_X_In and D_A1_Y_In must have same size" << endl;
    return false;
  }
  if (D_A1_EY_In.size() != I_NPts){
    cout << "MyFit::MPFitThreeGauss: ERROR: D_A1_X_In and D_A1_EY_In must have same size" << endl;
    return false;
  }
  if (D_A1_Guess_In.size() != I_NParams){
    cout << "MyFit::MPFitThreeGauss: ERROR: D_A1_Guess_In must have " << I_NParams << " elements" << endl;
    return false;
  }
  double x[I_NPts];
  double y[I_NPts];
  double ey[I_NPts];
  double p[I_NParams];
  for (int i_pt=0; i_pt<I_NPts; i_pt++){
    x[i_pt] = D_A1_X_In(i_pt);
    y[i_pt] = D_A1_Y_In(i_pt);
    ey[i_pt] = D_A1_EY_In(i_pt);
  }
  for (int i_par=0; i_par<I_NParams; i_par++){
    p[i_par] = D_A1_Guess_In(i_par);       /* Initial conditions */
  }
//  double pactual[] = {0.0, 4.70, 0.0, 0.5};/* Actual values used to make data*/
  double perror[I_NParams];                        /* Returned parameter errors */
  mp_par pars[I_NParams];                          /* Parameter constraints */
  D_A1_Coeffs_Out.resize(I_NParams);
  D_A1_ECoeffs_Out.resize(I_NParams);
  int i;
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

//  cout << "MyFit::MPFitThreeGauss: *** testgaussfit status = " << status << endl;
//  PrintResult(p, &result);

  for (int i_par=0; i_par<I_NParams; i_par++){
    D_A1_Coeffs_Out(i_par) = p[i_par];
    D_A1_ECoeffs_Out(i_par) = result.xerror[i_par];
  }

  return true;
}

bool MPFitThreeGaussFix(const blitz::Array<double, 1> &D_A1_X_In,
                        const blitz::Array<double, 1> &D_A1_Y_In,
                        const blitz::Array<double, 1> &D_A1_EY_In,
                        const blitz::Array<double, 1> &D_A1_Guess_In,
                        const blitz::Array<int, 1> &I_A1_Fix,
                        const bool B_WithConstantBackground,
                        const bool B_FitArea,
                        blitz::Array<double, 1> &D_A1_Coeffs_Out,
                        blitz::Array< double, 1 >& D_A1_ECoeffs_Out){
  int I_NParams = 8;
  if (!B_WithConstantBackground)
    I_NParams = 7;
  int I_NPts = D_A1_X_In.size();
  if (D_A1_Y_In.size() != I_NPts){
    cout << "MyFit::MPFitThreeGaussFix: ERROR: D_A1_X_In and D_A1_Y_In must have same size" << endl;
    return false;
  }
  if (D_A1_EY_In.size() != I_NPts){
    cout << "MyFit::MPFitThreeGaussFix: ERROR: D_A1_X_In and D_A1_EY_In must have same size" << endl;
    return false;
  }
  if (D_A1_Guess_In.size() != I_NParams){
    cout << "MyFit::MPFitThreeGaussFix: ERROR: D_A1_Guess_In must have " << I_NParams << " elements" << endl;
    return false;
  }
  double x[I_NPts];
  double y[I_NPts];
  double ey[I_NPts];
  double p[I_NParams];
  for (int i_pt=0; i_pt<I_NPts; i_pt++){
    x[i_pt] = D_A1_X_In(i_pt);
    y[i_pt] = D_A1_Y_In(i_pt);
    ey[i_pt] = D_A1_EY_In(i_pt);
  }
  for (int i_par=0; i_par<I_NParams; i_par++){
    p[i_par] = D_A1_Guess_In(i_par);       /* Initial conditions */
  }
  //  double pactual[] = {0.0, 4.70, 0.0, 0.5};/* Actual values used to make data*/
  double perror[I_NParams];                        /* Returned parameter errors */
  mp_par pars[I_NParams];                          /* Parameter constraints */
  D_A1_Coeffs_Out.resize(I_NParams);
  D_A1_ECoeffs_Out.resize(I_NParams);
  int i;
  struct vars_struct v;
  int status;
  mp_result result;

  memset(&result,0,sizeof(result));      /* Zero results structure */
  result.xerror = perror;

  memset(pars,0,sizeof(pars));        /* Initialize constraint structure */
  for (int i_par=0; i_par<I_NParams; i_par++){
    pars[i_par].fixed = I_A1_Fix(i_par);              /* Fix parameters 0 and 2 */
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

//  cout << "MyFit:MPFitThreeGaussFix: *** testgaussfix status = " << status << endl;
//  PrintResult(p, &result);

  for (int i_par=0; i_par<I_NParams; i_par++){
    D_A1_Coeffs_Out(i_par) = p[i_par];
    D_A1_ECoeffs_Out(i_par) = result.xerror[i_par];
  }

  return true;
}

bool MPFitThreeGaussLim(const blitz::Array<double, 1> &D_A1_X_In,
                        const blitz::Array<double, 1> &D_A1_Y_In,
                        const blitz::Array<double, 1> &D_A1_EY_In,
                        const blitz::Array<double, 1> &D_A1_Guess_In,
                        const blitz::Array<int, 2> &I_A2_Limited,
                        const blitz::Array<double, 2> &D_A2_Limits,
                        const bool B_WithConstantBackground,
                        const bool B_FitArea,
                        blitz::Array<double, 1> &D_A1_Coeffs_Out,
                        blitz::Array< double, 1 >& D_A1_ECoeffs_Out){
  int I_NParams = 8;
  if (!B_WithConstantBackground)
    I_NParams = 7;
  int I_NPts = D_A1_X_In.size();
  if (D_A1_Y_In.size() != I_NPts){
    cout << "MyFit::MPFitThreeGaussLim: ERROR: D_A1_X_In and D_A1_Y_In must have same size" << endl;
    return false;
  }
  if (D_A1_EY_In.size() != I_NPts){
    cout << "MyFit::MPFitThreeGaussLim: ERROR: D_A1_X_In and D_A1_EY_In must have same size" << endl;
    return false;
  }
  if (D_A1_Guess_In.size() != I_NParams){
    cout << "MyFit::MPFitThreeGaussLim: ERROR: D_A1_Guess_In must have " << I_NParams << " elements" << endl;
    return false;
  }
  double x[I_NPts];
  double y[I_NPts];
  double ey[I_NPts];
  double p[I_NParams];
  for (int i_pt=0; i_pt<I_NPts; i_pt++){
    x[i_pt] = D_A1_X_In(i_pt);
    y[i_pt] = D_A1_Y_In(i_pt);
    ey[i_pt] = D_A1_EY_In(i_pt);
  }
  for (int i_par=0; i_par<I_NParams; i_par++){
    p[i_par] = D_A1_Guess_In(i_par);       /* Initial conditions */
  }
  //  double pactual[] = {0.0, 4.70, 0.0, 0.5};/* Actual values used to make data*/
  double perror[I_NParams];                        /* Returned parameter errors */
  mp_par pars[I_NParams];                          /* Parameter constraints */
  D_A1_Coeffs_Out.resize(I_NParams);
  D_A1_ECoeffs_Out.resize(I_NParams);
  int i;
  struct vars_struct v;
  int status;
  mp_result result;

  memset(&result,0,sizeof(result));      /* Zero results structure */
  result.xerror = perror;

  memset(pars,0,sizeof(pars));        /* Initialize constraint structure */
  for (int i_par=0; i_par<I_NParams; i_par++){
    pars[i_par].limited[0] = I_A2_Limited(i_par,0);
    pars[i_par].limited[1] = I_A2_Limited(i_par,1);
    pars[i_par].limits[0] = D_A2_Limits(i_par,0);              /* lower parameter limit */
    pars[i_par].limits[1] = D_A2_Limits(i_par,1);              /* upper parameter limit */
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
    D_A1_Coeffs_Out(i_par) = p[i_par];
    D_A1_ECoeffs_Out(i_par) = result.xerror[i_par];
  }

//  cout << "MyFit:MPFitThreeGaussLim: *** testgaussfix status = " << status << endl;
//  PrintResult(p, &result);

  return true;
}


/*
 * gaussian fit function
 *
 * m - number of data points
 * n - number of parameters (5)
 * p - array of fit parameters
 *     p[0] = constant offset
 *     p[1] = peak z value Gauss
 *     p[2] = x centroid position Gauss
 *     p[3] = y centroid position Gauss
 *     p[4] = gaussian sigma width
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

  sig2 = p[4]*p[4];
  //  sig2_b = p[6]*p[6];

  for (i=0; i<m; i++) {
    xc = x[i]-p[2];
    yc = y[i]-p[3];
    dz[i] = (z[i] - p[1]*exp(-0.5*((xc*xc) + (yc * yc))/sig2) - p[0]) / sqrt(z[i]);
  }

  return 0;

}


bool MPFit2DGaussLim(const blitz::Array< double, 1 >& D_A1_X_In,
                     const blitz::Array< double, 1 >& D_A1_Y_In,
                     const blitz::Array< double, 1 >& D_A1_Z_In,
                     const blitz::Array< double, 1 >& D_A1_Guess_In,
                     const blitz::Array<int, 2> &I_A2_Limited,
                     const blitz::Array<double, 2> &D_A2_Limits,
//                     const bool B_WithConstantBackground,
//                     const bool B_FitArea,
                     blitz::Array< double, 1 >& D_A1_Coeffs_Out,
                     blitz::Array< double, 1 >& D_A1_ECoeffs_Out){
  int I_NParams = 5;
//  if (!B_WithConstantBackground)
//    I_NParams = 3;
  int I_NPts = D_A1_X_In.size();
  if (D_A1_Y_In.size() != I_NPts){
    cout << "MyFit::MPFitGaussLim: ERROR: D_A1_X_In and D_A1_Y_In must have same size" << endl;
    return false;
  }
  if (D_A1_Z_In.size() != I_NPts){
    cout << "MyFit::MPFitGaussLim: ERROR: D_A1_X_In and D_A1_Z_In must have same size" << endl;
    return false;
  }
  if (D_A1_Guess_In.size() != I_NParams){
    cout << "MyFit::MPFitGaussLim: ERROR: D_A1_Guess_In must have " << I_NParams << " elements" << endl;
    return false;
  }
  double x[I_NPts];
  double y[I_NPts];
  double z[I_NPts];
  double p[I_NParams];
  for (int i_pt=0; i_pt<I_NPts; i_pt++){
    x[i_pt] = D_A1_X_In(i_pt);
    y[i_pt] = D_A1_Y_In(i_pt);
    z[i_pt] = D_A1_Z_In(i_pt);
  }
  for (int i_par=0; i_par<I_NParams; i_par++){
    p[i_par] = D_A1_Guess_In(i_par);       /* Initial conditions */
  }
  //  double pactual[] = {0.0, 4.70, 0.0, 0.5};/* Actual values used to make data*/
  double perror[I_NParams];                        /* Returned parameter errors */
  mp_par pars[I_NParams];                          /* Parameter constraints */
  D_A1_Coeffs_Out.resize(I_NParams);
  D_A1_ECoeffs_Out.resize(I_NParams);
  int i;
  struct vars_struct v;
  int status;
  mp_result result;

  memset(&result,0,sizeof(result));      /* Zero results structure */
  result.xerror = perror;

  memset(pars,0,sizeof(pars));        /* Initialize constraint structure */
  for (int i_par=0; i_par<I_NParams; i_par++){
    pars[i_par].limited[0] = I_A2_Limited(i_par,0);
    pars[i_par].limited[1] = I_A2_Limited(i_par,1);
    pars[i_par].limits[0] = D_A2_Limits(i_par,0);              /* lower parameter limit */
    pars[i_par].limits[1] = D_A2_Limits(i_par,1);              /* upper parameter limit */
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
//  cout << "MyFit::MPFit2DGaussLim: Starting mpfit: I_NPts = " << I_NPts << ", I_NParams = " << I_NParams << endl;
  status = mpfit(MPFit2DGaussFuncCB, I_NPts, I_NParams, p, pars, 0, (void *) &v, &result);

  for (int i_par=0; i_par<I_NParams; i_par++){
    D_A1_Coeffs_Out(i_par) = p[i_par];
    D_A1_ECoeffs_Out(i_par) = result.xerror[i_par];
  }

//  cout << "MyFit:MPFit2DGaussLim: *** testgaussfix status = " << status << endl;
//  PrintResult(p, &result);

  return true;
}
