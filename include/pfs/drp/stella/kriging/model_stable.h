#ifndef VARIOGRAM_MODEL_STABLE_
#define VARIOGRAM_MODEL_STABLE_

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

// stable variograms
extern double stable(double n, double s, double r, double p, double d);

//
// for GSL least square fitting solver
//
extern int stb_f(const gsl_vector *x, void *params, gsl_vector *f);
extern int stb_df(const gsl_vector *x, void *params, gsl_matrix *J);
extern int stb_fdf(const gsl_vector *x, void *params, gsl_vector *f, gsl_matrix *J);

#endif  //VARIOGRAM_MODEL_STABLE_
