#ifndef VARIOGRAM_MODEL_SPHERICAL_
#define VARIOGRAM_MODEL_SPHERICAL_

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

// spherical variogram model
extern double spherical(double n, double s, double r, double d);

//
// for GSL least square fitting solver
//
extern int sph_f(const gsl_vector *x, void *params, gsl_vector *f);
extern int sph_df(const gsl_vector *x, void *params, gsl_matrix *J);
extern int sph_fdf(const gsl_vector *x, void *params, gsl_vector *f, gsl_matrix *J);

#endif  //VARIOGRAM_MODEL_SPHERICAL_
