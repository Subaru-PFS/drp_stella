#include <math.h>
#include "pfs/drp/stella/kriging/vario_model.h"

double spherical(double n, double s, double r, double d)
{
        if (d > r)
                return n + s;
        return n + s * (3.0 / 2.0 * fabs(d) / r - pow(fabs(d) / r, 3.0) / 2);
}


//
// following three functions are defined according to GSL Nonlinear Least-Squares Fitting Solver
// Related link: http://www.gnu.org/software/gsl/manual/gsl-ref_37.html#SEC476
//


int sph_f( const gsl_vector *x, void *params, gsl_vector *f )
{
        size_t n = ((LSFParam *)params)->n;
        double *dist = ((LSFParam *)params)->dist;
        double *vario = ((LSFParam *)params)->vario;

        double sill = gsl_vector_get(x, 0);
        double range = gsl_vector_get(x, 1);
        double nugget = gsl_vector_get(x, 2);

        for (size_t i = 0;i < n;i++)
        {
                double Yi = spherical(nugget, sill, range, dist[i]);
                gsl_vector_set(f, i, Yi - vario[i]);
        }

        return GSL_SUCCESS;
}

int sph_df(const gsl_vector *x, void *params, gsl_matrix *J)
{
        size_t n = ((LSFParam *)params)->n;
        double *dist = ((LSFParam *)params)->dist;
//        double *vario = ((LSFParam *)params)->vario;

//        double sill = gsl_vector_get(x, 0);
        double range = gsl_vector_get (x, 1);
//        double nugget = gsl_vector_get(x, 2);

        for (size_t i = 0;i < n;i++)
        {
                if (dist[i] > range)
                {
                        // df / d(sill)
                        gsl_matrix_set(J, i, 0, 1.0);
                        // df / d(range)
                        gsl_matrix_set(J, i, 1, 0.0);
                }
                else
                {
                        double sp = fabs(dist[i]) / range;
                        double tp = pow(sp, 3.0);
                        // df / d(sill)
                        gsl_matrix_set(J, i, 0, 3.0 / 2.0 * sp - tp / 2.0);
                        // df / d(range)
                        gsl_matrix_set(J, i, 1, 3.0 / 2.0 * sp / range - 3.0 / 2.0 * tp / range);
                }
                // df / d(nugget)
                gsl_matrix_set(J, i, 2, 1.0);
        }
        return GSL_SUCCESS;
}

int sph_fdf(const gsl_vector *x, void *params, gsl_vector *f, gsl_matrix *J)
{
        sph_f(x, params, f);
        sph_df(x, params, J);
        return GSL_SUCCESS;
}
