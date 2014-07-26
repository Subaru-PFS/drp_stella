#include <math.h>
#include "vario_model.h"

double stable(double n, double s, double r, double p, double d)
{
        double t = pow(d / r, p);
        if (t >= 700.0)
                return n + s;
        return n + s * (1.0 - exp(-t));
}


//
// following three functions are defined according to GSL Nonlinear Least-Squares Fitting Solver
// Related link: http://www.gnu.org/software/gsl/manual/gsl-ref_37.html#SEC476
//


int stb_f(const gsl_vector *x, void *params, gsl_vector *f)
{
        size_t n = ((LSFParam *)params)->n;
        double *dist = ((LSFParam *)params)->dist;
        double *vario = ((LSFParam *)params)->vario;
        double power = ((LSFParam *)params)->power;

        double sill = gsl_vector_get(x, 0);
        double range = gsl_vector_get(x, 1);
        double nugget = gsl_vector_get(x, 2);

        for (size_t i = 0;i < n;i++)
        {
                double Yi = stable(nugget, sill, range, power, dist[i]);
                gsl_vector_set(f, i, Yi - vario[i]);
        }

        return GSL_SUCCESS;
}

int stb_df(const gsl_vector *x, void *params, gsl_matrix *J)
{
        size_t n = ((LSFParam *)params)->n;
        double *dist = ((LSFParam *)params)->dist;
        double *vario = ((LSFParam *)params)->vario;
        double power = ((LSFParam *)params)->power;

        double sill = gsl_vector_get(x, 0);
        double range = gsl_vector_get (x, 1);
        double nugget = gsl_vector_get(x, 2);

        for (size_t i = 0;i < n;i++)
        {
                double e = exp(-pow(dist[i] / range, power));
                // df / d(sill)
                gsl_matrix_set(J, i, 0, 1.0 - e);
                // df / d(range)
                gsl_matrix_set(J, i, 1,  -power * sill / range * pow(dist[i] / range, power) * e);
                // df / d(nugget)
                gsl_matrix_set(J, i, 2, 1.0);
        }
        return GSL_SUCCESS;
}

int stb_fdf(const gsl_vector *x, void *params, gsl_vector *f, gsl_matrix *J)
{
        stb_f(x, params, f);
        stb_df(x, params, J);
        return GSL_SUCCESS;
}
