#include "variogram.h"
#include "vario_model.h"
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlin.h>

CVariogram::CVariogram(void)
{
        m_pDistance = NULL;
        m_pVariogram = NULL;
        m_samples = 0;

        m_sill = 0.0;
        m_range = 0.0;
        m_nugget = 0.0;
        m_power = 0.0;

        m_model = VARIO_NONE;
}

CVariogram::~CVariogram(void)
{
        destroy();
}

bool CVariogram::allocate(size_t smpl)
{
        destroy();

        m_pDistance = new double[smpl];
        m_pVariogram = new double[smpl];

        if (m_pDistance == NULL || m_pVariogram == NULL)
        {
                destroy();
                return false;
        }
        m_samples = smpl;

        return true;
}

void CVariogram::destroy(void)
{
        if (m_pDistance != NULL)
        {
                delete[] m_pDistance;
                m_pDistance = NULL;
        }
        if (m_pVariogram != NULL)
        {
                delete[] m_pVariogram;
                m_pVariogram = NULL;
        }

        m_model = VARIO_NONE;
        m_nugget = 0.0;
        m_sill = 0.0;
        m_range = 0.0;
        m_power = 0.0;
}

bool CVariogram::setSample(const std::vector<CVariogram::VarioItm> &vecSample)
{
        if (vecSample.empty())
                return false;

        if (!allocate(vecSample.size()))
                return false;

        for (size_t i = 0;i < samples();i++)
        {
                #ifdef __DEBUG_KRIGING__
                  cout << "CVariogram::setSample: running sample " << i << ": m_pDistance[i] = " << vecSample[i].distance << ", m_pVariogram[i] = " << vecSample[i].dissimilarity << endl;
                #endif
                m_pDistance[i] = vecSample[i].distance;
                m_pVariogram[i] = vecSample[i].dissimilarity;
        }

        cout << "CVariogram::setSample: Starting SortByDistance()" << endl;
        sortByDistance();
        m_model = VARIO_NONE;

        return true;
}

bool CVariogram::setModel(int model, double nugget, double sill, double range, double power, double step, double maxdist)
{
        if (model >= CVariogram::VARIO_NUM)
                return false;

        if (!allocate(static_cast<size_t>(fabs(maxdist / step))))
                return false;

        m_model = model;
        m_nugget = nugget;
        m_sill = sill;
        m_range = range;
        m_power = power;

        for (size_t i = 0;i < samples();i++)
        {
                m_pDistance[i] = step * i;
                m_pVariogram[i] = getModelData(step * i);
        }
        return true;
}

bool CVariogram::getSample(size_t smpl, double &dist, double &vario) const
{
        if (smpl >= samples())
                return false;

        dist = m_pDistance[smpl];
        vario = m_pVariogram[smpl];
        return true;
}

size_t CVariogram::countLessDist(double cap) const
{
        if (!isActive())
                return 0;

        size_t cnt =0;
        while (++cnt < m_samples && m_pDistance[cnt] < cap);
        return cnt;
}

int CVariogram::estimateModel(int model, double &nugget, double &sill, double &range, double power, double maxdist)
{
        double init_guess[3] = {sill, range, nugget};

        gsl_vector_view x;
        gsl_vector *g;

        // ƒpƒ‰ƒ[ƒ^ƒZƒbƒeƒBƒ“ƒO
        if (power < 0.0 || power > 2.0)
                return GSL_FAILURE;

        LSFParam para;
        para.dist = m_pDistance;
        para.vario = m_pVariogram;
        para.n = countLessDist(maxdist);
        para.power = power;

        // Å“K‰»‘ÎÛŠÖ”‚Ì“o˜^
        gsl_multifit_function_fdf f;

        m_model = model;
        switch (model)
        {
        case VARIO_SPH:
                f.f = &sph_f;
                f.df = &sph_df;
                f.fdf = &sph_fdf;
                break;
        case VARIO_STB:
                f.f = &stb_f;
                f.df = &stb_df;
                f.fdf = &stb_fdf;
                break;
        default:
                m_model = VARIO_NONE;
                return GSL_FAILURE;
        }

        f.n = para.n;
        f.params = &para;
        switch (model)
        {
        case VARIO_SPH:
        case VARIO_STB:
                f.p = 3;
                break;
        default:
                break;
        }

        x = gsl_vector_view_array(init_guess, f.p);
        g = gsl_vector_alloc(f.p);

        // initialize NLSF solver
        const gsl_multifit_fdfsolver_type *T = gsl_multifit_fdfsolver_lmsder;
        gsl_multifit_fdfsolver *s = gsl_multifit_fdfsolver_alloc(T, f.n, f.p);
        gsl_multifit_fdfsolver_set(s, &f, &x.vector);

        int status;
        size_t iter =0;

        // iteration
        do
        {
                iter++;
                status = gsl_multifit_fdfsolver_iterate(s);

                if (status > 0)
                        break;

                // convergence check by variation
//              status = gsl_multifit_test_delta(s->dx, s->x, 1e-4, 1e-4);

                // convergence check by gradient
                gsl_multifit_gradient(s->J, s->f, g);
                status = gsl_multifit_test_gradient(g, 0.01);

        } while (status == GSL_CONTINUE && iter < 500);

        if (iter >= 500)
        {
                GSL_ERROR_VAL("Variogram Estimator: Didn't converge...\n", status, false);
        }

        m_nugget = 0.0;
        m_sill = 0.0;
        m_range = 0.0;
        m_power = 0.0;

        switch (model)
        {
        case VARIO_SPH:
        case VARIO_STB:
                m_nugget = gsl_vector_get(s->x, 2);
                m_sill = gsl_vector_get(s->x, 0);
                m_range = fabs(gsl_vector_get(s->x, 1));
                break;

        default:
                break;
        }

        nugget = m_nugget;
        sill = m_sill;
        range = m_range;
        power = m_power;

        gsl_vector_free(g);
        gsl_multifit_fdfsolver_free(s);

        return status;
}

double CVariogram::getModelData(double dist) const
{
        switch (m_model)
        {
        case VARIO_SPH:
                return spherical(m_nugget, m_sill, m_range, dist);
        case VARIO_STB:
                return stable(m_nugget, m_sill, m_range, m_power, dist);
        default:
                break;
        }
        return -1;
}

double CVariogram::getModelCovariance(double dist) const
{
        switch (m_model)
        {
        case VARIO_SPH:
                return m_sill - spherical(m_nugget, m_sill, m_range, dist);
        case VARIO_STB:
                return m_sill - stable(m_nugget, m_sill, m_range, m_power, dist);
        default:
                break;
        }

        // if variogram does not has boundary ...
        return -1.0;
}

bool CVariogram::sortByDistance(void)
{
        if (!isActive())
                return false;

        for (size_t i = 0;i < m_samples;i++)
        {
	  cout << "CVariogram::sortByDistance: running sample " << i << " of " << m_samples << endl;
                for (size_t j = i + 1;j < m_samples;j++)
                {
                        if (m_pDistance[i] > m_pDistance[j])
                        {
                                std::swap(m_pDistance[i], m_pDistance[j]);
                                std::swap(m_pVariogram[i], m_pVariogram[j]);
                        }
                }
        }
        return true;
}

bool CVariogram::bubbleSortByDistance(void)
{
  if (!isActive())
    return false;

  long UpperLimit = m_samples - 1;
  long LastSwap;

  while(UpperLimit > 0)
  {
    LastSwap = 0;
    for(size_t Pos = 0;Pos < UpperLimit; ++Pos)
    {
      if(m_pDistance[Pos] > m_pDistance[Pos+1])
      {
        //        cout << "BubbleSort: (*P_D_A1_Out)(Pos=" << Pos << ")=" << (*P_D_A1_Out)(Pos) << " > (*P_D_A1_Out)(Pos+1=" << Pos+1 << ") = " << (*P_D_A1_Out)(Pos+1) << " => Swapping Positions" << endl;
        std::swap(m_pDistance[Pos], m_pDistance[Pos+1]);
        std::swap(m_pVariogram[Pos], m_pVariogram[Pos+1]);
        LastSwap = Pos;
//        cout << "BubbleSort: LastSwap set to " << LastSwap << endl;
      }
    }
    UpperLimit = LastSwap;
//    cout << "BubbleSort: UpperLimit set to " << LastSwap << endl;
  }
  return true;
}