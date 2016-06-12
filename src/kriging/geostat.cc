 #include "pfs/drp/stella/kriging/variogram.h"
 #include "pfs/drp/stella/kriging/geostat.h"
 #include <math.h>
 #include <map>
 #include <gsl/gsl_blas.h>
 #include <gsl/gsl_linalg.h>
 #include <algorithm>
 #include "pfs/drp/stella/kriging/matrix_ex.h"

 double isoDist( const gsl_vector *c1, const gsl_vector *c2 )
 {
         double norm;
         gsl_vector *tmp;

         tmp = gsl_vector_alloc(c1->size);
         gsl_vector_memcpy(tmp, c1);
         gsl_vector_sub(tmp, c2);
         ///TODO: check if sum(abs(tmp_i)) or sum(pow2(tmp_i)) would do (much faster)
         norm = gsl_blas_dnrm2(tmp);
         gsl_vector_free(tmp);

         return norm;
 }

 CGeostat::CGeostat(void)
 {
         m_coords.clear();
         m_pData = NULL;
         m_pTrend = NULL;
         m_domain.clear();
         m_pVarioCloud = NULL;
         m_pVariogram = NULL;
         m_pOKSys = NULL;
 }

 CGeostat::~CGeostat(void)
 {
         destroy();
 }

 bool CGeostat::initialize(size_t smpl, size_t dim)
 {
         destroy();
         return allocate(smpl, dim);
 }

 bool CGeostat::allocate(size_t smpl, size_t dim)
 {
         destroy();

         //cout << "CGeostat::allocate: Starting push_back1" << endl;
         for (size_t s = 0;s < smpl;s++)
                 m_coords.push_back(gsl_vector_alloc(dim));

         //cout << "CGeostat::allocate: Starting push_back2" << endl;
         for (size_t d = 0;d < dim;d++)
                 m_domain.push_back( std::pair<double, double>(0, 0) );

         //cout << "CGeostat::allocate: Starting m_pData gsl_vector_alloc" << endl;
         m_pData = gsl_vector_alloc(smpl);
         //cout << "CGeostat::allocate: Starting m_pTrend gsl_vector_alloc" << endl;
         m_pTrend = gsl_vector_calloc(dim + 1);
         //cout << "CGeostat::allocate: Creating new variograms" << endl;
         m_pVarioCloud = new CVariogram;
         m_pVariogram = new CVariogram;

         cout << "CGeostat::allocate: Starting m_pOKSys gsl_matrix_alloc: samples() = " << samples() << endl;
         m_pOKSys = gsl_matrix_alloc(samples() + 1, samples() + 1);

         return true;
 }

 void CGeostat::destroy(void)
 {
         while (!m_coords.empty())
         {
                 delete m_coords.back();
                 m_coords.pop_back();
         }
         if (m_pData != NULL)
         {
                 delete m_pData;
                 m_pData = NULL;
         }
         if (m_pTrend != NULL)
         {
                 delete m_pTrend;
                 m_pTrend = NULL;
         }
         if (m_pVariogram != NULL)
         {
                 delete m_pVariogram;
                 m_pVariogram = NULL;
         }
         if (m_pVarioCloud != NULL)
         {
                 delete m_pVarioCloud;
                 m_pVarioCloud = NULL;
         }
         if (m_pOKSys != NULL)
         {
                 delete m_pOKSys;
                 m_pOKSys = NULL;
         }
         m_domain.clear();
 }

 bool CGeostat::getCoordinate(gsl_vector *c, size_t s) const
 {
         if (!isActive() || s >= samples())
                 return false;
         gsl_vector_memcpy(c, m_coords[s]);

         return true;
 }

 double CGeostat::getData(size_t s) const
 {
         if (!isActive() || s >= samples())
                 return 0.0;

         gsl_vector *c = gsl_vector_alloc(dimension());
         getCoordinate(c, s);
         double d = getResidual(s) + getTrend(c);
         gsl_vector_free(c);
         return d;
 }

 double CGeostat::getTrend(const gsl_vector *c) const
 {
         if (!isActive())
                 return 0.0;

         double trend = gsl_vector_get(m_pTrend, 0);
         for (size_t d = 0;d < dimension();d++)
                 trend += gsl_vector_get(m_pTrend, d + 1) * gsl_vector_get(c, d);

         return trend;
 }

 double CGeostat::getResidual(size_t s) const
 {
         if (!isActive() || s >= samples())
                 return 0.0;

         return gsl_vector_get(m_pData, s);
 }

 bool CGeostat::getDomain(gsl_vector *lower, gsl_vector *upper) const
 {
         if (!isActive())
                 return false;

         for (size_t d = 0;d < dimension();d++)
         {
                 gsl_vector_set(lower, d, m_domain[d].first);
                 gsl_vector_set(upper, d, m_domain[d].second);
         }

         return true;
 }

 bool CGeostat::setDomain(const gsl_vector *lower, const gsl_vector *upper)
 {
         if (!isActive())
                 return false;

         for (size_t d = 0;d < dimension();d++)
         {
                 m_domain[d].first = gsl_vector_get(lower, d);
                 m_domain[d].second = gsl_vector_get(upper, d);
         }

         return true;
 }

 bool CGeostat::setCoordinate(size_t s, const gsl_vector *c)
 {
         if (!isActive() || s >= samples())
                 return false;

         gsl_vector_memcpy(m_coords[s], c);

         return true;
 }

 bool CGeostat::setData(size_t s, double d)
 {
         if (!isActive() || s >= samples())
                 return false;

         gsl_vector_set(m_pData, s, d);
         return true;
 }

 std::vector<CVariogram::VarioItm> CGeostat::computeVariogramCloud(void) const
 {
         std::vector<CVariogram::VarioItm> vecVario;
         CVariogram::VarioItm itm;
         gsl_vector *c1, *c2;
         double r1, r2;

         if (!isActive())
                 return vecVario;

         c1 = gsl_vector_alloc(dimension());
         c2 = gsl_vector_alloc(dimension());

         for (size_t i = 0;i < samples() - 1;i++)
         {
                 #ifdef __DEBUG_KRIGING__
                   cout << "CGeostat::computeVariogramCloud: running sample " << i << endl;
                 #endif
                 getCoordinate(c1, i);
                 r1 = getResidual(i);

                 for (size_t j = i + 1;j < samples();j++)
                 {
                         getCoordinate(c2, j);
                         r2 = getResidual(j);

                         itm.distance = isoDist(c1, c2);
                         itm.dissimilarity = (r1 - r2) * (r1 - r2) / 2.0;

                         vecVario.push_back(itm);
                 }
         }

         gsl_vector_free(c1);
         gsl_vector_free(c2);

         return vecVario;
 }

 std::vector<CVariogram::VarioItm> CGeostat::computeExperimentalVariogram(const std::vector<CVariogram::VarioItm> &vcloud, double step) const
 {
         std::vector<CVariogram::VarioItm> vmodel;
         std::map<int, CVariogram::VarioItm> variomap;
         CVariogram::VarioItm itm;

         for (size_t i = 0;i < vcloud.size();i++)
         {
                 int key = static_cast<int>(vcloud[i].distance / step);
                 if (variomap.find(key) == variomap.end())
                 {
                         variomap[key].distance = 1.0;
                         variomap[key].dissimilarity = vcloud[i].dissimilarity;
                 }
                 else
                 {
                         variomap[key].distance += 1.0;
                         variomap[key].dissimilarity += vcloud[i].dissimilarity;
                 }
         }

         for (std::map<int, CVariogram::VarioItm>::const_iterator it = variomap.begin();it != variomap.end();++it)
         {
                 itm.distance = it->first * step;
                 itm.dissimilarity = it->second.dissimilarity / it->second.distance;
                 vmodel.push_back(itm);
         }
         return vmodel;
 }

 bool CGeostat::estimate(int model, double power, double step)
 {
         if (!isActive())
                 return false;

         std::vector<CVariogram::VarioItm> vcloud;
         double n, s, r;

         cout << "CGeostat::estimate: starting approximateTrendSurface" << endl;
         approximateTrendSurface();

         cout << "CGeostat::estimate: starting computeVariogramCloud" << endl;
         vcloud = computeVariogramCloud();
         cout << "CGeostat::estimate: starting setSample" << endl;
         m_pVarioCloud->setSample(vcloud);

         cout << "CGeostat::estimate: starting computeExperimentalVariogram" << endl;
         m_pVariogram->setSample(computeExperimentalVariogram(vcloud, step));
         n = 0;
         m_pVariogram->getSample(m_pVariogram->samples() / 2, r, s);
         cout << "CGeostat::estimate: starting estimateModel" << endl;
         m_pVariogram->estimateModel(model, n, s, r, power, m_pVariogram->maxDistance() / 2.0);

         cout << "CGeostat::estimate: starting precomputeKrigingSystem" << endl;
         precomputeKrigingSystem();

         return true;
 }

 bool CGeostat::getModelParameters(double &sill, double &range, double &nugget, double &power) const
 {
         if (!isActive())
                 return false;

         sill = m_pVariogram->sill();
         range = m_pVariogram->range();
         nugget = m_pVariogram->nugget();
         power = m_pVariogram->power();

         return false;
 }

 bool CGeostat::precomputeKrigingSystem(void)
 {
         gsl_vector *c1, *c2;
         gsl_matrix *tmp;

         c1 = gsl_vector_alloc(dimension());
         c2 = gsl_vector_alloc(dimension());
         tmp = gsl_matrix_alloc(samples() + 1, samples() + 1);

         #ifdef __DEBUG_KRIGING__
           cout << "CGeostat::precomputeKrigingSystem: starting 1st for loop" << endl;
         #endif
         for (size_t s1 = 0;s1 < samples();s1++)
         {
                 getCoordinate(c1, s1);

                 #ifdef __DEBUG_KRIGING__
                   cout << "CGeostat::precomputeKrigingSystem: starting 2nd for loop" << endl;
                 #endif
                 for (size_t s2 = s1;s2 < samples();s2++)
                 {
                         getCoordinate(c2, s2);

                         double dist;
                         dist = isoDist(c1, c2);
                         #ifdef __DEBUG_KRIGING__
                           cout << "CGeostat::precomputeKrigingSystem: starting gsl_matrix_set(tmp, s1 = " << s1 << ", s2 = " << s2 << ", m_pVariogram->getModelData(dist=" << dist << ") = )" << m_pVariogram->getModelData(dist) << endl;
                         #endif
                         gsl_matrix_set(tmp, s1, s2, m_pVariogram->getModelData(dist));
                         gsl_matrix_set(tmp, s2, s1, gsl_matrix_get(tmp, s1, s2));
                 }
         }

         #ifdef __DEBUG_KRIGING__
           cout << "CGeostat::precomputeKrigingSystem: starting 3rd for loop" << endl;
         #endif
         for (size_t s = 0;s < samples();s++)
         {
                 gsl_matrix_set(tmp, s, samples(), 1.0);
                 gsl_matrix_set(tmp, samples(), s, 1.0);
         }
         gsl_matrix_set(tmp, samples(), samples(), 0.0);

         #ifdef __DEBUG_KRIGING__
           cout << "CGeostat::precomputeKrigingSystem: starting matrix_inverse" << endl;
         #endif
         MatrixEx::matrix_inverse(m_pOKSys, tmp);

         gsl_vector_free(c1);
         gsl_vector_free(c2);
         gsl_matrix_free(tmp);

         return true;
 }

 double CGeostat::maxDist() const
 {
         double dist =0.0;

         for (size_t d = 0;d < dimension();d++)
                 dist += pow(m_domain[d].second - m_domain[d].first, 2.0);
         return sqrt(dist);
 }

 bool CGeostat::isDomain(const gsl_vector *c) const
 {
         for (size_t d = 0;d < dimension();d++)
         {
                 double ct = gsl_vector_get(c, d);
                 if (ct < m_domain[d].first || ct > m_domain[d].second)
                         return false;
         }
         return true;
 }

 bool CGeostat::getWeightVector(gsl_vector *weight, const gsl_vector *c) const
 {
         if (!isActive())
                 return false;

         gsl_vector *spos, *vvario;

         spos = gsl_vector_alloc(dimension());
         vvario = gsl_vector_alloc(samples() + 1);

         for (size_t s = 0;s < samples();s++)
         {
                 getCoordinate(spos, s);
                 double dist = isoDist(c, spos);

                 gsl_vector_set(vvario, s, m_pVariogram->getModelData(dist));
         }
         gsl_vector_set(vvario, samples(), 1.0);

         gsl_blas_dgemv(CblasNoTrans, 1.0, m_pOKSys, vvario, 0.0, weight);


         gsl_vector_free(spos);
         gsl_vector_free(vvario);

         return true;
 }

 bool CGeostat::getPredictData(double &pred, double &var, const gsl_vector *c) const
 {
         if (!isActive())
                 return false;

         gsl_vector *spos;
         gsl_vector *vvario, *weight;
         spos = gsl_vector_alloc(dimension());
         vvario = gsl_vector_alloc(samples() + 1);
         weight = gsl_vector_alloc(samples() + 1);

         for (size_t s = 0;s < samples();s++)
         {
                 getCoordinate(spos, s);
                 double dist = isoDist(c, spos);

                 gsl_vector_set(vvario, s, m_pVariogram->getModelData(dist));
         }
         gsl_vector_set(vvario, samples(), 1.0);

         gsl_blas_dgemv(CblasNoTrans, 1.0, m_pOKSys, vvario, 0.0, weight);

         pred = getTrend(c);
         for (size_t s = 0;s < samples();s++)
                 pred += gsl_vector_get(weight, s) * getResidual(s);

         gsl_vector_set(vvario, samples(), 0.0);
         gsl_blas_ddot(weight, vvario, &var);

         gsl_vector_free(spos);
         gsl_vector_free(vvario);
         gsl_vector_free(weight);

         return true;
 }

 bool CGeostat::getPredictData(double &pred, const gsl_vector *c, const gsl_vector *weight) const
 {
         if (!isActive())
                 return false;

         pred = getTrend(c);
         for (size_t s = 0;s < samples();s++)
                 pred += gsl_vector_get(weight, s) * getResidual(s);

         return true;
 }

 bool CGeostat::approximateTrendSurface(void)
 {
         gsl_matrix *mPh, *mPhI;
         gsl_vector *c;

         mPh = gsl_matrix_alloc(dimension() + 1, samples());
         mPhI = gsl_matrix_alloc(samples(), dimension() + 1);
         c = gsl_vector_alloc(dimension());

         for (size_t s = 0;s < samples();s++)
         {
                 getCoordinate(c, s);
                 gsl_matrix_set(mPh, 0, s, 1.0);
                 for (size_t d = 0;d < dimension();d++){
                         gsl_matrix_set(mPh, d + 1, s, gsl_vector_get(c, d));
                         #ifdef __DEBUG_KRIGING__
                           cout << "CGeostat::approximateTrendSurface: mPh(" << d+1 << ", " << s << ") set to " << gsl_vector_get(c, d) << endl;
                         #endif
                 }
         }
         #ifdef __DEBUG_KRIGING__
           cout << "CGeostat::approximateTrendSurface: mPh = " << mPh << endl;
         #endif
         MatrixEx::matrix_pseudo_inverse(mPhI, mPh);
         gsl_blas_dgemv(CblasTrans, 1.0, mPhI, m_pData, 0.0, m_pTrend);

         for (size_t s = 0;s < samples();s++)
         {
                 getCoordinate(c, s);
                 *gsl_vector_ptr(m_pData, s) -= getTrend(c);
         }

         gsl_vector_free(c);
         gsl_matrix_free(mPh);
         gsl_matrix_free(mPhI);

         return true;
 }

 bool CGeostat::removeRedundantSample(size_t smpl)
 {
         if (!isActive() || smpl >= samples())
                 return false;

         std::vector<gsl_vector *> newCoord;
         gsl_vector *pcoord, *newData;

         newData = gsl_vector_alloc(samples() - 1);

         size_t sp1 =0;
         for (size_t s = 0;s < samples();s++)
         {
                 if (s == smpl)
                         continue;
                 gsl_vector_set(newData, sp1, gsl_vector_get(m_pData, s));

                 pcoord = gsl_vector_alloc(dimension());
                 getCoordinate(pcoord, s);
                 newCoord.push_back(pcoord);

                 sp1++;
         }

         while (!m_coords.empty())
         {
                 delete m_coords.back();
                 m_coords.pop_back();
         }
         delete m_pData;

         m_pData = newData;
         m_coords = newCoord;

         precomputeKrigingSystem();

         return true;
 }

 bool CGeostat::getPredictionErrorExcluded(double &pred, double &var, size_t s) const
 {
         if (!isActive() || s >= samples())
                 return false;

         gsl_vector *c, *c1, *c2;
         gsl_vector *vvario, *weight;
         gsl_matrix *mGamma, *mGammaI;
         size_t sp1, sp2;

         c = gsl_vector_alloc(dimension());
         c1 = gsl_vector_alloc(dimension());
         c2 = gsl_vector_alloc(dimension());

         vvario = gsl_vector_alloc(samples());
         weight = gsl_vector_alloc(samples());
         mGamma = gsl_matrix_alloc(samples(), samples());
         mGammaI = gsl_matrix_alloc(samples(), samples());

         getCoordinate(c, s);

         sp1 = 0;
         for (size_t s1 = 0;s1 < samples();s1++)
         {
                 if (s1 == s)
                         continue;

                 getCoordinate(c1, s1);

                 sp2 = 0;
                 for (size_t s2 = 0;s2 < samples();s2++)
                 {
                         if (s2 == s)
                                 continue;

                         getCoordinate(c2, s2);
                         double dist = isoDist(c1, c2);

                         gsl_matrix_set(mGamma, sp1, sp2, m_pVariogram->getModelData(dist));

                         sp2++;
                 }
                 sp1++;
         }

         for (size_t s = 0;s < samples() - 1;s++)
         {
                 gsl_matrix_set(mGamma, s, samples() - 1, 1.0);
                 gsl_matrix_set(mGamma, samples() - 1, s, 1.0);
         }
         gsl_matrix_set(mGamma, samples() - 1, samples() - 1, 0.0);
         MatrixEx::matrix_inverse(mGammaI, mGamma);


         sp1 = 0;
         for (size_t s = 0;s < samples();s++)
         {
//                 if (s == s)
                         continue;
                 getCoordinate(c1, s);
                 double dist = isoDist(c, c1);
                 gsl_vector_set(vvario, sp1++, m_pVariogram->getModelData(dist));
         }
         gsl_vector_set(vvario, samples() - 1, 1.0);
         gsl_blas_dgemv(CblasNoTrans, 1.0, mGamma, vvario, 0.0, weight);

         pred = getTrend(c);
         sp1 = 0;
         for (size_t s = 0;s < samples();s++)
         {
//                 if (s == s)
                         continue;
                 pred += gsl_vector_get(weight, sp1) * getResidual(s);
                 sp1++;
         }
         gsl_vector_set(vvario, samples() - 1, 0.0);
         gsl_blas_ddot(weight, vvario, &var);


         gsl_vector_free(c);
         gsl_vector_free(c1);
         gsl_vector_free(c2);

         gsl_matrix_free(mGamma);
         gsl_matrix_free(mGammaI);
         gsl_vector_free(vvario);
         gsl_vector_free(weight);

         return true;
 }

 double CGeostat::selectEliminatee(size_t &candidate) const
 {
         if (!isActive())
                 return 0;

         gsl_vector *c, *c1, *c2;
         gsl_vector *vvario, *weight;
         gsl_matrix *mGamma, *mGammaR, *mGammaI;
         size_t sp1, sp2;
         double var, varmin;

         c = gsl_vector_alloc(dimension());
         c1 = gsl_vector_alloc(dimension());
         c2 = gsl_vector_alloc(dimension());

         mGamma = gsl_matrix_alloc(samples(), samples());
         mGammaR = gsl_matrix_alloc(samples(), samples());
         mGammaI = gsl_matrix_alloc(samples(), samples());
         vvario = gsl_vector_alloc(samples());
         weight = gsl_vector_alloc(samples());

         for (size_t s1 = 0;s1 < samples();s1++)
         {
                 getCoordinate(c1, s1);
                 for (size_t s2 = s1;s2 < samples();s2++)
                 {
                         getCoordinate(c2, s2);
                         double dist = isoDist(c1, c2);
                         gsl_matrix_set(mGamma, s1, s2, m_pVariogram->getModelData(dist));
                         gsl_matrix_set(mGamma, s2, s1, gsl_matrix_get(mGamma, s1, s2));
                 }
         }
         for (size_t s = 0;s < samples() - 1;s++)
         {
                 gsl_matrix_set(mGammaR, s, samples() - 1, 1.0);
                 gsl_matrix_set(mGammaR, samples() - 1, s, 1.0);
         }
         gsl_matrix_set(mGammaR, samples() - 1, samples() - 1, 0.0);

         varmin = 1.0e6;
         for (size_t smpl = 0;smpl < samples();smpl++)
         {
                 getCoordinate(c, smpl);

                 sp1 = 0;
                 for (size_t s1 = 0;s1 < samples();s1++)
                 {
                         if (s1 != smpl)
                         {
                                 sp2 = 0;
                                 for (size_t s2 = 0;s2 < samples();s2++)
                                 {
                                         if (s2 != smpl)
                                                 gsl_matrix_set(mGammaR, sp1, sp2++, gsl_matrix_get(mGamma, s1, s2));
                                 }
                                 sp1++;
                         }
                 }
                 MatrixEx::matrix_inverse(mGammaI, mGammaR);


                 sp1 = 0;
                 for (size_t s = 0;s < samples();s++)
                 {
                         if (s != smpl)
                         {
                                 getCoordinate(c1, s);
                                 double dist = isoDist(c, c1);
                                 gsl_vector_set(vvario, sp1++, m_pVariogram->getModelData(dist));
                         }
                 }
                 gsl_vector_set(vvario, samples() - 1, 1.0);
                 gsl_blas_dgemv(CblasNoTrans, 1.0, mGammaI, vvario, 0.0, weight);

                 gsl_vector_set(vvario, samples() - 1, 0.0);
                 gsl_blas_ddot(weight, vvario, &var);

                 if (var < varmin)
                 {
                         candidate = smpl;
                         varmin = var;
                 }
         }

         gsl_vector_free(c);
         gsl_vector_free(c1);
         gsl_vector_free(c2);

         gsl_matrix_free(mGamma);
         gsl_matrix_free(mGammaR);
         gsl_matrix_free(mGammaI);

         gsl_vector_free(vvario);
         gsl_vector_free(weight);


         return 1.0 - varmin / m_pVariogram->sill();
 }

 double CGeostat::evalModelValidity(int model, double power, double step)
 {
         if (!isActive())
                 return 0.0;

         std::vector<CVariogram*> vecCloud, vecModel;
         std::vector<CVariogram::VarioItm> vcloud;
         double n =0, s =0, r =0;

         approximateTrendSurface();

         vcloud = computeVariogramCloud();
         m_pVarioCloud = new CVariogram;
         m_pVarioCloud->setSample(vcloud);

         m_pVariogram = new CVariogram;
         m_pVariogram->setSample(computeExperimentalVariogram(vcloud, step));
         m_pVariogram->getSample(m_pVariogram->samples() / 2, r, s);

         m_pVariogram->estimateModel(model, n, s, r, power, m_pVariogram->maxDistance() / 2.0);


         gsl_vector *c, *c1, *c2;
         gsl_vector *vvario, *weight;
         gsl_matrix *mGamma, *mGammaR, *mGammaI;
         size_t sp1, sp2;
         double pred, errsum;

         c = gsl_vector_alloc(dimension());
         c1 = gsl_vector_alloc(dimension());
         c2 = gsl_vector_alloc(dimension());

         mGamma = gsl_matrix_alloc(samples(), samples());
         mGammaR = gsl_matrix_alloc(samples(), samples());
         mGammaI = gsl_matrix_alloc(samples(), samples());
         vvario = gsl_vector_alloc(samples());
         weight = gsl_vector_alloc(samples());

         for (size_t s1 = 0;s1 < samples();s1++)
         {
                 getCoordinate(c1, s1);
                 for (size_t s2 = s1;s2 < samples();s2++)
                 {
                         getCoordinate(c2, s2);
                         double dist = isoDist(c1, c2);
                         gsl_matrix_set(mGamma, s1, s2, m_pVariogram->getModelData(dist));
                         gsl_matrix_set(mGamma, s2, s1, gsl_matrix_get(mGamma, s1, s2));
                 }
         }
         for (size_t s = 0;s < samples() - 1;s++)
         {
                 gsl_matrix_set(mGammaR, s, samples() - 1, 1.0);
                 gsl_matrix_set(mGammaR, samples() - 1, s, 1.0);
         }
         gsl_matrix_set(mGammaR, samples() - 1, samples() - 1, 0.0);


         errsum = 0.0;
         for (size_t smpl = 0;smpl < samples();smpl++)
         {
                 sp1 = 0;
                 for (size_t s1 = 0;s1 < samples();s1++)
                 {
                         if (s1 != smpl)
                         {
                                 sp2 = 0;
                                 for (size_t s2 = 0;s2 < samples();s2++)
                                 {
                                         if (s2 != smpl)
                                         {
                                                 gsl_matrix_set(mGammaR, sp1, sp2++, gsl_matrix_get(mGamma, s1, s2));
                                         }
                                 }
                                 sp1++;
                         }
                 }
                 MatrixEx::matrix_inverse(mGammaI, mGammaR);

                 getCoordinate(c, smpl);
                 sp1 = 0;
                 for (size_t s = 0;s < samples();s++)
                 {
                         if (s != smpl)
                         {
                                 getCoordinate(c1, s);
                                 double dist = isoDist(c, c1);
                                 gsl_vector_set(vvario, sp1++, m_pVariogram->getModelData(dist));
                         }
                 }
                 gsl_vector_set(vvario, samples() - 1, 1.0);
                 gsl_blas_dgemv(CblasNoTrans, 1.0, mGammaI, vvario, 0.0, weight);

                 sp1 = 0;
                 pred = 0.0;
                 for (size_t s = 0;s < samples();s++)
                 {
                         if (s != smpl)
                                 pred += gsl_vector_get(weight, sp1++) * getResidual(s);
                 }
                 errsum += pow(pred - getResidual(smpl), 2.0);
         }

         gsl_vector_free(c);
         gsl_vector_free(c1);
         gsl_vector_free(c2);

         gsl_matrix_free(mGamma);
         gsl_matrix_free(mGammaR);
         gsl_matrix_free(mGammaI);

         gsl_vector_free(vvario);
         gsl_vector_free(weight);

         return errsum;
 }
