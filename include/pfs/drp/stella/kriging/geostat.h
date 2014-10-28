#ifndef GEOSTATISTICS_CLASS_HEADER_FILE_
#define GEOSTATISTICS_CLASS_HEADER_FILE_

#include "variogram.h"

#include <iostream>
#include <vector>
#include <list>
#include <utility>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

using namespace std;

class CGeostat
{

//
// variables
private:
        std::vector<gsl_vector *> m_coords;
        gsl_vector *m_pData;
        gsl_vector *m_pTrend;
        std::vector< std::pair<double, double> > m_domain;
        CVariogram* m_pVarioCloud;
        CVariogram* m_pVariogram;
        gsl_matrix* m_pOKSys;

//
// constructors/destructor
public:
        // constructor
        CGeostat(void);
        // destructor
        virtual ~CGeostat(void);
                // initializer
        bool initialize(size_t smpl, size_t dim);

//
// attributes (const)
public:
        bool isActive(void) const
                {
                        return m_pData != NULL;
                }
        size_t samples(void) const
                {
                        return m_pData->size;
                }

        size_t dimension(void) const
                {
                        return m_domain.size();
                }
        double maxDist(void) const;
        // domain validation
        bool isDomain(const gsl_vector *c) const;

//
// data readers
//
public:
        // get coordinate of sample s
        bool getCoordinate(gsl_vector *c, size_t s) const;
        // get sample s
        double getData(size_t s) const;
        // get parameters of estimated theoretical variogram
        bool getModelParameters(double &sill, double &range, double &nugget, double &power) const;
        // get optimum weights for target parameter c
        bool getWeightVector(gsl_vector *w, const gsl_vector *c) const;
        // get prediction value & estimate variance
        bool getPredictData(double &pred, double &var, const gsl_vector *c) const;
        // get prediction value with weight
        bool getPredictData(double &pred, const gsl_vector *coord, const gsl_vector *w) const;
        // get control space domain
        bool getDomain(gsl_vector *lower, gsl_vector *upper) const;
        // get trend component
        double getTrend(const gsl_vector *c) const;
        // get residual component at sample location
        double getResidual(size_t s) const;

//
// data writers
public:
        // set control space domain
        bool setDomain(const gsl_vector *lower, const gsl_vector *upper);
        // set coordinate of sample s
        bool setCoordinate(size_t s, const gsl_vector *c);
        // set sample data
        bool setData(size_t s, double d);

//
// manipulators
public:
        // esimate trend surface and theoretical variogram
        bool estimate(int model, double power, double step);
        // remove redundant sample s while preserving estimated trend surface and theoretical variogram
        bool removeRedundantSample(size_t s);

//
// for cross validation
public:
        // validate the estimated theoretical variogram
        double evalModelValidity(int model, double power, double step);
        // leave-one-out cross validation
        bool getPredictionErrorExcluded(double &pred, double &var, size_t smpl) const;
        // greedy search of eliminatee
        double selectEliminatee(size_t &candidate) const;

//
// calculators
protected:
        // trend surface approximation
        bool approximateTrendSurface(void);
        // compute variogram cloud
        std::vector<CVariogram::VarioItm> computeVariogramCloud(void) const;
        // compute experimental variogram
        std::vector<CVariogram::VarioItm> computeExperimentalVariogram(const std::vector<CVariogram::VarioItm> &vcloud, double step) const;
        // precompute kriging system
        bool precomputeKrigingSystem(void);

//
// memory managers
private:
        // memory allocation
        bool allocate(size_t smpl, size_t dim);
        // memory destruction
        void destroy(void);
};

#endif  //GEOSTATISTICS_CLASS_HEADER_FILE_
