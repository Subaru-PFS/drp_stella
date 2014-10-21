#ifndef VARIOGRAM_HEADER_FILE_
#define VARIOGRAM_HEADER_FILE_

#include <vector>
#include <iostream>

using namespace std;

class CVariogram
{
//
// member enumeration
public:
        enum VariogramModel
        {
                VARIO_NONE,
                VARIO_SPH,
                VARIO_STB,
                VARIO_NUM
        };

//
// member classes
public:
        struct VarioItm
        {
                double distance;
                double dissimilarity;
        };

//
// variables
private:
        int m_model;
        size_t m_samples;
        double *m_pDistance;
        double *m_pVariogram;

        //
        // papameters of theoretical variogram
        //
        double m_nugget;
        double m_sill;
        double m_range;
        double m_power;

//
// constructor / destructor
//
public:
        // default constructor
        CVariogram(void);
        // destructor
        virtual ~CVariogram(void);

//
// attributes
public:
        bool isActive(void) const
                {
                        return m_pDistance != NULL && m_pVariogram;
                }
        bool isEstimated(void) const
                {
                        return m_model != VARIO_NONE;
                }
        size_t samples(void) const
                {
                        return m_samples;
                }
        double maxDistance(void) const
                {
                        return m_pDistance[m_samples - 1];
                }
        double minDistance(void) const
                {
                        return m_pDistance[0];
                }
        double nugget(void) const
                {
                        return m_nugget;
                }
        double sill(void) const
                {
                        return m_sill;
                }
        double range(void) const
                {
                        return m_range;
                }
        double power(void) const
                {
                        return m_power;
                }
        int modelType(void) const
                {
                        return m_model;
                }
        // get number of samples whose distance is less than given value
        size_t countLessDist(double cap) const;

//
// data reader
public:
        // get sample data (distance & dissimilarity)
        bool getSample(size_t smpl, double &dist, double &vario) const;
        // get theoretical dissimilarity
        double getModelData(double dist) const;
        // get covariance corresponding to theoretical dissimilarity
        double getModelCovariance(double dist) const;

//
// data writer
public:
        // set sample data (distance & dissimilarity)
        bool setSample(const std::vector<CVariogram::VarioItm> &vecSample);
        // set parameters of theoretical variogram (nugget, sill, range, power, and so on)
        bool setModel(int model, double nugget, double sill, double range, double power, double step, double maxdist);

//
// manipulator
protected:
        // sort container for dissimilarity by distance
        bool sortByDistance(void);
        bool bubbleSortByDistance(void);
public:
        // estimate theoretical variogram by non-linear least square fitting
        int estimateModel(int model, double &nugget, double &sill, double &range, double power, double maxdist =1.0e6);

//
// memory manager
private:
        // memory allocation
        bool allocate(size_t smpl);
        // memory destruction
        void destroy(void);
};

#endif  //VARIOGRAM_HEADER_FILE_
