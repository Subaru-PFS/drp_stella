#include <iostream>
#include <fstream>

#include "geostat.h"
#include "/home/azuri/entwicklung/stella/ses-pipeline/c/cfits/src/CFits.h"

using namespace std;

int main(void)
{
        // dimension of control space
        const size_t dim_cspace = 2;
        // number of samples
//        const size_t num_samples = 16;

        double i_a, i_b;
        int I_ClusterSize_X, I_ClusterSize_Y;
        int I_NClusters;
        CFits *P_MyCFits = new CFits();
        P_MyCFits->SetFileName(CString("/home/azuri/spectra/SEDIFU/SEDM-deep-sim-flat-2012-05-14_ScatterOnly.fits"));
        P_MyCFits->SetDatabaseFileName("/home/azuri/spectra/SEDIFU/database/apSEDM-deep-sim-flat-2012-05-14_Flat-back");
        P_MyCFits->ReadArray();
        cout << "main: P_MyCFits->GetPixArray(0,1) = " << (P_MyCFits->GetPixArray())(0,1) << endl;
        P_MyCFits->ReadDatabaseEntry();
        P_MyCFits->CalcTraceFunctions();
        P_MyCFits->Set_ApertureDataToZero();
        P_MyCFits->SetFileName(CString("/home/azuri/spectra/SEDIFU/SEDM-deep-sim-flat-2012-05-14_ScatterOnly_apZero.fits"));
        P_MyCFits->WriteArray();


        Array<double, 2> D_A2_ClusteredBig(2,2);
        Array<double, 2> D_A2_ClusteredSmall(2,2);
        Array<double, 2> D_A2_ToCluster(P_MyCFits->GetNRows(), P_MyCFits->GetNCols());
        D_A2_ToCluster = P_MyCFits->GetPixArray();
        P_MyCFits->WriteFits(&D_A2_ToCluster, CString("/home/azuri/spectra/SEDIFU/SEDM-deep-sim-flat-2012-05-14_ScatterOnly_apZero2.fits"));
        cout << "main: starting FindNonZeroClusters big" << endl;
        I_ClusterSize_X = 14;
        I_ClusterSize_Y = 30;
        if (!P_MyCFits->FindNonZeroClusters(D_A2_ToCluster, I_ClusterSize_X, I_ClusterSize_Y, 0, D_A2_ClusteredBig)){
          cout << "main: No big clusters found" << endl;
          return 0;
        }
        cout << "main: D_A2_ClusteredBig.rows() = " << D_A2_ClusteredBig.rows() << ", D_A2_ClusteredBig.cols() = " << D_A2_ClusteredBig.cols() << endl;
        P_MyCFits->WriteFits(&D_A2_ToCluster, CString("/home/azuri/spectra/SEDIFU/SEDM-deep-sim-flat-2012-05-14_ScatterOnly_apZero_cluster_big.fits"));
        P_MyCFits->WriteFits(&D_A2_ClusteredBig, CString("/home/azuri/spectra/SEDIFU/SEDM-deep-sim-flat-2012-05-14_ScatterOnly_apZero_clustered_big.fits"));
        cout << "main: starting FindNonZeroClusters small" << endl;
        Array<int, 2> I_A2_ClusterCoords(2,2);
        Array<int, 2> I_A2_WhereCluster(P_MyCFits->GetNRows(), P_MyCFits->GetNCols());
        Array<double, 2> D_A2_WhereClusterFAbs(P_MyCFits->GetNRows(), P_MyCFits->GetNCols());
        for (int i_cluster_run=0; i_cluster_run < 5; i_cluster_run++){
          if (i_cluster_run == 0){
            I_ClusterSize_X = 12;
            I_ClusterSize_Y = 28;
          }
          else if (i_cluster_run == 1){
            I_ClusterSize_X = 10;
            I_ClusterSize_Y = 28;
          }
          else if (i_cluster_run == 2){
            I_ClusterSize_X = 8;
            I_ClusterSize_Y = 28;
          }
          else if (i_cluster_run == 3){
            I_ClusterSize_X = 5;
            I_ClusterSize_Y = 28;
          }
          else if (i_cluster_run == 4){
            I_ClusterSize_X = 5;
            I_ClusterSize_Y = 10;
          }
          cout << "main: ClusterSize = " << I_ClusterSize_X << " x " << I_ClusterSize_Y << endl;
          if (!P_MyCFits->FindNonZeroClusters(D_A2_ToCluster, I_ClusterSize_X, I_ClusterSize_Y, 0, D_A2_ClusteredSmall)){
            cout << "main: No small clusters found" << endl;
            return 0;
          }
          cout << "main: D_A2_ClusteredSmall.rows() = " << D_A2_ClusteredSmall.rows() << ", D_A2_ClusteredSmall.cols() = " << D_A2_ClusteredSmall.cols() << endl;

          P_MyCFits->WriteFits(&D_A2_ToCluster, CString("/home/azuri/spectra/SEDIFU/SEDM-deep-sim-flat-2012-05-14_ScatterOnly_apZero_cluster_small.fits"));
          P_MyCFits->WriteFits(&D_A2_ClusteredSmall, CString("/home/azuri/spectra/SEDIFU/SEDM-deep-sim-flat-2012-05-14_ScatterOnly_apZero_clustered_small.fits"));
          cout << "main: starting fabs" << endl;
          D_A2_WhereClusterFAbs = fabs(D_A2_ClusteredSmall);
          P_MyCFits->WriteFits(&D_A2_WhereClusterFAbs, CString("/home/azuri/spectra/SEDIFU/SEDM-deep-sim-flat-2012-05-14_ScatterOnly_apZero_clustered_small_fabs.fits"));
          cout << "main: starting where" << endl;
          I_A2_WhereCluster = where(D_A2_WhereClusterFAbs >= 0.000001, 1, 0);

          //P_MyCFits->WriteFits(&I_A2_WhereCluster, CString("/home/azuri/spectra/SEDIFU/SEDM-deep-sim-flat-2012-05-14_ScatterOnly_apZero_clustered_small_fabs_where.fits"));
          cout << "main: sum(I_A2_WhereCluster) = " << sum(I_A2_WhereCluster) << endl;
          cout << "main: starting GetIndex" << endl;
          P_MyCFits->GetIndex(I_A2_WhereCluster, I_NClusters, I_A2_ClusterCoords);
          cout << "main: I_A2_ClusterCoords.rows() = " << I_A2_ClusterCoords.rows() << ", I_A2_ClusterCoords.cols() = " << I_A2_ClusterCoords.cols() << endl;
          cout << "main: I_A2_ClusterCoords = " << I_A2_ClusterCoords << endl;
          cout << "main: moving clusters to big array" << endl;
          for (int i_cl=0; i_cl < I_NClusters; i_cl++){
            D_A2_ClusteredBig(I_A2_ClusterCoords(i_cl, 0), I_A2_ClusterCoords(i_cl, 1)) = D_A2_ClusteredSmall(I_A2_ClusterCoords(i_cl, 0), I_A2_ClusterCoords(i_cl, 1));
          }
        }
        P_MyCFits->WriteFits(&D_A2_ClusteredBig, CString("/home/azuri/spectra/SEDIFU/SEDM-deep-sim-flat-2012-05-14_ScatterOnly_apZero_clustered_big_new.fits"));

        cout << "main: starting FindMeanValuesOfRectangles(98, 98)" << endl;
        if (!P_MyCFits->FindMeanValuesOfRectangles(98,98))
          return false;
        Array<int, 2> I_A2_WhereZero(P_MyCFits->GetNRows(), P_MyCFits->GetNCols());
        cout << "main: starting where2" << endl;
        I_A2_WhereZero = where(fabs(P_MyCFits->GetPixArray()) > 0.000001, 1, 0);
        //cout << "main: I_A2_WhereZero = " << I_A2_WhereZero << endl;
        cout << "main: I_A2_WhereZero.size() = " << I_A2_WhereZero.size() << endl;
        cout << "main: I_A2_WhereZero.rows() = " << I_A2_WhereZero.rows() << endl;
        cout << "main: I_A2_WhereZero.cols() = " << I_A2_WhereZero.cols() << endl;
        cout << "main: sum(I_A2_WhereZero) = " << sum(I_A2_WhereZero) << endl;
        //return 0;

        Array<int, 2> I_A2_IndZero(2,2);

        int I_NGood;
        cout << "main: starting GetIndex" << endl;
        P_MyCFits->GetIndex(I_A2_WhereZero, I_NGood, I_A2_IndZero);
        cout << "main: I_NGood = " << I_NGood << endl;
        cout << "main: I_A2_IndZero = " << I_A2_IndZero << endl;
        const size_t num_samples = I_NGood;

        // kriging predictor
        CGeostat krig;

        // initialize kriging predictor
        cout << "main: starting kriging initialize: num_samples = " << num_samples << ", dim_cspace = " << dim_cspace << endl;
        krig.initialize(num_samples, dim_cspace);


        //
        // define a domain of control space
        //
        cout << "main: starting gsl_vector_alloc" << endl;
        gsl_vector *lower = gsl_vector_alloc(dim_cspace);
        gsl_vector *upper = gsl_vector_alloc(dim_cspace);
        //
        //control parameter c1 : 0 <= c1 <= 4.0
        gsl_vector_set(lower, 0, 0.0);
        gsl_vector_set(upper, 0, P_MyCFits->GetNRows()-1);
        //
        // control parameter c2 : 0 <= c2 <= 5.0
        gsl_vector_set(lower, 1, 0.0);
        gsl_vector_set(upper, 1, P_MyCFits->GetNCols()-1);
        //
        // set domain
        cout << "main: starting kriging setDomain" << endl;
        krig.setDomain(lower, upper);
        //
        gsl_vector_free(lower);
        gsl_vector_free(upper);

        double tmp, pred, var;

        //
        // sample data
        //
        cout << "main: starting gsl_vector_alloc *c" << endl;
        gsl_vector *c = gsl_vector_alloc(dim_cspace);

        Array<double, 2> D_A2_ScatterFit(P_MyCFits->GetNRows(), P_MyCFits->GetNCols());
        cout << "main: D_A2_ScatterFit.rows() = " << D_A2_ScatterFit.rows() << endl;
        cout << "main: D_A2_ScatterFit.cols() = " << D_A2_ScatterFit.cols() << endl;
        cout << "main: starting kriging setData for loop" << endl;
        for (int i = 0; i < num_samples; i++){
          i_a = static_cast<double>(I_A2_IndZero(i,0));
          i_b = static_cast<double>(I_A2_IndZero(i,1));
          cout << "i_a = " << i_a << ", i_b = " << i_b << endl;
          gsl_vector_set(c, 0, i_a);
          gsl_vector_set(c, 1, i_b);
          //
          // coordinate of i-th sample
          krig.setCoordinate(i, c);
          krig.setData(i, (P_MyCFits->GetPixArray())(I_A2_IndZero(i,0), I_A2_IndZero(i,1)));
        }
/**
        i_a = 0.;
        i_b = 1.;
        cout << "i_a = " << i_a << ", i_b = " << i_b << endl;
        gsl_vector_set(c, 0, i_a);
        gsl_vector_set(c, 1, i_b);
        //
        // coordinate of i-th sample
        krig.setCoordinate(0, c);

        i_a = 0.;
        i_b = 2.;
        cout << "i_a = " << i_a << ", i_b = " << i_b << endl;
        gsl_vector_set(c, 0, i_a);
        gsl_vector_set(c, 1, i_b);
        //
        // coordinate of i-th sample
        krig.setCoordinate(1, c);

        i_a = 0.;
        i_b = 6.;
        cout << "i_a = " << i_a << ", i_b = " << i_b << endl;
        gsl_vector_set(c, 0, i_a);
        gsl_vector_set(c, 1, i_b);
        //
        // coordinate of i-th sample
        krig.setCoordinate(2, c);

        i_a = 0.;
        i_b = 7.;
        cout << "i_a = " << i_a << ", i_b = " << i_b << endl;
        gsl_vector_set(c, 0, i_a);
        gsl_vector_set(c, 1, i_b);
        //
        // coordinate of i-th sample
        krig.setCoordinate(3, c);

        i_a = 1.;
        i_b = 1.;
        cout << "i_a = " << i_a << ", i_b = " << i_b << endl;
        gsl_vector_set(c, 0, i_a);
        gsl_vector_set(c, 1, i_b);
        //
        // coordinate of i-th sample
        krig.setCoordinate(4, c);

        i_a = 1.;
        i_b = 2.;
        cout << "i_a = " << i_a << ", i_b = " << i_b << endl;
        gsl_vector_set(c, 0, i_a);
        gsl_vector_set(c, 1, i_b);
        //
        // coordinate of i-th sample
        krig.setCoordinate(5, c);

        i_a = 5.;
        i_b = 1.;
        cout << "i_a = " << i_a << ", i_b = " << i_b << endl;
        gsl_vector_set(c, 0, i_a);
        gsl_vector_set(c, 1, i_b);
        //
        // coordinate of i-th sample
        krig.setCoordinate(6, c);

        i_a = 5.;
        i_b = 10.;
        cout << "i_a = " << i_a << ", i_b = " << i_b << endl;
        gsl_vector_set(c, 0, i_a);
        gsl_vector_set(c, 1, i_b);
        //
        // coordinate of i-th sample
        krig.setCoordinate(7, c);

        i_a = 7.;
        i_b = 4.;
        cout << "i_a = " << i_a << ", i_b = " << i_b << endl;
        gsl_vector_set(c, 0, i_a);
        gsl_vector_set(c, 1, i_b);
        //
        // coordinate of i-th sample
        krig.setCoordinate(8, c);

        i_a = 7.;
        i_b = 10.;
        cout << "i_a = " << i_a << ", i_b = " << i_b << endl;
        gsl_vector_set(c, 0, i_a);
        gsl_vector_set(c, 1, i_b);
        //
        // coordinate of i-th sample
        krig.setCoordinate(9, c);

        i_a = 10.;
        i_b = 1.;
        cout << "i_a = " << i_a << ", i_b = " << i_b << endl;
        gsl_vector_set(c, 0, i_a);
        gsl_vector_set(c, 1, i_b);
        //
        // coordinate of i-th sample
        krig.setCoordinate(10, c);

        i_a = 10.;
        i_b = 10.;
        cout << "i_a = " << i_a << ", i_b = " << i_b << endl;
        gsl_vector_set(c, 0, i_a);
        gsl_vector_set(c, 1, i_b);
        //
        // coordinate of i-th sample
        krig.setCoordinate(11, c);

        i_a = 11.;
        i_b = 1.;
        cout << "i_a = " << i_a << ", i_b = " << i_b << endl;
        gsl_vector_set(c, 0, i_a);
        gsl_vector_set(c, 1, i_b);
        //
        // coordinate of i-th sample
        krig.setCoordinate(12, c);

        i_a = 11.;
        i_b = 7.;
        cout << "i_a = " << i_a << ", i_b = " << i_b << endl;
        gsl_vector_set(c, 0, i_a);
        gsl_vector_set(c, 1, i_b);
        //
        // coordinate of i-th sample
        krig.setCoordinate(13, c);

        i_a = 11.;
        i_b = 10.;
        cout << "i_a = " << i_a << ", i_b = " << i_b << endl;
        gsl_vector_set(c, 0, i_a);
        gsl_vector_set(c, 1, i_b);
        //
        // coordinate of i-th sample
        krig.setCoordinate(14, c);

        i_a = 12.;
        i_b = 3.;
        cout << "i_a = " << i_a << ", i_b = " << i_b << endl;
        gsl_vector_set(c, 0, i_a);
        gsl_vector_set(c, 1, i_b);
        //
        // coordinate of i-th sample
        krig.setCoordinate(15, c);

        double tmp, pred, var;
        for (size_t i = 0;i < num_samples;i++)
        {
                //
                // regular sampling point...
//                i_a = static_cast<double>(i / 4);
//                i_b = static_cast<double>(i % 4);
//                cout << "i = " << i << ": i_a = " << i_a << ", i_b = " << i_b << endl;
//                gsl_vector_set(c, 0, i_a);
//                gsl_vector_set(c, 1, i_b);
                //
                // coordinate of i-th sample
//                krig.setCoordinate(i, c);

                //
                // randomly generated sample data
                tmp = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
                tmp = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
                cout << "i = " << i << ": tmp = " << tmp << ", RAND_MAX = " << RAND_MAX << endl;

                //
                // data of i-th sample
                krig.setData(i, tmp);
        }
**/

        //
        // precomputation of statistical analysis
        //
        //  using Spherical variogram model
        //  power coefficient = 0 (only use for stable variogram model)
        //  step width (h_step = 0.1)
        //krig.estimate(CVariogram::VARIO_SPH, 0, 0.1);
        cout << "main: starting kriging estimate" << endl;
        krig.estimate(CVariogram::VARIO_SPH, 0, 1.);


        //
        // output file stream
        std::ofstream ofkrig;
        ofkrig.open("krig.csv");

        //
        // prediction: 100x100 sample point
        //
        cout << "main: starting kriging prediction" << endl;
        cout << "main: P_MyCFits->GetNRows() = " << P_MyCFits->GetNRows() << endl;
        cout << "main: P_MyCFits->GetNCols() = " << P_MyCFits->GetNCols() << endl;
        for (size_t i = 0;i < P_MyCFits->GetNRows();i++)
        {
          cout << "main: row i = " << i << endl;
                // target parameter c1
                gsl_vector_set(c, 0, i);
                for (size_t j = 0;j < P_MyCFits->GetNCols();j++)
                {
                        // target parameter c2
                        gsl_vector_set(c, 1, j);

                        // predictive value and estimate variance at parameter c
                        krig.getPredictData(pred, var, c);
                        ofkrig << pred << ", ";
                        D_A2_ScatterFit(static_cast<int>(i), static_cast<int>(j)) = pred;
                }
                ofkrig << std::endl;
        }
        ofkrig.close();
        cout << "main: D_A2_ScatterFit.rows() = " << D_A2_ScatterFit.rows() << endl;
        cout << "main: D_A2_ScatterFit.cols() = " << D_A2_ScatterFit.cols() << endl;

        gsl_vector_free(c);

        P_MyCFits->WriteFits(&D_A2_ScatterFit, CString("/home/azuri/spectra/SEDIFU/SEDM-deep-sim-flat-2012-05-14_ScatterFit_new.fits"));
        P_MyCFits->SetFileName(CString("/home/azuri/spectra/SEDIFU/SEDM-deep-sim-flat-2012-05-14_ScatterOnly.fits"));
        P_MyCFits->ReadArray();
        D_A2_ScatterFit = D_A2_ScatterFit - P_MyCFits->GetPixArray();
        P_MyCFits->WriteFits(&D_A2_ScatterFit, CString("/home/azuri/spectra/SEDIFU/SEDM-deep-sim-flat-2012-05-14_ScatterRes_new.fits"));

        return 0;
}
