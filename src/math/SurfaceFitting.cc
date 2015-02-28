#include "pfs/drp/stella/math/SurfaceFitting.h"
namespace pfs{ namespace drp{ namespace stella{ 
  namespace math{
    namespace tps{
    
      template <typename T> 
      int LU_Solve( boost::numeric::ublas::matrix<T> & a,
                    boost::numeric::ublas::matrix<T> & b )
      {
        // This routine is originally based on the public domain draft for JAMA,
        // Java matrix package available at http://math.nist.gov/javanumerics/jama/

        typedef boost::numeric::ublas::matrix<T> Matrix;
        typedef boost::numeric::ublas::matrix_row<Matrix> Matrix_Row;
        typedef boost::numeric::ublas::matrix_column<Matrix> Matrix_Col;

        if (a.size1() != b.size1())
          return 2;

        int m = a.size1(), n = a.size2();
        int pivsign = 0;
        int* piv = (int*)alloca( sizeof(int) * m);

        // PART 1: DECOMPOSITION
        //
        // For an m-by-n matrix A with m >= n, the LU decomposition is an m-by-n
        // unit lower triangular matrix L, an n-by-n upper triangular matrix U,
        // and a permutation vector piv of length m so that A(piv,:) = L*U.
        // If m < n, then L is m-by-m and U is m-by-n.
        {
          // Use a "left-looking", dot-product, Crout/Doolittle algorithm.
          for (int i = 0; i < m; ++i)
            piv[i] = i;
          pivsign = 1;

          // Outer loop.
          for (int j=0; j<n; ++j)
          {
            // Make a copy of the j-th column to localize references.
            Matrix_Col LUcolj(a,j);

            // Apply previous transformations.
            for (int i = 0; i < m; ++i)
            {
                Matrix_Row LUrowi(a,i);

                // This dot product is very expensive.
                // Optimize for SSE2?
                int kmax = (i<=j)?i:j;
                typename Matrix_Row::const_iterator ri_ite( LUrowi.begin());
                typename Matrix_Col::const_iterator cj_ite( LUcolj.begin());
                typename Matrix::value_type sum = 0.0;
                while( kmax-- > 0 )
                  sum += (*(ri_ite++)) * (*(cj_ite++));
                LUrowi[j] = LUcolj[i] -= sum;
            }

            // Find pivot and exchange if necessary.
            //
            // Slightly optimized version of:
            //  for (int i = j+1; i < m; ++i)
            //    if ( fabs(LUcolj[i]) > fabs(LUcolj[p]) )
            //      p = i;
            int p = j;
            typename Matrix::value_type coljp_abs = fabs(LUcolj[p]);
            for ( typename Matrix_Col::const_iterator
                    beg = LUcolj.begin(),
                    ite = beg + j+1,
                    end = LUcolj.end();
                  ite < end;
                  ++ite )
            {
              if (fabs(*ite) > coljp_abs)
              {
                p = ite-beg;
                coljp_abs = fabs(LUcolj[p]);
              }
            }

            if (p != j)
            {
                Matrix_Row raj(a,j);
                Matrix_Row(a,p).swap(raj);

                int tmp = piv[p];
                piv[p] = piv[j];
                piv[j] = tmp;
                pivsign = -pivsign;
            }

            // Compute multipliers.
            if (j < m && a(j,j) != 0.0)
                for (int i = j+1; i < m; ++i)
                  LUcolj[i] /= LUcolj[j];
          }
        }

        // PART 2: SOLVE

        // Check singluarity
        for (int j = 0; j < n; ++j)
          if (a(j,j) == 0)
            return 1;

        // Reorder b according to pivotting
        for (int i=0; i<m; ++i)
        {
          if ( piv[i] != i )
          {
            Matrix_Row b_ri( b, i );
            Matrix_Row( b, piv[i] ).swap( b_ri );
            for ( int j=i; j<m; ++j )
              if ( piv[j] == i )
              {
                piv[j] = piv[i];
                break;
              }
          }
        }

        // Solve L*Y = B(piv,:)
        for (int k=0; k<n; ++k)
        {
          const Matrix_Row& b_rk = Matrix_Row( b, k );
          for (int i = k+1; i < n; ++i)
          {
            const typename Matrix_Row::value_type aik = a(i,k);
            Matrix_Row( b, i ) -= b_rk * aik;
          }
        }

        // Solve U*X = Y;
        for (int k=n-1; k>=0; --k)
        {
          Matrix_Row(b,k) *= 1.0/a(k,k);

          const Matrix_Row& b_rk = Matrix_Row(b, k );
          for (int i=0; i<k; ++i)
          {
            const typename Matrix_Row::value_type aik = a(i,k);
            Matrix_Row(b,i) -= b_rk * aik;
          }
        }

        return 0;
      }

      template < typename T > 
      bool gauss_solve(boost::numeric::ublas::matrix<T> & a,
                       boost::numeric::ublas::matrix<T> & b )
      {
        int icol, irow;
        int n = a.size1();
        int m = b.size2();

        int* indxc = new int[n];
        int* indxr = new int[n];
        int* ipiv = new int[n];

        typedef boost::numeric::ublas::matrix<T> GJ_Mtx;
        typedef boost::numeric::ublas::matrix_row<GJ_Mtx> GJ_Mtx_Row;
        typedef boost::numeric::ublas::matrix_column<GJ_Mtx> GJ_Mtx_Col;

        for (int j=0; j<n; ++j)
          ipiv[j]=0;

        for (int i=0; i<n; ++i){
          T big=0.0;
          for (int j=0; j<n; j++){
            if (ipiv[j] != 1){
              for (int k=0; k<n; k++){
                if (ipiv[k] == 0){
                  T cmpa = a(j,k);
                  if ( cmpa < 0)
                    cmpa = -cmpa;
                  if (cmpa >= big){
                    big = cmpa;
                    irow=j;
                    icol=k;
                  }
                }
                else if (ipiv[k] > 1)
                   return false;
              }
            }
          }
          ++(ipiv[icol]);
          if (irow != icol){
            GJ_Mtx_Row ar1(a, irow), ar2 (a, icol);
            ar1.swap(ar2);

            GJ_Mtx_Row br1(b, irow), br2(b, icol);
            br1.swap(br2);
          }

          indxr[i] = irow;
          indxc[i] = icol;
          if (a(icol, icol) == 0.0)
            return false;

          T pivinv = 1.0 / a(icol, icol);
          a(icol, icol) = 1.0;
          GJ_Mtx_Row(a, icol) *= pivinv;
          GJ_Mtx_Row(b, icol) *= pivinv;

          for (int ll=0; ll<n; ll++){
            if (ll != icol){
              T dum = a(ll, icol);
              a(ll, icol) = 0.0;
              for (int l=0; l<n; l++)
                a(ll, l) -= a(icol, l) * dum;
              for (int l=0; l<m; l++)
                b(ll, l) -= b(icol, l) * dum;
            }
          }
        }

        // Unscramble A's columns
        for (int l=n-1; l>=0; --l){
          if (indxr[l] != indxc[l]){
            GJ_Mtx_Col ac1(a, indxr[l]), ac2(a, indxc[l]);
            ac1.swap(ac2);
          }
        }
        
        delete[] ipiv;
        delete[] indxr;
        delete[] indxc;

        return true;
      }
      
      static double tps_base_func(double r)
      {
        if ( r == 0.0 )
          return 0.0;
        else
          return r*r * log(r);
      }

      /*
       *  Calculate Thin Plate Spline (TPS) weights from
       *  control points and build a new height grid by
       *  interpolating with them.
       */
      template< typename T >
      static ndarray::Array<T, 2, 1> calc_tps(std::vector<T> const& xVec_In,
                                              std::vector<T> const& yVec_In,
                                              std::vector<T> const& zVec_In,
                                              int nRows,
                                              int nCols,
                                              double regularization)
      {
        std::vector< Vec > control_points;

        // Check input vectors
        if (xVec_In.size() != yVec_In.size()){
          string message("thin_plate_spline_fitting::calc_tps: ERROR: xVec_In.size(=");
          message += to_string(xVec_In.size()) + " != yVec_In.size(=" + to_string(yVec_In.size()) + ")";
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        if (xVec_In.size() != zVec_In.size()){
          string message("thin_plate_spline_fitting::calc_tps: ERROR: xVec_In.size(=");
          message += to_string(xVec_In.size()) + " != zVec_In.size(=" + to_string(zVec_In.size()) + ")";
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }

        // Populate control_points
        for (int i = 0; i < xVec_In.size(); ++i){
          Vec a(xVec_In[i] - (nCols / 2.), zVec_In[i], yVec_In[i] - (nRows / 2.));
          control_points.push_back(a);
          #ifdef __DEBUG_CALC_TPS__
            std::cout << "x[" << i << "] = " << a.x << ", y[" << i << "] = a.y, z[" << i << "] = " << a.z << std::endl;
          #endif
        }
        #ifdef __DEBUG_CALC_TPS__
          std::cout << "thin_plate_spline_fitting::calc_tps: control_points populated" << std::endl;
        #endif

        // You We need at least 3 points to define a plane
        if ( control_points.size() < 3 ){
          std::cout << "thin_plate_spline_fitting::calc_tps: ERROR: constrol_points.size(=" << control_points.size() << " < 3" << std::endl;
          exit(EXIT_FAILURE);
        }

        unsigned p = control_points.size();
        #ifdef __DEBUG_CALC_TPS__
          std::cout << "control_points.size() = " << control_points.size() << std::endl;
        #endif

        // Allocate the matrix and vector
        matrix<double> mtx_l(p+3, p+3);
        matrix<double> mtx_v(p+3, 1);
        matrix<double> mtx_orig_k(p, p);

        // Fill K (p x p, upper left of L) and calculate
        // mean edge length from control points
        //
        // K is symmetrical so we really have to
        // calculate only about half of the coefficients.
        double a = 0.0;
        for ( unsigned i=0; i<p; ++i )
        {
          for ( unsigned j=i+1; j<p; ++j )
          {
            Vec pt_i = control_points[i];
            Vec pt_j = control_points[j];
            pt_i.y = pt_j.y = 0;
            double elen = (pt_i - pt_j).len();
            mtx_l(i,j) = mtx_l(j,i) =
              mtx_orig_k(i,j) = mtx_orig_k(j,i) =
                tps_base_func(elen);
            a += elen * 2; // same for upper & lower tri
          }
        }
        a /= (double)(p*p);
        #ifdef __DEBUG_CALC_TPS__
          std::cout << "thin_plate_spline_fitting::calc_tps: a = " << a << std::endl;
        #endif

        // Fill the rest of L
        for ( unsigned i=0; i<p; ++i )
        {
          // diagonal: reqularization parameters (lambda * a^2)
          mtx_l(i,i) = mtx_orig_k(i,i) =
            regularization * (a*a);

          // P (p x 3, upper right)
          mtx_l(i, p+0) = 1.0;
          mtx_l(i, p+1) = control_points[i].x;
          mtx_l(i, p+2) = control_points[i].z;

          // P transposed (3 x p, bottom left)
          mtx_l(p+0, i) = 1.0;
          mtx_l(p+1, i) = control_points[i].x;
          mtx_l(p+2, i) = control_points[i].z;
        }
        // O (3 x 3, lower right)
        for ( unsigned i=p; i<p+3; ++i )
          for ( unsigned j=p; j<p+3; ++j )
            mtx_l(i,j) = 0.0;


        // Fill the right hand vector V
        for ( unsigned i=0; i<p; ++i )
          mtx_v(i,0) = control_points[i].y;
        mtx_v(p+0, 0) = mtx_v(p+1, 0) = mtx_v(p+2, 0) = 0.0;

        // Solve the linear system "inplace"
        #ifdef __DEBUG_CALC_TPS__
          std::cout << "thin_plate_spline_fitting::calc_tps: starting LU_Solve" << std::endl;
        #endif
        if (0 != LU_Solve(mtx_l, mtx_v))
        {
          puts( "Singular matrix! Aborting." );
          exit(1);
        }

        // Interpolate grid heights
        ndarray::Array<T, 2, 1> arr_Out = ndarray::allocate(nRows, nCols);
        arr_Out.deep() = 0.;
        for ( int x = -nCols / 2; x < nCols / 2; ++x )
        {
          for ( int z = -nRows / 2; z < nRows / 2; ++z )
          {
            double h = mtx_v(p+0, 0) + mtx_v(p+1, 0)*x + mtx_v(p+2, 0)*z;
            Vec pt_i, pt_cur(x,0,z);
            for ( unsigned i=0; i<p; ++i )
            {
              pt_i = control_points[i];
              pt_i.y = 0;
              h += mtx_v(i,0) * tps_base_func( ( pt_i - pt_cur ).len());
            }
            arr_Out[z + (nRows / 2)][x + (nCols / 2)] = h;
          }
          if (x % 100 == 0)
            std::cout << ".";
          #ifdef __DEBUG_CALC_TPS__
            std::cout << "thin_plate_spline_fitting::calc_tps: arr_Out[" << x + (nCols / 2) << "][0:10] = " << arr_Out[ndarray::view(x + (nCols / 2))(0:10)] << std::endl;
          #endif
        }

        // Calc bending energy
      /*  matrix<double> w( p, 1 );
        for ( int i=0; i<p; ++i )
          w(i,0) = mtx_v(i,0);
        matrix<double> be = prod( prod<matrix<double> >( trans(w), mtx_orig_k ), w );
        bending_energy = be(0,0);
        std::cout << "thin_plate_spline_fitting::calc_tps: bending_energy = " << bending_energy << std::endl;
      */  
        return arr_Out;
      }

      template int LU_Solve( boost::numeric::ublas::matrix<float> &, boost::numeric::ublas::matrix<float> &);
      template int LU_Solve( boost::numeric::ublas::matrix<double> &, boost::numeric::ublas::matrix<double> &);

      template bool gauss_solve(boost::numeric::ublas::matrix<float> &, boost::numeric::ublas::matrix<float> &);
      template bool gauss_solve(boost::numeric::ublas::matrix<double> &, boost::numeric::ublas::matrix<double> &);
      
      template ndarray::Array<float, 2, 1> calc_tps(std::vector<float> const&, std::vector<float> const&, std::vector<float> const&, int, int, double);
      template ndarray::Array<double, 2, 1> calc_tps(std::vector<double> const&, std::vector<double> const&, std::vector<double> const&, int, int, double);
    }  
  }
}}}
