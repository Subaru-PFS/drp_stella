/*
 * spline.h
 *
 * simple cubic spline interpolation library without external
 * dependencies
 *
 * ---------------------------------------------------------------------
 * Copyright (C) 2011, 2014 Tino Kluge (ttk448 at gmail.com)
 *
 *  This program is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU General Public License
 *  as published by the Free Software Foundation; either version 2
 *  of the License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * ---------------------------------------------------------------------
 *
 */

/*
 * Usage

#include <cstdio>
#include <cstdlib>
#include <vector>
#include "spline.h"

int main(int argc, char** argv) {

   std::vector<T> X(5), Y(5);
   X[0]=0.1; X[1]=0.4; X[2]=1.2; X[3]=1.8; X[4]=2.0;
   Y[0]=0.1; Y[1]=0.7; Y[2]=0.6; Y[3]=1.1; Y[4]=0.9;

   tk::spline s;
   s.set_points(X,Y);    // currently it is required that X is already sorted

   T x=1.5;

   printf("spline at %f is %f\n", x, s(x));

   return EXIT_SUCCESS;
}

Compile:

$ g++ -Wall demo.cpp -o demo
$ ./demo
spline at 1.500000 is 0.915345
*/

#ifndef _tk_spline_h
#define _tk_spline_h

#include <vector>

// unnamed namespace only because the implementation is in this
// header file and we don't want to export symbols to the obj files
namespace pfs { namespace drp { namespace stella { namespace math {
// band matrix solver
template<typename T>
class band_matrix {
private:
   std::vector< std::vector<T> > m_upper;  // upper band
   std::vector< std::vector<T> > m_lower;  // lower band
public:
   band_matrix() {};                             // constructor
   band_matrix(int dim, int n_u, int n_l);       // constructor
   ~band_matrix() {};                            // destructor
   void resize(int dim, int n_u, int n_l);      // init with dim,n_u,n_l
   int dim() const;                             // matrix dimension
   int num_upper() const {
      return m_upper.size()-1;
   }
   int num_lower() const {
      return m_lower.size()-1;
   }
   // access operator
   T & operator () (int i, int j);            // write
   T   operator () (int i, int j) const;      // read
   // we can store an additional diogonal (in m_lower)
   T& saved_diag(int i);
   T  saved_diag(int i) const;
   void lu_decompose();
   std::vector<T> r_solve(const std::vector<T>& b) const;
   std::vector<T> l_solve(const std::vector<T>& b) const;
   std::vector<T> lu_solve(const std::vector<T>& b,
                                bool is_lu_decomposed=false);

};


// spline interpolation
template<typename T>
class spline {
private:
   std::vector<T> m_x,m_y;           // x,y coordinates of points
   // interpolation parameters
   // f(x) = a*(x-x_i)^3 + b*(x-x_i)^2 + c*(x-x_i) + y_i
   std::vector<T> m_a,m_b,m_c;
public:
   void set_points(const std::vector<T>& x,
                   const std::vector<T>& y, bool cubic_spline=true);
   T operator() (T x) const;
};

}}}} // namespace pfs { namespace drp { namespace stella { namespace math {

#endif /* _tk_spline_h */
