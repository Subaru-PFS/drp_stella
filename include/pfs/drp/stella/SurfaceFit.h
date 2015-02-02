#ifndef __SURFACEFIT_H__
#define __SURFACEFIT_H__

#include <iostream>
using std::cout;
using std::endl;
#include <vector>
#include <cassert>
#include <cstdlib>
#include <algorithm>
#include <cmath>

extern "C" {
    void surfit_(int&,int&,float[],float[],float[],float[],float&,float&,float&,float&,int&,int&,float&,int&,int&,int&,float&,int&,float[],int&,float[],float[],float&,float[],int&,float[],int&,int[],int&,int&);
}
extern "C" {
    void bispev_(float[], int&, float[], int&, float[], int&, int&, float[], int&, float[], int&, float[], float[], int&, int[], int&, int&);
}


class SurfaceFit {
private:
  /**
   c parameters:
   c  iopt  : integer flag. on entry iopt must specify whether a weighted
   c          least-squares spline (iopt=-1) or a smoothing spline (iopt=0
   c          or 1) must be determined.
   c          if iopt=0 the routine will start with an initial set of knots
   c          tx(i)=xb,tx(i+kx+1)=xe,i=1,...,kx+1;ty(i)=yb,ty(i+ky+1)=ye,i=
   c          1,...,ky+1. if iopt=1 the routine will continue with the set
   c          of knots found at the last call of the routine.
   c          attention: a call with iopt=1 must always be immediately pre-
   c                     ceded by another call with iopt=1 or iopt=0.
   c          unchanged on exit.
   c  m     : integer. on entry m must specify the number of data points.
   c          m >= (kx+1)*(ky+1). unchanged on exit.
   c  x     : real array of dimension at least (m).
   c  y     : real array of dimension at least (m).
   c  z     : real array of dimension at least (m).
   c          before entry, x(i),y(i),z(i) must be set to the co-ordinates
   c          of the i-th data point, for i=1,...,m. the order of the data
   c          points is immaterial. unchanged on exit.
   c  w     : real array of dimension at least (m). before entry, w(i) must
   c          be set to the i-th value in the set of weights. the w(i) must
   c          be strictly positive. unchanged on exit.
   c  xb,xe : real values. on entry xb,xe,yb and ye must specify the bound-
   c  yb,ye   aries of the rectangular approximation domain.
   c          xb<=x(i)<=xe,yb<=y(i)<=ye,i=1,...,m. unchanged on exit.
   c  kx,ky : integer values. on entry kx and ky must specify the degrees
   c          of the spline. 1<=kx,ky<=5. it is recommended to use bicubic
   c          (kx=ky=3) splines. unchanged on exit.
   c  s     : real. on entry (in case iopt>=0) s must specify the smoothing
   c          factor. s >=0. unchanged on exit.
   c          for advice on the choice of s see further comments
   c  nxest : integer. unchanged on exit.
   c  nyest : integer. unchanged on exit.
   c          on entry, nxest and nyest must specify an upper bound for the
   c          number of knots required in the x- and y-directions respect.
   c          these numbers will also determine the storage space needed by
   c          the routine. nxest >= 2*(kx+1), nyest >= 2*(ky+1).
   c          in most practical situation nxest = kx+1+sqrt(m/2), nyest =
   c          ky+1+sqrt(m/2) will be sufficient. see also further comments.
   c  nmax  : integer. on entry nmax must specify the actual dimension of
   c          the arrays tx and ty. nmax >= nxest, nmax >=nyest.
   c          unchanged on exit.
   c  eps   : real.
   c          on entry, eps must specify a threshold for determining the
   c          effective rank of an over-determined linear system of equat-
   c          ions. 0 < eps < 1.  if the number of decimal digits in the
   c          computer representation of a real number is q, then 10**(-q)
   c          is a suitable value for eps in most practical applications.
   c          unchanged on exit.
   c  nx    : integer.
   c          unless ier=10 (in case iopt >=0), nx will contain the total
   c          number of knots with respect to the x-variable, of the spline
   c          approximation returned. if the computation mode iopt=1 is
   c          used, the value of nx should be left unchanged between sub-
   c          sequent calls.
   c          in case iopt=-1, the value of nx should be specified on entry
   c  tx    : real array of dimension nmax.
   c          on succesful exit, this array will contain the knots of the
   c          spline with respect to the x-variable, i.e. the position of
   c          the interior knots tx(kx+2),...,tx(nx-kx-1) as well as the
   c          position of the additional knots tx(1)=...=tx(kx+1)=xb and
   c          tx(nx-kx)=...=tx(nx)=xe needed for the b-spline representat.
   c          if the computation mode iopt=1 is used, the values of tx(1),
   c          ...,tx(nx) should be left unchanged between subsequent calls.
   c          if the computation mode iopt=-1 is used, the values tx(kx+2),
   c          ...tx(nx-kx-1) must be supplied by the user, before entry.
   c          see also the restrictions (ier=10).
   c  ny    : integer.
   c          unless ier=10 (in case iopt >=0), ny will contain the total
   c          number of knots with respect to the y-variable, of the spline
   c          approximation returned. if the computation mode iopt=1 is
   c          used, the value of ny should be left unchanged between sub-
   c          sequent calls.
   c          in case iopt=-1, the value of ny should be specified on entry
   c  ty    : real array of dimension nmax.
   c          on succesful exit, this array will contain the knots of the
   c          spline with respect to the y-variable, i.e. the position of
   c          the interior knots ty(ky+2),...,ty(ny-ky-1) as well as the
   c          position of the additional knots ty(1)=...=ty(ky+1)=yb and
   c          ty(ny-ky)=...=ty(ny)=ye needed for the b-spline representat.
   c          if the computation mode iopt=1 is used, the values of ty(1),
   c          ...,ty(ny) should be left unchanged between subsequent calls.
   c          if the computation mode iopt=-1 is used, the values ty(ky+2),
   c          ...ty(ny-ky-1) must be supplied by the user, before entry.
   c          see also the restrictions (ier=10).
   c  c     : real array of dimension at least (nxest-kx-1)*(nyest-ky-1).
   c          on succesful exit, c contains the coefficients of the spline
   c          approximation s(x,y)
   c  fp    : real. unless ier=10, fp contains the weighted sum of
   c          squared residuals of the spline approximation returned.
   c  wrk1  : real array of dimension (lwrk1). used as workspace.
   c          if the computation mode iopt=1 is used the value of wrk1(1)
   c          should be left unchanged between subsequent calls.
   c          on exit wrk1(2),wrk1(3),...,wrk1(1+(nx-kx-1)*(ny-ky-1)) will
   c          contain the values d(i)/max(d(i)),i=1,...,(nx-kx-1)*(ny-ky-1)
   c          with d(i) the i-th diagonal element of the reduced triangular
   c          matrix for calculating the b-spline coefficients. it includes
   c          those elements whose square is less than eps,which are treat-
   c          ed as 0 in the case of presumed rank deficiency (ier<-2).
   c  lwrk1 : integer. on entry lwrk1 must specify the actual dimension of
   c          the array wrk1 as declared in the calling (sub)program.
   c          lwrk1 must not be too small. let
   c            u = nxest-kx-1, v = nyest-ky-1, km = max(kx,ky)+1,
   c            ne = max(nxest,nyest), bx = kx*v+ky+1, by = ky*u+kx+1,
   c            if(bx.le.by) b1 = bx, b2 = b1+v-ky
   c            if(bx.gt.by) b1 = by, b2 = b1+u-kx  then
   c          lwrk1 >= u*v*(2+b1+b2)+2*(u+v+km*(m+ne)+ne-kx-ky)+b2+1
   c  wrk2  : real array of dimension (lwrk2). used as workspace, but
   c          only in the case a rank deficient system is encountered.
   c  lwrk2 : integer. on entry lwrk2 must specify the actual dimension of
   c          the array wrk2 as declared in the calling (sub)program.
   c          lwrk2 > 0 . a save upper boundfor lwrk2 = u*v*(b2+1)+b2
   c          where u,v and b2 are as above. if there are enough data
   c          points, scattered uniformly over the approximation domain
   c          and if the smoothing factor s is not too small, there is a
   c          good chance that this extra workspace is not needed. a lot
   c          of memory might therefore be saved by setting lwrk2=1.
   c          (see also ier > 10)
   c  iwrk  : integer array of dimension (kwrk). used as workspace.
   c  kwrk  : integer. on entry kwrk must specify the actual dimension of
   c          the array iwrk as declared in the calling (sub)program.
   c          kwrk >= m+(nxest-2*kx-1)*(nyest-2*ky-1).
   c  ier   : integer. unless the routine detects an error, ier contains a
   c          non-positive value on exit, i.e.
   c   ier=0  : normal return. the spline returned has a residual sum of
   c            squares fp such that abs(fp-s)/s <= tol with tol a relat-
   c            ive tolerance set to 0.001 by the program.
   c   ier=-1 : normal return. the spline returned is an interpolating
   c            spline (fp=0).
   c   ier=-2 : normal return. the spline returned is the weighted least-
   c            squares polynomial of degrees kx and ky. in this extreme
   c            case fp gives the upper bound for the smoothing factor s.
   c   ier<-2 : warning. the coefficients of the spline returned have been
   c            computed as the minimal norm least-squares solution of a
   c            (numerically) rank deficient system. (-ier) gives the rank.
   c            especially if the rank deficiency which can be computed as
   c            (nx-kx-1)*(ny-ky-1)+ier, is large the results may be inac-
   c            curate. they could also seriously depend on the value of
   c            eps.
   c   ier=1  : error. the required storage space exceeds the available
   c            storage space, as specified by the parameters nxest and
   c            nyest.
   c            probably causes : nxest or nyest too small. if these param-
   c            eters are already large, it may also indicate that s is
   c            too small
   c            the approximation returned is the weighted least-squares
   c            spline according to the current set of knots.
   c            the parameter fp gives the corresponding weighted sum of
   c            squared residuals (fp>s).
   c   ier=2  : error. a theoretically impossible result was found during
   c            the iteration proces for finding a smoothing spline with
   c            fp = s. probably causes : s too small or badly chosen eps.
   c            there is an approximation returned but the corresponding
   c            weighted sum of squared residuals does not satisfy the
   c            condition abs(fp-s)/s < tol.
   c   ier=3  : error. the maximal number of iterations maxit (set to 20
   c            by the program) allowed for finding a smoothing spline
   c            with fp=s has been reached. probably causes : s too small
   c            there is an approximation returned but the corresponding
   c            weighted sum of squared residuals does not satisfy the
   c            condition abs(fp-s)/s < tol.
   c   ier=4  : error. no more knots can be added because the number of
   c            b-spline coefficients (nx-kx-1)*(ny-ky-1) already exceeds
   c            the number of data points m.
   c            probably causes : either s or m too small.
   c            the approximation returned is the weighted least-squares
   c            spline according to the current set of knots.
   c            the parameter fp gives the corresponding weighted sum of
   c            squared residuals (fp>s).
   c   ier=5  : error. no more knots can be added because the additional
   c            knot would (quasi) coincide with an old one.
   c            probably causes : s too small or too large a weight to an
   c            inaccurate data point.
   c            the approximation returned is the weighted least-squares
   c            spline according to the current set of knots.
   c            the parameter fp gives the corresponding weighted sum of
   c            squared residuals (fp>s).
   c   ier=10 : error. on entry, the input data are controlled on validity
   c            the following restrictions must be satisfied.
   c            -1<=iopt<=1, 1<=kx,ky<=5, m>=(kx+1)*(ky+1), nxest>=2*kx+2,
   c            nyest>=2*ky+2, 0<eps<1, nmax>=nxest, nmax>=nyest,
   c            xb<=x(i)<=xe, yb<=y(i)<=ye, w(i)>0, i=1,...,m
   c            lwrk1 >= u*v*(2+b1+b2)+2*(u+v+km*(m+ne)+ne-kx-ky)+b2+1
   c            kwrk >= m+(nxest-2*kx-1)*(nyest-2*ky-1)
   c            if iopt=-1: 2*kx+2<=nx<=nxest
   c                        xb<tx(kx+2)<tx(kx+3)<...<tx(nx-kx-1)<xe
   c                        2*ky+2<=ny<=nyest
   c                        yb<ty(ky+2)<ty(ky+3)<...<ty(ny-ky-1)<ye
   c            if iopt>=0: s>=0
   c            if one of these conditions is found to be violated,control
   c            is immediately repassed to the calling program. in that
   c            case there is no approximation returned.
   c   ier>10 : error. lwrk2 is too small, i.e. there is not enough work-
   c            space for computing the minimal least-squares solution of
   c            a rank deficient system of linear equations. ier gives the
   c            requested value for lwrk2. there is no approximation re-
   c            turned but, having saved the information contained in nx,
   c            ny,tx,ty,wrk1, and having adjusted the value of lwrk2 and
   c            the dimension of the array wrk2 accordingly, the user can
   c            continue at the point the program was left, by calling
   c            surfit with iopt=1.
   **/
    std::vector<float> _tx;
    std::vector<float> _ty;
    std::vector<float> _coeffs;
    int _lwrk2;
    int _kwrk;
    int _kx;
    int _ky;
public:
    explicit SurfaceFit() : _tx(1), _ty(1), _coeffs(1), _lwrk2(1), _kwrk(1), _kx(3), _ky(3){}
    explicit SurfaceFit(const SurfaceFit &surfaceFit) : _tx(surfaceFit.getTX()),
                                                        _ty(surfaceFit.getTY()),
                                                        _coeffs(surfaceFit.getCoeffs()),
                                                        _lwrk2(surfaceFit.getLWrk2()),
                                                        _kwrk(surfaceFit.getKWrk()),
                                                        _kx(surfaceFit.getKX()),
                                                        _ky(surfaceFit.getKY()){}
    ~SurfaceFit()
    {
    }

    bool doFit(const std::vector<float> &xIn, const std::vector<float> &yIn, const std::vector<float> &zIn, const std::vector<float> &wIn, const unsigned int &nKnotsX, const unsigned int &nKnotsY, const float &smooth, const int iopt=-1);
    
    bool estimate(const std::vector<float> &xx, const std::vector<float> &yy, std::vector<float> &zz);
    
/*    std::vector<float> getY () const {
      std::vector<float> yvec;
      for (int i=0; i<_m; ++i)
        yvec.push_back(_y[i]);
      return yvec;
    }
    std::vector<float> getZ () const{
      std::vector<float> zvec;
      for (int i=0; i<_m; ++i)
        zvec.push_back(_z[i]);
      return zvec;
    }
    std::vector<float> getW () const {
      std::vector<float> wvec;
      for (int i=0; i<_m; ++i)
        wvec.push_back(_w[i]);
      return wvec;
    }
    float getXB() const {return _xb;}
    float getXE() const {return _xe;}
    float getYB() const {return _yb;}
    float getYE() const {return _ye;}*/
    int getKX() const {return _kx;}
    int getKY() const {return _ky;}
/*    float getSmooth() const {return _smooth;}
    int getNXEst() const {return _nxest;}
    int getNYEst() const {return _nyest;}
    int getNMax() const {return _nmax;}
    float getEps() const {return _eps;}
    int getNX() const {return _nx;}
    int getNY() const {return _ny;}*/
    std::vector<float> getTX () {return _tx;}
    std::vector<float> getTX () const{
      std::vector<float> xvec;
      for (int i=0; i<_tx.size(); ++i)
        xvec.push_back(_tx[i]);
      return xvec;
    }
    std::vector<float> getTY () {return _ty;}
    std::vector<float> getTY () const {
      std::vector<float> yvec;
      for (int i=0; i<_ty.size(); ++i)
        yvec.push_back(_ty[i]);
      return yvec;
    }
//    int getNC() const {return _nc;}
    std::vector<float> getCoeffs () {return _coeffs;}
    std::vector<float> getCoeffs () const {
      std::vector<float> cvec;
      for (int i=0; i<_coeffs.size(); ++i)
        cvec.push_back(_coeffs[i]);
      return cvec;
    }
//    float getFP() const {return _fp;}
//    int getLWrk1() const {return _lwrk1;}
    int getLWrk2() const {return _lwrk2;}
    int getKWrk() const {return _kwrk;}
//    int getIEr() const {return _ier;}
    /*
     *     int _iopt, _m;
     *    float *_x;
     *    float *_y;
     *    float *_z;
     *    float *_w;
     *    float _xb, _xe;
     *    float _yb, _ye;
     *    int _kx, _ky;
     *    float _smooth;
     *    int _nxest,_nyest,_nmax;
     *    float _eps;
     *    int _nx,_ny;
     *    float* _tx;
     *    float* _ty;
     *    float* _c;
     *    float _fp;
     *    float* _wrk1;
     *    float* _wrk2;
     *    int _lwrk1,_lwrk2;
     *    int* _iwrk;
     *    int _kwrk, _ier;
     * */
};

#endif // __SURFACEFIT_H__
