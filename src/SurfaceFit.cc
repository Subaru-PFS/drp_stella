#include "pfs/drp/stella/SurfaceFit.h"

    bool SurfaceFit::doFit(const std::vector<float> &xIn, const std::vector<float> &yIn, const std::vector<float> &zIn, const std::vector<float> &wIn, const unsigned int &nKnotsX, const unsigned int &nKnotsY, const float &smooth, const int iopt)
    {
      float ssmooth = smooth;
      assert(xIn.size() == yIn.size());
      assert(xIn.size() == zIn.size());
      int i_opt = iopt;
      int ier = 0;
      int vecSize = xIn.size();
      int nc, nxy;
      nxy = vecSize - (vecSize / 10);
      float *x = new float[nxy];
      float *y = new float[nxy];
      float *z = new float[nxy];
      float *w = new float[nxy];
      float xb, xe;
      float yb, ye;
      int nxest,nyest,nmax;
      float eps;
      int nx,ny;
      float* tx;
      float* ty;
      float* c;
      float fp;
      float* wrk1;
      float* wrk2;
      int lwrk1;
      int* iwrk;
      assert(static_cast<int>(nKnotsX) > _kx+2);
      assert(static_cast<int>(nKnotsY) > _ky+2);
      cout << "SurfaceFit::SurfaceFit: xIn.size() = " << xIn.size() << endl;
      //        int lwa;
      //        float * wa;
      for (int i = 0; i < nxy; ++i)
      {
        x[i] = xIn[i];
        y[i] = yIn[i];
        z[i] = zIn[i];
        w[i] = wIn[i];
        cout << "SurfaceFit::SurfaceFit: x[" << i << "] = " << x[i] << ", y[" << i << "] = " << y[i] << ", z[" << i << "] = " << z[i] << ", w[" << i << "] = " << w[i] << endl;
      }
      xb = (*(std::min_element(xIn.begin(), xIn.end())));
      xe = (*(std::max_element(xIn.begin(), xIn.end())));
      yb = (*(std::min_element(yIn.begin(), yIn.end())));
      ye = (*(std::max_element(yIn.begin(), yIn.end())));
      cout << "SurfaceFit::SurfaceFit: xb = " << xb << ", xe = " << xe << ", yb = " << yb << ", ye = " << ye << endl;
      
      nx = int(nKnotsX);//nxest / 2;
      ny = int(nKnotsY);//nyest / 2;
      cout << "SurfaceFit::SurfaceFit: nx = " << nx << ", ny = " << ny << endl;
      
      nxest = nx+1;// * (kx+1+std::sqrt(m/2));
      nyest = ny+1;// * (ky+1+std::sqrt(m/2));
      cout << "SurfaceFit::SurfaceFit: nxest = " << nxest << ", nyest = " << nyest << endl;
      
      nmax = nxest > nyest ? 2 * nxest : 2 * nyest;
      cout << "SurfaceFit::SurfaceFit: nmax = " << nmax << endl;
      
      tx = new float[nx];
      ty = new float[ny];
      tx[0] = xb;
      ty[0]= yb;
      double txStep = (xe-xb) / (nx-1);
      double tyStep = (ye-yb) / (ny-1);
      cout << "SurfaceFit::SurfaceFit: tx[0] = " << tx[0] << ", ty[0] = " << ty[0] << endl;
      for (int i=1; i<nx; ++i){
        tx[i] = tx[i-1] + txStep;
        cout << "SurfaceFit::SurfaceFit: tx[" << i << "] = " << tx[i] << endl;
      }
      for (int i=1; i<ny; ++i){
        ty[i] = ty[i-1] + tyStep;
        cout << "SurfaceFit::SurfaceFit: ty[" << i << "] = " << ty[i] << endl;
      }
      
      eps = std::pow(10.,-37.);
      cout << "SurfaceFit::SurfaceFit: eps = " << eps << endl;
      
      nc = (nxest-_kx-1)*(nyest-_ky-1) * 10;
      c = new float[nc];
      cout << "SurfaceFit::SurfaceFit: c.size = " << (nxest-_kx-1)*(nyest-_ky-1) * 10 << endl;
      
      fp = 100.;
      
      int u = nxest-_kx-1;
      cout << "SurfaceFit::SurfaceFit: u = " << u << endl;
      int v = nyest-_ky-1;
      cout << "SurfaceFit::SurfaceFit: v = " << v << endl;
      int km = std::max(_kx,_ky)+1;
      cout << "SurfaceFit::SurfaceFit: km = " << km << endl;
      int ne = std::max(nxest,nyest);
      cout << "SurfaceFit::SurfaceFit: ne = " << ne << endl;
      int bx = _kx*v+_ky+1;
      cout << "SurfaceFit::SurfaceFit: bx = " << bx << endl;
      int by = _ky*u+_kx+1;
      cout << "SurfaceFit::SurfaceFit: by = " << by << endl;
      int b1, b2;
      if(bx <= by){
        b1 = bx;
        b2 = b1+v-_ky;
      }
      else{
        b1 = by;
        b2 = b1+u-_kx;
      }
      cout << "SurfaceFit::SurfaceFit: b1 = " << b1 << endl;
      cout << "SurfaceFit::SurfaceFit: b2 = " << b2 << endl;
      lwrk1 = u*v*(2+b1+b2)+2*(u+v+km*(nxy+ne)+ne-_kx-_ky)+b2+2;
      //        lwrk1 = nmax*nmax*(2+(2*((1+nmax)*5+6)+nmax));
      cout << "SurfaceFit::SurfaceFit: lwrk1 = " << lwrk1 << endl;
      //        lwrk1 += 2*(nmax+nmax+nmax*(m+nmax)+nmax);
      //        cout << "SurfaceFit::SurfaceFit: lwrk1 = " << lwrk1 << endl;
      //        lwrk1 += ((1+nmax)*5 + 6)+nmax+1;
      //        cout << "SurfaceFit::SurfaceFit: lwrk1 = " << lwrk1 << endl;
      wrk1 = new float[lwrk1];
      
      _lwrk2=1;// = u*v*(b2+1)+b2+1;
      //        lwrk2 = nmax*nmax*((((1+nmax)*5) + 6 + nmax)+1)+((1+nmax)*5) + 6 + nmax;
      cout << "SurfaceFit::SurfaceFit: _lwrk2 = " << _lwrk2 << endl;
      wrk2 = new float[_lwrk2];
      
      _kwrk = nxy+(nmax*nmax);
      cout << "SurfaceFit::SurfaceFit: _kwrk = " << _kwrk << endl;
      iwrk = new int[_kwrk];
      /*
       *   c            -1<=iopt<=1, 1<=kx,ky<=5, m>=(kx+1)*(ky+1), nxest>=2*kx+2,
       *   c            nyest>=2*ky+2, 0<eps<1, nmax>=nxest, nmax>=nyest,*
       *        if ((iopt < -1) || iopt > 1){
       *          cout << "SurfaceFit::SurfaceFit: iopt=" << iopt << " outside range" << endl;
       *          exit(EXIT_FAILURE);
    }
    if ((_kx < 1) || (_kx > 5)){
      cout << "SurfaceFit::SurfaceFit: _kx=" << _kx << " outside range" << endl;
      exit(EXIT_FAILURE);
    }
    if ((_ky < 1) || (_ky > 5)){
      cout << "SurfaceFit::SurfaceFit: _ky=" << _ky << " outside range" << endl;
      exit(EXIT_FAILURE);
    }
    if (_nxest < (2*_kx+2)){
      cout << "SurfaceFit::SurfaceFit: _nxest=" << _nxest << " outside range" << endl;
      exit(EXIT_FAILURE);
    }
    if (_nyest < (2*_ky+2)){
      cout << "SurfaceFit::SurfaceFit: _nyest=" << _nyest << " outside range" << endl;
      exit(EXIT_FAILURE);
    }
    if ((_eps <= 0.) || (_eps >= 1.)){
      cout << "SurfaceFit::SurfaceFit: _eps=" << _eps << " outside range" << endl;
      exit(EXIT_FAILURE);
    }
    if ((_nmax < _nxest) || (_nmax < _nyest)){
      cout << "SurfaceFit::SurfaceFit: _nmax=" << _nmax << " outside range" << endl;
      exit(EXIT_FAILURE);
    }
    *   c            xb<=x(i)<=xe, yb<=y(i)<=ye, w(i)>0, i=1,...,m
    *   c            lwrk1 >= u*v*(2+b1+b2)+2*(u+v+km*(m+ne)+ne-kx-ky)+b2+1
    *   c            kwrk >= m+(nxest-2*kx-1)*(nyest-2*ky-1)
    *   c            if iopt=-1: 2*kx+2<=nx<=nxest
    *   c                        xb<tx(kx+2)<tx(kx+3)<...<tx(nx-kx-1)<xe
    *   c                        2*ky+2<=ny<=nyest
    *   c                        yb<ty(ky+2)<ty(ky+3)<...<ty(ny-ky-1)<ye
    *
    if (_lwrk1 < (u*v*(2+b1+b2)+2*(u+v+km*(_m+ne)+ne-_kx-_ky)+b2+1)){
      cout << "SurfaceFit::SurfaceFit: _lwrk=" << _lwrk1 << " outside range" << endl;
      exit(EXIT_FAILURE);
    }
    if (_kwrk < (_m+(_nxest-2*_kx-1)*(_nyest-2*_ky-1))){
      cout << "SurfaceFit::SurfaceFit: _kwrk=" << _kwrk << " outside range" << endl;
      exit(EXIT_FAILURE);
    }
    if ((_nx < (2*_kx+2)) || (_nx > _nxest)){
      cout << "SurfaceFit::SurfaceFit: _nx=" << _nx << " outside range: 2*_kx+2=" << 2*_kx+2 << ", _nxest=" << _nxest << endl;
      exit(EXIT_FAILURE);
    }
    if ((_ny < (2*_ky+2)) || (_ny > _nyest)){
      cout << "SurfaceFit::SurfaceFit: _ny=" << _ny << " outside range" << endl;
      exit(EXIT_FAILURE);
    }
    */
      //        s = 10.*m;
      //_smooth = smooth;
      //cout << "SurfaceFit::SurfaceFit: _smooth = " << _smooth << endl;
      
      surfit_(i_opt,nxy,x,y,z,w,xb,xe,yb,ye,_kx,_ky,ssmooth,nxest,nyest,nmax,eps,nx,tx,ny,ty,c,fp,wrk1,lwrk1,wrk2,_lwrk2,iwrk,_kwrk,ier);
      cout << "SurfaceFit::SurfaceFit: ier = " << ier << endl;
      while (ier > 10){
        _lwrk2 = 1.5 * ier;
        delete[] wrk2;
        wrk2 = new float[_lwrk2];
        surfit_(i_opt,nxy,x,y,z,w,xb,xe,yb,ye,_kx,_ky,ssmooth,nxest,nyest,nmax,eps,nx,tx,ny,ty,c,fp,wrk1,lwrk1,wrk2,_lwrk2,iwrk,_kwrk,ier);
        cout << "SurfaceFit::SurfaceFit: ier = " << ier << endl;
        if (ier > 10)
          cout << "SurfaceFit::SurfaceFit: starting loop again" << endl;
      }
      //        for (int i=0; i<((_nxest-_kx-1)*(_nyest-_ky-1) + 10 < 1000 ? (_nxest-_kx-1)*(_nyest-_ky-1) + 10 : 1000); i++)
      //          cout << "SurfaceFit::SurfaceFit: _c[" << i << "] = " << _c[i] << endl;
      cout << "SurfaceFit::SurfaceFit: fp = " << fp << endl;
//      return false;
      if (((fp - smooth)/smooth) > 0.001){
        cout << "SurfaceFit::SurfaceFit: fit failed!" << endl;
        return false;
      }
      
      delete[] x;
      delete[] y;
      delete[] z;
      delete[] w;
      _coeffs.resize(0);
      _coeffs.reserve(nc);
      for (int i=0; i<nc; i++){
        _coeffs.push_back(c[i]);
      }
      delete[] c;
      _tx.resize(0);
      _tx.reserve(nx);
      for (int i=0; i<nx; i++)
        _tx.push_back(tx[i]);
      delete[] tx;
      _ty.resize(0);
      _ty.reserve(ny);
      for (int i=0; i<ny; i++)
        _ty.push_back(ty[i]);
      delete[] ty;
      delete[] wrk1;
      delete[] wrk2;
      delete[] iwrk;
      return true;
    }
    
    bool SurfaceFit::estimate(const std::vector<float> &xx, const std::vector<float> &yy, std::vector<float> &zz)
    {
      //c   mx >=1, my >=1, lwrk>=mx*(kx+1)+my*(ky+1), kwrk>=mx+my
      //c   tx(kx+1) <= x(i-1) <= x(i) <= tx(nx-kx), i=2,...,mx
      //c   ty(ky+1) <= y(j-1) <= y(j) <= ty(ny-ky), j=2,...,my
      //       subroutine bispev(tx,nx,ty,ny,c,kx,ky,x,mx,y,my,z,wrk,lwrk,iwrk,kwrk,ier)
      //       bispev_(tx,nx,ty,ny,c,_kx,_ky,xxx,mmx,yyy,mmy,zzz,wrk2,_lwrk2,iwrk,_kwrk,ier);
      
      assert(xx.size() == yy.size());
      if(_lwrk2 < _kx+_ky+2)
        _lwrk2 = _kx+_ky+2;
      cout << "SurfaceFit::estimate: xx.size() = " << xx.size() << endl;
      zz.resize(xx.size());
      //bispev_(_tx,_nx,_ty,_ny,_c,_kx,_ky,xxx,mmx,yyy,mmy,zzz,_wrk2,_lwrk2,_iwrk,_kwrk,_ier);
      int nx = _tx.size();
      int ny = _ty.size();
      float* tx = new float[nx];
      float* ty = new float[ny];
      for (int i=0; i<nx; i++)
        tx[i] = _tx[i];
      for (int i=0; i<ny; i++)
        ty[i] = _ty[i];
      float* c = new float[_coeffs.size()];
      for (int i=0; i<_coeffs.size(); i++)
        c[i] = _coeffs[i];
      float* wrk2 = new float[_lwrk2];
      int* iwrk = new int[_kwrk];
      int ier=0;
      float* xxx = new float[1];
      float* yyy = new float[1];
      float* zzz = new float[1];
      int mmx = 1;
      int mmy = 1;
      for (int i=0; i<static_cast<int>(xx.size()); ++i){
        xxx[0] = xx[i];
        yyy[0] = yy[i];
        std::cout << "SurfaceFit::estimate: xx[" << i << "] = " << xx[i] << ", yy[" << i << "] = " << yy[i] << endl;
        if (i == 0){
          cout << "SurfaceFit::estimate: nx = " << nx << ", ny = " << ny << ", _kx = " << _kx << ", _ky = " << _ky << ", mmx = " << mmx << ", mmy = " << mmy << ", _lwrk2 = " << _lwrk2 << ", _kwrk = " << _kwrk << endl;
          //            for (int j=0; j<_nx; ++j)
          //              cout << "SurfaceFit::estimate: _tx[" << j << "] = " << _tx[j] << endl;
          //            for (int j=0; j<_ny; ++j)
          //              cout << "SurfaceFit::estimate: _ty[" << j << "] = " << _ty[j] << endl;
          //            for (int j=0; j<((_nxest-_kx-1)*(_nyest-_ky-1) + 10 < 1000 ? (_nxest-_kx-1)*(_nyest-_ky-1) + 10 : 1000); ++j)
          //              cout << "SurfaceFit::estimate: _c[" << j << "] = " << _c[j] << endl;
          //            for (int j=0; j<lwrk2; ++j)
          //              cout << "SurfaceFit::estimate: wrk2[" << j << "] = " << wrk2[j] << endl;
          //            for (int j=0; j<kwrk; ++j)
          //              cout << "SurfaceFit::estimate: iwrk[" << j << "] = " << iwrk[j] << endl;
        }
        bispev_(tx,nx,ty,ny,c,_kx,_ky,xxx,mmx,yyy,mmy,zzz,wrk2,_lwrk2,iwrk,_kwrk,ier);
        zz[i] = zzz[0];
        std::cout << "SurfaceFit::estimate: zz[" << i << "] = " << zz[i] << endl;
      }
      delete[] xxx;
      delete[] yyy;
      delete[] zzz;
      delete[] tx;
      delete[] ty;
      delete[] c;
      delete[] wrk2;
      delete[] iwrk;
      if (ier != 0){
        cout << "SurfaceFit::estimate: ERROR: ier = " << ier << " => Fit FAILED => Returning FALSE" << endl;
        return false;
      }
      return true;
    }
    