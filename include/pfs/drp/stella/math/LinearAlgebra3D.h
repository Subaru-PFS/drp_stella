#ifndef __LINEARALGEBRA3D_H__
#define __LINEARALGEBRA3D_H__

/*
 *  One file long C++ library of linear algebra primitives for
 *  simple 3D programs
 *
 *  Copyright (C) 2001-2003 by Jarno Elonen
 *
 *  Permission to use, copy, modify, distribute and sell this software
 *  and its documentation for any purpose is hereby granted without fee,
 *  provided that the above copyright notice appear in all copies and
 *  that both that copyright notice and this permission notice appear
 *  in supporting documentation.  The authors make no representations
 *  about the suitability of this software for any purpose.
 *  It is provided "as is" without express or implied warranty.
 */

#include <cmath>

#define __EPSILON__ 0.00001f
#define __CONST_PI__ 3.1415926
#define Deg2Rad(Ang) ((float)( Ang * __CONST_PI__ / 180.0 ))
#define Rad2Deg(Ang) ((float)( Ang * 180.0 / __CONST_PI__ ))

namespace pfs{ namespace drp{ namespace stella{ namespace math{
    // =========================================
    // 3-vector
    // =========================================
    class Vec
    {
    public:

      // Position
      float x, y, z;

      // Default constructor
      Vec()
      : x( 0 ), y( 0 ), z( 0 ) {}

      // Element constructor
      Vec( float x, float y, float z )
      : x( x ), y( y ), z( z ) {}

      // Copy constructor
      Vec( const Vec& a )
      : x( a.x ), y( a.y ), z( a.z ) {}

      // Norm (len^2)
      inline float norm() const { return x*x + y*y + z*z; }

      // Length of the vector
      inline float len() const { return (float)sqrt(norm()); }

      Vec &operator += ( const Vec &src ) { x += src.x; y += src.y; z += src.z; return *this; }
      Vec operator + ( const Vec &src ) const { Vec tmp( *this ); return ( tmp += src ); }
      Vec &operator -= ( const Vec &src ) { x -= src.x; y -= src.y; z -= src.z; return *this; }
      Vec operator - ( const Vec &src ) const { Vec tmp( *this ); return ( tmp -= src ); }

      Vec operator - () const { return Vec(-x,-y,-z); }

      Vec &operator *= ( const float src ) { x *= src; y *= src; z *= src;  return *this; }
      Vec operator * ( const float src ) const { Vec tmp( *this ); return ( tmp *= src ); }
      Vec &operator /= ( const float src ) { x /= src; y /= src; z /= src; return *this; }
      Vec operator / ( const float src ) const { Vec tmp( *this ); return ( tmp /= src ); }

      bool operator == ( const Vec& b) const { return ((*this)-b).norm() < __EPSILON__; }
      //bool operator == ( const Vec& b) const { return x==b.x && y==b.y && z==b.z; }
    };

      // Left hand float multplication
      inline Vec operator * ( const float src, const Vec& v ) { Vec tmp( v ); return ( tmp *= src ); }

      // Dot product
      inline float dot( const Vec& a, const Vec& b )
      { return a.x*b.x + a.y*b.y + a.z*b.z; }

      // Cross product
      inline Vec cross( const Vec &a, const Vec &b )
      { return Vec( a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x ); }


    // =========================================
    // 4 x 4 matrix
    // =========================================
    class Mtx
    {
    public:

      // 4x4, [[0 1 2 3] [4 5 6 7] [8 9 10 11] [12 13 14 15]]
      float data[ 16 ];

      // Creates an identity matrix
      Mtx()
      {
        for ( int i = 0; i < 16; ++i )
          data[ i ] = 0;
        data[ 0 + 0 ] = data[ 4 + 1 ] = data[ 8 + 2 ] = data[ 12 + 3 ] = 1;
      }

      // Returns the transpose of this matrix
      Mtx transpose() const
      {
        Mtx m;
        int idx = 0;
        for ( int row = 0; row < 4; ++row )
          for ( int col = 0; col < 4; ++col, ++idx )
            m.data[ idx ] = data[ row + ( col * 4 ) ];
        return m;
      }

      // Operators
      float operator () ( unsigned column, unsigned row )
      { return data[ column + ( row * 4 ) ]; }
    };

    // Creates a scale matrix
    Mtx scale( const Vec &scale );

    // Creates a translation matrix
    Mtx translate( const Vec &moveAmt );

    // Creates an euler rotation matrix (by X-axis)
    Mtx rotateX( float ang );

    // Creates an euler rotation matrix (by Y-axis)
    Mtx rotateY( float ang );

    // Creates an euler rotation matrix (by Z-axis)
    Mtx rotateZ( float ang );

    // Creates an euler rotation matrix (pitch/head/roll (x/y/z))
    Mtx rotate( float pitch, float head, float roll );

    // Creates an arbitraty rotation matrix
    Mtx makeRotationMatrix( const Vec &dir, const Vec &up );

    // Transforms a vector by a matrix
    inline Vec operator * ( const Vec& v, const Mtx& m );

    // Multiplies a matrix by another matrix
    Mtx operator * ( const Mtx& a, const Mtx& b );

    // =========================================
    // Plane
    // =========================================
    class Plane
    {
    public:
      enum PLANE_EVAL
      {
        EVAL_COINCIDENT,
        EVAL_IN_BACK_OF,
        EVAL_IN_FRONT_OF,
        EVAL_SPANNING
      };

      Vec normal;
      float d;

      // Default constructor
      Plane(): normal( 0,1,0 ), d( 0 ) {}

      // Vector form constructor
      //   normal = normalized normal of the plane
      //   pt = any point on the plane
      Plane( const Vec& normal, const Vec& pt )
        : normal( normal ), d( dot( -normal, pt )) {}

      // Copy constructor
      Plane( const Plane& a )
        : normal( a.normal ), d( a.d ) {}

      // Classifies a point (<0 == back, 0 == on plane, >0 == front)
      float classify( const Vec& pt ) const
      {
        float f = dot( normal, pt ) + d;
        return ( f > -__EPSILON__ && f < __EPSILON__ ) ? 0 : f;
      }
    };
}}}}
#endif
