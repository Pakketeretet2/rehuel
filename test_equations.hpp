/*
   Rehuel: a simple C++ library for solving ODEs
 

   Copyright 2017-2019, Stefan Paquay (stefanpaquay@gmail.com)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

============================================================================= */

/**
   \file test_equations.hpp
   \brief Contains some typical ODEs that are good to test solvers on.
*/

#ifndef TEST_EQUATIONS_HPP
#define TEST_EQUATIONS_HPP

#include "matrix_vector.hpp"
#include <cassert>

#include "functor.hpp"

namespace test_equations {


// Exponential function:
struct exponential : public functor
{
	typedef mat_type jac_type;
	explicit exponential( double l ) : l(l) {}

	vec_type sol( double t )
	{
		return { exp( l*t ) };
	}

	vec_type fun( double t, const vec_type &y )
	{
		return l*y;
	}

	jac_type jac( double t, const vec_type &y )
	{
		return {l};
	}

	double l;
};


// Exponential function:
struct harmonic : public functor
{
	typedef mat_type jac_type;
	explicit harmonic( double w ) : w(w) {}

	vec_type sol( double t )
	{
		return { sin(w*t), cos(w*t) };
	}

	vec_type fun( double t, const vec_type &y )
	{
		return { w*y[1], -w*y[0] };
	}

	jac_type jac( double t, const vec_type &y )
	{
		return { { 0, w }, { -w, 0 } };
	}

	double w;
};





// Van der Pol oscillator:
struct vdpol : public functor
{
	typedef mat_type jac_type;
	vdpol( double mu ) : mu(mu){}

	vec_type fun( double t, const vec_type &y )
	{
		return { y[1], ((1 - y[0]*y[0]) * y[1] - y[0]) / mu };
	}

	jac_type jac( double t, const vec_type &y )
	{
		jac_type J(2,2);
		J(0,0) = 0.0;
		J(0,1) = 1.0;

		J(1,0) = -(2.0 * y[0] * y[1] + 1.0) / mu;
		J(1,1) = ( 1.0 - y[0]*y[0] ) / mu;

		return J;
	}

	double mu;
};



// Brusselator:
struct bruss : public functor
{
	typedef mat_type jac_type;
	bruss( double a, double b ) : a(a), b(b) {}

	vec_type fun( double t, const vec_type &y )
	{
		return { a + y[0]*y[0]*y[1] - b*y[0] - y[0],
		                  b*y[0] - y[0]*y[0]*y[1] };
	}

	jac_type jac( double t, const vec_type &y )
	{
		jac_type J(2,2);
		J(0,0) = 2*y[0]*y[1] - b - 1;
		J(0,1) = y[0]*y[0];

		J(1,0) = b - 2*y[0]*y[1];
		J(1,1) = -y[0]*y[0];

		return J;
	}

	double a, b;
};



// Robertson oscillator:
struct rober :  public functor {
	typedef mat_type jac_type;

	vec_type fun( double t, const vec_type &y )
	{
		return { -0.04*y[0] + 1e4 * y[1]*y[2],
		          0.04*y[0] - 1e4 * y[1]*y[2] - 3e7*y[1]*y[1],
		                  3e7*y[1]*y[1] };
	}


	jac_type jac( double t, const vec_type &y )
	{
		jac_type J(3,3);
		J(0,0) = -0.04;
		J(0,1) = 1e4*y[2];
		J(0,2) = 1e4*y[1];

		J(1,0) = 0.04;
		J(1,1) = -1e4*y[2] - 6e7*y[1];
		J(1,2) = -1e4*y[1];

		J(2,0) = J(2,2) = 0.0;
		J(2,1) = 6e7*y[1];
		return J;
	}
};


// Simple dimerization reaction:
struct dimer :  public functor  {
	typedef mat_type jac_type;

	explicit dimer( double rate ) : rate(rate), irate(1.0/rate) {}

	vec_type fun( double t, const vec_type &y )
	{
		return { -2*rate*y[0]*y[0] + 2*irate*y[1],
		                  rate*y[0]*y[0] - irate*y[1] };
	}

	jac_type jac( double t, const vec_type &y )
	{
		jac_type J(2,2);
		J(0,0) = -4*rate*y[0];
		J(0,1) =  2*irate;

		J(1,0) =  2*rate*y[0];
		J(1,1) =   -irate;
		return J;
	}

	double rate, irate;
};



/*
// 1-dimensional reaction-diffusion model on [x0,x1]
struct reac_diff :  public functor_sparse_jac {
	// This represents the equations
	// dn1/dt = D1*laplace(n1) - 2*C*n1^2 + 2*n2/C
	// dn2/dt = D2*laplace(n2) + C*n1^2 - n2/C
	//
	// with boundary conditions
	//
	// n1(x=0)  = 1
	// n1'(x=1) = 0
	// n2(x=0)  = 0.5
	// n2'(x=1) = 0

	typedef sp_mat_type jac_type;


	reac_diff( int Nx, double D1, double D2, double rate )
		: Nx(Nx), D1(D1), D2(D2), rate(rate), irate(1.0/rate),
		  diff_matrix(2*Nx, 2*Nx), diff_rhs(2*Nx),
		  dx(1.0/(Nx-1.0)), dx2(dx*dx)
	{
		// The n1-part of the Laplace operator.
		double idx2 = 1.0 * D1 / dx2;

		diff_rhs    = zeros(2*Nx);
		diff_matrix = zeros(2*Nx,2*Nx);

		// Left side, n1(x=0) = 1.0;
		// So it starts at x = dt.
		diff_rhs(0)      =  1.0*D1*idx2;
		diff_matrix(0,0) = -2.0*D1*idx2;
		diff_matrix(0,1) =  1.0*D1*idx2;

		for( int i = 1; i < Nx-2; ++i ){
			diff_matrix(i,i)   = -2.0*D1*idx2;
			diff_matrix(i,i+1) = D1*idx2;
			diff_matrix(i,i-1) = D1*idx2;
		}

		// Right side, n1' = 0.
		diff_matrix(Nx-1,Nx-1) = -2.0*D1*idx2;
		diff_matrix(Nx-1,Nx-2) =  2.0*D1*idx2;

		// n2-part of the Laplace operator:
		diff_rhs(Nx)         =  0.5 * D2 * idx2;
		diff_matrix(Nx,Nx)   = -2.0 * D2 * idx2;
		diff_matrix(Nx,Nx+1) =  1.0 * D2 * idx2;

		for( int i = Nx + 1; i < 2*Nx-2; ++i ){
			diff_matrix(i,i)   = -2.0*D2*idx2;
			diff_matrix(i,i+1) = D2*idx2;
			diff_matrix(i,i-1) = D2*idx2;
		}

		diff_matrix(2*Nx-1, 2*Nx-1) = -2.0*D2*idx2;
		diff_matrix(2*Nx-1, 2*Nx-2) =  2.0*D2*idx2;

	}

	vec_type fun( double t, const vec_type &y )
	{
		vec_type rhs = diff_matrix * y + diff_rhs;

		// rhs.zeros( 2*Nx );
		// Reaction part:
		for( int i = 0; i < Nx; ++i ){
			rhs[i]    += -2*rate*y[i]*y[i] + 2*irate*y[i+Nx];
			rhs[i+Nx] += rate*y[i]*y[i]    - irate*y[i+Nx];
		}

		return rhs;
	}

	jac_type jac( double t, const vec_type &y )
	{
		jac_type J = diff_matrix;
		// J.zeros(2*Nx,2*Nx);
		// Reaction part:
		for( int i = 0; i < Nx; ++i ){
			J(i,i)    = -4*rate*y[i];
			J(i,i+Nx) =  2*irate;

			J(i+Nx,i)    =  2*rate*y[i];
			J(i+Nx,i+Nx) = -irate;

		}


		return J;
	}

	int Nx;
	double D1, D2, rate, irate;
	jac_type diff_matrix;
	vec_type diff_rhs;
	double dx, dx2;
};
*/



struct stiff_eq  :  public functor {

	// y'' + 1001 y' + 1000 x = 0,
	// y'(0) = 0, y(0) = 1.0;
	// u := y' -->
	// y' = u
	// u' = -1001 u - 1000 y
	// --> Y' = A*Y, Y = ( y, u )
	//

	typedef mat_type jac_type;

	stiff_eq() : A(2,2)
	{
		A(0,0) = 0.0;
		A(0,1) = 1.0;
		A(1,0) = -1000.0;
		A(1,1) = -1001.0;
	}

	vec_type sol( double t ) const
	{
		double ysol = -exp(-1000*t)/999.0 + 1000*exp(-t)/999.0;
		double usol = ( 1000.0*exp(-1000*t) - 1000.0*exp(-t) ) / 999.0;
		vec_type solution(2);
		solution(0) = ysol;
		solution(1) = usol;
		return solution;
	}

	vec_type fun( double t, const vec_type &y )
	{
		return A*y;
	}

	jac_type jac( double  t, const vec_type &y )
	{
		return A;
	}

private:
	jac_type A;

};



struct three_body : public functor
{
	typedef mat_type jac_type;

	three_body( double m1, double m2, double m3 )
		: m1r(m1/m3), m2r(m2/m3), m1(m1), m2(m2), m3(m3) {}

	virtual vec_type fun( double t, const vec_type &y )
	{
		// y contains ( q0, q1, q2, q3, q4, q5,
		//            ( p0, p1, p2, p3, p4, p5 )
		//
		// (q0, q1, q2, q3, q4, q5) are (x0, y0, x1, y1, x2, y2)

		vec_type dydt(12);
		dydt[0] =  y[6] / m1;
		dydt[1] =  y[7] / m1;
		dydt[2] =  y[8] / m2;
		dydt[3] =  y[9] / m2;
		dydt[4] = y[10] / m3;
		dydt[5] = y[11] / m3;

		double x2mx1 = y[2] - y[0];
		double y2my1 = y[3] - y[1];
		double x3mx1 = y[4] - y[0];
		double y3my1 = y[5] - y[1];
		double x3mx2 = y[4] - y[2];
		double y3my2 = y[5] - y[3];

		double r12_2 = x2mx1*x2mx1 + y2my1*y2my1;
		double r13_2 = x3mx1*x3mx1 + y3my1*y3my1;
		double r23_2 = x3mx2*x3mx2 + y3my2*y3my2;

		double r12_1 = sqrt(r12_2);
		double r13_1 = sqrt(r13_2);
		double r23_1 = sqrt(r23_2);

		double r12_3 = r12_2*r12_1;
		double r13_3 = r13_2*r13_1;
		double r23_3 = r23_2*r23_1;

		dydt[ 6] =  m1*m3*x3mx1 / r13_3 + m1*m2*x2mx1 / r12_3;
		dydt[ 7] =  m1*m3*y3my1 / r13_3 + m1*m2*y2my1 / r12_3;
		dydt[ 8] = -m1*m2*x2mx1 / r12_3 + m2*m3*x3mx2 / r23_3;
		dydt[ 9] = -m1*m2*y2my1 / r12_3 + m2*m3*x3mx2 / r23_3;
		dydt[10] = -m2*m3*x3mx2 / r23_3 - m1*m3*x3mx1 / r13_3;
		dydt[11] = -m2*m3*y3my2 / r23_3 - m1*m3*y3my1 / r13_3;

		return dydt;
	}

	virtual jac_type jac( double t, const vec_type &y )
	{
		mat_type J(12,12);

		double x2mx1 = y[2] - y[0];
		double y2my1 = y[3] - y[1];
		double x3mx1 = y[4] - y[0];
		double y3my1 = y[5] - y[1];
		double x3mx2 = y[4] - y[2];
		double y3my2 = y[5] - y[3];

		double x2mx1_2 = x2mx1*x2mx1;
		double y2my1_2 = y2my1*y2my1;
		double x3mx1_2 = x3mx1*x3mx1;
		double y3my1_2 = y3my1*y3my1;
		double x3mx2_2 = x3mx2*x3mx2;
		double y3my2_2 = y3my2*y3my2;

		double r12_2 = x2mx1_2 + y2my1_2;
		double r13_2 = x3mx1_2 + y3my1_2;
		double r23_2 = x3mx2_2 + y3my2_2;

		double r12_1 = sqrt(r12_2);
		double r13_1 = sqrt(r13_2);
		double r23_1 = sqrt(r23_2);

		double r12_3 = r12_2*r12_1;
		double r13_3 = r13_2*r13_1;
		double r23_3 = r23_2*r23_1;

		double r12_5 = r12_3*r12_2;
		double r13_5 = r13_3*r13_2;
		double r23_5 = r23_3*r23_2;


		J = arma::zeros(12,12);

		// [ 0..5 ] is p, [ 6..11 ] is x.
		J( 6, 6)  = -m1*m3 / r13_3 + 3*m1*m3*x3mx1_2 / r13_5
		          - m1*m2 / r12_3 + 3*m1*m2*x2mx1_2 / r12_5;

		J( 8, 8) = -m2*m3 / r23_3 + 3*m2*m3*x3mx2_2 / r23_5
		          - m1*m2 / r12_3 + 3*m1*m2*x2mx1_2 / r12_5;

		J(10,10) = -m1*m3 / r13_3 + 3*m1*m3*x3mx1_2 / r13_5
		          - m2*m3 / r23_3 + 3*m2*m3*x3mx2_2 / r23_5;

		J( 7, 7) = -m1*m3 / r13_3 + 3*m1*m3*y3my1_2 / r13_5
		          - m1*m2 / r12_3 + 3*m1*m2*y2my1_2 / r12_5;

		J( 9, 9) = -m2*m3 / r23_3 + 3*m2*m3*y3my2_2 / r23_5
		          - m1*m2 / r13_3 + 3*m1*m2*y2my1_2 / r12_5;

		J(11,11) = -m1*m3 / r13_3 + 3*m1*m3*y3my1_2 / r13_5
		          - m2*m3 / r23_3 + 3*m2*m3*y3my2_2 / r23_5;

		// Cross terms:
		J( 7, 6) = 3*m1*m3*x3mx1*y3my1 / r13_5
		           +3*m1*m2*x2mx1*y2my1 / r12_5;
		J( 8, 6) = m1*m2/r12_3 - 3*m1*m2*x2mx1_2 / r12_5;
		J( 9, 6) = -3*m1*m2*x2mx1*y2my1 / r12_5;
		J(10, 6) = m1*m3/r13_3 - 3*m1*m3*x3mx1_2 / r13_5;
		J(11, 6) = -3*m1*m3*x3mx1*y3my1 / r13_5;

		J( 8, 7) = -3*m1*m2*x2mx1*y2my1 / r12_5;
		J( 9, 7) = m1*m2/r12_3 - 3*m1*m2*y2my1_2 / r12_5;
		J(10, 7) = -3*m1*m3*x3mx1*y3my1 / r13_5;
		J(11, 7) = m1*m3/r13_2 - 3*m1*m3*y3my1_2 / r13_5;

		J( 9, 8) = 3*m2*m3*x3mx2*y3my2 / r23_5
		           + 3*m1*m2*x2mx1*y2my1 / r12_5;
		J(10, 8) = m2*m3 / r23_3 - 3*m2*m3*x3mx2_2 / r23_5;
		J(11, 8) = -3*m2*m3*x3mx2*y3my2 / r23_5;

		J(10, 9) = -3*m2*m3*x3mx2*y3my2 / r23_5;
		J(11, 9) = m2*m3/r23_3 - 3*m2*m3*y3my2_2 / r23_5;

		J(11,10) = 3*m2*m3*x3mx2*y3my2 / r23_5
		         + 3*m1*m3*x3mx1*y3my1 / r13_5;

		// symmetry:

		J( 6, 7) = J( 7, 6);

		J( 6, 8) = J( 8, 6);
		J( 7, 8) = J( 8, 7);

		J( 6, 9) = J( 9, 6);
		J( 7, 9) = J( 9, 7);
		J( 8, 9) = J( 9, 8);

		J( 6,10) = J(10, 6);
		J( 7,10) = J(10, 7);
		J( 8,10) = J(10, 8);
		J( 9,10) = J(10, 9);

		J( 6,11) = J(11, 6);
		J( 7,11) = J(11, 7);
		J( 8,11) = J(11, 8);
		J( 9,11) = J(11, 9);
		J(10,11) = J(11,10);

		// x to x:
		J(0,0) = 1.0 / m1;
		J(1,1) = 1.0 / m1;
		J(2,2) = 1.0 / m2;
		J(3,3) = 1.0 / m2;
		J(4,4) = 1.0 / m3;
		J(5,5) = 1.0 / m3;

		// x to p and p to x are 0.


		return -J;
	}


	double kin_energy( const vec_type &y )
	{
		double px0 = y[6];
		double py0 = y[7];
		double px1 = y[8];
		double py1 = y[9];
		double px2 = y[10];
		double py2 = y[11];

		double T = px0*px0 + py0*py0;
		T += px1*px1 + py1*py1;
		T += px2*px2 + py2*py2;

		return 0.5*T;
	}

	double pot_energy( const vec_type &y )
	{

		double r12_x = y[2] - y[0];
		double r12_y = y[3] - y[1];
		double r13_x = y[4] - y[0];
		double r13_y = y[5] - y[1];
		double r23_x = y[4] - y[2];
		double r23_y = y[5] - y[3];

		double r12_2 = r12_x*r12_x + r12_y*r12_y;
		double r13_2 = r13_x*r13_x + r13_y*r13_y;
		double r23_2 = r23_x*r23_x + r23_y*r23_y;

		double r12 = sqrt(r12_2);
		double r13 = sqrt(r13_2);
		double r23 = sqrt(r23_2);

		double V = -m1*m2/r12 - m1*m3/r13 - m2*m3/r23;
		return V;
	}


	double m1r, m2r;
	double m1, m2, m3;
};


struct kinetic_4 : public functor
{
	typedef mat_type jac_type;

	kinetic_4( double b2, double b3, double b4 )
		: b2(b2), b3(b3), b4(b4) {}

	virtual vec_type fun( double t, const vec_type &y )
	{
		vec_type rhs(4);
		rhs(0)  = -2*y(0)*y(0)  - y(0)*(y(1) + y(2));
		rhs(0) += 2*b2*y(1) + b3*y(2) + b4*y(3);

		rhs(1) = y(0)*y(0) - y(0)*y(1) + b3*y(2) - b2*y(1);
		rhs(2) = y(0)*y(1) - y(0)*y(2) + b4*y(3) - b3*y(2);
		rhs(3) = y(0)*y(2) - b4*y(3);

		return rhs;
	}


	virtual jac_type jac( double t, const vec_type &y )
	{
		jac_type J = { { -4*y(0) - y(1) - y(2), -y(0) + 2*b2, -y(0) + b3, b4 },
		               { 2*y(0) - y(1), -y(0) - b2, b3, 0.0 },
		               { y(1) - y(2), y(0), -y(0) - b3, b4  },
		               { y(2), 0, y(0), -b4} };
		return J;
	}

	double b2, b3, b4;
};


} // test_equations




#endif // TEST_EQUATIONS_HPP
