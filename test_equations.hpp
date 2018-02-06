/*
   Rehuel: a simple C++ library for solving ODEs


   Copyright 2017, Stefan Paquay (stefanpaquay@gmail.com)

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

#include <armadillo>

#include <cassert>

namespace test_equations {

// Van der Pol oscillator:
struct vdpol
{
	typedef arma::mat jac_type;
	vdpol( double mu ) : mu(mu){}

	arma::vec fun( double t, const arma::vec &y )
	{
		return { y[1], mu * ( 1 - y[0]*y[0] ) * y[1] - y[0] };
	}

	jac_type jac( double t, const arma::vec &y )
	{
		jac_type J(2,2);
		J(0,0) = 0.0;
		J(0,1) = 1.0;

		J(1,0) = -2*mu*y[0]*y[1] - 1.0;
		J(1,1) = mu*(1 - y[0]*y[0]);

		return J;
	}

	double mu;
};

// Robertson oscillator:
struct rober {
	typedef arma::mat jac_type;

	arma::vec fun( double t, const arma::vec &y )
	{
		return { -0.04*y[0] + 1e4 * y[1]*y[2],
		          0.04*y[0] - 1e4 * y[1]*y[2] - 3e7*y[2]*y[2],
		          3e7*y[2]*y[2] };
	}

	jac_type jac( double t, const arma::vec &y )
	{
		jac_type J(3,3);
		J(0,0) = -0.04;
		J(0,1) = 1e4*y[2];
		J(0,2) = 1e4*y[1];

		J(1,0) = 0.04;
		J(1,1) = -1e4*y[2];
		J(1,2) = -1e4*y[1] - 6e7*y[2];

		J(2,0) = J(2,1) = 0.0;
		J(2,2) = 6e7*y[2];
		return J;
	}
};


// Simple dimerization reaction:
struct dimer {
	typedef arma::mat jac_type;

	explicit dimer( double rate ) : rate(rate), irate(1.0/rate) {}

	arma::vec fun( double t, const arma::vec &y )
	{
		return { -2*rate*y[0]*y[0] + 2*irate*y[1],
		          rate*y[0]*y[0] - irate*y[1] };
	}

	jac_type jac( double t, const arma::vec &y )
	{
		jac_type J(2,2);
		J(0,0) = -4*rate*y[0];
		J(0,1) = -4*rate*y[0];

		J(1,0) = -4*rate*y[0];
		J(1,1) = -4*rate*y[0];
		return J;
	}

	double rate, irate;
};



// 1-dimensional reaction-diffusion model on [x0,x1]
struct reac_diff {
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
	//
	typedef arma::sp_mat jac_type;
	// typedef arma::mat jac_type;


	reac_diff( int Nx, double D1, double D2, double rate )
		: Nx(Nx), D1(D1), D2(D2), rate(rate), irate(1.0/rate),
		  diff_matrix(2*Nx, 2*Nx), diff_rhs(2*Nx),
		  dx(1.0/(Nx-1.0)), dx2(dx*dx)
	{
		// The n1-part of the Laplace operator.
		double idx2 = 1.0 * D1 / dx2;

		diff_rhs.zeros(2*Nx);
		diff_matrix.zeros(2*Nx,2*Nx);

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

	arma::vec fun( double t, const arma::vec &y )
	{
		arma::vec rhs = diff_matrix * y + diff_rhs;

		// rhs.zeros( 2*Nx );
		// Reaction part:
		for( std::size_t i = 0; i < Nx; ++i ){
			rhs[i]    += -2*rate*y[i]*y[i] + 2*irate*y[i+Nx];
			rhs[i+Nx] += rate*y[i]*y[i]    - irate*y[i+Nx];
		}

		return rhs;
	}

	jac_type jac( double t, const arma::vec &y )
	{
		jac_type J = diff_matrix;
		// J.zeros(2*Nx,2*Nx);
		// Reaction part:
		for( std::size_t i = 0; i < Nx; ++i ){
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
	arma::vec diff_rhs;
	double dx, dx2;
};




struct stiff_eq {

	// y'' + 1001 y' + 1000 x = 0,
	// y'(0) = 0, y(0) = 1.0;
	// u := y' -->
	// y' = u
	// u' = -1001 u - 1000 y
	// --> Y' = A*Y, Y = ( y, u )
	//

	typedef arma::mat jac_type;

	stiff_eq() : A(2,2)
	{
		A.zeros(2,2);
		A(0,0) = 0.0;
		A(0,1) = 1.0;
		A(1,0) = -1000.0;
		A(1,1) = -1001.0;
	}

	arma::vec sol( double t ) const
	{
		double ysol = -exp(-1000*t)/999.0 + 1000*exp(-t)/999.0;
		double usol = ( 1000.0*exp(-1000*t) - 1000.0*exp(-t) ) / 999.0;
		arma::vec solution(2);
		solution(0) = ysol;
		solution(1) = usol;
		return solution;
	}

	arma::vec fun( double t, const arma::vec &y )
	{
		return A*y;
	}

	jac_type jac( double  t, const arma::vec &y )
	{
		return A;
	}

private:
	jac_type A;

};


} // test_equations




#endif // TEST_EQUATIONS_HPP
