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

// 1-dimensional reaction-diffusion model on [x0,x1]
struct reac_diff {
	// This represents the equations
	// dn1/dt = D1*laplace(n1) - 2*C*n1^2
	// dn2/dt = D2*laplace(n2) + C*n1^2
	//
	// with boundary conditions
	//
	// n1(x=0)  = 1
	// n1'(x=1) = 0
	// n2(x=0)  = 0
	// n2(x=1)  = 0.5
	//
	typedef arma::sp_mat jac_type;

	reac_diff( int Nx, double D1, double D2, double rate )
		: Nx(Nx), D1(D1), D2(D2), rate(rate),
		  diff_matrix(2*Nx, 2*Nx), diff_rhs(2*Nx),
		  dx(1.0/(Nx-1.0)), dx2(dx*dx)
	{
		// The n1-part of the Laplace operator.
		double idx2 = 1.0 / dx2;
		diff_rhs.zeros(2*Nx);
		std::cerr << "Accessing diff_matrix(0,0)...\n";
		diff_matrix(0,0) = D1*idx2;
		// Left side, n1(x=0) = 1.0;
		diff_rhs(0) = D1*idx2 * 1.0;

		for( int i = 1; i < Nx-2; ++i ){
			std::cerr << "Accessing diff_matrix(" << i << ", " << i << ")...\n";
			diff_matrix(i,i)   = -2.0*D1*idx2;
			std::cerr << "Accessing diff_matrix(" << i << ", " << i+1 << ")...\n";
			diff_matrix(i,i+1) = D1*idx2;
			std::cerr << "Accessing diff_matrix(" << i << ", " << i-1 << ")...\n";
			diff_matrix(i,i-1) = D1*idx2;
		}
		// Right side, n1' = 0.
		std::cerr << "Accessing diff_matrix(" << Nx-1 << ", " << Nx-1 << ")...\n";
		diff_matrix(Nx-1,Nx-1) = -2.0*D1*idx2;
		diff_matrix(Nx-1,Nx-2) =  2.0*D1*idx2;

		// n2-part of the Laplace operator:
		std::cerr << "Accessing diff_matrix(" << Nx << ", " << Nx << ")...\n";
		diff_matrix(Nx,Nx) = D2*idx2;
		diff_rhs(Nx) = 0.0;

		for( int i = Nx + 1; i < 2*Nx-2; ++i ){
			std::cerr << "Accessing diff_matrix(" << i << ", " << i << ")...\n";
			diff_matrix(i,i)   = -2.0*D2*idx2;
			std::cerr << "Accessing diff_matrix(" << i << ", " << i+1 << ")...\n";
			diff_matrix(i,i+1) = D2*idx2;
			std::cerr << "Accessing diff_matrix(" << i << ", " << i-1 << ")...\n";
			diff_matrix(i,i-1) = D2*idx2;
		}
		std::cerr << "Accessing diff_matrix(" << 2*Nx-1 << ", " << 2*Nx-1 << ")...\n";
		diff_matrix(2*Nx-1, 2*Nx-1) = D2*idx2;
		diff_rhs(2*Nx-1) = 0.5*D2*idx2;
		std::cerr << "Done constructing matrix and rhs...\n";
	}

	arma::vec fun( double t, const arma::vec &y )
	{
		arma::vec rhs = diff_matrix * y + diff_rhs;
		return rhs;
	}

	jac_type jac( double t, const arma::vec &y )
	{
		return diff_matrix;
	}

	int Nx;
	double D1, D2, rate;
	jac_type diff_matrix;
	arma::vec diff_rhs;
	double dx, dx2;
};

} // test_equations




#endif // TEST_EQUATIONS_HPP
