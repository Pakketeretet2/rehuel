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
   \file bz.cpp

   \brief This file contains code to solve the Brusselator as example.

   Compile as g++ -O3 -I../ -lrehuel -larmadillo single_bz.cpp -o single_bz
*/

#include <iostream>

#include "rehuel.hpp"

/*
  ODEs are implemented through functors. They need to implement two functions:
  fun and jac.
  The first calculates the RHS of the ODE, the second the Jacobi matrix.
  It also needs a typedef, jac_type, to indicate whether or not the Jacobi
  matrix is sparse or not. Currently only non-sparse is supported.
*/
struct bz_funtor {
	typedef arma::mat jac_type;

	arma::vec fun( double t, const arma::vec &Y ) const
	{
		const double x = Y[0];
		const double y = Y[1];
		const double z = Y[2];
		const double u = Y[3];

		const double c_o_min_z = c_o - z;
		const double z_part  = c_o_min_z/( c_o_min_z + c_m);
		const double z_part2 = c_o_min_z/(b_c + 1.0);
		const double xy = x*y;
		const double x2 = x*x;
		return { -k1*xy + k2*y - 2*k3*x2 + k4*x*z_part,
		         -3*k1*xy - 2*k2*y - k3*x2 + k7*u + k9*z + kI*z_part2,
		         2*k4*x*z_part - k9*z - k10*z + kI * z_part2,
		         2*k1*xy + k2*y + k3*x2 - k7*u
		};
	}

	jac_type jac( double t, const arma::vec &Y ) const
	{
		const double x = Y[0];
		const double y = Y[1];
		const double z = Y[2];
		const double u = Y[3];

		const double c_o_min_z = c_o - z;
		const double n = c_o_min_z + c_m;
		const double z_part  = c_o_min_z/n;
		const double z_part2 = c_o_min_z/(b_c + 1.0);

		const double n2 = n*n;
		const double z_part_p  = -c_m / n2;
		const double z_part2_p = -1.0/(b_c + 1.0);
		
		const double xy = x*y;
		const double x2 = x*x;

		arma::mat A(4,4);

		A(0,0) = -k1*y - 4*k3*x + k4*z_part;
		A(0,1) = -k1*x + k2;
		A(0,2) = k4*x*z_part_p;
		A(0,3) = 0.0;

		A(1,0) = -3*k1*y - 2*k3*x;
		A(1,1) = -3*k1*x - 2*k2;
		A(1,2) = k9 + kI*z_part2_p;
		A(1,3) = k7;
			
		A(2,0) = 2*k4*x*z_part;
		A(2,1) = 0.0;
		A(2,2) = 2*k4*x*z_part_p - k9 - k10 + kI * z_part2_p;
		A(2,3) = 0.0;
				
		A(3,0) = 2*k1*y + 2*k3*x;
		A(3,1) = 2*k1*x + k2;
		A(3,2) = 0.0;
		A(3,3) = -k7;


		return A;
	}

	double k1, k2, k3, k4, k5, k6, k7, k8, k9, k10;
	double kr, kred, c_o, b_c, kI;

	double c_m;
};


int main()
{

	arma::vec Y0 = {2.80466e-7, 7.47363e-6, 0.0000163656, 1.25563e-7 };
	bz_funtor bzf;
	bzf.k1  = 3.2e5;
	bzf.k2  = 0.01536;
	bzf.k3  = 3e3;
	bzf.k4  = 2.016;
	bzf.k5  = 8e8;
	bzf.k6  = 10;
	bzf.k7  = 11.6;
	bzf.k8  = 3.72;
	bzf.k9  = 0.048;
	bzf.k10 = 0.02;

	bzf.kr   = 2e8;
	bzf.kred = 5e6; // street kred haha
	bzf.c_o  = 4.2e-3;
	bzf.b_c  = 0.05;
	bzf.kI   = 0.0;

	double cm2 = 2*bzf.kr*(bzf.k9 + bzf.k10)*bzf.c_o / (bzf.kred*bzf.kred);
	bzf.c_m  = std::sqrt(cm2);

	newton::options newton_opts;
	irk::solver_options solver_opts = irk::default_solver_options();
	solver_opts.rel_tol = 1e-7;
	solver_opts.abs_tol = 1e-7; // Put abs tol small if your absolute numbers are small.
	solver_opts.newton_opts = &newton_opts;
	int method = irk::RADAU_IIA_53;

	double t0 = 0.0;
	double t1 = 1e5;
	double t  = 0;

	// Check to make sure there are no errors in the implementation of
	// the Jacobi matrix.
	auto bzf_fun = [&bzf]( const arma::vec &Y ){ return bzf.fun(0,Y); };
	auto bzf_jac = [&bzf]( const arma::vec &Y ){ return bzf.jac(0,Y); };
	newton::newton_lambda_wrapper<decltype(bzf_fun),
	                              decltype(bzf_jac),
	                              bz_funtor::jac_type> bzf_wrap( bzf_fun, bzf_jac );
	if( newton::verify_jacobi_matrix( Y0, bzf_wrap ) ){
		std::cerr << "Jacobi matrix seems consistent at Y0\n";
	}else{
		std::cerr << "Jacobi matrix is fishy!\n";
		return -1;
	}

	my_timer timer(std::cerr);
	irk::rk_output sol = irk::odeint(bzf, t0, t1, Y0,
	                                 solver_opts, method);
	timer.toc("Solving BZ with RADAU IIA 53");

	std::cerr << "IRK solver: Made " << sol.count.attempt
	          << " attempted steps, of which " << sol.count.reject_newton
	          << " got rejected because of newton, "	          << sol.count.reject_err
	          << " because of err.\n";


	auto erk_solver_opts = erk::default_solver_options();
	erk_solver_opts.rel_tol = solver_opts.rel_tol;
	erk_solver_opts.abs_tol = solver_opts.abs_tol;

	timer.tic();
	erk::rk_output sol2 = erk::odeint(bzf, t0, t1, Y0,
	                                  erk_solver_opts,
	                                  erk::CASH_KARP_54);
	timer.toc("Solving BZ with CASH-KARP 54");
	
	std::cerr << "Made " << sol.count.attempt << " attempted steps, of which "
	          << sol.count.reject_newton << " got rejected because of newton, "
	          << sol.count.reject_err << " because of err.\n";

	
	std::cerr << "Solved ODE with " << sol.t_vals.size() << " time steps in "
	          << sol.elapsed_time / 1000.0 << " seconds.\n";
	for( std::size_t i = 0; i < sol.t_vals.size(); ++i ){
		std::cout << sol.t_vals[i];
		for( std::size_t j = 0; j < sol.y_vals[i].size(); ++j ){
			std::cout << " " << sol.y_vals[i][j];
		}
		std::cout << "\n";
	}
	
	return 0;
}
