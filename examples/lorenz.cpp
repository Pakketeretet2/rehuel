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
   \file lorenz.cpp

   \brief This file contains code to generate a Lorenz' strange attractor.

   Compile as g++ -O3 -I../ -fopenmp -march=native -lrehuel -larmadillo lorenz.cpp -o lorenz
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

struct lorenz {
	typedef arma::vec vec_type;
	typedef arma::mat mat_type;
	typedef mat_type jac_type;
	
	lorenz(double sigma = 10, double rho = 28, double beta = 8.0/3.0)
		: s(sigma), r(rho), b(beta) {}

	vec_type fun(double t, const vec_type &y) const
	{
		return { s*(y(1) - y(0)),
		         y(0)*(r - y(2)) - y(1),
		         y(0)*y(1) - b*y(2)
		};
	}

	mat_type jac(double t, const vec_type &y) const
	{
		return { { -s, s, 0 },
		         { r-y(2), -1.0, -y(0) },
		         {y(1), y(0), -b}
		};
	}

	double s, r, b;
};


int solve_irk(const std::string &method_str, lorenz &l,
              arma::vec Y0, double t0, double t1)
{
	newton::options newton_opts;
	irk::solver_options solver_opts = irk::default_solver_options();
	solver_opts.newton_opts = &newton_opts;
	int method = irk::RADAU_IIA_32;
	method = irk::name_to_method(method_str);
	irk::rk_output sol = irk::odeint( l, t0, t1, Y0,
	                                  solver_opts, method );

	std::cerr << "Solved ODE with " << sol.t_vals.size() << " time steps in "
	          << sol.elapsed_time / 1000.0 << " seconds ("
	          << sol.t_vals.size()*1000.0/sol.elapsed_time
	          << " time steps / s).\n";
	for( std::size_t i = 0; i < sol.t_vals.size(); ++i ){
		std::cout << sol.t_vals[i];
		for( std::size_t j = 0; j < sol.y_vals[i].size(); ++j ){
			std::cout << " " << sol.y_vals[i][j];
		}
		std::cout << "\n";
	}
	return 0;
}



int solve_erk(const std::string &method_str, lorenz &l,
              arma::vec Y0, double t0, double t1)
{
	erk::solver_options solver_opts = erk::default_solver_options();
	int method = erk::CASH_KARP_54;
	method = erk::name_to_method(method_str);
	solver_opts.adaptive_step_size = true;
	erk::rk_output sol = erk::odeint( l, t0, t1, Y0,
	                                  solver_opts, method );
	
	std::cerr << "Solved ODE with " << sol.t_vals.size() << " time steps in "
	          << sol.elapsed_time / 1000.0 << " seconds ("
	          << sol.t_vals.size()*1000.0/sol.elapsed_time
	          << " time steps / s).\n";
	for( std::size_t i = 0; i < sol.t_vals.size(); ++i ){
		std::cout << sol.t_vals[i];
		for( std::size_t j = 0; j < sol.y_vals[i].size(); ++j ){
			std::cout << " " << sol.y_vals[i][j];
		}
		std::cout << "\n";
	}
	return 0;
}



int main( int argc, char **argv )
{
	double t0 = 0.0;
	double t1 = 1e2;
	arma::vec Y0 = { 10.0, 1.0, 1.0 };

	lorenz l;
	std::string method_name = "RADAU_IIA_53";
	if (argc > 1) {
		method_name = argv[1];
	}

	int method = irk::name_to_method(method_name);
	if (method) {
		return solve_irk(method_name, l, Y0, t0, t1);
	}

	method = erk::name_to_method(method_name);
	if (method) {
		return solve_erk(method_name, l, Y0, t0, t1);
	}
	
	return 0;
}

