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
   \file robertson.cpp

   \brief This file contains code to solve the Robertson oscillator as example.

   Compile as g++ -O3 -I../ -fopenmp -march=native -lrehuel -larmadillo robertson.cpp -o robertson
*/

#include "rehuel.hpp"



// Robertson oscillator:
struct rober {
	typedef mat_type jac_type;

	rober(double a = 3e7, double b = 1e4) : a(a), b(b) {}
	
	
	vec_type fun( double t, const vec_type &y )
	{
		return { -0.04*y[0] + b * y[1]*y[2],
		          0.04*y[0] - b * y[1]*y[2] - a*y[1]*y[1],
		          a*y[1]*y[1] };
	}


	jac_type jac( double t, const vec_type &y )
	{
		jac_type J(3,3);
		J(0,0) = -0.04;
		J(0,1) = b*y[2];
		J(0,2) = b*y[1];

		J(1,0) = 0.04;
		J(1,1) = -b*y[2] - 2*a*y[1];
		J(1,2) = -b*y[1];

		J(2,0) = J(2,2) = 0.0;
		J(2,1) = 2*a*y[1];
		return J;
	}

	double a, b;
};




int solve_irk(const std::string &method_str, rober &robertson,
              arma::vec Y0, double t0, double t1)
{
	newton::options newton_opts;
	irk::solver_options solver_opts = irk::default_solver_options();
	solver_opts.newton_opts = &newton_opts;
	solver_opts.out_interval = 2e6;
	int method = irk::RADAU_IIA_32;
	method = irk::name_to_method(method_str);
	irk::rk_output sol = irk::odeint( robertson, t0, t1, Y0,
	                                  solver_opts, method );
	
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



int solve_erk(const std::string &method_str, rober &robertson,
              arma::vec Y0, double t0, double t1)
{
	erk::solver_options solver_opts = erk::default_solver_options();
	int method = erk::CASH_KARP_54;
	method = erk::name_to_method(method_str);
	solver_opts.adaptive_step_size = true;
		solver_opts.out_interval = 2e6;
	erk::rk_output sol = erk::odeint( robertson, t0, t1, Y0,
	                                  solver_opts, method, 1e-8 );
	
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



int main( int argc, char **argv )
{
	double t0 = 0.0;
	double t1 = 1e12;
	arma::vec Y0 = { 0.9, 0.0, 0.0 };

	rober robertson(3e1, 1e0);
	std::string method_name = "RADAU_IIA_53";
	if (argc > 1) {
		method_name = argv[1];
	}

	int method = irk::name_to_method(method_name);
	if (method) {
		// method = 0 indicates that this method is not an irk.
		std::cerr << "method = " << method << " for name "
		          << method_name << "\n";
		return solve_irk(method_name, robertson, Y0, t0, t1);
	}

	method = erk::name_to_method(method_name);
	if (method) {
		return solve_erk(method_name, robertson, Y0, t0, t1);
	}
	
	return 0;
}
