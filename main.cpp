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
   \file main.cpp
*/
#include "interpolate.hpp"
#include "newton.hpp"
#include "odes.hpp"
#include "irk.hpp"


#include <cmath>
#include <iostream>

/**
   \brief test the non-linear solvers on Rosenbrock's function.
*/
void test_newton_solvers()
{
	// Test newton and broyden solvers:
	arma::vec x0 = { 0.9, 0.9 };
	std::cerr << "Testing non-linear solvers...\n";

	newton::options opts;
	newton::status stats;
	opts.tol = 1e-8;
	opts.maxit = 50000;
	opts.time_internals = true;
	opts.max_step = 1.0;
	opts.refresh_jac = true;

	double a = 1.0;
	double b = 100;
	auto ff = [a,b](const arma::vec &x)
		{ return newton::test_functions::rosenbrock_F( x, a, b ); };
	auto JJ = [a,b](const arma::vec &x)
		{ return newton::test_functions::rosenbrock_J( x, a, b ); };


	arma::vec root_newton = newton::solve( ff, x0, opts, stats, JJ );

	std::cerr << "Full Newton ";
	if( stats.conv_status ){
		std::cerr << "failed to converge, final approx solution = ";
	}else{
		std::cerr << "converged, root = ";
	}
	std::cerr << "(" << root_newton(0) << ", "
	          << root_newton(1) << "); it took " << stats.iters
	          << " iterations (res = " << stats.res << ").\n";

	arma::vec root_broyden = newton::solve( ff, x0, opts, stats );

	std::cerr << "Broyden  ";
	if( stats.conv_status ){
		std::cerr << "failed to converge, final approx solution = ";
	}else{
		std::cerr << "converged, root = ";
	}
	std::cerr << "(" << root_broyden(0) << ", "
	          << root_broyden(1) << "); it took " << stats.iters
	          << " iterations (res = " << stats.res << ").\n";
}

/**
   \brief tests given ODE solver on a test problem.

   ys and times shall be unmodified upon failure.

   \param dt      Time step size to use
   \param method  Method to use
   \param t0      Initial time
   \param t1      Final time
   \param times   Will contain the time levels at which a solution exists
   \param ys      Will contain the solutions to the ODE.

   \returns 0 on success, non-zero otherwise.
*/
int solve_test_ode( double dt, int method, double t0, double t1,
                    std::vector<double> &times, std::vector<arma::vec> &ys )
{

	irk::solver_coeffs sc = irk::get_coefficients( method );
	irk::solver_options s_opts = irk::default_solver_options();

	sc.dt = dt;
	s_opts.local_tol = 1e-6;
	s_opts.internal_solver = irk::solver_options::BROYDEN;
	arma::vec y0 = { 1.0, 0.0 };

	double a = 1.0 / 20.0;
	double b = 1.0 / 15.0;
	double w = 0.05;

	auto ode = [a,b,w]( double t, const arma::vec &yy ){
		return odes::analytic_solvable_func( yy, a, b, w ); };
	auto ode_J = [a,b,w]( double t, const arma::vec &yy ){
		return odes::analytic_solvable_func_J( yy, a, b, w ); };

	if( y0.size() != ode(t0, y0).size() ){
		std::cerr << "Dimensions of initial condition and ODE "
		          << "do not match! Aborting!\n";
		return -1;
	}

	int odeint_status = irk::odeint( t0, t1, sc, s_opts, y0,
	                                 ode, ode_J, times, ys );
	return odeint_status;
}


/**
   \brief Makes a time grid.

   \param t0 Starting point
   \param t1 Final point

   \returns a time grid
*/
std::vector<double> make_time_grid( double t0, double t1 )
{
	double tc = t0;
	std::vector<double> t_grid;
	double dtc = 0.01;

	while( tc < 0.5 ){
		t_grid.push_back( tc );
		tc += dtc;
	}
	dtc = 0.05;
	while( tc < std::min( 2.5, t1 ) ){
		t_grid.push_back( tc );
		tc += dtc;
	}
	dtc = 0.2;
	while( tc < std::min( 10.0, t1 ) ){
		t_grid.push_back( tc );
		tc += dtc;
	}
	dtc = 0.5;
	while( tc < t1 ){
		t_grid.push_back( tc );
		tc += dtc;
	}
	t_grid.push_back(t1);

	return t_grid;
}


/**
   \brief Main entry point. Performs some basic operations.

   This function showcases some typical uses for the Rehuel-library.

   \param argc   Number of command line arguments
   \param argv   Command line arguments

   \returns 0 on success, non-zero otherwise.
*/
int main( int argc, char **argv )
{
	double dt = 0.025;
	int method = irk::LOBATTO_IIIA_43;

	double t0 = 0.0;
	double t1 = 50.0;

	if( argc > 1 ){
		int i = 1;
		while( i < argc ){
			const char *arg = argv[i];
			if( strcmp(arg, "-m") == 0 ){
				method = irk::name_to_method( argv[i+1] );
				std::cerr << "Got method code " << method << "\n";
				if( method < 0 ){
					std::cerr << "Unrecognized integrator "
					          << argv[i+1] << "!\n";
					return -2;
				}
				i += 2;
			}else if( strcmp(arg, "-dt") == 0 ){
				dt = std::atof( argv[i+1] );
				i += 2;
			}else{
				std::cerr << "Arg " << argv[i]
				          << " not recognized!\n";
				return -1;
			}
		}
	}

	std::vector<double> times;
	std::vector<arma::vec> ys;
	int odeint_status = solve_test_ode( dt, method, t0, t1, times, ys );

	if( odeint_status ){
		std::cerr << "Something went wrong solving the ODE!\n";
		return -2;
	}

	// Do not output all points but do interpolation on a mesh:
	std::vector<double> t_grid = make_time_grid(t0, t1);
	std::vector<arma::vec> y_interp = interpolate::linear( times, ys,
	                                                       t_grid );

	std::cerr << "Size of time grid: " << t_grid.size() << ".\n";
	std::cerr << "Size of interp. y: " << y_interp[0].size() << ".\n";

	for( std::size_t i = 0; i < t_grid.size(); ++i ){
		std::cout << t_grid[i];
		for( std::size_t j = 0; j < y_interp[i].size(); ++j ){
			std::cout << " " << y_interp[i][j];
		}
		std::cout << "\n";
	}

	return 0;
}
