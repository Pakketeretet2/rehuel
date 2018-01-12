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
#include "rehuel.hpp"
#include "odes.hpp"

#include <cmath>
#include <iostream>


struct exponential
{
	typedef arma::mat jac_type;

	exponential( double w, double a ) : w(w), a(a) {}

	double w, a;

	arma::vec sol( double t )
	{
		return { exp(-a*t)*cos(w*t), exp(-a*t)*sin(w*t) };
	}

	virtual arma::vec fun( double t, const arma::vec &y )
	{
		return { -a * y[0] - w * y[1], -a * y[1] + w * y[0] };
	}

	virtual jac_type jac( double t, const arma::vec &y )
	{
		return { {-a, -w}, {w, -a} };
	}
};


void test_newton()
{
	newton::test_functions::rosenbrock_func r( 1.0, 100.0 );
	arma::vec x0 = { 0.0, 0.0 };
	newton::options opts;
	newton::status stats;
	opts.maxit = 1000000;
	// arma::vec root = newton::broyden_iterate( r, x0, opts, stats );
	arma::vec root = newton::newton_iterate( r, x0, opts, stats );


	std::cerr << "Root is " << root << "\n";

}


void test_odeint( double dt )
{
	double t0 = 0.0;
	double t1 = 100.0;
	int method = irk::CASH_KARP_54;
	irk::solver_coeffs sc = irk::get_coefficients( method );
	irk::solver_options so = irk::default_solver_options();

	newton::options newton_opts;
	so.newton_opts = &newton_opts;

	std::vector<double> times;
	std::vector<arma::vec> ys;
	so.rel_tol = 1e-8;
	so.abs_tol = 1e-7;
	so.max_dt  = 2.0;
	so.store_in_vector_every = 1000;
	so.internal_solver = irk::solver_options::BROYDEN;
	so.adaptive_step_size = false;
	sc.dt = dt;

	exponential func( 5, 1.0 / 25.0 );
	arma::vec y0 = { 1.0, 0.0 };

	int status = irk::odeint( 0.0, t1, sc, so, y0, func, times, ys );
	std::cerr << "Status = " << status << ".\n";

	arma::vec max_err = { 0.0, 0.0 };

	for( std::size_t i = 0; i < times.size(); ++i ){
		double t = times[i];
		const arma::vec yi = ys[i];
		std::cout << t;
		for( std::size_t j = 0; j < yi.size(); ++j ){
			std::cout << " " << yi[j];
		}
		arma::vec sol = func.sol( t );
		std::cout << " " << sol[0] << " " << sol[1] << "\n";

		arma::vec current_err = sol - yi;
		for( std::size_t j = 0; j < yi.size(); ++j ){
			if( std::fabs( current_err[j] ) >
			    std::fabs( max_err[j] ) ){
				max_err[j] = current_err[j];
			}
		}
	}

	std::cerr << "Largest errors are";
	for( std::size_t j = 0; j < max_err.size(); ++j ){
		std::cerr << " " << max_err[j];
	}
	std::cerr << ".\n";
}


int main( int argc, char **argv )
{
	int i = 1;
	double dt = 1e-4;
	while( i < argc ){
		std::string arg = argv[i];
		if( arg == "--dt" ){
			dt = std::stof( argv[i+1] );
			i += 2;
		}else{
			std::cerr << "Argument " << arg << " not recognized!\n";
			return -1;
		}

	}
	test_odeint(dt);

	return 0;
}
