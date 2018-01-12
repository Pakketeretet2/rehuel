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

	exponential( double tau ) : inv_tau(1.0 / tau) {}

	double inv_tau;

	arma::vec sol( double t )
	{
		return { exp(-t * inv_tau) };
	}

	virtual arma::vec fun( double t, const arma::vec &y )
	{
		return { -inv_tau * y[0] };
	}

	virtual jac_type jac( double t, const arma::vec &y )
	{
		return { -inv_tau };
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




int main( int argc, char **argv )
{
	int i = 1;
	int method = irk::IMPLICIT_EULER;
	double dt = 1e-4;

	double t0 = 0.0;
	double t1 = 10.0;

	while( i < argc ){
		std::string arg = argv[i];
		if( arg == "--dt" ){
			dt = std::stof( argv[i+1] );
			i += 2;
		}else if( arg == "--method" ){
			method = irk::name_to_method( argv[i+1] );
			i += 2;
		}else if( arg == "--t0" ){
			t0 = std::stof( argv[i+1] );
			i += 2;
		}else if( arg == "--t1" ){
			t1 = std::stof( argv[i+1] );
			i += 2;

		}else{
			std::cerr << "Argument " << arg << " not recognized!\n";
			return -1;
		}
	}

	// Solve exponential ODE, check error:
	newton::options newton_opts;
	irk::solver_options so = irk::default_solver_options();
	irk::solver_coeffs  sc = irk::get_coefficients( method );

	std::cerr << "Finding largest error for dt = " << dt << " on interval ["
	          << t0 << ", " << t1 << "]\n";

	so.newton_opts = &newton_opts;
	sc.dt = dt;

	std::vector<double> times;
	std::vector<arma::vec> ys;

	std::ofstream timestep_file( "timesteps.dat" );

	so.internal_solver = irk::solver_options::BROYDEN;
	so.adaptive_step_size = false;
	so.rel_tol = 1e-12;
	so.abs_tol = 1e-10;
	so.timestep_info_out_interval = 1;
	so.timestep_out = &timestep_file;

	newton_opts.tol = 1e-1 * so.rel_tol;
	newton_opts.max_step = 1.0;
	newton_opts.refresh_jac = false;
	newton_opts.maxit = 100000;

	exponential func( 1.0 );
	arma::vec y0 = { 1.0 };
	int status = irk::odeint( t0, t1, sc, so, y0, func, times, ys );

	// Find largest error:
	double m_abs_err = 0.0;
	double m_rel_err = 0.0;
	for( std::size_t i = 0; i < times.size(); ++i ){
		double t = times[i];
		double yn = ys[i][0];
		double ye = func.sol( t )[0];
		double abs_err = std::fabs( yn - ye );
		double rel_err = abs_err / std::min( yn, ye );

		if( abs_err > m_abs_err ) m_abs_err = abs_err;
		if( rel_err > m_rel_err ) m_rel_err = rel_err;
	}

	std::cout << dt << " " << m_abs_err << " " << m_rel_err << "\n";


	return 0;
}
