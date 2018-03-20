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
#include "cpparser.hpp"
#include "integrator_io.hpp"
#include "rehuel.hpp"
#include "odes.hpp"
#include "test_equations.hpp"

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


template <typename functor>
struct newton_dummy
{
	typedef typename functor::jac_type jac_type;

	newton_dummy( functor &f, double t ) : func(f), t(t) {}

	arma::vec fun( const arma::vec x )
	{ return func.fun(t,x); }

	typename functor::jac_type jac( const arma::vec x )
	{ return func.jac(t,x); }

	double t;
	functor &func;
};


void test_newton()
{
	newton::test_functions::rosenbrock_func r( 1.0, 100.0 );
	arma::vec x0 = { 0.0, 0.0 };
	newton::options opts;
	newton::status stats;
	opts.maxit = 1000000;
	arma::vec root = newton::broyden_iterate( r, x0, opts, stats );
	std::cerr << "Broyden root is " << root << "\n";
	root = newton::newton_iterate( r, x0, opts, stats );
	std::cerr << "Newton root is " << root << "\n";

}


void test_ode_rk( int method, double t0, double t1, double dt, bool const_jac )
{
	// Solve exponential ODE, check error:
	newton::options newton_opts;
	irk::solver_options so = irk::default_solver_options();
	irk::solver_coeffs  sc = irk::get_coefficients( method );

	std::string fname = "ode_";
	fname += irk::method_to_name( method );
	fname += ".dat";
	std::ofstream out_file( fname );

	std::cerr << "Finding largest error for dt = " << dt << " on interval ["
	          << t0 << ", " << t1 << "]\n";

	so.newton_opts = &newton_opts;

	so.internal_solver = irk::solver_options::NEWTON;
	so.internal_solver = irk::solver_options::NEWTON;
	so.adaptive_step_size = false;
	so.rel_tol = 1e-12;
	so.abs_tol = 1e-10;


	newton_opts.tol = 1e-1 * so.rel_tol;
	newton_opts.max_step = 0.0;
	newton_opts.refresh_jac = false;
	newton_opts.maxit = 100000;

	exponential func( 1.0 );
	arma::vec y0 = func.sol( t0 );
	irk::rk_output sol  = irk::odeint( func, t0, t1, y0, so, dt );
	if( sol.status ){
		std::cerr << "Error solving ODE!\n";
		return;
	}

	// Find largest error:
	double m_abs_err = 0.0;
	double m_rel_err = 0.0;


	std::cout << dt << " " << m_abs_err << " " << m_rel_err << "\n";
}


void test_ode_multistep( int method, int order,
                         double t0, double t1, double dt )
{
	// Solve exponential ODE, check error:
	newton::options newton_opts;

	std::vector<double> t_vals;
	std::vector<arma::vec> y_vals;

	newton_opts.tol = 1e-14;
	newton_opts.max_step = 1.0;
	newton_opts.refresh_jac = false;
	newton_opts.maxit = 100000;

	exponential func( 1.0 );
	arma::vec y0 = { 1.0 };

	// Integrate with multistep method.
	multistep::solver_coeffs sc =
		multistep::get_coefficients( method, order );
	multistep::solver_options solver_opts
		= multistep::default_solver_options();

	solver_opts.newton_opts = &newton_opts;

	sc.dt = dt;

	int status = multistep::odeint( t0, t1, sc, solver_opts, y0, func );
	if( status ){
		std::cerr << "Error " << status << " while solving ODE.\n";
	}

	// Find largest error:
	double m_abs_err = 0.0;
	double m_rel_err = 0.0;
	for( std::size_t i = 0; i < t_vals.size(); ++i ){
		double t = t_vals[i];
		double yn = y_vals[i][0];
		double ye = func.sol( t )[0];
		double abs_err = std::fabs( yn - ye );
		double rel_err = abs_err / std::min( yn, ye );

		if( abs_err > m_abs_err ) m_abs_err = abs_err;
		if( rel_err > m_rel_err ) m_rel_err = rel_err;
	}

	std::cerr << "Largest error for order " << order << ": "
	          << m_abs_err << " " << m_rel_err << "\n";
	std::cout << dt << " " << m_rel_err << " " << m_abs_err << "\n";

}


void test_three_body( int method, double t0, double t1, double dt,
                      double m1, double m2, double m3,
                      bool adapt_dt, bool use_newton )
{
	test_equations::three_body three_bod(m1, m2, m3);
	// y0 = { x0, y0, x1, y1, x2, y2, px0, py0, px1, py1, px2, py2 }.
	arma::vec y0 = { 0.0, 0.0, 1.0, 0.0, 0.0, 4.0,
	                 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 };

	std::cerr << "dt = " << dt << "\n";
	newton_dummy<test_equations::three_body> dummy_3b(three_bod, 0.0);

	irk::solver_coeffs  sc = irk::get_coefficients( method );
	irk::solver_options so = irk::default_solver_options();
	newton::options no;

	so.internal_solver = irk::solver_options::NEWTON;
	so.rel_tol = 1e-4;
	so.abs_tol = 1e-3;

	so.out_interval = 10;
	no.maxit = 100;
	no.tol = 0.7*so.rel_tol;

	so.newton_opts = &no;

	if( !newton::verify_jacobi_matrix( y0, dummy_3b ) ){
		std::cerr << "Jacobi is fishy!\n";
		return;
	}

	irk::rk_output sol = irk::radau_IIA_53( three_bod, t0, t1, y0, so );

	std::cerr << "status is " << sol.status << "!\n";
	for( std::size_t i = 0; i < sol.t_vals.size(); ++i ){
		double t = sol.t_vals[i];
		arma::vec yi = sol.y_vals[i];

		// Calculate potential and kinetic energy:

		double T = three_bod.kin_energy( yi );
		double U = three_bod.pot_energy( yi );

		std::cout << t << " " << T << " " << U << " " << U+T;
		for( std::size_t j = 0; j < yi.size(); ++j ){
			std::cout << " " << yi[j];
		}
		std::cout << "\n";
	}
}


void test_cyclic_buffer()
{
	cyclic_buffer<double> cb(5);
	for( double t = 0; t < 12.0; t += 1.0 ) {
		cb.push_back(t);
		std::cerr << "Values:";
		for( std::size_t i = 0; i < cb.size(); ++i ){
			std::cerr << " (" << i << ", " << cb[i] << ")";
		}
		std::cerr << "\n";
	}
}


void print_all_methods()
{
	std::vector<std::string> methods = irk::all_method_names();
	for( const std::string &s : methods  ){
		std::cout << s << "\n";
	}
}


int main( int argc, char **argv )
{

	std::string method_str = irk::method_to_name( irk::IMPLICIT_EULER );

	double dt = 1e-4;
	double t0 = 0.0;
	double t1 = 10.0;

	bool test_exp = false;
	bool test_newt = false;
	bool show_all_methods = false;
	bool test_three_bod = false;

	double m1 = 1.0;
	double m2 = 1.0;
	double m3 = 1.0;

	bool adaptive_step = false;
	bool use_newton = false;
	bool test_mstep = false;
	bool test_cyc_buff = false;
	bool const_jac = false;

	int multistep_order = 2;


	cpparser parser( "rehuel", "A driver for the Rehuel library" );
	parser.add_switch( "", "test-exponential", false,
	                   "Tests given method on an exponential function." );
	parser.add_switch( "", "test-newton", false,
	                   "Test Newton library on Rosenbrock's function." );
	parser.add_switch( "", "print-all-methods", false,
	                   "Prints all methods available in Rehuel." );
	parser.add_switch( "", "test-multistep", false,
	                   "Test multistep methods on exponential function." );

	parser.add_option( "m", "method", true,
	                   method_str, "ODE method to use." );
	parser.add_option( "", "t0", true,
	                   t0, "Initial time for the ODE" );
	parser.add_option( "", "t1", true,
	                   t1, "Final time for the ODE." );
	parser.add_option( "", "dt", true,
	                   dt, "Time step size to use." );
	parser.add_switch( "", "test-three-body", false,
	                   "Runs a three-body problem." );
	parser.add_switch( "", "use-newton", false,
	                   "Use Newton's instead of Broyden's method." );

	parser.add_switch( "", "adaptive-step", false,
	                   "Use an adaptive time step size." );
	parser.add_option( "", "m1", true, m1,
	                   "Mass of first body in three body problem." );
	parser.add_option( "", "m2", true, m2,
	                   "Mass of second body in three body problem." );
	parser.add_option( "", "m3", true, m3,
	                   "Mass of third body in three body problem." );
	parser.add_switch( "", "test-cyclic-buffer", false,
	                   "Tests the cyclic buffer class." );

	parser.add_option( "", "multistep-order", true, multistep_order,
	                   "Sets the order of the multistep method." );
	parser.add_switch( "", "constant-jacobi-matrix", false,
	                   "Approximate the Jacobi matrix as constant each stage." );


	parser.parse_cmd_line( argc, argv );

	int parser_status = parser.option_by_long( "dt", dt );
	parser_status |= parser.option_by_long( "t0", t0 );
	parser_status |= parser.option_by_long( "t1", t1 );
	parser_status |= parser.option_by_long( "method", method_str );
	parser_status |= parser.option_by_long( "test-exponential", test_exp );
	parser_status |= parser.option_by_long( "test-newton", test_newt );
	parser_status |= parser.option_by_long( "print-all-methods", show_all_methods );

	parser_status |= parser.option_by_long( "adaptive-step", adaptive_step );
	parser_status |= parser.option_by_long( "test-three-body", test_three_bod );

	parser_status |= parser.option_by_long( "m1", m1 );
	parser_status |= parser.option_by_long( "m2", m2 );
	parser_status |= parser.option_by_long( "m3", m3 );

	parser_status |= parser.option_by_long( "use-newton", use_newton );
	parser_status |= parser.option_by_long( "test-multistep", test_mstep );

	parser_status |= parser.option_by_long( "test-cyclic-buffer", test_cyc_buff );
	parser_status |= parser.option_by_long( "multistep-order", multistep_order );
	parser_status |= parser.option_by_long( "constant-jacobi-matrix", const_jac );



	if( parser_status ){
		std::cerr << "Something went wrong parsing args!\n";
		return -1;
	}

	if( test_cyc_buff ) test_cyclic_buffer();

	if( adaptive_step ){
		std::cerr << "Using adaptive step!\n";
	}else{
		std::cerr << "Not using adaptive step!\n";
	}

	if( test_newt ) test_newton();

	if( test_exp ){
		int method = irk::name_to_method( method_str );
		std::cerr << "Method is " << method_str << ".\n";
		test_ode_rk( method, t0, t1, dt, const_jac );
	}

	if( show_all_methods ) print_all_methods();

	if( test_three_bod ){
		int method = irk::name_to_method( method_str );
		test_three_body( method, t0, t1, dt,
		                 m1, m2, m3, adaptive_step, use_newton );
	}

	if( test_mstep ){
		std::cerr << "Testing multistep with method "
		          << method_str << "\n";
		int method = multistep::name_to_method( method_str );
		std::cerr << "Method is "
		          << multistep::method_to_name(method) << ".\n";
		test_ode_multistep( method, multistep_order, t0, t1, dt );
	}

	return 0;
}
