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


struct three_body
{
	typedef arma::mat jac_type;

	three_body( double m1, double m2, double m3 )
		: m1r(m1/m3), m2r(m2/m3), m1(m1), m2(m2), m3(m3) {}

	virtual arma::vec fun( double t, const arma::vec &y )
	{
		// y contains ( q0, q1, q2, q3, q4, q5,
		//            ( p0, p1, p2, p3, p4, p5 )
		//
		// (q0, q1, q2, q3, q4, q5) are (x0, y0, x1, y1, x2, y2)

		arma::vec dydt(12);
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

	virtual jac_type jac( double t, const arma::vec &y )
	{
		arma::mat J(12,12);

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


		J.zeros(12,12);

		// [ 0..5 ] is p, [ 6..11 ] is x.
		J(6,6) = -m1*m3 / r13_3 + 3*m1*m3*x3mx1_2 / r13_5
			- m1*m2 / r12_3 + 3*m1*m2*x2mx1_2 / r12_5;

		J(8,8) = -m2*m3 / r23_3 + 3*m2*m3*x3mx2_2 / r12_5
			- m1*m2 / r12_3 + 3*m1*m2*x2mx1_2 / r12_5;

		J(10,10) = -m2*m3 / r23_3 + 3*m2*m3*x3mx2_2 / r23_5
			- m1*m3 / r13_3 + 3*m1*m3*x3mx1_2 / r13_5;

		J(7,7) = -m1*m3 / r13_3 + 3*m1*m3*y3my1_2 / r13_5
			- m1*m2 / r12_3 + 3*m1*m2*y2my1_2 / r12_5;

		J(9,9) = -m2*m3 / r23_3 + 3*m2*m3*y3my2_2 / r12_5
			- m1*m2 / r12_3 + 3*m1*m2*y2my1_2 / r12_5;

		J(11,11) = -m2*m3 / r23_3 + 3*m2*m3*y3my2_2 / r23_5
			- m1*m3 / r13_3 + 3*m1*m3*y3my1_2 / r13_5;

		// Cross terms:
		J(6,7)  =  3*m1*m3*( x3mx1*y3my1 + x2mx1*y2my1 );
		J(6,8)  = m1*m2 / r12_3 - 3*m1*m2*x2mx1_2 / r12_5;
		J(6,9)  = -3*m1*m2*x2mx1*y2my1 / r12_5;
		J(6,10) = m1*m3 / r13_3 - 3*m1*m3*x3mx1_2 / r13_5;
		J(6,11) = -3*m1*m3*x3mx1 * y3my1;

		J(7, 6) = J(6, 7);
		J(7, 8) = -3.0*m1*m2*x2mx1*y2my1 / r12_5;
		J(7, 9) = m1*m2 / r12_3 - 3*m1*m2*y2my1_2 / r12_5;
		J(7,10) = -3.0*m1*m3*x3mx1*y3my1 / r13_5;
		J(7,11) = m1*m3 / r13_3 - 3*m1*m3*y3my1_2 / r13_5;

		J(8, 6) = J(6,8);
		J(8, 7) = J(7,8);
		J(8, 9) = 3*m1*m2*x2mx1*y2my1 / r12_5 + 3*m2*m3*x3mx2*y3my2 / r23_5;
		J(8,10) = m2*m3 / r23_3 - 3*m2*m3*x3mx2_2 / r23_5;
		J(8,11) = -3*m2*m3*x3mx2*y3my2 / r23_5;

		J(9, 6) = J(6,9);
		J(9, 7) = J(7,9);
		J(9, 8) = J(8,9);
		J(9,10) = -3*m2*m3*x3mx2*y3my2 / r23_5;
		J(9,11) = m2*m3 / r23_3 - 3*m2*m3*y3my2_2 / r23_5;


		J(10, 6) = J(6,10);
		J(10, 7) = J(7,10);
		J(10, 8) = J(8,10);
		J(10, 9) = J(9,10);
		J(10,11) = 3*m2*m3*x3mx2*y3my2 / r23_5 + 3*m1*m3*x3mx1*y3my1 / r13_5;


		J(11, 6) = J( 6,11);
		J(11, 7) = J( 7,11);
		J(11, 8) = J( 8,11);
		J(11, 9) = J( 9,11);
		J(11,10) = J(10,11);

		// x to x:
		J(0,0) = 1.0 / m1;
		J(1,1) = 1.0 / m1;
		J(2,2) = 1.0 / m2;
		J(3,3) = 1.0 / m2;
		J(4,4) = 1.0 / m3;
		J(5,5) = 1.0 / m3;

		// x to p and p to x are 0.


		return J;
	}


	double kin_energy( const arma::vec &y )
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

	double pot_energy( const arma::vec &y )
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

	integrator_io::spaced_grid_info spaced_grid;
	spaced_grid.t0 = t0;
	spaced_grid.t1 = t1;
	int Npts = 1001;
	spaced_grid.dt = (t1 - t0) / (Npts - 1.0);
	std::cerr << "Printing sol every " << spaced_grid.dt << " times.\n";

	std::string fname = "ode_";
	fname += irk::method_to_name( method );
	fname += ".dat";
	auto output_mode = integrator_io::integrator_output::SPACED_GRID;
	std::ofstream out_file( fname );
	integrator_io::integrator_output output( output_mode, &out_file,
	                                         10, &std::cerr );
	output.set_spaced_grid( t0, t1, (t1 - t0) / (Npts - 1.0) );
	so.output = &output;

	std::cerr << "Finding largest error for dt = " << dt << " on interval ["
	          << t0 << ", " << t1 << "]\n";

	so.newton_opts = &newton_opts;
	sc.dt = dt;

	so.internal_solver = irk::solver_options::NEWTON;
	so.internal_solver = irk::solver_options::NEWTON;
	so.adaptive_step_size = false;
	so.rel_tol = 1e-12;
	so.abs_tol = 1e-10;

	integrator_io::vector_output vec_out;
	output.set_vector_output( 1, &vec_out );

	newton_opts.tol = 1e-1 * so.rel_tol;
	newton_opts.max_step = 0.0;
	newton_opts.refresh_jac = false;
	newton_opts.maxit = 100000;

	exponential func( 1.0 );
	arma::vec y0 = func.sol( t0 );
	int status = irk::odeint( t0, t1, sc, so, y0, func );
	if( status ){
		std::cerr << "Error solving ODE!\n";
		return;
	}

	// Find largest error:
	double m_abs_err = 0.0;
	double m_rel_err = 0.0;

	for( std::size_t i = 0; i < vec_out.t_vals.size(); ++i ){

		double t = vec_out.t_vals[i];
		double yn = vec_out.y_vals[i][0];
		double ye = func.sol(t)[0];
		double abs_err = std::fabs( yn - ye );
		double rel_err = abs_err == 0 ? 0.0 :
			abs_err / std::min( yn, ye );

		if( abs_err > m_abs_err ) m_abs_err = abs_err;
		if( rel_err > m_rel_err ) m_rel_err = rel_err;
	}

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
	three_body three_bod(m1, m2, m3);
	// y0 = { x0, y0, x1, y1, x2, y2, px0, py0, px1, py1, px2, py2 }.
	arma::vec y0 = { 0.0, 0.0, 1.0, 0.0, 0.0, 4.0,
	                 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 };

	std::cerr << "dt = " << dt << "\n";
	newton_dummy<three_body> dummy_3b(three_bod, 0.0);

	irk::solver_coeffs  sc = irk::get_coefficients( method );
	irk::solver_options so = irk::default_solver_options();
	newton::options no;

	so.internal_solver = irk::solver_options::NEWTON;
	so.adaptive_step_size = adapt_dt;
	so.rel_tol = 1e-4;
	so.abs_tol = 1e-3;

	so.verbosity = 0;
	so.use_newton_iters_adaptive_step = true;
	so.max_dt = 5.0;

	no.maxit = 25;
	no.precondition = true;
	no.tol = 1e-2*so.rel_tol;
	no.limit_step = true;
	no.refresh_jac = true;

	sc.dt = dt;


	so.newton_opts = &no;

	if( !newton::verify_jacobi_matrix( y0, dummy_3b ) ){
		std::cerr << "Jacobi is fishy!\n";
		return;
	}

	integrator_io::integrator_output output;
	integrator_io::vector_output vec_out;
	output.set_vector_output( 100, &vec_out );
	output.set_timestep_output( 100, &std::cerr );

	so.output = &output;

	std::vector<arma::vec> &ys = vec_out.y_vals;
	std::vector<double> &times = vec_out.t_vals;

	std::cerr << "Integrating three-body problem from "
	          << t0 << " to " << t1 << "...\n";
	int status = irk::odeint( t0, t1, sc, so, y0, three_bod );
	std::cerr << "status is " << status << "!\n";

	for( std::size_t i = 0; i < times.size(); ++i ){
		double t = times[i];
		arma::vec yi = ys[i];
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
