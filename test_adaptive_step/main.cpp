#include <iostream>

#include "integrator.hpp"
#include "rehuel.hpp"
#include "test_equations.hpp"

// Tests adaptive time step on various problems.

template <typename T>
void integrate( T &func, const arma::vec &y0, double t0, double t1, double dt,
                int met, const std::string &ode_name, irk::solver_options &so )
{
	//irk::rk_output sol = irk::radau_IIA_53( func, t0, t1, y0, so, dt );
	//irk::rk_output sol = irk::radau_IIA_32( func, t0, t1, y0, so, dt );

	irk::solver_coeffs  sc = irk::get_coefficients( met );
	irk::rk_output sol = irk::irk_guts( func, t0, t1, y0, so, dt, sc );

	if( sol.status ){
		std::cerr << "Error solving ODE with method "
		          << irk::method_to_name( met ) << "!\n";
		return;
	}

	std::string fname = ode_name + "_sol_";
	fname += irk::method_to_name(met);
	fname += ".dat";
	std::ofstream out( fname );
	out << std::setprecision(12);
	for( std::size_t i = 0; i < sol.t_vals.size(); ++i ){
		double ti = sol.t_vals[i];
		out << ti;
		const arma::vec &yi = sol.y_vals[i];
		for( std::size_t j = 0; j < yi.size(); ++j ){
			out << " " << yi[j];
		}
		out << "\n";
	}
}



int main( int argc, char **argv )
{

	irk::solver_options so = irk::default_solver_options();

	newton::options newt_opts;

	so.internal_solver = irk::solver_options::NEWTON;
	so.rel_tol = 1e-5;
	so.abs_tol = 1e-4;
	newt_opts.tol = 0.1*so.rel_tol;
	newt_opts.maxit = 10;
	newt_opts.refresh_jac = true;
	so.newton_opts = &newt_opts;
	so.out_interval = 0;
	std::vector<int> methods = { irk::RADAU_IIA_53, irk::RADAU_IIA_32 };
	arma::vec y0 = { 1.0 };

	double t0 = 0.0;
	double t1 = 20.0;
	double dt = 1e-4;

	test_equations::exponential exponen( -0.2 );
	for( int method : methods ){
		integrate( exponen, y0, t0, t1, dt, method, "exponential", so );
	}

	y0 = { 1.0, 0.0 };

	t0 = 0.0;
	t1 = 100.0;
	dt = 1e-4;

	test_equations::stiff_eq stiff;

	std::cerr << "\n****   Stiff equation coming up!  ****\n\n";
	for( int method : methods ){
		integrate( stiff, y0, t0, t1, dt, method, "stiff", so );
	}

	so.verbose_newton = true;

	test_equations::vdpol vdp( 1e-6 );
	y0 = { 2.0, -0.6 };

	t0 = 0.0;
	t1 = 2.0;
	dt = 1e-6;
	so.rel_tol = 1e-4;
	so.abs_tol = 1e-4;
	newt_opts.tol = 0.01*so.rel_tol;


	std::cerr << "\n****   Van der Pol coming up!  ****\n\n";
	for( int method : methods ){

		integrate( vdp, y0, t0, t1, dt, method, "vdpol", so );
	}

	so.verbose_newton = false;
	so.out_interval = 10;

	so.rel_tol = 1e-6;
	so.abs_tol = 1e-5;
	newt_opts.tol = 0.1*so.rel_tol;
	y0 = { 1.0, 0.0, 0.0 };
	dt = 1.0;
	t0 = 0.0;
	t1 = 1e6;

	test_equations::rober rob;
	std::cerr << "\n****   Robertson coming up!  ****\n\n";
	for( int method : methods ){
		integrate( rob, y0, t0, t1, dt, method, "rober", so );
	}

	test_equations::bruss brs( 1.0, 2.0 );
	test_equations::bruss brs2( 1.0, 3.5 );

	y0 = { 6.0, 3.0 };
	t0 = 0.0;
	t1 = 1e4;
	dt = 0.1;


	std::cerr << "\n****   Brusselator coming up!  ****\n\n";
	for( int method : methods ){
		integrate( brs, y0, t0, t1, dt, method, "bruss", so );
	}
	std::cerr << "\n****   Brusselator2 coming up!  ****\n\n";
	for( int method : methods ){
		integrate( brs2, y0, t0, t1, dt, method, "bruss2", so );
	}

	y0 = { 0.0, 0.0, 1.0, 0.0, 0.0, 4.0,
	       0.0, 0.0, 0.0, 1.0, 0.0, 0.0 };

	t0 = 0.0;
	t1 = 10.0;
	dt = 0.1;

	std::cerr << "\n****   Heads up! Tree body time!  ****\n\n";

	so.out_interval = 10000;
	test_equations::three_body three_b( 1.0, 1.0, 1.0 );
	for( int method : methods ){
		integrate( three_b, y0, t0, t1, dt, method, "threeb", so );
	}



	return 0;
}
