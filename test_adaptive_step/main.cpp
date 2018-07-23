#include <iostream>

#include "rehuel.hpp"
#include "test_equations.hpp"

// Tests adaptive time step on various problems.

template <typename T>
irk::rk_output integrate( T &func, const arma::vec &y0, double t0, double t1,
                          double dt, int met, const std::string &ode_name,
                          irk::solver_options &so )
{
	irk::solver_coeffs  sc = irk::get_coefficients( met );
	irk::rk_output sol = irk::irk_guts( func, t0, t1, y0, so, dt, sc );

	if( sol.status ){
		std::cerr << "Error solving ODE with method "
		          << irk::method_to_name( met ) << "!\n";
		return sol;
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
	return sol;
}



int main( int argc, char **argv )
{

	irk::solver_options so = irk::default_solver_options();

	newton::options newt_opts;

	so.internal_solver = irk::solver_options::NEWTON;
	so.rel_tol = 1e-5;
	so.abs_tol = 1e-4;
	newt_opts.tol = 0.1*so.rel_tol;
	newt_opts.dx_delta = 0.1*so.rel_tol;

	newt_opts.maxit = 10;
	newt_opts.refresh_jac = false; // true;
	so.newton_opts = &newt_opts;
	so.out_interval = 1000000;
	so.verbose_newton = false;
	std::vector<int> methods = { irk::RADAU_IIA_53, irk::RADAU_IIA_32,
	                             irk::RADAU_IIA_95 };
	arma::vec y0 = { 1.0 };

	std::vector<std::vector<double> > times( methods.size() );

	double t0 = 0.0;
	double t1 = 20.0;
	double dt = 1e-4;

	auto add_time = [&times]( int method, double time ){
		switch(method){
			case irk::RADAU_IIA_53:
				times[0].push_back( time );
				break;
			case irk::RADAU_IIA_32:
				times[1].push_back( time );
				break;
			case irk::RADAU_IIA_95:
				times[2].push_back( time );
				break;
			default:
				break;
		} };
	test_equations::exponential exponen( -0.2 );
	for( int method : methods ){
		auto sol = integrate( exponen, y0, t0, t1, dt, method, "exponential", so );
		add_time( method, sol.elapsed_time );
	}

	y0 = { 1.0, 0.0 };

	t0 = 0.0;
	t1 = 100.0;
	dt = 1e-4;

	test_equations::stiff_eq stiff;
	for( int method : methods ){
		auto sol = integrate( stiff, y0, t0, t1, dt, method, "stiff", so );
		add_time( method, sol.elapsed_time );
	}

	test_equations::vdpol vdp( 1e-6 );
	y0 = { 2.0, -0.6 };

	t0 = 0.0;
	t1 = 2.0;
	dt = 1e-6;
	so.rel_tol = 1e-4;
	so.abs_tol = 1e-4;
	newt_opts.tol = 0.01*so.rel_tol;
	newt_opts.dx_delta = 0.01*so.rel_tol;

	for( int method : methods ){

		auto sol = integrate( vdp, y0, t0, t1, dt, method, "vdpol", so );
		add_time( method, sol.elapsed_time );
	}

	so.rel_tol = 1e-6;
	so.abs_tol = 1e-5;
	newt_opts.tol = 0.1*so.rel_tol;
	newt_opts.dx_delta = 0.1*so.rel_tol;

	y0 = { 1.0, 0.0, 0.0 };
	dt = 1e-6;
	t0 = 0.0;
	t1 = 1e11;

	test_equations::rober rob;

	for( int method : methods ){
		auto sol = integrate( rob, y0, t0, t1, dt, method, "rober", so );
		add_time( method, sol.elapsed_time );
	}

	test_equations::bruss brs( 1.0, 2.0 );
	test_equations::bruss brs2( 1.0, 3.5 );

	y0 = { 6.0, 3.0 };
	t0 = 0.0;
	t1 = 1e4;
	dt = 0.1;


	std::cerr << "\n****   Brusselator coming up!  ****\n\n";
	for( int method : methods ){
		auto sol = integrate( brs, y0, t0, t1, dt, method, "bruss", so );
		add_time( method, sol.elapsed_time );
	}
	std::cerr << "\n****   Brusselator2 coming up!  ****\n\n";
	for( int method : methods ){
		auto sol = integrate( brs2, y0, t0, t1, dt, method, "bruss2", so );
		add_time( method, sol.elapsed_time );
	}

	y0 = { 0.0, 0.0, 1.0, 0.0, 0.0, 4.0,
	       0.0, 0.0, 0.0, 1.0, 0.0, 0.0 };

	t0 = 0.0;
	t1 = 10.0;
	dt = 0.1;

	std::cerr << "\n****   Heads up! Tree body time!  ****\n\n";

	so.rel_tol = 1e-3;
	so.abs_tol = 1e-2;
	newt_opts.tol = 0.1*so.rel_tol;
	newt_opts.dx_delta = 0.1*so.rel_tol;
	newt_opts.limit_step = true;
	newt_opts.maxit = 50;

	so.out_interval = 10;

	test_equations::three_body three_b( 1.0, 1.0, 1.0 );
	for( int method : methods ){
		auto sol = integrate( three_b, y0, t0, t1, dt, method, "threeb", so );
		add_time( method, sol.elapsed_time );
	}

	std::vector<std::string> eqs = { "Exponential", "Stiff", "Van der Pol",
	                                 "Robertson", "Brusselator",
	                                 "Brusselator2", "Three body" };
	std::cerr << "    Performance on test problems (ms elapsed):\n";
	std::cerr << "    Equation:  Radau IIA53   Radau IIA32:\n";

	for( std::size_t i = 0; i < eqs.size(); ++i ){
		std::cerr << "      " << eqs[i] << ":";
		for( std::size_t j = 0; j < methods.size(); ++j ){
			std::cerr << " " << times[j][i];
		}
		std::cerr << "\n";
	}


	return 0;
}
