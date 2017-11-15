#include "interpolate.hpp"
#include "newton.hpp"
#include "odes.hpp"
#include "irk.hpp"


#include <cmath>
#include <iostream>


void test_newton_solvers()
{
	// Test newton and broyden solvers:
	int iters;
	double res;
	arma::vec x0 = { 0.9, 0.9 };
	int status;
	std::cerr << "Testing solvers...\n";

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



int main( int argc, char **argv )
{
	// Do a simple ODE.
	double dt = 0.025;
	int method = irk::LOBATTO_IIIA_43;

	if( argc > 1 ){
		int i = 1;
		while( i < argc ){
			const char *arg = argv[i];
			if( strcmp(arg, "-m") == 0 ){
				method = irk::name_to_method( argv[i+1] );
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

	irk::solver_coeffs sc = irk::get_coefficients( method );
	irk::solver_options s_opts = irk::default_solver_options();

	sc.dt = dt;
	s_opts.local_tol = 1e-6;
	s_opts.internal_solver = irk::solver_options::NEWTON;
	std::vector<double> t;
	std::vector<arma::vec> y;
	arma::vec y0 = { 1.0, 0.0 };
	double t0 = 0.0;
	double t1 = 50.0;

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
	                                   ode, ode_J, t, y );

	// Do not output all points but do interpolation on a mesh:
	std::vector<double> t_grid;
	double tc = t0;
	double dtc = 0.01;
	while( tc < 0.5 ){
		t_grid.push_back( tc );
		tc += dtc;
	}
	dtc = 0.05;
	while( tc < 2.5 ){
		t_grid.push_back( tc );
		tc += dtc;
	}
	dtc = 0.2;
	while( tc < 10 ){
		t_grid.push_back( tc );
		tc += dtc;
	}
	dtc = 0.5;
	while( tc < t1 ){
		t_grid.push_back( tc );
		tc += dtc;
	}
	t_grid.push_back(t1);

	std::vector<arma::vec> y_interp = interpolate::linear( t, y, t_grid );

	for( std::size_t i = 0; i < t_grid.size(); ++i ){
		std::cout << t_grid[i];
		for( std::size_t j = 0; j < y_interp[i].size(); ++i ){
			std::cout << " " << y_interp[i][j];
		}
		std::cout << "\n";
	}

	return 0;
}
