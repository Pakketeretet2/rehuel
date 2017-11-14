#include "radau.hpp"
#include "odes.hpp"
#include "newton.hpp"

#include <cmath>
#include <iostream>



int main( int argc, char **argv )
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


	// Do a simple ODE.
	double dt = 0.025;
	int method = radau::LOBATTO_IIIA_43;
	method = radau::GAUSS_LEGENDRE_65;
	method = radau::DORMAND_PRINCE5_4;

	opts.time_internals = false;

	if( argc > 1 ){
		int i = 1;
		while( i < argc ){
			const char *arg = argv[i];
			if( strcmp(arg, "-m") == 0 ){
				method = std::atoi( argv[i+1] );
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

	radau::solver_coeffs sc = radau::get_coefficients( method );
	radau::solver_options s_opts = radau::default_solver_options();

	sc.dt = dt;
	s_opts.local_tol = 1e-3;

	std::vector<double> t;
	std::vector<arma::vec> y;
	arma::vec y0 = { 0.9, 1.0, 1.1 };
	double t0 = 0.0;
	double t1 = 50.0;

	double mu = 10.0;
	double eps = 3.0;
	auto ode   = [&mu, &eps]( double t, const arma::vec &y )
		{ return radau::odes::blue_sky_catastrophe( t, y, mu, eps ); };
	auto ode_J = [&mu, &eps]( double t, const arma::vec &y )
		{ return radau::odes::blue_sky_catastrophe_J( t, y, mu, eps); };

	int odeint_status = radau::odeint( t0, t1, sc, s_opts, y0,
	                                   ode, ode_J, t, y );

	for( std::size_t i = 0; i < t.size(); ++i ){
		std::cout << t[i];
		for( std::size_t j = 0; j < y[i].size(); ++j ){
			std::cout << " " << y[i][j];
		}
		std::cout << "\n";
	}

	return 0;
}
