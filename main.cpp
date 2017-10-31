#include "radau.hpp"
#include "newton.hpp"

#include <cmath>
#include <iostream>

arma::vec test_func( const arma::vec &x )
{
	double y1 = 1.0 - 2*std::sin(x[0])*std::cos(x[1]);
	double y2 = -x[1] - std::exp(-x[0]);
	return arma::vec( {y1, y2} );
}

arma::mat test_J( const arma::vec &x )
{
	auto J = arma::mat( 2, 2 );
	double s1 = std::sin( x[0] );
	double c1 = std::cos( x[0] );
	double s2 = std::sin( x[1] );
	double c2 = std::cos( x[1] );
	J(0,0) = -2.0*c1*c2;
	J(0,1) =  2.0*s1*s2;
	J(1,0) = -std::exp( -x[0] );
	J(1,1) = -1.0;

	return J;
}

arma::vec ode( double t, const arma::vec &y )
{
	return -y;
}

arma::mat ode_J( double t, const arma::vec &v )
{
	arma::mat JJ;
	JJ.eye(1,1);
	return JJ;
}

int main( int argc, char **argv )
{
	// Test newton on a simple function.

	arma::vec x0({ 0.0, 0.0 });
	std::cerr << "x0 = " << x0 << ".\n";
	std::cerr << "F(x0) = " << test_func(x0) << ".\n";
	std::cerr << "J(x0) = " << test_J(x0) << ".\n";

	double tol = 1e-4;
	int maxit = 1000;
	int status = 0;
	double res = 0.0;
	int iters = 0;
	arma::vec xsol = newton::newton_solve( test_func, test_J, x0, tol,
	                                       maxit, status, res, iters );
	if( status == newton::SUCCESS ){
		std::cerr << "Newton iteration successfully converged to "
		          << xsol << " with residual " << res << " after "
		          << iters << " iterations.\n";
	} else {
		std::cerr << "Newton iteration failed to converge within "
		          << maxit << " iterations! Final point was "
		          << xsol << " with residual " << res << "!\n";
	}

	// Do a simple ODE.
	radau::solver_coeffs sc;
	double dt = 0.1;

	sc.b = {1.0};
	sc.c = {1.0};
	sc.A = {1.0};
	sc.dt = dt;
	arma::vec F = { 0.0 };
	arma::mat J = { 0.0 };
	double t = 0.0;
	arma::vec y = {1.0};
	radau::construct_F_and_J( F, J, t, y, sc, ode, ode_J );



	return 0;
}
