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


arma::vec blue_sky_catastrophe( double t, const arma::vec &yy,
                                double mu, double eps )
{
	arma::vec rhs(3);
	double x = yy[0];
	double y = yy[1];
	double z = yy[2];

	double x2 = x*x;
	double y2 = y*y;
	double z2 = z*z;
	double z3 = z2*z;

	rhs(0) = x*(2 + mu - 10*(x2 + y2)) + z2 + x2 + y2 + 2*y;
	rhs(1) = -z3 - (1+y)*(z2+y2+2*y) - 4*x + mu*y;
	rhs(2) = (1+y)*z2 + x2 - eps;

	return rhs;
}

arma::mat blue_sky_catastrophe_J( double t, const arma::vec &yy,
                                  double mu, double eps )
{
	arma::mat J(3,3);

	double x = yy[0];
	double y = yy[1];
	double z = yy[2];

	double x2 = x*x;
	double y2 = y*y;
	double z2 = z*z;
	double z3 = z2*z;

	J(0,0) = (2 + mu - 10*(x2 + y2)) + 2*x + x*( -20*x);
	J(0,1) = x*(2 + mu - 20*y) + 2*y + 2;
	J(0,2) = 2*z;

	J(1,0) = -4.0;
	J(1,1) =  (z2+y2+2*y) + mu + y*( 2*y + 2 );
	J(1,2) = -3*z2 - (1+y)*2*z;

	J(2,0) = 2*x;
	J(2,1) = z2;
	J(2,2) = 2*z;

	return J;
}


int main( int argc, char **argv )
{
	// Do a simple ODE.
	double dt = 0.25;
	int method = radau::LOBATTO_IIIA_3;

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
	sc.dt = dt;

	std::vector<double> t;
	std::vector<arma::vec> y;
	arma::vec y0 = { 1.0, 1.0, 1.0 };
	double mu = 10.0;
	double eps = 3.0;
	auto ode   = [&mu, &eps]( double t, const arma::vec &y )
		{ return blue_sky_catastrophe( t, y, mu, eps ); };
	auto ode_J = [&mu, &eps]( double t, const arma::vec &y )
		{ return blue_sky_catastrophe_J( t, y, mu, eps); };


	int odeint_status = radau::odeint( 0.0, 500.0, sc, y0, ode, ode_J, t, y );

	for( std::size_t i = 0; i < t.size(); ++i ){
		std::cout << t[i];
		for( std::size_t j = 0; j < y[i].size(); ++j ){
			std::cout << " " << y[i][j];
		}
		std::cout << "\n";
	}

	return 0;
}
