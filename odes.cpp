#include "odes.hpp"

using namespace radau::odes;

namespace radau {

namespace odes {

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

arma::vec brusselator( double t, const arma::vec &yy, double a, double b )
{
	double x  = yy[0];
	double y  = yy[1];
	double x2 = x*x;

	arma::vec f = { a + x2 * y - b*x - x,
	                b*x - x2*y };
	return f;
}

arma::mat brusselator_J( double t, const arma::vec &yy, double a, double b )
{
	double x  = yy[0];
	double y  = yy[1];
	double x2 = x*x;

	arma::mat J = { { 2*x*y - b - 1.0, x2 },
	                { b - 2*x*y, -x2 } };
	return J;
}

} // namespace odes

} // namespace radau
