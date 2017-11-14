#include "newton.hpp"

double newton::test_functions::rosenbrock_f( const arma::vec &x,
                                             double a, double b )
{
	double ap = a - x[0];
	double bp = x[1] - x[0]*x[0];
	return ap*ap + b*bp*bp;
}

arma::vec newton::test_functions::rosenbrock_F( const arma::vec &x,
                                                double a, double b )
{
	double ap = a - x[0];
	double bp = x[1] - x[0]*x[0];

	return { -2*ap - 4*b*bp*x[0], 2*b*bp };

}

arma::mat newton::test_functions::rosenbrock_J( const arma::vec &x,
                                                double a, double b )
{
	double ap = a - x[0];
	double bp = x[1] - x[0]*x[0];

	return { {2 + 4*b*bp - 4*b*x[0]*x[0], 4*b*x[0]},
	         { -4*b*x[0], 2*b } };

	

}


