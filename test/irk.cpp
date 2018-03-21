// Tests parts of the IRK methods.

#define ARMA_USE_CXX11
#include <armadillo>

#include "catch.hpp"

#include "irk.hpp"

struct dahlquist
{
	dahlquist( double l ) : l(l) {}

	typedef arma::mat jac_type;

	arma::vec fun( double t, const arma::vec &y )
	{
		return l*y;
	}

	jac_type jac( double t, const arma::vec &y )
	{
		arma::mat J(1,1);
		J = l;
		return J;
	}

	double l;
};
