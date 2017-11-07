#include <armadillo>
#include <iostream>

#include "radau.hpp"

namespace radau {


bool verify_solver_coeffs( const solver_coeffs &sc )
{
	auto N = sc.b.size();
	if( N != sc.c.size() || N != sc.A.n_rows || N != sc.A.n_cols ){
		return false;
	}
	return true;
}

solver_coeffs get_coefficients( int method )
{
	solver_coeffs sc;
	double one_third = 1.0/3.0;
	double one_six = 1.0/6.0;
	double sqrt2 = sqrt(2);
	double sqrt3 = sqrt(3);


	switch(method){
		default:
			std::cerr << "Method " << method << " not supported!\n";
			break;
		case EXPLICIT_EULER:
			sc.A = { 0.0 };
			sc.b = { 1.0 };
			sc.c = { 0.0 };

			break;
		case IMPLICIT_EULER:
			sc.A = { 1.0 };
			sc.b = { 1.0 };
			sc.c = { 1.0 };

			break;
		case CLASSIC_RK:
			sc.A = { { 0.0, 0.0, 0.0, 0.0 },
			         { 0.5, 0.0, 0.0, 0.0 },
			         { 0.0, 0.5, 0.0, 0.0 },
			         { 0.0, 0.0, 1.0, 0.0 } };
			sc.b = { one_six, one_third, one_third, one_six };
			sc.c = { 0.0, 0.5, 0.5, 1.0 };

			break;
		case GAUSS_LEGENDRE_2:
			sc.A = { { 0.25, 0.25 - sqrt3/6.0 },
			         { 0.25 + sqrt3/6.0, 0.25 } };
			sc.c = { 0.5 - sqrt3/6.0, 0.5 + sqrt3/6.0 };
			sc.b = { 0.5, 0.5 };
			sc.b2= { 0.5 + 0.5*sqrt3, 0.5 - 0.5*sqrt3 };

			break;
		case RADAU_IIA_2:
			sc.A = { { 5.0 / 12.0, -1.0 / 12.0 },
			         { 3.0 / 4.0,   1.0 / 4.0 } };
			sc.c = { 1.0/3.0, 1.0 };
			sc.b = { 3.0/4.0, 1.0/4.0 };
			break;
		case LOBATTO_IIIA_3:
			sc.A = { {      0.0,     0.0,       0.0 },
			         { 5.0/24.0, 1.0/3.0, -1.0/24.0 },
			         {  1.0/6.0, 2.0/3.0,  1.0/6.0 } };
			sc.c = { 0.0, 0.5, 1.0 };
			sc.b = { 1.0/6.0, 2.0/3.0, 1.0/6.0 };
			sc.b2 = { -0.5, 2.0, -0.5 };

			break;
	}
	sc.dt = 0.05;
	return sc;
}


} // namespace radau
