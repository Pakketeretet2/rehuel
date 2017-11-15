#include <armadillo>
#include <iostream>
#include <string>

#include "irk.hpp"


namespace irk {


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
	double sqrt3 = sqrt(3);
	sc.FSAL = false;

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

		case RUNGE_KUTTA_4:
			sc.A = { { 0.0, 0.0, 0.0, 0.0 },
			         { 0.5, 0.0, 0.0, 0.0 },
			         { 0.0, 0.5, 0.0, 0.0 },
			         { 0.0, 0.0, 1.0, 0.0 } };
			sc.b = { one_six, one_third, one_third, one_six };
			sc.c = { 0.0, 0.5, 0.5, 1.0 };

			break;
		case GAUSS_LEGENDRE_65:
			sc.A = { { 0.25, 0.25 - sqrt3/6.0 },
			         { 0.25 + sqrt3/6.0, 0.25 } };
			sc.c = { 0.5 - sqrt3/6.0, 0.5 + sqrt3/6.0 };
			sc.b = { 0.5, 0.5 };
			sc.b2= { 0.5 + 0.5*sqrt3, 0.5 - 0.5*sqrt3 };

			break;
		case RADAU_IIA_32:
			sc.A = { { 5.0 / 12.0, -1.0 / 12.0 },
			         { 3.0 / 4.0,   1.0 / 4.0 } };
			sc.c = { 1.0/3.0, 1.0 };
			sc.b = { 3.0/4.0, 1.0/4.0 };

			break;
		case LOBATTO_IIIA_43:
			sc.A = { {      0.0,     0.0,       0.0 },
			         { 5.0/24.0, 1.0/3.0, -1.0/24.0 },
			         {  1.0/6.0, 2.0/3.0,  1.0/6.0 } };

			sc.c = { 0.0, 0.5, 1.0 };
			sc.b = { 1.0/6.0, 2.0/3.0, 1.0/6.0 };
			sc.b2 = { -0.5, 2.0, -0.5 };

			sc.FSAL = true;

			break;


		case BOGACKI_SHAMPINE_23:
			sc.A  = { {  0.0,  0.0, 0.0, 0.0 },
			          {  0.5,  0.0, 0.0, 0.0 },
			          {  0.0, 0.75, 0.0, 0.0 },
			          { 2.0/9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0 } };
			sc.b  = { 2.0/9.0, 1.0/3.0, 4.0/9.0, 0.0 };
			sc.b2 = { 7.0/24.0, 0.25, 1.0/3.0, 1.0/8.0 };

			sc.c  = { 0.0, 0.5, 0.75, 1.0 };

			sc.FSAL = true;


			break;
		case CASH_KARP_54:
			std::cerr << "CASH_KARP5_4 does not work properly!\n";
			std::terminate();

			sc.A = arma::mat( 6,6 );
			sc.A.zeros( 6,6 );

			sc.A(1,0) =  0.2;
			sc.A(2,0) =  3.0/40.0;
			sc.A(3,0) =  0.3;
			sc.A(4,0) = -11.0/54.0;
			sc.A(5,0) =  1631.0/55296.0;

			sc.A(2,1) =  9.0/40.0;
			sc.A(3,1) = -9.0/10.0;
			sc.A(4,1) =  5.0/2.0;
			sc.A(5,1) =  175.0/512.0;

			sc.A(3,2) =  6.0/5.0;
			sc.A(4,2) = -70.0/27.0;
			sc.A(5,2) =  575.0 / 13824.0;

			sc.A(4,3) =  35.0 / 27.0;
			sc.A(5,3) =  44275.0 / 110592.0;

			sc.A(5,4) =  253.0 / 4096.0;

			sc.b = {37.0 / 378.0, 0.0, 250.0 / 621.0, 125.0 / 594.0,
			        0.0, 512.0 / 1771.0 };
			sc.b2 = {2825.0/27648.0, 0.0, 18575.0 / 48384.0,
			         13525.0 / 55296.0, 277.0 / 14336.0, 0.25 };

			sc.c = { 0.0, 0.2, 0.3, 0.6, 1.0, 7.0/8.0 };


			break;
		case DORMAND_PRINCE_54:
			sc.A = arma::mat(7,7);
			sc.A.zeros( 7,7 );

			sc.A(1,0) =  1.0/5.0;
			sc.A(2,0) =  3.0/4.0;
			sc.A(3,0) =  44.0/45.0;
			sc.A(4,0) =  19372.0 / 6561.0;
			sc.A(5,0) =  9017.0 / 3168.0;
			sc.A(6,0) =  35.0/384.0;

			sc.A(2,1) =  9.0/40.0;
			sc.A(3,1) = -56.0/15.0;
			sc.A(4,1) = -25360.0/2187.0;
			sc.A(5,1) =  -355.0/33.0;
			sc.A(6,1) =  0.0;

			sc.A(3,2) =  32.0/9.0;
			sc.A(4,2) =  64448.0 / 6561.0;
			sc.A(5,2) =  46732.0 / 5247.0;
			sc.A(6,2) =  500.0 / 1113.0;

			sc.A(4,3) = -212.0 / 729.0;
			sc.A(5,3) =  49.0 / 176.0;
			sc.A(6,3) =  125.0/192.0;

			sc.A(5,4) = -5103.0 / 18656.0;
			sc.A(6,4) = -2187.0 / 6784.0;

			sc.A(6,5) =  11.0/84.0;

			sc.c  = {0.0, 0.2, 0.3, 0.8, 8.0/9.0, 1.0, 1.0};
			sc.b  = {35.0 / 34.0, 0.0, 500.0 / 1113.0,
			         125.0 / 192.0, -2187.0 / 6784.0,
			         -2187.0 / 6784.0, 11.0 / 84.0 };
			sc.b2 = {5179.57600, 0.0, 7571.0 / 16695.0,
			         393.0 / 640.0, -92097.0 / 339200.0,
			         187.0/2100.0, 1.0/40.0 };
			sc.FSAL = true;
			break;
	}
	sc.dt = 0.05;
	return sc;
}


solver_options default_solver_options()
{
	solver_options s;
	return s;
}

double get_better_time_step( double dt_old, double error_estimate,
                             const solver_options &opts )
{

	double thing = opts.local_tol / error_estimate;
	double factor = 0.9 * std::min( std::max( 0.3, thing ), 2.0 );
	return dt_old * factor;
}

unsigned int name_to_method( const char *name )
{
	std::string n(name);

	if( n == "explicit_euler" )      return EXPLICIT_EULER;
	if( n == "runge_kutta_4" )       return RUNGE_KUTTA_4;
	if( n == "bogacki_shampine_23" ) return BOGACKI_SHAMPINE_23;
	if( n == "cash_karp_54" )        return CASH_KARP_54;
	if( n == "dormand_prince_54" )   return DORMAND_PRINCE_54;

	if( n == "implicit_euler" )      return IMPLICIT_EULER;
	if( n == "radau_IIA_32" )        return RADAU_IIA_32;
	if( n == "lobatto_IIIA_43" )     return LOBATTO_IIIA_43;
	if( n == "gauss_legendre_65" )   return GAUSS_LEGENDRE_65;

	return -1337;
}


} // namespace irk
