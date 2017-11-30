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
	sc.name = method_to_name( method );

	switch(method){
		default:
			std::cerr << "Method " << method << " not supported!\n";
			break;
		case EXPLICIT_EULER:
			sc.A = { 0.0 };
			sc.b = { 1.0 };
			sc.c = { 0.0 };
			sc.order = 1;
			sc.order2 = 0;

			break;
		case IMPLICIT_EULER:
			sc.A = { 1.0 };
			sc.b = { 1.0 };
			sc.c = { 1.0 };
			sc.order = 1;
			sc.order2 = 0;
			break;

		case IMPLICIT_MIDPOINT:
			sc.A = { 0.5 };
			sc.b = { 1.0 };
			sc.c = { 0.5 };
			sc.order  = 2;
			sc.order2 = 0;

		case RUNGE_KUTTA_4:
			sc.A = { { 0.0, 0.0, 0.0, 0.0 },
			         { 0.5, 0.0, 0.0, 0.0 },
			         { 0.0, 0.5, 0.0, 0.0 },
			         { 0.0, 0.0, 1.0, 0.0 } };
			sc.b = { one_six, one_third, one_third, one_six };
			sc.c = { 0.0, 0.5, 0.5, 1.0 };
			sc.order = 4;
			sc.order2 = 0;
			break;
		case GAUSS_LEGENDRE_43:
			sc.A = { { 0.25, 0.25 - sqrt3/6.0 },
			         { 0.25 + sqrt3/6.0, 0.25 } };
			sc.c = { 0.5 - sqrt3/6.0, 0.5 + sqrt3/6.0 };
			sc.b = { 0.5, 0.5 };
			sc.b2= { 0.5 + 0.5*sqrt3, 0.5 - 0.5*sqrt3 };
			sc.order = 4;
			sc.order2 = 2;
			break;
		case RADAU_IIA_32:
			sc.A = { { 5.0 / 12.0, -1.0 / 12.0 },
			         { 3.0 / 4.0,   1.0 / 4.0 } };
			sc.c = { 1.0/3.0, 1.0 };
			sc.b = { 3.0/4.0, 1.0/4.0 };
			sc.order = 3;
			sc.order2 = 2;
			break;
		case LOBATTO_IIIA_43:
			sc.A = { {      0.0,     0.0,       0.0 },
			         { 5.0/24.0, 1.0/3.0, -1.0/24.0 },
			         {  1.0/6.0, 2.0/3.0,  1.0/6.0 } };

			sc.c = { 0.0, 0.5, 1.0 };
			sc.b = { 1.0/6.0, 2.0/3.0, 1.0/6.0 };
			sc.b2 = { -0.5, 2.0, -0.5 };
			sc.order = 4;
			sc.order2 = 3;
			sc.FSAL = true;

			break;


		case BOGACKI_SHAMPINE_23:
			sc.A  = { {  0.0,  0.0, 0.0, 0.0 },
			          {  0.5,  0.0, 0.0, 0.0 },
			          {  0.0, 0.75, 0.0, 0.0 },
			          { 2.0/9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0 } };
			sc.b  =   { 2.0/9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0 };

			sc.b2 = { 7.0/24.0, 0.25, 1.0/3.0, 1.0/8.0 };

			sc.c  = { 0.0, 0.5, 0.75, 1.0 };

			sc.FSAL = true;
			sc.order = 2;
			sc.order2 = 3;

			break;
		case CASH_KARP_54:
			sc.A = arma::mat( 6,6 );
			sc.A.zeros( 6,6 );

			sc.A(1,0) =  1.0/5.0;
			sc.A(2,0) =  3.0/40.0;
			sc.A(3,0) =  3.0/10.0;
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

			sc.b = {37.0 / 378.0,
			        0.0,
			        250.0 / 621.0,
			        125.0 / 594.0,
			        0.0,
			        512.0 / 1771.0 };
			sc.b2 = {2825.0/27648.0,
			         0.0,
			         18575.0 / 48384.0,
			         13525.0 / 55296.0,
			         277.0 / 14336.0,
			         1.0/4.0 };

			sc.c = { 0.0,
			         0.2,
			         0.3,
			         0.6,
			         1.0,
			         7.0/8.0 };

			sc.order = 5;
			sc.order2 = 4;

			break;
		case DORMAND_PRINCE_54:
			sc.A = arma::mat(7,7);
			sc.A.zeros( 7,7 );

			sc.A(1,0) =  1.0/5.0;
			sc.A(2,0) =  3.0/40.0;
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

			sc.c  = {0.0,
			         0.2,
			         0.3,
			         0.8,
			         8.0/9.0,
			         1.0,
			         1.0};
			sc.b  = {35.0 / 384.0,
			         0.0,
			         500.0 / 1113.0,
			         125.0/192.0,
			         -2187.0 / 6784.0,
			         11.0 / 84.0,
			         0.0 };
			sc.b2 = {5179.0/57600.0,
			         0.0,
			         7571.0 / 16695.0,
			         393.0 / 640.0,
			         -92097.0 / 339200.0,
			         187.0/2100.0,
			         1.0/40.0 };
			sc.order = 5;
			sc.order2 = 4;
			sc.FSAL = false;
			break;
	}

	// Some checks:
	for( std::size_t i = 0; i < sc.c.size(); ++i ){
		double ci = sc.c(i);
		double si = 0.0;
		for( std::size_t j = 0; j < sc.c.size(); ++j ){
			si += sc.A(i,j);
		}
		if( std::fabs( si - ci ) > 1e-5 ){
			std::cerr << "Warning! Mismatch between c and A(i,:) "
			          << "for i = " << i << ", method = "
			          << method_to_name( method ) << "!\n";
		}
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
                             const solver_options &opts,
                             const solver_coeffs &sc, double max_dt )
{
	// |y1 - y2| ~= C1 * dt^(min( sc.order, sc.order2 )) := error_estimate
	// We want this to be opts.rel_tol. So...
	// C1 * dt_new^(min( sc.order1, sc.order2 )) := opts.rel_tol.
	// After some algebra, that becomes this:
	double power  = 1.0 / ( std::min( sc.order, sc.order2 ) + 1.0 );
	double frac   = opts.rel_tol / error_estimate;
	double factor = std::pow( frac, power );
	double scale  = 0.9*std::min( 2.0, factor ); // Increase dt smoothly.
	double dt_new = std::min( dt_old * scale, max_dt );
	return dt_new;

}

int name_to_method( const char *name )
{
	std::string n(name);

	if( n == "explicit_euler" )      return EXPLICIT_EULER;
	if( n == "runge_kutta_4" )       return RUNGE_KUTTA_4;
	if( n == "bogacki_shampine_23" ) return BOGACKI_SHAMPINE_23;
	if( n == "cash_karp_54" )        return CASH_KARP_54;
	if( n == "dormand_prince_54" )   return DORMAND_PRINCE_54;

	if( n == "implicit_euler" )      return IMPLICIT_EULER;
	if( n == "implicit_midpoint" )   return IMPLICIT_MIDPOINT;
	if( n == "radau_IIA_32" )        return RADAU_IIA_32;
	if( n == "lobatto_IIIA_43" )     return LOBATTO_IIIA_43;
	if( n == "gauss_legendre_43" )   return GAUSS_LEGENDRE_43;

	return -1337;
}

const char *method_to_name( int method )
{
	if( method == EXPLICIT_EULER       ) return "explicit_euler";
	if( method == RUNGE_KUTTA_4        ) return "runge_kutta_4";
	if( method == BOGACKI_SHAMPINE_23  ) return "bogacki_shampine_23";
	if( method == CASH_KARP_54         ) return "cash_karp_54";
	if( method == DORMAND_PRINCE_54    ) return "dormand_prince_54";

	if( method == IMPLICIT_EULER       ) return "implicit_euler";
	if( method == IMPLICIT_MIDPOINT    ) return "implicit_midpoint";
	if( method == RADAU_IIA_32         ) return "radau_IIA_32";
	if( method == LOBATTO_IIIA_43      ) return "lobatto_IIIA_43";
	if( method == GAUSS_LEGENDRE_43    ) return "gauss_legendre_43";

	return "";
}



bool verify_solver_options( solver_options &opts )
{
	if( opts.newton_opts ) return true;
	std::cerr << "ERROR! solver_opts @" << &opts << " does not have "
	          << "newton::options set!\n";
	return false;
}


} // namespace irk
