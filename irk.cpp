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
	if( N == 0 ) return false;

	return true;
}


solver_coeffs get_coefficients( int method )
{
	solver_coeffs sc;
	double one_third = 1.0/3.0;
	double one_six = 1.0/6.0;

	double sqrt3 = sqrt(3.0);
	double sqrt5 = sqrt(5.0);
	double sqrt6 = sqrt(6.0);
	double sqrt15 = sqrt(15.0);


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


		case BOGACKI_SHAMPINE_32:
			sc.A  = { {  0.0,  0.0, 0.0, 0.0 },
			          {  0.5,  0.0, 0.0, 0.0 },
			          {  0.0, 0.75, 0.0, 0.0 },
			          { 2.0/9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0 } };
			sc.b  =   { 2.0/9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0 };

			sc.b2 = { 7.0/24.0, 0.25, 1.0/3.0, 1.0/8.0 };

			sc.c  = { 0.0, 0.5, 0.75, 1.0 };

			sc.FSAL = true;
			sc.order  = 3;
			sc.order2 = 2;

			break;

		case CASH_KARP_54:
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
			sc.FSAL = true;
			break;

		case FEHLBERG_54:
			sc.A.zeros( 6,6 );
			sc.A(1,0) =  1.0/4.0;
			sc.A(2,0) =  3.0/32.0;
			sc.A(3,0) =  1932.0/2197.0;
			sc.A(4,0) =  439.0/216.0;
			sc.A(5,0) = -8.0/27.0;

			sc.A(2,1) =  9.0/32.0;
			sc.A(3,1) = -7200.0/2197.0;
			sc.A(4,1) = -8.0;
			sc.A(5,1) =  2.0;

			sc.A(3,2) =  7296.0/2197.0;
			sc.A(4,2) =  3680.0/513.0;
			sc.A(5,2) = -3544.0/2565.0;

			sc.A(4,3) = -845.0/4104.0;
			sc.A(5,3) =  1859.0/4104.0;

			sc.A(5,4) = -11.0/40.0;

			sc.c = { 0.0,
			         0.25,
			         0.375,
			         12.0/13.0,
			         1.0,
			         0.5 };

			sc.b  = { 16.0/135.0, 0.0, 6656.0/12825.0,
			          28561.0 / 56430.0, -9.0/50.0, 2.0/55.0 };
			sc.b2 = { 25.0 / 216.0, 0.0, 1408.0/2565.0,
			          2197.0/4104.0, -1.0/5.0, 0.0 };

			sc.order  = 5;
			sc.order2 = 4;

			break;

			// IMPLICIT METHODS:

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
			break;

		case LOBATTO_IIIA_21:
			sc.A = { {0.0, 0.0 },
			         {0.5, 0.5 } };
			sc.b = { 0.5, 0.5 };
			sc.c = { 1.0, 0.0 };
			sc.b2 = { 0.25, 0.75 };
			sc.order  = 2;
			sc.order2 = 1;
			break;

		case LOBATTO_IIIC_21:
			sc.A  = { {0.5, -0.5},
			          {0.5,  0.5 } };
			sc.b  = { 0.5, 0.5 };
			sc.b2 = { 3.0/4.0, 1.0/4.0 };
			sc.c  = { 0.0, 1.0 };
			sc.order = 2;
			sc.order2 = 1;

			break;


		case RADAU_IA_31:
			sc.A = { { 1.0 / 4.0, -1.0 / 4.0  },
			         { 1.0 / 4.0,  5.0 / 12.0 } };

			sc.c = { 0.0, 2.0/3.0 };
			sc.b = { 1.0/4.0, 3.0/4.0 };
			sc.b2 = { 0.5, 0.5 };


			sc.order = 3;
			sc.order2 = 1;

			break;

		case RADAU_IIA_31:

			// Try adding an explicit Euler stage?

			sc.A = { { 5.0 / 12.0, -1.0 / 12.0 },
			         { 3.0 / 4.0,   1.0 / 4.0 } };
			sc.c = { 1.0/3.0, 1.0 };
			sc.b = { 3.0/4.0, 1.0/4.0 };
			sc.b2 = { 5.0/8.0, 3.0/8.0 };

			sc.order = 3;
			sc.order2 = 1;
			break;


		case LOBATTO_IIIA_42:

			sc.A = { {      0.0,     0.0,       0.0 },
			         { 5.0/24.0, 1.0/3.0, -1.0/24.0 },
			         {  1.0/6.0, 2.0/3.0,  1.0/6.0 } };

			sc.c = { 0.0, 0.5, 1.0 };
			sc.b = { 1.0/6.0, 2.0/3.0, 1.0/6.0 };
			sc.b2 = { -0.5, 2.0, -0.5 };
			sc.order = 4;
			sc.order2 = 2;
			sc.FSAL = true;

			break;

		case LOBATTO_IIIC_42:

			sc.A = { { 1.0/6.0, -1.0/3.0, 1.0/6.0 },
			         { 1.0/6.0, 5.0/12.0, -1.0/12.0 },
			         { 1.0/6.0, 2.0/3.0, 1.0/6.0 } };
			sc.b = { 1.0/6.0, 2.0/3.0, 1.0/6.0 };
			sc.b2 = { -0.5, 2.0, -0.5 };
			sc.c = { 0.0, 0.5, 1.0 };
			sc.order = 4;
			sc.order2 = 2;

			break;


		case GAUSS_LEGENDRE_42:

			sc.A = { { 0.25, 0.25 - sqrt3/6.0, 0.0 },
			         { 0.25 + sqrt3/6.0, 0.25, 0.0 },
			         { 0.0, 0.0, 0.0 } };
			sc.c = { 0.5 - sqrt3/6.0, 0.5 + sqrt3/6.0, 0.0 };
			sc.b = { 0.5, 0.5, 0.0 };
			sc.b2= { (3*sqrt3 + 1)/12.0, (7-sqrt3)/12.0, (2-sqrt3)/6.0 };
			sc.order = 4;
			sc.order2 = 2;
			break;


		case RADAU_IA_53:

			sc.A = { { 0, 0, 0, 0 },
			         { 0, 1.0/9.0, (-1 - sqrt6)/18.0, (-1 + sqrt6)/18.0},
			         { 0, 1.0/9.0, (88.0 + 7*sqrt6)/360.0, (88 - 43*sqrt6)/360.0 },
			         { 0, 1.0/9.0, (88 + 43*sqrt6)/360.0, (88.0 - 7*sqrt6)/360.0 } };
			sc.c  = { 0.0, 0.0, (6.0 - sqrt6)/10.0, (6.0 + sqrt6)/10.0 };
			sc.b  = { 0, 1.0/9.0, (16.0 + sqrt6)/36.0, (16.0 - sqrt6)/36.0 };
			sc.b2 = { 5.0/2.0, -43/18.0, (sqrt(6)+16.0)/36.0, (16-sqrt(6))/36.0 };

			sc.order = 5;
			sc.order2 = 3;

			break;
		case RADAU_IIA_53:

			sc.A = { { 0, 0, 0, 0 },
			         { 0, (88 - 7*sqrt6)/360.0, (296 - 169*sqrt6)/1800.0, (-2+3*sqrt6)/225.0 },
			         { 0, (296 + 169*sqrt6)/1800.0, (88 + 7*sqrt6)/360.0, (-2-3*sqrt6)/225.0 },
			         { 0, (16.0 - sqrt6)/36.0, (16 + sqrt6)/36.0, 1.0 / 9.0 } };
			sc.c  = {  0, (4.0-sqrt6)/10.0, (4.0+sqrt6) / 10.0, 1.0 };
			sc.b  = {  0, (16 - sqrt6)/36.0, (16 + sqrt6)/36.0, 1.0 / 9.0 };
			sc.b2 = { 5/2.0, -(23*sqrt(6) + 7)/18.0, (23*sqrt(6) - 7)/18.0, -13/18.0 };
			sc.order = 5;
			sc.order2 = 3;

			break;

			// Up to here everything is nicely paired/embedded.
		case GAUSS_LEGENDRE_62:
			sc.A = { { 5.0/36.0, 2.0/9.0 - sqrt15 / 15.0, 5.0/36.0 - sqrt15 / 30.0, 0 },
			         { 5.0/36.0 + sqrt15 / 24.0, 2.0/9.0, 5.0/36.0 - sqrt15 / 24.0, 0 },
			         { 5.0/36.0 + sqrt15 / 30.0, 2.0/9.0 + sqrt15 / 15.0, 5.0/36.0, 0 },
			         { 0, 0, 0, 0 } };
			sc.b = { 5.0/18.0, 4.0/9.0, 5.0/18.0, 0 };
			sc.c = { 0.5 - sqrt15/10.0, 0.5, 0.5 + sqrt15/10.0, 0.0 };

			sc.b2 = {2/9.0, 5/9.0, 2/9.0, 0.0 };
			sc.order = 6;
			sc.order2 = 2;

			break;


			// Up to here everything is nicely paired/embedded.
		case LOBATTO_IIIC_63:{
			double a1 = 1.0 / 12.0;
			double a2 = sqrt5/12.0;
			double a3 = 0.25;
			double a4 = 1.0/6.0;
			double a5 = sqrt5/60.0;
			sc.A = { { a1, -a2, a2, -a1 },
			         { a1, a3, a4 - 7*a5, a5 },
			         { a1, a4 + 7*a5, a3, -a5 },
			         { a1, 5*a1, 5*a1, a1 } };
			sc.b = { a1, 5*a1, 5*a1, a1 };
			sc.b2 = { 4.0/3.0, (5 - 15*sqrt5)/12.0, (5 + 15*sqrt5)/12.0, -7.0/6.0 };

			sc.c = { 0.0, a1 + a3 + a4 - 7*a5 + a5,
			         a1 + a4 + 7*a5  + a3 - a5, 1.0 };
			sc.order = 6;
			sc.order2 = 3;

			break;
		}


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



double get_better_time_step( double dt_n, double dt_nm, double err,
                             double old_err, double tol,
                             int newton_iters, int n_rejected,
                             const solver_options &opts,
                             const solver_coeffs &sc, double max_dt )
{
	// From "Stiff differential equations solved by Radau methods.
	old_err = std::max( 2e-16, old_err );
	err     = std::max( 2e-16, err );

	double min_order = std::min( sc.order, sc.order2 );
	double inv_err = 1.0 / err;
	double fac = 0.9 / sqrt( 1.0 + newton_iters );
	fac /= static_cast<double>( 1.0 + n_rejected );
	double err_frac = old_err / err;
	double pow = 1.0 / ( 1.0 + min_order );
	double dt_frac = dt_n / dt_nm;

	double scale_27 = std::pow( inv_err, pow );
	double scale_28 = scale_27 * std::pow( err_frac, pow ) * dt_frac;

	double min_scale = std::min( scale_27, scale_28 );
	/*
	std::cerr << "    Rehuel: (err, old_err) = ( " << err << ", " << old_err << ")\n";
	std::cerr << "    Rehuel: err_frac = " << err_frac << ", pow = "
	          << pow << ", min_order = " << min_order << ".\n";
	std::cerr << "    Rehuel: (err, old_err) = ( " << err << ", " << old_err << ")\n";
	std::cerr << "    Rehuel: scale_27 = " << scale_27 << ", scale_28 = " << scale_28 << "\n";
	*/

	double dt_new = fac * dt_n * min_scale;
	// if( dt_new >
	return dt_new;

}


bool verify_solver_options( solver_options &opts )
{
	if( opts.newton_opts ) return true;
	std::cerr << "ERROR! solver_opts @" << &opts << " does not have "
	          << "newton::options set!\n";
	return false;
}


const char *method_to_name( int method )
{
	return irk::rk_method_to_string[method].c_str();
}


int name_to_method( const std::string &name )
{
	return irk::rk_string_to_method[name];
}


std::vector<std::string> all_method_names()
{
	std::vector<std::string> methods;
	for( auto pair : rk_string_to_method ){
		methods.push_back( pair.first );
	}
	return methods;
}


bool is_method_explicit( const solver_coeffs &sc )
{
	for( std::size_t i = 0; i < sc.b.size(); ++i ){
		if( sc.A(i,i) != 0.0 ){
			return false;
		}
	}
	return true;
}


bool is_method_dirk( const solver_coeffs &sc )
{
	bool is_explicit = is_method_explicit(sc);
	if( is_explicit ) return false;

	for( std::size_t i = 0; i < sc.b.size(); ++i ){
		// The diagonal is already determined to not be 0.
		for( std::size_t j = i+1; j < sc.b.size(); ++j ){
			if( sc.A(i,j) != 0 ){
				return false;
			}
		}
	}

	return true;
}


bool is_method_sdirk( const solver_coeffs &sc )
{
	bool is_dirk = is_method_dirk( sc );
	if( !is_dirk ) return false;

	// Check that all diagonal elements are equal:
	for( std::size_t i = 0; i < sc.b.size()-1; ++i ){
		if( sc.A(i,i) != sc.A(i+1,i+1) ) return false;
	}

	return true;
}


} // namespace irk
