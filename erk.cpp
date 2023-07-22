#include "erk.hpp"


using namespace erk;


namespace erk {

solver_coeffs get_coefficients( int method )
{
	solver_coeffs sc;
	double one_third = 1.0/3.0;
	double one_six = 1.0/6.0;

	sc.FSAL = false;
	sc.name = method_to_name( method );
	sc.gamma = 0.0;


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

	return sc;
}


solver_options default_solver_options()
{
	solver_options s;
	return s;
}


bool verify_solver_options(solver_options &opts)
{
	return true;
}


bool verify_solver_coeffs( const erk::solver_coeffs &sc )
{
	auto N = sc.b.size();
	if( N != sc.c.size() || N != sc.A.n_rows || N != sc.A.n_cols ){
		return false;
	}
	if( N == 0 ) return false;

	for (std::size_t i = 0; i < N; ++i) {
		for (std::size_t j = i; j < N; ++j) {
			if (sc.A(i,j) != 0.0) {
				std::cerr << "Coefficient matrix not consistent"
				          << " with explicit RK method!\n";
				return false;
			}
		}
	}

	return true;
}


const char *method_to_name( int method )
{
	return erk::rk_method_to_string[method].c_str();
}


int name_to_method( const std::string &name )
{
	return erk::rk_string_to_method[name];
}


std::vector<std::string> all_method_names()
{
	std::vector<std::string> methods;
	for (auto pair : rk_string_to_method){
		methods.push_back(pair.first);
	}
	return methods;
}


vec_type project_b( double theta, const erk::solver_coeffs &sc )
{
	long int Ns = sc.b.size();
	const mat_type &bcs = sc.b_interp;
	assert( bcs.size() > 0 && "Chosen method does not have dense output!" );

	vec_type bs(Ns), ts(Ns);

	// ts will contain { t, t^2, t^3, ..., t^{Ns} }
	double tt = theta;
	for( long int i = 0; i < Ns; ++i ){
		int j = Ns - i - 1;
		ts[j] = tt;
		tt *= theta;
	}

	// Now bs = sc.b_interp * ts;
	return sc.b_interp * ts;
}


}
