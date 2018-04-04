#include <iostream>
#include <string>

#include "irk.hpp"


namespace irk {



/**
   \brief expands the coefficient lists.

   This is needed to automatically calculate the interpolating coefficients.

   This is like operator expansion of operator( c1, c2 )
   if c1 = { a1 + a2 } and c2 = { b1 + b2 }
   and we encode for that as c1 = { {1}, {2} }; c2 = { {3}, {4} }
   then the expansion would be operator(c1,c2) =
   { a1b1 + a1b2 + a2b1 + a2b2 } which would be encoded as
   { {1,3}, {1,4}, {2,3}, {2,4} }.
   operator( ( a1b1 + a1b2 + a2b1 + a2b2 ), (x1 + x2) ) follows from induction.
*/
typedef std::vector<std::vector<int> > coeff_list;
coeff_list expand( const coeff_list &c1,
                   const coeff_list &c2 )
{
	// Test driven:
	// 1. expand( { {1}, {2} }, { {3}, {4} } ) should lead to
	// { {1,3}, {1,4}, {2,3}, {2,4} }
	//
	// 2. expand( { {1,3}, {1,4}, {2,3}, {2,4} }, { {5}, {6} } ) leads to
	// { {1,3,5}, {1,4,5}, {2,3,5}, {2,4,5},
	//   {1,3,6}, {1,4,6}, {2,3,6}, {2,4,6} }
	//
	// 3. expand( { {1,2,3}, {4,5} } ) should be
	// ( a1 + a2 + a3 ) ( b1 + b2 ) =
	// ( a1b1 + a2b1 + a3b1 + a1b2 + a2b2 + a3b2 ) =
	// { {1,4}, {2,4}, {3,4}, {1,5}, {2,5}, {2,6} }
	//
	coeff_list c3;

	for( std::size_t i = 0; i < c1.size(); ++i ){
		for( std::size_t j = 0; j < c2.size(); ++j ){
			std::vector<int> vij( c1[i].begin(), c1[i].end() );
			vij.insert( vij.end(), c2[j].begin(), c2[j].end() );
			c3.push_back( vij );
		}
	}
	return c3;
}


/**
   \brief Output operator for a coefficient list.
*/
std::ostream &operator<<( std::ostream &o, const coeff_list &c )
{
	o << " { ";
	for( std::size_t i = 0; i < c.size(); ++i ){
		o << "{";
		for( std::size_t j = 0; j < c[i].size(); ++j ){
			o << " " << c[i][j];
		}
		o << " }";
		if( i < c.size() - 1 ) o << ", ";
	}
	o << " }";
	return o;
}


/**
   \brief Simple calculation of the factorial of n.
*/
int factorial( int n )
{
	if( n <= 1 ) return 1;

	int res = 1;
	for( int i = 1; i <= n; ++i ){
		res *= i;
	}

	return res;
}

/**
   \brief binomial coefficient (n above b).
*/
int binom_coeff( int n, int b )
{
	return factorial( n ) / ( factorial(b)*factorial(n-b) );
}



bool verify_solver_coeffs( const solver_coeffs &sc )
{
	auto N = sc.b.size();
	if( N != sc.c.size() || N != sc.A.n_rows || N != sc.A.n_cols ){
		return false;
	}
	if( N == 0 ) return false;

	return true;
}


arma::mat collocation_interpolate_coeffs( const arma::vec &c )
{
	// Interpolates on a solution interval as
	// b_j(t) = b_interp(j,0)*t + b_interp(j,1)*t^2
	//          + b_interp(j,2)*t^3 + ...
	// The coefficients arise from the following equation:
	//
	// bj(t) = integral( lp_j(x), dx ),
	// where lp_j(x) = (x-c2)(x-c1)(x-c3) / (cj-c2)(cj-c1)(cj-c3)
	//
	// Therefore, to derive the shape of the polynomial, we have
	// to perform some expansion in terms of all the coefficients.
	//
	// For example, if we have three stages, then we have
	//
	// lp_1(x) = (x - c2)(x - c3)/(c1-c2)(c1-c3)
	// lp_2(x) = (x - c1)(x - c3)/(c2-c1)(c2-c3)
	// lp_3(x) = (x - c1)(x - c2)/(c3-c1)(c3-c2)
	//
	// The denominators are easily dealt with. For the nominators, we
	// need to expand the products. We encode x with -1, c1 with 0, etc.
	// Therefore, for three stages we should get...
	//
	// expand( { {-1}, {1} }, { {-1}, {2} } ) =
	// { {-1 -1}, {-1 2}, {-1 1}, {1 2} }.
	// This means x^2  + c2 x + c1 x + c1c2
	//
	// Afterwards, we integrate in x, so for this case, we'd obtain
	// x^3/3  + c2 x^2 / 2 + c1 x ^ 2 / 2 + c1c2 * x
	//

	std::size_t Ns = c.size();
	arma::vec d( Ns );
	for( std::size_t i = 0; i < Ns; ++i ){
		double cfacs = 1.0;
		for( std::size_t j = 0; j < Ns; ++j ){
			if( j == i ) continue;
			cfacs *= c(i) - c(j);
		}
		d(i) = cfacs;
	}

	std::vector<coeff_list> poly_coefficients(Ns);
	for( std::size_t i = 0; i < Ns; ++i ){
		coeff_list ci;
		for( std::size_t j = 0; j < Ns; ++j ){
			if( i == j ) continue;

			int jj = j;
			// -1 codes for x.
			if( ci.size() == 0 ){
				ci = {{-1}, {jj}};
			}else{
				coeff_list tmp = expand( ci, { {-1}, {jj} } );
				ci = tmp;
			}
		}
		poly_coefficients[i] = ci;
	}

	// The coefficients construct the polynomial, so from this, we can
	// exactly calculate the polynomial coefficients. It is convoluted
	// but it works...
	//
	// The number of -1 s encode for the power of the term, with
	// four -1 s meaning it is a fifth order term, etc.
	arma::mat b_interp;
	b_interp.zeros( Ns, Ns );
	for( std::size_t i = 0; i < Ns; ++i ){
		for( const std::vector<int> cf : poly_coefficients[i] ){
			// Check the order of the term.
			int order = 0;
			for( int j : cf ){
				if( j == - 1 ) order++;
			}
			// order0 is the constant term that will become linear.
			// If you grab the highest order term, ignore it.
			// if( order == static_cast<int>(Ns) ) continue;

			double coeff = 1.0 / d(i) / (1.0 + order);
			for( int j : cf ){
				if( j != -1 ){
					// Multiply by the right c:
					coeff *= -c[j];
				}
			}

			int b_interp_idx = Ns - order - 1;
			int Ns_nosign = Ns;
			assert( (b_interp_idx >= 0) && "index out of range" );
			assert( (b_interp_idx < Ns_nosign) &&
			        "index out of range" );

			b_interp(i, b_interp_idx) += coeff;
		}
	}

	return b_interp;
}




solver_coeffs get_coefficients( int method )
{
	solver_coeffs sc;
	double one_third = 1.0/3.0;
	double one_six = 1.0/6.0;

	double sqrt3 = sqrt(3.0);
	// double sqrt5 = sqrt(5.0);
	double sqrt6 = sqrt(6.0);
	// double sqrt15 = sqrt(15.0);

	// Methods that need adding:
	// RADAU_137, LOBATTO_IIA_{43,86,129},
	// LOBATTO_IIIC_{43,86,129}, GAUSS_LEGENDRE_{42,84,126}

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

		case LOBATTO_IIIA_43:

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

		// case LOBATTO_IIIA_86:
		// case LOBATTO_IIIA_129:


		case LOBATTO_IIIC_43:

			sc.A = { { 1.0/6.0, -1.0/3.0, 1.0/6.0 },
			         { 1.0/6.0, 5.0/12.0, -1.0/12.0 },
			         { 1.0/6.0, 2.0/3.0, 1.0/6.0 } };
			sc.b = { 1.0/6.0, 2.0/3.0, 1.0/6.0 };
			sc.b2 = { -0.5, 2.0, -0.5 };
			sc.c = { 0.0, 0.5, 1.0 };
			sc.order = 4;
			sc.order2 = 2;

			break;

		// case LOBATTO_IIIC_86:
		// case LOBATTO_IIIC_129:

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

		// case GAUSS_LEGENDRE_84:
		// case GAUSS_LEGENDRE_126:

		case RADAU_IIA_32:{

			sc.A = { {5.0/12.0, -1.0/12.0},
			         {3.0/4.0, 1.0/4.0 } };
			// A does not have real eigenvalues...
			// We take the real part of the complex values.
			sc.gamma = 1.0/3.0;

			sc.c  = { 1.0/3.0, 1.0 };
			sc.b  = { 3.0/4.0, 1.0/4.0 };

			sc.b2 = { (-6*sc.gamma + 3.0) / 4.0,
			          ( 2*sc.gamma + 1.0) / 4.0 };

			sc.order = 3;
			sc.order2 = 2;

			sc.b_interp = collocation_interpolate_coeffs( sc.c );

			break;
		}


		case RADAU_IIA_53:{

			sc.A = { { (88 - 7*sqrt6)/360.0, (296 - 169*sqrt6)/1800.0, (-2+3*sqrt6)/225.0 },
			         { (296 + 169*sqrt6)/1800.0, (88 + 7*sqrt6)/360.0, (-2-3*sqrt6)/225.0 },
			         { (16.0 - sqrt6)/36.0, (16 + sqrt6)/36.0, 1.0 / 9.0 } };
			// gamma is the real eigenvalue of A.
			sc.gamma = 2.74888829595677e-01;


			sc.c  = {  (4.0-sqrt6)/10.0,  (4.0+sqrt6) / 10.0, 1.0 };
			sc.b  = {  (16 - sqrt6)/36.0,
			           (16 + sqrt6)/36.0,
			           1.0 / 9.0 };

			sc.b2 = { -((18*sqrt6 + 12)*sc.gamma - 16 + sqrt6)/36.0,
			           ((18*sqrt6 - 12)*sc.gamma + 16 + sqrt6)/36.0,
			          -(3*sc.gamma - 1) / 9.0 };

			sc.order = 5;
			sc.order2 = 3;

			sc.b_interp = collocation_interpolate_coeffs( sc.c );



			break;
		}

		case RADAU_IIA_95:{
			sc.A = {{ 0.0729988643179033243, -0.0267353311079455719,
			          0.0186769297639843544, -0.0128791060933064399,
			          0.00504283923388201521 },
			        { 0.153775231479182469, 0.146214867847493507,
			          -0.036444568905128090, 0.021233063119304719,
			          -0.007935579902728778 },
			        { 0.14006304568480987, 0.29896712949128348,
			          0.16758507013524896, -0.03396910168661775,
			          0.01094428874419225 },
			        { 0.14489430810953476, 0.2765000687601592,
			          0.3257979229104210, 0.1287567532549098,
			          -0.01570891737880533 },
			        { 0.1437135607912259, 0.2813560151494621,
			          0.3118265229757413, 0.2231039010835707,
			          0.04 } };

			sc.c = { 0.05710419611451768219312119255411562124,
			         0.27684301363812382768004599768562514112,
			         0.58359043236891682005669766866291724869,
			         0.86024013565621944784791291887511976674,
			         1.0 };

			sc.b = { 0.1437135607912259,
			         0.2813560151494621,
			         0.3118265229757413,
			         0.2231039010835707,
			         0.04 };

			// gamma is the real eigenvalue of A.
			sc.gamma = 0.1590658444274690;

			sc.b2 = { sc.b(0) - 1.5864079001863282*sc.gamma,
			          sc.b(1) + 1.0081178814983730*sc.gamma,
			          sc.b(2) - 0.73097486615978746*sc.gamma,
			          sc.b(3) + 0.50926488484774272*sc.gamma,
			          sc.b(4) - 0.2*sc.gamma };

			sc.order  = 9;
			sc.order2 = 5;

			sc.b_interp = collocation_interpolate_coeffs( sc.c );

			// Interpolates on a solution interval as
			// b_j(t) = b_interp(j,0)*t + b_interp(j,1)*t^2
			//          + b_interp(j,2)*t^3 + ...
			/*
			double c1 = sc.c[0];
			double c2 = sc.c[1];
			double c3 = sc.c[2];

			double d1 = (c1-c2)*(c1-c3);
			double d2 = (c2-c1)*(c2-c3);
			double d3 = (c3-c1)*(c3-c2);

			sc.b_interp = { { c2*c3/d1, -(c2+c3)/(2.0*d1), (1.0/3.0)/d1 },
			                { c1*c3/d2, -(c1+c3)/(2.0*d2), (1.0/3.0)/d2 },
			                { c1*c2/d3, -(c1+c2)/(2.0*d3), (1.0/3.0)/d3 } };
			*/

			         break;
		}
		// case RADAU_IIA_137

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

arma::vec project_b( double theta, const irk::solver_coeffs &sc )
{
	std::size_t Ns = sc.b.size();
	const arma::mat &bcs = sc.b_interp;
	assert( bcs.size() > 0 && "Chosen method does not have dense output!" );

	arma::vec bs(Ns), ts(Ns);

	// ts will contain { t, t^2, t^3, ..., t^{Ns} }
	double tt = theta;
	for( std::size_t i = 0; i < Ns; ++i ){
		int j = Ns - i - 1;
		ts[j] = tt;
		tt *= theta;
	}

	// Now bs = sc.b_interp * ts;
	return sc.b_interp * ts;
}




} // namespace irk
