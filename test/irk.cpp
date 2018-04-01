// Tests parts of the IRK methods.

#include "../arma_include.hpp"

#include "catch.hpp"

#include "irk.hpp"

/**
   \brief Combine all "cross-terms". That is, if
   v1 = { [1], [2] } and v2 = 3
   it should return
   { [1, 2], [1, 3] }
   if
   v1 = { [ 1, 2 ], [ 1, 3 ] }, v2 = 4, then it should return
   { [ 1, 2, 4 ], [ 1, 3, 4 ] }
*/
typedef std::vector<std::vector<int> > coeff_list;
coeff_list cross_term_add( const coeff_list &v1,
                           const coeff_list &v2 )
{
	coeff_list v3;
	for( std::size_t i = 0; i < v1.size(); ++i ){
		std::vector<int> v1i( v1[i].begin(), v1[i].end() );
		for( std::size_t j = 0; j < v2.size(); ++j ){
			// Add the elements of v2[j] to v1i;
			v1i.insert( v1i.end(), v2[j].begin(), v2[j].end() );
		}
		v3.push_back(v1i);
	}
	return v3;
}


std::ostream &operator<<( std::ostream &o, const coeff_list &c )
{
	for( std::size_t i = 0; i < c.size(); ++i ){
		o << " " << i << ": ";
		for( std::size_t j = 0; j < c[i].size(); ++j ){
			o << " " << c[i][j];
		}
		o << "\n";
	}
	return o;
}


int factorial( int n )
{
	if( n <= 1 ) return 1;

	int res = 1;
	for( int i = 1; i <= n; ++i ){
		res *= i;
	}

	return res;
}


int binom_coeff( int n, int b )
{
	return factorial( n ) / ( factorial(b)*factorial(n-b) );
}


TEST_CASE( "Test the factorial and binomial coefficient functions", "[binom]" )
{
	REQUIRE( factorial(0) == 1 );
	REQUIRE( factorial(1) == 1 );
	REQUIRE( factorial(2) == 2 );
	REQUIRE( factorial(3) == 2*3 );
	REQUIRE( factorial(4) == 2*3*4 );
	REQUIRE( factorial(5) == 2*3*4*5 );

	REQUIRE( binom_coeff(4,0) == 1 );
	REQUIRE( binom_coeff(4,1) == 4 );
	REQUIRE( binom_coeff(4,2) == 6 );
	REQUIRE( binom_coeff(4,3) == 4 );
	REQUIRE( binom_coeff(4,4) == 1 );

	REQUIRE( binom_coeff(4,0) == 1 );
	REQUIRE( binom_coeff(4,1) == 4 );
	REQUIRE( binom_coeff(4,2) == 6 );
	REQUIRE( binom_coeff(4,3) == 4 );
	REQUIRE( binom_coeff(4,4) == 1 );


	coeff_list v1 = { { 1, 2 } };
	coeff_list v2 = { { 1, 2, 3 }, { 1, 2, 4 } };

	REQUIRE( v1[0][0] == 1 );
	REQUIRE( v1[0][1] == 2 );

	REQUIRE( v2[0][0] == 1 );
	REQUIRE( v2[0][1] == 2 );
	REQUIRE( v2[0][2] == 3 );
	REQUIRE( v2[1][0] == 1 );
	REQUIRE( v2[1][1] == 2 );
	REQUIRE( v2[1][2] == 4 );

	REQUIRE( v1.size() == 1 );
	REQUIRE( v1[0].size() == 2 );

	REQUIRE( v2.size() == 2 );
	REQUIRE( v2[0].size() == 3 );
	REQUIRE( v2[1].size() == 3 );

	coeff_list p1 = cross_term_add( v1, { {4} } );
	coeff_list p2 = cross_term_add( v2, { {4} } );
	std::cerr << "After adding cross-terms, v1 becomes\n"
	          << p1 << "\n";
	std::cerr << "After adding cross-terms, v2 becomes\n"
	          << p2 << "\n";

	// Test it against the real thing:
	coeff_list css = { {0}, {1}, {2}, {3}, {4} };
	coeff_list cs  = { {0,1,2,3,4} };

	std::cerr << "cs:\n" << css << "\n";
	for( int i = 0; i < 5; ++i ){
		css = cross_term_add( css, cs );
		std::cerr << "Iter " << i+1 << ":\n";
		std::cerr << css << "\n";
	}
}




TEST_CASE( "Test if the product generator works.", "[collocation]" )
{
	using namespace irk;

	SECTION( "RADAU_IIA_32" ){
		auto coeffs = get_coefficients( RADAU_IIA_32 );
		auto b_interp = collocation_interpolate_coeffs( coeffs.c );
	}

	SECTION( "RADAU_IIA_53" ){
		auto coeffs = get_coefficients( RADAU_IIA_53 );
		auto b_interp = collocation_interpolate_coeffs( coeffs.c );
	}

	SECTION( "RADAU_IIA_95" ){
		auto coeffs = get_coefficients( RADAU_IIA_95 );
		auto b_interp = collocation_interpolate_coeffs( coeffs.c );
	}


}
