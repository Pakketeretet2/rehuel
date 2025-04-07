// This provides some tests for the test equations.

#include <catch2/catch_all.hpp>

#include "newton.hpp"
#include "test_equations.hpp"

TEST_CASE( "Test the test equations", "[test_eqs]" ){
	using namespace test_equations;
	using namespace newton;

	std::vector<double> times = { 0, 1, 2, 5, 10, 100, 1e3,
	                              1e4, 1e5, 1e6 };


	SECTION( "Van der Pol oscillator" ){
		for( double t : times ){
			std::vector<double> mus = { 1e-2, 1e-1, 1, 1e1, 1e2 };
			for( double mu : mus ){
				vdpol vdp( mu );

				vec_type y0 = { 1.0, 1.0 };

				newton_functor_wrapper<vdpol> nw( vdp, t );

				REQUIRE( verify_jacobi_matrix( y0, nw ) );
			}
		}
	}


	SECTION( "Robertson oscillator" ){
		for( double t : times ){
			rober r;
			vec_type y0 = { 1.0, 0.0, 0.0 };
			vec_type y1 = { 0.8, 0.2, 0.0 };
			vec_type y2 = { 1.0 - 1e-3 - 1e-6, 1e-3, 1e-6 };
			vec_type y3 = { 1 - 0.005 - 0.99, 0.005, 0.99 };

			newton_functor_wrapper<rober> nw( r, t );
			for( vec_type yy : { y0, y1, y2, y3 } ){
				
				REQUIRE( verify_jacobi_matrix( yy, nw ) );
			}
		}
	}

	SECTION( "Dimerization problem" ){
		for( double t : times ){
			std::vector<double> rates = { 1e-2, 1e-1, 1, 1e1, 1e2 };
			for( double r : rates ){
				vec_type y0 = { 1.0, 0.0 };
				dimer d( r );
				newton_functor_wrapper<dimer> nw( d, t );
				REQUIRE( verify_jacobi_matrix( y0, nw ) );
			}
		}
	}

	
	SECTION( "Stiff equation" ){
		for( double t : times ){
			stiff_eq se;
			newton_functor_wrapper<stiff_eq> nw( se, t );
			vec_type y0 = { 1.0, 0.0 };
			REQUIRE( verify_jacobi_matrix( y0, nw ) );
		}
	}


	SECTION( "Three body problem" ){
		for( double t : times ){
			three_body tb( 1.0, 1.0, 1.0 );
			newton_functor_wrapper<three_body> nw( tb, t );
			vec_type y0 = { 0.0, 0.0, 1.0, 0.0, 0.0, 4.0,
			                         0.0, 0.0, 0.0, 1.0, 0.0, 0.0 };

			REQUIRE( verify_jacobi_matrix( y0, nw ) );
		}
	}

}
