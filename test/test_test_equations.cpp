// This provides some tests for the test equations.

#include "catch.hpp"

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

				arma::vec y0 = { 1.0, 1.0 };

				newton_functor_wrapper<vdpol> nw( vdp, t );

				REQUIRE( verify_jacobi_matrix( y0, nw ) );
			}
		}
	}


	SECTION( "Robertson oscillator" ){
		for( double t : times ){
			rober r;
			arma::vec y0 = { 1.0, 0.0, 0.0 };
			arma::vec y1 = { 0.8, 0.2, 0.0 };
			arma::vec y2 = { 1.0 - 1e-3 - 1e-6, 1e-3, 1e-6 };
			arma::vec y3 = { 1 - 0.005 - 0.99, 0.005, 0.99 };

			newton_functor_wrapper<rober> nw( r, t );
			for( arma::vec yy : { y0, y1, y2, y3 } ){
				REQUIRE( verify_jacobi_matrix( yy, nw ) );
			}
		}
	}

	SECTION( "Dimerization problem" ){
		for( double t : times ){
			std::vector<double> rates = { 1e-2, 1e-1, 1, 1e1, 1e2 };
			for( double r : rates ){
				arma::vec y0 = { 1.0, 0.0 };
				dimer d( r );
				newton_functor_wrapper<dimer> nw( d, t );
				REQUIRE( verify_jacobi_matrix( y0, nw ) );
			}
		}
	}

	SECTION( "Reaction-diffusion equation" ){
		for( double t : times ){
			std::vector<double> rates = { 0.0, 1e-2, 1e-1, 1, 1e1 };
			double D1 = 1.0;
			double D2 = 0.5;
			std::size_t Nx = 128;
			for( double r : rates ){
				reac_diff reac( Nx, D1, D2, r );
				arma::vec y0( 2*Nx );
				for( std::size_t i = 0; i < Nx; ++i ){
					y0[i] = 0.5 + 0.5*( i < Nx/2);
					y0[i+1] = 0.0;
				}
				//newton_functor_wrapper<reac_diff> nw( reac, t );
				//REQUIRE( verify_jacobi_matrix( y0, nw ) );
			}
		}
	}

	SECTION( "Stiff equation" ){
		for( double t : times ){
			stiff_eq se;
			newton_functor_wrapper<stiff_eq> nw( se, t );
			arma::vec y0 = { 1.0, 0.0 };
			REQUIRE( verify_jacobi_matrix( y0, nw ) );
		}
	}


	SECTION( "Three body problem" ){
		for( double t : times ){
			three_body tb( 1.0, 1.0, 1.0 );
			newton_functor_wrapper<three_body> nw( tb, t );
			arma::vec y0 = { 0.0, 0.0, 1.0, 0.0, 0.0, 4.0,
			                 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 };

			REQUIRE( verify_jacobi_matrix( y0, nw ) );
		}
	}

}
