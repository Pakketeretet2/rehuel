#include <catch2/catch_all.hpp>

#include "../irk.hpp"

TEST_CASE( "Dense output and interpolation", "[dense_output]" )
{
		
	
	SECTION( "RADAU_IIA_32" ){
		irk::solver_coeffs sc = irk::get_coefficients( irk::RADAU_IIA_32 );
		vec_type bs = project_b( 1.0, sc );
		
		REQUIRE( bs(0) == Catch::Approx(3.0/4.0) );
		REQUIRE( bs(1) == Catch::Approx(1.0/4.0) );

		bs = project_b( 1.0/3.0, sc );

		REQUIRE( bs(0) == Catch::Approx( 5.0/12.0) );
		REQUIRE( bs(1) == Catch::Approx(-1.0/12.0) );
	}

	SECTION( "RADAU_IIA_53" ){
		irk::solver_coeffs sc = irk::get_coefficients( irk::RADAU_IIA_53 );
		vec_type bs = project_b( 1.0, sc );

		for( std::size_t i = 0; i < bs.size(); ++i ){
			REQUIRE( bs(i) == Catch::Approx(sc.b(i)) );
		}

		bs = project_b( sc.c(0), sc );
		for( std::size_t i = 0; i < bs.size(); ++i ){
			REQUIRE( bs(i) == Catch::Approx(sc.A(0,i)) );
		}

		bs = project_b( sc.c(1), sc );
		for( std::size_t i = 0; i < bs.size(); ++i ){
			REQUIRE( bs(i) == Catch::Approx(sc.A(1,i)) );
		}

	}



}
