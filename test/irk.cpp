// Tests parts of the IRK methods.

#include "../arma_include.hpp"

#include "catch.hpp"
#include "irk.hpp"
#include "test_equations.hpp"



TEST_CASE( "Test the expansion of polynomial coefficients.", "[poly-expand]" )
{
	using namespace irk;

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

	coeff_list c1 = { {1}, {2} };
	coeff_list c2 = { {3}, {4} };
	coeff_list c3 = expand( c1, c2 );
	REQUIRE( c1.size() == 2 );
	REQUIRE( c1[0].size() == 1 );
	REQUIRE( c2.size() == c1.size() );
	REQUIRE( c2[0].size() == 1 );

	REQUIRE( c3.size() == 4 );
	REQUIRE( c3[0].size() == 2 );
	std::cerr << "c1: " << c1 << "\n";
	std::cerr << "c2: " << c2 << "\n";
	std::cerr << "c3: " << c3 << "\n";

	REQUIRE( c3[0][0] == 1 );
	REQUIRE( c3[0][1] == 3 );
	REQUIRE( c3[1][0] == 1 );
	REQUIRE( c3[1][1] == 4 );
	REQUIRE( c3[2][0] == 2 );
	REQUIRE( c3[2][1] == 3 );
	REQUIRE( c3[3][0] == 2 );
	REQUIRE( c3[3][1] == 4 );

	// { {1,3,5}, {1,4,5}, {2,3,5}, {2,4,5},
	//   {1,3,6}, {1,4,6}, {2,3,6}, {2,4,6} }

	coeff_list c4 = { {5}, {6} };
	coeff_list c5 = expand( c3, c4 );
	REQUIRE( c5.size() == 8 );
	REQUIRE( c5[0].size() == 3 );

	std::cerr << "c5: " << c5 << "\n";

	REQUIRE( c5[0][0] == 1 );
	REQUIRE( c5[0][1] == 3 );
	REQUIRE( c5[0][2] == 5 );

	REQUIRE( c5[1][0] == 1 );
	REQUIRE( c5[1][1] == 3 );
	REQUIRE( c5[1][2] == 6 );

	REQUIRE( c5[2][0] == 1 );
	REQUIRE( c5[2][1] == 4 );
	REQUIRE( c5[2][2] == 5 );

	REQUIRE( c5[3][0] == 1 );
	REQUIRE( c5[3][1] == 4 );
	REQUIRE( c5[3][2] == 6 );

	REQUIRE( c5[4][0] == 2 );
	REQUIRE( c5[4][1] == 3 );
	REQUIRE( c5[4][2] == 5 );

	REQUIRE( c5[5][0] == 2 );
	REQUIRE( c5[5][1] == 3 );
	REQUIRE( c5[5][2] == 6 );

	REQUIRE( c5[6][0] == 2 );
	REQUIRE( c5[6][1] == 4 );
	REQUIRE( c5[6][2] == 5 );

	REQUIRE( c5[7][0] == 2 );
	REQUIRE( c5[7][1] == 4 );
	REQUIRE( c5[7][2] == 6 );

	// 3. expand( { {1,2,3}, {4,5} } ) should be
	// ( a1 + a2 + a3 ) ( b1 + b2 ) =
	// ( a1b1 + a2b1 + a3b1 + a1b2 + a2b2 + a3b2 ) =
	// { {1,4}, {2,4}, {3,4}, {1,5}, {2,5}, {2,6} }
	coeff_list c21 = { {1}, {2}, {3} };
	coeff_list c22 = { {4}, {5} };
	coeff_list c23 = expand( c21, c22 );

	std::cerr << "c21:\n" << c21 << "\n";
	std::cerr << "c22:\n" << c22 << "\n";
	std::cerr << "c23:\n" << c23 << "\n";

	REQUIRE( c23.size() == 6 );
	REQUIRE( c23[0].size() == 2 );

	REQUIRE( c23[0][0] == 1 );
	REQUIRE( c23[0][1] == 4 );
	REQUIRE( c23[1][0] == 1 );
	REQUIRE( c23[1][1] == 5 );

	REQUIRE( c23[2][0] == 2 );
	REQUIRE( c23[2][1] == 4 );
	REQUIRE( c23[3][0] == 2 );
	REQUIRE( c23[3][1] == 5 );

	REQUIRE( c23[4][0] == 3 );
	REQUIRE( c23[4][1] == 4 );
	REQUIRE( c23[5][0] == 3 );
	REQUIRE( c23[5][1] == 5 );


}



TEST_CASE( "Test if the product generator works.", "[collocation]" )
{
	using namespace irk;
	std::cerr << "\n";

	// For Radau 32 it should be obvious.
	// c1 = 1/3, c2 = 1
	// hence lp1(x) = ( x - c2 ) / ( c1 - c2 )
	//              = -1.5x + 1.5
	//       lp2(x) = ( x - c1 ) / ( c2 - c1 )
	//              =  1.5x + 0.5
	//

	SECTION( "RADAU_IIA_32" ){
		std::cerr << "Checking coeffs for RADAU_IIA_32:\n";
		auto coeffs = get_coefficients( RADAU_IIA_32 );
		mat_type b_interp = collocation_interpolate_coeffs( coeffs.c );

		std::cerr << "Size of b_interp is " << b_interp.size() << "\n";
		std::cerr << "Here is the matrix:\n";
		std::cerr << b_interp << "\n";

		double b1 = coeffs.b[0];
		double b2 = coeffs.b[1];
		double bb1 = 0.0;
		double bb2 = 0.0;
		for( std::size_t i = 0; i < 2; ++i ){
			bb1 += b_interp(0,i);
			bb2 += b_interp(1,i);
		}

		REQUIRE( b1 == Approx(bb1) );
		REQUIRE( b2 == Approx(bb2) );

	}

	SECTION( "RADAU_IIA_53" ){
		std::cerr << "Checking coeffs for RADAU_IIA_53:\n";
		auto coeffs = get_coefficients( RADAU_IIA_53 );
		mat_type b_interp = collocation_interpolate_coeffs( coeffs.c );
	}

	SECTION( "RADAU_IIA_95" ){
		std::cerr << "Checking coeffs for RADAU_IIA_95:\n";
		auto coeffs = get_coefficients( RADAU_IIA_95 );
		mat_type b_interp = collocation_interpolate_coeffs( coeffs.c );
	}
}



TEST_CASE( "Test if merging two solution objects works.", "[sol_merge]" )
{
	using namespace irk;
	rk_output sol1;
	rk_output sol2;

	sol1.t_vals = { 0.0, 0.1, 0.2 };
	sol2.t_vals = { 0.3, 0.4, 0.5 };

	sol1.y_vals = { {1.0},
	                {0.9},
	                {0.9*0.9} };
	sol2.y_vals = { {0.9*0.9*0.9},
	                {0.9*0.9*0.9*0.9},
	                {0.9*0.9*0.9*0.9*0.9} };


	rk_output sol3 = merge_rk_output( sol1, sol2 );
	REQUIRE( sol3.t_vals[0] == 0.0 );
	REQUIRE( sol3.t_vals[1] == 0.1 );
	REQUIRE( sol3.t_vals[2] == 0.2 );
	REQUIRE( sol3.t_vals[3] == 0.3 );
	REQUIRE( sol3.t_vals[4] == 0.4 );
	REQUIRE( sol3.t_vals[5] == 0.5 );

	REQUIRE( sol3.y_vals[0](0) == 1.0 );
	REQUIRE( sol3.y_vals[1](0) == 0.9 );
	REQUIRE( sol3.y_vals[2](0) == (0.9*0.9) );
	REQUIRE( sol3.y_vals[3](0) == (0.9*0.9*0.9) );
	REQUIRE( sol3.y_vals[4](0) == (0.9*0.9*0.9*0.9) );
	REQUIRE( sol3.y_vals[5](0) == (0.9*0.9*0.9*0.9*0.9) );

	SECTION( "Applied to an aqual ODE." ){

		auto so = default_solver_options();
		newton::options opts;
		opts.tol = 0.1*so.rel_tol;
		vec_type Y0 = { 1.0 };
		test_equations::exponential func( -0.4 );
		so.out_interval = 1;
		so.newton_opts = &opts;
		rk_output soltmp1 = irk::odeint( func, 0.0, 2.0, Y0, so );
		std::size_t nsol1 = soltmp1.t_vals.size();
	        vec_type Y1 = soltmp1.y_vals[nsol1-1];
		double dt = soltmp1.t_vals[nsol1-1] - soltmp1.t_vals[nsol1-2];
		rk_output soltmp2 = irk::odeint( func, 2.0, 4.0,
		                                       Y1, so, dt );

		auto merge = merge_rk_output( soltmp1, soltmp2 );

		for( std::size_t i = 0; i < merge.t_vals.size(); ++i ){
			if( i < nsol1 ){
				REQUIRE( merge.t_vals[i] == soltmp1.t_vals[i] );
			}else{
				auto j = i - nsol1;
				REQUIRE( merge.t_vals[i] == soltmp2.t_vals[j] );
			}
		}
	}


}
