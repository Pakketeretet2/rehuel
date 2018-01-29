#include <armadillo>
#include <iostream>
#include <string>

#include "multistep.hpp"


namespace multistep {



solver_coeffs get_coefficients( int method, int order )
{
	solver_coeffs sc;
	switch(method){
		case ADAMS_BASHFORTH:
			if( order > 5 ){
				std::cerr << "No stable Adams-Bashforth of "
				          << "order > 5 exists!\n";
				return sc;
			}
			break;
		case ADAMS_MOULTON:
			if( order > 5 ){
				std::cerr << "No stable Adams-Moulton of "
				          << "order > 5 exists!\n";
				return sc;
			}
			break;
		case BDF:
			if( order > 6 ){
				std::cerr << "No stable BDF of "
				          << "order > 6 exists!\n";
				return sc;
			}
			break;
		default:
			std::cerr << "Method " << method << " unknown!\n";
			return sc;
	}


	sc.method_type = method;
	sc.order = order;
	sc.dt = 1e-4;

	// These are all known from literature (see Wikipedia):
	sc.cs_ab = { { 1.0,            0.0,         0.0,        0.0,     0.0},
	             { 3.0/2.0,   -1.0/2.0,         0.0,        0.0,     0.0},
	             { 23.0/12.0, -4.0/3.0,    5.0/12.0,        0.0,     0.0},
	             { 55.0/24.0, -59.0/24.0, 37.0/24.0,   -3.0/8.0,     0.0},
	             { 1901.0/720.0,      -1387.0/360.0, 109.0/30.0,
	               -637.0/360.0,      251.0/720.0} };

	sc.cs_am = { {      1.0, 0.0,           0.0,       0.0,       0.0 },
	             {      0.5, 0.5,           0.0,       0.0,       0.0 },
	             { 5.0/12.0, 2.0/3.0, -1.0/12.0,       0.0,       0.0 },
	             {  3.0/8.0, 19.0/24.0, -5.0/24.0, 1.0/24.0, 0.0 },
	             { 251.0/720.0, 646.0/720.0, -264.0 / 720.0,
	               106.0/720.0, -19.0/720.0 } };

	sc.cs_bdf = { { -1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
	              { -4.0/3.0, 1.0/3.0, 0.0, 0.0, 0.0, 0.0 },
	              { -18.0/11.0, 9.0/11.0, -2.0/11.0, 0.0, 0.0, 0.0 },
	              { -48.0/25.0, 36.0/25.0, -16.0/25.0, 3.0/25.0, 0.0, 0.0 },

	              { -300.0/137.0, 300.0/137.0, -200.0/137.0,
	                75.0/137.0, -12.0/137.0, 0.0 },

	              { -360.0/147.0, 450.0/147.0, -400.0/147.0,
	                225.0/147.0, -72.0/147.0, 10.0/147.0 } };
	sc.b = { 1.0,
	         2.0/3.0,
	         6.0/11.0,
	         12.0/25.0,
	         60.0 / 137.0,
	         60.0 / 147.0 };



	return sc;
}


solver_options default_solver_options()
{
	solver_options so;
	return so;
}


const char *method_to_name( int method )
{
	return multistep::ms_method_to_string[method].c_str();
}


int name_to_method( const std::string &name )
{
	return multistep::ms_string_to_method[name];
}


std::vector<std::string> all_method_names()
{
	std::vector<std::string> methods;
	for( auto pair : ms_string_to_method ){
		methods.push_back( pair.first );
	}
	return methods;
}


} // namespace multistep
