#include <armadillo>
#include <iostream>

#include "radau.hpp"

namespace radau {


bool verify_solver_coeffs( const solver_coeffs &sc )
{
	auto N = sc.b.size();
	if( N != sc.c.size() || N != sc.A.n_rows || N != sc.A.n_cols ){
		return false;
	}
	return true;
}



} // namespace radau
