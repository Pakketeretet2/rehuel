/*
   Rehuel: a simple C++ library for solving ODEs


   Copyright 2017, Stefan Paquay (stefanpaquay@gmail.com)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

============================================================================= */

/**
   \file test_equations.hpp
   \brief Contains some typical ODEs that are good to test solvers on.
*/

#ifndef TEST_EQUATIONS_HPP
#define TEST_EQUATIONS_HPP

#include <armadillo>

namespace test_equations {

struct vdpol
{
	typedef arma::mat jac_type;
	vdpol( double mu ) : mu(mu){}

	arma::vec fun( double t, const arma::vec &y )
	{
		return { y[1], mu * ( 1 - y[0]*y[0] ) * y[1] - y[0] };
	}

	jac_type jac( double t, const arma::vec &y )
	{
		jac_type J(2,2);
		J(0,0) = 0.0;
		J(0,1) = 1.0;

		J(1,0) = -2*mu*y[0]*y[1] - 1.0;
		J(1,1) = mu*(1 - y[0]*y[0]);

		return J;
	}

	double mu;
};

} // test_equations




#endif // TEST_EQUATIONS_HPP
