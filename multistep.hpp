/*
   Rehuel: a simple C++ library for solving ODEs


   Copyright 2017-2019, Stefan Paquay (stefanpaquay@gmail.com)

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
   \file irk.hpp

   \brief Contains functions related to performing time integration with
   multi-step methods.
*/

#ifndef MULTISTEP_HPP
#define MULTISTEP_HPP

#include <cassert>
#include <limits>
#include <iomanip>

#include "enums.hpp"
#include "my_timer.hpp"
#include "newton.hpp"
#include "options.hpp"
#include "output.hpp"


/**
   \brief Contains functions related to multi-step methods.
*/
namespace multistep {

typedef arma::vec vec_type;
typedef arma::mat mat_type;

struct solver_options {
	int order;
};

struct multistep_output : basic_output
{};


/**
   \brief Formulae for Adams-Bashforth methods.

   \note history contains the previous solutions, with
   history[hist_offset] = y_n, history[hist_offset-1 % 6] = y_{n-1},
*/
template <typename functor_type> inline
void adams_bashforth(functor_type &func, int order, vec_type &result,
                     double t, double dt, int hist_offset,
                     const std::vector<vec_type> &history)
{
	int i0 = hist_offset;
	// By doing + 6 mod 6, you always get the right modulo:
	int i1 = (i0 +  7) % 6;
	int i2 = (i0 +  8) % 6;
	int i3 = (i0 +  9) % 6;
	int i4 = (i0 + 10) % 6;
	int i5 = (i0 + 11) % 6;
	result = history[i0];
	constexpr const double C[5][5] =
		{{ 1.0,    0, 0, 0, 0},
		 { 1.5, -0.5, 0, 0, 0},
		 { 23.0/12.0, -16.0/12.0, 5.0/12.0, 0, 0},
		 { 55.0/24.0, -59.0/24.0, 37.0/24.0, -9.0/24.0, 0},
		 { 1901.0/720.0, -2774.0/720.0, 2616.0/720.0,
		   -1274.0/720.0, 251.0/720.0} };

	switch(order) {
		case 1:
			result += dt*func.fun(t, history[i0]);
			break;
		case 2:
			result += dt*(C[1][0]*func.fun(t, history[i0]) +
			              C[1][1]*func.fun(t, history[i1]));
			break;
		case 3:
			result += dt*(C[2][0]*func.fun(t, history[i0]) +
			              C[2][1]*func.fun(t, history[i1]) +
			              C[2][2]*func.fun(t, history[i2]));
			break;
		case 4:
			result += dt*(C[3][0]*func.fun(t, history[i0]) +
			              C[3][1]*func.fun(t, history[i1]) +
			              C[3][2]*func.fun(t, history[i2]) +
			              C[3][3]*func.fun(t, history[i3]));
			break;
		case 5:
			result += dt*(C[4][0]*func.fun(t, history[i0]) +
			              C[4][1]*func.fun(t, history[i1]) +
			              C[4][2]*func.fun(t, history[i2]) +
			              C[4][3]*func.fun(t, history[i3]) +
			              C[4][4]*func.fun(t, history[i4]));
			break;

		default:
			assert("Multistep methods cannot have order > 6 or < 0!"
			       && false);
			break;
	}
}


template <typename functor_type> inline
basic_output adams_bashforth(functor_type &func, double t0, double t1,
                             const vec_type &y0,
                             const solver_options &solver_opts, double dt)
{
	if (t0 + dt > t1) {
		std::cerr << "    Rehuel: Initial dt (" << dt;
		dt = t1 - t0;
		std::cerr << ") too large for interval! Reducing to "
		          << dt << "\n";
	}
	std::cerr << "    Rehuel: Integrating over interval [ "
	          << t0 << ", " << t1 << " ]...\n"
	          << "            Method = Adams-Bashforth\n";


	my_timer timer;
	double t = t0;
	multistep_output sol;
	sol.status = SUCCESS;

	assert(dt > 0 && "Cannot use time step size <= 0!");
	std::size_t Neq = y0.size();
	vec_type y = y0;
	long long int step = 0;



}



} // namespace multistep



#endif // MULTISTEP_HPP
