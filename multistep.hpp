/*
   Rehuel: a simple C++ library for solving ODEs


   Copyright 2017-2023 Stefan Paquay (stefanpaquay@gmail.com)

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

#include "cyclic_buffer.hpp"
#include "enums.hpp"
#include "irk.hpp"
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

   Updates Y to the new value and also appends its value to history.
*/
template <typename functor_type> inline
void adams_bashforth_step(functor_type &func, int order,
                          vec_type &Y, double t, double dt,
                          const cyclic_buffer<vec_type> &history)
{

	constexpr const double C[5][5] =
		{{ 1.0,    0, 0, 0, 0},
		 { 1.5, -0.5, 0, 0, 0},
		 { 23.0/12.0, -16.0/12.0, 5.0/12.0, 0, 0},
		 { 55.0/24.0, -59.0/24.0, 37.0/24.0, -9.0/24.0, 0},
		 { 1901.0/720.0, -2774.0/720.0, 2616.0/720.0,
		   -1274.0/720.0, 251.0/720.0} };

	switch(order) {
	case 1:
		Y += dt*func.fun(t, history[0]);
		break;
	case 2:
		Y += dt*(C[1][0]*func.fun(t, history[0]) +
		         C[1][1]*func.fun(t, history[1]));
		break;
	case 3:
		Y += dt*(C[2][0]*func.fun(t, history[0]) +
		         C[2][1]*func.fun(t, history[1]) +
		         C[2][2]*func.fun(t, history[2]));
		break;
	case 4:
		Y += dt*(C[3][0]*func.fun(t, history[0]) +
		         C[3][1]*func.fun(t, history[1]) +
		         C[3][2]*func.fun(t, history[2]) +
		         C[3][3]*func.fun(t, history[3]));
		break;
	case 5:
		Y += dt*(C[4][0]*func.fun(t, history[0]) +
		         C[4][1]*func.fun(t, history[1]) +
		         C[4][2]*func.fun(t, history[2]) +
		         C[4][3]*func.fun(t, history[3]) +
		         C[4][4]*func.fun(t, history[4]));
		break;

	default:
		assert("Adams-Bashforth cannot have order > 5 or order < 0"
		       && false);
		break;
	}
}



template <typename functor_type> inline
basic_output bootstrap_history(functor_type &func, int order, const vec_type &y,
                               double t, double dt)
{
	output_options output_opts;
	output_opts.output_mode = output_options::NO_OUTPUT;

	basic_output start;
	start.t_vals.push_back(t);
	start.y_vals.push_back(y);

	// For bootstrapping, just use a high-order RK method:
	int method = irk::LOBATTO_IIIC_85;
	auto opts = irk::default_solver_options();
	newton::options n_opts;
	opts.newton_opts = &n_opts;
	opts.rel_tol = 1e-10;
	opts.abs_tol = 1e-9;

	for (int i = 1; i < order; ++i) {
		auto tmp_sol = irk::odeint(func,  t, t+dt, start.y_vals[i-1],
		                           opts, output_opts, method);
		std::size_t Nt = tmp_sol.t_vals.size();
		t += dt;
		start.t_vals.push_back(t);
		start.y_vals.push_back(tmp_sol.y_vals[Nt-1]);
	}

	return start;
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


	double t = t0;
	multistep_output sol;
	sol.status = SUCCESS;

	assert(dt > 0 && "Cannot use time step size <= 0!");

	my_timer timer;
	vec_type y = y0;
	long long int step = 0;

	cyclic_buffer<vec_type> history(solver_opts.order);
	// For multistep methods we need to do some bootstrapping:
	basic_output hist = bootstrap_history(func, solver_opts.order, y, t, dt);


	for (std::size_t i = 0; i < hist.t_vals.size(); ++i) {
		sol.t_vals.push_back(hist.t_vals[i]);
		sol.y_vals.push_back(hist.y_vals[i]);
		history.push_back(hist.y_vals[i]);
	}
	std::size_t hist_size = sol.t_vals.size();
	t = sol.t_vals[hist_size-1];
	y = sol.y_vals[hist_size-1];

	while (t < t1) {
		adams_bashforth_step(func, solver_opts.order, y, t, dt, history);
		history.push_back(y);
		t += dt;
		++step;
		sol.t_vals.push_back(t);
		sol.y_vals.push_back(y);
	}
	timer.toc("    Solving with Adams-Bashforth method");
	return sol;
}



template <typename functor_type> inline
multistep::multistep_output bdf(functor_type &func, double t0, double t1,
                                const vec_type &y0,
                                const solver_options &solver_opts, double dt)
{
	double t = t0;
	multistep_output sol;
	sol.status = SUCCESS;

	if (solver_opts.order > 6) {
		std::cerr << "BDF methods over order 6 do not exist!\n";
		sol.status = GENERAL_ERROR;
		return sol;
	}
	if (solver_opts.order <= 0) {
		std::cerr << "BDF method of order <= 0 does not make sense!\n";
		sol.status = GENERAL_ERROR;
		return sol;
	}

	if (t0 + dt > t1) {
		std::cerr << "    Rehuel: Initial dt (" << dt;
		dt = t1 - t0;
		std::cerr << ") too large for interval! Reducing to "
		          << dt << "\n";
	}
	std::cerr << "    Rehuel: Integrating over interval [ "
	          << t0 << ", " << t1 << " ]...\n"
	          << "            Method = BDF-" << solver_opts.order << "\n";

	assert(dt > 0 && "Cannot use time step size <= 0!");

	// Solution is constructed by
	// y_{n+1} + C[2][0]*y_n + C[2][1]*y_{n-1} = d[2]*dt*f(t_{n+1}, y_{n+1})
	constexpr const double C[6][6] =
		{{-1.0,    0,   0,   0,   0,   0},  // BDF 1
		 {-4.0/3.0, 1.0/3.0, 0,  0,  0,  0},  // BDF 2
		 {-18.0/11.0, 9.0/11.0, -2.0/11.0, 0, 0, 0 }, // BDF 3
		 {-48.0/25.0, 36.0/25.0, -16.0/25.0, 3.0/25.0, 0, 0},  // BDF 4
		 {-300/137.0, 300/137.0, -200/137.0, 75/137.0, -12/137.0, 0}, // BDF 5
		 {-360/147.0, 450/147.0, -400/147.0, 225/147.0, -72/147.0, 10/147.0}
		};
	constexpr const double bdf_rhs_weight[6] =
		{1.0, 2/3.0, 6/11.0, 12/25.0, 60/137.0, 60/147.0};

	double bdf_params[6];
	double bdf_params_alt[6];
	for (std::size_t i = 0; i < 6; ++i) {
		bdf_params[i] = C[solver_opts.order-1][i];
		if (solver_opts.order < 6) {
			bdf_params_alt[i] = C[solver_opts.order][i];
		}
	}

	my_timer timer;
	vec_type y = y0;
	long long int step = 0;

	cyclic_buffer<vec_type> history(solver_opts.order+1);
	// For multistep methods we need to do some bootstrapping:
	basic_output hist = bootstrap_history(func, solver_opts.order+1, y, t, dt);

	for (std::size_t i = 0; i < hist.t_vals.size(); ++i) {
		sol.t_vals.push_back(hist.t_vals[i]);
		sol.y_vals.push_back(hist.y_vals[i]);
		history.push_back(hist.y_vals[i]);
	}
	std::size_t hist_size = sol.t_vals.size();
	t = sol.t_vals[hist_size-1];
	y = sol.y_vals[hist_size-1];

	std::cerr << "  Bootstrapping done. Time grid:\n   ";
	for (double tn : sol.t_vals) {
		std::cerr << " " << tn;
	}
	std::cerr << "\n";
	timer.toc("    Solving with BDF method");
	std::cout << "Starting with position " << y << " at time " << t << "\n";

	newton::options newton_opts;
	newton::status newton_status;
	newton_opts.tol = 1e-4;
	newton_opts.dx_delta = 1e-10;
	newton_opts.maxit = 500;
	newton_opts.time_internals = true;

	struct newton_multistep_helper {
		typedef arma::vec vec_type;
		typedef arma::mat jac_type;

		newton_multistep_helper(functor_type &func_object,
		                        std::size_t system_size,
		                        double func_weight):
			func(func_object), func_weight(func_weight),
			I(system_size, system_size, arma::fill::eye)
		{}

		vec_type fun(const arma::vec &y)
		{ return y + const_part - func_weight*func.fun(t_nplus, y); }

		jac_type jac(const arma::vec &y)
		{ return I - func_weight*func.jac(t_nplus, y); }

	        functor_type func;
		double t_nplus;
		arma::vec const_part;
		double func_weight;

		jac_type I;
	};



	while (t < t1) {
		vec_type interpolation_part;
		interpolation_part.zeros(y0.size());
		for (std::size_t i = 0; i < solver_opts.order; ++i) {
			interpolation_part += bdf_params[i]*history[i];
		}

		// y is y_n, yn_plus = y_{n+1}
		newton_multistep_helper newton_functor(
			func, y.size(), dt*bdf_rhs_weight[solver_opts.order-1]);
		newton_functor.t_nplus = t + dt;
		newton_functor.const_part = interpolation_part;

		vec_type yn_plus = newton::newton_iterate(
			newton_functor, y, newton_opts, newton_status);

		if (newton_status.conv_status != newton::SUCCESS) {
			std::cerr << "Newton iteration failed on BDF system!\n";
			sol.status = INTERNAL_SOLVE_FAILURE;
			return sol;
		}

		t += dt;
		++step;
		y = yn_plus;
		sol.t_vals.push_back(t);
		sol.y_vals.push_back(y);
		history.push_back(y);
	}



	return sol;
}





} // namespace multistep



#endif // MULTISTEP_HPP
