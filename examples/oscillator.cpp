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
   \file oscillator.cpp

   \brief This file contains code to solve a harmonic oscillator as example.

   Compile as g++ -O3 -I../ -lrehuel -larmadillo oscillator.cpp -o oscillator
*/

#include <iostream>

#include "rehuel.hpp"


struct oscillator_funtor {
	/*
	  Harmonic oscillator equation with external forcing:
	  m*x'' = -k*x - gamma*x' + f0*sin(w*t);
	  
	  This is converted into a system of two single ODEs:
	  x' := u
	  u' = -(k/m)*x - (gamma/m)*u + (f0/m)*sin(w*t)
	  
	 */
	typedef mat_eigen jac_type;


	oscillator_funtor() : m(1.0), k(1.0), gamma(0.0), f0(0.0), w(1.0) {}

	vec_eigen fun( double t, const vec_eigen &Y ) const
	{
		double x = Y(0);
		double u = Y(1);

		double kk = k / m;
		double gg = gamma / m;
		double ff = f0 / m;

		vec_eigen F(2);
		F(0) = u;
		F(1) = -kk*x - gg*u + ff*sin(w*t);

		return F;
	}

	
	jac_type jac( double t, const vec_eigen &Y ) const
	{
		double kk = k / m;
		double gg = gamma / m;
		double ff = f0 / m;
		
		jac_type A = {{0.0, 1.0},
		              {-kk, ff}};
		return A;
	}

	double m;
	double k, gamma, f0, w;
};


int main(int argc, char **argv)
{
	vec_eigen y0 = {2.0, 0.0};
	oscillator_funtor of;

	// WIth these settings you get a resonance:
	// [m*x''] = [kg m/s^2]
	of.gamma = 0.1; // [gamma*x'] = [kg m/s^2] so [gamma] = [kg/s]
	of.f0 = 2.0;    // [f0] = [kg m/s^2]
	of.w = 1.6;     // [w]  = [1/s]
	of.m = 1.0;
	// [k*x] = [kg m/s^2] so [k] = kg/s^2
	of.k = of.m*of.w*of.w*1.2;

	newton::options newton_opts;
	irk::solver_options irk_opts = irk::default_solver_options();
	erk::solver_options erk_opts = erk::default_solver_options();

	irk_opts.rel_tol = 1e-5;
	irk_opts.abs_tol = 1e-4;
	irk_opts.newton_opts = &newton_opts;

	erk_opts.rel_tol = 1e-5;
	erk_opts.abs_tol = 1e-4;

	double t0 = 0.0;
	double t1 = 5e2;

	my_timer timer(std::cerr);
	irk::rk_output irk_sol = irk::odeint(of, t0, t1, y0, irk_opts,
	                                     irk::RADAU_IIA_53);
	timer.toc("Solving with RADAU IIA-53");
	timer.tic();
	erk::rk_output erk_sol = erk::odeint(of, t0, t1, y0, erk_opts,
	                                     erk::CASH_KARP_54);
	timer.toc("Solving with CASH-KARP 54");
	std::cerr << "IRK solver: Made " << irk_sol.count.attempt
	          << " attempted steps, of which " << irk_sol.count.reject_newton
	          << " got rejected because of newton, "	          << irk_sol.count.reject_err
	          << " because of err.\n";

	
	std::cerr << "ERK solver: Made " << erk_sol.count.attempt
	          << " attempted steps, of which " << erk_sol.count.reject_err
	          << " were rejected because of err.\n";
	
	std::ofstream irk_out("oscillator_irk.dat");
	std::ofstream erk_out("oscillator_erk.dat");
	
	for (std::size_t i = 0; i < irk_sol.t_vals.size(); ++i) {
		irk_out << irk_sol.t_vals[i];
		for (std::size_t j = 0; j < irk_sol.y_vals[i].size(); ++j) {
			irk_out << " " << irk_sol.y_vals[i][j];
		}
		irk_out << "\n";
	}

	
	for (std::size_t i = 0; i < erk_sol.t_vals.size(); ++i) {
		erk_out << erk_sol.t_vals[i];
		for (std::size_t j = 0; j < erk_sol.y_vals[i].size(); ++j) {
			erk_out << " " << erk_sol.y_vals[i][j];
		}
		erk_out << "\n";
	}

	



}
