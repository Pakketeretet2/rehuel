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
   \file bessel.cpp

   \brief This file contains code to solve bessel's equation as example.

   Compile as g++ -O3 -I../ -lrehuel -larmadillo bessel.cpp -o bessel
*/

#include <iostream>

#include "rehuel.hpp"


struct bessel_functor {
	/*
	  Bessel's equation:
	  t^2*y'' + t*y' + (t^2-a^2)*y = 0

	  Define u = y'
	  then
	  y' = u
	  u' = -u/t - (1 - a^2/t^2)*y = 0
	 */
	typedef arma::mat jac_type;

	bessel_functor() : a(1.0) {}

	arma::vec fun( double t, const arma::vec &Y ) const
	{
		double y = Y(0);
		double u = Y(1);

		arma::vec F(2);
		F(0) = u;
		F(1) = -u/t - (1.0 - a*a/t/t)*y;

		return F;
	}

	
	jac_type jac( double t, const arma::vec &Y ) const
	{
		jac_type A = {{0.0, 1.0},
		              {a*a/t/t, -1.0/t}};
		return A;
	}

	double a;
};


int main(int argc, char **argv)
{
	arma::vec y0 = {4.0, 0.0};
	bessel_functor bf;

	newton::options newton_opts;
	irk::solver_options irk_opts = irk::default_solver_options();
	erk::solver_options erk_opts = erk::default_solver_options();

	irk_opts.rel_tol = 1e-5;
	irk_opts.abs_tol = 1e-4;
	irk_opts.newton_opts = &newton_opts;

	erk_opts.rel_tol = 1e-5;
	erk_opts.abs_tol = 1e-4;

	double t0 = 1.0;
	double t1 = 5e2;

	my_timer timer(std::cerr);
	irk::rk_output irk_sol = irk::odeint(bf, t0, t1, y0, irk_opts,
	                                     irk::RADAU_IIA_53);
	timer.toc("Solving with RADAU IIA-53");
	timer.tic();
	erk::rk_output erk_sol = erk::odeint(bf, t0, t1, y0, erk_opts,
	                                     erk::CASH_KARP_54);
	timer.toc("Solving with CASH-KARP 54");
	std::cerr << "IRK solver: Made " << irk_sol.count.attempt
	          << " attempted steps, of which " << irk_sol.count.reject_newton
	          << " got rejected because of newton, "	          << irk_sol.count.reject_err
	          << " because of err.\n";

	
	std::cerr << "ERK solver: Made " << erk_sol.count.attempt
	          << " attempted steps, of which " << erk_sol.count.reject_err
	          << " were rejected because of err.\n";
	
	std::ofstream irk_out("bessel_irk.dat");
	std::ofstream erk_out("bessel_erk.dat");
	
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
