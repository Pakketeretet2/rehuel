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
   \file erk.hpp

   \brief Functions related to performing time integration with
   explicit Runge-Kutta (RK) methods.
*/
#ifndef ERK_HPP
#define ERK_HPP


#include <cassert>
#include <limits>
#include <iomanip>

#include "arma_include.hpp"
#include "enums.hpp"
#include "my_timer.hpp"
#include "newton.hpp"
#include "options.hpp"



/**
   \brief Contains functions related to implicit Runge-Kutta methods.
 */
namespace erk {

	
#ifdef DEBUG_OUTPUT
constexpr const bool debug = true;
#else
constexpr const bool debug = false;
#endif // DEBUG_OUTPUT

	
typedef arma::mat mat_type;
typedef arma::vec vec_type;


	
/**
   Contains the Butcher tableau plus time step size.
*/
struct solver_coeffs
{
	const char *name; ///< Human-friendly name for the method.
	vec_type b;      ///< weights for the new y-value
	vec_type c;      ///< these set the intermediate time points
	mat_type A;      ///< alpha coefficients in Butcher tableau

	vec_type b2; ///< weights for the new y-value of the embedded RK method

	int order;   ///< Local convergence order for main method
	int order2;  ///< Local convergence order for embedded method

	/// If the method satisfies first-same-as-last (FSAL), set this to
	/// true to enable the optimizations associated with FSAL.
	bool FSAL;

	/// This gamma is used as weight for the current value for embedding.
	double gamma;

	/// This matrix defines the interpolating polynomial, if available.
	mat_type b_interp;
};



/**
   \brief options for the time integrator.
 */
struct solver_options : common_solver_options {

	/// \brief Constructor with default values.
	solver_options() : adaptive_step_size(true),
	                   extrapolate_stage(false)
	{ }

	~solver_options()
	{ }

	/// If true, attempt to perform adaptive time stepping using
	/// an embedded pair.
	bool adaptive_step_size;

	/// If true, use current stages and extrapolate to the next time level.
	/// \note This is hard for RK-methods not based on quadrature, so is
	/// generally not supported.
	bool extrapolate_stage;
};


static std::map<int,std::string> rk_method_to_string = {
	FOREACH_ERK_METHOD(GENERATE_STRING)
};

static std::map<std::string,int> rk_string_to_method = {
	FOREACH_ERK_METHOD(GENERATE_MAP)
};


/**
   \brief a struct that contains time stamps and stages that can be used for
   constructing the solution all time points in the interval (dense output).
*/
struct rk_output
{
	struct counters {
		counters() : attempt(0), reject_err(0) {}

		std::size_t attempt, reject_err;
	};

	int status;

	std::vector<double> t_vals;
	std::vector<vec_type> y_vals;
	std::vector<vec_type> stages;

	std::vector<vec_type> err_est;
	std::vector<double>   err;

	double elapsed_time, accept_frac;

	counters count;
};


/**
   \brief Returns a vector with all method names.
*/
std::vector<std::string> all_method_names();



/**
   \brief Returns coefficients belonging to the given method.

   See erk::rk_methods for all methods.

   \note If the method is not recognized, the coefficients
         returned will not pass verify_solver_coefficients.

   \param method The method to return coefficients for.

   \returns coefficients belonging to given method.
*/
solver_coeffs get_coefficients( int method );


/**
   \brief Default solver options:
*/
solver_options default_solver_options();

	
/**
   \brief Checks if all options are set to sane values.

   \returns true if all options checked out, false otherwise.

*/
bool verify_solver_options( const solver_options &opts );


/**
   \brief Checks whether or not the given coefficients are consistent in size.

   \param sc the coefficients to check.

   \returns true if the coefficients are valid, false otherwise.
*/
bool verify_solver_coeffs( const erk::solver_coeffs &sc );


/**
   \brief Converts a string with a method name to an int.

   \param name A string describing the method.

   \returns the enum corresponding to given method. See \ref rk_methods
*/
int name_to_method( const std::string &name );


/**
   \brief Converts method code to a human-readable string.

   \note This output shall satisfy
         name_to_method( method_to_name( method ) ) == method.

   \param method The method to convert to a name.

   \returns a string literal representing the method.
*/
const char *method_to_name( int method );




// This is copy-pasta from irk.cpp, should be moved I think.
vec_type project_b( double theta, const erk::solver_coeffs &sc );


/**
   \brief Guts of the explicit RK integrator.
   Time-integrates a given ODE from t0 to t1, starting at y0

   t_vals and y_vals shall be unmodified upon failure.

   \param func         Functor of the ODE to integrate
   \param t0           Starting time
   \param t1           Final time
   \param y0           Initial values
   \param solver_opts  Options for the internal solver.
   \param dt           Initial time step size.
   \param sc           Coefficients of the solver.

   \returns an output struct with the solution.
*/
template <typename functor_type> inline
rk_output erk_guts(functor_type &func, double t0, double t1, const vec_type &y0,
                   const solver_options &solver_opts, double dt,
                   const solver_coeffs &sc )
{
	if( t0 + dt > t1 ){
		std::cerr << "    Rehuel: Initial dt (" << dt;
		dt = t1 - t0;
		std::cerr << ") too large for interval! Reducing to "
		          << dt << "\n";
	}

	std::cerr << "    Rehuel: Integrating over interval [ "
	          << t0 << ", " << t1 << " ]...\n"
	          << "            Method = " << sc.name << "\n";


	// Explicit RK methods are a lot simpler.
	// First you compute the k stages explicitly:
	
	my_timer timer;
	double t = t0;
	rk_output sol;
	sol.status = SUCCESS;

	assert(dt > 0 && "Cannot use time step size <= 0!");
	std::size_t Neq = y0.size();
	std::size_t Ns  = sc.b.size();

	vec_type y = y0;
	vec_type yo(Neq);
	// Ks is the stages at the new time step.
	mat_type Ks(Neq,Ns);
	long long int step = 0;
	// For time step size control.
	double dts[3] = {dt, dt, dt}, errs[3] = {0.9,0.9,0.9};

	if( solver_opts.out_interval > 0 ){
		std::cerr  << "    Rehuel: step  t  dt   err\n";
	}

	vec_type err_est = arma::zeros(Neq);
	sol.t_vals.push_back(t);
	sol.y_vals.push_back(y);
	sol.stages.push_back(vectorise(Ks));
	sol.err_est.push_back( err_est );
	sol.err.push_back( 0.0 );

	// This will keep track of error estimates during integration.
	std::size_t min_order = std::min( sc.order, sc.order2 );

	while( t < t1 ) {
		// ****************  Calculate stages:   ************
		// Make sure you stop exactly at t = t1.
		if( t + dt > t1 ){
			dt = t1 - t;
		}
		sol.count.attempt++;

		int integrator_status = 0;

		// Formula for explicit stages are
		// k_i = f(t + ci*dt, y0 + sum_{j=1}^{i-1} A(i,j)*k_j)
		for (std::size_t i = 0; i < Ns; ++i) {
			vec_type tmp = y;
			for (std::size_t j = 0; j < i; ++j) {
				tmp += dt*sc.A(i,j)*Ks.col(j);
			}
			Ks.col(i) = func.fun(t + sc.c(i)*dt, tmp);
		}
	
		// ************* Form solution at t + dt: ***********
		vec_type delta_y   = arma::zeros(Neq);  // Increment to y
		for (std::size_t i = 0; i < Ns; ++i) {
			delta_y   += sc.b(i)*Ks.col(i);
		}
		vec_type y_n   = y + dt*delta_y;
		
		double err = 0.0;
		double new_dt = dt;
		
		// If you have no adaptive step size, error calculation
		// might not be very sensible.
		if (solver_opts.adaptive_step_size) {
			vec_type delta_alt = arma::zeros(Neq);
			for (std::size_t i = 0; i < Ns; ++i) {
				delta_alt += sc.b2(i)*Ks.col(i);
			}
		
			// ************* Error estimate: ***********
	
			double err_tot = 0.0;
			double atol = solver_opts.abs_tol, rtol = solver_opts.rel_tol;
			
			err_est = dt*(delta_alt - delta_y);
			
			for (std::size_t i = 0; i < Neq; ++i) {
				double erri = err_est(i);
				double y0i  = std::fabs(y(i));
				double y1i  = std::fabs(y_n(i));
				double sci  = atol + rtol * std::max(y0i, y1i);
				double add = erri / sci;
				err_tot += add*add;
			}
			assert( err_tot >= 0.0 && "Error cannot be negative!" );
			err = std::sqrt(err_tot / static_cast<double>(Neq));
			
			if (err < machine_precision) {
				err = machine_precision;
			}
			errs[2] = errs[1];
			errs[1] = errs[0];
			errs[0] = err;
			
			// ************* Adaptive time step size control: ***********
			
			// Error is too large to tolerate:
			if (solver_opts.adaptive_step_size && (err > 1.0)) {
				integrator_status = 1;
				sol.count.reject_err++;
			}
			double fac  = 0.9;
			double expt = 1.0 / (1.0 + min_order);
			double err_inv = 1.0 / err;
			double scale_27 = std::pow(err_inv, expt);
			double dt_rat = dts[0] / dts[1];
			double err_frac = errs[1] / errs[0];
			if (errs[1] == 0 || errs[0] == 0) {
				err_frac = 1.0;
			}
			double err_rat = std::pow(err_frac, expt);
			double scale_28 = scale_27 * dt_rat * err_rat;
			double min_scale = std::min(scale_27, scale_28);
			
			if (debug) {
				std::cerr << "    Rehuel: Time step controller:\n"
				          << "            err      = " << err << "\n"
				          << "            err_inv  = " << err_inv << "\n"
				          << "            dt_rat   = " << dt_rat << "\n"
				          << "            err_frac = " << err_frac << "\n"
				          << "            err_rat  = " << err_rat << "\n"
				          << "            scale_27 = " << scale_27 << "\n"
				          << "            scale_28 = " << scale_28 << "\n\n";
			}
			double new_dt = fac * dt * std::min(4.0, min_scale);
			if (solver_opts.max_dt > 0) {
				new_dt = std::min(solver_opts.max_dt, new_dt);
			}
		}
		
		// ********************* Update y and time ***************
		if (solver_opts.out_interval > 0 &&
		    (step % solver_opts.out_interval == 0) ) {
			std::cerr << "    Rehuel: " << step << " " << t
			          << " " <<  dt << " " << err << "\n";
		}
		
		if (!solver_opts.adaptive_step_size || integrator_status == 0) {
			yo = y;
			y  = y_n;
			t += dt;
			++step;
			
			sol.t_vals.push_back(t);
			sol.y_vals.push_back(y_n);
			// Since K is a matrix, it needs to be flattened:
			sol.stages.push_back(arma::vectorise(Ks));
			sol.err_est.push_back(err_est);
			sol.err.push_back(err);
		}

		// **************** Set the new time step size. *********************
		if (solver_opts.adaptive_step_size) {
			dt = new_dt;
		}
		dts[2] = dts[1];
		dts[1] = dts[0];
		dts[0] = dt;
	}
	double elapsed = timer.toc();
	sol.elapsed_time = elapsed;
	sol.accept_frac = static_cast<double>(step) / sol.count.attempt;
	return sol;
}

	
/**
   \brief Time-integrate a given ODE from t0 to t1, starting at y0

   t_vals and y_vals shall be unmodified upon failure.

   \param func         Functor of the ODE to integrate
   \param t0           Starting time
   \param t1           Final time
   \param y0           Initial values
   \param dt           Initial time step size.
   \param solver_opts  Options for the internal solver.

   \returns a status code (see \ref odeint_status_codes)
*/
template <typename functor_type> inline
rk_output odeint(functor_type &func, double t0, double t1, const vec_type &y0,
                 solver_options solver_opts,
                 int method = erk::CASH_KARP_54, double dt = 1e-6)
{
	solver_coeffs sc = get_coefficients(method);
	if (solver_opts.adaptive_step_size && sc.b2.size() == 0) {
		std::cerr << "    Rehuel: WARNING: Cannot have adaptive time "
		          << "step with non-embedding method! Disabling "
		          << "adaptive time step size!\n";
		solver_opts.adaptive_step_size = false;
	}

	assert( verify_solver_coeffs( sc ) && "Invalid solver coefficients!" );

	return erk_guts(func, t0, t1, y0, solver_opts, dt, sc);
}

} // namespace erk


#endif // ERK_HPP
