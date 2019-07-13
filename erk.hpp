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

	/// If true, use the current stages and extrapolate to the next time level.
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
	std::vector<double>    err;

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

	rk_output sol;
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
                 const solver_options &solver_opts,
                 int method = erk::CASH_KARP_54, double dt = 1e-6)
{
	solver_coeffs sc = get_coefficients(method);

	assert( verify_solver_coeffs( sc ) && "Invalid solver coefficients!" );

	return erk_guts(func, t0, t1, y0, solver_opts, dt, sc);
}

} // namespace erk


#endif // ERK_HPP
