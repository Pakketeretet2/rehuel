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
   \file irk.hpp

   \brief Contains functions related to performing time integration with
   Runge-Kutta (RK) methods.
*/

#ifndef IRK_HPP
#define IRK_HPP

#define ARMA_USE_CXX11
#include <armadillo>
#include <cassert>
#include <limits>
#include <iomanip>

#include "enums.hpp"
#include "integrator_io.hpp"
#include "my_timer.hpp"
#include "newton.hpp"
#include "options.hpp"

/**
   \brief Namespace containing functions related to Runge-Kutta methods.
 */
namespace irk {

/**
   Contains the Butcher tableau plus time step size.
*/
struct solver_coeffs
{
	const char *name; ///< Human-friendly name for the method.
	arma::vec b; ///< weights for the new y-value
	arma::vec c; ///< these set the intermediate time points
	arma::mat A; ///< alpha coefficients in Butcher tableau

	arma::vec b2; ///< weights for the new y-value of the embedded RK method

	int order;   ///< Local convergence order for main method
	int order2;  ///< Local convergence order for embedded method

	/// If the method satisfies first-same-as-last (FSAL), set this to
	/// true to enable the optimizations associated with FSAL.
	bool FSAL;

	/// This gamma is used as weight for the current value for embedding.
	double gamma;

	/// This matrix defines the interpolating polynomial, if available.
	arma::mat b_interp;
};


/**
   \brief options for the time integrator.
 */
struct solver_options : common_solver_options {
	/// \brief Enumerates the possible internal non-linear solvers
	enum internal_solvers {
		BROYDEN = 0, ///< Broyden's method
		NEWTON = 1   ///< Newton's method
	};

	/// \brief Constructor with default values.
	solver_options() : adaptive_step_size(true),
	                   use_newton_iters_adaptive_step( false )
	{ }

	~solver_options()
	{ }

	/// If true, attempt to perform adaptive time stepping using
	/// an embedded pair.
	bool adaptive_step_size;

	/// If true, use newton iteration info in determining adaptive step size
	bool use_newton_iters_adaptive_step;
};


static std::map<int,std::string> rk_method_to_string = {
	FOREACH_RK_METHOD(GENERATE_STRING)
};

static std::map<std::string,int> rk_string_to_method = {
	FOREACH_RK_METHOD(GENERATE_MAP)
};


/**
   \brief a struct that contains time stamps and stages that can be used for
   constructing the solution all time points in the interval (dense output).
*/
struct rk_output
{
	int status;

	std::vector<double> t_vals;
	std::vector<arma::vec> y_vals;
	std::vector<arma::vec> stages;

	std::vector<arma::vec> err_est;
	std::vector<double>    err;

	std::size_t n_jac_evals, n_func_evals;
};



/**
   \brief Returns a vector with all method names.
*/
std::vector<std::string> all_method_names();


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
bool verify_solver_coeffs( const solver_coeffs &sc );


/**
   \brief Returns coefficients belonging to the given method.

   See irk::rk_methods for all methods.

   For solving stiff systems, we recommend using a solver that is
   L-stable and has support for adaptive time stepping.

   \note If the method is not recognized, the coefficients
         returned will not pass verify_solver_coefficients.

   \param method The method to return coefficients for.

   \returns coefficients belonging to given method.
*/
solver_coeffs get_coefficients( int method );


/**
   \brief Checks if the given method is explicit.
*/
bool is_method_explicit( const solver_coeffs &sc );


/**
   \brief Checks if the given method is diagonally implicit.
*/
bool is_method_dirk( const solver_coeffs &sc );


/**
   \brief Checks if the given method is singly diagonally implicit.
*/
bool is_method_sdirk( const solver_coeffs &sc );


/**
   \brief Returns default solver options.
   \returns default solver options.
*/
solver_options default_solver_options();


/**
   \brief Attempts to find a more optimal time step size based on error estimate

   \param dt_n           Time step size of last taken step
   \param dt_nm          Time step size of previous taken step.
   \param abs_err        Absolute error estimate
   \param rel_err        Current error estimate
   \param newton_iters   Number of Newton iterations used.
   \param opts           Solver options
   \param sc             Solver coefficients.
   \param max_dt         Largest accepted time step size.


   \return A time step size that is estimated to be more optimal.
*/
double get_better_time_step( double dt_n, double dt_nm, double err,
                             double old_err, double tol,
                             int newton_iters, int maxit,
                             const solver_options &opts,
                             const solver_coeffs &sc, double max_dt );

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





/**
   \brief Constructs a non-linear system the stages have to satisfy.

   \param t    Current time
   \param y    Current solution to the ODE at t
   \param K    Initial guess for the stages
   \param sc   Solver coefficients
   \param fun  The RHS of the ODE
   \param jac  The Jacobian of the RHS of the ODE.

   \returns the value for the non-linear system whose root is the new stages.
*/
template <typename functor_type> inline
arma::vec construct_F( double t, const arma::vec &y, const arma::vec &K,
                       double dt, const irk::solver_coeffs &sc,
                       functor_type &func )
{
	auto Ns  = sc.b.size();
	auto Neq = y.size();
	auto NN = Ns*Neq;

	arma::vec F( NN );

	assert( K.size() == NN  && "Size of K is not right!" );
	assert( y.size() == Neq && "Size of y is not right!" );

	const arma::vec &c = sc.c;
	const arma::mat &A = sc.A;

	for( unsigned int i = 0; i < Ns; ++i ){
		double ti = t + dt * c[i];
		arma::vec yi = y;
		arma::vec delta;
		delta.zeros( Neq );
		for( unsigned int j = 0; j < Ns; ++j ){
			unsigned int offset = j*Neq;
			delta += A(i,j) * K.subvec( offset, offset + Neq - 1 );
		}
		yi += dt*delta;
		arma::vec ki = K.subvec( i*Neq, i*Neq + Neq - 1 );
		arma::vec tmp = func.fun( ti, yi );
		F.subvec( i*Neq, i*Neq + Neq - 1 ) = tmp - ki;
	}
	return F;
}

/**
   \brief Constructs the Jacobi matrix of the non-linear system for the stages

   \param t    Current time
   \param y    Current solution to the ODE at t
   \param K    Initial guess for the stages
   \param sc   Solver coefficients
   \param fun  The RHS of the ODE
   \param jac  The Jacobian of the RHS of the ODE.

   \returns the Jacobi matrix of the non-linear system for the new stages.
*/
template <typename functor_type> inline
typename functor_type::jac_type construct_J( double t, const arma::vec &y,
                                             const arma::vec &K,
                                             double dt,
                                             const irk::solver_coeffs &sc,
                                             functor_type &func )
{
	auto Ns  = sc.b.size();
	auto Neq = y.size();
	auto NN = Ns*Neq;

	arma::vec F( NN );

	assert( K.size() == NN  && "Size of K is not right!" );
	assert( y.size() == Neq && "Size of y is not right!" );

	const arma::vec &c = sc.c;
	const arma::mat &A = sc.A;

	typename functor_type::jac_type J( NN, NN );
	J.eye( NN, NN );
	J *= -1.0;

	// i is column, j is row.
	for( unsigned int i = 0; i < Ns; ++i ){
		double ti = t + dt * c[i];
		arma::vec yi = y;

		for( unsigned int j = 0; j < Ns; ++j ){
			unsigned int offset = j*Neq;
			yi += dt * A(i,j) * K.subvec( offset, offset + Neq - 1 );
		}
		auto Ji = func.jac( ti, yi );

		// j is row.
		for( unsigned int j = 0; j < Ns; ++j ){
			// Block i*Neq by j*Neq has to be filled with
			// d F(t + ci, y + dt*sum_{k=0}^N-1 (a_{i,k}*k_k)) / d k_j
			// which is
			// F(t + ci, y + sum_{k=0}^N-1 (a_{i,k}*k_k))' * a_{i,j}*dt
			auto Jc = J.submat( i*Neq, j*Neq, i*Neq + Neq - 1,
			                    j*Neq + Neq - 1 );
			double a_part = dt * A(i,j);
			Jc += Ji * a_part;
		}
	}

	return J;
}




/**
   \brief Time-integrate a given ODE from t0 to t1, starting at y0

   t_vals and y_vals shall be unmodified upon failure.

   \param func         Functor of the ODE to integrate
   \param t0           Starting time
   \param t1           Final time
   \param y0           Initial values
   \param sc           Solver coefficients
   \param solver_opts  Options for the internal solver.

   \returns a status code (see \ref odeint_status_codes)
*/
template <typename functor_type> inline
rk_output radau_IIA_53( functor_type &func, double t0, double t1, const arma::vec &y0,
                        const solver_options &solver_opts, double dt = 1e-6 )
{
	double t = t0;
	rk_output sol;
	sol.status = SUCCESS;

	sol.t_vals.push_back(t);
	sol.y_vals.push_back(y0);

	solver_coeffs sc = get_coefficients( irk::RADAU_IIA_53 );

	assert( strcmp( sc.name, "RADAU_IIA_53" ) == 0 &&
	        "For some reason get_coefficients returned wrong coeffs!" );
	assert( verify_solver_coeffs( sc ) && "Invalid solver coefficients!" );
	assert( solver_opts.newton_opts && "Newton solver options not set!" );
	assert( dt > 0 && "Cannot use time step size <= 0!" );

	const newton::options &newton_opts = *solver_opts.newton_opts;

	std::size_t Neq = y0.size();
	std::size_t Ns  = sc.b.size();
	std::size_t N   = Neq * Ns;

	arma::vec y = y0;
	arma::vec K_np, K_n;
	K_np.zeros( N );
	K_n.zeros( N );

	newton::status newton_stats;
	long long int step = 0;

	double dts[3];
	double errs[3];
	dts[0] = dts[1] = dts[2] = dt;
	errs[0] = errs[1] = errs[2] = 0.9;

	std::cerr  << "    Rehuel: step  t  dt   err   iters\n";

	std::size_t n_attempt = 0;
	std::size_t n_reject_newton = 0;
	std::size_t n_reject_err    = 0;

	std::ofstream err_info( "err.log" );
	err_info << "# step time  err  iters\n";

	bool alternative_error_formula = true;

	// Make sure you stop exactly at t = t1.
	while( t < t1 ){

		// ****************  Calculate stages:   ************
		if( t + dt > t1 ){
			dt = t1 - t;
		}
		++n_attempt;

		int integrator_status = 0;

		// Use newton iteration to find the Ks for the next level:
		auto stages_func = [&t, &y, &dt, &sc, &func]( const arma::vec &K ){
			return construct_F( t, y, K, dt, sc, func );
		};

		auto stages_jac = [&t, &y, &dt, &sc, &func]( const arma::vec &K ){
			return construct_J( t, y, K, dt, sc, func );
		};

		newton::newton_lambda_wrapper<decltype(stages_func),
		                              decltype(stages_jac),
		                              typename functor_type::jac_type>
			nw( stages_func, stages_jac );

		K_np = newton::newton_iterate( nw, K_n, newton_opts,
		                               newton_stats, true /* false */ );
		int newton_status = newton_stats.conv_status;

		// *********** Verify Newton iteration convergence ************
		if( newton_status != newton::SUCCESS ){
			dt *= 0.1;
			++n_reject_newton;
			continue;
		}


		// ****************  Construct solution at t + dt   ************
		arma::vec delta_y, delta_alt;
		std::size_t Neq = y.size();
		double gamma = sc.gamma;
		delta_y.zeros( Neq );
		delta_alt.zeros( Neq );

		for( std::size_t i = 0; i < Ns; ++i ){
			std::size_t i0 = i*Neq;
			std::size_t i1 = (i+1)*Neq - 1;
			auto Ki = K_np.subvec( i0, i1 );
			delta_y   += dt * sc.b[i]  * Ki;
			delta_alt += dt * sc.b2[i] * Ki;
		}

		arma::vec dy_alt = dt*gamma * func.fun( t, y ) + delta_alt;
		arma::vec y_n    = y + delta_y;
		arma::vec yp     = y + dy_alt;
		arma::vec delta_delta = dy_alt - delta_y;


		/*
		std::cerr << "    Rehuel: old y, new y, y' and dy:\n";
		for( std::size_t i = 0; i < y_n.size(); ++i ){
			std::cerr << "        " << y[i] << "   " << y_n[i] << "   " << yp[i]
			          << "  " << std::scientific << delta_delta[i]
			          << "\n";
		}
		*/
		// **************      Estimate error:    **********************
		arma::mat I, J0;
		I.eye(Neq, Neq);
		J0 = func.jac( t, y );
		// Formula 8.19:
		arma::vec err_8_19 = arma::solve( I - gamma * dt * J0,
		                                  delta_delta );
		arma::vec err_est = err_8_19;


		// Alternative formula 8.20:
		if( alternative_error_formula ){
			// Use the alternative formulation:
			arma::vec dy_alt_alt = dt*gamma*func.fun(t, y+err_est);
			dy_alt_alt += delta_alt;
			arma::vec err_alt = dy_alt_alt - delta_y;
			err_est = arma::solve( I - gamma*dt*J0, err_alt );
		}

		double err_tot = 0.0;
		double n = 0.0;
		for( std::size_t i = 0; i < err_est.size(); ++i ){
			double erri = err_est[i];
			double sci  = solver_opts.abs_tol;
			double y0i  = std::fabs( y[i] );
			double y1i  = std::fabs( y_n[i] );

			sci += solver_opts.rel_tol * std::max( y0i, y1i );
			err_tot += erri*erri / sci / sci;
			n += 1.0;
		}

		double err = std::sqrt( err_tot / n );

		assert( err >= 0.0 && "Error cannot be negative!" );
		if( err < machine_precision ){
			err = machine_precision;
		}

		errs[2] = errs[1];
		errs[1] = errs[0];
		errs[0] = err;

		if( err > 1.0 ){
			// This is bad.
			alternative_error_formula = true;
			integrator_status = 1;
			++n_reject_err;


			err_info << step << " " << t << " " << dt << " "
			         << err << " " << newton_stats.iters << "\n";
		}


		// **************      Find new dt:    **********************


		double fac = 0.9 * ( newton_opts.maxit + 1.0 );
		fac /= ( newton_opts.maxit + newton_stats.iters );

		double expt = 1.0 / ( 1.0 + std::min( sc.order, sc.order2 ) );
		double err_inv = 1.0 / err;
		double scale_27 = std::pow( err_inv, expt );
		double dt_rat = dts[0] / dts[1];
		double err_frac = errs[1] / errs[0];
		double err_rat = std::pow( err_frac, expt );
		double scale_28 = scale_27 * dt_rat * err_rat;
		/*
		std::cerr << "    Rehuel: Time step controller:\n"
		          << "            err      = " << err << "\n"
		          << "            err_inv  = " << err_inv << "\n"
		          << "            dt_rat   = " << dt_rat << "\n"
		          << "            err_frac = " << err_frac << "\n"
		          << "            err_rat  = " << err_rat << "\n"
		          << "            scale_27 = " << scale_27 << "\n"
		          << "            scale_28 = " << scale_28 << "\n\n";
		*/
		double new_dt = fac * dt * std::min( scale_27, scale_28 );

		// **************    Update y and time   ********************

		if( solver_opts.out_interval > 0 &&
		    (step % solver_opts.out_interval == 0) ){
			std::cerr  << "    Rehuel: " << step << " " << t
			           << " " <<  dt << " " << err << " "
			           << newton_stats.iters << "\n";
		}


		if( integrator_status == 0 ){

			y = y_n;
			t += dt;
			++step;

			sol.t_vals.push_back(t);
			sol.y_vals.push_back(y_n);
			sol.stages.push_back(K_np);
			sol.err_est.push_back( err_est );
			sol.err.push_back( err );

			alternative_error_formula = false;

			if( err < 0.01 ){
				change_dt = true;
			}else if( err >= 1.0 ){
				change_dt = true;
			}
		}


			err_info << step << " " << t << " " << dt << " "
			         << err << " " << newton_stats.iters << "\n";

		}

		// **************      Actually set the new dt:    **********************

		dt = new_dt;
		dts[2] = dts[1];
		dts[1] = dts[0];
		dts[0] = dt;

		if( integrator_status == 0 ){

			// ***************   Estimate new stages:  *************
		}
	}

	double accept_rat = static_cast<double>(step) / n_attempt;

	std::cerr << "    Rehuel: Done integrating ODE over [ " << t0 << ", "
	          << t1 << " ].\n";
	std::cerr << "            Number of succesful steps: " << step
	          << " / " << n_attempt
	          << ". Accept ratio = " << accept_rat
	          << ", rejected due to newton: " << n_reject_newton
	          << ", rejected due to err: " << n_reject_err << "\n";

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
rk_output odeint( functor_type &func, double t0, double t1, const arma::vec &y0,
                  const solver_options &solver_opts, double dt = 1e-6 )
{
	return radau_IIA_53( func, t0, t1, y0, solver_opts );
}



} // namespace irk


#endif // IRK_HPP
