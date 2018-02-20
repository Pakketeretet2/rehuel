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
	double dt;   ///< time step size to use

	arma::vec b2; ///< weights for the new y-value of the embedded RK method

	int order;   ///< Local convergence order for main method
	int order2;  ///< Local convergence order for embedded method

	/// If the method satisfies first-same-as-last (FSAL), set this to
	/// true to enable the optimizations associated with FSAL.
	bool FSAL;

	/// True if the method has an interpolating formula for dense output.
	bool dense_output;
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
   \brief This allows for optimizations when determining stages.
*/
enum method_traits { IMPLICIT = 0,
                     EXPLICIT = 1,
                     SDIRK    = 2 };


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

   \param dt_old         Old time step size
   \param abs_err        Absolute error estimate
   \param rel_err        Current error estimate
   \param newton_iters   Number of Newton iterations used.
   \param opts           Solver options
   \param sc             Solver coefficients.
   \param max_dt         Largest accepted time step size.


   \return A time step size that is estimated to be more optimal.
*/
double get_better_time_step( double dt_old, double err, double old_err,
                             double tol, int newton_iters, int n_rejected,
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
		for( unsigned int j = 0; j < Ns; ++j ){
			unsigned int offset = j*Neq;
			yi += dt * A(i,j) * K.subvec( offset, offset + Neq - 1 );
		}
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
   \brief A wrapper struct to make Newton iteration easier.

*/
template <typename f_func, typename J_func, typename Jac_type>
struct newton_wrapper
{
	typedef Jac_type jac_type;

	newton_wrapper( f_func &f, J_func &J ) : f(f), J(J) {}

	arma::vec fun( const arma::vec &K )
	{
		return f(K);
	}

	jac_type jac( const arma::vec &K )
	{
		return J(K);
	}

	f_func &f;
	J_func &J;
};

/**
   \brief Calculates the RK stages.

*/
template <typename functor_type, int method_type> inline
arma::vec get_rk_stages( double t, const arma::vec &y, double dt,
                         const solver_coeffs &sc,
                         const solver_options &solver_opts,
                         functor_type &func, newton::status &stats,
                         const arma::vec &K, bool &success )
{
	arma::vec KK = K;
	const newton::options &opts = *solver_opts.newton_opts;
	my_timer timer_step( std::cerr );

	std::size_t Neq = y.size();
	std::size_t Ns  = sc.b.size();

	// Check if the method is explicit or implicit:

	if( method_type & EXPLICIT ){
		// No need to solve with Newton:
		for( std::size_t i = 0; i < Ns; ++i ){
			std::size_t offset = i*Neq;
			arma::vec yi = y;
			for( std::size_t j = 0; j < i; ++j ){
				std::size_t delta = j*Neq;
				auto Ki = KK.subvec( delta, delta + Neq - 1 );
				yi += dt * sc.A(i,j) * Ki;
			}
			KK.subvec( offset, offset + Neq - 1 )
				= func.fun( t + sc.c[i]*dt, yi );
		}
		success = true;

	}else if( false && (method_type & SDIRK) ){
		// In this case the solution is simplified, because Newton
		// iteration simplifies to Ns times solving a system of NeqxNeq

		auto single_stage_func = [&t, &y, &dt, &sc, &func]
			(const arma::vec &k1, const arma::vec &k_rest ){
			return func.fun( t + sc.c[0]*dt, y + dt*sc.A(0,0)*k1 + k_rest ) - k1;
		};

		typename functor_type::jac_type I_k1( y.size(), y.size() );
		I_k1.eye();

		auto single_stage_jac = [&t, &y, &dt, &sc, &func, &I_k1]
			( const arma::vec &k1, const arma::vec &k_rest ){
			return func.jac( t + sc.c[0]*dt, y + dt*sc.A(0,0)*k1 + k_rest ) - I_k1;
		};


		for( std::size_t i = 0; i < Ns; ++i ){
			arma::vec k_rest;
			k_rest.zeros( Neq );
			for( std::size_t j = 0; j < i; ++j ){
				std::size_t delta = j*Neq;
				auto Ki = KK.subvec( delta, delta + Neq - 1 );
				k_rest += dt * sc.A(i,j) * Ki;
			}

			auto ki_func = [&single_stage_func, &k_rest]( const arma::vec &ki ){
				return single_stage_func( ki, k_rest ); };
			auto ki_jac  = [&single_stage_jac, k_rest]( const arma::vec &ki ){
				return single_stage_jac( ki, k_rest ); };

			newton_wrapper<decltype(ki_func),
			               decltype(ki_jac),
			               typename functor_type::jac_type>
				nw_ki( ki_func, ki_jac );

			arma::vec ki_sol;

			switch( solver_opts.internal_solver ){
				case solver_options::NEWTON:
					ki_sol = newton::newton_iterate( nw_ki, k_rest,
					                                 opts, stats );
				default:
				case solver_options::BROYDEN:
					ki_sol = newton::broyden_iterate( nw_ki, k_rest,
					                                  opts, stats );
					break;
			}

			success = stats.conv_status == newton::SUCCESS;
			if( !success ) return KK;
			std::size_t offset = i * Neq;
			std::size_t delta  = Neq;
			KK.subvec(offset, offset + delta - 1) = ki_sol;
		}
	}else{

		// For the SDIRK, use constant J but keep same for the rest.

		auto J = construct_J( t, y, KK, dt, sc, func );

		// Use newton iteration to find the Ks for the next level:
		auto stages_func = [&t, &y, &dt, &sc, &func]( const arma::vec &K ){
			return construct_F( t, y, K, dt, sc, func );
		};

		auto stages_jac = [&t, &y, &dt, &sc, &func]( const arma::vec &K ){
			return construct_J( t, y, K, dt, sc, func );
		};

		auto stages_jac_const = [&J]( const arma::vec &K ){
			return J;
		};

		newton_wrapper<decltype(stages_func), decltype(stages_jac_const),
		               typename functor_type::jac_type>
			nw_const_J( stages_func, stages_jac_const );

		newton_wrapper<decltype(stages_func), decltype(stages_jac),
		               typename functor_type::jac_type>
			nw_update_jac( stages_func, stages_jac );


		if( solver_opts.constant_jac_approx ){
			switch( solver_opts.internal_solver ){
				case solver_options::NEWTON:
					KK = newton::newton_iterate( nw_const_J, K,
					                             opts, stats );
					break;
				default:
				case solver_options::BROYDEN:
					KK = newton::broyden_iterate( nw_const_J, K,
					                              opts, stats );
					break;
			}
		}else{
			switch( solver_opts.internal_solver ){
				case solver_options::NEWTON:
					KK = newton::newton_iterate( nw_update_jac, K,
					                             opts, stats );
					break;
				default:
				case solver_options::BROYDEN:
					KK = newton::broyden_iterate( nw_update_jac, K,
					                              opts, stats );
					break;
			}
		}
		success = stats.conv_status == newton::SUCCESS;
	}


	return KK;
}


/**
   \brief Performs one time step from (t,y) to (t+dt, y+dy)

   K and y shall be unmodified upon failure

   \param t             Current time
   \param y             Current solution to the ODE at t
   \param dt            Current time step size
   \param sc            Solver coefficients
   \param solver_opts   Options for the solver
   \param fun           The RHS of the ODE
   \param jac           The Jacobian of the RHS of the ODE.
   \param adaptive_dt   If true, also update embedded pair.
   \param err           Will contain an error estimate, if available.
   \param K             Will contain the new stages on success.

   \returns a status code (see \ref odeint_status_codes)
*/
template <typename functor_type, int method_type> inline
int take_time_step( double t, arma::vec &y, double dt,
                    newton::status &stats,
                    const irk::solver_coeffs &sc,
                    const irk::solver_options &solver_opts,
                    functor_type &func,
                    bool adaptive_dt, double &err, arma::vec &K )
{
	std::size_t Neq = y.size();
	std::size_t Ns  = sc.b.size();

	bool success = false;
	arma::vec KK = get_rk_stages<functor_type, method_type>( t, y, dt, sc,
	                                                         solver_opts,
	                                                         func, stats, K,
	                                                         success );
	if( solver_opts.abort_on_solver_fail && !success ){
		std::cerr << "Internal solver failed! Aborting!\n";
		return GENERAL_ERROR;
	}
	bool increase_dt = false;


	if( success ){
		arma::vec y_alt;
		// First, do adaptive time step:
		if( adaptive_dt ){
			y_alt = y;
			for( unsigned int i = 0; i < Ns; ++i ){
				unsigned int offset = i*Neq;
				const auto &Ki = KK.subvec( offset,
				                            offset + Neq - 1 );
				y_alt += dt * sc.b2[i] * Ki;
			}
		}

		arma::vec yn = y;

		for( unsigned int i = 0; i < Ns; ++i ){
			unsigned int offset = i*Neq;
			const auto &Ki = KK.subvec( offset, offset + Neq - 1 );
			yn += dt * sc.b[i] * Ki;
		}

		if( adaptive_dt ){
			arma::vec y_err = yn - y_alt;
			double ynorm = arma::norm( yn, "inf" );

			// Use absolute error:
			double abs_err = arma::norm( y_err, "inf" );
			// Use relative error:
			arma::vec rel_y_err = y_err;
			for( std::size_t i = 0; i < y_err.size(); ++i ){
				double yi_abs = std::abs( yn[i] );
				double ya_abs = std::abs( y_alt[i] );
				double y_min = std::min( yi_abs, ya_abs );
				rel_y_err[i] /= y_min;
			}

			bool use_rel_tol = false;
			double rel_err = arma::norm( rel_y_err, "inf" );
			double tol = solver_opts.abs_tol;
			if( ynorm < solver_opts.abs_tol ){
				err = abs_err;
			}else{
				use_rel_tol = true;
				err = rel_err;
				tol = solver_opts.rel_tol;
			}

			if( err < 0.1*tol ){
				increase_dt = true;
			}else if( err > tol ){
				// Returning here prevents y from being updated.
				int retcode = DT_TOO_LARGE;
				if( use_rel_tol )
					retcode |= ERROR_LARGER_THAN_RELTOL;
				else
					retcode |= ERROR_LARGER_THAN_ABSTOL;
				return retcode;
			}
		}

		// If you get here, either adaptive_dt == false
		// or everything is fine.
		// Store new y:
		y = yn;
		int ret_code = 0;

		if( increase_dt )      ret_code |= DT_TOO_SMALL;
		else                   ret_code |= SUCCESS;
		if( stats.iters < 10 ) ret_code |= INTERNAL_SOLVE_FEW_ITERS;



		K = KK; // Store the new stages.
		return ret_code;

	}else{
		// At this point, y has not changed. Neither has K. Only err.
		if( solver_opts.verbosity ){
			std::cerr << "Internal solver failed to converge at "
			          << "t = " << t << ". Final dt = " << dt << ", final res = "
			          << stats.res << "\n";
		}
		return INTERNAL_SOLVE_FAILURE;
	}

	return SUCCESS;
}


/**
   \brief Time-integrate a given ODE from t0 to t1, starting at y0

   t_vals and y_vals shall be unmodified upon failure.

   \param t0           Starting time
   \param t1           Final time
   \param sc           Solver coefficients
   \param solver_opts  Options for the internal solver
   \param y0           Initial values
   \param func         Functor of the ODE to integrate

   \returns a status code (see \ref odeint_status_codes)
*/
template <typename functor_type> inline
int odeint( double t0, double t1, const solver_coeffs &sc,
            const solver_options &solver_opts,
            const arma::vec &y0, functor_type &func )
{
	double t = t0;
	assert( verify_solver_coeffs( sc ) && "Invalid solver coefficients!" );
	assert( solver_opts.newton_opts && "Newton solver options not set!" );
	assert( sc.dt > 0 && "Cannot use time step size of 0!" );


	double dt = sc.dt;
	double err = 0.0;
	double old_err = err;
	bool adaptive_dt = solver_opts.adaptive_step_size;

	if( adaptive_dt ){
		if( sc.b2.size() != sc.b.size() ){
			std::cerr << "WARNING: Adaptive time step requested "
			          << "but chosen method has no embedded "
			          << " pair!\n";
			adaptive_dt = false;
		}
	}else if( solver_opts.verbosity ){
		std::cerr << "Not using adaptive time step.\n";
	}

	arma::vec y = y0;
	std::vector<double> tt;
	std::vector<arma::vec> yy;
	unsigned long long int step = 0;

	newton::status newton_stats; // Use these for adaptive time step control.


	// Prepare output if necessary:
	if( solver_opts.output ){

		auto *output = solver_opts.output;
		output->set_storage_size( sc.order + 1 );
		output->add_solution( t, y );
		output->write_solution();

		output->store_vector_solution( step, t, y );
	}



	// Main integration loop:
	int status = 0;

	my_timer timer( std::cerr );
	timer.tic();

	// Prepare stages:
	std::size_t Neq = y.size();
	std::size_t Ns  = sc.b.size();
	std::size_t NN = Ns * Neq;

	arma::vec K = arma::zeros( NN, 1 );
	for( std::size_t i = 0; i < Ns; ++i ){
		std::size_t i0 = i*Neq;
		std::size_t i1 = (i+1)*Neq - 1;
		K.subvec( i0, i1 ) = func.fun( t, y );
	}

	// Grab max_dt for convenience:
	double max_dt = solver_opts.max_dt;

	bool explicit_method = irk::is_method_explicit( sc );
	bool sdirk_method    = irk::is_method_sdirk( sc );

	int rejected_step = 0;
	int latest_n_rejected_step = 0;

	int method_type = 0;
	if( explicit_method ){
		method_type += EXPLICIT;
	}
	if( sdirk_method ){
		method_type += SDIRK;
	}

	while( t < t1 ){

		bool change_dt = false;
		double old_dt = dt;
		old_err = err;

		// Check if you need to lower dt to exactly hit
		if( t + dt > t1 ){
			dt = t1 - t;
			old_dt = dt;
			std::cerr << "    Rehuel: This is probably the last step, t = "
			          << t << " and dt = " << dt << "!\n";
		}

		/*
		if( solver_opts.output ){
			double to = solver_opts.output->next_output_time_step();
			if( t + dt > to ){
				dt = to - t;
				old_dt = dt;
			}
		}
		*/

		if( method_type & EXPLICIT ){
			const auto& stepper = take_time_step<functor_type, EXPLICIT>;
			status = stepper( t, y, dt, newton_stats, sc,
			                  solver_opts, func, adaptive_dt,
			                  err, K );

			// To make sure get_better_timestep works:
			newton_stats.iters = 1;
		}else if( method_type & SDIRK ){
			const auto& stepper = take_time_step<functor_type, SDIRK>;
			status = stepper( t, y, dt, newton_stats, sc,
			                  solver_opts, func, adaptive_dt,
			                  err, K );

		}else{
			const auto& stepper = take_time_step<functor_type, IMPLICIT>;
			status = stepper( t, y, dt, newton_stats, sc,
			                  solver_opts, func, adaptive_dt,
			                  err, K );
		}

		if( solver_opts.verbosity ){
			std::cerr << "t = " << t << ", step " << step
			          << ", status = " << status << "\n";
			std::vector<std::pair<int,std::string> > pairs =
				{ {SUCCESS                 , "SUCCESS"},
				  {DT_TOO_SMALL            , "DT_TOO_SMALL            "},
				  {INTERNAL_SOLVE_FEW_ITERS, "INTERNAL_SOLVE_FEW_ITERS"},
				  {GENERAL_ERROR           , "GENERAL_ERROR           "},
				  {DT_TOO_LARGE            , "DT_TOO_LARGE            "},
				  {INTERNAL_SOLVE_FAILURE  , "INTERNAL_SOLVE_FAILURE  "},
				  {ERROR_LARGER_THAN_RELTOL, "ERROR_LARGER_THAN_RELTOL"},
				  {ERROR_LARGER_THAN_ABSTOL, "ERROR_LARGER_THAN_ABSTOL"} };
			for( const auto &p : pairs ){
				if( status & p.first ){
					std::cerr << "    " << p.second << "\n";
				}
			}
		}


		if( status == GENERAL_ERROR ){
			if( solver_opts.verbosity ){
				std::cerr << "Generic error in odeint!\n";
			}
			return GENERAL_ERROR;
		}

		if( status & INTERNAL_SOLVE_FAILURE ){
			status |= DT_TOO_LARGE;
		}

		if( status & DT_TOO_LARGE ){
			change_dt = adaptive_dt;

			if( !change_dt ){
				dt *= 0.3;
			}
		}

		if( status & INTERNAL_SOLVE_FEW_ITERS ){
			// This means you can probably increase max_dt and dt:
			status |= DT_TOO_SMALL;
		}
		if( status & DT_TOO_SMALL ){
			change_dt = adaptive_dt;
		}


		if( change_dt ){
			double ynorm = arma::norm( y, "inf" );
			double rel_tol = solver_opts.rel_tol;
			double abs_tol = solver_opts.abs_tol;

			double tol = abs_tol;
			if( ynorm < 0 ){
				// Yes. For some reason the norm of y
				// overflowed to negative inf.
				ynorm = std::abs(ynorm);
			}

			if( ynorm < abs_tol ){
				tol = rel_tol * ynorm;
			}
			err = std::abs(err); // To take care of -inf. :/
			if( solver_opts.verbosity ){
				if( err > tol ){
					std::cerr << "Err was too big: " << err
					          << ".\nThis was step "
					          << latest_n_rejected_step
					          << " rejected.\n";
				}else{
					std::cerr << "Err was too small: "
					          << err << ".\n";
				}
				std::cerr << "Tol is " << tol << " ("
				          << rel_tol*ynorm
				          << " vs. " << abs_tol << ")\n";
			}
			double old_dt = dt;

			dt = get_better_time_step( dt, err, old_err, tol,
			                           newton_stats.iters,
			                           latest_n_rejected_step,
			                           solver_opts, sc, max_dt );

			if( solver_opts.verbosity ){
				std::cerr << "changing dt from " << old_dt
				          << "... to " << dt << "\n";
			}
			if( status & INTERNAL_SOLVE_FAILURE ){
				if( dt >= old_dt ){
					dt = 0.5*old_dt;
				}
			}
		}
		
		bool move_on = (status == SUCCESS);
		if( status & DT_TOO_SMALL || status & INTERNAL_SOLVE_FEW_ITERS ){
			move_on = true;
		}

		if( !move_on ){
			// Restore err to the old error:
			err = old_err;
			++rejected_step;
			++latest_n_rejected_step;

			continue;
		}


		// OK, advance time:
		t += old_dt;
		++step;
		latest_n_rejected_step = 0;


		// Some FSAL preparation here:
		std::size_t i0 = (Ns-1)*Neq;
		std::size_t i1 = Ns*Neq - 1;
		if( sc.FSAL ){
			K.subvec( 0, Neq - 1 ) = K.subvec( i0, i1 );
		}else{
			K.subvec( 0, Neq - 1 ) = func.fun( t, y );
		}

		for( std::size_t i = 1; i < Ns; ++i ){
			i0 = i*Neq;
			i1 = (i+1)*Neq - 1;
			K.subvec( i0, i1 ) = func.fun( t, y );
		}


		// Output:
		auto *output = solver_opts.output;
		if( output ){
			output->add_solution( t, y );
			output->write_solution();
			output->store_vector_solution( step, t, y );

			output->write_timestep_info( step, t, dt, err, old_err,
			                             newton_stats.iters );

		}
	}

	std::string timer_msg = "    Rehuel: Solving ODE with ";
	timer_msg += sc.name;
	timer_msg += " and internal solver ";
	if( solver_opts.internal_solver == solver_options::NEWTON ){
		timer_msg += "newton";
	}else if( solver_opts.internal_solver == solver_options::BROYDEN ){
		timer_msg += "broyden";
	}
	timer_msg += " (" + std::to_string( step ) + " steps)";

	double t_diff_msec = timer.toc( timer_msg );
	double throughput = 1000*step / t_diff_msec;
	std::cerr << "    Rehuel: Performance: " << throughput
	          << " steps/second.\n\n";

	return 0;
}




} // namespace irk


#endif // IRK_HPP
