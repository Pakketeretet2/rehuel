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

#define AMRA_USE_CXX11
#include <armadillo>
#include <cassert>

#include "enums.hpp"
#include "my_timer.hpp"
#include "newton.hpp"

/**
   \brief Namespace containing functions related to Runge-Kutta methods.
 */
namespace irk {

/**
   Contains the Butcher tableau plus time step size.
*/
struct solver_coeffs
{
	arma::vec b; ///< weights for the new y-value
	arma::vec c; ///< these set the intermediate time points
	arma::mat A; ///< alpha coefficients in Butcher tableau
	double dt;   ///< time step size to use

	arma::vec b2; ///< weights for the new y-value of the embedded RK method

	/// If the method satisfies first-same-as-last (FSAL), set this to
	/// true to enable the optimizations associated with FSAL.
	bool FSAL;
};

/**
   \brief options for the time integrator.
 */
struct solver_options {
	/// \brief Enumerates the possible internal non-linear solvers
	enum internal_solvers {
		BROYDEN = 0, ///< Broyden's method
		NEWTON = 1   ///< Newton's method
	};

	/// \brief Constructor with default values.
	solver_options()
		: internal_solver(BROYDEN), adaptive_step_size(true),
		  local_tol(1e-6) {}

	/// Internal non-linear solver used (see \ref internal_solvers)
	/// Broyden typically gives good results in less time.
	int internal_solver;
	/// If true, attempt to perform adaptive time stepping using
	/// an embedded pair.
	bool adaptive_step_size;
	/// Local tolerance to satisfy when adaptive time stepping
	double local_tol;
};

/**
   \brief Checks if the solver Butcher tableau is consistent.
   \param sc The coefficients to check
   \returns true if the coefficients are alright, false otherwise
*/
bool verify_solver_coeffs( const solver_coeffs &sc );

/**
   \brief Returns coefficients for given integrator.
   \param method The time integrator to use. See \ref rk_methods
   \returns solver coefficients for given method.
*/
solver_coeffs get_coefficients( int method );

/**
   \brief Returns default solver options.
   \returns default solver options.
*/
solver_options default_solver_options();


/**
   \brief Attempts to find a more optimal time step size based on error estimate

   \param dt_old           Old time step size
   \param error_estimate   Current error estimate
   \param opts             Solver options

   \return A time step size that is estimated to be more optimal.
*/
double get_better_time_step( double dt_old, double error_estimate,
                             const solver_options &opts );

/**
   \brief Converts a string with a method name to an int.

   \param name A string describing the method.

   \returns the enum corresponding to given method. See \ref rk_methods
*/
int name_to_method( const char *name );


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
template <typename func_type, typename Jac_type> inline
arma::vec construct_F( double t, const arma::vec &y, const arma::vec &K,
                       const irk::solver_coeffs &sc,
                       const func_type &fun, const Jac_type &jac )
{
	auto Ns  = sc.b.size();
	auto Neq = y.size();
	auto NN = Ns*Neq;

	arma::vec F( NN );

	assert( K.size() == NN  && "Size of K is not right!" );
	assert( y.size() == Neq && "Size of y is not right!" );

	const arma::vec &c = sc.c;
	const arma::mat &A = sc.A;
	double dt = sc.dt;


	for( unsigned int i = 0; i < Ns; ++i ){
		double ti = t + dt * c[i];
		arma::vec yi = y;
		for( unsigned int j = 0; j < Ns; ++j ){
			unsigned int offset = j*Neq;
			yi += dt * A(i,j) * K.subvec( offset, offset + Neq - 1 );
		}
		arma::mat Ji = jac( ti, yi );
		arma::vec ki = K.subvec( i*Neq, i*Neq + Neq - 1 );
		F.subvec( i*Neq, i*Neq + Neq - 1 ) = fun( ti, yi ) - ki;
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
template <typename func_type, typename Jac_type> inline
arma::mat construct_J( double t, const arma::vec &y, const arma::vec &K,
                       const irk::solver_coeffs &sc,
                       const func_type &fun, const Jac_type &jac )
{
	auto Ns  = sc.b.size();
	auto Neq = y.size();
	auto NN = Ns*Neq;

	arma::vec F( NN );

	assert( K.size() == NN  && "Size of K is not right!" );
	assert( y.size() == Neq && "Size of y is not right!" );

	const arma::vec &c = sc.c;
	const arma::mat &A = sc.A;
	double dt = sc.dt;

	arma::mat J( NN, NN );
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
		arma::mat Ji = jac( ti, yi );

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
   \brief Performs one time step from (t,y) to (t+dt, y+dy)

   K shall be unmodified upon failure

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
template <typename func_type, typename Jac_type> inline
int take_time_step( double t, arma::vec &y, double dt,
                    const irk::solver_coeffs &sc,
                    const irk::solver_options &solver_opts,
                    const func_type &fun, const Jac_type &jac,
                    bool adaptive_dt, double &err,
                    arma::vec &K )
{
	auto Ns  = sc.b.size();
	auto Neq = y.size();
	auto NN = Ns*Neq;

	arma::mat J( NN, NN );
	arma::vec KK = K;

	// Use newton iteration to find the Ks for the next level:
	auto stages_func = [&t, &y, &sc, &fun, &jac]( const arma::vec &K ){
		return construct_F( t, y, K, sc, fun, jac );
	};
	auto stages_jac  = [&t, &y, &sc, &fun, &jac]( const arma::vec &K ){
		return construct_J( t, y, K, sc, fun, jac );
	};

	newton::options opts;
	newton::status stats;

	opts.tol = 1e-8;
	opts.time_internals = false;
	opts.refresh_jac = false;
	opts.maxit = 10000;

	switch( solver_opts.internal_solver ){
		case solver_options::NEWTON:
			KK = newton::solve( stages_func, K, opts, stats, stages_jac );
			break;
		default:
		case solver_options::BROYDEN:
			KK = newton::solve( stages_func, K, opts, stats );
			break;
	}

	bool increase_dt = false;

	if( stats.conv_status == newton::SUCCESS ){
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
			auto y_err = yn - y_alt;
			double err_est = arma::norm( y_err, "inf" );
			err = err_est;
			if( err > solver_opts.local_tol ){
				return DT_TOO_LARGE;
			}else if( err < solver_opts.local_tol * 0.2 ){
				// Flag this but do update y anyway.
				increase_dt = true;
			}
		}

		// If you get here, either adaptive_dt == false
		// or everything is fine.
		y = yn;

		if( increase_dt ) return DT_TOO_SMALL;
		else              return SUCCESS;

	}else{
		std::cerr << "Internal solver did not converge to tol "
		          << opts.tol << " in " << opts.maxit
			  << " iterations! Final residual is "
			  << stats.res << "!\n";
		return INTERNAL_SOLVE_FAILURE;
	}

	K = KK; // Store the new stages.
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
   \param fun          RHS to the ODE to integrate
   \param jac          Jacobi matrix to the ODE to integrate
   \param t_vals       Will contain the time points corresponding to obtained y
   \param y_vals       Will contain the numerical solution to the ODE.

   \returns a status code (see \ref odeint_status_codes)
*/
template <typename func_type, typename Jac_type> inline
int odeint( double t0, double t1, const solver_coeffs &sc,
            const solver_options &solver_opts,
            const arma::vec &y0, const func_type &fun, const Jac_type &jac,
            std::vector<double> &t_vals, std::vector<arma::vec> &y_vals )
{
	double t = t0;
	assert( verify_solver_coeffs( sc ) && "Invalid solver coefficients!" );

	if( !y_vals.empty() || !t_vals.empty() ){
		std::cerr << "Vectors for storing are not empty! I will _not_ "
		          << "clear them!\n";
	}

	assert( sc.dt > 0 && "Cannot use time step size of 0!" );

	double dt = sc.dt;

	double err = 0.0; // For adaptive time step size.

	bool adaptive_dt = solver_opts.adaptive_step_size;
	if( adaptive_dt ){
		if( sc.b2.size() != sc.b.size() ){
			std::cerr << "Adaptive time step requested but chosen "
			          << "method has no embedded auxillary pair!\n";
			adaptive_dt = false;
		}
	}

	arma::vec y = y0;
	std::vector<double> tt;
	std::vector<arma::vec> yy;

	tt.push_back(t);
	yy.push_back(y);


	// Main integration loop:
	int status = 0;
	int steps = 0;

	std::cerr << "# step  t    dt   err\n";
	auto print_integrator_stats = [&steps, &t, &dt, &err]{
		std::cerr << steps << "  " << t << "  " << dt
		<< " " << err << "\n"; };
	print_integrator_stats();

	my_timer timer( std::cerr );
	timer.tic();


	std::size_t Neq = y.size();
	std::size_t Ns  = sc.b.size();
	std::size_t NN = Ns * Neq;

	arma::vec K( NN );
	for( std::size_t i = 0; i < Ns; ++i ){
		K.subvec( i, i + Neq - 1 ) = y;
	}


	while( t < t1 ){
		bool something_changed = false;

		double old_dt = dt;
		status = take_time_step( t, y, dt, sc, solver_opts, fun, jac,
		                         adaptive_dt, err, K );

		switch( status ){
			default:
			case GENERAL_ERROR:
				std::cerr << "Generic error in odeint!\n";
				return GENERAL_ERROR;

				break;


			case INTERNAL_SOLVE_FAILURE:
				std::cerr << "Internal solver failed to converge! ";
			case DT_TOO_LARGE:
				if( adaptive_dt ){
					dt = get_better_time_step( dt, err,
					                           solver_opts );
				}else{
					dt *= 0.9*0.3;
				}

				if( dt < 1e-10 ){
					std::cerr << "dt is too small!\n";
					return TIME_STEP_TOO_SMALL;
				}

				something_changed = true;
				break;

			case DT_TOO_SMALL:
				if( adaptive_dt ){
					dt = get_better_time_step( dt, err,
					                           solver_opts );
				}else{
					dt *= 1.2;
				}
				something_changed = true;
			case SUCCESS:
				// OK.
				t += old_dt;
				++steps;

				tt.push_back( t );
				yy.push_back( y );

				break;
		}

		if( status != SUCCESS && status != DT_TOO_SMALL ){
			continue;
		}

		// Do some preparation for the next time step here:
		if( sc.FSAL ){
			std::size_t i0 = (Ns-1)*Neq;
			std::size_t i1 = Ns*Neq - 1;
			K.subvec( 0, Neq - 1 ) = K.subvec( i0, i1 );
		}else{
			K.subvec( 0, Neq - 1 ) = y;
		}
		for( std::size_t i = 1; i < Ns; ++i ){

			K.subvec( i, i + Neq - 1 ) = y;
		}

		if( steps % 1000 == 0 || something_changed ){
			print_integrator_stats();
		}


	}

	if( solver_opts.internal_solver == solver_options::NEWTON ){
		timer.toc( "Integrating with Newton iteration" );
	}else if( solver_opts.internal_solver == solver_options::BROYDEN ){
		timer.toc( "Integrating with Broyden iteration" );
	}

	// If everything was fine, store the obtained results
	t_vals = tt;
	y_vals = yy;

	return 0;
}




} // namespace irk


#endif // IRK_HPP
