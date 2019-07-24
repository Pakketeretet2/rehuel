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
   Runge-Kutta (RK) methods.
*/

#ifndef IRK_HPP
#define IRK_HPP

#include <cassert>
#include <limits>
#include <iomanip>

#include "enums.hpp"
#include "my_timer.hpp"
#include "newton.hpp"
#include "options.hpp"
#include "output.hpp"


/**
   \brief Contains functions related to implicit Runge-Kutta methods.
 */
namespace irk {

	
#ifdef DEBUG_OUTPUT
constexpr const bool debug = true;
#else
constexpr const bool debug = false;
#endif // DEBUG_OUTPUT

	
typedef arma::vec vec_type;
typedef arma::mat mat_type;


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
	/// \brief Enumerates the possible internal non-linear solvers
	enum internal_solvers {
		BROYDEN = 0, ///< Broyden's method
		NEWTON = 1   ///< Newton's method
	};

	/// \brief Constructor with default values.
	solver_options() : adaptive_step_size(true),
	                   use_newton_iters_adaptive_step(true),
	                   verbose_newton(false),
	                   extrapolate_stage(false)
	{ }

	~solver_options()
	{ }

	/// If true, attempt to perform adaptive time stepping using
	/// an embedded pair.
	bool adaptive_step_size;

	/// If true, use newton iteration info in determining adaptive step size
	bool use_newton_iters_adaptive_step;

	/// If true, make the Newton iterator print output.
	bool verbose_newton;

	/// If true, use the current stages and extrapolate to the next time level.
	bool extrapolate_stage;
};


static std::map<int,std::string> rk_method_to_string = {
	FOREACH_IRK_METHOD(GENERATE_STRING)
};

static std::map<std::string,int> rk_string_to_method = {
	FOREACH_IRK_METHOD(GENERATE_MAP)
};


/**
   \brief a struct that contains time stamps and stages that can be used for
   constructing the solution all time points in the interval (dense output).
*/
struct rk_output : basic_output
{
	struct counters {
		counters() : attempt(0), reject_newton(0), reject_err(0),
		             newton_success(0), newton_incr_diverge(0),
		             newton_iter_error_too_large(0),
		             newton_maxit_exceed(0),
		             fun_evals(0), jac_evals(0) {}

		std::size_t attempt, reject_newton, reject_err;

		std::size_t newton_success, newton_incr_diverge,
			newton_iter_error_too_large, newton_maxit_exceed;
		std::size_t fun_evals, jac_evals;
	};

	std::vector<vec_type> stages;
	std::vector<vec_type> err_est;
	std::vector<double>   err;

	double elapsed_time, accept_frac;

	counters count;
};


/**
   \brief Merges two rk_output structs.

   \param sol1 First rk_output struct.
   \param sol2 Second rk_output sctruct.

   \returns a solution struct that contains the merged contents of both.
*/
rk_output merge_rk_output( const rk_output &sol1, const rk_output &sol2 );


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
   \brief evaluates the inter/extrapolated weight functions to given theta.

   \param theta    The value to inter/extrapolate to.
   \param sc       The coefficients to use to inter/extrapolate

   \returns A vector containing the values { b1(theta), b2(theta)... }.
*/
vec_type project_b( double theta, const irk::solver_coeffs &sc );


/**
   \brief expands the coefficient lists.

   This is needed to automatically calculate the interpolating coefficients.

   \param c1 The first coefficient list
   \param c2 The second coefficient list

   This is like operator expansion of operator( c1, c2 )
   if c1 = { a1 + a2 } and c2 = { b1 + b2 }
   and we encode for that as c1 = { {1}, {2} }; c2 = { {3}, {4} }
   then the expansion would be operator(c1,c2) =
   { a1b1 + a1b2 + a2b1 + a2b2 } which would be encoded as
   { {1,3}, {1,4}, {2,3}, {2,4} }.
   operator( ( a1b1 + a1b2 + a2b1 + a2b2 ), (x1 + x2) ) follows from induction.

   \returns the expanded coefficient list.
*/
typedef std::vector<std::vector<int> > coeff_list;
coeff_list expand( const coeff_list &c1, const coeff_list &c2 );


/**
   \brief Prints the coeff_list to output stream.

   \param o  The output stream
   \param c  The coefficient list.

   \returns the output stream.
*/
std::ostream &operator<<( std::ostream &o, const coeff_list &c );


/**
   \brief Generates the interpolation polynomial coefficients
          for collocation methods

   \param c    The collocation points of the method.

   \returns The interpolation coefficient matrix.
*/
mat_type collocation_interpolate_coeffs( const vec_type &c );




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
vec_type construct_F( double t, const vec_type &y, const vec_type &K,
                      double dt, const irk::solver_coeffs &sc,
                      functor_type &func, std::size_t &fun_evals )
{
	if (debug) std::cerr << "Constructing F...\n";
	auto Ns  = sc.b.size();
	auto Neq = y.size();
	auto NN = Ns*Neq;

	if (debug) std::cerr << "NN = " << NN << ", K.size() is " << K.size() << "\n";
	vec_type F(NN);
	if (debug) std::cerr << "Done constructing F...\n";
	assert( K.size() == NN  && "Size of K is not right!" );
	assert( y.size() == Neq && "Size of y is not right!" );

	const vec_type &c = sc.c;
	const mat_type &A = sc.A;
	auto eval_fun = [&func,&fun_evals](double t, const vec_type &Y)
	                {
		                ++fun_evals;
		                return func.fun(t,Y); 
	                };

	for( unsigned int i = 0; i < Ns; ++i ){
		double ti = t + dt * c[i];
		if (debug) std::cerr << "Constructing yi and delta...\n";
		vec_type yi = y;
		vec_type delta = arma::zeros(Neq);
		for( unsigned int j = 0; j < Ns; ++j ){
			unsigned int offset = j*Neq;
			// vec_eigen needs a read-only segment function.
			//auto K_part = K.segment(offset, offset+Neq);
			delta += A(i,j) * K.subvec(offset, offset+Neq-1);
		}
		if (debug) std::cerr << "Constructed delta...\n";
		yi += dt*delta;
		auto ki = K.subvec( i*Neq, i*Neq + Neq - 1 );
		// vec_type tmp = func.fun( ti, yi );
		vec_type tmp = eval_fun(ti, yi);
		
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
typename functor_type::jac_type construct_J( double t, const vec_type &y,
                                             const vec_type &K,
                                             double dt,
                                             const irk::solver_coeffs &sc,
                                             functor_type &func,
                                             std::size_t &jac_evals)
{
	auto Ns  = sc.b.size();
	auto Neq = y.size();
	auto NN = Ns*Neq;

	vec_type F( NN );

	assert( K.size() == NN  && "Size of K is not right!" );
	assert( y.size() == Neq && "Size of y is not right!" );

	const vec_type &c = sc.c;
	const mat_type &A = sc.A;

	typename functor_type::jac_type J = -arma::eye( NN, NN );
	auto eval_jac = [&func,&jac_evals](double t, const vec_type &Y)
	                {
		                ++jac_evals;
		                return func.jac(t,Y); 
	                };

	// i is column, j is row.
	for( unsigned int i = 0; i < Ns; ++i ){
		double ti = t + dt * c[i];
		vec_type yi = y;

		for( unsigned int j = 0; j < Ns; ++j ){
			unsigned int offset = j*Neq;
			yi += dt*A(i,j)*K.subvec(offset, offset + Neq-1);
		}
		//auto Ji = func.jac( ti, yi );
		auto Ji = eval_jac(ti, yi);

		// j is row.
		for( unsigned int j = 0; j < Ns; ++j ){
			// Block i*Neq by j*Neq has to be filled with
			// d F(t + ci, y + dt*sum_{k=0}^N-1 (a_{i,k}*k_k)) / d k_j
			// which is
			// F(t + ci, y + sum_{k=0}^N-1 (a_{i,k}*k_k))' * a_{i,j}*dt
			// Armadillo syntax is (from, to)
			auto Jc = J.submat( i*Neq, j*Neq,
			                    i*Neq + Neq - 1,
			                    j*Neq + Neq - 1 );
			//auto Jc = J.block(i*Neq, j*Neq, Neq, Neq);
			double a_part = dt * A(i,j);
			Jc += Ji * a_part;
		}
	}

	return J;
}





/**
   \brief Generic time integration function for IRK methods

   \param func         Functor of the ODE to integrate
   \param t0           Starting time
   \param t1           Final time
   \param y0           Initial values
   \param sc           Solver coefficients
   \param solver_opts  Options for the internal solver.

   \returns a struct that contains status, solution, etc. (see irk::rk_output).
*/
template <typename functor_type> inline
rk_output irk_guts( functor_type &func, double t0, double t1, const vec_type &y0,
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

	my_timer timer;

	double t = t0;
	rk_output sol;
	sol.status = SUCCESS;

	assert( solver_opts.newton_opts && "Newton solver options not set!" );
	assert( dt > 0 && "Cannot use time step size <= 0!" );

	const newton::options &newton_opts = *solver_opts.newton_opts;

	std::size_t Neq = y0.size();
	std::size_t Ns  = sc.b.size();
	std::size_t N   = Neq * Ns;

	vec_type y  = y0;
	vec_type yo(N);
	vec_type K_np = arma::zeros(N), K_n = arma::zeros(N);

	newton::status newton_stats;
	long long int step = 0;

	auto eval_fun = [&func,&sol](double t, const vec_type &Y)
	{
		++sol.count.fun_evals;
		return func.fun(t,Y); 
	};


	double dts[3];
	double errs[3];
	double err = 0.0;
	double new_dt = dt;
	dts[0] = dts[1] = dts[2] = dt;
	errs[0] = errs[1] = errs[2] = 0.9;

	if( solver_opts.out_interval > 0 ){
		std::cerr  << "    Rehuel: step  t  dt   err   iters\n";
	}

	vec_type err_est = arma::zeros( y.size() );

	sol.t_vals.push_back(t);
	sol.y_vals.push_back(y);
	sol.stages.push_back(K_n);
	sol.err_est.push_back( err_est );
	sol.err.push_back( 0.0 );

	bool alternative_error_formula = true;
	std::size_t min_order = std::min( sc.order, sc.order2 );

	while( t < t1 ){
		// ****************  Calculate stages:   ************
		// Make sure you stop exactly at t = t1.
		if( t + dt > t1 ){
			dt = t1 - t;
		}
		sol.count.attempt++;

		int integrator_status = 0;

		// Use newton iteration to find the Ks for the next level:
		auto stages_func = [&t, &y, &dt, &sc, &func, &sol]( const vec_type &K ){
			return construct_F( t, y, K, dt, sc, func, sol.count.fun_evals );
		};

		auto stages_jac = [&t, &y, &dt, &sc, &func, &sol]( const vec_type &K ){
			return construct_J( t, y, K, dt, sc, func, sol.count.jac_evals );
		};

		newton::newton_lambda_wrapper<decltype(stages_func),
		                              decltype(stages_jac),
		                              typename functor_type::jac_type>
			nw( stages_func, stages_jac );

		K_np = newton::newton_iterate( nw, K_n, newton_opts,
		                               newton_stats,
		                               !solver_opts.verbose_newton );
		int newton_status = newton_stats.conv_status;

		// *********** Verify Newton iteration convergence ************
		if( newton_status != newton::SUCCESS ){
			if( !solver_opts.adaptive_step_size ){
				// In this case, you can do nothing but error.
				sol.status = GENERAL_ERROR;
				std::cerr << "    Rehuel: Newton iteration "
				          << "failed for constant time step "
				          << "size! Aborting!\n";
				return sol;
			}

			//std::cerr << "    Rehuel: Newton iteration failed! "
			//          << "Retrying with dt = " << dt << "\n";
			dt *= 0.5;
			sol.count.reject_newton++;
			if( newton_status == newton::INCREMENT_DIVERGE ){
				sol.count.newton_incr_diverge++;
			}else if( newton_status == newton::ITERATION_ERROR_TOO_LARGE ){
				sol.count.newton_iter_error_too_large++;
			}else if( newton_status == newton::MAXIT_EXCEEDED ){
				sol.count.newton_maxit_exceed++;
			}
			continue;
		}else{
			sol.count.newton_success++;
		}


		// ****************  Construct solution at t + dt   ************
		vec_type delta_y, delta_alt, dy_alt;
		std::size_t Neq = y.size();
		double gamma = sc.gamma;

		// Vectorized version of the loop below:
		mat_type Ks = arma::reshape(K_np, Neq, Ns);
		delta_y = Ks*sc.b;
		vec_type y_n    = y + dt*delta_y;
		
		if (solver_opts.adaptive_step_size) {
			delta_alt = Ks*sc.b2;
			dy_alt = gamma * eval_fun( t, y ) + delta_alt;
			vec_type yp     = y + dt*dy_alt;
			vec_type delta_delta = dy_alt - delta_y;

			// **************      Estimate error:    **************
			mat_type I, J0;
			I = arma::eye(Neq, Neq);
			J0 = func.jac( t, y );
			// Formula 8.19:
			mat_type solve_tmp = I - gamma*dt*J0;
			vec_type err_8_19 = dt*arma::solve(solve_tmp, delta_delta);
			err_est = err_8_19;
			
			// Alternative formula 8.20:
			if( alternative_error_formula ){
				// Use the alternative formulation:
				// vec_type dy_alt_alt = gamma*func.fun(t, y+err_est);
				vec_type dy_alt_alt = gamma*eval_fun(t, y + err_est);
				dy_alt_alt += delta_alt;
				vec_type err_alt = dy_alt_alt - delta_y;
				err_est = dt*arma::solve(solve_tmp, err_alt);
			}
			
			double err_tot = 0.0;
			double n = 0.0;
			double atol = solver_opts.abs_tol;
			double rtol = solver_opts.rel_tol;
			for( std::size_t i = 0; i < err_est.size(); ++i ){
				double erri = err_est[i];
				double y0i  = std::fabs( y[i] );
				double y1i  = std::fabs( y_n[i] );
				double sci  = atol + rtol * std::max( y0i, y1i );
				
				double add = erri / sci;
				err_tot += add * add;
				n += 1.0;
			}
			
			assert( err_tot >= 0.0 && "Error cannot be negative!" );
		        err = std::sqrt( err_tot / n );
			
			
			if( err < machine_precision ){
				err = machine_precision;
			}
			
			errs[2] = errs[1];
			errs[1] = errs[0];
			errs[0] = err;
			
			if(err > 1.0){
				// This is bad.
				alternative_error_formula = true;
				integrator_status = 1;
				sol.count.reject_err++;
			}

			
			// **************      Find new dt:    **********************
			
			double fac = 0.9 * ( newton_opts.maxit + 1.0 );
			fac /= ( newton_opts.maxit + newton_stats.iters );
			
			double expt = 1.0 / ( 1.0 + min_order );
			double err_inv = 1.0 / err;
			double scale_27 = std::pow( err_inv, expt );
			double dt_rat = dts[0] / dts[1];
			double err_frac = errs[1] / errs[0];
			if( errs[1] == 0 || errs[0] == 0 ){
				err_frac = 1.0;
			}
			double err_rat = std::pow( err_frac, expt );
			double scale_28 = scale_27 * dt_rat * err_rat;
			
			double min_scales = std::min( scale_27, scale_28 );
			// When growing dt, don't grow more than a factor 4:
			new_dt = fac * dt * std::min( 4.0, min_scales );
			if( solver_opts.max_dt > 0 ){
				new_dt = std::min( solver_opts.max_dt, new_dt );
			}
			
		}
		
		// **************    Update y and time   ********************

		if( solver_opts.out_interval > 0 &&
		    (step % solver_opts.out_interval == 0) ){
			std::cerr  << "    Rehuel: " << step << " " << t
			           << " " <<  dt << " " << err << " "
			           << newton_stats.iters << "\n";
		}
		

		if( !solver_opts.adaptive_step_size || integrator_status == 0 ){
			yo = y;
			y  = y_n;
			t += dt;
			++step;

			sol.t_vals.push_back(t);
			sol.y_vals.push_back(y_n);
			sol.stages.push_back(K_np);
			sol.err_est.push_back( err_est );
			sol.err.push_back( err );

			alternative_error_formula = false;
		}

		// **************      Actually set the new dt:    **********************

		if( solver_opts.adaptive_step_size ) {
			dt = new_dt;
		}
		dts[2] = dts[1];
		dts[1] = dts[0];
		dts[0] = dt;

		if( solver_opts.extrapolate_stage && (integrator_status == 0) ){
			// ***************   Estimate new stages:  *************
			// NOTE: This depends on the definition of the variables
			// that are the stages. If the code is later changed to
			// solve the stages according to Hairer & Wanner, this
			// too needs to change.
			double dt_frac = dts[0]/dts[1];
			for( std::size_t i = 0; i < Ns; ++i ){
				double ci = sc.c(i);

				double theta = 1.0 + dt_frac * ci;
				vec_type bs = project_b( theta, sc );

				vec_type Yi = yo;
				for( std::size_t j = 0; j < Ns; ++j ){
					std::size_t j0 = j*Neq;
					std::size_t j1 = (j+1)*Neq;
					auto Kj = K_np.subvec( j0, j1-1 );
					Yi += dt * bs[j] * Kj;
				}

				// Now you have an approximation for the time
				// point at t + ci*dt. Use the rhs evaluated
				// at this point as an approximation for the
				// next stage.
				std::size_t i0 = i*Neq;
				std::size_t i1 = (i+1)*Neq;
				double ti = t + dt*ci;
				K_n.subvec( i0, i1-1 ) = eval_fun(ti,Yi);
			}
		}
	}

	double elapsed = timer.toc();
	sol.elapsed_time = elapsed;
	sol.accept_frac = static_cast<double>(step) / sol.count.attempt;

	return sol;
}


/**
   \brief Time-integrate a given ODE from t0 to t1, starting at y0

   \param func         Functor of the ODE to integrate
   \param t0           Starting time
   \param t1           Final time
   \param y0           Initial values
   \param dt           Initial time step size.
   \param solver_opts  Options for the internal solver.

   \returns a struct with the solution and info about the solution quality.
*/
template <typename functor_type> inline
rk_output odeint( functor_type &func, double t0, double t1, const vec_type &y0,
                  solver_options solver_opts,
                  int method = irk::RADAU_IIA_53, double dt = 1e-6 )
{
	solver_coeffs sc = get_coefficients( method );
	if (solver_opts.adaptive_step_size && sc.b2.size() == 0) {
		std::cerr << "    Rehuel: WARNING: Cannot have adaptive time "
		          << "step with non-embedding method! Disabling "
		          << "adaptive time step size!\n";
		solver_opts.adaptive_step_size = false;
	}
	assert( verify_solver_coeffs( sc ) && "Invalid solver coefficients!" );
	return irk_guts( func, t0, t1, y0, solver_opts, dt, sc );
}



/**
   \brief Time-integrate a given ODE from t0 to t1, starting at y0.

   This function is supposed to provide a sane "default" implicit solver,
   a la ode15s in Matlab.

   \param func         Functor of the ODE to integrate
   \param t0           Starting time
   \param t1           Final time
   \param y0           Initial values

   \returns a struct with the solution and info about the solution quality.
*/
template <typename functor_type> inline
rk_output odeint( functor_type &func, double t0, double t1, const vec_type &y0)
{
	solver_options s_opts = default_solver_options();
	newton::options n_opts;
	n_opts.tol = 0.1*std::min(s_opts.abs_tol, s_opts.rel_tol);
	
	return odeint(func, t0, t1, y0, s_opts);
}






} // namespace irk


#endif // IRK_HPP
