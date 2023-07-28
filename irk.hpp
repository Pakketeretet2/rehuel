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

#include <functional>

#include "enums.hpp"
#include "my_timer.hpp"
#include "newton.hpp"
#include "options.hpp"
#include "output.hpp"


/**
   \namespace irk
   \brief Contains functions related to implicit Runge-Kutta methods.
 */
namespace irk {
/**
  \addtogroup Integrators

*/

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
	vec_type b;       ///< weights for the new y-value
	vec_type c;       ///< these set the intermediate time points
	mat_type A;       ///< alpha coefficients in Butcher tableau

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





inline void print_timing_breakdown(const std::vector<double> &timings,
                                   std::ostream &out = std::cerr)
{
	double total_t = std::accumulate(timings.begin(), timings.end(), 0.0);

	auto print_line = [&out](double t, double total, const char *txt)
	{
		const char *w = "      ";
		double prct = 100.0*t/total;
	        out << w << txt << std::setw(7) << t
	            << " | " << std::setprecision(3)
	            << prct << "\n";
	};

	// Determine time unit:
	std::string time_unit = "ms";
	double time_scale = 1.0;
	if (total_t > 1000) {
		time_scale = 0.0001;
		time_unit = " s";
	}
	total_t *= time_scale;

	out << "     Rehuel: IRK done solving, timings (" << time_unit << " | %):\n";
	print_line(total_t,    total_t, "Total time:                  ");
	print_line(time_scale*timings[0], total_t, "Vector setup:                ");
	print_line(time_scale*timings[1], total_t, "Stages update:               ");
	print_line(time_scale*timings[2], total_t, "Solution update:             ");
	print_line(time_scale*timings[3], total_t, "Store solution:              ");
	print_line(time_scale*timings[4], total_t, "Error estimate:              ");
	print_line(time_scale*timings[5], total_t, "Calculate optimal step:      ");
}


/**
   \brief Construct the residual vector of the non-linear systme to solve.
*/
template <typename functor_type> inline
arma::vec construct_R(functor_type &func,
                      const vec_type &y, double t, double dt,
                      const solver_coeffs &sc, const vec_type &Y,
                      const mat_type &I_neq)
{
	vec_type F(Y.size());
	std::size_t Ns = sc.b.size();
	std::size_t Neq = y.size();
	vec_type e = arma::ones(Ns);
	arma::vec R = Y;
	for (std::size_t i = 0; i < Ns; ++i) {
		std::size_t i0 = Neq*i;
		std::size_t i1 = i0 + Neq - 1;

		auto Yi = Y.subvec(i0,i1);
		F.subvec(i0, i1) = func.fun(t + sc.c(i)*dt, y + Yi);
	}
	R -= dt*arma::kron(sc.A, I_neq)*F;
	return R;
}



/**
   \brief Performs simplified Newton iteration for IRKs to find stages

   Stages are defined by (Y_1, Y_2, ...)^T = dt*(kron(A,I)*(k_1, k_2, ...)^T
   with k_i the original stages.

   \param Y Contains the stages
*/
template <typename functor_type,
          bool adaptive_step=true,
          bool PLU_decomposition=false> inline
int newton_solve_stages(functor_type &func, const vec_type &y, double t,
                        double dt, const solver_coeffs &sc,
                        int maxit, int refresh_jac,
                        double xtol, double Rtol, vec_type &Y, mat_type &J,
                        newton::status &stats,
                        std::size_t &fun_evals, std::size_t &jac_evals)
{
	const std::size_t Neq = y.size();
	const std::size_t Ns  = sc.b.size();
	const std::size_t NN  = Ns*Neq;

	const mat_type I_neq   = arma::eye(Neq, Neq);

	// Construct the initial system:
	Y = arma::zeros(NN);

	// Jacobi matrix:
	// Idea: Refresh Jacobi matrix after every so many iterations.
	mat_type J_Y;
	mat_type L, U, P;

	auto refresh_jacobi_matrix =
		[&func, &J, &J_Y, NN, &L, &U, &P, t, dt, y, sc, &jac_evals]()
		{
			J = func.jac(t,y);
			J_Y = arma::eye(NN,NN);
			J_Y -= dt*kron(sc.A,J);

			// Since we re-use the same Jacobi matrix,
			// pre-construct the LU decomposition:
			if (PLU_decomposition) {
				assert(arma::lu(L,U,P, J_Y) &&
				       "LU decomposition of Jacobi matrix failed!");
			}
			++jac_evals;
		};

	refresh_jacobi_matrix();

	// Start iterating:
	double xtol2 = xtol*xtol;
	double Rtol2 = Rtol*Rtol;
	vec_type R = construct_R(func, y, t, dt, sc, Y, I_neq);
	fun_evals += Ns;
	double step = 1.0;
	double Rnorm2 = arma::dot(R, R);;
	if (adaptive_step) {
		step /= sqrt(1.0 + Rnorm2);
	}
	double xnorm2 = 0;

	// During iteration, we sometimes temporarily increase the
	// maximum number of allowed iterations when the error is very small.
	// Therefore, keep a copy of the original value here:

	int status = newton::MAXIT_EXCEEDED;
	stats.iters = 1;
	for ( ; stats.iters < maxit; ++stats.iters) {
		vec_type dY;
		if (PLU_decomposition) {
			vec_type tmp = arma::solve(arma::trimatl(-L), P*R);
			dY  = arma::solve(arma::trimatu(U), tmp);
		} else {
			dY  = -arma::solve(J_Y, R);
		}
		xnorm2 = arma::dot(dY, dY);

		Y += step*dY;
		R = construct_R(func, y, t, dt, sc, Y, I_neq);

		fun_evals += Ns;
		Rnorm2 = arma::dot(R,R);
		if (Rnorm2 < Rtol2) {
			status = newton::SUCCESS;
			break;
		}
		if (xnorm2 < xtol2) {
			status = newton::SUCCESS;
			break;
		}
		if (adaptive_step) {
			step = 1.0 / sqrt(1.0 + Rnorm2);
		}

		if (stats.iters % refresh_jac == 0) {
			refresh_jacobi_matrix();
		}
	}
	stats.res = Rnorm2;
	stats.conv_status = status;

	return status;
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
rk_output irk_guts(functor_type &func, double t0, double t1, const vec_type &y0,
                   const solver_options &solver_opts, double dt,
                   const solver_coeffs &sc, const output_options &output_opts)
{
	if (t0 + dt > t1) {
		output_opts.log_out << "    Rehuel: Initial dt (" << dt;
		dt = t1 - t0;
		output_opts.log_out << ") too large for interval! Reducing to "
		          << dt << "\n";

	}

	output_opts.log_out << "    Rehuel: Integrating over interval [ "
	                    << t0 << ", " << t1 << " ]...\n"
	                    << "            Method = " << sc.name << "\n";

	const bool time_internals = solver_opts.time_internals;
	my_timer timer;
	timeval irk_start = timer.get_tic();

	// This table keeps track of internal timings (in ms):
	enum timing_entries {
		VECTOR_SETUP = 0,
		UPDATE_STAGES,
		UPDATE_Y,
		STORE_SOL,
		ESTIMATE_ERROR,
		ESTIMATE_DT,
		CUSTOM_OUTPUT_CALLBACK,
		// Dummy for number of elements:
		N_TIMING_ENTRIES
	};
	std::vector<double> timings(N_TIMING_ENTRIES, 0);

	double t = t0;
	rk_output sol;
	sol.status = SUCCESS;

	assert( solver_opts.newton_opts && "Newton solver options not set!" );
	assert( dt > 0 && "Cannot use time step size <= 0!" );

	const newton::options &newton_opts = *solver_opts.newton_opts;
	const int newton_maxit0 = newton_opts.maxit;
	int newton_maxit = newton_maxit0;
	long long last_maxit_relax_step = 0;

	std::size_t Neq = y0.size();
	std::size_t Ns  = sc.b.size();
	std::size_t N   = Neq * Ns;


	if (time_internals) timer.tic();
	vec_type y  = y0;
	vec_type yo(N);
	vec_type K_np = arma::zeros(N), K_n = arma::zeros(N);
	if (time_internals) timings[VECTOR_SETUP] += timer.toc();
	long long int step = 0;

	double dts[3];
	double errs[3];
	double err = 0.0;
	double new_dt = dt;
	dts[0] = dts[1] = dts[2] = dt;
	errs[0] = errs[1] = errs[2] = 0.9;

	if( solver_opts.out_interval > 0 ){
		output_opts.log_out  << "    Rehuel: step  t  dt   err   iters\n";
	}

	vec_type err_est = arma::zeros( y.size() );
	if (time_internals) timer.tic();
	sol.t_vals.push_back(t);
	sol.y_vals.push_back(y);
	sol.stages.push_back(K_n);
	sol.err_est.push_back( err_est );
	sol.err.push_back( 0.0 );
	if (time_internals) timings[STORE_SOL] += timer.toc();
	bool alternative_error_formula = true;
	std::size_t min_order = std::min( sc.order, sc.order2 );


	// Variables/parameters for Newton iteration:
	vec_type Y; // Contains the stages.
	mat_type J; // Contains Jacobi matrix
	double xtol = newton_opts.dx_delta;
	double Rtol = newton_opts.tol;
	newton::status newton_stats;

	// Construct the alternative weights:
	mat_type Ai = arma::inv(sc.A);
	vec_type d_weights  = (Ai.t())*sc.b;
	vec_type d2_weights = (Ai.t())*sc.b2;


	while (t < t1) {
		// ****************  Calculate stages:   ************
		// Make sure you stop exactly at t = t1.
		if( t + dt > t1 ){
			dt = t1 - t;
		}
		sol.count.attempt++;

		if (solver_opts.max_steps >= 0 && step > solver_opts.max_steps) {
			output_opts.log_out << "    Rehuel: Maximum number of attempts exceeded.\n";
			sol.status = ERROR_MAX_STEPS_EXCEEDED;
			return sol;
		}

		int integrator_status = 0;

		// Use newton iteration to find the Ks for the next level:
		int newton_status = newton_solve_stages<functor_type,
		                                        false, true>(
			func, y, t, dt, sc,
			newton_maxit,
			newton_opts.refresh_jac,
			xtol, Rtol, Y, J,
			newton_stats,
			sol.count.fun_evals,
			sol.count.jac_evals);



		if (time_internals) timings[UPDATE_STAGES] += timer.toc();

		// *********** Verify Newton iteration convergence ************
		if (newton_status != newton::SUCCESS){
			if (!solver_opts.adaptive_step_size) {
				// In this case, you can do nothing but error.
				sol.status = GENERAL_ERROR;
				output_opts.log_out << "   Rehuel: Newton iteration "
				          << "failed for constant time step "
				          << "size! Aborting!\n";
				return sol;
			}

			dt *= 0.7;
			if (step - last_maxit_relax_step > 15) {
				newton_maxit += newton_maxit0;
				last_maxit_relax_step = step;
			}
			/*
			output_opts.log_out << "   Rehuel: step " << step
			          << ", t = " << t
			          << ": Newton iteration failed! Status: "
			          << newton::status_message(newton_status)
			          << ".\n        Retrying with dt = " << dt
			          << " and maxit = " << newton_maxit << "\n";
			*/
			sol.count.reject_newton++;
			if (newton_status == newton::INCREMENT_DIVERGE){
				sol.count.newton_incr_diverge++;
			}else if (newton_status == newton::ITERATION_ERROR_TOO_LARGE){
				sol.count.newton_iter_error_too_large++;
			}else if (newton_status == newton::MAXIT_EXCEEDED){
				sol.count.newton_maxit_exceed++;
			}
			continue;
		} else {
			sol.count.newton_success++;

			// It is possible we incremented dt and maxit
			// before. We should slowly relax them back...
			/*
			if (step - last_maxit_relax_step > 10) {
				output_opts.log_out << "   Rehuel: step " << step
				          << ": Resetting maxit from "
				          << newton_maxit << " to "
				          << newton_maxit0 << ".\n";
				newton_maxit = newton_maxit0;
			}
			*/
		}


		// ****************  Construct solution at t + dt   ************
		if (time_internals) timer.tic();

		// At this point, Y contains the stages defined by
		// Y_i = dt*(a_i1*k1 + a_i2*k2)...
		// The update to y is given by d := b*inv(A)*Y;


		vec_type delta_y, delta_alt;
		double gam = sc.gamma*dt;

		// Vectorized version of the loop below:
		mat_type YYs = arma::reshape(Y, Neq, Ns);
		delta_y = YYs*d_weights;

		if (solver_opts.adaptive_step_size) {
			delta_alt = YYs*d2_weights;
		}

		vec_type dy_alt = gam * func.fun(t,y) + delta_alt;
		++sol.count.fun_evals;

		vec_type y_n    = y + delta_y;
		vec_type yp     = y + dy_alt;
		vec_type delta_delta = dy_alt - delta_y;
		if (time_internals) {
			timings[UPDATE_Y] += timer.toc();
			timer.tic();
		}

		// **************      Estimate error:    **********************
		// Formula 8.19:
		// J0 = func.jac( t, y );
		// J was already calculated for us in newton_solve_stages:
		mat_type solve_tmp = arma::eye(Neq,Neq) - gam*J;
		vec_type err_8_19 = dt*arma::solve(solve_tmp, delta_delta);
		err_est = err_8_19;

		// Alternative formula 8.20:
		if( alternative_error_formula ){
			// Use the alternative formulation:
			// vec_type dy_alt_alt = gamma*func.fun(t, y+err_est);
			vec_type dy_alt_alt = gam*func.fun(t, y + err_est);
			++sol.count.fun_evals;

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
		if (time_internals) timings[ESTIMATE_ERROR] += timer.toc();

		if( solver_opts.adaptive_step_size && (err > 1.0) ){
			// This is bad.
			alternative_error_formula = true;
			integrator_status = 1;
			sol.count.reject_err++;
		}


		// **************      Find new dt:    **********************
		if (time_internals) timer.tic();
		double fac = 0.9 * (newton_maxit + 1.0);
		fac /= (newton_maxit + newton_stats.iters);

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

		/*
		output_opts.log_out << "    Rehuel: Time step controller:\n"
		          << "            err      = " << err << "\n"
		          << "            err_inv  = " << err_inv << "\n"
		          << "            dt_rat   = " << dt_rat << "\n"
		          << "            err_frac = " << err_frac << "\n"
		          << "            err_rat  = " << err_rat << "\n"
		          << "            scale_27 = " << scale_27 << "\n"
		          << "            scale_28 = " << scale_28 << "\n\n";
		*/

		double min_scales = std::min( scale_27, scale_28 );
		// When growing dt, don't grow more than a factor 4:
		new_dt = fac * dt * std::min( 8.0, min_scales );
		if( solver_opts.max_dt > 0 ){
			new_dt = std::min( solver_opts.max_dt, new_dt );
		}
		if (time_internals) timings[ESTIMATE_DT] += timer.toc();

		// **************    Update y and time   ********************
		if( solver_opts.out_interval > 0 &&
		    (step % solver_opts.out_interval == 0) ){
			output_opts.log_out  << "    Rehuel: " << step << " " << t
			                     << " " <<  dt << " " << err << " "
			                     << newton_stats.iters << "\n";
		}


		if (!solver_opts.adaptive_step_size || integrator_status == 0) {
			if (time_internals) timer.tic();
			yo = y;
			y  = y_n;
			t += dt;
			++step;

			if (time_internals) {
				timings[UPDATE_Y] += timer.toc();
				timer.tic();
			}

			if (step % output_opts.output_interval == 0) {
				if (output_opts.store_in_vectors()) {
					sol.t_vals.push_back(t);
					sol.y_vals.push_back(y_n);
					sol.stages.push_back(K_np);
					sol.err_est.push_back( err_est );
					sol.err.push_back( err );

					if (time_internals) {
						timings[STORE_SOL] += timer.toc();
					}
				}
				if (output_opts.write_to_file()) {
					*output_opts.output_stream << t;
					for (std::size_t y_idx = 0; y_idx < y_n.size(); ++y_idx) {
						*output_opts.output_stream << " " << y_n[y_idx];
					}
					*output_opts.output_stream << "\n";
				}
			}
			alternative_error_formula = false;
		}

		// **************      Actually set the new dt:    **********************

		if( solver_opts.adaptive_step_size ) {
			dt = new_dt;
		}
		dts[2] = dts[1];
		dts[1] = dts[0];
		dts[0] = dt;

		if (solver_opts.extrapolate_stage && (integrator_status == 0)) {
			// TODO
		}
	}

	double elapsed = timer.get_elapsed(irk_start);
	sol.elapsed_time = elapsed;
	sol.accept_frac = static_cast<double>(step) / sol.count.attempt;

	if (time_internals) print_timing_breakdown(timings);

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
rk_output odeint(functor_type &func, double t0, double t1, const vec_type &y0,
                 solver_options solver_opts, const output_options &output_opts,
                 int method = irk::RADAU_IIA_53, double dt = 1e-6)
{
	solver_coeffs sc = get_coefficients( method );
	if (solver_opts.adaptive_step_size && sc.b2.size() == 0) {
		output_opts.log_out << "    Rehuel: WARNING: Cannot have adaptive time "
		          << "step with non-embedding method! Disabling "
		          << "adaptive time step size!\n";
		solver_opts.adaptive_step_size = false;
	}
	assert( verify_solver_coeffs( sc ) && "Invalid solver coefficients!" );
	return irk_guts(func, t0, t1, y0, solver_opts, dt, sc, output_opts);
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
rk_output odeint(functor_type &func, double t0, double t1, const vec_type &y0)
{
	solver_options s_opts = default_solver_options();
	newton::options n_opts;
	n_opts.refresh_jac = 25;
	n_opts.tol = 0.1*std::min(s_opts.abs_tol, s_opts.rel_tol);

	s_opts.newton_opts = &n_opts;
	return odeint(func, t0, t1, y0, s_opts);
}

/**
   @}
*/


} // namespace irk


#endif // IRK_HPP
