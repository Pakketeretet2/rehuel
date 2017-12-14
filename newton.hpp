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
   \file newton.hpp
*/

#ifndef NEWTON_HPP
#define NEWTON_HPP

// Functions for performing Newton iteration.

#define AMRA_USE_CXX11

#include <armadillo>
#include <fstream>

#include "my_timer.hpp"


/// \brief A namespace with solvers for non-linear systems of equations.
namespace newton {

/// \brief Return codes for newton_solve.
enum newton_solve_ret_codes {
	SUCCESS = 0,       ///< Converged to tolerance
	NOT_CONVERGED = 1  ///< Did not converge to tolerance within maxit
};


/**
   \brief contains options for the solver.
*/
struct options {
	options() : tol(1e-4), maxit(500), time_internals(false),
	            max_step(-1), refresh_jac(true), precondition(true){}

	double tol;           ///< Desired tolerance.
	int maxit;            ///< Maximum number of iterations
	bool time_internals;  ///< Print timings for solver
	double max_step;      ///< Limit update to this length in N-D.

	/// When Using Newton's method, if this is false, the Jacobi matrix
	/// is constructed only once at the beginning and never updated.
	bool refresh_jac;

	/// If true, precondition the Jacobi matrix.
	bool precondition;
};

/**
   \brief contains status for the solver.
*/
struct status {
	status() : conv_status(SUCCESS), res(0.0), iters(0),
	           store_final_F(false), store_final_J(false){}


	int conv_status;  ///< Status code, see \ref newton_solve_ret_codes
	double res;       ///< Final residual (F(x_root)^2)
	int iters;        ///< Number of iterations actually used

	bool store_final_F;  ///< If true, the final value of F(x) is stored
	bool store_final_J;  ///< If true, the final value of J(x) is stored
	arma::vec final_F;   ///< if(store_final_F), contains F(x) at found root
	arma::mat final_J;   ///< if(store_final_J), contains J(x) at found root
};


/**
    \brief Approximates the Jacobi matrix with finite differences.

    \param y    Point about which to approximate Jacobi matrix
    \param fun  Function to determine Jacobi matrix for
    \param h    Finite difference step size.
*/
template <typename func_type> inline
arma::mat approx_jacobi_matrix( const arma::vec &y,
                                const func_type &fun, double h )
{
	std::size_t N = y.size();
	arma::mat J_approx(N,N);
	J_approx.zeros(N,N);
	arma::vec f0 = fun(y);
	arma::vec new_yp = y;
	arma::vec new_ym = y;

	for( std::size_t j = 0; j < N; ++j ){
		double old_y_j = y(j);

		new_yp(j) += h;
		new_ym(j) -= h;
		arma::vec fp = fun( new_yp );
		arma::vec fm = fun( new_ym );

		arma::vec delta = fp - fm;
		delta /= (2.0*h);

		for( std::size_t i = 0; i < N; ++i ){
			J_approx(i,j) = delta(i);
		}


		new_yp(j) = old_y_j;
		new_ym(j) = old_y_j;

	}
	return J_approx;
}

/**
    \brief Verifies that the function jac produces accurate Jacobi matrix at y.

    \param y   Point to test Jacobi matrix at
    \param fun Function handle for non-linear system.
    \param J   Function handle for Jacobi matrix for fun.

    \returns true if jac is accurate, false otherwise.
*/
template <typename func_type, typename Jac_type> inline
bool verify_jacobi_matrix( const arma::vec &y, const func_type &fun,
                           const Jac_type &J )
{
	arma::mat J_approx = approx_jacobi_matrix( y, fun, 1e-4 );
	arma::mat J_fun = J( y );
	std::size_t N = y.size();
	double max_diff2 = 0;
	for( std::size_t i = 0; i < N; ++i ){
		for( std::size_t j = 0; j < N; ++j ){
			double delta = J_approx(i,j) - J_fun(i,j);
			double delta2 = delta*delta;
			if( delta2 > max_diff2 ) delta2 = max_diff2;
		}
	}

	double max_diff = sqrt( max_diff2 );
	if( max_diff > 1e-12 ){
		return false;
	}else{
		return true;
	}
}


/**
   \brief Performs Broyden's method to solve non-linear system F(x) = 0.

   \param F     The non-linear system whose root to find.
   \param x     Initial guess for root.
   \param opts  Options for solver (see \ref options)
   \param stats Will contain solver statistics (see \ref status)

   \returns the root of F(x).
*/
template <typename func_rhs>
arma::vec broyden_iterate( const func_rhs &F, arma::vec x,
                           const options &opts, status &stats )
{
	stats.conv_status = SUCCESS;
	double tol2 = opts.tol*opts.tol;
	arma::vec fn = F(x);
	double res2 = arma::dot( fn, fn );
	stats.iters = 1;

	std::size_t N = x.size();
	arma::vec x0 = x;
	arma::vec f0 = fn;

	arma::mat Jaci( N, N );
	Jaci.eye(N,N);


	double max_step2;
	if( opts.max_step > 0 ) max_step2 = opts.max_step*opts.max_step;
	else max_step2 = -1;

	while( res2 > tol2 && stats.iters < opts.maxit ){
		double lambda = std::max( 1e-10, 1.0 / (1.0 + res2 ) );
		arma::vec direction = -lambda*Jaci*f0;

		if( max_step2 > 0 ){
			double norm2 = arma::dot( direction, direction );
			if( lambda*lambda*norm2 > max_step2 ){
				direction *= std::sqrt(max_step2/norm2)/lambda;
			}
		}


		x = x0 + direction;
		fn = F(x);
		arma::vec dx = x - x0;
		arma::vec df = fn - f0;

		arma::vec normm = (dx.t() * Jaci) * df;
		double norm = normm(0);
		arma::vec left_part = ( dx - Jaci*df );
		arma::rowvec right_part = dx.t() * Jaci;
		Jaci += left_part * right_part / norm;
		res2 = arma::dot( fn, fn );

		++stats.iters;

		f0 = fn;
		x0 = x;
	}
	if( stats.iters == opts.maxit && res2 > tol2 ){
		stats.conv_status = NOT_CONVERGED;
	}else{
		stats.conv_status = SUCCESS;
	}
	stats.res = std::sqrt( res2 );
	return x;
}


/**
   \brief Performs Gauss-Seide to solve system F(x) = 0.

   \param F     The non-linear system whose root to find.
   \param x     Initial guess for root.
   \param opts  Options for solver (see \p options)
   \param stats Will contain solver statistics (see \p status)

   \returns the root of F(x).
*/
template <typename func_rhs> inline
arma::vec gauss_seidel( const func_rhs &F, arma::vec x,
                        const options &opts, status &stats )
{

	arma::vec yn = F(x);
	double res2 = arma::norm( yn, 2 );
	double tol2 = opts.tol * opts.tol;
	std::cerr << "Starting Gauss-Seidel at " << stats.iters
	          << " iters, res = " << std::sqrt(res2) << "\n";
	while( res2 > tol2 && stats.iters < opts.maxit ){
		x = yn;
		yn = F(x);
		++stats.iters;
		res2 = arma::norm( yn, 2 );
	}

	return x;
}

/**
   \brief Performs Newton's method to solve non-linear system F(x) = 0.

   \param F     The non-linear system whose root to find.
   \param x     Initial guess for root.
   \param opts  Options for solver (see \p options)
   \param stats Will contain solver statistics (see \p status)
   \param J     The Jacobi-matrix of F(x).

   \returns the root of F(x).
*/
template <typename func_rhs, typename func_Jac > inline
arma::vec newton_iterate( const func_rhs &F, arma::vec x,
                          const options &opts, status &stats,
                          const func_Jac &J )
{
	stats.conv_status = SUCCESS;
	double tol2 = opts.tol*opts.tol;
	arma::vec r = F(x);
	double res2 = arma::dot( r, r );

	stats.iters = 1;
	bool low_rcond_warn = false;

	double max_step2;
	if( opts.max_step > 0 ) max_step2 = opts.max_step*opts.max_step;
	else max_step2 = -1;

	arma::mat Jac = J(x);
	arma::mat P, Pi;
	std::size_t N = x.size();
	P.eye( N, N );
	Pi.eye( N, N );
	if( opts.precondition ){
		// We solve direction = -Jac^{-1} r
		// So A*direction = -A*direction*Jac^{-1} r
		for( std::size_t i = 0; i < N; ++i ){
			P(i,i) = 1.0 / Jac(i,i);
			Pi(i,i) = Jac(i,i);
		}
	}


	bool no_jac = false;
	while( res2 > tol2 && stats.iters < opts.maxit ){

		double lambda = std::max( 1e-10, 1.0 / (1.0 + res2 ) );
		arma::vec direction;
		// P*direction = -arma::solve(P*jac,r)
		// direction = -Pi*(arma::solve(P*jac,r));
		direction = -Pi*arma::solve(P*Jac,r);

		if( max_step2 > 0 ){
			double norm2 = arma::dot( direction, direction );
			if( lambda*lambda*norm2 > max_step2 ){
				direction *= std::sqrt(max_step2/norm2)/lambda;
			}
		}

		x += lambda*direction;

		++stats.iters;
		r = F(x);
		res2 = dot(r,r);

		if( no_jac ) continue;

		arma::mat Jacn = J(x);
		if( arma::rcond(Jacn) < opts.tol ){
			if( !low_rcond_warn ){
				std::cerr << "Warning! J close to "
				          << "singular! Falling back "
				          << "on old value!\n";
				low_rcond_warn = true;
			}
		}else{
			Jac = Jacn;
			if( opts.precondition ){
				// We solve direction = -Jac^{-1} r
				// So A*direction = -A*direction*Jac^{-1} r
				for( std::size_t i = 0; i < N; ++i ){
					P(i,i) = 1.0 / Jac(i,i);
					Pi(i,i) = Jac(i,i);
				}
			}
		}
	}
	if( stats.iters == opts.maxit && res2 > tol2 ){
		stats.conv_status = NOT_CONVERGED;
	}else{
		stats.conv_status = SUCCESS;
	}
	stats.res = std::sqrt( res2 );
	return x;
}


/**
   \brief Solves F(x) = 0 with x0 as initial guess.

   This function attempts to find a root to the nonlinear
   system of equations F(x) = 0. If the Jacobi matrix is
   given it uses full Newton iteration. If not, it uses
   Broyden's "good" method.

   \param F     Function to find root of
   \param x     Initial guess, will contain the solution.
   \param opts  Solver options struct.
   \param stats Solver status struct.
   \param J     Jacobi matrix of function (optional)

   \returns the root.
*/
template <typename func_rhs, typename func_Jac >
arma::vec solve( const func_rhs &F, arma::vec x,
                 const options &opts, status &stats,
                 const func_Jac &J  )
{
	my_timer *timer = nullptr;
	if( opts.time_internals ){
		timer = new my_timer( std::cerr );
		if(timer) timer->tic();
	}
	arma::vec root = newton_iterate( F, x, opts, stats, J );
	if( opts.time_internals ){
		if(timer){
			if( stats.conv_status == NOT_CONVERGED ){
				timer->toc("Newton iteration (failed)");
			}else{
				timer->toc("Newton iteration (success)");
			}
			delete timer;
		}
	}
	return root;
}

/**
    \brief Solves F(x) = 0 with x0 as initial guess.

    This uses Broyden's method.

    \overload solve.
*/
template <typename func_rhs>
arma::vec solve( const func_rhs &F, arma::vec x,
                 const options &opts, status &stats )
{
	my_timer *timer = nullptr;
	if( opts.time_internals ){
		timer = new my_timer( std::cerr );
		if(timer) timer->tic();
	}
	arma::vec root = broyden_iterate( F, x, opts, stats );
	if( opts.time_internals ){
		if(timer){
			timer->toc("Broyden iteration");
			delete timer;
		}
	}
	return root;
}


/// \brief A namespace with some functions to test the solvers on.
namespace test_functions {

/// \brief Rosenbrock's function
double rosenbrock_f( const arma::vec &x, double a, double b );
/// \brief Gradient of Rosenbrock's function
arma::vec rosenbrock_F( const arma::vec &x, double a, double b );
/// \brief Hessian of Rosenbrock's function
arma::mat rosenbrock_J( const arma::vec &x, double a, double b );

} // namespace test_functions

} // namespace newton


#endif // NEWTON
