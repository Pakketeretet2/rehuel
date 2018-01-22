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
	            max_step(-1), refresh_jac(true), precondition(false){}

	double tol;           ///< Desired tolerance.
	int maxit;            ///< Maximum number of iterations
	bool time_internals;  ///< Print timings for solver
	double max_step;      ///< Limit update to this length in N-D.

	/// When Using Newton's method, if this is false, the Jacobi matrix
	/// is constructed only once at the beginning and never updated.
	bool refresh_jac;

	/// If true, precondition the Jacobi matrix.
	bool precondition;

	/// If true, tries to limit the step to two extremes.
	bool limit_step;
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
};



/**
    \brief Approximates the Jacobi matrix with finite differences.

    \param y    Point about which to approximate Jacobi matrix
    \param fun  Function to determine Jacobi matrix for
    \param h    Finite difference step size.
*/
template <typename functor_type> inline
arma::mat approx_jacobi_matrix( const arma::vec &y, functor_type &func,
                                double h )
{
	std::size_t N = y.size();
	arma::mat J_approx(N,N);
	J_approx.zeros(N,N);
	arma::vec f0 = func.fun(y);
	arma::vec new_yp = y;
	arma::vec new_ym = y;

	for( std::size_t j = 0; j < N; ++j ){
		double old_y_j = y(j);

		new_yp(j) += h;
		new_ym(j) -= h;
		arma::vec fp = func.fun( new_yp );
		arma::vec fm = func.fun( new_ym );

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
template <typename functor_type> inline
bool verify_jacobi_matrix( const arma::vec &y, functor_type &func )
{
	arma::mat J_approx = approx_jacobi_matrix( y, func, 1e-4 );
	arma::mat J_fun = func.jac( y );
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
		std::cerr << "Jacobi matrix seems iffy with max diff of "
		          << max_diff << "\n";
		return false;
	}else{
		std::cerr << "Jacobi matrix seems fine with max diff of "
		          << max_diff << "\n";
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
template <typename functor_type>
arma::vec broyden_iterate( functor_type &func, arma::vec x,
                           const options &opts, status &stats )
{
	stats.conv_status = SUCCESS;
	double tol2 = opts.tol*opts.tol;
	arma::vec r = func.fun(x);
	double res2 = arma::dot( r, r );
	stats.iters = 1;

	std::size_t N = x.size();
	arma::vec x0 = x;
	arma::vec f0 = r;

	arma::mat Jaci( N, N );
	Jaci.eye(N,N);

	auto print_stuff = [&stats, &x, &res2](){
		std::cerr << "Step " << stats.iters << ", res2 = " << res2 << ", x =";
		for( std::size_t i = 0; i < x.size(); ++i ){
			std::cerr << " " << x[i];
		}
		std::cerr << "\n";};

	// print_stuff();

	double max_step2;
	if( opts.max_step > 0 ) max_step2 = opts.max_step*opts.max_step;
	else max_step2 = -1;

	while( res2 > tol2 && stats.iters < opts.maxit ){
		double lambda = 1.0 / sqrt(1.0 + res2 );
		arma::vec direction = -lambda*Jaci*f0;

		if( max_step2 > 0 ){
			double norm2 = arma::dot( direction, direction );
			if( lambda*lambda*norm2 > max_step2 ){
				direction *= std::sqrt(max_step2/norm2)/lambda;
			}
		}


		x = x0 + direction;
		r = func.fun(x);
		arma::vec dx = x - x0;
		arma::vec df = r - f0;

		arma::vec normm = (dx.t() * Jaci) * df;
		double norm = normm(0);
		arma::vec left_part = ( dx - Jaci*df );
		arma::rowvec right_part = dx.t() * Jaci;
		Jaci += left_part * right_part / norm;
		res2 = arma::dot( r, r );

		++stats.iters;

		// print_stuff();

		f0 = r;
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
template <typename functor_type> inline
arma::vec gauss_seidel( functor_type &func, arma::vec x,
                        const options &opts, status &stats )
{
	arma::vec yn = func.fun(x);
	double res2 = arma::norm( yn, 2 );
	double tol2 = opts.tol * opts.tol;
	std::cerr << "Starting Gauss-Seidel at " << stats.iters
	          << " iters, res = " << std::sqrt(res2) << "\n";
	while( res2 > tol2 && stats.iters < opts.maxit ){
		x = yn;
		yn = func.fun(x);
		++stats.iters;
		res2 = arma::norm( yn, 2 );
	}

	return x;
}



/**
   \brief Templated implementation of Newton's method.

   \param func  The functor for which the root of func.fun is to be found.
   \param x     Initial guess for root.
   \param opts  Options for solver (see \p options)
   \param stats Will contain solver statistics (see \p status)

   \returns the root of F(x).
*/
template <typename functor_type, bool refresh_jac, bool precondition,
          bool time_internals, bool limit_step> inline
arma::vec newton_iterate_impl( functor_type &func, arma::vec x,
                               const options &opts, status &stats )
{
	stats.conv_status = SUCCESS;
	double tol2 = opts.tol*opts.tol;
	arma::vec r = func.fun(x);
	double res2 = arma::dot( r, r );
	stats.iters = 1;

	std::size_t N = x.size();
	arma::vec x0 = x;
	arma::vec f0 = r;

	arma::vec direction;

	typename functor_type::jac_type L(N, N), U(N,N);


	double max_step2;
	if( opts.max_step > 0 ) max_step2 = opts.max_step*opts.max_step;
	else max_step2 = -1;

	auto J = func.jac(x);
	if( !refresh_jac ){
		// Then you might as well LU decompose the system here:
		arma::lu( L, U, J );
	}

	typename functor_type::jac_type P;
	P.eye(N,N);


	auto print_stuff = [&stats, &x, &res2](){
		std::cerr << "Step " << stats.iters << ", res2 = " << res2 << ", x =";
		for( std::size_t i = 0; i < x.size(); ++i ){
			std::cerr << " " << x[i];
		}
		std::cerr << "\n";};

	// print_stuff();

	while( res2 > tol2 && stats.iters < opts.maxit ){
		double lambda = 1.0;
		if( limit_step ){
			lambda = (1.0 + opts.tol*res2)  / (1.0 + res2);
		}

		if( precondition ){

			for( std::size_t i = 0; i < N; ++i ){
				P(i,i) = 1.0 / J(i,i);
			}
			direction = -arma::solve(P*J, P*r);

		}else if( !refresh_jac ){

			arma::vec tmp = arma::solve( arma::trimatl(L), r );
			direction = -arma::solve(arma::trimatu(U), tmp);

		}else{
			direction = -arma::solve(J, r);
		}

		if( max_step2 > 0 ){
			double norm2 = arma::dot( direction, direction );
			if( lambda*lambda*norm2 > max_step2 ){
				direction *= std::sqrt(max_step2/norm2)/lambda;
			}
		}


		x = x0 + direction;
		if( refresh_jac ){
			auto Jn = func.jac(x);
			if( arma::rcond(Jn) >= opts.tol ) J = Jn;
		}

		r = func.fun(x);
		res2 = arma::dot( r, r );

		++stats.iters;
		// print_stuff();
		f0 = r;
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
   \brief Performs Newton's method to solve non-linear system F(x) = 0.

   \param func  The functor for which the root of func.fun is to be found.
   \param x     Initial guess for root.
   \param opts  Options for solver (see \p options)
   \param stats Will contain solver statistics (see \p status)

   \returns the root of F(x).
*/
template <typename functor_type> inline
arma::vec newton_iterate( functor_type &func, arma::vec x,
                          const options &opts, status &stats )
{
	int option_bits = 0;
	option_bits += 1 * ( opts.refresh_jac == true );
	option_bits += 2 * ( opts.precondition == true );
	option_bits += 4 * ( opts.time_internals == true );
	option_bits += 8 * ( opts.limit_step == true );

	//std::cerr << "Option combo is " << option_bits << ".\n";

	switch(option_bits){
		default:
		case 0:
			return newton_iterate_impl<functor_type, 0, 0, 0, 0>(
				func, x, opts, stats );
		case 1:
			return newton_iterate_impl<functor_type, 1, 0, 0, 0>(
				func, x, opts, stats );
		case 2:
			return newton_iterate_impl<functor_type, 0, 1, 0, 0>(
				func, x, opts, stats );
		case 3:
			return newton_iterate_impl<functor_type, 1, 1, 0, 0>(
				func, x, opts, stats );
		case 4:
			return newton_iterate_impl<functor_type, 0, 0, 1, 0>(
				func, x, opts, stats );
		case 5:
			return newton_iterate_impl<functor_type, 1, 0, 1, 0>(
				func, x, opts, stats );
		case 6:
			return newton_iterate_impl<functor_type, 0, 1, 1, 0>(
				func, x, opts, stats );
		case 7:
			return newton_iterate_impl<functor_type, 1, 1, 1, 0>(
				func, x, opts, stats );
		case 8:
			return newton_iterate_impl<functor_type, 0, 0, 0, 1>(
				func, x, opts, stats );
		case 9:
			return newton_iterate_impl<functor_type, 1, 0, 0, 1>(
				func, x, opts, stats );
		case 10:
			return newton_iterate_impl<functor_type, 0, 1, 0, 1>(
				func, x, opts, stats );
		case 11:
			return newton_iterate_impl<functor_type, 1, 1, 0, 1>(
				func, x, opts, stats );
		case 12:
			return newton_iterate_impl<functor_type, 0, 0, 1, 1>(
				func, x, opts, stats );
		case 13:
			return newton_iterate_impl<functor_type, 1, 0, 1, 1>(
				func, x, opts, stats );
		case 14:
			return newton_iterate_impl<functor_type, 0, 1, 1, 1>(
				func, x, opts, stats );
		case 15:
			return newton_iterate_impl<functor_type, 1, 1, 1, 1>(
				func, x, opts, stats );
	}
}



/// \brief A namespace with some functions to test the solvers on.
namespace test_functions {

/// \brief Rosenbrock's function functor
struct rosenbrock_func
{
	typedef arma::mat jac_type;

	rosenbrock_func(double a, double b) : a(a), b(b){}
	/// \brief Function itself
	double f( const arma::vec &x );

	/// \brief Gradient of Rosenbrock's function
	virtual arma::vec fun( const arma::vec &x );
	/// \brief Hessian of Rosenbrock's function
	virtual arma::mat jac( const arma::vec &x );

	double a, b;
};

} // namespace test_functions

} // namespace newton


#endif // NEWTON
