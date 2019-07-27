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
   \file newton.hpp
*/

#ifndef NEWTON_HPP
#define NEWTON_HPP

// Functions for performing Newton iteration.

#include "my_timer.hpp"

#include "arma_include.hpp"
#include <iomanip>
#include <fstream>

typedef arma::vec vec_type;
typedef arma::mat mat_type;



/// \brief A namespace with solvers for non-linear systems of equations.
namespace newton {

/// \brief Return codes for newton_solve.
enum newton_solve_ret_codes {
	SUCCESS = 0,                ///< Converged to tolerance
	INCREMENT_DIVERGE,          ///< Increment in x increased
	ITERATION_ERROR_TOO_LARGE,  ///< Estimated iteration error is too large
	MAXIT_EXCEEDED,             ///< Number of iterations too large
	GENERIC_ERROR               ///< Generic failure
};


/**
   \brief contains options for the solver.
*/
struct options {
	options() : tol(1e-4), dx_delta(1e-4), maxit(500),
	            time_internals(false), max_step(-1), refresh_jac(true),
	            precondition(false), limit_step(false) {}

	double tol;           ///< Desired tolerance.
	double dx_delta;      ///< Terminate if the increment is below this.
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
	status() : conv_status(SUCCESS), res(0.0), iters(0), eta_final(0.0){}


	int conv_status;  ///< Status code, see \ref newton_solve_ret_codes
	double res;       ///< Final residual (F(x_root)^2)
	int iters;        ///< Number of iterations actually used
	double eta_final; ///< Last value of eta_k
};



/**
   \brief A wrapper struct to make Newton iteration easier.

   It wraps a function that returns an RHS and a function that
   returns a Jacobi matrix.
*/
template <typename f_func, typename J_func, typename Jac_type>
struct newton_lambda_wrapper
{
	typedef Jac_type jac_type;

	newton_lambda_wrapper( f_func &f, J_func &J ) : f(f), J(J) {}

	vec_type fun( const vec_type &K )
	{
		return f(K);
	}

	jac_type jac( const vec_type &K )
	{
		return J(K);
	}

	f_func &f;
	J_func &J;
};




/**
   \brief A wrapper struct to make Newton iteration easier.

   It wraps an ODE functor that returns an RHS and a function that
   returns a Jacobi matrix.
*/
template <typename functor_type>
struct newton_functor_wrapper
{
	typedef typename functor_type::jac_type jac_type;

	newton_functor_wrapper( functor_type &func, double t ) : func(func), t(t) {}

	vec_type fun( const vec_type &Y )
	{
		return func.fun( t, Y );
	}

	jac_type jac( const vec_type &Y )
	{
		return func.jac( t, Y );
	}

	functor_type &func;
	double t;
};




/**
   \brief converts t to a string and pads c until it is width wide.
*/
template <typename T>
std::string to_fixed_w_string( T t, std::size_t wide, char c = ' ' )
{
	std::stringstream ss;
	ss << std::setw(wide) << std::scientific << t;
	std::string s = ss.str();

	if( s.length() > wide ){
		// That won't work...
		std::cerr << "String is wider than width!\n";
	}else{
		std::size_t add = wide - s.length();
		std::string app( add, c );
		s.append( app );
	}
	return s;
}




/**
    \brief Approximates the Jacobi matrix with finite differences.

    \param y    Point about which to approximate Jacobi matrix
    \param fun  Function to determine Jacobi matrix for
    \param h    Finite difference step size.
*/
template <typename functor_type> inline
mat_type approx_jacobi_matrix( const vec_type &y, functor_type &func,
                                double h )
{
	std::size_t N = y.size();
	mat_type J_approx;
	J_approx.zeros(N,N);
	vec_type f0 = func.fun(y);
	vec_type new_yp = y;
	vec_type new_ym = y;

	for( std::size_t j = 0; j < N; ++j ){
		double old_y_j = y(j);

		new_yp(j) += h;
		new_ym(j) -= h;
		vec_type fp = func.fun( new_yp );
		vec_type fm = func.fun( new_ym );

		vec_type delta = fp - fm;
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
bool verify_jacobi_matrix(const vec_type &y, functor_type &func)
{
	mat_type J_approx = approx_jacobi_matrix( y, func, 1e-4 );
	mat_type J_fun = func.jac( y );
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
		return true;
	}
}


/**
   \brief Performs Broyden's method to solve non-linear system F(x) = 0.

   \param F     The non-linear system whose root to find.
   \param x     Initial guess for root.
   \param opts  Options for solver (see \ref options)
   \param stats Will contain solver statistics (see \ref status)
   \param quiet If true, will not print output.

   \returns the root of F(x).
*/
template <typename functor_type>
vec_type broyden_iterate( functor_type &func, vec_type x,
                           const options &opts, status &stats,
                           bool quiet = true )
{
	stats.conv_status = SUCCESS;
	double tol2 = opts.tol*opts.tol;
	vec_type r = func.fun(x);
	double res2 = dot( r, r );
	stats.iters = 1;

	std::size_t N = x.size();
	vec_type x0 = x;
	vec_type f0 = r;

	mat_type Jaci(N,N);
	Jaci.eye( N, N );

	double incr = 2*opts.dx_delta;

	double max_step2;
	if( opts.max_step > 0 ) max_step2 = opts.max_step*opts.max_step;
	else max_step2 = -1;
	bool terminate = false;

	auto print_stats =
		[&opts, &x, &stats, &res2, &incr](){
		int w = 14;
		std::cerr << "    Broyden: " << stats.iters << "/" << opts.maxit
		<< "  " << to_fixed_w_string( res2, w )
		<< "  " << to_fixed_w_string( incr, w )
		<< "          x : (";
		for( std::size_t i = 0; i < x.size(); ++i ){
			std::cerr << " " << x[i];
		}
		std::cerr << " )\n";
	};


	while( !terminate && stats.iters < opts.maxit ){
		if( !quiet ) print_stats();

		double lambda = 1.0 / sqrt(1.0 + res2);
		vec_type direction = -lambda*(Jaci*f0);

		if( max_step2 > 0 ){
			double norm2 = dot( direction, direction );
			if( lambda*lambda*norm2 > max_step2 ){
				direction *= std::sqrt(max_step2/norm2)/lambda;
			}
		}
		incr = arma::norm(direction, "inf");

		x = x0 + direction;
		r = func.fun(x);
		vec_type dx = x - x0;
		vec_type df = r - f0;

		// This is x^T*Jaci...
		arma::rowvec xt_J = dx.t()*Jaci;
		double norm = arma::dot(xt_J, df);
		vec_type left_part = ( dx - Jaci*df );

		Jaci += left_part*xt_J / norm;
		res2 = dot( r, r );

		if( incr < opts.dx_delta ){
			terminate = true;
			stats.conv_status = SUCCESS;
		}
		if( res2 < tol2 ){
			terminate = true;
			stats.conv_status = SUCCESS;
		}

		++stats.iters;

		f0 = r;
		x0 = x;
	}

	stats.res = std::sqrt( res2 );
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
          bool time_internals, bool limit_step, bool quiet> inline
vec_type newton_iterate_impl( functor_type &func, vec_type x,
                               const options &opts, status &stats )
{
	stats.conv_status = SUCCESS;
	stats.iters = 0;
	vec_type r = func.fun(x);
	double res2 = dot( r, r );

	std::size_t N = x.size();
	vec_type x0 = x;
	vec_type xn = x;
	vec_type f0 = r;

	vec_type direction;

	typename functor_type::jac_type L(N, N), U(N,N);

	double max_step2;
	if( opts.max_step > 0 ) max_step2 = opts.max_step*opts.max_step;
	else max_step2 = -1;

	auto J = func.jac(x);
	typename functor_type::jac_type P;
	P.eye(N,N);

	double incr1 = 0;
	double incr0 = 0;

	// Estimated convergence rate:
	double theta_k = 1.0;
	double eta_k = 1.0;

	auto print_stats =
		[&opts, &xn, &stats, &res2, &incr1, &incr0, &theta_k, &eta_k](){
		int w = 14;
		std::cerr << "    Newton: " << stats.iters << "/" << opts.maxit
		          << "  " << to_fixed_w_string( res2, w )
		          << "  " << to_fixed_w_string( incr1, w )
		          << "  " << to_fixed_w_string( incr0, w )
		          << "  " << to_fixed_w_string( theta_k, w )
		          << "  " << to_fixed_w_string( eta_k, w )
		          << "  " << eta_k * incr1 << "/" << opts.tol << "\n"
		          << "          x : (";
		for( std::size_t i = 0; i < xn.size(); ++i ){
			std::cerr << " " << xn[i];
		}
		std::cerr << " )\n";
	};

	bool terminate = false;


	while( !terminate && (stats.iters < opts.maxit) ){
		if( !quiet ) print_stats();

		double lambda = 1.0;
		if( limit_step ){
			lambda = 1.0 / (1.0 + res2);
		}

		try{
			if( precondition ){
				for( std::size_t i = 0; i < N; ++i ){
					if( J(i,i) != 0.0 ){
						P(i,i) = 1.0 / J(i,i);
					}else{
						P(i,i) = 1.0;
					}
				}
				direction = -arma::solve(P*J, P*r);

			}else{
				direction = -arma::solve(J,r);
			}
		}catch( std::exception &e ){
			stats.conv_status = GENERIC_ERROR;
			std::cerr << "Newton caught generic error!\n";
			return x;
		}
		//std::cerr << "Step direction: " << direction << "\n";

		if( max_step2 > 0 ){
			double norm2 = dot( direction, direction );
			if( lambda*lambda*norm2 > max_step2 ){
				direction *= std::sqrt(max_step2/norm2)/lambda;
			}
		}

		xn = x0 + lambda*direction;
		if( refresh_jac ){
			J = func.jac(xn);
		}

		r = func.fun(xn);
		res2 = dot( r, r );

		f0 = r;
		x0 = xn;

		incr0 = incr1;
		incr1 = arma::norm(direction, "inf");

		++stats.iters;

		/*
		  This was supposed to accelerate convergence or something
		  but it does not work.
		if( stats.iters > 1 ){
			theta_k = incr1 / incr0;
			eta_k = theta_k / ( 1.0 - theta_k );
			if( theta_k > 1.0 ){
				//std::cerr << "    Newton: Divergence at step "
				//          << stats.iters << "!\n";
				stats.conv_status = INCREMENT_DIVERGE;
				terminate = true;
			}
		}
		*/

		/*
		if( res2 < tol2 ){
			terminate = true;
			stats.conv_status = SUCCESS;
		}
		*/

		if( incr1 < opts.dx_delta ){
			terminate = true;
			stats.conv_status = SUCCESS;
		}
	}


	if( stats.iters == opts.maxit ){
		stats.conv_status = MAXIT_EXCEEDED;
	}
	stats.res = std::sqrt( res2 );
	stats.eta_final = eta_k;
	// Check whether or not res is NaN or inf or somesuch.
	if( !std::isfinite( stats.res ) ){
		stats.conv_status = GENERIC_ERROR;
	}

	if( !quiet ){
		print_stats();
		std::cerr << "\n";
	}

	if( stats.conv_status == SUCCESS ){
		return xn;
	}else{
		return x;
	}
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
vec_type newton_iterate( functor_type &func, vec_type x,
                          const options &opts, status &stats,
                          bool quiet = true )
{
	int option_bits = 0;
	option_bits += 1 * ( opts.refresh_jac == true );
	option_bits += 2 * ( opts.precondition == true );
	option_bits += 4 * ( opts.time_internals == true );
	option_bits += 8 * ( opts.limit_step == true );
	option_bits += 16* ( quiet == true );

	//std::cerr << "Option combo is " << option_bits << ".\n";

	switch(option_bits){
		default:
		case 0:
			return newton_iterate_impl<functor_type, 0, 0, 0, 0,0>(
				func, x, opts, stats );
		case 1:
			return newton_iterate_impl<functor_type, 1, 0, 0, 0,0>(
				func, x, opts, stats );
		case 2:
			return newton_iterate_impl<functor_type, 0, 1, 0, 0,0>(
				func, x, opts, stats );
		case 3:
			return newton_iterate_impl<functor_type, 1, 1, 0, 0,0>(
				func, x, opts, stats );
		case 4:
			return newton_iterate_impl<functor_type, 0, 0, 1, 0,0>(
				func, x, opts, stats );
		case 5:
			return newton_iterate_impl<functor_type, 1, 0, 1, 0,0>(
				func, x, opts, stats );
		case 6:
			return newton_iterate_impl<functor_type, 0, 1, 1, 0,0>(
				func, x, opts, stats );
		case 7:
			return newton_iterate_impl<functor_type, 1, 1, 1, 0,0>(
				func, x, opts, stats );
		case 8:
			return newton_iterate_impl<functor_type, 0, 0, 0, 1,0>(
				func, x, opts, stats );
		case 9:
			return newton_iterate_impl<functor_type, 1, 0, 0, 1,0>(
				func, x, opts, stats );
		case 10:
			return newton_iterate_impl<functor_type, 0, 1, 0, 1,0>(
				func, x, opts, stats );
		case 11:
			return newton_iterate_impl<functor_type, 1, 1, 0, 1,0>(
				func, x, opts, stats );
		case 12:
			return newton_iterate_impl<functor_type, 0, 0, 1, 1,0>(
				func, x, opts, stats );
		case 13:
			return newton_iterate_impl<functor_type, 1, 0, 1, 1,0>(
				func, x, opts, stats );
		case 14:
			return newton_iterate_impl<functor_type, 0, 1, 1, 1,0>(
				func, x, opts, stats );
		case 15:
			return newton_iterate_impl<functor_type, 1, 1, 1, 1,0>(
				func, x, opts, stats );
		case 16:
			return newton_iterate_impl<functor_type, 0, 0, 0, 0,1>(
				func, x, opts, stats );
		case 17:
			return newton_iterate_impl<functor_type, 1, 0, 0, 0,1>(
				func, x, opts, stats );
		case 18:
			return newton_iterate_impl<functor_type, 0, 1, 0, 0,1>(
				func, x, opts, stats );
		case 19:
			return newton_iterate_impl<functor_type, 1, 1, 0, 0,1>(
				func, x, opts, stats );
		case 20:
			return newton_iterate_impl<functor_type, 0, 0, 1, 0,1>(
				func, x, opts, stats );
		case 21:
			return newton_iterate_impl<functor_type, 1, 0, 1, 0,1>(
				func, x, opts, stats );
		case 22:
			return newton_iterate_impl<functor_type, 0, 1, 1, 0,1>(
				func, x, opts, stats );
		case 23:
			return newton_iterate_impl<functor_type, 1, 1, 1, 0,1>(
				func, x, opts, stats );
		case 24:
			return newton_iterate_impl<functor_type, 0, 0, 0, 1,1>(
				func, x, opts, stats );
		case 25:
			return newton_iterate_impl<functor_type, 1, 0, 0, 1,1>(
				func, x, opts, stats );
		case 26:
			return newton_iterate_impl<functor_type, 0, 1, 0, 1,1>(
				func, x, opts, stats );
		case 27:
			return newton_iterate_impl<functor_type, 1, 1, 0, 1,1>(
				func, x, opts, stats );
		case 28:
			return newton_iterate_impl<functor_type, 0, 0, 1, 1,1>(
				func, x, opts, stats );
		case 29:
			return newton_iterate_impl<functor_type, 1, 0, 1, 1,1>(
				func, x, opts, stats );
		case 30:
			return newton_iterate_impl<functor_type, 0, 1, 1, 1,1>(
				func, x, opts, stats );
		case 31:
			return newton_iterate_impl<functor_type, 1, 1, 1, 1,1>(
				func, x, opts, stats );

	}
}


} // namespace newton


#endif // NEWTON
