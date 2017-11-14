#ifndef NEWTON_HPP
#define NEWTON_HPP

// Functions for performing Newton iteration.

#define AMRA_USE_CXX11

#include <armadillo>
#include <fstream>

#include "my_timer.hpp"


namespace newton {

/// \brief Return codes for newton_solve.
enum newton_solve_ret_codes {
	SUCCESS = 0,
	NOT_CONVERGED = 1
};


struct options {
	options() : tol(1e-4), maxit(500), time_internals(false),
	            max_step(-1), refresh_jac(true){}
	double tol;
	int maxit;
	bool time_internals;
	double max_step;
	bool refresh_jac;
};

struct status {
	status() : conv_status(SUCCESS), res(0.0), iters(0),
	           store_final_f(false), store_final_J(false){}

	int conv_status;
	double res;
	int iters;

	bool store_final_f, store_final_J;
	arma::vec final_f;
	arma::mat final_J;
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
    \param fun Function handle.
    \param jac Jacobi matrix for fun.

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
	for( int i = 0; i < N; ++i ){
		for( int j = 0; j < N; ++j ){
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


template <typename func_rhs>
arma::vec broyden_iterate( const func_rhs &F, arma::vec x,
                           const options &opts, status &stats )
{
	stats.conv_status = SUCCESS;
	double tol2 = opts.tol*opts.tol;
	arma::vec fn = F(x);
	double res2 = arma::dot( fn, fn );
	stats.iters = 0;

	std::size_t N = x.size();
	arma::vec x0 = x;
	arma::vec f0 = fn;

	arma::mat Jaci( N, N );
	double h = 1e-4;
	arma::mat Jac_approx = approx_jacobi_matrix( x, F, h );

	double rcond = arma::rcond(Jac_approx);

	Jaci.zeros( N, N );
	if( rcond > opts.tol ){
		// Approximate the Jacobi matrix with the inverse diagonal:
		for( std::size_t i = 0; i < N; ++i ){
			Jaci(i,i) = 1.0 / Jac_approx(i,i);
		}
	}else{
		Jaci.eye(N,N);
	}


	double max_step2;
	if( opts.max_step > 0 ) max_step2 = opts.max_step*opts.max_step;
	else max_step2 = -1;

	std::ofstream out( "broyden_iterate.dat" );
	auto output_status = [&out, &x, &res2, &stats]() {
		out << stats.iters << " " << res2;
		for( std::size_t i = 0; i < x.size(); ++i ){
			out << " " << x[i];
		}
		out << "\n"; };
	output_status();

	while( res2 > tol2 && stats.iters < opts.maxit ){
		double lambda = 1.0 / ( 1.0 + res2 );
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
		output_status();

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


template <typename func_rhs, typename func_Jac > inline
arma::vec newton_iterate( const func_rhs &F, arma::vec x,
                          const options &opts, status &stats,
                          const func_Jac &J )
{
	if( !verify_jacobi_matrix( x, F, J ) ){
		std::cerr << "Jacobi matrix function seems incorrect! "
		          << "Falling back on Broyden instead!\n";
		return broyden_iterate( F, x, opts, stats );
	}

	stats.conv_status = SUCCESS;
	double tol2 = opts.tol*opts.tol;
	arma::vec r = F(x);
	double res2 = arma::dot( r, r );

	stats.iters = 0;
	unsigned int N = x.size();
	bool low_rcond_warn = false;

	double max_step2;
	if( opts.max_step > 0 ) max_step2 = opts.max_step*opts.max_step;
	else max_step2 = -1;

	arma::mat Jac = J(x);
	if( arma::rcond(Jac) < opts.tol ){
		std::cerr << "Warning! Initial J is close to singular!\n";
	}

	std::ofstream out( "newton_iterate.dat" );
	auto output_status = [&out, &x, &res2, &stats]() {
		out << stats.iters << " " << res2;
		for( std::size_t i = 0; i < x.size(); ++i ){
			out << " " << x[i];
		}
		out << "\n"; };
	output_status();

	while( res2 > tol2 && stats.iters < opts.maxit ){

		double lambda = 1.0 / (1.0 + res2 );
		arma::vec direction = -arma::solve(Jac,r);
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

		output_status();

		if( opts.refresh_jac ){
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
			timer->toc("Newton iteration");
			delete timer;
		}
	}
	return root;
}

/**
    \brief Solves F(x) = 0 with x0 as initial guess.

    This uses Broyden's method.

    \overloads solve.
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


namespace test_functions {

double rosenbrock_f( const arma::vec &x, double a, double b );
arma::vec rosenbrock_F( const arma::vec &x, double a, double b );
arma::mat rosenbrock_J( const arma::vec &x, double a, double b );

} // namespace test_functions

} // namespace newton


#endif // NEWTON
