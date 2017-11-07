#ifndef RADAU_HPP
#define RADAU_HPP

// A library to solve ODEs

// Constructs the correct Jacobi matrix to solve
// F( t + ci*dt, y + dt*Sum_{j=0}^{N-1} (a_ij k_j ) ) - k_i

#define AMRA_USE_CXX11
#include <armadillo>
#include <cassert>

#include "newton.hpp"

namespace radau {

struct solver_coeffs
{
	arma::vec b, c;
	arma::mat A;
	double dt;
	arma::vec b2; // To use for embedding.
};

enum rk_methods {
	EXPLICIT_EULER = 0,
	IMPLICIT_EULER = 1,
	CLASSIC_RK = 2,
	GAUSS_LEGENDRE_2 = 3,
	RADAU_IIA_2 = 4,
	LOBATTO_IIIA_3 = 5
};

bool verify_solver_coeffs( const solver_coeffs &sc );

solver_coeffs get_coefficients( int method );

template <typename func_type, typename Jac_type> inline
arma::vec construct_F( double t, const arma::vec &y, const arma::vec &K,
                       const radau::solver_coeffs &sc,
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

template <typename func_type, typename Jac_type> inline
arma::mat construct_J( double t, const arma::vec &y, const arma::vec &K,
                       const radau::solver_coeffs &sc,
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

template <typename func_type, typename Jac_type> inline
int take_time_step( double t, arma::vec &y,
                    const radau::solver_coeffs &sc,
                    const func_type &fun, const Jac_type &jac )
{
	auto Ns  = sc.b.size();
	auto Neq = y.size();
	auto NN = Ns*Neq;

	arma::vec F( NN );
	arma::mat J( NN, NN );

	// Use newton iteration to find the Ks for the next level:
	arma::vec K( NN );
	K.zeros();

	double tol = 1e-8;
	double res = 0.0;
	int maxit = 100000;
	int iters = 0;
	int status = 0;

	auto stages_func = [&t, &y, &sc, &fun, &jac]( const arma::vec &K ){
		return construct_F( t, y, K, sc, fun, jac );
	};
	auto stages_jac  = [&t, &y, &sc, &fun, &jac]( const arma::vec &K ){
		return construct_J( t, y, K, sc, fun, jac );
	};

	K = newton::solve( stages_func, stages_jac, K, tol, maxit,
	                   status, res, iters );
	if( status == newton::SUCCESS ){
		// You can update the time step.
		for( unsigned int i = 0; i < Ns; ++i ){
			unsigned int offset = i*Neq;
			const auto &Ki = K.subvec( offset, offset + Neq - 1 );
			y += sc.dt * sc.b[i] * Ki;
		}
	}else{
		// Somehow signal something went wrong.
		return -1;
	}

	return 0;
}


template <typename func_type, typename Jac_type> inline
int odeint( double t0, double t1, const solver_coeffs &sc,
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

	arma::vec y = y0;
	y_vals.push_back( y );
	t_vals.push_back( t );
	double dt = sc.dt;

	// Main integration loop:
	int status = 0;
	int steps = 0;

	std::cerr << "# step  t    dt\n";
	std::cerr << steps << "  " << t << "  " << dt << "\n";

	while( t < t1 ){
		status = take_time_step( sc.dt, y, sc, fun, jac );
		if( status ){
			// Handle special status...
			dt *= 0.5;
			std::cerr << "Newton iteration failed to converge! "
			          << "Halving time step to " << dt << "!\n";
		}else{
			// OK.
			t += sc.dt;
			t_vals.push_back( t );
			y_vals.push_back( y );
			++steps;
			if( steps % 100 == 0 ){
				std::cerr << steps << "  " << t << "  " << dt << "\n";
			}
		}
	}

	return 0;
}

template <typename func_type, typename Jac_type> inline
bool verify_jacobi_matrix( double t, const arma::vec &y,
                           const func_type &fun, const Jac_type &jac )
{
	double h = 1e-4;
	std::size_t N = y.size();
	arma::mat J_approx(N,N);
	J_approx.zeros(N,N);

	for( int j = 0; j < N; ++j ){
		arma::vec new_yp = y;
		arma::vec new_ym = y;

		new_yp(j) += h;
		new_ym(j) -= h;
		arma::vec fp = fun( t, new_yp );
		arma::vec fm = fun( t, new_yp );
		arma::vec delta = fp - fm;
		delta /= 2.0*h;

		for( int i = 0; i < N; ++i ){
			J_approx(i,j) = delta(i,j);
		}
	}

	arma::mat J_fun = jac( t, y );

	double max_diff2 = 0;
	for( int i = 0; i < N; ++i ){
		for( int j = 0; j < N; ++j ){
			double delta = J_approx(i,j) - J_fun(i,j);
			double delta2 = delta*delta;
			if( delta2 > max_diff2 ) delta2 = max_diff2;
		}
	}

	double max_diff = sqrt( max_diff2 );
	std::cerr << "Largest diff is " << max_diff << "\n";
	if( max_diff > 1e-8 ){
		return false;
	}else{
		return true;
	}
}



} // namespace radau


#endif // RADAU_HPP
