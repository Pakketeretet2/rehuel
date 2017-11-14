#ifndef RADAU_HPP
#define RADAU_HPP

// A library to solve ODEs

// Constructs the correct Jacobi matrix to solve
// F( t + ci*dt, y + dt*Sum_{j=0}^{N-1} (a_ij k_j ) ) - k_i

#define AMRA_USE_CXX11
#include <armadillo>
#include <cassert>

#include "my_timer.hpp"
#include "newton.hpp"

namespace radau {

/**
   Contains the Butcher tableau plus time step size.
*/
struct solver_coeffs
{
	arma::vec b, c;
	arma::mat A;
	double dt;

	arma::vec b2; // To use for embedding.
	bool FSAL;    // If FSAL, an optimization is possible.
};


struct solver_options {
	enum internal_solvers {
		BROYDEN = 0,
		NEWTON = 1
	};

	solver_options()
		: internal_solver(BROYDEN), adaptive_step_size(true),
		  local_tol(1e-6) {}

	int internal_solver;
	bool adaptive_step_size;
	double local_tol;
};

enum rk_methods {
	EXPLICIT_EULER      = 10,
	CLASSIC_RK4         = 11,
	BOGACKI_SHAMPINE2_3 = 12,
	CASH_KARP5_4        = 13,
	DORMAND_PRINCE5_4   = 14,

	IMPLICIT_EULER      = 20,
	RADAU_IIA_32        = 21,
	LOBATTO_IIIA_43     = 22,
	GAUSS_LEGENDRE_65   = 23
};

enum status_codes {
	SUCCESS = 0,
	DT_TOO_LARGE  =  1,
	DT_TOO_SMALL  =  2,
	GENERAL_ERROR = -1,
	INTERNAL_SOLVE_FAILURE = -3,
	TIME_STEP_TOO_SMALL = -4
};

bool verify_solver_coeffs( const solver_coeffs &sc );

solver_coeffs get_coefficients( int method );
solver_options default_solver_options();


// Attempts to find a more optimal time step size
double get_better_time_step( double dt_old, double error_estimate,
                             const solver_options &opts );



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
int take_time_step( double t, arma::vec &y, double dt,
                    const radau::solver_coeffs &sc,
                    const radau::solver_options &solver_opts,
                    const func_type &fun, const Jac_type &jac,
                    bool adaptive_dt, double &err,
                    arma::vec &K )
{
	auto Ns  = sc.b.size();
	auto Neq = y.size();
	auto NN = Ns*Neq;

	arma::mat J( NN, NN );

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
			K = newton::solve( stages_func, K, opts, stats, stages_jac );
			break;
		default:
		case solver_options::BROYDEN:
			K = newton::solve( stages_func, K, opts, stats );
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
				const auto &Ki = K.subvec( offset,
				                           offset + Neq - 1 );
				y_alt += dt * sc.b2[i] * Ki;
			}
		}

		arma::vec yn = y;

		for( unsigned int i = 0; i < Ns; ++i ){
			unsigned int offset = i*Neq;
			const auto &Ki = K.subvec( offset, offset + Neq - 1 );
			yn += dt * sc.b[i] * Ki;
		}

		if( adaptive_dt ){
			auto y_err = yn - y_alt;
			double err_est = arma::norm( y_err, "inf" );
			err = err_est;
			if( err > solver_opts.local_tol ){
				std::cerr << "Error estimate " << err
				          << " is too large!\n";
				return DT_TOO_LARGE;
			}else if( err < solver_opts.local_tol * 0.05 ){
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

	return SUCCESS;
}


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

	t_vals.push_back( t );
	y_vals.push_back( y );

	// Main integration loop:
	int status = 0;
	int steps = 0;

	std::cerr << "# step  t    dt\n";
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
	K.zeros( NN );
	// This might be a better guess:
	/*
	for( int i = 0; i < Ns; ++i ){
		K.subvec( i, i + Neq - 1 ) = y;
	}
	*/

	while( t < t1 ){

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
				std::cerr << "dt is now " << dt << ", was " << old_dt << "\n";
				break;


			case DT_TOO_SMALL:
				if( adaptive_dt ){
					dt = get_better_time_step( dt, err,
					                           solver_opts );
				}else{
					dt *= 1.2;
				}
			case SUCCESS:
				// OK.
				t += old_dt;
				++steps;
				if( steps % 50000 == 0 ){
					print_integrator_stats();
				}

				t_vals.push_back( t );
				y_vals.push_back( y );

				break;
		}

		if( status != SUCCESS && status != DT_TOO_SMALL ){
			continue;
		}

		// Do some other post-processing here.

	}

	if( solver_opts.internal_solver == solver_options::NEWTON ){
		timer.toc( "Integrating with Newton iteration" );
	}else if( solver_opts.internal_solver == solver_options::BROYDEN ){
		timer.toc( "Integrating with Broyden iteration" );
	}

	return 0;
}




} // namespace radau


#endif // RADAU_HPP
