#include "integrator.hpp"


template <typename functor_type>
int integrator::odeint( functor_type &func, const arma::vec &y0,
                        double t0, double t1, double dt )
{
	double t = t0;

	std::size_t Neq = y0.size();
	std::size_t Ns  = sc.b.size();
	std::size_t N   = Neq * Ns;

	arma::vec y = y0;
	arma::vec K_np, K_n;
	K_np.zeros( N );
	K_n.zeros( N );

	newton::options opts;
	opts.maxit = 50;
	opts.tol = 1e-1 * int_opts.rel_tol;
	opts.refresh_jac = false;
	// opts.limit_step = true;

	newton::status newton_stats;

	const irk::solver_coeffs &sc = sc;

	while( t < t1 ){
		if( t + dt > t1 ){
			dt = t1 - t;
		}

		// Calculate the stages:
		auto J = irk::construct_J( t, y, K_n, dt, sc, func );

		// Use newton iteration to find the Ks for the next level:
		auto stages_func = [&t, &y, &dt, &sc, &func]( const arma::vec &K ){
			return irk::construct_F( t, y, K, dt, sc, func );
		};

		auto stages_jac_const = [&J]( const arma::vec &K ){
			return J;
		};


		irk::newton_wrapper<decltype(stages_func),
		                    decltype(stages_jac_const),
		                    typename functor_type::jac_type>
			nw( stages_func, stages_jac_const );

		K_np = newton::newton_iterate( nw, K_n, opts, newton_stats );

		int newton_status = newton_stats.conv_status;
		if( newton_status != newton::status::SUCCESS ){
			// Iteration failed. Reject.
			dt *= 0.5;
			continue;
		}

		// Apparently your stages are alright. Calculate the error:
		arma::vec delta_y, delta_alt;
		delta_y.zeros( Neq );
		delta_alt.zeros( Neq );

		for( std::size_t i = 0; i < Ns; ++i ){
			std::size_t i0 = i*Neq;
			std::size_t i1 = (i+1)*Neq - 1;

			delta_y   += sc.b[i]  * K_np.subvec( i0, i1 );
			delta_alt += sc.b2[i] * K_np.subvec( i0, i1 );

		}

		arma::vec yn = y + dt * delta_y;
		double err = estimate_error( K_np, y, yn, yalt, dt );






	}

	return 0;
}


template <typename functor_type>
double integrator::estimate_error( const arma::vec &K, const arma::vec &y_new,
                                   const arma::vec &y_alt,
                                   const typename functor_type::jac_type &J,
                                   double dt )
{
	double err = 0.0;

	std::size_t Neq = y0.size();
	std::size_t Ns  = sc.b.size();
	std::size_t N   = Neq * Ns;

	typename functor_type::jac_type I;
	double gamma0 = sc.b2(0);
	I.eye( Neq, Neq );
	arma::vec delta_y = y_alt - y_new;
	arma::vec err = arma::solve( I - dt * gamma0 * J, delta_y );
	double n = 0.0;
	for( std::size_t i = 0; i < Neq; ++i ){
		double erri = err(i);
		double y0i = std::fabs( y_new(i) );
		double y1i = std::fabs( y_alt(i) );

		double sci = int_opts.abs_tol + std::max( y0i, y1i ) * int_opts.rel_tol;

		err += erri*erri / (sci*sci);
		n += 1.0;
	}

	return std::sqrt( err / n );
}
