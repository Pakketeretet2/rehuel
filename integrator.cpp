#include "integrator.hpp"


int integrator::odeint( functor &func, const arma::vec &y0,
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
	long long int step = 0;


	std::ofstream err_log( "error.dat" );
	std::ofstream sol_log( "sols.dat" );

	sol_log << step << " " << t;
	for( std::size_t i = 0; i < Neq; ++i ){
		sol_log << " " << y[i];
	}
	sol_log << "\n";

	double dt_old = dt;

	double err = 0.0;
	double err_old = err;

	// Make sure you stop exactly at t = t1.
	while( true ){
		if( t + dt >= t1 ){
			dt = t1 - t;
		}

		int integrator_status = 0;
		// Calculate the stages:
		auto J = construct_J( t, y, K_n, dt, func );

		// Use newton iteration to find the Ks for the next level:
		auto stages_func = [&t, &y, &dt, &func, this]( const arma::vec &K ){
			return construct_F( t, y, K, dt, func );
		};

		auto stages_jac_const = [&J]( const arma::vec &K ){
			return J;
		};


		newton::newton_lambda_wrapper<decltype(stages_func),
		                              decltype(stages_jac_const),
		                              functor::jac_type>
			nw( stages_func, stages_jac_const );

		K_np = newton::newton_iterate( nw, K_n, opts, newton_stats );

		int newton_status = newton_stats.conv_status;
		if( newton_status != newton::SUCCESS ){
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

		arma::vec yn   = y + dt * delta_y;
		arma::vec yalt = y + dt * delta_alt;
		double gamma0 = 0.274888829595677;
		arma::mat J0 = func.jac( t, y );

		err_old = err;
		err = estimate_error( yn, yalt, J0, gamma0, dt );

		if( err > 1.0 ){
			// This is bad.
			std::cerr << "    Rehuel: Error of " << err
			          << " too large!\n";
			integrator_status = 1;
		}

		arma::mat I;
		I.eye( Neq, Neq );
		arma::vec dy = yalt - yn;
		arma::vec err_est = arma::solve( I - dt * gamma0 * J0, dy );

		double new_dt = get_new_dt( dt, dt_old, err, err_old );

		/*
		std::cerr << "    Rehuel: Step and errors at t = " << t
		          << ", dt = " << dt << ":\n";
		std::cerr << "            y   yn   yalt   dy   erry\n";

		for( std::size_t i = 0; i < Neq; ++i ){
			std::cerr << "           " << y[i] << "    " << yn[i]
			          << "   " << yalt[i] << "   " << "   " << dy[i]
			          << "     " << err_est[i] << "\n";
		}
		std::cerr << "            Error estimate = " << err << ".\n";
		*/
		err_log << step << " " << t << " " << dt << " " << err << "\n";

		// Update y and the time step:
		if( integrator_status == 0 ){
			y = yn;
			t += dt;
			++step;

			sol_log << step << " " << t;

			for( std::size_t i = 0; i < Neq; ++i ){
				sol_log << " " << y[i];
			}
			sol_log << "\n";
		}

		// Adjust time step size:
		dt_old = dt;
		dt = new_dt;

		if( t >= t1 ){
			break;
		}
	}

	return 0;
}



double integrator::estimate_error( const arma::vec &y_new,
                                   const arma::vec &y_alt,
                                   const arma::mat &J,
                                   double gamma, double dt )
{
	std::size_t Neq = y_new.size();

        functor::jac_type I;

	I.eye( Neq, Neq );
	arma::vec delta_y = y_alt - y_new;

	arma::vec err = arma::solve( I - dt * gamma * J, delta_y );
	double n = 0.0;
	double tot_err = 0.0;
	for( std::size_t i = 0; i < Neq; ++i ){
		double erri = err(i);
		double y0i = std::fabs( y_new(i) );
		double y1i = std::fabs( y_alt(i) );

		double sci = int_opts.abs_tol + std::max( y0i, y1i ) * int_opts.rel_tol;

		tot_err += erri*erri / (sci*sci);
		n += 1.0;
	}

	return std::sqrt( tot_err / n );
}


arma::vec integrator::construct_F( double t, const arma::vec &y,
                                   const arma::vec &K, double dt,
                                   functor &func )
{
	auto Ns  = sc.b.size();
	auto Neq = y.size();
	auto NN = Ns*Neq;

	arma::vec F( NN );

	assert( K.size() == NN  && "Size of K is not right!" );
	assert( y.size() == Neq && "Size of y is not right!" );

	const arma::vec &c = sc.c;
	const arma::mat &A = sc.A;

	for( unsigned int i = 0; i < Ns; ++i ){
		double ti = t + dt * c[i];
		arma::vec yi = y;
		for( unsigned int j = 0; j < Ns; ++j ){
			unsigned int offset = j*Neq;
			yi += dt * A(i,j) * K.subvec( offset, offset + Neq - 1 );
		}
		arma::vec ki = K.subvec( i*Neq, i*Neq + Neq - 1 );
		arma::vec tmp = func.fun( ti, yi );
		F.subvec( i*Neq, i*Neq + Neq - 1 ) = tmp - ki;
	}
	return F;
}

arma::mat integrator::construct_J( double t, const arma::vec &y,
                                   const arma::vec &K, double dt,
                                   functor &func )
{
	auto Ns  = sc.b.size();
	auto Neq = y.size();
	auto NN  = Ns*Neq;

	arma::vec F( NN );

	assert( K.size() == NN  && "Size of K is not right!" );
	assert( y.size() == Neq && "Size of y is not right!" );

	const arma::vec &c = sc.c;
	const arma::mat &A = sc.A;

	functor::jac_type J( NN, NN );
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
		auto Ji = func.jac( ti, yi );

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



double integrator::get_new_dt( double dt1, double dt0, double err, double err_old )
{
	double min_order = std::min( sc.order, sc.order2 );
	double expt = 1.0 / ( 1.0 + min_order );
	double fac = 0.9;
	err     = std::max( err, 1e-16 );
	err_old = std::max( err, 1e-16 );

	double inv_err = 1.0 / err;
	// double err_frac = err_old / err;
	double scale_27 = fac * std::pow( inv_err, expt );
	// double scale_28 = (dt0/dt1)*std::pow( err_frac, expt );
	return dt1 * scale_27; // std::min( scale_27, scale_28 );
}
