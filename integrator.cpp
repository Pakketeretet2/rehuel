#include "integrator.hpp"



int integrator::odeint( functor &func, const arma::vec &y0,
                        double t0, double t1, double dt,
                        std::vector<double> &tvals,
                        std::vector<arma::vec> &yvals )
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
	opts.maxit = 250;
	opts.tol = 0.7 * int_opts.rel_tol;
	opts.limit_step = false;

	newton::status newton_stats;
	long long int step = 0;


	std::ofstream err_log( "error.dat" );
	std::ofstream sol_log( "sols.dat" );

	sol_log << step << " " << t;
	for( std::size_t i = 0; i < Neq; ++i ){
		sol_log << " " << y[i];
	}
	sol_log << "\n";

	double dts[3];
	double errs[3];
	dts[0] = dts[1] = dts[2] = dt;
	errs[0] = errs[1] = errs[2] = 0.9;

	std::cerr  << "    Rehuel: step  t  dt   err   iters\n";

	err_log << "# step t dt err iters res scale27 scale28 new_dt\n";

	std::size_t n_reject  = 0;
	std::size_t n_attempt = 0;

	bool alternative_error_formula = true;

	// Make sure you stop exactly at t = t1.
	while( t < t1 ){

		// ****************  Calculate stages:   ************
		if( t + dt > t1 ){
			dt = t1 - t;
			std::cerr << "    Rehuel: Last step coming up... t = "
			          << t << ", dt = " << dt << ".\n";
		}
		++n_attempt;

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
		// K_np = newton::broyden_iterate( nw, K_n, opts, newton_stats );

		// *********** Verify Newton iteration convergence ************
		int newton_status = newton_stats.conv_status;
		if( newton_status != newton::SUCCESS ){
			dt *= 0.1;
			++n_reject;
			continue;
		} else {
			// std::cerr << "    Rehuel: Newton succeeded with dt = "
			//           << dt << "\n";
		}


		// ****************  Construct solution at t + dt   ************
		arma::vec delta_y, delta_alt;
		std::size_t Neq = y.size();
		delta_y.zeros( Neq );
		delta_alt.zeros(Neq);

		for( std::size_t i = 0; i < Ns; ++i ){
			std::size_t i0 = i*Neq;
			std::size_t i1 = (i+1)*Neq - 1;
			auto Ki = K_np.subvec( i0, i1 );
			delta_y   += sc.b[i]  * Ki;
			delta_alt += sc.b2[i]  * Ki;
		}

		arma::vec y_n   = y + dt * delta_y;
		arma::vec y_alt = y + dt * delta_alt;

		arma::vec delta_delta = dt*( delta_alt - delta_y );

		// **************      Estimate error:    **********************
		double gamma = sc.b2[0];
		arma::mat I, J0;
		I.eye(Neq, Neq);
		J0 = func.jac( t, y );
		// Formula 8.19:
		arma::vec err_8_19 = arma::solve( I - gamma * dt * J0,
		                                  delta_delta );
		arma::vec err_est = err_8_19;

		// Alternative formula 8.20:

		if( alternative_error_formula ){
			// Use the alternative formulation:
			arma::vec stage_0_delta = gamma*func.fun( t, y + err_est );
			arma::vec delta_alt_alt = stage_0_delta;
			for( std::size_t i = 1; i < Ns; ++i ){
				std::size_t i0 = i*Neq;
				std::size_t i1 = (i+1)*Neq - 1;
				auto Ki = K_np.subvec( i0, i1 );
				delta_alt_alt += sc.b2[i] * Ki;
			}
			arma::vec dd_alt = dt*( delta_alt_alt - delta_y );
			err_est = arma::solve( I - gamma * dt * J0, dd_alt );
		}

		double err_tot = 0.0;
		double n = 0.0;
		for( std::size_t i = 0; i < err_est.size(); ++i ){
			double erri = err_est[i];
			double sci  = int_opts.abs_tol;
			double y0i = std::fabs( y[i] );
			double y1i = std::fabs( y_n[i] );

			sci += int_opts.rel_tol * std::max( y0i, y1i );
			err_tot += erri*erri / sci / sci;
			n += 1.0;
		}

		double err = std::sqrt( err_tot / n );
		if( err == 0 ) err = 3e-16;
		errs[2] = errs[1];
		errs[1] = errs[0];
		errs[0] = err;

		if( err > 1.0 ){
			// This is bad.
			alternative_error_formula = true;
			integrator_status = 1;
			++n_reject;
		}


		// **************      Find new dt:    **********************
		double fac = 0.9 * ( 2*opts.maxit + 1.0 );
		fac /= ( 2*opts.maxit + newton_stats.iters );

		double expt = 1.0 / ( 1.0 + std::min( sc.order, sc.order2 ) );
		double err_inv = 1.0 / err;
		double scale_27 = std::pow( err_inv, expt );
		double dt_rat = dts[0] / dts[1];
		double err_frac = errs[1] / errs[0];
		double err_rat = std::pow( err_frac, expt );
		double scale_28 = scale_27 * dt_rat * err_rat;

		double new_dt = fac * dt * std::min( scale_27, scale_28 );


		// **************    Update y and time   ********************


		err_log << step << " " << t << " " << dt << " " << err << " "
		        << newton_stats.iters << " " << newton_stats.res << " "
		        << scale_27 << " " << scale_28 << " " << new_dt << "\n";

		if( integrator_status == 0 ){

			y = y_n;
			t += dt;
			++step;

			tvals.push_back(t);
			yvals.push_back(y_n);


			alternative_error_formula = false;

			sol_log << step << " " << t;

			for( std::size_t i = 0; i < Neq; ++i ){
				sol_log << " " << y[i];
			}
			sol_log << "\n";

			// K_n = K_np; // This seems to do better.
		}

		// **************      Actually set the new dt:    **********************

		dt = new_dt;
		dts[2] = dts[1];
		dts[1] = dts[0];
		dts[0] = dt;

	}

	double accept_rat = static_cast<double>(step) / n_attempt;

	std::cerr << "    Rehuel: Done integrating ODE over [ " << t0 << ", "
	          << t1 << " ].\n";
	std::cerr << "            Number of succesful steps: " << step
	          << " / " << n_attempt
	          << ", number of rejected steps: " << n_reject << ".\n"
	          << "            Accept ratio: " << accept_rat
	          << ", reject ratio: " << 1.0 - accept_rat << "\n";

	return 0;
}



double integrator::estimate_error( functor &func,
                                   double t,
                                   const arma::vec &y0,
                                   const arma::vec &yn,
                                   const arma::vec &yalt,
                                   const arma::vec &Ks,
                                   const arma::mat &J,
                                   double gamma, double dt,
                                   arma::vec &err_est, bool alt_formula )
{
	std::size_t Neq = y0.size();
	arma::mat I( Neq, Neq );

	arma::vec delta_y = yalt - yn;
	// Solving stiff ODEs eq. 8.19:
	arma::vec err_8_19 = arma::solve( I - dt*gamma*J, delta_y );

	err_est = err_8_19;
	if( false && alt_formula ){
		std::size_t Ns = sc.b.size();
		arma::vec y0_alt = y0 + err_8_19;
		arma::vec stage_0_alt = gamma*func.fun( t, y0_alt );

		arma::vec yalt_alt = stage_0_alt;
		for( std::size_t i = 1; i < Ns; ++i ){
			std::size_t i0 = i*Neq;
			std::size_t i1 = (i+1)*Neq - 1;
			auto Ki = Ks.subvec( i0, i1 );
			yalt_alt += sc.b2[i] * Ki;
		}
		yalt_alt *= dt;
		arma::vec delta_y_alt = yalt_alt - yn;
		arma::vec err_8_20 = arma::solve( I - dt*gamma*J, delta_y_alt );

		err_est = err_8_20;
	}

	double err_tot = 0.0;
	double n = 0.0;
	for( std::size_t i = 0; i < Neq; ++i ){
		double erri_2 = err_est(i)*err_est(i);
		double sci    = int_opts.abs_tol;
		double y0i = std::fabs( y0[i] );
		double y1i = std::fabs( yn[i] );

		sci += std::max( y0i, y1i ) * int_opts.rel_tol;

		double sci_2  = sci*sci;
		err_tot += erri_2 / sci_2;
		n += 1.0;
	}
	return std::sqrt( err_tot / n );
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



double integrator::get_new_dt( double dts[3], double errs[3],
                               int n_iters, int n_maxit )
{
	double min_order = std::min( sc.order, sc.order2 );
	double expt = 1.0 / ( 1.0 + min_order );

	double newt_weight = (1.0 + 2*n_maxit) / ( n_iters + 2.0*n_maxit );
	double fac = 0.9 * newt_weight;

	double err     = std::max( errs[0], 3e-40 );
	double err_old = std::max( errs[1], 3e-40 );

	double inv_err = 1.0 / err;
	double err_frac = err_old / err;



	double scale_27 = fac * std::pow( inv_err, expt );
	double scale_28 = scale_27*(dts[0]/dts[1])*std::pow( err_frac, expt );

	// std::cerr  << "scales are " << scale_27 << ", " << scale_28 << "\n";

	return dts[0] * std::min( scale_27, scale_28 );

}
