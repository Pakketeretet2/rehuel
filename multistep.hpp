#ifndef MULTISTEP_HPP
#define MULTISTEP_HPP

#include "cyclic_buffer.hpp"
#include "enums.hpp"
#include "newton.hpp"


#include <armadillo>
#include <vector>

namespace multistep {

/**
   \brief options for the time integrator.
 */
struct solver_options {
	/// \brief Constructor with default values.
	solver_options() : solver_type(ms_methods::BDF),
	                   order(1), out_int( 1000 ) {}

	~solver_options()
	{ }


	/// Solver type
	int solver_type;
	int order;

	/// Output interval;
	int out_int;

};

template <typename func_type, typename Jac_type> inline
int odeint_BDF( double t0, double t1, double dt,
         const solver_options &solver_opts,
         const arma::vec &y0, const func_type &fun, const Jac_type &jac,
         std::vector<double> &t_vals, std::vector<arma::vec> &y_vals )
{
	double t = t0;
	int step = 0;

	arma::vec y = y0;
	y_vals.push_back(y);
	t_vals.push_back(t);

	// Coefficients for the different BDF methods:
	arma::mat Cfs = { {    -1.0,     0.0, 0.0, 0.0, 0.0, 0.0 },
	                  { -4.0/3.0, 1.0/3.0, 0.0, 0.0, 0.0, 0.0 },
	                  { -18./11.,  9./11., 2.0/11., 0.0, 0.0, 0.0 },
	                  { -48./25., 36./25., -16./25., 12./25., 0.0, 0.0 },
	                  { -300./137., 300./137., -200./137., 75./137., -12./137., 0.0 },
	                  { -360./147., 450./147., -400./147., 225./147., -72./147., 10./147.} };
	arma::vec Bfs = { 1.0, 2.0/3.0, 6.0/11.0, 12.0/25.0, 60.0/137.0, 60.0/147.0 };


	std::size_t N = y0.size();
	arma::mat I(N, N);
	I.eye(N,N);

	newton::options opts;
	opts.tol = 1e-9;
	opts.maxit = 1000;
	newton::status stats;

	cyclic_buffer<arma::vec> previous_ys( solver_opts.order );
	previous_ys.push_back( y );
	arma::vec history( N );
	history.zeros( N,1 );
	while( t < t1 ){
		std::size_t coeff_idx = step;
		if( coeff_idx >= previous_ys.size() ){
			coeff_idx = previous_ys.size()-1;
		}

		for( int i = 0; i < previous_ys.size(); ++i ){
			double ci = Cfs(coeff_idx, i);
			history += ci * previous_ys[ i ];
		}
		double beta = Bfs(coeff_idx);

		// Solve yn + history = f(t+dt, yn) -->
		auto rhs_F = [&fun,&history, &t, &dt, &beta]( arma::vec yn )
			{ return beta*dt*fun(t + dt, yn) - history - yn; };
		auto rhs_J = [&jac, &history, &t, &dt, &I, &beta]( arma::vec yn )
			{ return beta*dt*jac(t + dt, yn) - I; };

		arma::vec yn = newton::solve( rhs_F, y, opts, stats );

		previous_ys.push_back( yn );

		t += dt;
		++step;
		if( step % 1000 == 0 ){
			std::cerr << "At t = " << t << "\n";
		}
		t_vals.push_back(t);
		y_vals.push_back(yn);
		y = yn;
	}

	return 0;
}


/**
   \brief Time-integrate a given ODE from t0 to t1, starting at y0

   t_vals and y_vals shall be unmodified upon failure.

   \param t0           Starting time
   \param t1           Final time
   \param dt           Time step size
   \param solver_opts  Options for the internal solver
   \param y0           Initial values
   \param fun          RHS to the ODE to integrate
   \param t_vals       Will contain the time points corresponding to obtained y
   \param y_vals       Will contain the numerical solution to the ODE.

   \returns a status code (see \ref odeint_status_codes)
*/
template <typename func_type, typename Jac_type> inline
int odeint( double t0, double t1, double dt,
            const solver_options &solver_opts,
            const arma::vec &y0, const func_type &fun, const Jac_type &jac,
            std::vector<double> &t_vals, std::vector<arma::vec> &y_vals,
            std::ostream *errout )
{
	switch( solver_opts.solver_type ){
		case BDF:
		default:
			return odeint_BDF( t0, t1, dt, solver_opts, y0, fun,
			                   jac, t_vals, y_vals );
	}
}

} // namespace multistep


#endif // MULTISTEP_HPP
