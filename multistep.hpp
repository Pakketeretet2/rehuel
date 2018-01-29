#ifndef MULTISTEP_HPP
#define MULTISTEP_HPP

/**
   \file multistep.hpp
   Contains stuff for multistep methods.
*/

#include "cyclic_buffer.hpp"
#include "enums.hpp"
#include "newton.hpp"
#include "options.hpp"


#include <armadillo>
#include <cassert>
#include <utility>
#include <vector>

namespace multistep {

/// Ease-of-use typedef for history
typedef std::pair<double, arma::vec> historic_vec;

/**
   \brief options for the time integrator.
 */
struct solver_options : common_solver_options {
	/// \brief Constructor with default values.
	solver_options()
	{ }

	~solver_options()
	{ }
};

/**
   \brief Coefficients for solver.
*/
struct solver_coeffs {

	// For a specific order, the updates are:
	//    cs[i]*y(n+1-i) + ... = b * f(t+dt, y(n+1))     (BDF)
	//    cs[i]*f( t - i*dt, y(n-i) )    + ... = y(n+1)  (Aadams-Bashforth)
	//    cs[i]*f( t+(1-i)dt, y(n+1-i) ) + ... = y(n+1)  (Aadams-Moulton)

	int method_type;
	int order;
	double dt;

	arma::mat cs_bdf;
	arma::mat cs_ab;
	arma::mat cs_am;

	arma::vec b;

};


/**
   \brief A wrapper struct to make Newton iteration easier.

*/
template <typename f_func, typename J_func, typename Jac_type>
struct newton_wrapper
{
	typedef Jac_type jac_type;

	newton_wrapper( f_func &f, J_func &J ) : f(f), J(J) {}

	arma::vec fun( const arma::vec &K )
	{
		return f(K);
	}

	jac_type jac( const arma::vec &K )
	{
		return J(K);
	}

	f_func &f;
	J_func &J;
};



static std::map<int,std::string> ms_method_to_string = {
	FOREACH_MULTISTEP_METHOD(GENERATE_STRING)
};

static std::map<std::string,int> ms_string_to_method = {
	FOREACH_MULTISTEP_METHOD(GENERATE_MAP)
};



/**
   \brief Returns a vector with all method names.
*/
std::vector<std::string> all_method_names();


/**
   \brief Returns default solver options.
   \returns default solver options.
*/
solver_options default_solver_options();


/**
   \brief Returns solver coefficients for given method and order.
*/
solver_coeffs get_coefficients( int method, int order );


/**
   \brief Converts a string with a method name to an int.

   \param name A string describing the method.

   \returns the enum corresponding to given method. See \ref rk_methods
*/
int name_to_method( const std::string &name );


/**
   \brief Converts method code to a human-readable string.

   \note This output shall satisfy
         name_to_method( method_to_name( method ) ) == method.

   \param method The method to convert to a name.

   \returns a string literal representing the method.
*/
const char *method_to_name( int method );




template <typename functor_type> inline
int take_time_step_ab( double t, arma::vec &ynp,
                       const cyclic_buffer<historic_vec> &last_fs,
                       double dt,
                       newton::status &stats,
                       const multistep::solver_coeffs &sc,
                       const multistep::solver_options &solver_opts,
                       functor_type &func )
{
	assert( false && "Adams-Bashforth not implemented!" );
	return 0;
}


template <typename functor_type> inline
int take_time_step_am( double t, arma::vec &ynp,
                       const cyclic_buffer<historic_vec> &last_fs,
                       double dt,
                       newton::status &stats,
                       const multistep::solver_coeffs &sc,
                       const multistep::solver_options &solver_opts,
                       functor_type &func )
{
	assert( false && "Adams-Moulton not implemented!" );
	return 0;
}




template <typename functor_type> inline
int take_time_step_bdf( double t, arma::vec &ynp,
                        const cyclic_buffer<historic_vec> &last_ys,
                        double dt,
                        newton::status &stats,
                        const multistep::solver_coeffs &sc,
                        const multistep::solver_options &solver_opts,
                        functor_type &func )
{
	arma::vec history;
	std::size_t N = last_ys[0].second.size();
	history.zeros( N );
	typename functor_type::jac_type I( N, N );
	I.eye(N,N);

	int ns = last_ys.size();
	int current_order = ( ns < sc.order ) ? ns : sc.order;

	const auto &cs = sc.cs_bdf.row(current_order-1);
	double bb = sc.b(current_order-1);

	bool print_formula = false;

	if( print_formula ){
		std::cerr << "Order is " << current_order << "\n";
		std::cerr << "Formula: y(n+1) ";
	}
	for( std::size_t i = 0; i < last_ys.size(); ++i ){
		history += cs[i] * last_ys[i].second;

		if( print_formula ) {
			if( i == 0 ){
				std::cerr << " + " << cs[i] << "*y(n)";
			}else{
				std::cerr << " + " << cs[i] << "*y(n-" << i << ")";
			}
		}

	}
	if( print_formula ) std::cerr << " = " << bb << " * dt * f(t(n+1), y(n+1)).\n";

	/*
	std::cerr << "Times of the history:";
	for( std::size_t i = 0; i < last_ys.size(); ++i ){
		std::cerr << " " << last_ys[i].first;
	}
	std::cerr << "\n";
	*/

	// Now, for BDF we solve y_n+1 + history = dt*f(t_(n+1), y_{n+1}) or
	// bb*dt*f(t_{n+1}, y_{n+1}) - y_{n+1} - history = 0.
	auto system_fun = [&t, &dt, &bb, &history, &func]( const arma::vec &y ) {
		return bb*dt * func.fun( t + dt, y ) - history - y; };

	auto system_jac = [&t, &dt, &bb, &history, &func, &I]( const arma::vec &y ) {
		return bb*dt * func.jac( t + dt, y ) - I; };

	newton_wrapper<decltype(system_fun), decltype(system_jac),
	               typename functor_type::jac_type>
		nw( system_fun, system_jac );

	arma::vec root = newton::broyden_iterate( nw, ynp,
	                                         *solver_opts.newton_opts,
	                                         stats );

	if( stats.conv_status == newton::SUCCESS ){
		ynp = root;
		return 0;
	}else{
		std::cerr << "Internal solver failed to converge!\n";
		return -1;
	}
}


template <typename functor_type> inline
int odeint( double t0, double t1,
            const solver_coeffs &sc,
            const solver_options &solver_opts,
            const arma::vec &y0,
            functor_type &func,
            std::vector<double> &t_vals,
            std::vector<arma::vec> &y_vals )
{
	double t = t0;
	int step = 0;

	assert( solver_opts.newton_opts && "Newton options not set!" );

	assert( sc.dt > 0 && "Cannot use time step size <= 0!" );
	assert( (sc.method_type == ADAMS_BASHFORTH ||
	         sc.method_type == ADAMS_MOULTON   ||
	         sc.method_type == BDF) && "Unrecognized method!" );

	double dt = sc.dt;

	arma::vec y = y0;
	std::size_t N = y0.size();
	arma::mat I(N, N);
	I.eye(N,N);

	newton::status stats;


	cyclic_buffer<historic_vec> last_ys( sc.order );
	cyclic_buffer<historic_vec> last_fs( sc.order );

	y_vals.push_back(y);
	t_vals.push_back(t);

	last_ys.push_back( std::make_pair( t, y ) );
	last_fs.push_back( std::make_pair( t, func.fun( t, y ) ) );

	// If your method is a multistep one then you want a high-accuracy
	// approx to the first order - 1 stages. We use recursion here:

	// TODO: Approximate initial last_ys and last_ts!
	//       Add option of passing them in somehow!

	while( t < t1 ){
		arma::vec yn(y);
		int status = 0;
		switch( sc.method_type ){
			case ADAMS_BASHFORTH:
				status = take_time_step_ab( t, yn, last_fs, dt,
				                            stats, sc,
				                            solver_opts, func );

				break;
			case ADAMS_MOULTON:
				status = take_time_step_am( t, yn, last_fs, dt,
				                            stats, sc,
				                            solver_opts, func );
				break;
			case BDF:
				status = take_time_step_bdf( t, yn, last_ys, dt,
				                             stats, sc,
				                             solver_opts, func );
				break;
		}

		if( status ){
			std::cerr << "Got status " << status << "!\n";
			return status;
		}else{
			y = yn;
			t += dt;
			++step;

			last_ys.push_back( std::make_pair(t, y) );
			last_fs.push_back( std::make_pair(t, func.fun(t, y)) );


			if( step % solver_opts.store_in_vector_every == 0 ){
				y_vals.push_back(y);
				t_vals.push_back(t);
			}
		}
	}

	return 0;
}


} // namespace multistep


#endif // MULTISTEP_HPP
