#include <iostream>

#include "rehuel.hpp"
#include "test_equations.hpp"

// Tests adaptive time step on various problems.


int main( int argc, char **argv )
{

	test_equations::stiff_eq stiff;

	irk::solver_options so = irk::default_solver_options();
	arma::vec y0 = { 1.0, 0.0 };

	double t0 = 0.0;
	double t1 = 10.0;
	double dt = 1e-12;

	newton::options newt_opts;

	so.internal_solver = irk::solver_options::NEWTON;
	so.rel_tol = 1e-6;
	so.abs_tol = 1e-5;
	newt_opts.tol = 0.1*so.rel_tol;
	newt_opts.maxit = 10000;
	so.newton_opts = &newt_opts;
	so.adaptive_step_size = true;

	std::vector<int> methods = { irk::RADAU_IIA_53 };


	for( int method : methods ){

		irk::solver_coeffs  sc = irk::get_coefficients( method );
		integrator_io::vector_output vec_out;
		integrator_io::integrator_output output;
		int interval = 10;
		output.set_vector_output( interval, &vec_out );
		output.set_timestep_output( 10000, &std::cerr );
		so.output = &output;
		so.verbosity = 0;

		sc.dt = dt;
		int status = irk::odeint( t0, t1, sc, so, y0, stiff );
		if( status ){
			std::cerr << "Error solving ODE with method "
			          << irk::method_to_name( method ) << "!\n";
			continue;
		}

		std::string fname = "stiff_sol_";
		fname += irk::method_to_name(method);
		fname += ".dat";
		std::ofstream out( fname );

		for( std::size_t i = 0; i < vec_out.t_vals.size(); ++i ){
			double ti = vec_out.t_vals[i];
			out << ti << " ";
			const arma::vec &yi = vec_out.y_vals[i];
			out << yi[0] << " " << yi[1] << " ";
			arma::vec ye = stiff.sol( ti );
			out << ye[0] << " " << ye[1];
			out << "\n";
		}
	}



	return 0;
}
