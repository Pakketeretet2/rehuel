#include <iomanip>
#include <iostream>
#include <fstream>

#include "../rehuel.hpp"
#include "../newton.hpp"
#include "../test_equations.hpp"


#include "clara.hpp"




int main( int argc, char **argv )
{

	bool help = false;

	double w = 0.2;
	test_equations::harmonic func( w );
	double t1 = 50000.0 * w;

	std::string method = "RADAU_IIA_32";

	auto cli = clara::Help( help )["-h"]["--help"]
		( "Prints help." )
		| clara::Opt( t1, "t1" )["--final-time"]
		( "The time to stop the integration." )
		| clara::Opt( method, "method" )["-m"]["--method"]
		( "The method to use." );

	auto args = clara::Args( argc, argv );
	auto result = cli.parse( args );
	if( !result ){
		std::cerr << "Error in command line: "
		          << result.errorMessage() << "\n";
	}

	if( help ){
		std::cerr << cli << "\n";
		return 0;
	}

	int m = irk::name_to_method( method );
	std::vector<double> dts = { 5e-3,
	                            1e-2, 2e-2, 5e-2,
	                            1e-1, 2e-1, 5e-1,
	                            1e0,  2e0,  5e0,
	                            1e1,  2e1,  5e1 };
	auto so = irk::default_solver_options();
	auto sc = irk::get_coefficients( m );
	newton::options no;
	no.maxit = 100000;
	so.adaptive_step_size = false;
	so.newton_opts = &no;
	so.out_interval = 1000000;

	std::string fname = "sol_";
	fname += method;
	fname += ".dat";
	std::ofstream sol_out( fname );

	for( double dt : dts ){
		bool output = false;
		if( std::fabs( 2e-1 - dt ) < 1e-12 ) output = true;

		arma::vec y0 = { 0.0, 1.0 };
		irk::rk_output sol = irk::odeint( func, 0, t1, y0, so, m, dt );
		double max_rel_err = 0.0;
		double max_abs_err = 0.0;
		if( output ){
			sol_out << std::setprecision(16) << std::setw(20);
			sol_out << 0.0 << " " << y0[0] << " " << y0[1] << "\n";
		}
		for( std::size_t i = 0; i < sol.t_vals.size(); ++i ){
			double ti = sol.t_vals[i];
			arma::vec yi = sol.y_vals[i];

			double yreal = func.sol( ti )[0];
			double ynumer = yi[0];

			double abs_err = yreal - ynumer;
			double rel_err = abs_err / yreal;

			max_rel_err = std::max( rel_err, max_rel_err );
			max_abs_err = std::max( abs_err, max_abs_err );

			if( output ){
				sol_out << ti << " " << yi[0] << " "
				        << yi[1] << "\n";
			}

		}
		std::cout << dt << " " << max_rel_err << " " << max_abs_err << "\n";
	}



	return 0;
}
