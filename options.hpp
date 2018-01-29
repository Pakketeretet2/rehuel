#ifndef OPTIONS_HPP
#define OPTIONS_HPP

/**
   \file options.hpp
   Contains common solver options.
*/

#include <iosfwd>


namespace newton {
struct options;
} // namespace newton

/**
   \brief struct for common solver options.
*/
struct common_solver_options
{
	/// \brief Enumerates the possible internal non-linear solvers
	enum internal_solvers {
		BROYDEN = 0, ///< Broyden's method
		NEWTON = 1   ///< Newton's method
	};

	/// \brief Constructor with default values.
	common_solver_options()
		: internal_solver(BROYDEN),
		  rel_tol(1e-5),
		  abs_tol(10*rel_tol),
		  max_dt( 100.0 ),
		  solution_out_interval( 1000 ),
		  timestep_info_out_interval( 1000 ),
		  newton_opts( nullptr ),
		  store_in_vector_every( 1 ),
		  solution_out( nullptr ),
		  timestep_out( nullptr ),
		  verbosity( 0 ),
		  constant_jac_approx( false ),
		  abort_on_solver_fail( false )
	{ }

	~common_solver_options()
	{ }

	/// Internal non-linear solver used (see \ref internal_solvers)
	/// Broyden typically gives good results in less time.
	int internal_solver;

	/// Relative tolerance to satisfy when adaptive time stepping
	double rel_tol;
	/// Absolute tolerance to satisfy when adaptive time stepping
	double abs_tol;
	/// Maximum time step size
	double max_dt;

	/// Output interval for solution:
	int solution_out_interval;

	/// Output interval for time step size and error.
	int timestep_info_out_interval;

	/// Options for the internal solver.
	const newton::options *newton_opts;

	/// Store solution in vector every this many steps (0 to disable)
	int store_in_vector_every;

	/// Write solution to this output stream.
	std::ostream *solution_out;

	/// Write time step info to this output stream.
	std::ostream *timestep_out;

	/// if > 0, print some output.
	int verbosity;

	/// If true, use a constant Jacobi matrix approximation
	bool constant_jac_approx;

	/// If true, abort if the internal solver failed.
	bool abort_on_solver_fail;

};



#endif // OPTIONS_HPP
