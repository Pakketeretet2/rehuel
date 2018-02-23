#ifndef OPTIONS_HPP
#define OPTIONS_HPP

/**
   \file options.hpp
   Contains common solver options.
*/

#include <iosfwd>

#include "integrator_io.hpp"

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
		  newton_opts( nullptr ),
		  output( nullptr ),
		  verbosity( 0 ),
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

	/// Options for the internal solver.
	const newton::options *newton_opts;

	/// To output solutions.
	integrator_io::integrator_output *output;

	/// if > 0, print some output.
	int verbosity;

	/// If true, abort if the internal solver failed.
	bool abort_on_solver_fail;

};



#endif // OPTIONS_HPP
