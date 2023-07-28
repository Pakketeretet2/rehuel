/*
   Rehuel: a simple C++ library for solving ODEs


   Copyright 2017-2019, Stefan Paquay (stefanpaquay@gmail.com)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

============================================================================= */

/**
   \file options.hpp
   Contains common solver options.
*/


#ifndef OPTIONS_HPP
#define OPTIONS_HPP

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
		: internal_solver(NEWTON),
		  rel_tol(1e-4),
		  abs_tol(10*rel_tol),
		  max_dt(0.0),
		  max_steps(-1),
		  newton_opts(nullptr),
		  out_interval(0),
		  time_internals(false)
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

	/// Maximum number of time steps to take (negative to allow infinite).
	long long int max_steps;

	/// Options for the internal solver.
	const newton::options *newton_opts;

	/// Output interval for error and step:
	int out_interval;

	/// Keep track of the timings of various parts in solver?
	bool time_internals;
};


/**
   \brief Specifies how the user wants output. Default is
   for the solver to populate the t_vals and y_vals vectors in a basic_output
   struct.
*/
struct output_options {

	output_options(){}
	enum output_bits
	{
		NO_OUTPUT = 0,
		STORE_IN_VECTORS = 1,
		WRITE_TO_FILE = 2
	};

	std::size_t output_mode = STORE_IN_VECTORS;
	std::size_t output_interval = 1;
	std::ostream &log_out = std::cout;
	std::ostream *output_stream;

	bool store_in_vectors() const
	{
		return output_mode & (1 << (STORE_IN_VECTORS - 1));
	}
	bool write_to_file() const
	{
		return output_mode & (1 << (WRITE_TO_FILE - 1));
	}

	void set_output_stream(std::ostream &out_stream)
	{
		output_stream = &out_stream;
		output_mode |= (1 << (WRITE_TO_FILE - 1));
	}


	void enable_store_in_vectors()
	{
		output_mode |= (1 << (STORE_IN_VECTORS - 1));
	}
	void disable_store_in_vectors()
	{
		output_mode &= ~(1 << (STORE_IN_VECTORS - 1));
	}

};






#endif // OPTIONS_HPP
