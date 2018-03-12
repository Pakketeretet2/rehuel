/*
   Rehuel: a simple C++ library for solving ODEs


   Copyright 2017, Stefan Paquay (stefanpaquay@gmail.com)

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
   \brief This file contains some structs that allow for a generalized
   way of outputting the numerical solutions from various kinds of integrators.
*/


#ifndef INTEGRATOR_OUTPUT_HPP
#define INTEGRATOR_OUTPUT_HPP

#define ARMA_USE_CXX11

#include <armadillo>
#include <cassert>
#include <iosfwd>
#include <limits>

#include "cyclic_buffer.hpp"
#include "interpolate.hpp"

namespace integrator_io {


struct spaced_grid_info
{
	double t0, t1, dt;
};


struct given_time_values_info
{
	std::vector<double> output_times;
};


struct vector_output
{
	std::vector<double> t_vals;
	std::vector<arma::vec> y_vals;
};


/**
   \brief this class provides a generalized, flexible interface
   to manage how the various integrator types will output the solutions.
*/
class integrator_output {

private:

	/// Stores the particular way output is written
	int output_mode;

	/// Keeps track of when the last write was.
	double last_write;

	/// Keeps track of whether or not something was written at all.
	bool have_written;

	/// Contains the last N solution times.
	cyclic_buffer<double>    sol_times;

	/// Contains the last N solution values.
	cyclic_buffer<arma::vec> sol_values;

	/// Contains the spaced grid info:
	spaced_grid_info *spaced_grid;

	/// Contains the output grid info:
	given_time_values_info *given_time_values;

	/// Stream to write solution to.
	std::ostream *solution_out;

	/// Output time step size and error estimate every this many steps.
	int timestep_out_interval;
	/// Stream to dump the time step size and error info in.
	std::ostream *timestep_out;


	/// Output into vector every this many steps:
	int vector_out_interval;

	/// Pointer to struct to store the vector info into:
	vector_output *vec_out;



	/// Checks the validity of some settings.
	bool check()
	{
		bool is_valid = true;

		return is_valid;
	}

public:
	/// Enumerates the possible output methods
	enum output_modes {
		NO_OUTPUT = 0,
		SPACED_GRID = 1,
		GIVEN_TIME_VALUES = 2
	};


	/// Constructor sets most settings to either defaults or user input
	integrator_output( int output_mode = NO_OUTPUT,
	                   std::ostream *solution_out = nullptr,
	                   int timestep_out_interval = 0,
	                   std::ostream *timestep_out = nullptr )
		: output_mode( output_mode ),
		  last_write(-100),
		  have_written( false ),
		  spaced_grid( nullptr ),
		  given_time_values( nullptr ),
		  solution_out( solution_out ),
		  timestep_out_interval( timestep_out_interval ),
		  timestep_out( timestep_out ),
		  vector_out_interval( 0 ),
		  vec_out( nullptr )

	{
		check();

		if( output_mode & SPACED_GRID ){
			spaced_grid = new spaced_grid_info;
			assert( spaced_grid &&
			        "Failed to allocate spaced grid!" );
		}
		if( output_mode & GIVEN_TIME_VALUES ){
			given_time_values = new given_time_values_info;
			assert( given_time_values &&
			        "Failed to allocate given time value struct!" );
		}
	}


	/// Destructor takes care of heap-allocated output info.
	~integrator_output()
	{
		if( spaced_grid )       delete spaced_grid;
		if( given_time_values ) delete given_time_values;
	}


	/**
	  \brief Sets the internal spacial grid.

	  \returns 0 on success, non-zero otherwise
	*/
	int set_spaced_grid( double t0, double t1, double dt )
	{
		if( output_mode != SPACED_GRID ){
			std::cerr << "WARNING: Cannot set spaced grid for "
			          << "non-spaced-grid output!\n";
			return -1;
		}

		spaced_grid->t0 = t0;
		spaced_grid->t1 = t1;
		spaced_grid->dt = dt;

		return 0;
	}


	/**
	   \brief Sets the timed values struct and checks validity.

	   \returns 0 on success, non-zero otherwise.
	*/
	int set_output_time_values( const std::vector<double> &output_times )
	{
		if( output_mode != GIVEN_TIME_VALUES ){
			std::cerr << "WARNING: Cannot set output time values "
			          << "for non-given-time-values output!\n";
			return -1;
		}

		if( !std::is_sorted(output_times.begin(), output_times.end())){
			std::cerr << "WARNING: Given output times are not "
			          << "sorted! Storing sorted copy!\n";
		}

		given_time_values->output_times = output_times;
		std::sort( given_time_values->output_times.begin(),
		           given_time_values->output_times.end() );

	}


	/**
	   \brief Adds the latest solution to the cyclic buffers:
	*/
	void add_solution( double t, const arma::vec &sol )
	{
		sol_times.push_back(t);
		sol_values.push_back(sol);
	}


	/**
	   \brief Writes solution info between last two times.
	*/
	void write_solution()
	{
		if( output_mode & SPACED_GRID ){
			write_solution_spaced_grid();
		}
	}


	/**
	   \brief Writes time step info and stuff.
	*/
	void write_timestep_info( unsigned long long int step, double t,
	                          double dt, double err, double old_err, int iters )
	{
		if( timestep_out && (step % timestep_out_interval == 0) ){
			std::ostream &out = *timestep_out;

			out << step << " " << t << " " << dt << " "
			    << err << " " << old_err << " " << iters << "\n";
		}
	}

	/**
	   \brief Writes only the time step and iterations.
	*/
	void write_timestep_info( unsigned long long int step,
	                          double t, double dt )
	{
		if( timestep_out && (step % timestep_out_interval == 0) ){
			std::ostream &out = *timestep_out;

			out << step << " " << t << " " << dt << "\n";
		}
	}



	/**

	 */
	void write_solution_spaced_grid()
	{
		double next_write = last_write + spaced_grid->dt;

		double t_now = sol_times[0];
		if( t_now > next_write ){
			*solution_out << t_now;
			const arma::vec &y = sol_values[0];
			for( std::size_t i = 0; i < y.size(); ++i ){
				*solution_out << " " << y[i];
			}
			*solution_out << "\n";
			last_write = sol_times[0];
		}
	}



	/**
	   \brief Returns the next time output is needed.

	   This function should be used for the first time steps
	   at which the full number of points has not yet been acquired.
	*/
	double next_output_time_step()
	{
		double next_time = std::numeric_limits<double>::max();

		if( output_mode & SPACED_GRID ){
			next_time = std::min( next_time,
			                      sol_times[0] + spaced_grid->dt );
		}
		if( output_mode & GIVEN_TIME_VALUES ){

		}
		return next_time;
	}


	/**
	   \brief Sets how many time points are stored and
	   hence the interpolation accuracy.
	*/
	void set_storage_size( std::size_t size )
	{
		sol_times.resize(size);
		sol_values.resize(size);
	}


	/**
	   \brief Sets the vector output.

	   \param out_interval  The interval with which to store data in vector.
	   \param vec_out       Pointer to the struct to store output in.

	   \returns 0 on success, non-zero otherwise.
	*/
	int set_vector_output( int out_interval, vector_output *vector_out )
	{
		if( out_interval <= 0 ){
			std::cerr << "WARNING: Vector out interval "
			          << "has to be positive! "
			          << "Ignoring set_vector_output call!\n";
			return -1;
		}
		if( !vector_out ){
			std::cerr << "WARNING: Vector output struct is null! "
			          << "Ignoring set_vector_output call!\n";
			return -2;
		}

		if( vec_out ){
			std::cerr << "WARNING: Output vector already set!\n";
		}

		vector_out_interval = out_interval;
		vec_out = vector_out;

		return 0;
	}

	/**
	   \brief Returns read-only ptr to vector output struct.
	*/
	const vector_output *get_vector_output() const
	{
		return vec_out;
	}

	/**
	   \brief This adds a time point to the vectors in vec_out.

	   \note the solution is only stored if the current step
	         matches vector_output_interval

	   \param step        Current simulation time step
	   \param time        Current simulation time
	   \param y           Current simulation solution
	   \param force_add   If true, add solution regardless of step.

	*/
	void store_vector_solution( int step, double time, const arma::vec &y,
	                            bool force_add = false)
	{
		if( vec_out ){
			if( (step % vector_out_interval == 0)
			    || force_add ) {
				vec_out->t_vals.push_back(time);
				vec_out->y_vals.push_back(y);
			}
		}
	}

	/**
	   \brief Sets the timestep out info.

	   \param out_interval         Output interval
	   \param timestep_out_stream  Output stream for time step.
	*/
	int set_timestep_output( int out_interval,
	                         std::ostream *timestep_out_stream )
	{
		timestep_out_interval = out_interval;
		if( timestep_out ){
			std::cerr << "WARNING: Output stream for time "
			          << "step already set!\n";
		}
		timestep_out = timestep_out_stream;

		return 0;
	}

};

} // namespace output


#endif // INTEGRATOR_OUTPUT_HPP
