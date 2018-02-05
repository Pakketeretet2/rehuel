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

#include <iosfwd>

namespace integrator_io {

struct integrator_output {

	integrator_output( int sol_out_int, int step_out_int, int store_vec_int,
	                   std::ostream *sol_out, std::ostream *step_out )
		: solution_out_interval( sol_out_int ),
		  timestep_out_interval( step_out_int ),
		  store_in_vector_every( store_vec_int ),
		  solution_out( sol_out ),
		  timestep_out( step_out ) {}

	int solution_out_interval;
	int timestep_out_interval;
	int store_in_vector_every;

	std::ostream *solution_out;
	std::ostream *timestep_out;



};

} // namespace output


#endif // INTEGRATOR_OUTPUT_HPP
