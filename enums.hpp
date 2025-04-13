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
   \file enums.hpp

   \brief some enums that should be exposed publicly.
*/

#ifndef ENUMS_HPP
#define ENUMS_HPP


#define FOREACH_MULTISTEP_METHOD(METHOD) \
	METHOD(ADAMS_BASHFORTH, 10) \
	METHOD(ADAMS_MOULTON,11)    \
	METHOD(BDF, 20)             \


// Some macro magic for enum to string and back.
#define FOREACH_IRK_METHOD(METHOD)        \
	METHOD(IMPLICIT_EULER,200)        \
	                                  \
	METHOD(RADAU_IIA_32,  210)        \
	METHOD(RADAU_IIA_53,  211)        \
	METHOD(RADAU_IIA_95,  212)        \
	METHOD(RADAU_IIA_137, 213)        \
	                                  \
	METHOD(LOBATTO_IIIA_43,  220)     \
	METHOD(LOBATTO_IIIA_85,  221)     \
	METHOD(LOBATTO_IIIA_127, 222)     \
	                                  \
	METHOD(LOBATTO_IIIC_43,  230)     \
	METHOD(LOBATTO_IIIC_64,  231)     \
	METHOD(LOBATTO_IIIC_85,  232)     \
	METHOD(LOBATTO_IIIC_127, 233)     \
	                                  \
	METHOD(GAUSS_LEGENDRE_42,  240)   \
	METHOD(GAUSS_LEGENDRE_63,  241)   \
	METHOD(GAUSS_LEGENDRE_105, 242)   \
	METHOD(GAUSS_LEGENDRE_147, 243)


#define FOREACH_ERK_METHOD(METHOD)        \
	METHOD(EXPLICIT_EULER, 100)       \
	METHOD(RUNGE_KUTTA_4,  110)       \
	                                  \
	METHOD(BOGACKI_SHAMPINE_32,120)	  \
	                                  \
	METHOD(CASH_KARP_54,130)          \
	METHOD(DORMAND_PRINCE_54,131)     \
	METHOD(FEHLBERG_54,132)



#define FOREACH_ROSENBROCK_METHOD(METHOD)


#define GENERATE_ENUM(ENUM, VAL) ENUM = VAL,
#define GENERATE_STRING(STRING, VAL) {VAL,#STRING},
#define GENERATE_MAP(STRING, VAL) {#STRING, VAL},


namespace irk {

/// \brief enumerates all implemented implicit RK methods.
enum rk_methods {
	FOREACH_IRK_METHOD(GENERATE_ENUM)

};

} // namespace irk

/// \brief enumerates all implemented explicit RK methods.
namespace erk {

enum rk_methods {
	FOREACH_ERK_METHOD(GENERATE_ENUM)
};

} // namespace erk



/// \brief enumerates possible return codes.
enum odeint_status_codes {
	SUCCESS = 0, ///< Everything is A-OK.

	/// The error estimate is too small, so attempt a larger step
	DT_TOO_SMALL  =  1,
	/// The internal solver used very few iterations
	INTERNAL_SOLVE_FEW_ITERS = 2,

	GENERAL_ERROR = 4, ///< A generic error

	/// The error estimate became too large, so attempt a smaller step size
	DT_TOO_LARGE  =  8,

	/// Internal solver failed to calculate stages
	INTERNAL_SOLVE_FAILURE = 16,

	/// The error exceeded the relative tolerance
	ERROR_LARGER_THAN_RELTOL = 32,

	/// The error exceeded the absolute tolerance
	ERROR_LARGER_THAN_ABSTOL = 64,

	ERROR_MAX_STEPS_EXCEEDED = 128

};

static constexpr const double machine_precision = 1e-17;

#endif // ENUMS_HPP
