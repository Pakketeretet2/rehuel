#ifndef ENUMS_HPP
#define ENUMS_HPP

/**
   \file enums.hpp

   \brief some enums that should be exposed publicly.
*/

namespace irk {

/// \brief enumerates all implemented RK methods.
enum rk_methods {
	EXPLICIT_EULER      = 10,
        RUNGE_KUTTA_4       = 11,
	BOGACKI_SHAMPINE_23 = 12,
	CASH_KARP_54        = 13,
	DORMAND_PRINCE_54   = 14,

	IMPLICIT_EULER      = 20,
	IMPLICIT_MIDPOINT   = 21,
	RADAU_IIA_32        = 22,
	LOBATTO_IIIA_43     = 23,
	GAUSS_LEGENDRE_43   = 24
};

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
	/// Time step is unacceptably small, problem is likely stiff.
	TIME_STEP_TOO_SMALL = 32
};

} // namespace irk

#endif // ENUMS_HPP
