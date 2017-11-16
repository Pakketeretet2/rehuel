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
	RADAU_IIA_32        = 21,
	LOBATTO_IIIA_43     = 22,
	GAUSS_LEGENDRE_43   = 23
};

/// \brief enumerates possible return codes.
enum odeint_status_codes {
	SUCCESS = 0, ///< Everything is A-OK.

	/// The error estimate became too large, so attempt a smaller step size
	DT_TOO_LARGE  =  1,
	/// The error estimate is too small, so attempt a larger step
	DT_TOO_SMALL  =  2,

	GENERAL_ERROR = -1, ///< A generic error

	/// Internal solver failed to calculate stages
	INTERNAL_SOLVE_FAILURE = -3,
	/// Time step is unacceptably small, problem is likely stiff.
	TIME_STEP_TOO_SMALL = -4
};

} // namespace irk

#endif // ENUMS_HPP
