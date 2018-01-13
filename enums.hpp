#ifndef ENUMS_HPP
#define ENUMS_HPP

/**
   \file enums.hpp

   \brief some enums that should be exposed publicly.
*/

// Some macro magic for enum to string and back.
#define FOREACH_RK_METHOD(METHOD)         \
	METHOD(EXPLICIT_EULER, 100)       \
	METHOD(RUNGE_KUTTA_4,101)         \
	METHOD(BOGACKI_SHAMPINE_32,102)	  \
	METHOD(CASH_KARP_54,103)          \
	METHOD(DORMAND_PRINCE_54,104)     \
	METHOD(FEHLBERG_54,105)           \
	                                  \
	METHOD(IMPLICIT_EULER,200)        \
	METHOD(IMPLICIT_MIDPOINT,201)     \
	                                  \
	METHOD(RADAU_IA_32,202)           \
	METHOD(RADAU_IIA_32,203)	  \
	METHOD(LOBATTO_IIIA_21,204)	  \
	METHOD(LOBATTO_IIIC_21,205)	  \
					  \
	METHOD(LOBATTO_IIIA_43,206)	  \
	METHOD(LOBATTO_IIIC_43,207)	  \
					  \
	METHOD(GAUSS_LEGENDRE_42,208)	  \
	METHOD(RADAU_IA_54,209)		  \
	METHOD(RADAU_IIA_54,210)          \
					  \
	METHOD(GAUSS_LEGENDRE_63, 211)    \
	METHOD(LOBATTO_IIIA_65,   212)    \
	METHOD(LOBATTO_IIIC_65,   213)    \


#define GENERATE_ENUM(ENUM, VAL) ENUM = VAL,
#define GENERATE_STRING(STRING, VAL) {VAL,#STRING},
#define GENERATE_MAP(STRING, VAL) {#STRING, VAL},


namespace irk {


/// \brief enumerates all implemented RK methods.
enum rk_methods {
	FOREACH_RK_METHOD(GENERATE_ENUM)
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
	TIME_STEP_TOO_SMALL = 32,

	/// The error exceeded the relative tolerance
	ERROR_LARGER_THAN_RELTOL = 64,

	/// The error exceeded the absolute tolerance
	ERROR_LARGER_THAN_ABSTOL = 128


};

} // namespace irk

#endif // ENUMS_HPP
