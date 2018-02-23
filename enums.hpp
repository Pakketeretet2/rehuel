#ifndef ENUMS_HPP
#define ENUMS_HPP

/**
   \file enums.hpp

   \brief some enums that should be exposed publicly.
*/


#define FOREACH_MULTISTEP_METHOD(METHOD) \
	METHOD(ADAMS_BASHFORTH, 10) \
	METHOD(ADAMS_MOULTON,11)    \
	METHOD(BDF, 20)             \


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
	METHOD(RADAU_IIA_32,  202)	  \
	METHOD(RADAU_IA_53,   203)        \
	METHOD(RADAU_IIA_53,  204)        \
	                                  \
	METHOD(IMPLICIT_MIDPOINT, 210)    \
	METHOD(GAUSS_LEGENDRE_42, 211)	  \
	METHOD(GAUSS_LEGENDRE_62, 212)    \
	                                  \
	METHOD(LOBATTO_IIIA_21, 220)	  \
	METHOD(LOBATTO_IIIC_21, 221)	  \
	METHOD(LOBATTO_IIIA_42, 222)	  \
	METHOD(LOBATTO_IIIC_42, 223)	  \
	METHOD(LOBATTO_IIIC_63, 224)      \

//	METHOD(SDIRK_L_43, 230)



#define FOREACH_ROSENBROCK_METHOD(METHOD)


#define GENERATE_ENUM(ENUM, VAL) ENUM = VAL,
#define GENERATE_STRING(STRING, VAL) {VAL,#STRING},
#define GENERATE_MAP(STRING, VAL) {#STRING, VAL},


namespace multistep {

/// \brief enumerates all implemented multistep methods.
enum ms_methods {
	FOREACH_MULTISTEP_METHOD(GENERATE_ENUM)
};

/// \brief testing doxygen.
enum poop{
	WHUT
};

} // namespace multistep


namespace irk {

/// \brief enumerates all implemented RK methods.
enum rk_methods {
	FOREACH_RK_METHOD(GENERATE_ENUM)

};


} // namespace irk


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
	ERROR_LARGER_THAN_ABSTOL = 64
};


#endif // ENUMS_HPP
