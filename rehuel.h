#ifndef REHUEL_H
#define REHUEL_H

/**
   \file rehuel.h

   \brief This is a header for a C-style interface to the Rehuel library.
*/
#include "rehuel.hpp"

#ifdef __cplusplus
extern "C" {
#endif

/// The standard form of the RHS of an ODE for the C interface
typedef void (*ode_rhs)(double, int, const double *, double *);

/// The standard form of the Jacobi matrix of an ODE for the C interface
typedef void (*ode_jac)(double, int, const double *, double **);


/**
   A struct with pointers to solver_coeffs and solver_options.
*/
typedef struct  {
	rehuel::solver_coeffs *coeffs;
	rehuel::solver_options *options;
} rehuel_handle;


/// \brief Initialize the rehuel interface
rehuel_handle *rehuel_initialize();
/// \brief Finalize the rehuel interface.
void rehuel_finalize( rehuel_handle * );



#ifdef __cplusplus
} /* extern "C" */
#endif


#endif // REHUEL_H
