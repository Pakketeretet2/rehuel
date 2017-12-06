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

#ifndef REHUEL_H
#define REHUEL_H

/**
   \file This file contains a C interface to the rehuel library.
*/

#ifdef __cplusplus
extern "C" {
#endif // CPP


/* Opaque pointer: */
struct rehuel_handle_guts;

typedef struct rehuel_handle_guts * rehuel_handle;

/* Initialization and finalization: */
rehuel_handle rehuel_initialize();
void rehuel_finalize( rehuel_handle );

/* Setting coefficients: */
/* Newton */
void rehuel_newton_set_tol( rehuel_handle, double );
void rehuel_newton_set_maxit( rehuel_handle, int );
void rehuel_newton_set_time_internals( rehuel_handle, int );
void rehuel_newton_set_max_step( rehuel_handle, double );
void rehuel_newton_set_refresh_jacobi_matrix( rehuel_handle, int );

/* IRK */
void rehuel_irk_set_solver_coefficients( rehuel_handle, int );
void rehuel_irk_set_internal_solver( rehuel_handle, int );
void rehuel_irk_set_adaptive_time_step( rehuel_handle, int );
void rehuel_irk_set_relative_tolerance( rehuel_handle, double );
void rehuel_irk_set_absolute_tolerance( rehuel_handle, double );
void rehuel_irk_set_maximum_time_step( rehuel_handle, double );
void rehuel_irk_set_output_interval( rehuel_handle, int );




/* Multistep: */





#ifdef __cplusplus
} /* extern "C" */
#endif /* CPP */



#endif /* REHUEL_H */
