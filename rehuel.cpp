#include "rehuel.h"

#include "irk.hpp"
#include "newton.hpp"

// This file contains implementation details of the C interface.

struct rehuel_handle_guts {
	irk::solver_coeffs  *irk_coeffs;
	irk::solver_options *irk_opts;
	newton::options     *newton_opts;

};


// Init and finalize:
rehuel_handle rehuel_initialize()
{
	rehuel_handle rh = new rehuel_handle_guts;
	// The solver always needs a newton::options.
	// The rest can be null until needed.
	rh->irk_coeffs = nullptr;
	rh->irk_opts = nullptr;
	rh->newton_opts = new newton::options;


	return rh;
}

void rehuel_finalize( rehuel_handle rh )
{
	if( rh->newton_opts ) delete rh->newton_opts;
	if( rh->irk_coeffs  ) delete rh->irk_coeffs;
	if( rh->irk_opts  )   delete rh->irk_opts;

	delete rh;
}



// Setings:
// Newton:
void rehuel_newton_set_tol( rehuel_handle rh, double tol )
{
	rh->newton_opts->tol = tol;
}

void rehuel_newton_set_maxit( rehuel_handle rh, int maxit )
{
	rh->newton_opts->maxit = maxit;
}

void rehuel_newton_set_time_internals( rehuel_handle rh, int enable )
{
	rh->newton_opts->time_internals = enable;
}

void rehuel_newton_set_max_step( rehuel_handle rh, double max_step )
{
	rh->newton_opts->max_step = max_step;
}

void rehuel_newton_set_refresh_jacobi_matrix( rehuel_handle rh, int refresh )
{
	rh->newton_opts->refresh_jac = refresh;
}


// IRK:
void rehuel_irk_set_solver_coefficients( rehuel_handle rh, int method )
{
	if( !rh->irk_coeffs ){
		rh->irk_coeffs = new irk::solver_coeffs;
	}
	*rh->irk_coeffs = irk::get_coefficients( method );
}

void rehuel_irk_set_internal_solver( rehuel_handle rh, int internal_solver )
{
	if( !rh->irk_opts ){
		rh->irk_opts = new irk::solver_options;
		rh->irk_opts->newton_opts = rh->newton_opts;
		*rh->irk_opts = irk::default_solver_options();
	}
	rh->irk_opts->internal_solver = internal_solver;
}

void rehuel_irk_set_adaptive_time_step( rehuel_handle rh, int adaptive_dt )
{
	if( !rh->irk_opts ){
		rh->irk_opts = new irk::solver_options;
		rh->irk_opts->newton_opts = rh->newton_opts;
		*rh->irk_opts = irk::default_solver_options();
	}
	rh->irk_opts->adaptive_step_size = adaptive_dt;
}

void rehuel_irk_set_relative_tolerance( rehuel_handle rh, double rel_tol )
{
	if( !rh->irk_opts ){
		rh->irk_opts = new irk::solver_options;
		rh->irk_opts->newton_opts = rh->newton_opts;
		*rh->irk_opts = irk::default_solver_options();
	}
	rh->irk_opts->rel_tol = rel_tol;
}

void rehuel_irk_set_absolute_tolerance( rehuel_handle rh, double abs_tol )
{
	if( !rh->irk_opts ){
		rh->irk_opts = new irk::solver_options;
		rh->irk_opts->newton_opts = rh->newton_opts;
		*rh->irk_opts = irk::default_solver_options();
	}
	rh->irk_opts->abs_tol = abs_tol;
}

void rehuel_irk_set_maximum_time_step( rehuel_handle rh, double max_dt )
{
	if( !rh->irk_opts ){
		rh->irk_opts = new irk::solver_options;
		rh->irk_opts->newton_opts = rh->newton_opts;
		*rh->irk_opts = irk::default_solver_options();
	}
}

void rehuel_irk_set_output_interval( rehuel_handle rh, int output_interval )
{
	if( !rh->irk_opts ){
		rh->irk_opts = new irk::solver_options;
		rh->irk_opts->newton_opts = rh->newton_opts;
		*rh->irk_opts = irk::default_solver_options();
	}
	rh->irk_opts->out_int = output_interval;
}
