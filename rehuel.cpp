#include "rehuel.h"

// This C++ source file contains the C routines.


rehuel_handle *rehuel_initialize()
{
	rehuel_handle *rh = new rehuel_handle;
	rh->coeffs = new rehuel::solver_coeffs;
	rh->options = new rehuel::solver_options;

	return rh;
}


void rehuel_finalize( rehuel_handle *rh )
{
	delete rh->coeffs;
	delete rh->options;

	delete rh;
}


void rehuel_set_solver_coefficients( rehuel_handle *rh, int solver )
{
	*rh->coeffs = rehuel::get_solver_coefficients( solver );
}

int rehuel_verify_solver_coefficients( rehuel_handle *rh )
{
	return rehuel::verify_solver_coeffs( *rh->coeffs );
}

int rehuel_integrate_ode( rehuel_handle *rh, double t0, double t1, int size,
                          ode_rhs f, ode_jac J, int n_vals,
                          double *t_vals, double **y_vals )
{
	// This is tricky...
	int odeint_status = 0;

	return odeint_status;
}
