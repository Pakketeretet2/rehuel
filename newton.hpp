// Functions for performing Newton iteration.

#include <armadillo>

namespace newton {

/// \brief Return codes for newton_solve.
enum newton_solve_ret_codes {
	SUCCESS = 0,
	NOT_CONVERGED = 1
};

/**
   \brief Solves F(x) = 0 with x0 as initial guess.

   \param F     Function to find root of
   \param J     Jacobi matrix of function
   \param x     Initial guess, will contain the solution.
   \param tol   Tolerance
   \param maxit Maximum number of iterations.

   \returns A status code.
*/
template <typename func_rhs, typename func_Jac >
arma::vec newton_solve( const func_rhs &F, const func_Jac &J, arma::vec x,
                        double tol, int maxit, int &status,
                        double &res, int &iters )
{
	status = SUCCESS;
	double tol2 = tol*tol;
	arma::vec r = F(x);
	double res2 = arma::dot( r, r );
	iters = 0;

	while( res2 > tol2 && iters < maxit ){
		arma::mat Jac = J(x);
		// x_{k+1} = x_k - J^{-1}r
		x -= arma::solve( Jac, r );
		++iters;
		r = F(x);
		res2 = dot(r,r);
	}
	if( iters == maxit && res2 > tol2 ){
		status = NOT_CONVERGED;
	}else{
		status = SUCCESS;
	}
	res = std::sqrt( res2 );
	return x;
}

} // namespace newton
