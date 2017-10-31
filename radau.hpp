// A library to solve ODEs

// Constructs the correct Jacobi matrix to solve
// F( t + ci*dt, y + dt*Sum_{j=0}^{N-1} (a_ij k_j ) ) - k_i

#include <armadillo>

namespace radau {

struct solver_coeffs
{
	arma::vec b, c;
	arma::mat A;
	double dt;
};

bool verify_solver_coeffs( const solver_coeffs &sc );

template <typename func_type, typename Jac_type> inline
void construct_F_and_J( arma::vec &F, arma::mat &J, double t,
                        const arma::vec &y,
                        const radau::solver_coeffs &sc,
                        const func_type &fun, const Jac_type &jac )
{
	auto Ns  = sc.b.size();
	auto Neq = y.size();
	auto NN = Ns*Neq;
	arma::vec K( Neq );

	const arma::vec &b = sc.b;
	const arma::vec &c = sc.c;
	const arma::mat &A = sc.A;
	double dt = sc.dt;

	J.eye( NN, NN );
	J *= -1.0;
	for( unsigned int i = 0; i < Ns; ++i ){
		double ti = t + dt * c[i];
		arma::vec yi = y;

		for( unsigned int j = 0; j < Ns; ++j ){
			unsigned int offset = j*Neq;
			yi += dt * A(i,j) * K.subvec( offset, offset + Neq );
		}
		arma::mat Ji = jac( ti, yi );
		arma::vec ki = K.subvec( i*Neq, i*Neq + Neq );
		F.subvec( i*Neq, i*Neq + Neq ) = fun( ti, yi ) - ki;
	}
	// Construct J.

	// i is column, j is row.
	for( unsigned int i = 0; i < Ns; ++i ){
		double ti = t + dt * c[i];
		arma::vec yi = y;

		for( unsigned int j = 0; j < Ns; ++j ){
			unsigned int offset = j*Neq;
			yi += dt * A(i,j) * K.subvec( offset, offset + Neq );
		}
		arma::mat Ji = jac( ti, yi );

		// j is row.
		for( unsigned int j = 0; j < Ns; ++j ){
			// Block i*Neq by j*Neq has to be filled with
			// d F(t + ci, y + dt*sum_{k=0}^N-1 (a_{i,k}*k_k)) / d k_j
			// which is
			// F(t + ci, y + sum_{k=0}^N-1 (a_{i,k}*k_k))' * a_{i,j}*dt
			auto Jc = J.submat( i*Neq, j*Neq, i*Neq + Neq,
			                    j*Neq + Neq );
			double iets = dt * A(i,j);
			Jc += Ji * iets;
		}
	}
}


} // namespace radau
