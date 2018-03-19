#ifndef FUNCTOR_HPP
#define FUNCTOR_HPP

/**
   \file functor.hpp Constrains definitions for functors that describe ODEs.
*/


#include <armadillo>

/**
   \brief This class describes how a functor that describes an
   ODE is to look like.
*/
class functor {
public:
	typedef arma::mat jac_type;

	virtual arma::vec fun( double t, const arma::vec &y ) = 0;
	virtual jac_type jac( double t, const arma::vec &y ) = 0;
};


/**
   \brief A similar functor but for the case of a sparse Jacobian matrix.
*/
class functor_sparse_jac {
public:
	typedef arma::sp_mat jac_type;
	virtual arma::vec fun( double t, const arma::vec &y ) = 0;
	virtual jac_type jac( double t, const arma::vec &y ) = 0;
};

#endif // FUNCTOR_HPP
