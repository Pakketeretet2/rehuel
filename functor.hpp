/*
   Rehuel: a simple C++ library for solving ODEs


   Copyright 2017-2019, Stefan Paquay (stefanpaquay@gmail.com)

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

/**
   \file functor.hpp Constrains definitions for functors that describe ODEs.
*/

#ifndef FUNCTOR_HPP
#define FUNCTOR_HPP


#include "arma_include.hpp"

/**
   \brief This class describes how a functor that describes an
   ODE is to look like.
*/
class functor {
public:
	/// Provide a typedef for the Jacobi matrix to allow for
	/// sparse vs. dense matrices.
	typedef arma::mat jac_type;

	/// Evaluates the RHS of the differential equation.
	virtual arma::vec fun( double t, const vec_type &y ) = 0;
	/// Evaluates the Jacobi matrix of the ODE RHS.
	virtual jac_type jac( double t, const vec_type &y ) = 0;

	/**
	   \brief calculates the ordinary differential equation's
	   RHS and Jacobi matrix at the same time. Can be overridden for
	   efficiency. The default implementation just calls jac and fun.

	   \note This function should satisfy the following code:
	   \code{
	   functor F;
	   arma::vec y0 = <initial conditions>;
	   F::jac_type J(y0.size(), y0.size());
	   evaluate(t, y, J) == fun(t,y);
	   J == jac(t,y);
	   }
	   
	   \param t    Current time
	   \param y    Current y-vector
	   \param J    Will contain the Jacobi matrix.
	   
	   \returns    The RHS of the ODE
	*/
	virtual arma::vec evaluate(double t, const vec_type &y,
	                           jac_type &J)
	{
		J = jac(t, y);
		return fun(t,y);
	}
};


/**
   \brief A similar functor but for the case of a sparse Jacobian matrix.
*/
/*
class functor_sparse_jac {
public:
	typedef sp_mat_type jac_type;
	virtual vec_type fun( double t, const vec_type &y ) = 0;
	virtual jac_type jac( double t, const vec_type &y ) = 0;
};
*/

#endif // FUNCTOR_HPP
