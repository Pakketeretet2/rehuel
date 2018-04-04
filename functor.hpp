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
