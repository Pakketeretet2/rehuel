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
   \file odes.hpp
*/

#ifndef ODES_HPP
#define ODES_HPP

// This file contains a bunch of "standard" ODEs to play with.

#include <armadillo>

namespace odes {

arma::vec test_func( double t, const arma::vec &x );
arma::mat test_J( double t, const arma::vec &x );

arma::vec blue_sky_catastrophe( double t, const arma::vec &yy,
                                double mu, double eps );
arma::mat blue_sky_catastrophe_J( double t, const arma::vec &yy,
                                  double mu, double eps );

arma::vec brusselator( double t, const arma::vec &yy, double a, double b );
arma::mat brusselator_J( double t, const arma::vec &yy, double a, double b );

arma::vec analytic_solvable_func( double t, const arma::vec &yy,
                                  double a, double b, double w );
arma::mat analytic_solvable_func_J( double t, const arma::vec &yy,
                                    double a, double b, double w );

arma::vec analytic_stiff_sol( double t );
arma::vec analytic_stiff( double t, const arma::vec &yy );
arma::mat analytic_stiff_J( double t, const arma::vec &yy );



} // namespace odes

#endif // ODES_HPP
