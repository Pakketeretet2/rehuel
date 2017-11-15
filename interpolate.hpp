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
   \file interpolate.hpp
*/

#ifndef INTERPOLATE_HPP
#define INTERPOLATE_HPP

#include <armadillo>
#include <vector>

/**
   \brief Contains functions for interpolating data.
*/
namespace interpolate {

/**
   \brief Interpolates linearly to given grid.

   \param x_vals  The x coordinates of the points
   \param y_vals  The values corresponding to the points
   \param x_grid  The grid to interpolate to.

   \returns the interpolation of y on t_grid.
*/
std::vector<arma::vec> linear( const std::vector<double> &x_vals,
                               const std::vector<arma::vec> &y_vals,
                               const std::vector<double> &x_grid );


} // namespace interpolate

#endif // INTERPOLATE_HPP
