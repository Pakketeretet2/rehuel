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
   \file arma_include.hpp

   \brief This file includes the Armadillo library and sets up some options
   so that these are consistently set across all files.
*/


#ifndef ARMA_INCLUDE_HPP
#define ARMA_INCLUDE_HPP

#define ARMA_USE_CXX11
#define ARMA_USE_OPENBLAS
#define ARMA_USE_SUPERLU

#include <armadillo>


#endif // ARMA_INCLUDE_HPP
