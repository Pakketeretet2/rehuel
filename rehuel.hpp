#ifndef REHUEL_HPP
#define REHUEL_HPP

/**
   \file rehuel.hpp

   \brief This is the C++ header to the Rehuel library.
*/

#include <armadillo>


#include "enums.hpp"
#include "irk.hpp"


/**
   \brief exposes a more constant interface to the underlying IRK functions.
*/
namespace rehuel {



/// Enumerates all implemented Runge-Kutta methods.
typedef irk::rk_methods rk_methods;
/// Enumerates possible return codes of odeint.
typedef irk::odeint_status_codes odeint_status_codes;



} // namespace rehuel


#endif // REHUEL_HPP
