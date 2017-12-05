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


/// The standard form of the RHS of an ODE.
typedef arma::vec (*standard_ode_rhs)(double, const arma::vec&);
/// The standard form of the Jacobi matrix of the RHS ODE.
typedef arma::mat (*standard_ode_jac)(double, const arma::vec&);


/// Enumerates all implemented Runge-Kutta methods.
typedef irk::rk_methods rk_methods;
/// Enumerates possible return codes of odeint.
typedef irk::odeint_status_codes odeint_status_codes;


/// \brief Integrates ODE. See \ref irk::odeint.
static const auto &integrate_ode = irk::odeint<standard_ode_rhs, standard_ode_jac>;

/// \brief Vefifies solver correctness. See \ref irk::verify_solver_coeffs.
static const auto &verify_solver_coeffs = irk::verify_solver_coeffs;

/// \brief Returns coefficients for given solver. \ref irk::get_coefficients.
static const auto &get_solver_coefficients = irk::get_coefficients;

/// \brief Returns default solver options. See \ref irk::default_solver_options.
static const auto&get_default_solver_options = irk::default_solver_options;




} // namespace rehuel


#endif // REHUEL_HPP
