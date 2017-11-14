#ifndef ODES_HPP
#define ODES_HPP

// This file contains a bunch of "standard" ODEs to play with.

#include <armadillo>

namespace radau {

namespace odes {

arma::vec test_func( const arma::vec &x );
arma::mat test_J( const arma::vec &x );

arma::vec blue_sky_catastrophe( double t, const arma::vec &yy,
                                double mu, double eps );
arma::mat blue_sky_catastrophe_J( double t, const arma::vec &yy,
                                  double mu, double eps );

arma::vec brusselator( double t, const arma::vec &yy, double a, double b );
arma::mat brusselator_J( double t, const arma::vec &yy, double a, double b );

} // namespace odes

} //namespace radau

#endif // ODES_HPP
