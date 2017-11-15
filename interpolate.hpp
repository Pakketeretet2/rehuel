#ifndef INTERPOLATE_HPP
#define INTERPOLATE_HPP

#include <armadillo>
#include <vector>


namespace interpolate {


std::vector<arma::vec> linear( const std::vector<double> &x_vals,
                               const std::vector<arma::vec> &y_vals,
                               const std::vector<double> &t_grid );


} // namespace interpolate

#endif // INTERPOLATE_HPP
