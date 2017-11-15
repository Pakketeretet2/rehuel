#include "interpolate.hpp"

namespace interpolate {

std::vector<arma::vec> linear( const std::vector<double> &x_vals,
                               const std::vector<arma::vec> &y_vals,
                               const std::vector<double> &x_grid )
{
	double x0 = x_grid[0];
	double x1 = 0;

	std::size_t Ng = x_grid.size();
	std::vector<arma::vec> interp( Ng );

	// Ensure t_grid is monotonically increasing.

	for( std::size_t i = 1; i < Ng; ++i ){
		x1 = x_grid[i];
		double dx = x1 - x0;
		if( dx <= 0 ){
			std::cerr << "x_grid should be strictly "
			          << "non-decreasing!\n";
			return interp;
		}
		x0 = x1;
	}

	int ix = 0;
	std::size_t Ny = y_vals[0].size();
	for( std::size_t i = 0; i < Ng; ++i ){
		double xgi = x_grid[i];

		double xm = x_vals[ix];
		double xp = x_vals[ix+1];

		while( xp < xgi ){
			++ix;
			xm = x_vals[ix];
			xp = x_vals[ix+1];
		}

		double dx = xp - xm;
		double f  = xgi - xm;
		f /= dx;

		arma::vec yi( Ny );
		yi.zeros(Ny);

		for( std::size_t j = 0; j < Ny; ++j ){
			double ym = y_vals[ix][j];
			double yp = y_vals[ix+1][j];
			double dy = yp - ym;
			double y_interp = ym + f*dy;

			yi[j] = y_interp;
		}
		interp[i] = yi;
	}

	return interp;
}

} // namespace interpolate
