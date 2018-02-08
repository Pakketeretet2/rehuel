#include "interpolate.hpp"

#include <cassert>
#include <typeinfo>


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



/**
   \brief This is the actual implementation of the formula.

   \note Make sure the reference is initialized with zeros!
*/
template <typename T>
void newton_interpolation_formula( const std::vector<double> &x_pts,
                                   const std::vector<T> &y_pts,
                                   double x, T &res )
{
	// Construct the polynomial.
	// int poly_order = x_pts.size() - 1;
	assert( x_pts.size() == y_pts.size() &&
	        "Interpolate should have equal number of xs and ys!" );

	double xlo = *std::min_element( x_pts.begin(), x_pts.end() );
	double xhi = *std::max_element( x_pts.begin(), x_pts.end() );

	// Check to make sure x is inside x_pts.
	if( x < xlo || x > xhi ){
		std::cerr << "Warning! Target x " << x << " is out of range [ "
		          << xlo << ", " << xhi << " ]!\n";
	}

	int npts = x_pts.size();

	std::vector<double> dx( npts );
	for( int i = 0; i < npts; ++i ){
		dx[i] = x - x_pts[i];
	}

	for( int i = 0; i < npts; ++i ){
		double nom = 1.0;
		double den = 1.0;
		for( int j = 0; j < i; ++j ){
			nom *= dx[j];
			den *= x_pts[i] - x_pts[j];
		}
		for( int j = i+1; j < npts; ++j ){
			nom *= dx[j];
			den *= x_pts[i] - x_pts[j];
		}
		res += nom * y_pts[i] / den;
	}
}




double newton( const std::vector<double> &x_pts,
               const std::vector<double> &y_pts, double x )
{

	// Construct the Legendre polynomials:
	double y = 0.0;
	//newton_interpolation_formula<double>( x_pts, y_pts, x, y );
	return y;
}


arma::vec newton( const std::vector<double> &x_pts,
                  const std::vector<arma::vec> &y_pts, double x )
{

	// Construct the Legendre polynomials:
	arma::vec y;
	y.zeros( y_pts[0].size() );
	newton_interpolation_formula<arma::vec>( x_pts, y_pts, x, y );
	return y;
}





} // namespace interpolate
