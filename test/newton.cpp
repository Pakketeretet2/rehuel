#include "catch.hpp"

#include "../newton.hpp"


/// \brief Rosenbrock's function functor
struct rosenbrock_func
{
	typedef arma::mat jac_type;

	rosenbrock_func(double a, double b) : a(a), b(b){}
	/// \brief Function itself
	double f( const arma::vec &x )
	{
		double ap = a - x[0];
		double bp = x[1] - x[0]*x[0];
		return ap*ap + b*bp*bp;

	}

	/// \brief Gradient of Rosenbrock's function
	virtual arma::vec fun( const arma::vec &x )
	{
		double ap = a - x[0];
		double bp = x[1] - x[0]*x[0];

		return { -2*ap - 4*b*bp*x[0], 2*b*bp };
	}

	/// \brief Hessian of Rosenbrock's function
	virtual arma::mat jac( const arma::vec &x )
	{
		double bp = x[1] - x[0]*x[0];

		return { { 2 + 8 * b*x[0]*x[0] - 4*b*bp, -4*b*x[0] },
			{ -4*b*x[0], 2*b } };
	}

	double a, b;
};


/// \brief An inverse power functino
struct inverse_power
{
	typedef arma::mat jac_type;

	inverse_power( double a ) : a(a) {}


	/// \brief Gradient of Rosenbrock's function
	virtual arma::vec fun( const arma::vec &x )
	{
		double x32 = x[0]*sqrt(x[0]);
		double xm = 1.0 - x[0];
		double xm32 = xm*sqrt(xm);

		return { a / x32 - 1.0 / xm32 };
	}

	/// \brief Hessian of Rosenbrock's function
	virtual arma::mat jac( const arma::vec &x )
	{
		double x52 = x[0]*x[0]*sqrt(x[0]);
		double xm = 1 - x[0];
		double xm52 = xm*xm*sqrt(xm);

		return { -1.5*( a / x52 + 1.0 / xm52 ) };
	}



	double a;
};


TEST_CASE( "Newton iteration on various functions.", "[newton_iter]" )
{
	SECTION( "Rosenbrock" ){
		std::vector<double> as = { 1.0, 2.0, 3.0, 10.0 };
		std::vector<double> bs = { 100.0, 200.0, 20.0, 1.0 };

		for( std::size_t i = 0; i < as.size(); ++i ){
			double a = as[i];
			for( std::size_t j = 0; j < bs.size(); ++j ){
				double b = bs[i];

				arma::vec x0;
				x0.randu(2);

				std::cerr << "  x0 = " << x0 << "\n";

				rosenbrock_func ros( a, b );

				newton::options opts;
				opts.tol = 1e-12;
				opts.dx_delta = 1e-12;
				opts.maxit = 1000;

				arma::vec real_root = { a, a*a };
				newton::status stats;
				arma::vec root = newton::newton_iterate( ros, x0, opts,
				                                         stats, false );

				CHECK( stats.conv_status == newton::SUCCESS );

				double dx = real_root(0) - root(0);
				double dy = real_root(1) - root(1);
				double fr = ros.f(root);
				std::cerr << "  status = " << stats.conv_status << ".\n";
				std::cerr << "  (dx,dy) = ( " << dx << ", " << dy << " )\n";
				std::cerr << "  f(root) = " << fr << "\n";
				std::cerr << "  tol = " << opts.tol << "\n";
				bool root_conv = ( fr < opts.tol );
				bool incr_conv = (std::fabs(dx) < opts.tol) &&
					(std::fabs(dy) < opts.tol);
				REQUIRE( (root_conv || incr_conv) );
			}
		}
	}

}
