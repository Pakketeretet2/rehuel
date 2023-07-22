#include <catch2/catch.hpp>

#include "../newton.hpp"


/// \brief Rosenbrock's function functor
struct rosenbrock_func
{
	typedef mat_type jac_type;

	rosenbrock_func(double a, double b) : a(a), b(b){}
	/// \brief Function itself
	double f( const vec_type &x )
	{
		double ap = a - x[0];
		double bp = x[1] - x[0]*x[0];
		return ap*ap + b*bp*bp;

	}

	/// \brief Gradient of Rosenbrock's function
	virtual vec_type fun( const vec_type &x )
	{
		double ap = a - x[0];
		double bp = x[1] - x[0]*x[0];

		return { -2*ap - 4*b*bp*x[0], 2*b*bp };
	}

	/// \brief Hessian of Rosenbrock's function
	virtual mat_type jac( const vec_type &x )
	{
		double bp = x[1] - x[0]*x[0];

		return { { 2 + 8 * b*x[0]*x[0] - 4*b*bp, -4*b*x[0] },
		         { -4*b*x[0], 2*b } };
	}

	double a, b;
};


/// \brief Higher order Rosenbrock's function functor
struct higher_order_rosenbrock_func
{
	typedef mat_type jac_type;

	higher_order_rosenbrock_func(double a, double b, std::size_t N)
		: a(a), b(b), N(N){}

	/// \brief Function itself
	double f( const vec_type &x )
	{
		double res = 0;
		for (std::size_t i = 0; i < 2*N; i += 2) {
			double ap = a - x[i];
			double bp = x[i] - x[i+1]*x[i+1];
			res += ap*ap + b*bp*bp;
		}
		return res;
	}

	/// \brief Gradient of Rosenbrock's function
	virtual vec_type fun( const vec_type &x )
	{
		arma::vec ret(2*N);

		for (std::size_t i = 0; i < 2*N; i+=2) {
			double ap = a - x[i];
			double bp = x[i] - x[i+1]*x[i+1];
			ret[i] = -2*ap + 2*b*bp;
			ret[i+1] = -4*b*bp*x[i+1];
		}
		return ret;
	}

	/// \brief Hessian of Rosenbrock's function
	virtual mat_type jac( const vec_type &x )
	{
		mat_type J(2*N, 2*N, arma::fill::zeros);

		for (std::size_t i = 0; i < 2*N; i += 2) {
			double bp = x[i] - x[i+1]*x[i+1];
			J(i, i) = 2.0*(1 + b);
			J(i, i+1) = 0;
			J(i+1, i) = J(i, i+1);
			J(i+1, i+1) = -4*b*bp - 4*b*x[i+1]*(-2*x[i+1]);
		}
		return J;
	}

	double a, b;
	std::size_t N;
};



/// \brief An inverse power functino
struct inverse_power
{
	typedef mat_type jac_type;

	inverse_power( double a ) : a(a) {}


	/// \brief Gradient of Rosenbrock's function
	virtual vec_type fun( const vec_type &x )
	{
		double x32 = x[0]*sqrt(x[0]);
		double xm = 1.0 - x[0];
		double xm32 = xm*sqrt(xm);

		return { a / x32 - 1.0 / xm32 };
	}

	/// \brief Hessian of Rosenbrock's function
	virtual mat_type jac( const vec_type &x )
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
	my_timer timer(std::cout);
	std::size_t rosenbrock_order = 4;
	SECTION( "Rosenbrock" ){
		timer.tic();

		std::vector<double> as = { 1.0 };
		std::vector<double> bs = { 1.0, 10.0, 100.0 };

		for( std::size_t i = 0; i < as.size(); ++i ){
			double a = as[i];
			for( std::size_t j = 0; j < bs.size(); ++j ){
				double b = bs[j];

				vec_type x0(2*rosenbrock_order);
				for (std::size_t kkk = 0; kkk < 2*rosenbrock_order; kkk += 2) {
					x0[kkk] = -1.231;
					x0[kkk+1] = 4.213;
				}
				// std::cerr << "  x0 = " << x0 << "\n";

				higher_order_rosenbrock_func ros( a, b, rosenbrock_order );
				newton::options opts;

				opts.refresh_jac = 1;
				opts.tol = 1e-12;
				opts.dx_delta = 1e-12;
				opts.maxit = 10000;

				vec_type real_root = { a, a*a };
				newton::status stats;
				vec_type root = newton::newton_iterate(ros, x0, opts,
				                                       stats, true);

				CHECK( stats.conv_status == newton::SUCCESS );

				double dx = real_root(0) - root(0);
				double dy = real_root(1) - root(1);
				double fr = ros.f(root);
				/*
				std::cerr << "  status = " << stats.conv_status << ".\n";
				std::cerr << "  (dx,dy) = ( " << dx << ", " << dy << " )\n";
				std::cerr << "  f(root) = " << fr << "\n";
				std::cerr << "  tol = " << opts.tol << "\n";
				*/
				bool root_conv = ( fr < opts.tol );
				bool incr_conv = (std::fabs(dx) < opts.tol) &&
					(std::fabs(dy) < opts.tol);
				REQUIRE( (root_conv || incr_conv) );
			}
		}

		timer.toc("Rosenbrock, normal way");
	}

	SECTION( "Rosenbrock with reuse of LU" ){
		timer.tic();

		std::vector<double> as = { 1.0 };
		std::vector<double> bs = { 1.0, 10.0, 100.0 };

		for( std::size_t i = 0; i < as.size(); ++i ){
			double a = as[i];
			for( std::size_t j = 0; j < bs.size(); ++j ){
				double b = bs[j];

				vec_type x0(2*rosenbrock_order);
				for (std::size_t kkk = 0; kkk < 2*rosenbrock_order; kkk += 2) {
					x0[kkk] = -1.231;
					x0[kkk+1] = 4.213;
				}
				higher_order_rosenbrock_func ros( a, b, rosenbrock_order );
				newton::options opts;

				opts.refresh_jac = 10;
				opts.tol = 1e-12;
				opts.dx_delta = 1e-12;
				opts.maxit = 10000;

				vec_type real_root = { a, a*a };
				newton::status stats;
				vec_type root = newton::newton_iterate(ros, x0, opts,
				                                       stats, false);

				CHECK( stats.conv_status == newton::SUCCESS );

				double dx = real_root(0) - root(0);
				double dy = real_root(1) - root(1);
				double fr = ros.f(root);
				/*
				std::cerr << "  status = " << stats.conv_status << ".\n";
				std::cerr << "  (dx,dy) = ( " << dx << ", " << dy << " )\n";
				std::cerr << "  f(root) = " << fr << "\n";
				std::cerr << "  tol = " << opts.tol << "\n";
				*/
				bool root_conv = ( fr < opts.tol );
				bool incr_conv = (std::fabs(dx) < opts.tol) &&
					(std::fabs(dy) < opts.tol);
				REQUIRE( (root_conv || incr_conv) );
			}
		}
		timer.toc("Rosenbrock, reuse LU");
	}
}





TEST_CASE( "Broyden iteration on various functions.", "[broyden_iter]" )
{
	SECTION( "Rosenbrock" ){
		std::cerr << "Running on Rosenbrock, might take a while.\n";
		std::vector<double> as = { 1.0 };
		std::vector<double> bs = { 1.0, 10.0, 100.0 };

		for( std::size_t i = 0; i < as.size(); ++i ){
			double a = as[i];
			for( std::size_t j = 0; j < bs.size(); ++j ){
				double b = bs[j];

				vec_type x0 = {-1.231,4.213};
				std::cerr << "  x0 = " << x0 << "\n";

				rosenbrock_func ros( a, b );
				newton::options opts;

				opts.refresh_jac = true;
				opts.tol = 1e-12;
				opts.dx_delta = 1e-12;
				// Smaller step help converge for stiffer Rosenbrock problems:
				opts.max_step = 1.0 / b;
				opts.maxit = 10000;

				vec_type real_root = { a, a*a };
				newton::status stats;
				vec_type root = newton::broyden_iterate( ros, x0, opts,
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
