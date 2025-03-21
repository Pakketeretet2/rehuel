#include "multistep.hpp"
#include "test_equations.hpp"
#include "catch2/catch_all.hpp"

TEST_CASE("Multistep methods", "[multistep]")
{
	test_equations::exponential E(-1.0);
	double t0 = 0.0, t1 = 10.0;
	multistep::solver_options opts;
	vec_type Y0 = { 1.0 };

	std::vector<double> dts = { 1e-6, 1e-5, 1e-4,
	                            1e-3, 1e-2, 1e-1, 1 };
	std::vector<double> errs(dts.size());

	for (int order = 1; order <= 5; ++order) {
		opts.order = order;
		std::ofstream out("adams_bashforth_err_order_" +
		                  std::to_string(order) + ".dat");
		
		for (std::size_t i = 0; i < dts.size(); ++i) {
			double dt = dts[i];
			basic_output sol = multistep::adams_bashforth(E, t0, t1, Y0, opts, dt);
			std::size_t N = sol.t_vals.size();
			double max_err2 = 0.0;
			for (std::size_t nt = 0; nt < N; ++nt) {
				double y_true = exp(-sol.t_vals[nt]);
				double y_err = sol.y_vals[nt][0] - y_true;
				double err2 = y_err*y_err;
				max_err2 = std::max(max_err2, err2);
			}
			errs[i] = std::sqrt(max_err2);

			out << dt << " " << std::setprecision(16) << errs[i] << "\n";
		}
	}
}
