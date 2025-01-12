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
   \file example_equations.cpp

   \brief contains some example code that integrates various equations.

   The main function is a simple driver that a user can supply an equation with
   parameters to, as well as a method and options.
*/

#include <iostream>
#include <string>
#include <vector>

#include "arma_include.hpp"
#include "rehuel.hpp"          // This includes most of the library
#include "test_equations.hpp"  // This contains some example ODEs



struct user_options {
	user_options() : eq("exponential"), method("LOBATTO_IIIC_85"),
	                 t0(0.0), t1(10.0), dt(1e-2),
	                 rel_tol(1e-5), abs_tol(1e-4), out_interval(0) {}
	std::string eq, method;

	double t0, t1, dt;
	double rel_tol, abs_tol;

	bool time_internals;
	int out_interval;

	std::string output_fname;
};


void print_usage()
{
	std::string exe_name = "./rehuel_example";
	std::string w(exe_name.length(), ' ');
	std::cerr << "Usage:\n"
	          << exe_name << " --equation <equation> --method <method>\n"
	          << w << " --time-span <t0> <t1> --dt <dt>\n"
	          << w << " --rel-tol <rtol> --abs-tol <atol>\n"
	          << w << " --out-interval <interval> --time-internals <0/1>,\n"
	          << w << " --output-file <fname>\n"
	          << "with\n"
	          << "\t<equation>: Equation to solve. Possible values:\n"
	          << "\t            values: exponential, stiff-equation,\n"
	          << "\t            robertson, three-body, van-der-pol,\n"
	          << "\t            brusselator, lorenz\n\n"
	          << "\t<method>:   Time integration method. See Doxygen\n"
	          << "\t            documentation for supported methods.\n"
	          << "\t            Default is LOBATTO_IIIC_85\n\n"
	          << "\t<t0>, <t1>: Initial and final time to integrate.\n\n"
	          << "\t<dt>        Initial time step size to use. Adaptive\n"
	          << "\t            integrators change this during integration\n\n"
	          << "\t<rtol>:     Relative error tolerance for integration\n\n"
	          << "\t<atol>:     Absolute error tolerance for integration\n\n"
	          << "\t<interval>: Output interval.\n\n"
	          << "\t<0/1>:      0 or 1 (for booleans)\n\n"
	          << "\t<fname>:    Output file name\n\n";
}




int parse_opts(int argc, char **argv, user_options &u_opts)
{
	int i = 1;
	while (i < argc) {
		std::string arg = argv[i];
		if (arg == "-h" || arg == "--help") {
			print_usage();
			return 1;
		} else if (arg == "--equation") {
			if (i+1 == argc) {
				std::cerr << "Option \"" << arg
				          << "\" needs a value!\n";
				return -1;
			}
			u_opts.eq = argv[i+1];
			i += 2;
		} else if (arg == "--method") {
			if (i+1 == argc) {
				std::cerr << "Option \"" << arg
				          << "\" needs a value!\n";
				return -1;
			}
			u_opts.method = argv[i+1];
			i += 2;
		} else if (arg == "--time-span") {
			if (i+2 >= argc) {
				std::cerr << "Option \"" << arg
				          << "\" needs two values!\n";
				return -1;
			}
			u_opts.t0 = std::stof(argv[i+1]);
			u_opts.t1 = std::stof(argv[i+2]);
			i += 3;
		} else if (arg == "--dt") {
			if (i+1 == argc) {
				std::cerr << "Option \"" << arg
				          << "\" needs a value!\n";
				return -1;
			}
			u_opts.dt = std::stof(argv[i+1]);
			i += 2;
		} else if (arg == "--rel-tol") {
			if (i+1 == argc) {
				std::cerr << "Option \"" << arg
				          << "\" needs a value!\n";
				return -1;
			}
			u_opts.rel_tol = std::stof(argv[i+1]);
			i += 2;
		} else if (arg == "--abs-tol") {
			if (i+1 == argc) {
				std::cerr << "Option \"" << arg
				          << "\" needs a value!\n";
				return -1;
			}
			u_opts.abs_tol = std::stof(argv[i+1]);
			i += 2;
		} else if (arg == "--out-interval") {
			if (i+1 == argc) {
				std::cerr << "Option \"" << arg
				          << "\" needs a value!\n";
			}
			u_opts.out_interval = std::stoi(argv[i+1]);
			i += 2;
		} else if (arg == "--time-internals") {
			if (i+1 == argc) {
				std::cerr << "Option \"" << arg
				          << "\" needs 0 or 1.\n";
				return -1;
			}
			int val = std::stoi(argv[i+1]);
			i += 2;
			if (val != 0 && val != 1) {
				std::cerr << "Got an ambiguous value for a "
				          << "bool! Please use 0 or 1 only!\n";
				return -2;
			}
			u_opts.time_internals = val;
		} else if (arg == "--output-file") {
			if (i+1 == argc) {
				std::cerr << "Option \"" << arg
				          << "\" needs an output file name!\n";
				return -1;
			}
			u_opts.output_fname = argv[i+1];
			i += 2;
		} else {
			std::cerr << "Unrecognized arg \"" << arg << "\"!\n";
			return -1;
		}
	}

	return 0;
}


template <typename T>
bool read_param(const std::string &param_name, const std::string &eq_name, T &t)
{
	std::cerr << param_name << " parameter for " << eq_name << "?\n";
	std::cin >> t;
	return std::cin.good();
}




// templated becaues irk::rk_output and erk::rk_output are different types.
template <typename rk_output>
void print_output(const rk_output &sol)
{
	std::size_t n_times = sol.t_vals.size();
	for (std::size_t n = 0; n < n_times; ++n) {
		std::cout << sol.t_vals[n];
		std::size_t n_ys = sol.y_vals[n].size();
		for (std::size_t j = 0; j < n_ys; ++j) {
			std::cout << " " << sol.y_vals[n](j);
		}
		std::cout << "\n";
	}
}


void set_irk_options(irk::solver_options &s_opts, newton::options &n_opts,
                     const user_options &u_opts)
{
	// With the current API, the user needs to explicitly set the newton
	// iterator options for the irk solver by setting the n_opts poitner.
	s_opts.newton_opts = &n_opts;
	s_opts.rel_tol = u_opts.rel_tol;
	s_opts.abs_tol = u_opts.abs_tol;
	s_opts.out_interval = u_opts.out_interval;
	s_opts.time_internals = u_opts.time_internals;
	n_opts.tol = 0.1*std::min(s_opts.rel_tol, s_opts.abs_tol);
}


void set_erk_options(erk::solver_options &s_opts, const user_options &u_opts)
{
	s_opts.rel_tol = u_opts.rel_tol;
	s_opts.abs_tol = u_opts.abs_tol;
	s_opts.out_interval = u_opts.out_interval;
	s_opts.time_internals = u_opts.time_internals;
}



void print_fun_jac_counters(const irk::rk_output &sol)
{
	std::cerr << sol.count.fun_evals << " function evaluations.\n"
	          << sol.count.jac_evals << " Jacobi matrix evaluations.\n";
}


void print_fun_jac_counters(const erk::rk_output &sol)
{
	std::cerr << sol.count.fun_evals << " function evaluations.\n";
}



template <typename rk_output>
void print_performance(const rk_output &sol)
{
	double elapsed_time = sol.elapsed_time;
	std::string time_units = "ms";
	if (elapsed_time > 1000) {
		elapsed_time /= 1000;
		time_units = "s";
	} else if (elapsed_time < 1e-3) {
		elapsed_time *= 1000;
		time_units = "us";
	}
	std::cerr << "Solved equation in "
	          << elapsed_time << " " << time_units << ".\n";
	std::cerr << "Accepted " << 100*sol.accept_frac << "% of the steps\n";
	print_fun_jac_counters(sol);
}


template <typename functor_type>
int int_equation(functor_type &F, const user_options &u_opts, vec_type Y0)
{
	output_options output_opts;
	std::ofstream output_file_stream;
	if (!u_opts.output_fname.empty()) {
		std::cerr << "Writing to file\n";
		output_file_stream.open(u_opts.output_fname);
		output_opts.set_output_stream(output_file_stream);
		output_opts.disable_store_in_vectors();
	}
	int method = irk::name_to_method(u_opts.method);
	if (method) {
		// This means the user-supplied solver is indeed an IRK.
		irk::solver_options s_opts = irk::default_solver_options();

		newton::options n_opts;
		set_irk_options(s_opts, n_opts, u_opts);
		irk::rk_output sol = irk::odeint(F, u_opts.t0, u_opts.t1, Y0,
		                                 s_opts, output_opts, method,
		                                 u_opts.dt);
		if (sol.status) {
			std::cerr << "Got an error integrating ODE. :/\n";
			return 2;
		}
		print_performance(sol);
		return 0;
	}
	method = erk::name_to_method(u_opts.method);
	if (method) {
		// This means the user-supplied solver is indeed an IRK.
		erk::solver_options s_opts = erk::default_solver_options();
		set_erk_options(s_opts, u_opts);
		erk::rk_output sol = erk::odeint(F, u_opts.t0, u_opts.t1, Y0,
		                                 s_opts, output_opts, method, u_opts.dt);
		if (sol.status) {
			std::cerr << "Got an error integrating ODE. :/\n";
			return 2;
		}

		print_performance(sol);
		return 0;
	}
	std::cerr << "Method \"" << u_opts.method << "\" not recognized!\n";
	return -1;
}



int solve_robertson(const user_options &opts)
{
	std::cerr << "Solving the Robertson problem. This is a system of very "
	          << "stiff equations. Using an explicit method will probably "
	          << "blow up your RAM if t1 is the typical value of 1e12.\n"
	          << "  !! Be careful !!  \n";
	std::cerr << "Continue with method " << opts.method << "? (y/N)?\n";
	std::string answer;
	bool resume = false;
	while (std::cin >> answer) {
		if (answer == "Y" || answer == "y" ||
		    answer == "yes" || answer == "Yes") {
			resume = true;
		}
		break;
	}
	if (!resume) {
		return 1;
	}
	test_equations::rober R;
	return int_equation(R, opts, {1.0, 0.0, 0.0});

}


int solve_three_body(const user_options &u_opts,
                     vec_type Y0 = { 1.0, 0.0,  3.0, 2.0,  0.0, 3.0,
                                     0.0, 2.0, -1.0, 0.0, -1.0, 0.0})
{
	std::cerr << "Solving the three body problem. Both explicit and "
	          << "implicit methods should work reasonably well, unless "
	          << "the mass differences become very large.\n";
	test_equations::three_body TB;
	double m1, m2, m3;
	read_param("mass 1", "three body problem", m1);
	read_param("mass 2", "three body problem", m2);
	read_param("mass 3", "three body problem", m3);
	TB.set_m1(m1);
	TB.set_m2(m2);
	TB.set_m3(m3);

	return int_equation(TB, u_opts, Y0);
}


int solve_van_der_pol(const user_options &opts)
{
	std::cerr << "Solving the Van der Pol oscillator. It has a limit cycle "
	          << "for mu > 0. For very small mu the equation is stiff.\n";
	test_equations::vdpol V;
	read_param("mu", "Van der Pol", V.mu);
	return int_equation(V, opts, {2.0, 0.0});
}


int solve_brusselator(const user_options &opts)
{
	std::cerr << "Solving the Van der Pol oscillator. It has a limit cycle "
	          << "if b > 1 + a^2. Even for large b it is not very stiff.\n";
	test_equations::bruss B;
	read_param("a", "Brusselator", B.a);
	read_param("b", "Brusselator", B.b);

	return int_equation(B, opts, {2.5, 2.5});
}


int solve_exponential(const user_options &opts)
{
	std::cerr << "Solving a simple exponential function. Any method should "
	          << "be good for this, explicit are probably faster.\n"
	          << "The equation is y' = l*y so remember the minus sign!\n";
	test_equations::exponential E;
	read_param("lambda", "exponential", E.l);
	return int_equation(E, opts, {1.0});
}


int solve_stiff_equation(const user_options &opts)
{
	std::cerr << "Solving a stiff equation. Implicit solvers should be "
	          << "a lot faster on this problem.\n";
	test_equations::stiff_eq S;
	return int_equation(S, opts, {1.0, 1.0});
}


int solve_lorenz(const user_options &opts)
{
	std::cerr << "Solving a Lorenz problem. Any method should be fine.\n"
	          << "This system of ODEs has chaotic solutions, including a "
	          << "strange attractor\n"
	          << "for rho = 28, sigma = 10 and beta = 2.66667.\n";
	test_equations::lorenz L;
	read_param("rho",   "Lorenz oscillator", L.r);
	read_param("sigma", "Lorenz oscillator", L.s);
	read_param("beta",  "Lorenz oscillator", L.b);
	return int_equation(L, opts, {1.0, 1.0, 1.0});
}


int run_batch()
{
	test_equations::rober R;

	vec_type Y0 = { 1.0, 0.0,  3.0, 2.0,  0.0, 3.0,
		0.0, 2.0, -1.0, 0.0, -1.0, 0.0};
	test_equations::three_body TB;
	TB.set_m1(1.0);
	TB.set_m2(1.0);
	TB.set_m3(1000.0);

	test_equations::vdpol V;
	V.mu = 1e-4;

	test_equations::lorenz L;
	L.r = 1;
	L.s = 1;
	L.b = 1;

	user_options opts;
	opts.method = "LOBATTO_IIIC_85";
	opts.out_interval = 1000;

	int status_TB = int_equation(TB, opts, Y0);
	int status_L = int_equation(L, opts, {1.0, 1.0, 1.0});
	int status_R = int_equation(R, opts, {1.0, 0.0, 0.0});
	int status_V = int_equation(V, opts, {1.0, 0.0, 0.0});

	std::cerr << "Status for three-body:  " << status_TB << "\n"
	          << "       for Lorenz:      " << status_L << "\n"
	          << "       for Robertson:   " << status_R << "\n"
	          << "       for Van der Pol: " << status_V << "\n";
	return (status_TB & status_L & status_R & status_V);
}



int main(int argc, char **argv)
{

	std::cerr << "Welcome to the Rehuel example program.\n";

	user_options u_opts;

	if (argc <= 1) {
		std::cerr << "\n";
		print_usage();
		return 3;
	}
	int parse_status = parse_opts(argc, argv, u_opts);
	if (parse_status < 0) {
		std::cerr << "Error parsing user options!\n";
		return -1;
	} else if (parse_status > 0) {
		return 0;
	}

	// Determine the equation:
	if (u_opts.eq == "exponential") {
		return solve_exponential(u_opts);
	} else if (u_opts.eq == "stiff-equation") {
		return solve_stiff_equation(u_opts);
	} else if (u_opts.eq == "robertson") {
		return solve_robertson(u_opts);
	} else if (u_opts.eq == "three-body") {
		return solve_three_body(u_opts);
	} else if (u_opts.eq == "three-body-resume") {
		vec_type Y0 = { -161.263, 195.287, -161.26,
		                195.316, -1940.43, -2519.87,
		                21.0719, -1.56866, -21.7366,
		                2.36384, -1.33533, -1.73402 };
		return solve_three_body(u_opts, Y0);
	} else if (u_opts.eq == "van-der-pol") {
		return solve_van_der_pol(u_opts);
	} else if (u_opts.eq == "brusselator") {
		return solve_brusselator(u_opts);
	} else if (u_opts.eq == "lorenz") {
		return solve_lorenz(u_opts);
	} else {
		std::cerr << "Equation \"" << u_opts.eq
		          << "\" not recognized!\n";
		return -1;
	}


	return 0;
}
