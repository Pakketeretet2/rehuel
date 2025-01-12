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
   \brief This little program attempts to numerically compute the stability
   region of a given method.
*/
#include <complex>
#include <iostream>

#include "arma_include.hpp"
#include "rehuel.hpp"
#include "functor.hpp"


std::complex<double> stability_function(std::complex<double> z,
                                        const arma::mat &A,
                                        const arma::vec &b)
{
	std::size_t N = b.size();
	arma::vec e = arma::ones(N);
	arma::mat I = arma::eye(N,N);
	arma::cx_mat M = I - z*A;
	auto Ai = arma::inv(M);
	std::complex<double> one(1.0,0.0);
	return one + z*arma::dot(b, Ai*e);
}


template <typename solver_coefficients>
arma::mat rk_to_stab_region(const arma::vec &X, const arma::vec &Y,
                            const solver_coefficients &sc,
                            bool embedded_method)
{
	// The stability region of a RK method follows from applying it to
	// f(t,y) = lambda*y. After some math, you obtain that the stages
	// satisfy K = \lambda inv(I - lambda*dt*A)*(y0, y0, y0, ..., y0)^T
	// and hence
	// y1/y0 = 1 + z * b * inv(I - z*A)*(1, 1, 1, ..., 1)^T
	// with z = lambda*dt
	auto N = X.size();
	auto M = Y.size();
	arma::mat S(N, M);

	if (S.n_rows == 0 || S.n_cols == 0) {
		std::cerr << "Allocation of S failed? Is " << N << "x" << M
		          << " = " << M*N << " really that big?\n";
	}

	for (std::size_t i = 0; i < N; ++i) {
		for (std::size_t j = 0; j < M; ++j) {
			std::complex<double> z(X(i), Y(j));
			if (embedded_method) {
				S(i,j) = std::fabs(stability_function(z, sc.A, sc.b2));
			} else {
				S(i,j) = std::fabs(stability_function(z, sc.A, sc.b));
			}
		}
	}

	return S;
}


void stability_limit(const std::string &method, std::ostream &out)
{
	std::vector<double> X = {-1e8, -1e7, -1e6, -1e5, -1e4, -1e3, -1e2, -10,
		-9, -8, -7, -6, -5, -4, -3, -2, -1.5, -1, -0.9, -0.8, -0.7, -0.6,
		-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 1};

	out << "#z Rz Rz_embed\n";
	if (int m = irk::name_to_method(method)) {
		irk::solver_coeffs sc = irk::get_coefficients(m);
		for (double x : X) {
			out << x << " "
			    << std::fabs(stability_function({x, 0}, sc.A, sc.b)) << " "
			    << std::fabs(stability_function({x, 0}, sc.A, sc.b2)) << "\n";
		}
	} else if(int m = erk::name_to_method(method)) {
		erk::solver_coeffs sc = erk::get_coefficients(m);
		for (double x : X) {
			out << x << " "
			    << std::fabs(stability_function({x, 0}, sc.A, sc.b)) << " "
			    << std::fabs(stability_function({x, 0}, sc.A, sc.b2)) << "\n";
		}
	}
}


void stability_region(const std::string &method, bool embedded_method,
                      std::ostream &out)
{
	int m = irk::name_to_method(method);
	arma::vec X = arma::linspace(-190, 10, 400);
	arma::vec Y = arma::linspace( -20, 20, 80);
	arma::mat S;

	if (m) {
		auto sc = irk::get_coefficients(m);
		S = rk_to_stab_region(X, Y, sc, embedded_method);
	} else {
		std::cerr << "Method " << method << " not recognized as IRK!\n";
	}
	m = erk::name_to_method(method);
	if (m) {
		auto sc = irk::get_coefficients(m);
		// std::cerr << "Coeffs:\n" << sc << "\n";
		S = rk_to_stab_region(X, Y, sc, embedded_method);
	} else {
		std::cerr << "Method " << method << " not recognized as ERK!\n";
	}

	for (std::size_t j = 0; j < Y.size(); ++j) {
		for (std::size_t i = 0; i < X.size(); ++i) {
			out << X(i) << " " << Y(j) << " " << S(i,j) << "\n";
		}
		out << "\n";
	}
}


int main(int argc, char **argv)
{
	if (argc < 2) {
		std::cerr << "Pass a method to calculate stability region for!\n";
		return -1;
	}

	std::cerr << "args are ";
	for (int i = 1; i < argc; ++i) {
		std::cerr << "(" << i << ") " << argv[i];
	}
	std::cerr << "\n";

	std::string method(argv[1]);
	if (int method_enum = irk::name_to_method(method)) {
		auto sc = irk::get_coefficients(method_enum);
		std::cerr << "Method coeffs:\n" << sc << "\n";
	}
	std::cerr.flush();

	std::string stab_region_fname = "stability_" + method + "_region.dat";
	std::ofstream stab_region_fout(stab_region_fname);
	stability_region(method, false, stab_region_fout);

	std::string stab_region_embed_fname = "stability_" + method + "_embedded_region.dat";
	std::ofstream stab_region_embed_fout(stab_region_embed_fname);
	stability_region(method, true, stab_region_embed_fout);

	std::string out_fname = "stability_" + method + "_limit.dat";

	std::ofstream limit_fout(out_fname);
	stability_limit(method, limit_fout);

	return 0;
}
