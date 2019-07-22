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

template <typename solver_coefficients>
arma::mat rk_to_stab_region(const arma::vec &X, const arma::vec &Y,
                            const solver_coefficients &sc)
{
	// The stability region of a RK method follows from applying it to
	// f(t,y) = lambda*y. After some math, you obtain that the stages
	// satisfy K = \lambda inv(I - lambda*dt*A)*(y0, y0, y0, ..., y0)^T
	// and hence
	// y1/y0 = 1 + z * b * inv(I - z*A)*(1, 1, 1, ..., 1)^T
	// with z = lambda*dt

	auto stab_func = [&sc](std::complex<double> z) -> arma::cx_double
	                 {
		                 auto N = sc.b.size();
		                 arma::vec e = arma::ones(N);
		                 arma::mat I = arma::eye(N,N);
		                 arma::cx_mat M = I - z*sc.A;
		                 auto Ai = arma::inv(M);
		                 std::complex<double> one(1.0,0.0);
		                 return one + z*arma::dot(sc.b, Ai*e);
	                 };
	
	auto N = X.size();
	auto M = Y.size();
	arma::mat S(N, M);
	for (std::size_t i = 0; i < N; ++i) {
		for (std::size_t j = 0; j < M; ++j) {
			std::complex<double> z(X(i), Y(j));
			S(i,j) = std::fabs(stab_func(z));
		}
	}

	return S;
}


void stability_region(const std::string &method)
{
	int m = irk::name_to_method(method);
	arma::vec X = arma::linspace(-30,10,250);
	arma::vec Y = arma::linspace(-20,20,300);
	arma::mat S;
	
	if (m) {
		auto sc = irk::get_coefficients(m);
		S = rk_to_stab_region(X, Y, sc);
	}
	m = erk::name_to_method(method);
	if (m) {
		auto sc = erk::get_coefficients(m);
		S = rk_to_stab_region(X, Y, sc);
	}
	
	for (std::size_t j = 0; j < Y.size(); ++j) {
		for (std::size_t i = 0; i < X.size(); ++i) {
			std::cout << X(i) << " " << Y(j) << " " << S(i,j) << "\n";
		}
		std::cout << "\n";
	}
	
}




int main(int argc, char **argv)
{
	if (argc < 2) {
		std::cerr << "Pass a method to calculate stability region for!\n";
		return -1;
	}
	stability_region(argv[1]);
	
	return 0;
}
