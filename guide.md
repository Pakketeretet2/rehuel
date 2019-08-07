Rehuel guide
====

This is a short tutorial on how to use Rehuel in your C++ project.

### Non-stiff problems ###

Suppose you want to integrate the Van der Pol oscillator:  

\f[ y' = w \\
  w' = \mu (1 - y^2)w + y
\f]

We first need to write a functor that calculates the right-hand-side for us.
Note that the functor should have the same interface as the \ref functor class defined in \ref functor.hpp.
Hence, the following code will do:
~~~~{.cpp}
#include "rehuel.hpp"

struct vdpol
{
	typedef arma::mat jac_type;
	vdpol( double mu = 1.0 ) : mu(mu){}

	arma::vec fun( double t, const arma::vec &y )
	{
		return { y[1], ((1 - y[0]*y[0]) * y[1] - y[0]) / mu };
	}

	jac_type jac( double t, const arma::vec &y )
	{
		jac_type J(2,2);
		J(0,0) = 0.0;
		J(0,1) = 1.0;

		J(1,0) = -(2.0 * y[0] * y[1] + 1.0) / mu;
		J(1,1) = ( 1.0 - y[0]*y[0] ) / mu;

		return J;
	}

	double mu;
};
~~~~
We include the Rehuel header above so that the necessary libraries (specifically Armadillo) are included.
Now we can write our main function that will provide the functor to a Rehuel integrator and integrate it, then print it to stdout.

~~~~{.cpp}
#include <iostream>

int main(int argc, char **argv)
{
        vdpol V(1.0);
        double t0 = 0.0;             // Initial time
        double t1 = 3000.0;          // Final time
        arma::vec Y0 = { 2.0, 0.0 }; // Initial vector
        // erk is a good place to start. If your problem is stiff, you will find out.
        auto sol = erk::odeint(V, t0, t1, Y0);
        std::cerr << "Solving took " << sol.elapsed_time << " ms.\n";
        // Write the solution to stdout:
        std::size_t Nt = sol.t_vals.size();
        for (std::size_t nt = 0; nt < Nt; ++nt) {
            std::cout << sol.t_vals[nt];
            for (std::size_t j = 0; j < sol.y_vals[nt].size(); ++j) {
                std::cout << " " << sol.y_vals[nt](j);
            }
            std::cout << "\n";
        }
        
        return 0;
}
~~~~
Output from time integrators in Rehuel are in the form of a "solution" struct, which contains various fields.
These fields differ per method, but they always contain status, t_vals and y_vals.
Status is an integer that specifies if the solve was succesful (in which case it is 0).
t_vals is a std::vector<double> containing the time values at which a solution was produced and
y_vals is a std::vector<arma::vec> containing the solutions corresponding to the time points.
For irk and erk they also contain information about the performance.

The program should be compiled and run as
~~~~{.sh}
g++ -O3 -I<location of Rehuel source dir> -L<location of rehuel shared library> \
    -lrehuel -larmadillo example.cpp -o example
./example > example.dat
~~~~
Running it will produce the following output:
~~~~{.sh}
    Rehuel: Integrating over interval [ 0, 3000 ]...
            Method = DORMAND_PRINCE_54
Solving took 18.674 ms.
~~~~
Of course the timing for you might be different.

### Stiff problems ###

The Van der Pol oscillator has one parameter \f$ \mu \f$. For very small \f$\mu\f$, the equation becomes stiff.
We can modify the previous code slightly to show the effect of the parameter \f$mu\f$:
~~~~{.cpp}
#include <iostream>

int main(int argc, char **argv)
{
        double t0 = 0.0;             // Initial time
        double t1 = 3000.0;          // Final time
        arma::vec Y0 = { 2.0, 0.0 }; // Initial vector

        for (double mu : { 1.0, 5e-4 };
                auto sol1 = erk::odeint(V, t0, t1, Y0);
                std::cerr << "Solving for mu = " << mu
                          << " took " << sol.elapsed_time << " ms.\n";

                std::size_t Nt = sol1.t_vals.size();
                for (std::size_t nt = 0; nt < Nt; ++nt) {
                    std::cout << sol1.t_vals[nt];
                    for (std::size_t j = 0; j < sol1.y_vals[nt].size(); ++j) {
                        std::cout << " " << sol1.y_vals[nt](j);
                    }
                    std::cout << "\n";
                }
                std::cout << "\n\n";

                auto sol2 = irk::odeint(V, t0, t1, Y0);
                std::cerr << "Solving for mu = " << mu
                          << " took " << sol2.elapsed_time << " ms.\n";

                Nt = sol2.t_vals.size();
                for (std::size_t nt = 0; nt < Nt; ++nt) {
                    std::cout << sol2.t_vals[nt];
                    for (std::size_t j = 0; j < sol2.y_vals[nt].size(); ++j) {
                        std::cout << " " << sol2.y_vals[nt](j);
                    }
                    std::cout << "\n";
                }
                std::cout << "\n\n";


        }
        
        return 0;
}
~~~~