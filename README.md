NOTE: Development of Rehuel moved to Gitlab: https://gitlab.com/Pakketeretet2/rehuel

Rehuel
==============
A simple C++11 library for solving ordinary differential equations.
--------------

Rehuel is a relatively simple C++11 library for solving ordinary differential
equations using Runge-Kutta methods, with an emphasis on implicit Runge-Kutta
methods. It is named after the Dutch mathematician Rehuel Lobatto.

Most of the design and choices are based on the following excellent resources
by Ernst Hairer and Gerhard Wanner:
 - Stiff differential equations solved by Radau methods
 - Solving Ordinary Differential Equations II

The library needs some compiling but a lot of it is defined in headers.
The Makefile creates a shared library for linking from other projects

Rehuel contains a bunch of routines it requires itself but might also be useful
for other projects. Right now they include non-linear solvers (Newton and
Broyden).

Its only dependency is a C++11-capable compiler and the Armadillo library.
I have only tested g++ 7.2.0 but the code should be fairly portable.

At some point I might add a C interface.

See test_adaptive_step/main.cpp for code that applies the methods to
some test problems. There are also some tests in the test/ directory.

You can use Doxygen to generate the source documentation.

