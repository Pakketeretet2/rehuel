Rehuel
==============
A simple C++11 library for solving ordinary differential equations.
--------------

Rehuel is a relatively simple C++11 library for solving ordinary differential
equations using Runge-Kutta methods, with an emphasis on implicit Runge-Kutta
methods. It is named after the Dutch mathematician Rehuel Lobatto.

It needs some compiling but a lot of it is defined in headers.
There are two Makefiles:
 - Makefile, creates a standalone executable that integrates a simple ODE
 - Makefile_lib, creates a shared library for linking from other projects

Rehuel contains a bunch of routines it requires itself but might also be useful
for other projects. Right now they include non-linear solvers (Newton and
Broyden) and some interpolation routines.

Its only dependency is a C++11-capable compiler and the Armadillo library.
I have only tested g++ 7.2.0 but the code should be fairly portable.
*NOTE*: I recommend compiling with O2 instead of O3 as O3 seems to give
some issues with various Armadillo functionality...

At some point I might add a C interface.

See main.cpp for a typical use case.

You can use Doxygen to generate the source documentation.

