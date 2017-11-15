Rehuel
==============
A simple C++11 library for solving ordinary differential equations.
--------------

Rehuel is a relatively simple C++11 library for solving ordinary differential
equations using Runge-Kutta methods, with an emphasis on implicit Runge-Kutta
methods. It is named after the Dutch mathematician Rehuel Lobatto.

It needs some compiling but a lot of it is defined in headers.

It contains a bunch of routines it requires itself but might also be useful
for other projects. Right now they include non-linear solvers (Newton and
Broyden) and some interpolation routines.

Its only dependency is a C++11-capable compiler and the Armadillo library.

At some point I might add a C interface.

See main.cpp for a typical use case.

You can use Doxygen to generate the source documentation.

