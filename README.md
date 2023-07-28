Rehuel
==============
A simple C++11 library for solving ordinary differential equations.
--------------

Rehuel is a relatively simple C++11 library for solving ordinary differential
equations. It is named after the Dutch mathematician Rehuel Lobatto.

The goals of the project are two-fold: On the one hand, we aim to provide
a high quality solver for various kinds of methods that can be drag-and-dropped
into other projects, similar to Matlab's ode45 and ode15s.
On the other hand, the project also provides me with an interesting test bed
for the various methods out there.

For a quick tutorial/guide, see the guide in \ref guide.md

---------------
Recommended solvers
---------------

<b>Non-stiff:</b>
    DORMAND_PRINCE_54    (A good default explicit RK method (think ode45))
    CASH_KARP_54         (Ditto, might be optimized later to "fail early"
                         on moderately stiff systems)
    BOGACKI_SHAMPINE_32  (If you need less accuracy and more speed)

<b>Moderately stiff</b>:
    It depends really. Try DORMAND_PRINCE_54, and monitor your RAM. If you are
running out of RAM and/or the solution takes forever, integrate over a
shorter interval. If the solution still takes too long, try a stiff solver.
    Alternatively, you can just try an IRK method and benefit from its higher accuracy too.

<b>Very stiff</b>:
    LOBATTO_IIIC_85  (this should probably be your default stiff solver)
    LOBATTO_IIIC_43  (might be faster than the 85 method but seems to have trouble with some problems)
    RADAU_IIA_53     (should work decently well too)

In examples/example_equations.cpp we provide a small driver program that can
apply various methods to various problems so you can get a feel for the
strengths and weaknesses of each method.

-------------------------
Building/installing
-------------------------
The library needs some compiling but a lot of it is defined in headers.
The Makefile creates a shared library for linking from other projects.

Its only dependency is a C++11-capable compiler and the Armadillo library.

See examples/example_equations.cpp for code that applies the methods to
some test problems. There are also some tests in the test/ directory.
These directories both have their own Makefile.

You can use Doxygen to generate the source documentation.

----------------
Resources
----------------
Most of the design and choices are based on the following excellent resources
by Ernst Hairer and Gerhard Wanner:
 - Stiff differential equations solved by Radau methods
 - Solving Ordinary Differential Equations I
 - Solving Ordinary Differential Equations II
