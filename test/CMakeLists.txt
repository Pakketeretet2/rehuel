cmake_minimum_required(VERSION 3.28)
project(RehuelTest)

find_package(Catch2 3 REQUIRED)
find_package(Armadillo REQUIRED)

add_executable(test armadillo.cpp cyclic_vector.cpp irk.cpp newton.cpp test.cpp
               test_interpolate.cpp test_multistep.cpp test_test_equations.cpp)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/.." ${ARMADILLO_INCLUDE_DIRS})
target_link_directories(test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_link_libraries(test PRIVATE Catch2::Catch2WithMain ${ARMADILLO_LIBRARIES} rehuel)
