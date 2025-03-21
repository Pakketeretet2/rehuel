#include "catch2/catch_all.hpp"
#include "cyclic_buffer.hpp"

#include <iostream>

TEST_CASE("Tests that cyclic buffer works as expected.", "[cyclic_buffer]")
{
	// Cyclic buffer operates as a stack, last in is first.
	cyclic_buffer<int> v(5);

	REQUIRE(v.size() == 0);
	REQUIRE(v.empty());

	v.push_back(1);
	v.print_internal_state(std::cout) << "\n";

	v.push_back(2);
	v.push_back(3);
	REQUIRE(v.size() == 3);
	v.push_back(4);

	v.push_back(5);
	v.print_internal_state(std::cout) << "\n";

	REQUIRE(v.size() == 5);
	REQUIRE(v[0] == 5);

	REQUIRE(v[1] == 4);
	REQUIRE(v[2] == 3);
	REQUIRE(v[3] == 2);

	REQUIRE(v[4] == 1);
	REQUIRE(!v.empty());
	v.print_internal_state(std::cout) << "\n";

	// Cyclic buffer operates as a stack, last in is first.
	v.push_back(1);
	std::cout << "After push_back(1), before push_back(2):\n";
	v.print_internal_state(std::cout) << "\n";
	v.push_back(2);
	v.print_internal_state(std::cout) << "\n";
	REQUIRE(v[0] == 2);
	REQUIRE(v[1] == 1);

	REQUIRE(v[2] == 5);
	REQUIRE(v[3] == 4);
	REQUIRE(v[4] == 3);
	REQUIRE(v.size() == 5);


	std::cout << "Internal state of cyclic vector before pushing 4 9s:\n";
	v.print_internal_state(std::cout) << "\n";

	for (std::size_t i = 0; i < 4; ++i) {
		v.push_back(9);
	}

	std::cout << "Internal state of cyclic vector:\n";
	v.print_internal_state(std::cout) << "\n";





}
