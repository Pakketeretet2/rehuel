#include "catch.hpp"

#include "cyclic_buffer.hpp"

TEST_CASE("Tests that cyclic buffer works as expected.", "[cyclic_buffer]")
{
	// Cyclic buffer operates as a stack, last in is first.
	cyclic_buffer<int> v(5);

	REQUIRE(v.size() == 0);
	REQUIRE(v.empty());
	
	v.push_back(1);
	v.push_back(2);
	v.push_back(3);
	REQUIRE(v.size() == 3);
	v.push_back(4);
	v.push_back(5);
	REQUIRE(v.size() == 5);
	REQUIRE(v[0] == 5);
	REQUIRE(v[1] == 4);
	REQUIRE(v[2] == 3);
	REQUIRE(v[3] == 2);
	REQUIRE(v[4] == 1);
	REQUIRE(!v.empty());
	
	// Cyclic buffer operates as a stack, last in is first.
	v.push_back(1);
	v.push_back(2);
	REQUIRE(v[0] == 2);
	REQUIRE(v[1] == 1);
	REQUIRE(v[2] == 5);
	REQUIRE(v[3] == 4);
	REQUIRE(v[4] == 3);
	REQUIRE(v.size() == 5);

	
}
