#include "newton.hpp"


namespace newton {

const char *status_message(int status)
{
	switch (status) {
	case SUCCESS:
		return "SUCCESS";
	case INCREMENT_DIVERGE:
		return "INCREMENT_DIVERGE";
	case ITERATION_ERROR_TOO_LARGE:
		return "ITERATION_ERROR_TOO_LARGE";
	case MAXIT_EXCEEDED:
		return "MAXIT_EXCEEDED";
	case GENERIC_ERROR:
		return "GENERIC_ERROR";
	default:
		return "Unrecognized status code!";
	}
}

}
