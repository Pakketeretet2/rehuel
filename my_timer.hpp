#ifndef MY_TIMER_HPP
#define MY_TIMER_HPP

/**
   \file my_timer.hpp
*/
#include <iostream>
#include <memory>


#ifndef _WIN32
#include <sys/time.h>

/**
  \brief A simple timer class based on sys/time

  This class is a fairly accurate timer. Useful for timing parts of programs
  like loops and stuff.
*/
class my_timer {
public:
	/// Default constructor, no output.
	my_timer() : out(nullptr), t_tic{0}, t_toc{0}
	{ init_tic_toc(); }

	/// Constructor that takes an std::ostream to which stuff is
	/// occasionally printed.
	explicit my_timer(std::ostream &out_stream) : out(&out_stream), t_tic{0}, t_toc{0}
	{ init_tic_toc(); }

	/// Empty destructor
	~my_timer(){}

	/// Sets the "tic"-time to current time.
	void tic()
	{ gettimeofday(&t_tic, nullptr); }


	/**
	  \brief Computes difference between the "tic"-time and current time.

	  \param msg   A message to print to out in addition
	               to the elapsed time (optional)
	  \returns     The difference between tic-time and
	               current time in milliseconds.
	*/
	double toc( const std::string &msg, const std::string &post )
	{
		gettimeofday(&t_toc, nullptr);
		double diff_msec = (t_toc.tv_usec - t_tic.tv_usec)*1e-3 +
			(t_toc.tv_sec  - t_tic.tv_sec)*1000.0;
		double diff_sec  = diff_msec*1e-3;
		if( out ){
			if( !msg.empty() ){
				*out << msg << ": ";
			}
			*out << diff_msec << " ms elapsed ("
			     << diff_sec << " s).";
			if( !post.empty() ){
				*out << " " << post;
			}
			*out << "\n";
		}
		return diff_msec;
	}

	double toc( const std::string &msg )
	{
		return toc( msg, "" );
	}
	double toc( )
	{
		return toc( "", "" );
	}


	/// Enables output stream and sets it to o.
	void enable_output( std::ostream &o )
	{
		out = &o;
	}

	/**
	  \brief Disables the output stream.

	  \warning After calling this, out is lost!
	*/
	void disable_output()
	{ out = nullptr; }


private:
	std::ostream *out;
	timeval t_tic, t_toc;

	void init_tic_toc()
	{
		gettimeofday(&t_tic, nullptr);
		gettimeofday(&t_toc, nullptr);
	}

	// This class is not copy-able:
	my_timer( const my_timer &o ) = delete;
	my_timer &operator=( const my_timer &o ) = delete;
};

#else
// On Windows, treat everything as No-op.
class my_timer {
public:
	/// Default constructor, no output.
	my_timer(){}

	/// Constructor that takes an std::ostream to which stuff is
	/// occasionally printed.
	explicit my_timer(std::ostream &out_stream) {}

	/// Empty destructor
	~my_timer(){}

	/// Sets the "tic"-time to current time.
	void tic() {}

	/**
	  \brief Computes difference between the "tic"-time and current time.

	  \param msg   A message to print to out in addition
	               to the elapsed time (optional)
	  \returns     The difference between tic-time and
	               current time in milliseconds.
	*/
	double toc( const std::string &msg, const std::string &post )
	{
		return 0.0;
	}
	double toc( const std::string &msg )
	{
		return toc( msg, "" );
	}
	double toc( )
	{
		return toc( "", "" );
	}


	/// Enables output stream and sets it to o.
	void enable_output( std::ostream &o )
	{ }

	/**
	  \brief Disables the output stream.

	  \warning After calling this, out is lost!
	*/
	void disable_output()
	{ }

private:
	void init_tic_toc()
	{ }

	// This class is not copy-able:
	my_timer( const my_timer &o ) = delete;
	my_timer &operator=( const my_timer &o ) = delete;
};


#endif // _WIN32


#endif // MY_TIMER_HPP
