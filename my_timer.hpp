/*
   Rehuel: a simple C++ library for solving ODEs


   Copyright 2017-2019, Stefan Paquay (stefanpaquay@gmail.com)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

============================================================================= */

/**
   \file my_timer.hpp
*/

#ifndef MY_TIMER_HPP
#define MY_TIMER_HPP

#include <iostream>
#include <memory>

#ifndef _WIN32
#include <sys/time.h>
#endif // _WIN32


/**
  \brief A simple timer class based on sys/time for Unix platforms.

  This class is a fairly accurate timer. Useful for timing parts of programs
  like loops and stuff.
*/
class my_timer_linux {
public:

	
	/// Default constructor, no output.
	my_timer_linux() : out(nullptr), t_tic{0,0}
	{ init(); }

	/**
	   \brief Constructor that sets an std::ostream for output printing.

	   \param out_stream  Print output to here.
	*/
	explicit my_timer_linux(std::ostream &out_stream)
		: out(&out_stream), t_tic{0,0}
	{ init(); }

	/// Empty destructor
	~my_timer_linux(){}

	/// Sets the "tic"-time to current time.
	void tic()
	{ gettimeofday(&t_tic, nullptr); }


	/**
	  \brief Computes difference between the "tic"-time and current time.

	  \param msg   A message to print to out in addition
	               to the elapsed time (optional)
          \param post  A message to print after the elapsed time (optional)
	  \returns     The difference between tic-time and
	               current time in milliseconds.
	*/
	double toc( const std::string &msg, const std::string &post )
	{
		double diff_msec = get_elapsed(t_tic);
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

	/**
	   \brief Print just a message before timing but not after.

	   \overload toc
	*/
	double toc( const std::string &msg )
	{
		return toc( msg, "" );
	}

	/**
	   \brief Print just the timing.

	   \overload toc
	*/
	double toc( )
	{
		return toc( "", "" );
	}


	/**
	   Enables output and sets the output stream.

	   \param o The output stream to use.
	*/
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

	/**
	   \brief Returns the current "tic" so you can store it.
	*/
	timeval get_tic() const { return t_tic; }


	/**
	   \brief Returns the elapsed time (in ms) between given tic and now.
	*/
	double get_elapsed(const timeval &user_tic) const
	{
		timeval t_toc;
		gettimeofday(&t_toc, nullptr);
		
		return (t_toc.tv_usec - user_tic.tv_usec)*1e-3 +
			(t_toc.tv_sec  - user_tic.tv_sec)*1000.0;
	}


private:
	std::ostream *out;  ///< Pointer to the output stream to use
	timeval t_tic;      ///< The time stamp at which tic was last called

	/**
	   \brief Initializes t_tic to current time.
	*/
	void init()
	{
		gettimeofday(&t_tic, nullptr);
	}

	/// Deleted copy-constructor
	my_timer_linux( const my_timer_linux &o ) = delete;
	/// Deleted assignment operator
	my_timer_linux &operator=( const my_timer_linux &o ) = delete;
};


/**
  \brief dummy timer for Windows.

  This dummy class provides the same interface as my_timer_linux but
  produces no-ops for all calls.
*/
class my_timer_windows {
public:
	my_timer_windows(){}

	explicit my_timer_windows(std::ostream &) {}

	/// Empty destructor
	~my_timer_windows(){}

	void tic() {}
	double toc( const std::string &, const std::string & )
	{ return 0.0; }
	double toc( const std::string &msg )
	{ return toc( msg, "" ); }
	double toc( )
	{ return toc( "", "" ); }

	void enable_output( std::ostream & ) { }
	void disable_output() { }

private:
	// void init() { }
	my_timer_windows( const my_timer_windows &o ) = delete;
	my_timer_windows &operator=( const my_timer_windows &o ) = delete;
};

#ifndef _WIN32
typedef my_timer_linux my_timer;
#else
typedef my_timer_windows my_timer;
#endif // _WIN32


#endif // MY_TIMER_HPP
