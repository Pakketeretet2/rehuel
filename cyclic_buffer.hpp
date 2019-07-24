#ifndef CYCLIC_BUFFER
#define CYCLIC_BUFFER

#include <vector>

template <typename T> class cyclic_buffer;

template <typename T>
class cyclic_buffer {

public:


	cyclic_buffer() : period_(0), current_(0), storage_()
	{
		storage_.reserve(period_);
	}

	explicit cyclic_buffer( int period )
		: period_(period), current_(0), storage_()
	{
		storage_.reserve(period_);
	}
	cyclic_buffer( const cyclic_buffer &o )
		: period_( o.period() ), current_(0), storage_( o.storage() ) {}


	cyclic_buffer &operator=( const cyclic_buffer &o )
	{
		if( *this != &o ){
			if( period_ >= o.period() ){
				// No need to resize.
			}

			// Copy contents of storage:
			storage_.resize( o.period() );
			period_ = o.period();
			current_ = o.current();
			std::copy( o.storage().begin(), o.storage().end(),
			           storage_.begin() );
		}

		return *this;
	}

	~cyclic_buffer(){}

	std::vector<T> &storage()
	{
		return storage_;
	}
	const std::vector<T> &storage() const
	{
		return storage_;
	}

	T* data() { return storage_.data(); }
	const T* data() const { return storage_.data(); }

	/**
	   \brief is the buffer empty?
	   \returns true if buffer is empty, false otherwise.
	*/
	bool empty() const {
		return storage_.empty();
	}

	T operator[]( std::size_t i ) const
	{
		long j = current_ - i - 1;
		if( j < 0 ) j += period_;

		return storage_[j];
	}

	T& operator[]( std::size_t i )
	{
		long j = current_ - i - 1;
		if( j < 0 ) j += period_;

		return storage_[j];
	}

	void push_back( const T &t )
	{
		int storage_size = storage_.size();
		if( storage_size < period_ ){
			storage_.push_back(t);
		}else{
			if( current_ == period_ ){
				current_ = 0;
			}
			storage_[current_] = t;
		}
		++current_;
	}


	std::size_t size() const { return storage_.size(); }
	std::size_t period() const { return period_; }
	int current() const { return current_; }

	/**
	   \brief Resizes the internal storage.

	   \warning This invalidates the contents of the cyclic buffer!
	*/
	void resize( std::size_t size )
	{
		storage_.clear();
		storage_.reserve( size );
		period_  = size;
		current_ = 0;
	}
	
	/*
	max_size
	capacity
	empty
	reserve
	shrink_to_fit
	*/


private:
	int period_;
	int current_;

	std::vector<T> storage_;
};


#endif // CYCLIC_BUFFER
