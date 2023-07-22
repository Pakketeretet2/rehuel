#ifndef CYCLIC_BUFFER
#define CYCLIC_BUFFER

#include <cassert>
#include <vector>
#include <iostream>


template <typename T> class cyclic_buffer;

template <typename T>
class cyclic_buffer {

public:
	cyclic_buffer() : period_(0), current_(0), size_(0)
	{ }

	explicit cyclic_buffer(int period)
		: period_(period), current_(0), size_(0), storage_(2*period_)
	{ }
	cyclic_buffer( const cyclic_buffer &o )
		: period_(o.period()), current_(o.current()), size_(o.size()), storage_(o.storage()) {}


	cyclic_buffer &operator=( const cyclic_buffer &o )
	{
		if(*this != &o){
			if(period_ >= o.period()){
				// No need to resize.
			}

			// Copy contents of storage:
			storage_.resize(2*o.period());
			period_ = o.period();
			current_ = o.current();
			size_ = o.size();
			std::copy(o.storage().begin(), o.storage().end(), storage_.begin());
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
		return size_ == 0;
	}

	// Access and storing is optimized by storing twice the
	// vector so we can always iterate linearly. Most recent
	// should be considered the top.
	T operator[](std::size_t i) const
	{
		assert (i < period_ && "i should be < period!");
		assert (current_ >= i && "i out of bounds");
		std::size_t j = current_ - i;

		return storage_[j];
	}

	T& operator[](std::size_t i)
	{
		assert (i < period_ && "i should be < period!");
		assert (current_ >= i && "i out of bounds");
		std::size_t j = current_ - i;

		return storage_[j];
	}

	// We store the buffer twice so we can always iterate linearly
	// If we add 1 then e.g. we have 1 _ _ 1 _ _
	//                              |^
	// 2: 1 2 _ 1 2 _
	//   |  ^
	// 3: 1 2 3 1 2 3
	//   |    ^
	// 4: 4 2 3 4 2 3
	//     |    ^
	// 5: 4 5 3 4 5 3
	//       |    ^
	// 6: 4 5 6 4 5 6
	//         |    ^
	// 7: 7 5 6 7 5 6
	//     |    ^

	void push_back( const T &t )
	{
		// Add t to the internal buffer.
		long next_spot = current_ + 1;
		if (size_ == 0) {
			next_spot = 0;
		}
		if (next_spot >= period_) {

			storage_[next_spot] = t;
			storage_[next_spot - period_] = t;
		} else {
			storage_[next_spot] = t;
		}
		current_ = next_spot;
		if (next_spot == 2*period_ - 1) {
			current_ = period_ - 1;
		}

		if (size_ < period_) ++size_;
	}


	std::size_t size() const
	{ return size_; }

	std::size_t period() const { return period_; }
	int current() const { return current_; }

	struct iterator
	{
		iterator(long idx, typename std::vector<T>::iterator it){}

		// We iterate backwards through the vector...
		iterator &operator++()
		{
			--idx_;
			return *this;
		}

		bool operator==(const iterator &o)
		{
			return o.idx_ == idx_;
		}
		bool operator!=(const iterator &o)
		{
			return o.idx_ != idx_;
		}

		T operator*() const
		{ return *(it_ + idx_); }

		T &operator*()
		{ return *(it_ + idx_); }

		long idx_;
		typename std::vector<T>::iterator it_;
	};

	iterator begin()
	{
		return iterator(current_, storage_.begin());
	}

	iterator end()
	{
		if (current_ >= period_) {
			return iterator(current_ = period_, storage_.begin());
		} else {
			return iterator(0, storage_.begin());
		}
	}

	/**
	   \brief Resizes the internal storage.

	   \warning This invalidates the contents of the cyclic buffer!
	*/
	void resize(std::size_t size)
	{
		storage_.clear();
		storage_.reserve(2*size);
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

	std::ostream &print_internal_state(std::ostream &o) const
	{
		o << "Cyclic vector state:"
		  << " period  " << period_
		  << " current " << current_ << "\n"
		  << "Storage:\n";

		for (std::size_t i = 0; i < storage_.size(); ++i) {
			o << storage_[i];
			if (std::size_t(current_) == i) {
				o << '*';
			}
			o << " ";
		}
		return o;
	}

private:
	long period_;
        long current_;
	long size_;

	std::vector<T> storage_;
};


#endif // CYCLIC_BUFFER
