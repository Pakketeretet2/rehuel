#ifndef CYCLIC_BUFFER
#define CYCLIC_BUFFER

#include <vector>

template <typename T> class cyclic_buffer;

template <typename T>
class cyclic_buffer {

public:

	class const_iterator {
	public:
		const_iterator() : t(nullptr), start(nullptr), e(nullptr),
		                   b(nullptr), period(0){}
		const_iterator( const const_iterator &it )
			: t(it.t), start(it.start), e(it.e),
			  b(it.b), period(it.period){}
		virtual ~const_iterator(){}
		virtual const_iterator& operator=( const const_iterator &it )
		{
			t = it.t;
			start = it.start;
			e = it.e;
			b = it.b;
			period = it.period;
		}

		virtual bool operator==( const const_iterator &it )
		{ return t == it.t && start == it.start && e == it.e
				&& b == it.b && period = it.period;
		}

		bool operator!=( const const_iterator &it )
		{ return !( it == *this ); }

		T operator*() const
		{ return t; }

		T operator->()
		{ return t; }


		const_iterator &operator++()
		{
			t++;
			if( t == e && e != start ){
				t = b;
			}else{
				// Leave it at end.
			}
		}
		const_iterator operator++(int)
		{
			const_iterator cp( *this );
			this->operator++();
			return cp;
		}

		const_iterator &operator--()
		{
			if( t == b ){
				t = e - 1;
			}else{
				t--;
			}

			if( t == start ){
				t = e;
			}
		}
		const_iterator operator--(int)
		{
			const_iterator cp( *this );
			this->operator++();
			return cp;
		}



		T* t;
		T* start, e, b;
		int period;
	};



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
		: period_( o.period() ), current_(0), storage_( o.get_storage() ) {}

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
	}

	~cyclic_buffer(){}

	std::vector<T> &storage(){}
	const std::vector<T> &storage() const {}

	T* data() { return storage_.data(); }
	const T* data() const { return storage_.data(); }

	bool empty() const { return storage_.empty(); }

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
		if( storage_.size() < period_ ){
			storage_.push_back(t);
		}else{
			if( current_ == period_ ){
				current_ = 0;
			}
			storage_[current_] = t;
		}
		++current_;
	}

	/*
	iterator begin() const {}
	iterator end() const {}
	iterator rbegin() const {}
	iterator rend() const {}
	*/

	/*
	const_iterator cbegin() const
	{
		const_iterator it;
		T *storage_start = storage_.data();
		it.t = storage_start + current_;
		it.start = it.t;
		it.e = storage_start + period_;
		it.b = storage_start;
		it.period = period_;
	}

	const_iterator cend() const
	{
		const_iterator it;
		T *storage_start = storage_.data();
		it.t = storage_start + period_;
		it.start = it.t;
		it.e = storage_start + period_;
		it.b = storage_start;
		it.period = period_;
	}

	iterator begin() const
	{
		iterator it;
		T *storage_start = storage_.data();
		it.t = storage_start + current_;
		it.start = it.t;
		it.e = storage_start + period_;
		it.b = storage_start;
		it.period = period_;
	}

	iterator end() const
	{
		iterator it;
		T *storage_start = storage_.data();
		it.t = storage_start + period_;
		it.start = it.t;
		it.e = storage_start + period_;
		it.b = storage_start;
		it.period = period_;
	}
	*/
	/*
	const_iterator crbegin() const {}
	const_iterator crend() const {}
	*/
	std::size_t size() const { return storage_.size(); }
	int period() const { return period_; }
	int current() const { return current_; }

	/*
	max_size
	resize
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
