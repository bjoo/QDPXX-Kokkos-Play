/*
 * test.hpp
 *
 *  Created on: Mar 3, 2020
 *      Author: bjoo
 */

#ifndef INCLUDE_TEST_HPP_
#define INCLUDE_TEST_HPP_

#include <cstddef>
#include <array>

namespace Playground {

// The IndexOrderLeft and IndexOrderRight
// basically generate the necessary offsets for left and right
// indexing. The trick to them is that they are recursive.
// They thus need info from the parent ( an offset and a scale to use )
// and in some cases the dimensions.

// Not all the same data is used in IndexOrderLeft and IndexOrderRight
// but to aid duck-typing and keeping function signatures the same, the
// member functions all have similar signatures

template<int dim>
class IndexOrderLeft {
private:
	const std::size_t parent_offset_;
	const std::size_t parent_scale_;
	const std::array<size_t,dim> my_dims_;
	std::size_t child_scale_;
public:
	explicit IndexOrderLeft( std::size_t parent_offset,
							 std::size_t parent_scale,
							 std::array<size_t,dim> my_dims,
							 std::size_t prod_rest_of_sizes) : parent_offset_(parent_offset),
							 	 	 	 	 	 	parent_scale_(parent_scale),
													my_dims_(my_dims) {
		child_scale_= parent_scale_;
		for(int i=0; i<dim; ++i) child_scale_ *= my_dims_[i];

	};

	// Linear Offsets
	inline
	std::size_t offset(std::size_t index ) const {
		return parent_offset_ + parent_scale_*index;
	}


	// Matrix Offsets
	inline
	std::size_t offset(std::size_t row, std::size_t col) const {
		return parent_offset_ + parent_scale_*(row + my_dims_[0]*col);

	}

	// Linear Child Offsets
	inline
	std::size_t child_offset( std::size_t index ) const {
		return (*this).offset(index);

	}

	// Matrix Child offset
	inline
	std::size_t child_offset( std::size_t row, std::size_t col ) const {
		// offset takes care of any row/col swapping
		return (*this).offset(row,col);
	}

	// Child Scaling
	inline
	std::size_t child_scale() const {
		return child_scale_;
	}
};

template<int dim>
class IndexOrderRight {
private:
	const std::size_t offset_;
	const std::size_t scale_;
	const std::array<std::size_t,dim> my_dims_;
public:
	explicit IndexOrderRight( std::size_t parent_offset,
							  std::size_t parent_scale,
							  std::array<std::size_t,dim> my_dims,
							  std::size_t prod_rest_of_sizes) : offset_(parent_offset),
							 	 	 	 	 	 	scale_(prod_rest_of_sizes),
													my_dims_(my_dims){};

	// Linear Offsets
	inline std::size_t offset(std::size_t index ) const {
		return offset_ + scale_*index;
	}

	// Matrix Offsets
	inline std::size_t offset(std::size_t row, std::size_t col) const {
		return offset_  + scale_*(col + my_dims_[0]*row);
	}

	// Linear Child Offset
	inline std::size_t child_offset( std::size_t index ) const {
		return (*this).offset(index);

	}

	// Matrix Child offset
	inline std::size_t child_offset( std::size_t row, std::size_t col ) const {
		// Offset flips row/col for appropriate runnign
		return (*this).offset(row,col);
	}

	// Child scale. Layout right, so children will run fast.
	inline std::size_t child_scale() const {
		return 1;
	}
};

// T has to be a view: It should have a constructor signature:
//  T(child_offset, child_scale)
//  T::total_size() statically
template<typename T, typename IndexOrder>
class OLatticeView {
private:
	const std::size_t size_;
	const IndexOrder order_;
public:

	/** Explicit constructor: Just sets the number of elements of type T */
	explicit OLatticeView(std::size_t size) : size_(size), order_(0,1,{size_}, T::total_size()) {}

	/** The number of items of type T */
	inline
	std::size_t num_elem() const {
		return size_;
	}

	/** I am the top of the food chain, and my size is variable
	 * but I still support a non-static total_size() in case you need
	 * it for allocations etc.
	 */
	inline
	std::size_t total_size() const {
		return size_*T::total_size();
	}

	/** Give me the offset for index */
	inline
	std::size_t offset(std::size_t index) const {
		return order_.offset(index);
	}

	/** Give me the subview (i.e. T) for index i */
	inline
	T subview(std::size_t i) const {
		T ret_val(order_.child_offset(i), order_.child_scale());
		return ret_val;
	}
}; // Class


/** RScalarView is always a recursive base case: it just stores the offset to its element
 *  it does not support subview() as it has no substructure. Likewise it does not need an Index
 *  order since it has no indices. We don't need to keep the type of data being referenced, that
 *  is in the buffer. We only do indexing
 */
class RScalarView {
private:
	const std::size_t offset_;
public:
	explicit RScalarView( std::size_t parent_offset, std::size_t scale) : offset_(parent_offset) {}

	/* I am the one and only */
	inline
	std::size_t num_elem() const {
		return 1;
	}

	/* I have no substructure, so I am the one and only */
	static constexpr
	std::size_t total_size() {
		return 1;
	}

	/* Give back the offset people want so bad */
	inline
	std::size_t offset() const {
		return offset_;
	}
};

/** I am a complex number, which means I am a recursive base case,
 *  so I have no substructure (no subview). In other respects I behave
 *  like a PVector with two fixed elements, so I need an index order
 */
template<typename IndexOrder>
class RComplexView {
private:
	const IndexOrder order_;
public:
	explicit RComplexView( std::size_t parent_offset, std::size_t scale ) :
		order_(parent_offset, scale, {2}, 1) {}

	/* I hold two numbers */
	inline constexpr
	std::size_t num_elem() const {
		return 2;
	}

	/* I have no substructure, so my total elements is also 2 */
	static constexpr
	std::size_t total_size() {
		return 2;
	}

	/** Offset to the real part */
	inline
	std::size_t offsetReal() const {
		return order_.offset(0);
	}

	/** Offset to imaginary part */
	inline
	std::size_t offsetImag() const {
		return order_.offset(1);
	}

	/* Do I want to do an offset() with an index?
	 * Guess it can't hurt. Caveat emptor... this needs to be 0 or 1
	 */
	inline
	std::size_t offset(std::size_t reim) const {
		return order_.offset(reim);
	}
};


/* I am a fixed size vector */
template<typename T, int N, typename IndexOrder>
class PVectorView {
private:
	const IndexOrder order_;
public:
	explicit PVectorView( std::size_t parent_offset, std::size_t scale ) :
		order_(parent_offset, scale, {N}, T::total_size()) {}


	inline constexpr
	std::size_t num_elem() const {
		return N;
	}

	static constexpr
	std::size_t total_size() {
		return N*T::total_size();
	}

	inline
	std::size_t offset(std::size_t index) const {
		return order_.offset(index);
	}

	inline
	T subview(std::size_t i) {
		T ret_val(order_.child_offset(i), order_.child_scale());
		return ret_val;
	}
};

/** I am a fixed sized square matrix */
template<typename T, int N, typename IndexOrder>
class PMatrixView {
private:
	const IndexOrder order_;
public:
	explicit PMatrixView( std::size_t parent_offset, std::size_t scale ) :
		order_(parent_offset, scale, {N,N}, T::total_size()) {}

	inline constexpr
	std::size_t num_elem() const {
		return N*N;
	}

	static constexpr
	std::size_t total_size() {
		return N*N*T::total_size();
	}

	inline
	std::size_t offset(std::size_t row, std::size_t col) const {
		return order_.offset(row,col);
	}

	inline
	T subview(std::size_t row, std::size_t col) const {
		T ret_val(order_.child_offset(row,col), order_.child_scale());
		return ret_val;
	}
};



} // Namespace


#endif /* INCLUDE_TEST_HPP_ */
