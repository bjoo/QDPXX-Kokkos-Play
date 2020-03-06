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

#if 0
template<typename T>
struct BaseTypeOf;

template<>
struct BaseTypeOf<float> {
	using Type_t = float;
};

template<>
struct BaseTypeOf<double> {
	using Type_t = double;
};


/* Forward declare this */
template<typename T, typename IndexOrder>
class OLatticeView;

template<typename T, typename IndexOrder>
struct BaseTypeOf<OLatticeView<T,IndexOrder>> {
	using Type_t = typename BaseTypeOf<T>::Type_t;
};

#endif

template<int dim>
class IndexOrderLeft {
private:
	std::size_t parent_offset_;
	std::size_t parent_scale_;
	std::array<size_t,dim> my_dims_;
public:
	explicit IndexOrderLeft( std::size_t parent_offset,
							 std::size_t parent_scale,
							 std::array<size_t,dim> my_dims,
							 std::size_t prod_rest_of_sizes) : parent_offset_(parent_offset),
							 	 	 	 	 	 	parent_scale_(parent_scale),
													my_dims_(my_dims) {};


	inline std::size_t offset(std::size_t index ) const {
		return parent_offset_ + parent_scale_*index;
	}

	inline std::size_t offset(std::size_t row, std::size_t col) const {
		return parent_offset_ + parent_scale_*(row + my_dims_[0]*col);

	}

	inline std::size_t child_offset( std::size_t index ) const {
		return (*this).offset(index);

	}

	inline std::size_t child_offset( std::size_t row, std::size_t col ) const {
			return (*this).offset(row,col);

	}

	inline std::size_t child_scale() const {
		std::size_t child_scale = parent_scale_;
		for(int i=0; i<dim; ++i) child_scale *= my_dims_[i];
		return child_scale;
	}
};

#if 1
template<int dim>
class IndexOrderRight {
private:
	std::size_t offset_;
	std::size_t scale_;
	std::array<std::size_t,dim> my_dims_;
public:
	explicit IndexOrderRight( std::size_t parent_offset,
							  std::size_t parent_scale,
							  std::array<std::size_t,dim> my_dims,
							  std::size_t prod_rest_of_sizes) : offset_(parent_offset),
							 	 	 	 	 	 	scale_(prod_rest_of_sizes),
													my_dims_(my_dims){};

	inline std::size_t offset(std::size_t index ) const {
		return offset_ + scale_*index;
	}

#if 1
	inline std::size_t offset(std::size_t row, std::size_t col) const {
		return offset_ + scale_*(col + my_dims_[1]*row);
	}
#endif

	inline std::size_t child_offset( std::size_t index ) const {
		return (*this).offset(index);

	}

	inline std::size_t child_offset( std::size_t row, std::size_t col ) const {
			return (*this).offset(row,col);
	}

	inline std::size_t child_scale() const {
		return 1;
	}
};
#endif


template<typename T, typename IndexOrder>
class OLatticeView {
private:
	const std::size_t size_;
	const IndexOrder order_;
public:

	/** Explicit constructor: Just sets the number of elements of type T */
	explicit OLatticeView(std::size_t size) : size_(size), order_(0,1,{size_}, T::total_size()) {}

	/** The number of items of type T */
	std::size_t num_elem() const {
		return size_;
	}

	// Give me the offset
	inline
	std::size_t offset(std::size_t index) const {
		return order_.offset(index);
	}

	// Subview
	T subview(std::size_t i) const {
		T ret_val(order_.child_offset(i), order_.child_scale());
		return ret_val;
	}
}; // Class


class RScalarView {
private:
	const std::size_t offset_;
public:
	explicit RScalarView( std::size_t parent_offset, std::size_t scale) : offset_(parent_offset) {}

	inline
	std::size_t num_elem() const {
		return 1;
	}


	static constexpr
	std::size_t total_size() {
		return 1;
	}

	std::size_t offset() const {
		return offset_;
	}
};


template<typename T, int N, typename IndexOrder>
class PVectorView {
private:
	const IndexOrder order_;
public:
	explicit PVectorView( std::size_t parent_offset, std::size_t scale ) :
		order_(parent_offset, scale, {N}, T::total_size()) {}

	inline
	std::size_t num_elem() const {
		return N;
	}

	static constexpr
	std::size_t total_size() {
		return N*T::total_size();
	}

	std::size_t offset(std::size_t index) const {
		return order_.offset(index);
	}

	T subview(std::size_t i) {
		T ret_val(order_.child_offset(i), order_.child_scale());
		return ret_val;
	}
};

template<typename T, int N, typename IndexOrder>
class PMatrixView {
private:
	const IndexOrder order_;
public:
	explicit PMatrixView( std::size_t parent_offset, std::size_t scale ) :
		order_(parent_offset, scale, {N,N}, T::total_size()) {}

	inline
	std::size_t num_elem() const {
		return N*N;
	}

	static constexpr
	std::size_t total_size() {
		return N*N*T::total_size();
	}

	std::size_t offset(std::size_t row, std::size_t col) const {
		return order_.offset(row,col);
	}

	T subview(std::size_t row, std::size_t col) const {
		T ret_val(order_.child_offset(row,col), order_.child_scale());
		return ret_val;
	}
};



} // Namespace


#endif /* INCLUDE_TEST_HPP_ */
