/*
 * test3.hpp
 *
 *  Created on: Mar 24, 2020
 *      Author: bjoo
 */

#pragma once

#include "Kokkos_Core.hpp"
namespace Playground {

// Type Traits for Type->LocalType conversions
//
//
template<typename T>
struct LocalType;


// Scalar Type
//
//  Contains a type T which should be a POD type
//
//
//  We declare: A type based on views
//              A type based on local storage (registerize on GPUs)
//              A type trait to get local from view based
//
//
//  Forward declare local type
template<typename T>
struct RScalarLocal;

// The view based type
template<typename T, typename ViewType, size_t _NumDims=1>
struct RScalar {

	// This is a view which should be ok on device
	ViewType _data;

	// The indices already frozen
	std::array<size_t,8> _indices;

    // Construct from a view and fixed indices
	KOKKOS_INLINE_FUNCTION
	RScalar(ViewType data_in, std::array<size_t,8> indices) : _data(data_in), _indices(indices){};

	// Construct from a view (assume indices are all 0) -- most for testing purposes
	KOKKOS_INLINE_FUNCTION
	RScalar(ViewType data_in) : _data(data_in), _indices{0,0,0,0,0,0,0,0}{}

	// Construct from RScalarLocal
	// Forward declare since RScalarLocal is incomplete
	RScalar(const RScalarLocal<T>& t);

	// Op assign from
	RScalar& operator=(const RScalarLocal<T>& t );

	// View types cannot be default initialized
	RScalar() = delete;

	KOKKOS_INLINE_FUNCTION
	T& elem() const {
		if constexpr ( _NumDims == 1 ) {
			return _data(_indices[0]);
		}

		if constexpr ( _NumDims == 2 ) {
			return _data( _indices[0], _indices[1] );
		}

		if constexpr ( _NumDims == 3) {
			return _data( _indices[0], _indices[1], _indices[2] );
		}

		if constexpr ( _NumDims == 4 ) {
			return _data( _indices[0], _indices[1], _indices[2], _indices[3] );
		}

		if constexpr ( _NumDims == 5 ) {
			return _data( _indices[0], _indices[1], _indices[2], _indices[3],
					      _indices[4]);
		}

		if constexpr ( _NumDims == 6 ) {
			return _data( _indices[0], _indices[1], _indices[2], _indices[3],
					      _indices[4], _indices[5]);
		}

		if constexpr ( _NumDims == 7 ) {
			return _data( _indices[0], _indices[1], _indices[2], _indices[3],
					      _indices[4], _indices[5], _indices[6]);
		}

		if constexpr ( _NumDims == 8 ) {
			return _data( _indices[0], _indices[1], _indices[2], _indices[3],
					      _indices[4], _indices[5], _indices[6], _indices[7]);
		}

	}

};


// A thread local RScalar
// with compact storage
template<typename T>
struct RScalarLocal {

	// The Data
	T _data;


	// Initialize with the data
	KOKKOS_INLINE_FUNCTION
	RScalarLocal(const T& data_in) : _data(data_in) {};


	// Init from an RScalarView (this is known
	template<typename ViewType, size_t N>
	RScalarLocal(const RScalar<T,ViewType,N>& view_in) : _data(view_in.elem()) {};


	template<typename ViewType, size_t N>
	KOKKOS_INLINE_FUNCTION
	RScalarLocal<T>& operator=(const RScalar<T,ViewType,N>& view_in) {
		(*this).elem()=view_in.elem();
		return (*this);
	}

	// Const getter
	KOKKOS_INLINE_FUNCTION
	const T& elem() const {
	  return _data;
	}

	// setter
	KOKKOS_INLINE_FUNCTION
	T& elem() {
		return _data;
	}
};

template<typename T, typename ViewType, size_t N>
RScalar<T,ViewType,N>::RScalar( const RScalarLocal<T>& local_in ) {
	(*this).elem() = local_in.elem();
}

template<typename T, typename ViewType, size_t N>
KOKKOS_INLINE_FUNCTION
RScalar<T,ViewType,N>& RScalar<T,ViewType,N>::operator=(const RScalarLocal<T>& local_in ) {
	(*this).elem() = local_in.elem();
	return (*this);
}

template<typename T>
struct RComplexLocal;

template<typename T, typename ViewType, size_t _IdxPos, size_t _NumDims >
struct RComplex {
	ViewType _data;
	std::array<size_t,8> _indices;

	KOKKOS_INLINE_FUNCTION
	RComplex(ViewType data_in, std::array<size_t,8> indices) : _data(data_in), _indices(indices){};

	// View Types Cannot be Default constructed
	RComplex() = delete;

	// Forward declare these since RComplexLocal is only
	// forward declared
	RComplex(const RComplexLocal<T>& in);
	RComplex& operator=(const RComplexLocal<T>& in);

	// Getters and Setters
	KOKKOS_INLINE_FUNCTION
	T& real() const {
		std::array<size_t,8> new_idx(_indices);
		new_idx[ _IdxPos ] = 0;
		return get( new_idx );
	}

	KOKKOS_INLINE_FUNCTION
	T& imag() const {
		std::array<size_t,8> new_idx(_indices);
		new_idx[ _IdxPos ] = 1;
		return get( new_idx );
	}

	KOKKOS_INLINE_FUNCTION
	T& get( std::array<size_t,8> new_idx ) const {
		if constexpr ( _NumDims == 1 ) {
			return _data(new_idx[0]);
		}

		if constexpr ( _NumDims == 2 ) {
			return _data(new_idx[0],new_idx[1]);
		}

		if constexpr ( _NumDims == 3) {
			return _data( new_idx[0], new_idx[1], new_idx[2] );
		}

		if constexpr ( _NumDims == 4 ) {
			return _data( new_idx[0], new_idx[1], new_idx[2], new_idx[3] );
		}

		if constexpr ( _NumDims == 5 ) {
			return _data( new_idx[0], new_idx[1], new_idx[2], new_idx[3],
					new_idx[4]);
		}

		if constexpr ( _NumDims == 6 ) {
			return _data( new_idx[0], new_idx[1], new_idx[2], new_idx[3],
					new_idx[4], new_idx[5]);
		}

		if constexpr ( _NumDims == 7 ) {
			return _data( new_idx[0], new_idx[1], new_idx[2], new_idx[3],
					new_idx[4], new_idx[5], new_idx[6]);
		}

		if constexpr ( _NumDims == 8 ) {
			return _data( new_idx[0], new_idx[1], new_idx[2], new_idx[3],
					new_idx[4], new_idx[5], new_idx[6], new_idx[7]);
		}

	}


};

// ThreadLocal Type
template<typename T >
struct RComplexLocal {
	T _data[2];

	RComplexLocal() {}

	KOKKOS_INLINE_FUNCTION
	RComplexLocal( const T& re, const T& im ) : _data{re,im} {}

	template<typename ViewType, size_t IdxPos, size_t NumDims >
	KOKKOS_INLINE_FUNCTION
	RComplexLocal( const RComplex<T,ViewType,IdxPos,NumDims>& in) : _data{ in.real(), in.imag() } {}

	template<typename ViewType, size_t IdxPos, size_t NumDims >
	KOKKOS_INLINE_FUNCTION
	RComplexLocal& operator=(const RComplex<T,ViewType,IdxPos,NumDims>& in) {
		_data[0] = in.real();
		_data[1] = in.imag();
		return (*this);
 	}

	KOKKOS_INLINE_FUNCTION
	const T& real() const {
		return _data[0];
	}

	KOKKOS_INLINE_FUNCTION
	T& real() {
		return _data[0];
	}

	KOKKOS_INLINE_FUNCTION
	const T& imag() const {
		return _data[1];
	}

	KOKKOS_INLINE_FUNCTION
	T& imag() {
		return _data[1];
	}


};

template<typename T, typename ViewType, size_t IdxPos, size_t NumDims >
KOKKOS_INLINE_FUNCTION
RComplex<T,ViewType,IdxPos,NumDims>::RComplex( const RComplexLocal<T>& in ) : _data{in.real(),in.imag()} {}

template<typename T, typename ViewType, size_t IdxPos, size_t NumDims >
KOKKOS_INLINE_FUNCTION
RComplex<T,ViewType,IdxPos,NumDims>&
RComplex<T,ViewType,IdxPos,NumDims>::operator=(const RComplexLocal<T>& in ) {
	(*this).real() = in.real();
	(*this).imag() = in.imag();
	return (*this);

}

// Forward declare local type
template<typename T, size_t N>
struct PVectorLocal;

template<typename T, typename ViewType, size_t _N, size_t _IdxPos >
struct PVector {
	ViewType _data;
	std::array<size_t,8> _indices;

	using my_local_type = PVectorLocal<typename LocalType<T>::type, _N>;
	// View Type cannot be default constructed
	PVector() = delete;

	KOKKOS_INLINE_FUNCTION
	PVector(ViewType data_in, std::array<size_t,8> indices) : _data(data_in), _indices(indices){};

	// Initialize from a local type
	PVector(const my_local_type & in) ;

	// Op Assign from a local type
	PVector& operator=(const my_local_type& in);


	// Op Assign from a local type
	PVector& operator=(my_local_type&& in);

	KOKKOS_INLINE_FUNCTION
	auto elem(size_t i) const {
		std::array<size_t,8> new_idx(_indices);
		new_idx[_IdxPos] = i;
		return T(_data, new_idx);
	}

};

template<typename T, size_t _N>
struct PVectorLocal {

	// Will bottom out at a float or double or some such
	using local_subtype = typename LocalType<T>::type;

	// An array of those
	local_subtype _data[_N];

	// Default constructor
	explicit PVectorLocal() {}

	// size query for loops so I can save passing N around
	static constexpr
	KOKKOS_INLINE_FUNCTION
	size_t size(void) { return _N; }


	// Instantiate from a PVector
	template<typename ViewType, size_t _IdxPos>
	KOKKOS_INLINE_FUNCTION
	PVectorLocal(const PVector<T,ViewType,_N,_IdxPos>& in )
	{
		for(size_t i=0; i < _N; ++i) {
			(*this).elem(i) = in.elem(i);
		}
	}

	// Assign from a PVector
	template<typename ViewType, size_t _IdxPos>
	KOKKOS_INLINE_FUNCTION
	PVectorLocal<T,_N>& operator=(const PVector<T,ViewType,_N,_IdxPos>& in) {
		for(size_t i=0; i < _N; ++i) {
			(*this).elem(i) = in.elem(i);
		}
		return *this;
	}

	// Assign from a PVector
	template<typename ViewType, size_t _IdxPos>
	KOKKOS_INLINE_FUNCTION
	PVectorLocal<T,_N>& operator=(PVector<T,ViewType,_N,_IdxPos>&& in) {
		for(size_t i=0; i < _N; ++i) {
			(*this).elem(i) = in.elem(i);
		}
		return *this;
	}

	// Getters and setters
	KOKKOS_INLINE_FUNCTION
	const local_subtype&  elem(size_t i) const {
		return _data[i];
	}

	KOKKOS_INLINE_FUNCTION
	local_subtype&  elem(size_t i)  {
		return _data[i];
	}
};


// Initialize from a local type -- this was predeclared now we can write it
template<typename T, typename ViewType, size_t _N, size_t _IdxPos>
KOKKOS_INLINE_FUNCTION
PVector<T,ViewType,_N,_IdxPos>::PVector(const typename PVector::my_local_type& in) {
	for(size_t i=0; i < _N; ++i) {
		(*this).elem(i) = in.elem(i);
	}
}

// Assign from a local type
template<typename T, typename ViewType, size_t _N, size_t _IdxPos>
KOKKOS_INLINE_FUNCTION
PVector<T,ViewType,_N,_IdxPos>&
PVector<T,ViewType,_N,_IdxPos>::operator=(const PVector::my_local_type& in ){
	for(size_t i=0; i < _N; ++i) {
		(*this).elem(i) = in.elem(i);
	}
}


// Move assign from a local type
template<typename T, typename ViewType, size_t _N, size_t _IdxPos>
KOKKOS_INLINE_FUNCTION
PVector<T,ViewType,_N,_IdxPos>&
PVector<T,ViewType,_N,_IdxPos>::operator=(PVector::my_local_type&& in ){
	for(size_t i=0; i < _N; ++i) {
		(*this).elem(i) = in.elem(i);
	}
}

template<typename T, size_t _N>
struct PMatrixLocal;

template<typename T, typename ViewType, size_t _N, size_t _IdxPos1, size_t _IdxPos2 >
struct PMatrix {
	ViewType _data;
	std::array<size_t,8> _indices;
	using my_local_type = PMatrixLocal<typename LocalType<T>::type, _N>;

	KOKKOS_INLINE_FUNCTION
	PMatrix(ViewType data_in, std::array<size_t,8> indices) : _data(data_in), _indices(indices){};


	// Forward declare, and define after PMatrixLocal is defined
	PMatrix(const my_local_type& in);
	PMatrix& operator=(const my_local_type& in);
	PMatrix& operator=(my_local_type&& in);
	KOKKOS_INLINE_FUNCTION
	auto elem(size_t i, size_t j) const {
		std::array<size_t,8> new_idx(_indices);
		new_idx[_IdxPos1] = i;
		new_idx[_IdxPos2] = j;
		return T(_data, new_idx);
	}

};

template<typename T, size_t N>
struct PMatrixLocal {
	using local_subtype = typename LocalType<T>::type;
	local_subtype _data[N][N];

	static constexpr
	KOKKOS_INLINE_FUNCTION
	size_t size(void) { return N; }

	KOKKOS_INLINE_FUNCTION
	explicit PMatrixLocal() {}

	template<typename ViewType, size_t _IdxPos1, size_t _IdxPos2>
	KOKKOS_INLINE_FUNCTION
	PMatrixLocal(const PMatrix<T,ViewType,N, _IdxPos1, _IdxPos2>& in) {
		for(int j=0; j < N; ++j) {
			for(int i=0; i < N; ++i) {
				(*this).elem(i,j) = in.elem(i,j);
			}
		}
	}

	template<typename ViewType, size_t _IdxPos1, size_t _IdxPos2>
	KOKKOS_INLINE_FUNCTION
	PMatrixLocal& operator=(const PMatrix<T,ViewType,N, _IdxPos1, _IdxPos2>& in) {
		for(int j=0; j < N; ++j) {
			for(int i=0; i < N; ++i) {
				(*this).elem(i,j) = in.elem(i,j);
			}
		}
		return (*this);
	}

	template<typename ViewType, size_t _IdxPos1, size_t _IdxPos2>
	KOKKOS_INLINE_FUNCTION
	PMatrixLocal& operator=(PMatrix<T,ViewType,N, _IdxPos1, _IdxPos2>&& in) {
		for(int j=0; j < N; ++j) {
			for(int i=0; i < N; ++i) {
				(*this).elem(i,j) = in.elem(i,j);
			}
		}
		return (*this);
	}


	KOKKOS_INLINE_FUNCTION
	const local_subtype& elem(size_t i, size_t j) const {
		return _data[i][j];
	}

	KOKKOS_INLINE_FUNCTION
	local_subtype& elem(size_t i, size_t j)  {
		return _data[i][j];
	}
};

template<typename T, typename ViewType, size_t _N, size_t _IdxPos1, size_t _IdxPos2>
KOKKOS_INLINE_FUNCTION
PMatrix<T,ViewType,_N, _IdxPos1, _IdxPos2>::PMatrix( const PMatrix::my_local_type& in) {
	for(int j=0; j < _N; ++j) {
		for(int i=0; i < _N; ++i) {
			(*this).elem(i,j) = in.elem(i,j);
		}
	}

}

template<typename T, typename ViewType, size_t _N, size_t _IdxPos1, size_t _IdxPos2>
KOKKOS_INLINE_FUNCTION
PMatrix<T,ViewType,_N, _IdxPos1, _IdxPos2>&
PMatrix<T,ViewType,_N, _IdxPos1, _IdxPos2>::operator=( const PMatrix::my_local_type& in) {
	for(int j=0; j < _N; ++j) {
		for(int i=0; i < _N; ++i) {
			(*this).elem(i,j) = in.elem(i,j);
		}
	}
	return (*this);
}

template<typename T, typename ViewType, size_t _N, size_t _IdxPos1, size_t _IdxPos2>
KOKKOS_INLINE_FUNCTION
PMatrix<T,ViewType,_N, _IdxPos1, _IdxPos2>&
PMatrix<T,ViewType,_N, _IdxPos1, _IdxPos2>::operator=(PMatrix::my_local_type&& in) {
	for(int j=0; j < _N; ++j) {
		for(int i=0; i < _N; ++i) {
			(*this).elem(i,j) = in.elem(i,j);
		}
	}
	return (*this);
}

// ----- Terminal types
template<>
struct LocalType<float> {
	using type=float;
};

template<>
struct LocalType<double> {
	using type=double;
};


template<>
struct LocalType<short> {
	using type=short;
};

template<>
struct LocalType<int> {
	using type=int;
};

template<>
struct LocalType<size_t> {
	using type=size_t;
};

template<>
struct LocalType<ptrdiff_t> {
	using type=ptrdiff_t;
};

// The local type of a view type is its local type at this level templated
// on the local type of its substructure.
// is its local type templated on the terminal type (float etc0)
template<typename T, typename ViewType, size_t NumDims>
struct LocalType< RScalar<T, ViewType, NumDims> > {
	using type =  RScalarLocal<typename LocalType<T>::type>;
};

template<typename T, typename ViewType, size_t IdxPos, size_t NumDims>
struct LocalType< RComplex<T, ViewType,IdxPos,NumDims> > {
	using type = RComplexLocal<typename LocalType<T>::type>;
};


// THe local type for a local type is itself
template<typename T>
struct LocalType< RScalarLocal<T> > {
	using type =  RScalarLocal<T>;
};

template<typename T>
struct LocalType< RComplexLocal<T> > {
	using type = RComplexLocal<T>;
};

// The local type for a view based type is the local type on its level templated
// on the local types of its subtypes
template<typename T, typename ViewType, size_t N, size_t IdxPos1>
struct LocalType< PVector<T, ViewType, N, IdxPos1> > {
	using type = PVectorLocal<typename LocalType<T>::type,N>;
};

// The local tpye for a local type is itself
template<typename T, size_t N>
struct LocalType< PVectorLocal<T,N> > {
	using type = PVectorLocal<T,N>;
};

// The local type for a view based type is its local type templated on its local subtypes
template<typename T, typename ViewType, size_t N, size_t IdxPos1, size_t IdxPos2>
struct LocalType< PMatrix<T, ViewType, N, IdxPos1, IdxPos2 > > {
	using type = PMatrixLocal<typename LocalType<T>::type,N>;
};

// The local type for a local tpe is itself
template<typename T, size_t N>
struct LocalType< PMatrixLocal<T,N> > {
	using type = PMatrixLocal<T,N>;
};

// Base Type Traits:

template<typename T>
struct BaseType;

template<>
struct BaseType<float> {
	using type=float;
};

template<>
struct BaseType<double> {
	using type=double;
};


template<>
struct BaseType<short> {
	using type=short;
};

template<>
struct BaseType<int> {
	using type=int;
};

template<>
struct BaseType<size_t> {
	using type=size_t;
};

template<>
struct BaseType<ptrdiff_t> {
	using type=ptrdiff_t;
};

template<typename T, typename ViewType, size_t NumDims>
struct BaseType< RScalar<T,ViewType,NumDims> > {
	using type = typename BaseType<T>::type;
};

template<typename T>
struct BaseType< RScalarLocal<T> >  {
	using type = typename BaseType<T>::type;
};

template<typename T, typename ViewType, size_t IdxPos, size_t NumDims>
struct BaseType< RComplex<T,ViewType,IdxPos,NumDims> > {
	using type = typename BaseType<T>::type;
};

template<typename T>
struct BaseType< RComplexLocal<T> > {
	using type = typename BaseType<T>::type;
};

template<typename T, typename ViewType, size_t N, size_t IdxPos>
struct BaseType< PVector<T,ViewType,N,IdxPos> > {
	using type = typename BaseType<T>::type;
};

template<typename T, size_t N>
struct BaseType< PVectorLocal<T,N> > {
	using type = typename BaseType<T>::type;
};

template<typename T, typename ViewType, size_t N, size_t IdxPos1, size_t IdxPos2>
struct BaseType< PMatrix<T,ViewType,N,IdxPos1,IdxPos2> > {
	using type = typename BaseType<T>::type;
};

template<typename T, size_t N>
struct BaseType< PMatrixLocal<T,N> > {
	using type = typename BaseType<T>::type;
};
//
//
//
// OLattice type
//
//
template<typename T, typename ViewType, size_t _IdxPos1>
struct OLattice {

	ViewType _data;
	std::size_t _n_elem;
	KOKKOS_INLINE_FUNCTION

	OLattice(size_t n_elem) : _data("internal", n_elem), _n_elem(n_elem) {}

	KOKKOS_INLINE_FUNCTION
	OLattice(ViewType t) : _data(t), _n_elem(t.extent(0)) {}

	KOKKOS_INLINE_FUNCTION
	auto elem(size_t i) const {
		std::array<size_t,8> index{0,0,0,0, 0,0,0,0};
		index[_IdxPos1 ] = i;
		return T(_data, index );
	}

	KOKKOS_INLINE_FUNCTION
	size_t num_elem() const {
		return _n_elem;
	}
};

template<typename T, typename ViewType, size_t _IdxPos1>
struct BaseType< OLattice<T,ViewType,_IdxPos1> > {
	using type = typename BaseType<T>::type;
};

}// namespace
