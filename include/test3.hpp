/*
 * test3.hpp
 *
 *  Created on: Mar 24, 2020
 *      Author: bjoo
 */

#pragma once

#include "Kokkos_Core.hpp"
namespace Playground {

using KokkosIndices = Kokkos::Array<size_t,8>;

// Type Traits for Type->LocalType conversions
//
//
template<typename T>
struct LocalType;

#if 0
#define IF_CONSTEXPR    if constexpr
#else
#define IF_CONSTEXPR    if
#endif

  template<typename ViewType, size_t N>
  struct Indexer;

  template<typename ViewType>
  struct Indexer<ViewType,1> {
    
    KOKKOS_INLINE_FUNCTION
    auto& operator()( const ViewType& v, const KokkosIndices& _indices ) const  {
      return v(_indices[0]);
    }
  };

  template<typename ViewType>
  struct Indexer<ViewType,2> {
   
    KOKKOS_INLINE_FUNCTION 
    auto& operator()( const ViewType& v, const KokkosIndices& _indices) const {
          return v( _indices[0], _indices[1] );
    }
  };

  template<typename ViewType>
  struct Indexer<ViewType,3> {
    KOKKOS_INLINE_FUNCTION 
    auto& operator()( const ViewType& v, const KokkosIndices& _indices ) const {
      return v( _indices[0], _indices[1], _indices[2] );
    }
  };

  template<typename ViewType>
  struct Indexer<ViewType,4> {
    KOKKOS_INLINE_FUNCTION
    auto& operator()( const ViewType& v, const KokkosIndices& _indices ) const {
    return v( _indices[0], _indices[1], _indices[2], _indices[3] );
    }
  };

  template<typename ViewType>
  struct Indexer<ViewType,5> {
    KOKKOS_INLINE_FUNCTION
    auto& operator()( const ViewType& v, const KokkosIndices& _indices ) const {
    return v( _indices[0], _indices[1], _indices[2], _indices[3],
		  _indices[4]);

    }
  };

  template<typename ViewType>
  struct Indexer<ViewType,6> {
    KOKKOS_INLINE_FUNCTION
    auto& operator()( const ViewType& v, const KokkosIndices& _indices ) const {
      return v( _indices[0], _indices[1], _indices[2], _indices[3],
		  _indices[4], _indices[5]);
    }
  };

  template<typename ViewType>
  struct Indexer<ViewType,7> {
    KOKKOS_INLINE_FUNCTION
    auto& operator()( const ViewType& v,  const KokkosIndices& _indices) const {
    return v( _indices[0], _indices[1], _indices[2], _indices[3],
		  _indices[4], _indices[5], _indices[6]);
    }
  };

  template<typename ViewType>
  struct Indexer<ViewType,8> {
    KOKKOS_INLINE_FUNCTION
    auto& operator()( const ViewType& v, const KokkosIndices& _indices)  const {
    return v( _indices[0], _indices[1], _indices[2], _indices[3],
		  _indices[4], _indices[5], _indices[6], _indices[7]);
    }
  };
  


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
template<typename T, typename ViewType, size_t ParentNumDims=1>
struct RScalar {

  // This is a view which should be ok on device
  ViewType _data;

  // The indices already frozen
  KokkosIndices _indices;
  Indexer<ViewType, ParentNumDims> _idx;
  
  // Construct from a view and fixed indices
  KOKKOS_INLINE_FUNCTION
  RScalar(ViewType data_in, KokkosIndices indices) : _data(data_in), _indices(indices), _idx() {};
  
  // Construct from a view (assume indices are all 0) -- most for testing purposes
  KOKKOS_INLINE_FUNCTION
  RScalar(ViewType data_in) : _data(data_in), _indices{0,0,0,0,0,0,0,0}, _idx(){}
  
  // Construct from RScalarLocal
  // Forward declare since RScalarLocal is incomplete
  KOKKOS_FUNCTION
  RScalar(const RScalarLocal<T>& t);

  // Op assign from
  KOKKOS_FUNCTION
  RScalar& operator=(const RScalarLocal<T>& t );

  // View types cannot be default initialized
  RScalar() = delete;

  KOKKOS_INLINE_FUNCTION
  T& elem() const {
    return _idx(_data,_indices);
  }

};


// A thread local RScalar
// with compact storage
template<typename T>
struct RScalarLocal {

	// The Data
	T _data;

	// Use array type following Kokkos convention
	using array_type = T;

	template<typename ViewType, size_t NumDims=1>
	using GlobalType = RScalar<T,ViewType,NumDims>;

	KOKKOS_INLINE_FUNCTION
	RScalarLocal(void) {}

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

template<typename T, typename ViewType, size_t NumDims>
KOKKOS_INLINE_FUNCTION
RScalar<T,ViewType,NumDims>::RScalar( const RScalarLocal<T>& local_in ) {
	(*this).elem() = local_in.elem();
}

template<typename T, typename ViewType,  size_t NumDims>
KOKKOS_INLINE_FUNCTION
RScalar<T,ViewType,NumDims>& RScalar<T,ViewType,NumDims>::operator=(const RScalarLocal<T>& local_in ) {
	(*this).elem() = local_in.elem();
	return (*this);
}

template<typename T>
struct RComplexLocal;

template<typename T, typename ViewType, size_t ParentNumDims >
struct RComplex {
	ViewType _data;
	KokkosIndices _indices;
	Indexer<ViewType, ParentNumDims+1> _idx;       // Num dims is of the parent. I have 1 extra
	

	KOKKOS_INLINE_FUNCTION
	RComplex(ViewType data_in, KokkosIndices indices) : _data(data_in), _indices(indices), _idx() {};

	// View Types Cannot be Default constructed
	RComplex() = delete;

	// Forward declare these since RComplexLocal is only
	// forward declared
	KOKKOS_FUNCTION
	RComplex(const RComplexLocal<T>& in);

	KOKKOS_FUNCTION
	RComplex& operator=(const RComplexLocal<T>& in);

	// Getters and Setters
	KOKKOS_INLINE_FUNCTION
	T& real() const {
		KokkosIndices new_idx(_indices);
		new_idx[ ParentNumDims ] = 0;
		return _idx(_data, new_idx );
	}

	KOKKOS_INLINE_FUNCTION
	T& imag() const {
		KokkosIndices new_idx(_indices);
		new_idx[ ParentNumDims ] = 1;
		return _idx(_data, new_idx );
	}

};

// ThreadLocal Type
template<typename T >
struct RComplexLocal {
	T _data[2];

	using array_type = T[2];

	// IdxPos is the last index of the parent
	// NumDims is the total number of dimensions
	template<typename ViewType, size_t ParentNumDims >
	using GlobalType = RComplex<T,ViewType,ParentNumDims>;

	KOKKOS_INLINE_FUNCTION
	RComplexLocal() {}

	KOKKOS_INLINE_FUNCTION
	RComplexLocal( const T& re, const T& im ) : _data{re,im} {}

	template<typename ViewType, size_t ParentNumDims>
	KOKKOS_INLINE_FUNCTION
	RComplexLocal( const RComplex<T,ViewType,ParentNumDims>& in) : _data{ in.real(), in.imag() } {}

	template<typename ViewType, size_t ParentNumDims >
	KOKKOS_INLINE_FUNCTION
	RComplexLocal& operator=(const RComplex<T,ViewType,ParentNumDims>& in) {
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

template<typename T, typename ViewType,  size_t ParentNumDims >
KOKKOS_INLINE_FUNCTION
RComplex<T,ViewType,ParentNumDims>::RComplex( const RComplexLocal<T>& in ) : _data{in.real(),in.imag()} {}

template<typename T, typename ViewType, size_t ParentNumDims >
KOKKOS_INLINE_FUNCTION
RComplex<T,ViewType,ParentNumDims>&
RComplex<T,ViewType,ParentNumDims>::operator=(const RComplexLocal<T>& in ) {
	(*this).real() = in.real();
	(*this).imag() = in.imag();
	return (*this);

}


// Forward declar  local type
template<typename T>
struct PScalarLocal;

template<typename T, typename ViewType, size_t ParentNumDims >
struct PScalar {
	ViewType _data;
	KokkosIndices _indices;
	using my_local_type = PScalarLocal<typename LocalType<T>::type>;

	PScalar() = delete;

	KOKKOS_INLINE_FUNCTION
	PScalar(ViewType data_in, KokkosIndices indices) : _data(data_in), _indices(indices) {}

	// Forward declare Initialize from local type
	KOKKOS_FUNCTION
	PScalar(const my_local_type & in );

	KOKKOS_FUNCTION
	PScalar& operator=(const my_local_type& in);

	// Op Assign from a local type
	KOKKOS_FUNCTION
	PScalar& operator=(my_local_type&& in);

	KOKKOS_INLINE_FUNCTION
	auto elem() const {
		// Scalar so no increase in the dimension
		using Ret_type = typename T::template GlobalType<ViewType, ParentNumDims>;
		return  Ret_type(_data, _indices );
	}
};

// A thread local RScalar
// with compact storage
template<typename T>
struct PScalarLocal {

	// The Data
	T _data;

	// Use array type following Kokkos convention
	using array_type = typename T::array_type;

	template<typename ViewType, size_t NumDims=1>
	using GlobalType = PScalar<T,ViewType,NumDims>;

	KOKKOS_INLINE_FUNCTION
	PScalarLocal(void) {}

	// Initialize with the data
	KOKKOS_INLINE_FUNCTION
	PScalarLocal(const T& data_in) : _data(data_in) {};


	// Init from an RScalarView (this is known
	template<typename ViewType, size_t N>
	PScalarLocal(const PScalar<T,ViewType,N>& view_in) : _data(view_in.elem()) {};


	template<typename ViewType, size_t N>
	KOKKOS_INLINE_FUNCTION
	PScalarLocal<T>& operator=(const PScalar<T,ViewType,N>& view_in) {
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

template<typename T, typename ViewType, size_t NumDims>
KOKKOS_INLINE_FUNCTION
PScalar<T,ViewType,NumDims>::PScalar( const PScalar::my_local_type& local_in ) {
	(*this).elem() = local_in.elem();
}

template<typename T, typename ViewType,  size_t NumDims>
KOKKOS_INLINE_FUNCTION
PScalar<T,ViewType,NumDims>& PScalar<T,ViewType,NumDims>::operator=(const PScalar::my_local_type& local_in ) {
	(*this).elem() = local_in.elem();
	return (*this);
}

template<typename T, typename ViewType,  size_t NumDims>
KOKKOS_INLINE_FUNCTION
PScalar<T,ViewType,NumDims>& PScalar<T,ViewType,NumDims>::operator=(PScalar::my_local_type&& local_in ) {
	(*this).elem() = local_in.elem();
	return (*this);
}


// Forward declare local type
template<typename T, size_t N>
struct PVectorLocal;

template<typename T, typename ViewType, size_t _N, size_t ParentNumDims >
struct PVector {
	ViewType _data;
	KokkosIndices _indices;


	using my_local_type = PVectorLocal<typename LocalType<T>::type, _N>;



	// View Type cannot be default constructed
	PVector() = delete;

	KOKKOS_INLINE_FUNCTION
	PVector(ViewType data_in, KokkosIndices indices) : _data(data_in), _indices(indices){};

	// Initialize from a local type
	KOKKOS_FUNCTION
	PVector(const my_local_type & in) ;

	// Op Assign from a local type
	KOKKOS_FUNCTION
	PVector& operator=(const my_local_type& in);


	// Op Assign from a local type
	KOKKOS_FUNCTION
	PVector& operator=(my_local_type&& in);

	KOKKOS_INLINE_FUNCTION
	auto elem(size_t i) const {
		KokkosIndices new_idx(_indices);
		new_idx[ ParentNumDims ] = i;
		using Ret_type = typename T::template GlobalType<ViewType, ParentNumDims+1>;
		return  Ret_type(_data, new_idx );
	}

};

template<typename T, size_t _N>
struct PVectorLocal {

	template<typename ViewType, size_t ParentNumDims >
	using GlobalType =  PVector<T, ViewType, _N, ParentNumDims>;

	using array_type = typename T::array_type[_N];

	// Will bottom out at a float or double or some such
	using local_subtype = typename LocalType<T>::type;

	// An array of those
	local_subtype _data[_N];

	// Default constructor
	KOKKOS_INLINE_FUNCTION
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
template<typename T, typename ViewType, size_t _N, size_t ParentNumDims>
KOKKOS_INLINE_FUNCTION
PVector<T,ViewType,_N, ParentNumDims>::PVector(const typename PVector::my_local_type& in) {
	for(size_t i=0; i < _N; ++i) {
		(*this).elem(i) = in.elem(i);
	}
}

// Assign from a local type
template<typename T, typename ViewType, size_t _N, size_t ParentNumDims>
KOKKOS_INLINE_FUNCTION
PVector<T,ViewType,_N,ParentNumDims>&
PVector<T,ViewType,_N,ParentNumDims>::operator=(const PVector::my_local_type& in ){
	for(size_t i=0; i < _N; ++i) {
		(*this).elem(i) = in.elem(i);
	}
	return (*this);
}


// Move assign from a local type
template<typename T, typename ViewType, size_t _N, size_t ParentNumDims>
KOKKOS_INLINE_FUNCTION
PVector<T,ViewType,_N, ParentNumDims>&
PVector<T,ViewType,_N, ParentNumDims>::operator=(PVector::my_local_type&& in ){
	for(size_t i=0; i < _N; ++i) {
		(*this).elem(i) = in.elem(i);
	}
	return (*this);
}

template<typename T, size_t _N>
struct PMatrixLocal;

template<typename T, typename ViewType, size_t _N, size_t ParentNumDims >
struct PMatrix {

	ViewType _data;
	KokkosIndices _indices;
	using my_local_type = PMatrixLocal<typename LocalType<T>::type, _N>;

	KOKKOS_INLINE_FUNCTION
	PMatrix(ViewType data_in, KokkosIndices indices) : _data(data_in), _indices(indices){};

	PMatrix() = delete;

	// Forward declare, and define after PMatrixLocal is defined
	KOKKOS_FUNCTION
	PMatrix(const my_local_type& in);

	KOKKOS_FUNCTION
	PMatrix& operator=(const my_local_type& in);

	KOKKOS_FUNCTION
	PMatrix& operator=(my_local_type&& in);

	KOKKOS_INLINE_FUNCTION
	auto elem(size_t i, size_t j) const {
		KokkosIndices new_idx(_indices);
		new_idx[ ParentNumDims ] = i;
		new_idx[ ParentNumDims +1 ] = j;
		using Ret_type = typename T::template GlobalType<ViewType,ParentNumDims+2>;
		return Ret_type(_data, new_idx);
	}

};

template<typename T, size_t N>
struct PMatrixLocal {
	using local_subtype = typename LocalType<T>::type;
	using array_type = typename T::array_type[N][N];

	local_subtype _data[N][N];

	template<typename ViewType, size_t ParentNumDims >
	using GlobalType = PMatrix<T, ViewType, N, ParentNumDims>;

	static constexpr
	KOKKOS_INLINE_FUNCTION
	size_t size(void) { return N; }

	KOKKOS_INLINE_FUNCTION
	explicit PMatrixLocal() {}

	template<typename ViewType, size_t ParentNumDims>
	KOKKOS_INLINE_FUNCTION
	PMatrixLocal(const PMatrix<T,ViewType,N, ParentNumDims>& in) {
		for(int j=0; j < N; ++j) {
			for(int i=0; i < N; ++i) {
				(*this).elem(i,j) = in.elem(i,j);
			}
		}
	}

	template<typename ViewType, size_t ParentNumDims>
	KOKKOS_INLINE_FUNCTION
	PMatrixLocal& operator=(const PMatrix<T,ViewType,N, ParentNumDims>& in) {
		for(int j=0; j < N; ++j) {
			for(int i=0; i < N; ++i) {
				(*this).elem(i,j) = in.elem(i,j);
			}
		}
		return (*this);
	}

	template<typename ViewType, size_t ParentNumDims>
	KOKKOS_INLINE_FUNCTION
	PMatrixLocal& operator=(PMatrix<T,ViewType,N,ParentNumDims>&& in) {
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

template<typename T, typename ViewType, size_t _N, size_t ParentNumDims>
KOKKOS_INLINE_FUNCTION
PMatrix<T,ViewType,_N, ParentNumDims>::PMatrix( const PMatrix::my_local_type& in) {
	for(int j=0; j < _N; ++j) {
		for(int i=0; i < _N; ++i) {
			(*this).elem(i,j) = in.elem(i,j);
		}
	}

}

template<typename T, typename ViewType, size_t _N, size_t ParentNumDims>
KOKKOS_INLINE_FUNCTION
PMatrix<T,ViewType,_N, ParentNumDims>&
PMatrix<T,ViewType,_N, ParentNumDims>::operator=( const PMatrix::my_local_type& in) {
	for(int j=0; j < _N; ++j) {
		for(int i=0; i < _N; ++i) {
			(*this).elem(i,j) = in.elem(i,j);
		}
	}
	return (*this);
}

template<typename T, typename ViewType, size_t _N, size_t ParentNumDims>
KOKKOS_INLINE_FUNCTION
PMatrix<T,ViewType,_N, ParentNumDims>&
PMatrix<T,ViewType,_N, ParentNumDims>::operator=(PMatrix::my_local_type&& in) {
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

template<typename T, typename ViewType, size_t ParentNumDims>
struct LocalType< RComplex<T, ViewType, ParentNumDims> > {
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
template<typename T, typename ViewType, size_t ParentNumDims>
struct LocalType< PScalar<T, ViewType, ParentNumDims> > {
	using type = PScalarLocal<typename LocalType<T>::type>;
};

// The local tpye for a local type is itself
template<typename T>
struct LocalType< PScalarLocal<T> > {
	using type = PScalarLocal<T>;
};


// The local type for a view based type is the local type on its level templated
// on the local types of its subtypes
template<typename T, typename ViewType, size_t N, size_t ParentNumDims>
struct LocalType< PVector<T, ViewType, N, ParentNumDims> > {
	using type = PVectorLocal<typename LocalType<T>::type,N>;
};

// The local tpye for a local type is itself
template<typename T, size_t N>
struct LocalType< PVectorLocal<T,N> > {
	using type = PVectorLocal<T,N>;
};

// The local type for a view based type is its local type templated on its local subtypes
template<typename T, typename ViewType, size_t N, size_t ParentNumDims>
struct LocalType< PMatrix<T, ViewType, N, ParentNumDims > > {
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

template<typename T, typename ViewType, size_t ParentNumDims>
struct BaseType< RScalar<T,ViewType,ParentNumDims> > {
	using type = typename BaseType<T>::type;
};

template<typename T>
struct BaseType< RScalarLocal<T> >  {
	using type = typename BaseType<T>::type;
};

template<typename T, typename ViewType, size_t ParentNumDims>
struct BaseType< RComplex<T,ViewType,ParentNumDims> > {
	using type = typename BaseType<T>::type;
};

template<typename T>
struct BaseType< RComplexLocal<T> > {
	using type = typename BaseType<T>::type;
};

template<typename T, typename ViewType, size_t ParentNumDims>
struct BaseType< PScalar<T,ViewType,ParentNumDims> > {
	using type = typename BaseType<T>::type;
};

template<typename T>
struct BaseType< PScalarLocal<T> >  {
	using type = typename BaseType<T>::type;
};

template<typename T, typename ViewType, size_t N, size_t ParentNumDims>
struct BaseType< PVector<T,ViewType,N,ParentNumDims> > {
	using type = typename BaseType<T>::type;
};

template<typename T, size_t N>
struct BaseType< PVectorLocal<T,N> > {
	using type = typename BaseType<T>::type;
};

template<typename T, typename ViewType, size_t N, size_t ParentNumDims>
struct BaseType< PMatrix<T,ViewType,N,ParentNumDims> > {
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
template<typename T, class S=Kokkos::DefaultExecutionSpace::memory_space>
struct OLattice {

	// Need to compute ViewType
	using array_type = typename T::array_type*;
	using ViewType = Kokkos::View<array_type, S>;
	ViewType _data;
	std::size_t _n_elem;

	OLattice(size_t n_elem) : _data("olattice_data", n_elem), _n_elem(n_elem) {}

	OLattice(ViewType t) : _data(t), _n_elem(t.extent(0)) {}

	KOKKOS_INLINE_FUNCTION
	auto elem(size_t i) const {
		KokkosIndices index{i,0,0,0, 0,0,0,0};

		using Ret_type = typename T::template GlobalType<ViewType,1>;
		return Ret_type(_data, index );
  }

  KOKKOS_INLINE_FUNCTION
  size_t num_elem() const {
    return _n_elem;
  }
};

template<typename T,typename MemSpace>
struct BaseType< OLattice<T,MemSpace> > {
	using type = typename BaseType<T>::type;
};

}// namespace
