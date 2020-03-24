/*
 * test2.hpp
 *
 *  Created on: Mar 19, 2020
 *      Author: bjoo
 */

#pragma once

#include <Kokkos_Core.hpp>

#define DEBUG
#ifdef DEBUG
#include <iostream>
#define DEBUG_MSG(str)   { std::cout << "DEBUG: " << str << std::endl << std::flush; }
#else
#define DEBUG_MSG(str)   {}
#endif

namespace Playground {

using Index = size_t;


template<typename T>
struct base_type;

template<typename T>
struct num_dims;

// FLOAT
template<>
struct base_type<float> {
	using type_t = float;
};

template<>
struct num_dims<float> {
	static constexpr Index value=0;
};

// DOUBLE
template<>
struct base_type<double> {
	using type_t = double;
};

template<>
struct num_dims<double> {
	static constexpr Index value=0;
};


#if 0
template<typename T, Index N,
	typename MemorySpace=Kokkos::DefaultExecutionSpace::memory_space,
	typename KokkosLayout=Kokkos::DefaultExecutionSpace::array_layout>
class ILattice {
public:
	using base_type = base_type<T;
	using array_type = T[N];
	using view_type = typename Kokkos::View<array_type,KokkosLayout,MemorySpace>;


	~ILattice() = default;
	ILattice(const ILattice& p) = default;
	ILattice(ILattice&& p) = default;
	ILattice& operator=(const ILattice& p) = default;
};

template<typename T,
	typename MemorySpace=Kokkos::DefaultExecutionSpace::memory_space,
	typename KokkosLayout=Kokkos::DefaultExecutionSpace::array_layout>
struct RScalar {

	using array_type = typename T;
	using base_type = typename base_type<T>::type_t;
	using view_type = typename Kokkos::View<array_type,KokkosLayout,MemorySpace>;

	view_type _data_view;
	explicit RScalar(view_type in) : _data_view(in) {}
	T& elem() const {
		_data_view(0);
	}
};

template<typename T, typename MemSpace, typename Layout>
struct base_type< RScalar<T,MemSpace,Layout> > {
	using type_t = typename base_type<T>::type_t;
};

template<typename T, typename MemSpace, typename Layout>
struct num_dims< RScalar<T,MemSpace,Layout> >
{
	static constexpr Index value = num_dims<T>::value;
};

template<typename T,
	typename MemorySpace=Kokkos::DefaultExecutionSpace::memory_space,
	typename KokkosLayout=Kokkos::DefaultExecutionSpace::array_layout>
struct RComplex {

	using array_type = typename T::array_type[2];
	using base_type = typename T::base_type;
	using view_type = typename Kokkos::View<array_type,KokkosLayout,MemorySpace>;

};
#endif

template<typename T, Index N,
	typename MemorySpace=Kokkos::DefaultExecutionSpace::memory_space,
	typename KokkosLayout=Kokkos::DefaultExecutionSpace::array_layout>
struct PVector {

	using array_type = T[N];
	using base_type = typename base_type<T>::type_t;
	using view_type = typename Kokkos::View<array_type,KokkosLayout,MemorySpace>;

	view_type _data_view;

	explicit PVector(view_type view_in) : _data_view( view_in ) {}

	T& elem(Index i) const {
		return _data_view(i);
	}

};

template<typename T, Index N, typename MemSpace, typename Layout>
struct base_type< PVector<T,N,MemSpace,Layout> > {
	using type_t = typename base_type<T>::type_t;
};

template<typename T, Index N, typename MemSpace, typename Layout>
struct num_dims< PVector<T,N,MemSpace,Layout> >
{
	static constexpr Index value = 1 + num_dims<T>::value;
};

template<typename T, Index N,
	typename MemorySpace=Kokkos::DefaultExecutionSpace::memory_space,
	typename KokkosLayout=Kokkos::DefaultExecutionSpace::array_layout>
struct PMatrix {


	using  array_type = typename T::array_type[N][N];
	using  base_type = typename base_type<T>::type_t;
	using  view_type = typename Kokkos::View<array_type,KokkosLayout,MemorySpace>;

	view_type _data_view;

	explicit PMatrix(view_type view_in) : _data_view( view_in ) {}

	T elem(Index i, Index j) const {
		if constexpr( num_dims<T>::value == 1 ) {
			return T( Kokkos::subview(_data_view,i,j,Kokkos::ALL));
		}
		if constexpr( num_dims<T>::value == 2 ) {
			return T( Kokkos::subview(_data_view,i,j,Kokkos::ALL, Kokkos::ALL));
		}
	}

};

template<typename T, Index N, typename MemSpace, typename Layout>
struct base_type< PMatrix<T,N,MemSpace,Layout> > {
	using type_t = typename base_type<T>::type_t;
};

template<typename T, Index N, typename MemSpace, typename Layout>
struct num_dims< PMatrix<T,N,MemSpace,Layout> >
{
	static constexpr Index value = 2 + num_dims<T>::value;
};



// Assumption: T itself is a recursive view, of
// Type PMatrix, PVector, RScalar, or RComplex
//
template<typename T,
	typename MemorySpace=Kokkos::DefaultExecutionSpace::memory_space,
	typename KokkosLayout=Kokkos::DefaultExecutionSpace::array_layout>
struct OLattice {
	const Index _n_sites;
	using array_type = typename T::array_type*;
	using base_type = typename T::base_type;
	using view_type = typename Kokkos::View<array_type,KokkosLayout,MemorySpace>;


	// Doesn't need to be mutable... (I think internally the data in a View already is)
	view_type _data_view;

	// Allocate:
	// NB: This constructor will:
	//   a) Allocate memory for the view
	//   b) Call default constructor on each element using 'parallel for'

	OLattice(const Index n_sites) : _n_sites(n_sites), _data_view("OLatticeData", n_sites) {}

	// Copy
	OLattice(const OLattice& in) : _n_sites(in._n_sites), _data_view(in._data_view) {}

	//Move
	OLattice(OLattice&& in) : _n_sites(in._n_sites), _data_view(std::move(in._data_view)) {}

	OLattice& operator=(const OLattice& in) {
		_n_sites = in._n_sites;
		_data_view = in._data_view;
	}

	constexpr
	KOKKOS_INLINE_FUNCTION
	T& elem(const Index i) const {
		if constexpr ( num_dims<T>::value == 1 ) {
			return T(  Kokkos::subview(_data_view, i, Kokkos::ALL));
		}

		if constexpr ( num_dims<T>::value == 2 ) {
			return T(  Kokkos::subview(_data_view, i, Kokkos::ALL, Kokkos::ALL));
		}
	}
};

}


