/*
 * test2.hpp
 *
 *  Created on: Mar 19, 2020
 *      Author: bjoo
 */

#pragma once
#include <Kokkos_Core.hpp>


namespace Playground {

using Index = size_t;



template<typename T, Index N,
	typename MemorySpace=Kokkos::DefaultExecutionSpace::memory_space,
	typename KokkosLayout=Kokkos::DefaultExecutionSpace::array_layout>
struct ILattice {

	static constexpr
	KOKKOS_INLINE_FUNCTION
	Index num_dims() {
		return 1;
	}

	static constexpr
	KOKKOS_INLINE_FUNCTION
	Index num_elem() {
		return N;
	}

	using array_type = T[N];
	using base_type = T;
	using view_type = typename Kokkos::View<array_type,KokkosLayout,MemorySpace>;

	view_type _data;


	// Default creation spawns a view
	explicit ILattice() = default;

	// Custom creation allows passing in a view
	explicit ILattice(view_type subview_of_parent) : _data(subview_of_parent) {}


	T& elem(Index i) const {
		return _data(i);
	}


};


template<typename T,
	typename MemorySpace=Kokkos::DefaultExecutionSpace::memory_space,
	typename KokkosLayout=Kokkos::DefaultExecutionSpace::array_layout>
struct RScalar {
	static constexpr
	KOKKOS_INLINE_FUNCTION
	Index num_dims() {
		// Not introducing an extra dimension of length=1
		return 1+T::num_dims();
	}

	static constexpr
	KOKKOS_INLINE_FUNCTION
	Index num_elem() {
		// The num elem in T
		return 1*T::num_elem();
	}

	using array_type = typename T::array_type[1];
	using base_type = typename T::base_type;
	using view_type = typename Kokkos::View<array_type,KokkosLayout,MemorySpace>;

	T _data;

	explicit RScalar() = default;
	explicit RScalar(view_type subview_of_parent) :
					_data(Kokkos::subview(subview_of_parent,0,Kokkos::ALL)) {}

	KOKKOS_INLINE_FUNCTION
	const T& elem() const {
		return _data;
	}

	T& elem() {
		return _data;
	}
};

template<typename T,
	typename MemorySpace=Kokkos::DefaultExecutionSpace::memory_space,
	typename KokkosLayout=Kokkos::DefaultExecutionSpace::array_layout>
struct RComplex {
	static constexpr
	KOKKOS_INLINE_FUNCTION
	Index num_dims() {
		return 1+T::num_dims();
	}

	static constexpr
	KOKKOS_INLINE_FUNCTION
	Index num_elem() {
		return 2*T::num_elem();
	}

	using array_type = typename T::array_type[2];
	using base_type = typename T::base_type;
	using view_type = typename Kokkos::View<array_type,KokkosLayout,MemorySpace>;

	mutable T _data[2];

	explicit RComplex() = default;
	explicit RComplex(view_type in) : _data{
			T(Kokkos::subview(in,0,Kokkos::ALL)),
			T(Kokkos::subview(in,1,Kokkos::ALL)) } {}


	T& real() const { return _data[0]; }
	T& imag() const { return _data[1]; }

};

template<typename T, Index N,
	typename MemorySpace=Kokkos::DefaultExecutionSpace::memory_space,
	typename KokkosLayout=Kokkos::DefaultExecutionSpace::array_layout>
struct PVector {

	static constexpr
	KOKKOS_INLINE_FUNCTION
	Index num_dims() {
		return 1+T::num_dims();
	}

	static constexpr
	KOKKOS_INLINE_FUNCTION
	Index num_elem() {
		return N*T::num_elem();
	}

	mutable T _data[N];

	using array_type = typename T::array_type[N];
	using base_type = typename T::base_type;
	using view_type = typename Kokkos::View<array_type,KokkosLayout,MemorySpace>;

	explicit PVector() = default;

	explicit PVector(view_type in) {
		for(int i=0; i < N; ++i) {
			// This is ugly -- use if constexpr to pick the right one.
			//
			if constexpr( T::num_dims() == 2 ) {
				_data[i] = T( Kokkos::subview(in,i,Kokkos::ALL, Kokkos::ALL));
			}

			if constexpr( T::num_dims() == 3 ) {
				_data[i] = T( Kokkos::subview(in,i,Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
			}

			if constexpr( T::num_dims() == 4 ) {
				_data[i] = T( Kokkos::subview(in,i,Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
												   Kokkos::ALL));
			}

			if constexpr( T::num_dims() == 5 ) {
				_data[i] = T( Kokkos::subview(in,i,Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
												   Kokkos::ALL, Kokkos::ALL));
			}

			if constexpr( T::num_dims() == 6 ) {
				_data[i] = T( Kokkos::subview(in,i,Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
												   Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
			}
			if constexpr( T::num_dims() == 7 ) {
				_data[i] = T( Kokkos::subview(in,i,Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
												   Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
												   Kokkos::ALL));
			}
		}}


	T& elem(Index i) const { return _data[i]; }
};

template<typename T, Index N,
	typename MemorySpace=Kokkos::DefaultExecutionSpace::memory_space,
	typename KokkosLayout=Kokkos::DefaultExecutionSpace::array_layout>
struct PMatrix {

	static constexpr
	KOKKOS_INLINE_FUNCTION
	Index num_dims() {
		return 2+T::num_dims();
	}

	static constexpr
	KOKKOS_INLINE_FUNCTION
	Index num_elem() {
		return N*N*T::num_elem();
	}

	mutable T _data[N][N];

	using array_type = typename T::array_type[N][N];
	using base_type = typename T::base_type;
	using view_type = typename Kokkos::View<array_type,KokkosLayout,MemorySpace>;

	explicit PMatrix() = default;
	explicit PMatrix(view_type in) {
		for(int j=0; j < N; ++j) {
		 for(int i=0; i < N; ++i) {
			// This is ugly -- use if constexpr to pick the right one.
			//
			if constexpr( T::num_dims() == 2 ) {
				_data[i][j] = T( Kokkos::subview(in,i,j,Kokkos::ALL, Kokkos::ALL));
			}

			if constexpr( T::num_dims() == 3 ) {
				_data[i][j] = T( Kokkos::subview(in,i,j,Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
			}

			if constexpr( T::num_dims() == 4 ) {
				_data[i][j] = T( Kokkos::subview(in,i,j,Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
												   Kokkos::ALL));
			}

			if constexpr( T::num_dims() == 5 ) {
				_data[i][j] = T( Kokkos::subview(in,i,j,Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
												   Kokkos::ALL, Kokkos::ALL));
			}

			if constexpr( T::num_dims() == 6 ) {
				_data[i][j] = T( Kokkos::subview(in,i,j,Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
												   Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
			}
		 }
		}
	}

	T& elem(Index i, Index j) const { return _data[i][j]; }
};

template<typename T,
	typename MemorySpace=Kokkos::DefaultExecutionSpace::memory_space,
	typename KokkosLayout=Kokkos::DefaultExecutionSpace::array_layout>
struct OLattice {
	const Index _n_sites;
	using array_type = typename T::array_type*;
	using base_type = typename T::base_type;
	using view_type = typename Kokkos::View<array_type,KokkosLayout,MemorySpace>;
	static constexpr
		KOKKOS_INLINE_FUNCTION
		Index num_dims()  {
			return 1+T::num_dims();
		}

		KOKKOS_INLINE_FUNCTION
		Index num_elem() const {
			return _n_sites*T::num_elem();
		}


	view_type _data;


	explicit OLattice(const Index n_sites) : _n_sites(n_sites), _data("OLatticeData", n_sites)
	{
	}


	T elem(int i) const {
	   if constexpr ( T::num_dims() == 1 ) {
		   return T( Kokkos::subview(_data,i, Kokkos::ALL ));
	   }

	   if constexpr ( T::num_dims() == 2 ) {
		   return T( Kokkos::subview(_data,i, Kokkos::ALL, Kokkos::ALL ));
	   }

	   if constexpr ( T::num_dims() == 3 ) {
		   return T( Kokkos::subview(_data,i, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL ));
	   }

	   if constexpr ( T::num_dims() == 4 ) {
	  		   return T( Kokkos::subview(_data,i, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL ));
	   }

	   if constexpr( T::num_dims() == 5 ) {
		   return T( Kokkos::subview(_data,i, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
				   	                          Kokkos::ALL));
	   }

	   if constexpr( T::num_dims() == 6 ) {
		   return T( Kokkos::subview(_data,i, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
				   	                          Kokkos::ALL, Kokkos::ALL));
	   }
	   if constexpr( T::num_dims() == 7 ) {
		   return T( Kokkos::subview(_data,i, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
				   	                          Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
	   }
	}

};
}


