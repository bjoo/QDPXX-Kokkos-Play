/*
 * binop_add.hpp
 *
 *  Created on: Mar 20, 2020
 *      Author: bjoo
 */

#pragma once

#include "test2.hpp"

namespace Playground {

// ILattice
template<typename T,
		 Index N,
		 typename MS=Kokkos::DefaultExecutionSpace::memory_space,
		 typename KL=Kokkos::DefaultExecutionSpace::array_layout>
KOKKOS_INLINE_FUNCTION
auto operator+(const ILattice<T,N,MS,KL>& left, const ILattice<T,N,MS,KL>& right)
{
	ILattice<T,N,MS,KL> ret_val;

	for(Index i=0; i < N; ++i) {
		ret_val.elem(i) = left.elem(i)+right.elem(i);
	}
	return ret_val;
}

// RScalar
template<typename T,
		 typename MS=Kokkos::DefaultExecutionSpace::memory_space,
		 typename KL=Kokkos::DefaultExecutionSpace::array_layout>
KOKKOS_INLINE_FUNCTION
auto operator+(const RScalar<T,MS,KL>& left, const RScalar<T,MS,KL>& right)
{
	RScalar<T,MS,KL> ret_val;
	ret_val.elem() = left.elem() + right.elem();
	return ret_val;
}


// RComplex
template<typename T,
		 typename MS=Kokkos::DefaultExecutionSpace::memory_space,
		 typename KL=Kokkos::DefaultExecutionSpace::array_layout>
KOKKOS_INLINE_FUNCTION
auto operator+(const RComplex<T,MS,KL>& left, const RComplex<T,MS,KL>& right)
{
	RComplex<T,MS,KL> ret_val;
	ret_val.real() = left.real() + right.real();
	ret_val.imag() = left.imag() + right.imag();
	return ret_val;
}


#if 0
// PVector
template<typename T,
		 Index N,
		 typename MS=Kokkos::DefaultExecutionSpace::memory_space,
		 typename KL=Kokkos::DefaultExecutionSpace::array_layout>
KOKKOS_INLINE_FUNCTION
auto operator+(const PVector<T,N,MS,KL>& left, const PVector<T,N,MS,KL>& right)
{
	PVector<T,N,MS,KL> ret_val;

	for(int i=0; i < N; ++i) {
		ret_val.elem(i) = left.elem(i) + right.elem(i);
	}
	return ret_val;
}

// PMatrix
template<typename T,
		 Index N,
		 typename MS=Kokkos::DefaultExecutionSpace::memory_space,
		 typename KL=Kokkos::DefaultExecutionSpace::array_layout>
KOKKOS_INLINE_FUNCTION
auto operator+(const PMatrix<T,N,MS,KL>& left, const PMatrix<T,N,MS,KL>& right)
{
	PMatrix<T,N,MS,KL> ret_val;

	for(int i=0; i < N; ++i) {
		for(int j=0; j < N; ++j) {
			ret_val.elem(i,j) = left.elem(i,j) + right.elem(i,j);
		}
	}
	return ret_val;
}

template<typename T,
		 typename MS=Kokkos::DefaultExecutionSpace::memory_space,
		 typename KL=Kokkos::DefaultExecutionSpace::array_layout>
KOKKOS_INLINE_FUNCTION
void op_add_evaluate(OLattice<T,MS,KL> dst, const OLattice<T,MS,KL> left, const OLattice<T,MS,KL> right)
{
	if( dst._n_sites == left._n_sites && left._n_sites == right._n_sites) {
		for(Index i=0;  i < dst._n_sites; ++i) {
			dst.elem(i) = left.elem(i) + right.elem(i);
		}
	}
}
#endif

} // Namespace


