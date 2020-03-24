/*
 * test3.hpp
 *
 *  Created on: Mar 24, 2020
 *      Author: bjoo
 */

#pragma once

#include "Kokkos_Core.hpp"
namespace Playground {



template<typename T, typename ViewType, size_t _NumDims>
struct RScalar {
	ViewType _data;
	std::array<size_t,8> _indices;


	RScalar(ViewType data_in, std::array<size_t,8> indices) : _data(data_in), _indices(indices){};

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

template<typename T, typename ViewType, size_t _IdxPos, size_t _NumDims >
struct RComplex {
	ViewType _data;
	std::array<size_t,8> _indices;

	RComplex(ViewType data_in, std::array<size_t,8> indices) : _data(data_in), _indices(indices){};

	T& real() const {
		std::array<size_t,8> new_idx(_indices);
		new_idx[ _IdxPos ] = 0;
		return get( new_idx );
	}

	T& imag() const {
		std::array<size_t,8> new_idx(_indices);
		new_idx[ _IdxPos ] = 1;
		return get( new_idx );
	}

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

template<typename T, typename ViewType, size_t _N, size_t _IdxPos >
struct PVector {
	ViewType _data;
	std::array<size_t,8> _indices;

	PVector(ViewType data_in, std::array<size_t,8> indices) : _data(data_in), _indices(indices){};

	auto elem(size_t i) const {
		std::array<size_t,8> new_idx(_indices);
		new_idx[_IdxPos] = i;
		return T(_data, new_idx);
	}

};

template<typename T, typename ViewType, size_t _N, size_t _IdxPos1, size_t _IdxPos2 >
struct PMatrix {
	ViewType _data;
	std::array<size_t,8> _indices;

	PMatrix(ViewType data_in, std::array<size_t,8> indices) : _data(data_in), _indices(indices){};

	auto elem(size_t i, size_t j) const {
		std::array<size_t,8> new_idx(_indices);
		new_idx[_IdxPos1] = i;
		new_idx[_IdxPos2] = j;
		return T(_data, new_idx);
	}

};


// Eventually deduce Kokkos array_type of the subtype T,
// and can compute view type

template<typename T, typename ViewType, size_t _IdxPos1>
struct OLattice {

	ViewType _data;

	OLattice(size_t n_elem) : _data("internal", n_elem) {}
	OLattice(ViewType t) : _data(t) {}

	auto elem(size_t i) const {
		std::array<size_t,8> index{0,0,0,0, 0,0,0,0};
		index[_IdxPos1 ] = i;
		return T(_data, index );
	}

};

}// namespace
