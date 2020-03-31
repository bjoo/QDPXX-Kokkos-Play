/*
 * binop_add.hpp
 *
 *  Created on: Mar 20, 2020
 *      Author: bjoo
 */

#pragma once

#include "test3.hpp"

namespace Playground {



// CRTP
template<typename OpKind>
struct RExpr {
	// Shorthand to do the static cast
	OpKind const& self() const {
		return static_cast<const OpKind &>(*this);
	}

	// Apply the () of the 'OpKind()'
	auto  operator()() const {
		self()();
	}
};

template<typename T>
class RLeaf : public RExpr<RLeaf<T>> {
public:
	explicit RLeaf(const T& value) : _value(value){}
	using dst_type = T;
	auto operator()() const {
		return _value;
	}
private:
	const T _value;
};


template<typename DstType, typename OpType, typename LeftType, typename RightType>
struct RBinOp : RExpr< RBinOp<DstType,OpType,LeftType,RightType>> {
	explicit RBinOp(const LeftType left, const RightType right) : _left(left), _right(right) {}

	LeftType _left;
	RightType _right;
	OpType _op;
	using dst_type = DstType;

	auto operator()() const {
		return _op( _left(), _right() );
	}
};


struct _rleaf_plus {
	template<typename T, typename V, size_t N>
	auto operator()(const RScalar<T,V,N>& l,
			const RScalar<T,V,N>& r) const {
		return RScalarLocal<T>( l.elem() + r.elem() );
	}

	template<typename T, typename V, size_t N>
	auto operator()(const RScalar<T,V,N>& l,
			const RScalarLocal<T>& r) const {
		return RScalarLocal<T>( l.elem() + r.elem() );
	}

	template<typename T, typename V, size_t N>
	auto operator()(const RScalarLocal<T>& l,
			const RScalar<T,V,N>& r) const {
		return RScalarLocal<T>( l.elem() + r.elem() );
	}

	template<typename T>
	auto operator()(const RScalarLocal<T>& l,
			const RScalarLocal<T>& r) const {
			return RScalarLocal<T>( l.elem() + r.elem() );
		}
};

template<typename E1, typename E2>
auto operator+( const RExpr<E1>& l, const RExpr<E2>& r ) {
	return RBinOp< typename E1::dst_type, _rleaf_plus, E1, E2>( l.self(), r.self());
}


} // Namespace


