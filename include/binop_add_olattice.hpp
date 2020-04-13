/*
 * binop_add_olattice.hpp
 *
 *  Created on: Apr 1, 2020
 *      Author: bjoo
 */

#pragma once

#include <cstddef>
#include "test3.hpp"
#include "Kokkos_Core.hpp"

namespace Playground {

// CRTP
template<typename OpKind>
struct OLatExpr {
	// Shorthand to do the static cast
	KOKKOS_INLINE_FUNCTION
	OpKind const& self() const {
		return static_cast<const OpKind &>(*this);
	}

	// Apply the () of the 'OpKind()'
	KOKKOS_INLINE_FUNCTION
	auto  operator()(const std::size_t site ) const {
		self()(site);
	}
};

// Leaf type -- to hold an OLattice
template<typename T>
class OLeaf : public OLatExpr<OLeaf<T>> {
public:
	KOKKOS_INLINE_FUNCTION
	explicit OLeaf(const T& value) : _value(value){}
	using dst_type = T;

	KOKKOS_INLINE_FUNCTION
	auto operator()(size_t site) const {
		return _value.elem(site);
	}
private:
	const T _value;
};

// A BinOp
template<typename OpType, typename LeftType, typename RightType>
struct OLatBinOp : OLatExpr< OLatBinOp<OpType,LeftType,RightType>> {
	explicit OLatBinOp(const LeftType left, const RightType right) : _left(left), _right(right) {}

	LeftType _left;
	RightType _right;
	OpType _op;


	// Execut BinOp for a site
	KOKKOS_INLINE_FUNCTION
	auto operator()(const std::size_t site) const {
		return _op( _left(site), _right(site) );
	}
};

// All the plus operations for all the subtypes
// for both view + view, local + view, view + local, local + local
struct _plus {
	//
	// PMatrix
    //
	//
	// LeftType is either  PMatrixLocal (local) or a PMatrix (view)
	// Right Type is either PMatrixLocal (local) or a PMatrix (view)
	template<typename LeftType, typename RightType>
	KOKKOS_INLINE_FUNCTION
	auto add_mat( const LeftType& l, const RightType& r) const {

		// Change (const LeftType&) into a LeftType so we can get its local
		// type
		using LT = typename std::remove_reference<LeftType>::type;
		using CLT = typename std::remove_const<LT>::type;

		using local_type = typename LocalType<CLT>::type;
		local_type ret_val;
		for(size_t i=0; i < local_type::size(); ++i) {
			for(size_t j=0; j < local_type::size(); ++j) {
				ret_val.elem(i,j) = (*this)( l.elem(i,j) , r.elem(i,j));
			}
		}
		return ret_val;
	}


	// PMatrix: view + view
	// For correct Both PMatrices must have the same subtype
	template<typename T,typename V, size_t N, size_t ParentNumDims>
	KOKKOS_INLINE_FUNCTION
	auto operator()(const PMatrix<T,V,N,ParentNumDims>& l,
		 	        const PMatrix<T,V,N,ParentNumDims>& r) const {
		return add_mat<decltype(l), decltype(r)>(l,r);
	}

	// PMatrix: local + view
	template<typename T, typename V, size_t N, size_t ParentNumDims>
	KOKKOS_INLINE_FUNCTION
	auto operator()(const PMatrixLocal<typename LocalType<T>::type,N>& l,
		 	        const PMatrix<T,V,N,ParentNumDims>& r) const {
		return add_mat< decltype(l), decltype(r)>(l,r);
	}

	// PMatrix: view + local
	template<typename T, typename V, size_t N, size_t ParentNumDims>
	KOKKOS_INLINE_FUNCTION
	auto operator()(const PMatrix<T,V,N,ParentNumDims>& l,
		 	        const PMatrixLocal<typename LocalType<T>::type,N>& r) const {
		return add_mat< decltype(l), decltype(r)>(l,r);
	}

	// PMatrix: local + local
	template<typename T, size_t N>
	KOKKOS_INLINE_FUNCTION
	auto operator()(const PMatrixLocal<T,N>& l,
			const PMatrixLocal<T,N>& r) const {
		return add_mat< decltype(l), decltype(r)>(l,r);
	}

	//-------------------------------------------
	// PVector:
	// ------------------------------------------
	//
	template<typename LeftType, typename RightType>
	KOKKOS_INLINE_FUNCTION
	auto add_vector(const LeftType& l, const RightType& r) const {
		// Remove const and & to just get the left type
		// The local types of left and right must be the same
		// so it doesn't matter which we use.
		using LT = typename std::remove_reference<LeftType>::type;
		using CLT = typename std::remove_const<LT>::type;

		using local_type = typename LocalType<CLT>::type;
		local_type ret_val;
		for(size_t i=0; i < local_type::size(); ++i) {
			ret_val.elem(i) = (*this)( l.elem(i), r.elem(i) );
		}
		return ret_val;
	}

	// PVector: view + view
	template<typename T1,typename T2, typename V, size_t N, size_t ParentNumDims>
	KOKKOS_INLINE_FUNCTION
	auto operator()(const PVector<T1,V,N,ParentNumDims>& l, const PVector<T2,V,N,ParentNumDims>& r) const {
		return add_vector<decltype(l),decltype(r)>(l,r);
	}

	// PVector: local + view
	template<typename T1, typename T2, typename V, size_t N, size_t ParentNumDims>
	KOKKOS_INLINE_FUNCTION
	auto operator()(const PVectorLocal<T1,N>& l, const PVector<T2,V,N,ParentNumDims>& r) const {
		return add_vector<decltype(l),decltype(r)>(l,r);
	}

	// PVector: view + local
	template<typename T1,typename T2, typename V, size_t N, size_t ParentNumDims>
	KOKKOS_INLINE_FUNCTION
	auto operator()(const PVector<T1,V,N,ParentNumDims>& l, const PVectorLocal<T2,N>& r) const {
		return add_vector<decltype(l),decltype(r)>(l,r);
	}

	// PVector: local+ local
	template<typename T1, typename T2, size_t N>
	KOKKOS_INLINE_FUNCTION
	auto operator()(const PVectorLocal<T1,N>& l, const PVectorLocal<T2,N>& r) const {
		return add_vector<decltype(l),decltype(r)>(l,r);
	}

	// --------------------------------
	// RComplex:
	// --------------------------------
	template<typename LeftType, typename RightType>
	KOKKOS_INLINE_FUNCTION
	auto add_cmplx(const LeftType& l, const RightType& r) const {
		using LT = typename LocalType<
				     typename std::remove_const<
				       typename std::remove_reference<LeftType>::type>::type>::type;

		return LT( l.real() + r.real(), l.imag() + r.imag() );
	}

	// Complex: view + view
	template<typename T,typename V, size_t ParentNumDims>
	KOKKOS_INLINE_FUNCTION
	auto operator()(const RComplex<T,V,ParentNumDims>& l,
			        const RComplex<T,V,ParentNumDims>& r) const {
		return add_cmplx<decltype(l),decltype(r)>(l,r);
	}

	// Complex: local + view
	template<typename T,typename V, size_t ParentNumDims>
	KOKKOS_INLINE_FUNCTION
	auto operator()(const RComplexLocal<T>& l,
			const RComplex<T,V,ParentNumDims>& r) const {
		return add_cmplx<decltype(l),decltype(r)>(l,r);
	}

	// Complex: view + local
	template<typename T,typename V, size_t ParentNumDims>
	KOKKOS_INLINE_FUNCTION
	auto operator()(const RComplex<T,V,ParentNumDims>& l,
			const RComplexLocal<T>& r) const {
		return add_cmplx<decltype(l),decltype(r)>(l,r);
	}

	// Complex: local + local
	template<typename T>
	KOKKOS_INLINE_FUNCTION
	auto operator()(const RComplexLocal<T>& l,
			const RComplexLocal<T>& r) const {
		return add_cmplx<decltype(l),decltype(r)>(l,r);
	}

	// -------------------------------------------------------
	// RScalar
	// -------------------------------------------------------
	template<typename LeftType, typename RightType>
	KOKKOS_INLINE_FUNCTION
	auto add_scalar( const LeftType& l, const RightType& r) const {
		using LT = typename LocalType<
						     typename std::remove_const<
						       typename std::remove_reference<LeftType>::type>::type>::type;
		return LT(l.elem() + r.elem());
	}

	// RScalar: view + view
	template<typename T, typename V, size_t ParentNumDims>
	KOKKOS_INLINE_FUNCTION
	auto operator()( const RScalar<T,V,ParentNumDims>& l, const RScalar<T,V,ParentNumDims>& r) const {
		return add_scalar<decltype(l),decltype(r)>(l,r);
	}

	// RScalar: local + view
	template<typename T, typename V, size_t ParentNumDims>
	KOKKOS_INLINE_FUNCTION
	auto operator()( const RScalarLocal<T>& l, const RScalar<T,V,ParentNumDims>& r) const {
		return add_scalar<decltype(l),decltype(r)>(l,r);
	}

	// RScalar: view + local
	template<typename T, typename V, size_t ParentNumDims>
	KOKKOS_INLINE_FUNCTION
	auto operator()( const RScalar<T,V,ParentNumDims>& l, const RScalarLocal<T>& r) const {
		return add_scalar<decltype(l),decltype(r)>(l,r);
	}

	// RScalar: local + local
	template<typename T>
	KOKKOS_INLINE_FUNCTION
	auto operator()( const RScalarLocal<T>& l, const RScalarLocal<T>& r) const {
		return add_scalar<decltype(l),decltype(r)>(l,r);
	}
};

// This takes two OLatExpr-s and creates a binop
template<typename E1, typename E2>
auto operator+( const E1& l, const E2& r ) {
	return OLatBinOp< _plus,
					E1, E2>( l.self(), r.self());
}

#if 1
// This OLat + an OLatExpr and creates a BinOp
template<typename T, class MemSpace, typename Expr>
auto operator+( const OLattice<T,MemSpace>& l, const Expr& r ) {
	return OLatBinOp< _plus,
					 OLeaf<OLattice<T,MemSpace>>,
					 Expr>(
							OLeaf<OLattice<T,MemSpace>>(l), r.self());
}
#endif
// This OLat + an OLatExpr and creates a BinOp
template<typename T, class MemSpace, typename Expr>
auto operator+( const Expr& l, const OLattice<T,MemSpace>& r) {
	return OLatBinOp<  _plus,
					Expr,
					OLeaf<OLattice<T,MemSpace>> >(
							l.self(), OLeaf<OLattice<T,MemSpace>>(r));
}


// This OLat + OLat and creates a BinOp
template<typename T, class MemSpace>
auto operator+( const OLattice<T,MemSpace> l, const OLattice<T,MemSpace>& r) {
	return OLatBinOp< _plus,
					OLeaf<OLattice<T,MemSpace>>,
					OLeaf<OLattice<T,MemSpace>> >(
							OLeaf<OLattice<T,MemSpace>>(l), OLeaf<OLattice<T,MemSpace>>(r));
}

template<typename T, class MemSpace, typename Expr>
void evaluate(OLattice<T,MemSpace>& dest, const Expr& expression ) {

	const std::size_t n_sites= dest.num_elem();
	Kokkos::parallel_for(n_sites,KOKKOS_LAMBDA(const size_t site) {
		dest.elem(site) = expression(site);
	});
	Kokkos::fence(); // Apparently this is needed for consistency in UVM space
}


} // namespace


