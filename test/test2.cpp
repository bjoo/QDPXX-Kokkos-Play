/*
 * test2.cpp
 *
 *  Created on: Mar 17, 2020
 *      Author: bjoo
 */

#include "gtest/gtest.h"

#include "test2.hpp"
#include <type_traits>

using namespace Playground;

#if 0
TEST(Test2, TestILatType)
{
	using ILatType = ILattice<float,8>;
	bool array_type_assertion = std::is_same_v< ILatType::array_type, float[8]>;
	bool base_type_assertion = std::is_same_v< ILatType::base_type, float>;

	ASSERT_TRUE( array_type_assertion );
	ASSERT_TRUE( base_type_assertion );
	ASSERT_EQ( ILatType::num_elem(), 8);
	ASSERT_EQ( ILatType::num_dims(), 1);
	bool is_std_layout = std::is_standard_layout< ILatType >::value;
	ASSERT_TRUE( is_std_layout );
}


TEST(Test2, TestRScalarILatType)
{
	using ILatType = ILattice<float,8>;
	using ScalarType = RScalar<ILatType>;

	bool array_type_assertion = std::is_same_v< ScalarType::array_type, float[8][1]>;
	bool base_type_assertion = std::is_same_v< ScalarType::base_type, float>;

	ASSERT_EQ( ScalarType::num_elem(), ILatType::num_elem() );
	ASSERT_EQ( ScalarType::num_dims(), ILatType::num_dims());


}



TEST(Test2, TestRComplexILatType)
{
	using ILatType = ILattice<float,8>;
	using ComplexType = RComplex<ILatType>;

	bool array_type_assertion = std::is_same_v< ComplexType::array_type, float[8][2]>;
	bool base_type_assertion = std::is_same_v< ComplexType::base_type, float>;

	ASSERT_EQ( ComplexType::num_elem(), 2*ILatType::num_elem() );
	ASSERT_EQ( ComplexType::num_dims(), 2 );

}


TEST(Test2, TestPVectorRComplexILatType)
{
	using ILatType = ILattice<float,8>;
	using ComplexType = RComplex<ILatType>;
	using PVectorType = PVector<ComplexType,3>;

	bool array_type_assertion = std::is_same_v< PVectorType::array_type, float[8][2][3]>;
	bool base_type_assertion = std::is_same_v< PVectorType::base_type, float>;

	ASSERT_EQ( PVectorType::num_elem(), 3*2*ILatType::num_elem() );
	ASSERT_EQ( PVectorType::num_dims(), 3 );
}

TEST(Test2, TestPMatrixRComplexILatType)
{
	using ILatType = ILattice<float,8>;
	using ComplexType = RComplex<ILatType>;
	using PMatrixType = PMatrix<ComplexType,3>;


	bool array_type_assertion = std::is_same_v< PMatrixType::array_type, float[8][2][3][3]>;
	bool base_type_assertion = std::is_same_v< PMatrixType::base_type, float>;

	ASSERT_EQ( PMatrixType::num_elem(), 3*3*2*ILatType::num_elem() );
	ASSERT_EQ( PMatrixType::num_dims(), 4 );
}

TEST(Test2, TestPropRComplexILatType)
{
	using ILatType = ILattice<float,8>;
	using ComplexType = RComplex<ILatType>;
	using PColMatrixType = PMatrix<ComplexType,3>;
	using PSpinMatrixType = PMatrix<PColMatrixType,4>;

	bool array_type_assertion = std::is_same_v< PSpinMatrixType::array_type, float[8][2][3][3][4][4]>;
	bool base_type_assertion = std::is_same_v< PSpinMatrixType::base_type, float>;

	ASSERT_EQ( PSpinMatrixType::num_elem(), 4*4*3*3*2*ILatType::num_elem() );
	ASSERT_EQ( PSpinMatrixType::num_dims(), 6 );
}

TEST(Test2, TestOLatticePropILatType)
{
	using ILatType = ILattice<float,8>;
	using ComplexType = RComplex<ILatType>;
	using PColMatrixType = PMatrix<ComplexType,3>;
	using PPropType = PMatrix<PColMatrixType,4>;
	using OLatticeType = OLattice<PPropType>;

	OLatticeType testlat(20);

	bool array_type_assertion = std::is_same_v< OLatticeType::array_type, float*[8][2][3][3][4][4]>;
	bool base_type_assertion = std::is_same_v<OLatticeType::base_type, float>;

	ASSERT_EQ( testlat.num_elem(), 20*4*4*3*3*2*ILatType::num_elem() );
	ASSERT_EQ( OLatticeType::num_dims(), 7 );

	// Now I need to check on my view
	ASSERT_EQ(testlat._data_view.rank, OLatticeType::num_dims());

	ASSERT_EQ( testlat._data_view.span(), 20*3*3*4*4*2*8);
	ASSERT_EQ( testlat._data_view.extent(0),20);
	ASSERT_EQ( testlat._data_view.extent(1),4);
	ASSERT_EQ( testlat._data_view.extent(2),4);
	ASSERT_EQ( testlat._data_view.extent(3),3);
	ASSERT_EQ( testlat._data_view.extent(4),3);
	ASSERT_EQ( testlat._data_view.extent(5),2);
	ASSERT_EQ( testlat._data_view.extent(6),8);

}

TEST(Test2,ILatticeStorage)
{
	using ILatType = ILattice<float,8>;

	Kokkos::View<float[8]> a("a");

	ILatType fred(a); // Should copy in 'a' and use it as a view

	Kokkos::parallel_for(8,[=](Index i){
		fred.elem(i) = static_cast<float>(i);
	});

	typename Kokkos::View<float[8]>::HostMirror b = Kokkos::create_mirror_view(a);
	Kokkos::deep_copy(b,a);

	for(int i=0; i < 8; ++i) {
		ASSERT_FLOAT_EQ( b(i), static_cast<float>(i));
	}

}

TEST(Test2, TestRScalILatStorage)
{
	using ILatType = ILattice<float,8>;
	using RScalType = RScalar<ILatType>;

	using RScalView = RScalType::view_type;

	RScalView storage("label");

	RScalType a(storage);

	Kokkos::parallel_for(8,[=](const Index i){
		(a.elem()).elem(i) = i;
	});

	typename RScalView::HostMirror b = Kokkos::create_mirror_view(storage);
	Kokkos::deep_copy(b,storage);

	for(int i=0; i < 8; ++i) {
		ASSERT_FLOAT_EQ( b(0,i), static_cast<float>(i));
	}
}


TEST(Test2, TestComplexILatStorage)
{
	using ILatType = ILattice<float,8>;
	using RComplexType = RComplex<ILatType>;

	using RComplexView = RComplexType::view_type;

	RComplexView storage("label");

	RComplexType a(storage);

	Kokkos::parallel_for(8,[=](const Index i){
		a.real().elem(i) = 2*i;
		a.imag().elem(i) = 2*i+1;
	});

	typename RComplexView::HostMirror b = Kokkos::create_mirror_view(storage);
	Kokkos::deep_copy(b,storage);

	for(int i=0; i < 8; ++i) {
		ASSERT_FLOAT_EQ( b(0,i), static_cast<float>(2*i));
		ASSERT_FLOAT_EQ( b(1,i), static_cast<float>(2*i+1));
	}
}

TEST(Test2, TestPVectorComplexILatStorage)
{
	using ILatType = ILattice<float,8>;
	using RComplexType = RComplex<ILatType>;
	using PVecType = PVector<RComplexType,4>;

	using PVecView = PVecType::view_type;

	PVecView storage("label");

	PVecType a(storage);

	Kokkos::parallel_for(8*2*4,[=](const Index i){

		Index lane = i%8;
		Index compspin = i/8;

		Index comp = compspin %2;
		Index spin = compspin /2;

		comp == 0 ? a.elem(spin).real().elem(lane) = i : a.elem(spin).imag().elem(lane) = i;

	});


	typename PVecView::HostMirror b = Kokkos::create_mirror_view(storage);
	Kokkos::deep_copy(b,storage);

	for(int s=0; s < 4; ++s) {
		for(int reim=0; reim < 2; ++reim) {
			for(int lane=0; lane < 8; ++lane) {

				Index i= lane + 8*(reim + 2*s);

				ASSERT_FLOAT_EQ( b(s,reim,lane), static_cast<float>(i));
			}
		}
	}
}

TEST(Test2, TestPPropComplexILatStorage)
{
	using ILatType = ILattice<float,8>;
	using RComplexType = RComplex<ILatType>;
	using PColMatType = PMatrix<RComplexType,3>;
	using PPropType = PMatrix<PColMatType,4>;

	using PPropView = PPropType::view_type;

	PPropView storage("label");

	PPropType a(storage);

	Kokkos::parallel_for(8*2*3*3*4*4,[=](const Index i){

		Index lane = i%8;
		Index compcolcolspinspin = i/8;

		Index comp = compcolcolspinspin % 2;
		Index colcolspinspin = compcolcolspinspin/2;

		Index col1 = colcolspinspin % 3;
		Index colspinspin = colcolspinspin / 3;

		Index col2 = colspinspin % 3;
		Index spinspin = colspinspin / 3;

		Index spin1 = spinspin % 4;
		Index spin2 = spinspin / 4;



		comp == 0 ? a.elem(spin2,spin1).elem(col2,col1).real().elem(lane) = i
				: a.elem(spin2,spin1).elem(col2,col1).imag().elem(lane) = i;

	});


	typename PPropView::HostMirror b = Kokkos::create_mirror_view(storage);
	Kokkos::deep_copy(b,storage);

	for(int s2=0; s2 < 4; ++s2) {
		for(int s1=0; s1 < 4; ++s1) {

			for(int c2=0; c2 < 3; ++c2 ) {
				for(int c1=0; c1 < 3; ++c1 ) {

					for(int reim=0; reim < 2; ++reim) {
						for(int lane=0; lane < 8; ++lane) {

							Index i= lane + 8*(reim + 2*(c1 + 3*(c2 + 3*(s1 + 4*s2))));

							ASSERT_FLOAT_EQ( b(s2,s1,c2,c1,reim,lane), static_cast<float>(i));
						}
					}
				}
			}
		}
	}
}

#endif

TEST(Test2, TestOLatPPropComplexILatStorage)
{
	using PVecType = PVector<float,3>;
	using PPropType = PMatrix<PVecType,4>;
	using OLatType = OLattice<PPropType>;

	OLatType the_prop(20);
	auto elem_0 = the_prop.elem(0);

	for(int i=0; i < 4; ++i) {
		for(int j=0; j < 4; ++j ) {
			for(int k=0; k < 3; ++k){
				elem_0.elem(i,j).elem(k) = static_cast<float>(i + 4*(j+4*k));
			}
		}
	}

	for(int i=0; i < 4; ++i) {
		for(int j=0; j < 4; ++j ) {
			for(int k=0; k < 4; ++k ) {
				ASSERT_FLOAT_EQ( the_prop._data_view(0,i,j,k), static_cast<float>(i + 4*(j+4*k)) );
			}
		}
	}


}

