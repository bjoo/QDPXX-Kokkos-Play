/*
 * test_add.cpp
 *
 *  Created on: Mar 20, 2020
 *      Author: bjoo
 */


#include "gtest/gtest.h"

#include "binop_add.hpp"
#include "Kokkos_Core.hpp"


using namespace Playground;

TEST(TestAdd, TestILatticeAdd)
{
	ILattice<float,8> a,b;

	for(Index i=0; i < 8; ++i) {
		a.elem(i) = static_cast<float>(i);
		b.elem(i) = static_cast<float>(2*i);

	}

	auto c = a + b ;

	for(Index i=0; i < 8; ++i) {
		ASSERT_FLOAT_EQ( c.elem(i), static_cast<float>(3*i) ) ;
	}
}

TEST(TestAdd, RScalarAdd)
{
	DEBUG_MSG("Instantiate a and b");
	RScalar<ILattice<float,8>> a;
	RScalar<ILattice<float,8>> b;



	DEBUG_MSG("Setting lanes of a_inner and b_inner");
	for(Index i=0; i < 8; ++i) {
		a.elem().elem(i) = static_cast<float>(i);
		b.elem().elem(i) = static_cast<float>(2*i);
	}

	DEBUG_MSG("Entering add operation");
	auto c = a + b ;

	DEBUG_MSG("Finishing add");
	for(Index i=0; i < 8; ++i) {
		ASSERT_FLOAT_EQ( c.elem().elem(i), static_cast<float>(3*i) ) ;
	}
}


TEST(TestAdd, RComplexAdd)
{
	RComplex<ILattice<float,8>> a,b;

	for(Index i=0; i < 8; ++i) {
		a.real().elem(i) = static_cast<float>(2*i);
		a.imag().elem(i) = static_cast<float>(2*i+1);
		b.real().elem(i) = static_cast<float>(4*i);
		b.imag().elem(i) = static_cast<float>(4*i+2);

	}

	auto c = a + b ;

	for(Index i=0; i < 8; ++i) {
		ASSERT_FLOAT_EQ( c.real().elem(i), static_cast<float>(6*i) ) ;
		ASSERT_FLOAT_EQ( c.imag().elem(i), static_cast<float>(6*i+3) ) ;
	}
}

#if 0
TEST(TestAdd, PVectorAdd)
{
	PVector<RComplex<ILattice<float,8>>,4> a,b;

	for(int vrow=0; vrow < 4; ++vrow) {
		for(Index i=0; i < 8; ++i) {
			a.elem(vrow).real().elem(i) = static_cast<float>(2*i + 2*8*vrow);
			a.elem(vrow).imag().elem(i) = static_cast<float>(2*i+1 + 2*8*vrow);
			b.elem(vrow).real().elem(i) = static_cast<float>(4*i + 2*8*vrow);
			b.elem(vrow).imag().elem(i) = static_cast<float>(4*i+2 + 2*8*vrow);
		}
	}

	auto c = a + b ;

	for(int vrow=0; vrow<4; ++vrow) {
		for(Index i=0; i < 8; ++i) {
			ASSERT_FLOAT_EQ( c.elem(vrow).real().elem(i), static_cast<float>(6*i + 32*vrow) ) ;
			ASSERT_FLOAT_EQ( c.elem(vrow).imag().elem(i), static_cast<float>(6*i+3 + 32*vrow) ) ;
		}
	}
}

TEST(TestAdd, PMatrixAdd)
{
	PMatrix<RComplex<ILattice<float,8>>,4> a,b;

	for(int row=0; row < 4; ++row) {
		for(int col=0; col < 4; ++col) {
			for(Index i=0; i < 8; ++i) {
				a.elem(row,col).real().elem(i) = static_cast<float>(2*i + 2*8*(col+4*row));
				a.elem(row,col).imag().elem(i) = static_cast<float>(2*i+1 + 2*8*(col+4*row));
				b.elem(row,col).real().elem(i) = static_cast<float>(4*i + 2*8*(col + 4*row));
				b.elem(row,col).imag().elem(i) = static_cast<float>(4*i+2 + 2*8*(col + 4*row));
			}
		}
	}

	auto c = a + b ;

	for(int row=0;  row<4; ++ row) {
		for(int col=0; col < 4; ++col) {
		for(Index i=0; i < 8; ++i) {
			ASSERT_FLOAT_EQ( c.elem(row,col).real().elem(i), static_cast<float>(6*i + 32*(col + 4*row)) ) ;
			ASSERT_FLOAT_EQ( c.elem(row,col).imag().elem(i), static_cast<float>(6*i+3 + 32*(col + 4*row)) ) ;
		}
	}
	}
}

TEST(TestAdd, OLatticeAdd)
{
	OLattice<PMatrix<RComplex<ILattice<float,8>>,4>,Kokkos::OpenMP,Kokkos::LayoutRight> a(20),b(20),c(20);

	for(Index site=0; site < 20; ++ site) {
		for(int row=0; row < 4; ++row) {
			for(int col=0; col < 4; ++col) {
				for(Index i=0; i < 8; ++i) {
					a.elem(site).elem(row,col).real().elem(i) = static_cast<float>(2*i + 2*8*(col+4*(row+4*site)));
					a.elem(site).elem(row,col).imag().elem(i) = static_cast<float>(2*i+1 + 2*8*(col+4*(row+4*site)));
					b.elem(site).elem(row,col).real().elem(i) = static_cast<float>(4*i + 2*8*(col + 4*(row+4*site)));
					b.elem(site).elem(row,col).imag().elem(i) = static_cast<float>(4*i+2 + 2*8*(col + 4*(row+4*site)));
				}
			}
		}
	}

	op_add_evaluate(c,a,b);
/*
	for(Index site=0; site < 20; ++site) {
		for(int row=0;  row<4; ++ row) {
			for(int col=0; col < 4; ++col) {
				for(Index i=0; i < 8; ++i) {
					ASSERT_FLOAT_EQ( c.elem(site).elem(row,col).real().elem(i), static_cast<float>(6*i + 32*(col + 4*(row+4*site))) ) ;
					ASSERT_FLOAT_EQ( c.elem(site).elem(row,col).imag().elem(i), static_cast<float>(6*i+3 + 32*(col + 4*(row+4*site))) ) ;
				}
			}
		}
	}
	*/
}
#endif
