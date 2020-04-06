/*
 * test_add_olattice.cpp
 *
 *  Created on: Apr 1, 2020
 *      Author: bjoo
 */


#include "gtest/gtest.h"
#include "test3.hpp"
#include "binop_add_olattice.hpp"
#include "Kokkos_Core.hpp"
#include <type_traits>

using namespace Playground;
using TestMemSpace = Kokkos::CudaUVMSpace;

TEST(Test4, OLatticeSpinorAdd)
{
	using storage=typename Kokkos::View<float*[4][3][2],TestMemSpace>;
	using LatticeFermion = OLattice<
							PVector<
								PVector<
								  RComplex<float,storage,3,4>,
								storage,3,2>,
	                        storage,4,1 >,
	                       storage,0>;

	LatticeFermion x(20);
	LatticeFermion y(20);
	LatticeFermion z(20);

	// Poor man's fill
	Kokkos::parallel_for(20,KOKKOS_LAMBDA(const size_t site) {
		for(size_t spin=0; spin < 4; ++spin) {
			for(size_t color=0; color < 3; ++color) {
				float r = static_cast<float>(0 + 2*(color + 3*(spin + 4*site)));
				float i = static_cast<float>(1 + 2*(color + 3*(spin + 4*site)));
				x.elem(site).elem(spin).elem(color).real() =r;
				x.elem(site).elem(spin).elem(color).imag() = i;

				y.elem(site).elem(spin).elem(color).real() = 2.0f*r;
				y.elem(site).elem(spin).elem(color).imag() = 2.0f*i;

				z.elem(site).elem(spin).elem(color).real() = 0.0f;
				z.elem(site).elem(spin).elem(color).imag() = 0.0f;
			}
		}
	});

	// Some expression
	evaluate( z, x + (x + y) );

	// Check
	for(int site = 0; site < 20; ++site) {
		for(size_t spin=0; spin < 4; ++spin) {
			for(size_t color=0; color < 3; ++color) {
				float r = static_cast<float>(0 + 2*(color + 3*(spin + 4*site)));
				float i = static_cast<float>(1 + 2*(color + 3*(spin + 4*site)));
				ASSERT_FLOAT_EQ( z.elem(site).elem(spin).elem(color).real(), 4.0f*r);
				ASSERT_FLOAT_EQ( z.elem(site).elem(spin).elem(color).imag(), 4.0f*i);
			}
		}

	}


}

// Test PMatrix
TEST(Test4, OLatticePropAdd)
{
	// types
	using storage=typename Kokkos::View<float*[4][4][3][3][2],TestMemSpace>;
	using LatticePropagator = OLattice<
							PMatrix<
								PMatrix<
								  RComplex<float,storage,5,6>,
								storage,3,3,4>,
	                        storage,4,1,2 >,
	                       storage,0>;

	LatticePropagator x(20);
	LatticePropagator y(20);
	LatticePropagator z(20);

	// Poor man's fill
	Kokkos::parallel_for(20,KOKKOS_LAMBDA(const size_t site) {
		for(size_t spin2=0; spin2 < 4; ++spin2) {
			for(size_t spin1=0; spin1 < 4; ++spin1) {
				for(size_t color2=0; color2 < 3; ++color2) {
					for(size_t color1=0;color1 < 3; ++color1) {

						float r = static_cast<float>(0 + 2*(color1 + 3*(color2 + 3*(spin1 + 4*(spin2 + 4*site)))));
						float i = static_cast<float>(1 + 2*(color1 + 3*(color2 + 3*(spin1 + 4*(spin2 + 4*site)))));
						x.elem(site).elem(spin2,spin1).elem(color2,color1).real() =r;
						x.elem(site).elem(spin2,spin1).elem(color2,color1).imag() = i;

						y.elem(site).elem(spin2,spin1).elem(color2,color1).real() = 2.0f*r;
						y.elem(site).elem(spin2,spin1).elem(color2,color1).imag() = 2.0f*i;

						z.elem(site).elem(spin2,spin1).elem(color2,color1).real() = 0.0f;
						z.elem(site).elem(spin2,spin1).elem(color2,color1).imag() = 0.0f;
					}
				}
			}
		}
	});

	// The evaluate
	evaluate(z, (x + y) + y);

	// Check
	for(size_t site=0; site < 20; ++site) {
		for(size_t spin2=0; spin2 < 4; ++spin2) {
			for(size_t spin1=0; spin1 < 4; ++spin1) {
				for(size_t color2=0; color2 < 3; ++color2) {
					for(size_t color1=0;color1 < 3; ++color1) {

						float r = static_cast<float>(0 + 2*(color1 + 3*(color2 + 3*(spin1 + 4*(spin2 + 4*site)))));
						float i = static_cast<float>(1 + 2*(color1 + 3*(color2 + 3*(spin1 + 4*(spin2 + 4*site)))));
						ASSERT_FLOAT_EQ( z.elem(site).elem(spin2,spin1).elem(color2,color1).real(), 5.0f*r);
						ASSERT_FLOAT_EQ( z.elem(site).elem(spin2,spin1).elem(color2,color1).imag(), 5.0f*i);
					}
				}
			}
		}
	}


}
