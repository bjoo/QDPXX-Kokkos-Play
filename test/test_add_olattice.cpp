/*
 * test_add_olattice.cpp
 *
 *  Created on: Apr 1, 2020
 *      Author: bjoo
 */


#include "gtest/gtest.h"
#include "test3.hpp"
#include "binop_add_olattice.hpp"
#include "Kokkos_Macros.hpp"
#include "Kokkos_Core.hpp"
#include <type_traits>

using namespace Playground;

#if defined(KOKKOS_ENABLE_CUDA)
using TestMemSpace=Kokkos::CudaSpace;
#elif defined(KOKKOS_ENABLE_HIP)
using TestMemSpace = Kokkos::Experimental::HIPSpace;
#elif defined(KOKKOS_ENABLE_OPENMP)
using TestMemSpace = Kokkos::HostSpace;
#elif defined(KOKKOS_ENABLE_OPENMPTARGET)
using TestMemSpace = Kokkos::Experimental::OpenMPTargetSpace;
#endif

void testOLatSpinorAdd(void)
{

  using LatticeFermion = OLattice<
    PVectorLocal<
      PVectorLocal<
	     RComplexLocal<float>,
	  3>,
     4>, TestMemSpace>;

  
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
  Kokkos::fence();

  // Some expression
  evaluate( z, x + (x + y) );

  {
    auto z_mirror = Kokkos::create_mirror(z._data);
    Kokkos::deep_copy(z_mirror, z._data);
    
    // Check -- always on host
    for(int site = 0; site < 20; ++site) {
      for(size_t spin=0; spin < 4; ++spin) {
	for(size_t color=0; color < 3; ++color) {
	  float r = static_cast<float>(0 + 2*(color + 3*(spin + 4*site)));
	  float i = static_cast<float>(1 + 2*(color + 3*(spin + 4*site)));
		ASSERT_FLOAT_EQ( z_mirror(site,spin,color,0), 4.0f*r);
		ASSERT_FLOAT_EQ( z_mirror(site,spin,color,1), 4.0f*i);
	}
      }
    }
  }
  
}

TEST(Test4, OLatticeSpinorAdd)
{
  testOLatSpinorAdd();
}


void testOLatPropAdd(void)
{

  using LatticePropagator = OLattice<
    PMatrixLocal<
      PMatrixLocal<
	   RComplexLocal<float>,
	  3>,
    4>,
    TestMemSpace>;
  
  LatticePropagator x(20);
  LatticePropagator y(20);
  LatticePropagator z(20);
  
  
  // Poor man's fill: on-device
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
  Kokkos::fence();
  
  // The evaluate
  evaluate(z, (x + y) + y);
  
  {
    // Check -- always on the host
    auto z_mirror = Kokkos::create_mirror(z._data);
    Kokkos::deep_copy(z_mirror, z._data);
    
    for(size_t site=0; site < 20; ++site) {
      for(size_t spin2=0; spin2 < 4; ++spin2) {
	for(size_t spin1=0; spin1 < 4; ++spin1) {
	  for(size_t color2=0; color2 < 3; ++color2) {
	    for(size_t color1=0;color1 < 3; ++color1) {
	      
	      float r = static_cast<float>(0 + 2*(color1 + 3*(color2 + 3*(spin1 + 4*(spin2 + 4*site)))));
	      float i = static_cast<float>(1 + 2*(color1 + 3*(color2 + 3*(spin1 + 4*(spin2 + 4*site)))));
	      ASSERT_FLOAT_EQ( z_mirror(site,spin2,spin1,color2,color1,0), 5.0f*r);
	      ASSERT_FLOAT_EQ( z_mirror(site,spin2,spin1,color2,color1,1), 5.0f*i);
	    }
	  }
	}
      }
    }
  }
}


TEST(Test4, OLatticePropAdd)
{
  testOLatPropAdd();
}
