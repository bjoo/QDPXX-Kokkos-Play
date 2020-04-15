/*
 * test2.cpp
 *
 *  Created on: Mar 17, 2020
 *      Author: bjoo
 */

#include "gtest/gtest.h"

#include "test3.hpp"
#include "Kokkos_Macros.hpp"

using namespace Playground;

#if defined(KOKKOS_ENABLE_CUDA)
using TestMemSpace=Kokkos::CudaUVMSpace;
#elif defined(KOKKOS_ENABLE_HIP)
using TestMemSpace = Kokkos::Experimental::HIPHostPinnedSpace;
#elif defined(KOKKOS_ENABLE_OPENMP)
using TestMemSpace = Kokkos::HostSpace;
#endif

TEST(Test3, TestRScalar)
{
	using storage = Kokkos::View<float[3],TestMemSpace>;
	storage a_storage("a");
	
	a_storage(0)=1.5;
	a_storage(1)=2.5;
	a_storage(2)=3.5;
	

	KokkosIndices indices{2,0,0,0, 0,0,0,0};
	RScalar<float,storage,1 > a(a_storage, indices);

	// Needs to be done on device
	ASSERT_FLOAT_EQ( a.elem(), a_storage(2));
	a.elem() = 3.2;
	ASSERT_FLOAT_EQ( a_storage(2), 3.2);

}

TEST(Test3, TestRScalarFromLocal)
{
	using storage = Kokkos::View<float[3],TestMemSpace>;
	storage a_storage("a");
	a_storage(0)=1.5;



	a_storage(1)=2.5;
	a_storage(2)=3.5;

	KokkosIndices indices{2,0,0,0, 0,0,0,0};
	RScalar<float,storage,1 > a(a_storage, indices);

	RScalarLocal<float> local(0.5);

	a = local;

	ASSERT_FLOAT_EQ( a_storage(2),0.5);

	a_storage(2) = 0.35;
	local = a;

	ASSERT_FLOAT_EQ( local.elem(), 0.35);
}

TEST(Test3, TestRComplex)
{
	using storage = typename Kokkos::View<float[2],TestMemSpace>;
	storage a_storage("a");
	a_storage(0)=1.5;
	a_storage(1)=2.5;

	KokkosIndices indices{0,0,0,0, 0,0,0,0};
	RComplex<float,storage,0> a(a_storage, indices);

	ASSERT_FLOAT_EQ( a.real(), a_storage(0));
	ASSERT_FLOAT_EQ( a.imag(), a_storage(1));

	a.real() = 3.9;
	a.imag() = 4.2;

	ASSERT_FLOAT_EQ( a_storage(0),3.9);
	ASSERT_FLOAT_EQ( a_storage(1), 4.2);
}

TEST(Test3, TestRPVectorScalar)
{
	Kokkos::View<float[3],TestMemSpace> a_storage("a");
		a_storage(0)=1.5;
		a_storage(1)=2.5;
		a_storage(2)=3.5;

	KokkosIndices indices{0,0,0,0, 0,0,0,0};
	using storage = typename Kokkos::View<float[3],TestMemSpace>;
	PVector< RScalarLocal<float>, storage, 4, 0> a(a_storage,indices);

	ASSERT_FLOAT_EQ( a.elem(2).elem(), a_storage(2));

	a.elem(1).elem() = 7.2;
	ASSERT_FLOAT_EQ( a_storage(1), 7.2);
}

TEST(Test3, TestPScalarPVectorScalar)
{
	using array_type = float[3][2];
	using test_type = PScalarLocal< PVectorLocal < RComplexLocal< float >,3>>;
	bool value =  std::is_same< array_type, test_type::array_type>::value;
	ASSERT_TRUE(value);
}

TEST(Test3, TestPScalar2)
{	using s_type = Kokkos::View<float[10], Kokkos::HostSpace>;

	s_type a_storage("a");
	for(int i=0; i < 10; ++i) {
		a_storage(i) = i;
	}

	KokkosIndices ind={0,0,0,0,0,0,0,0};
	PScalar<
	  PScalarLocal<
	    PVectorLocal< RScalarLocal<float >,10>
	  >,
	s_type,0 >  foo(a_storage,ind);

	for(int i=0; i < 10; ++i) {
		ASSERT_FLOAT_EQ( foo.elem().elem().elem(i).elem(), a_storage(i));
	}
}


TEST(Test3, TestPMatrixScalar)
{
	using storage = typename Kokkos::View<float[3][4],TestMemSpace>;
	storage a_storage("a");
	for(int j=0; j < 4; ++j) {
		for(int i=0; i < 3; ++i ) {
			a_storage(i,j) = i+3*j;
		}
	}


	KokkosIndices indices{0,0,0,0, 0,0,0,0};
	PMatrix< RScalarLocal<float>, storage, 4, 0> a(a_storage, indices);

	ASSERT_FLOAT_EQ( a.elem(1,2).elem(), a_storage(1,2));


}

TEST(Test3, TestVecVecScalar)
{
	using storage = typename Kokkos::View<float[3][4],TestMemSpace>;
	storage a_storage("a");
	for(int j=0; j < 4; ++j) {
		for(int i=0; i < 3; ++i ) {
			a_storage(i,j) = i+3*j;
		}
	}


	KokkosIndices  indices{0,0,0,0, 0,0,0,0};
	PVector< PVectorLocal< RScalarLocal<float>,3>, storage, 4, 0> a(a_storage, indices);

	ASSERT_FLOAT_EQ( a.elem(1).elem(2).elem(), a_storage(1,2));


}


TEST(Test3, TestSiteProp)
{
	using storage =typename Kokkos::View<float[2][3][3][4][4],TestMemSpace>;
	storage prop_storage("p");

	for(int spin2=0; spin2 < 4; ++spin2) {
		for(int spin1=0; spin1 < 4; ++spin1 ) {
			for(int col2=0; col2 < 3; ++col2) {
				for(int col1=0; col1 < 3; ++col1) {
					for(int reim=0; reim < 2; ++reim ) {
						prop_storage(spin2,spin1,col2,col1,reim) =
							static_cast<float>(reim + 2*(col1 + 3*(col2 + 3*(spin1 + 4*spin2))));
					}
				}
			}
		}
	}

	KokkosIndices indices{0,0,0,0, 0,0,0,0};
	using PropType =  PMatrix<
			           PMatrixLocal<
					     RComplexLocal<float>,  // Complex index 4, altogether 5 dims.
					   3>,   // colormatrix: dim=3, indices 2,3
			           storage, 4, 0>;  // spinmatrix: dim=4, indices 0,1

	PropType p(prop_storage, indices);


	for(int spin2=0; spin2 < 4; ++spin2) {
		for(int spin1=0; spin1 < 4; ++spin1 ) {
			for(int col2=0; col2 < 3; ++col2) {
				for(int col1=0; col1 < 3; ++col1) {
					for(int reim=0; reim < 2; ++reim ) {
						float result = prop_storage(spin2,spin1,col2,col1,reim);


						float res2 =  reim == 0 ? p.elem(spin2,spin1).elem(col2,col1).real()
								: p.elem(spin2,spin1).elem(col2,col1).imag();

						ASSERT_FLOAT_EQ(result,res2);
					}
				}
			}
		}
	}
}

void testLatColorMatrix(void)
{
	using storage=typename Kokkos::View<float*[3][3], TestMemSpace>;

	storage ref_storage("ref",20);
	OLattice< PScalarLocal< PMatrixLocal< RScalarLocal<float>, 3> >, TestMemSpace> latcm(20);

	Kokkos::parallel_for(20, KOKKOS_LAMBDA( const size_t site){
		for(int i=0; i < 3; ++i) {
			for(int j=0; j < 3; ++j ) {
				ref_storage(site,i,j) = static_cast<float>( site + 20*(i + 3*j));
				latcm.elem(site).elem().elem(i,j).elem() = static_cast<float>( site + 20*(i + 3*j));
			}
		}

	});
	Kokkos::fence();


	for(int site=0; site < 20; site++) {
		for(int i=0; i < 3; ++i) {
			for(int j=0; j < 3; ++j ) {
				ref_storage(site,i,j) = static_cast<float>( site + 20*(i + 3*j));
				latcm.elem(site).elem().elem(i,j).elem() = static_cast<float>( site + 20*(i + 3*j));
			}
		}

		}
}

TEST(Test3, testLatColMatrix)
{
	testLatColorMatrix();
}
void testLatTestProp(void)
{
  using storage =typename Kokkos::View<float*[4][4][3][3][2],TestMemSpace>;
  storage prop_storage("p", 20); // 20 sites
  storage ref_storage("p_ref", 20);
	
  Kokkos::parallel_for( 20, KOKKOS_LAMBDA(const int site){

      for(int spin2=0; spin2 < 4; ++spin2) {
	for(int spin1=0; spin1 < 4; ++spin1 ) {
	  for(int col2=0; col2 < 3; ++col2) {
	    for(int col1=0; col1 < 3; ++col1) {
	      for(int reim=0; reim < 2; ++reim ) {

		float i =  static_cast<float>(reim + 2*(col1 + 3*(col2	
						   + 3*(spin1 + 4*(spin2+4*site)))));
		
		prop_storage(site,spin2,spin1,col2,col1,reim) = i;
		ref_storage(site,spin2,spin1,col2,col1,reim) = 0.5*i + 2.6;
	      }
	    }
	  }
	}
      }
    });

  Kokkos::fence();
  
  using PropType =  OLattice<
    PMatrixLocal<
      PMatrixLocal<
	    RComplexLocal<float>,
				  3>,
				4>, TestMemSpace>;
  
  PropType p(prop_storage);

  Kokkos::parallel_for(20, KOKKOS_LAMBDA(const int site) {
      for(int spin2=0; spin2 < 4; ++spin2) {
	for(int spin1=0; spin1 < 4; ++spin1 ) {
	  for(int col2=0; col2 < 3; ++col2) {
	    for(int col1=0; col1 < 3; ++col1) {
	      
	      p.elem(site).elem(spin2,spin1).elem(col2,col1).real() *= 0.5;
	      p.elem(site).elem(spin2,spin1).elem(col2,col1).real() += 2.6;
	      
	      p.elem(site).elem(spin2,spin1).elem(col2,col1).imag() *= 0.5;
	      p.elem(site).elem(spin2,spin1).elem(col2,col1).imag() += 2.6;
	      
	    }
	    
	  }
	}
      }
    });
  Kokkos::fence(); // Fence is necessary
  
  // Checking is off device
  for(int site=0; site < 20; ++site) {
    for(int spin2=0; spin2 < 4; ++spin2) {
      for(int spin1=0; spin1 < 4; ++spin1 ) {
	for(int col2=0; col2 < 3; ++col2) {
	  for(int col1=0; col1 < 3; ++col1) {
	    for(int reim=0; reim < 2; ++reim) {
	      ASSERT_FLOAT_EQ( prop_storage(site,spin2,spin1,col2,col1,reim),
			       ref_storage(site,spin2,spin1,col2,col1,reim) );
	    }
	  }
	}
      }
    }
  }
}

TEST(Test3, TestLatPropProp)
{

  testLatTestProp();

}
