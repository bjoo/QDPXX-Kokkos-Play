#include "gtest/gtest.h"

#include "test.hpp"
#include "Kokkos_Core.hpp"

using namespace Playground;
TEST(Play, TestCreateLatScalarLeft)
{

	OLatticeView<RScalarView, IndexOrderLeft<1>> latscalar(24);

	for(int i=0; i < 24; ++i) {
		ASSERT_EQ( latscalar.offset(i), i );
	}
}

TEST(Play, TestCreateLatVectorScalarLeft)
{
	OLatticeView< PVectorView< RScalarView, 3, IndexOrderLeft<1>>, IndexOrderLeft<1>> lat3vec(16);

	for(int site=0; site < 16; ++site) {
		for(int color=0; color < 3; ++color) {
			ASSERT_EQ( lat3vec.subview(site).offset(color), site+16*color );
			ASSERT_EQ( lat3vec.subview(site).subview(color).offset(), site+16*color );
		}
	}

}


TEST(Play, TestCreateLatMatScalarLeft)
{
	OLatticeView< PMatrixView< RScalarView, 3, IndexOrderLeft<2>>, IndexOrderLeft<1>> lat3vec(16);

	for(int site=0; site < 16; ++site) {
		for(int row=0; row < 3; ++row) {
			for(int col=0; col < 3; ++col) {
				ASSERT_EQ( lat3vec.subview(site).offset(row,col), site+16*(row + 3*col) );
				ASSERT_EQ( lat3vec.subview(site).subview(row,col).offset(), site+16*(row+3*col) );
			}
		}
	}
}

TEST(Play, TestCreateMatMatViewLeft)
{
	OLatticeView<PMatrixView<PMatrixView<RScalarView, 3, IndexOrderLeft<2>>, 4, IndexOrderLeft<2>>,IndexOrderLeft<1>> lat3vec(16);

	for(int site=0; site < 16; ++site) {
		for(int srow=0; srow < 4; ++srow) {
		  for(int scol=0; scol < 4; ++scol ) {

		     for(int crow=0; crow < 3; ++crow) {
			   for(int ccol=0; ccol < 3; ++ccol) {
				 ASSERT_EQ( lat3vec.subview(site).subview(srow,scol).offset(crow,ccol),
						 site+16*(srow + 4*(scol + 4*(crow + 3*ccol))) );
				 ASSERT_EQ( lat3vec.subview(site).subview(srow,scol).subview(crow,ccol).offset(),
						 site+16*(srow + 4*(scol + 4*(crow +3*ccol))) );
			   }
		     }
		  }
		}
	}
}

TEST(Play, TestCreatePropViewLeft)
{
	OLatticeView<PMatrixView<PMatrixView<RComplexView<IndexOrderLeft<1>>, 3, IndexOrderLeft<2>>, 4, IndexOrderLeft<2>>,IndexOrderLeft<1>> lat3vec(16);

	for(int site=0; site < 16; ++site) {
		for(int srow=0; srow < 4; ++srow) {
		  for(int scol=0; scol < 4; ++scol ) {

		     for(int crow=0; crow < 3; ++crow) {
			   for(int ccol=0; ccol < 3; ++ccol) {

				 ASSERT_EQ( lat3vec.subview(site).subview(srow,scol).subview(crow,ccol).offsetReal(),
						 site+16*(srow + 4*(scol + 4*(crow +3*(ccol + 3*0)))) );
				 ASSERT_EQ( lat3vec.subview(site).subview(srow,scol).subview(crow,ccol).offsetImag(),
				 		site+16*(srow + 4*(scol + 4*(crow +3*(ccol + 3*1)))) );
			   }
		     }
		  }
		}
	}
}

TEST(Play, TestCreatePropViewLeft2)
{
	OLatticeView<PMatrixView<PMatrixView<RComplexView<IndexOrderLeft<1>>, 3, IndexOrderLeft<2>>, 4, IndexOrderLeft<2>>,IndexOrderLeft<1>> lat3vec(16);

	for(int site=0; site < 16; ++site) {
		for(int srow=0; srow < 4; ++srow) {
		  for(int scol=0; scol < 4; ++scol ) {

		     for(int crow=0; crow < 3; ++crow) {
			   for(int ccol=0; ccol < 3; ++ccol) {

				 ASSERT_EQ( lat3vec.subview(site).subview(srow,scol).subview(crow,ccol).offset(0),
						 site+16*(srow + 4*(scol + 4*(crow +3*(ccol + 3*0)))) );
				 ASSERT_EQ( lat3vec.subview(site).subview(srow,scol).subview(crow,ccol).offset(1),
				 		site+16*(srow + 4*(scol + 4*(crow +3*(ccol + 3*1)))) );
			   }
		     }
		  }
		}
	}
}


TEST(Play, TestCreateLatScalarRight)
{

	OLatticeView<RScalarView, IndexOrderRight<1>> latscalar(24);

	for(int i=0; i < 24; ++i) {
		ASSERT_EQ( latscalar.offset(i), i );
	}
}

TEST(Play, TestCreateLatVectorScalarRight)
{
	OLatticeView< PVectorView< RScalarView, 3, IndexOrderRight<1>>, IndexOrderRight<1>> lat3vec(16);

	for(int site=0; site < 16; ++site) {
		for(int color=0; color < 3; ++color) {
			ASSERT_EQ( lat3vec.subview(site).offset(color), 3*site+color );
			ASSERT_EQ( lat3vec.subview(site).subview(color).offset(), 3*site+color );
		}
	}

}


TEST(Play, TestCreateLatMatScalarRight)
{
	OLatticeView< PMatrixView< RScalarView, 3, IndexOrderRight<2>>, IndexOrderRight<1>> lat3vec(16);

	for(int site=0; site < 16; ++site) {
		for(int row=0; row < 3; ++row) {
			for(int col=0; col < 3; ++col) {
				ASSERT_EQ( lat3vec.subview(site).offset(row,col), col + 3*(row + 3*site) );
				ASSERT_EQ( lat3vec.subview(site).subview(row,col).offset(), col+3*(row + 3*site) );
			}
		}
	}
}


TEST(Play, TestCreateMatMatViewRight)
{
	OLatticeView<PMatrixView<PMatrixView<RScalarView, 3, IndexOrderRight<2>>, 4, IndexOrderRight<2>>,
	IndexOrderRight<1>> lat3vec(16);

	for(int site=0; site < 16; ++site) {
		for(int srow=0; srow < 4; ++srow) {
		  for(int scol=0; scol < 4; ++scol ) {

		     for(int crow=0; crow < 3; ++crow) {
			   for(int ccol=0; ccol < 3; ++ccol) {
				 ASSERT_EQ( lat3vec.subview(site).subview(srow,scol).offset(crow,ccol),
						 ccol + 3*(crow + 3*(scol + 4*(srow + 4*site))));
				 ASSERT_EQ( lat3vec.subview(site).subview(srow,scol).subview(crow,ccol).offset(),
						 ccol + 3*(crow + 3*(scol + 4*(srow + 4*site))));
			   }
		     }
		  }
		}
	}
}

TEST(Play, TestCreatePropViewRight)
{
	OLatticeView<PMatrixView<PMatrixView<RComplexView<IndexOrderRight<1>>, 3, IndexOrderRight<2>>, 4, IndexOrderRight<2>>,
	IndexOrderRight<1>> lat3vec(16);

	for(int site=0; site < 16; ++site) {
		for(int srow=0; srow < 4; ++srow) {
		  for(int scol=0; scol < 4; ++scol ) {

		     for(int crow=0; crow < 3; ++crow) {
			   for(int ccol=0; ccol < 3; ++ccol) {
				 ASSERT_EQ( lat3vec.subview(site).subview(srow,scol).subview(crow,ccol).offsetReal(),
						 0 + 2*(ccol + 3*(crow + 3*(scol + 4*(srow + 4*site)))));
				 ASSERT_EQ( lat3vec.subview(site).subview(srow,scol).subview(crow,ccol).offsetImag(),
						 1 + 2*(ccol + 3*(crow + 3*(scol + 4*(srow + 4*site)))));
			   }
		     }
		  }
		}
	}
}

TEST(Play, TestCreatePropViewRight2)
{
	OLatticeView<PMatrixView<PMatrixView<RComplexView<IndexOrderRight<1>>, 3, IndexOrderRight<2>>, 4, IndexOrderRight<2>>,
	IndexOrderRight<1>> lat3vec(16);

	for(int site=0; site < 16; ++site) {
		for(int srow=0; srow < 4; ++srow) {
		  for(int scol=0; scol < 4; ++scol ) {

		     for(int crow=0; crow < 3; ++crow) {
			   for(int ccol=0; ccol < 3; ++ccol) {
				 ASSERT_EQ( lat3vec.subview(site).subview(srow,scol).subview(crow,ccol).offset(0),
						 0 + 2*(ccol + 3*(crow + 3*(scol + 4*(srow + 4*site)))));
				 ASSERT_EQ( lat3vec.subview(site).subview(srow,scol).subview(crow,ccol).offset(1),
						 1 + 2*(ccol + 3*(crow + 3*(scol + 4*(srow + 4*site)))));
			   }
		     }
		  }
		}
	}
}

TEST(Play, TestCountDimsOLatScalar)
{
	using Foo = OLatticeView<RScalarView,IndexOrderRight<1>>;
	ASSERT_EQ( Foo::num_dims(), 1);
}

TEST(Play, TestCountDimsOLatVectorComplex)
{
	using Foo = OLatticeView<PVectorView<RComplexView<IndexOrderRight<1>>,3,IndexOrderRight<1>>,IndexOrderRight<1>>;
	ASSERT_EQ( Foo::num_dims(),3);
}

TEST(Play, TestCountDimsOLatMatrixComplex)
{
	using Foo = OLatticeView<PMatrixView<RComplexView<IndexOrderRight<1>>,3,IndexOrderRight<1>>,IndexOrderRight<1>>;
	ASSERT_EQ( Foo::num_dims(),4);
}

TEST(Play, TestCountDimsOLatVectorVectorComplex)
{
	using Foo = OLatticeView<PVectorView<PVectorView<RComplexView<IndexOrderRight<1>>,4,IndexOrderRight<1>>,3,IndexOrderRight<1>>,IndexOrderRight<1>>;
	ASSERT_EQ( Foo::num_dims(),4);
}

TEST(Play, TestCountDimsOLatPropComplex)
{
	using Foo = OLatticeView<PMatrixView<PMatrixView<RComplexView<IndexOrderRight<1>>,4,IndexOrderRight<1>>,3,IndexOrderRight<1>>,IndexOrderRight<1>>;
	ASSERT_EQ( Foo::num_dims(),6);
}

TEST(Play, TestGetDimsOLatticeScalar)
{
	using Foo=OLatticeView<RScalarView,IndexOrderLeft<1>>;
	Foo testlat(20);
	auto dims=testlat.get_dims();
	std::cout << "Dims size=" << dims.size() << std::endl;
	std::cout << "{ ";
	for(int i=0; i < Foo::num_dims(); ++i) {
		std::cout << dims[i] << " ";
	}
	std::cout << "}\n";
	ASSERT_EQ(Foo::num_dims(),1);
	ASSERT_EQ(dims[0], 20);

}

TEST(Play, TestGetDimsOLatticePropagator)
{
	using Foo = OLatticeView<PMatrixView<PMatrixView<RComplexView<IndexOrderRight<1>>,4,IndexOrderRight<1>>,3,IndexOrderRight<1>>,IndexOrderRight<1>>;

	Foo testlat(20);
	auto dims=testlat.get_dims();
	std::cout << "Dims size=" << dims.size() << std::endl;
	std::cout << "{ ";
	for(int i=0; i < dims.size(); ++i) {
		std::cout << dims[i] << " ";
	}
	std::cout << "}\n";

	std::array<std::size_t,8> expected_dims{20,3,3,4,4,2,1,1};
	for(int i=0; i < Foo::num_dims(); ++i) {
			ASSERT_EQ(dims[i],expected_dims[i]);
	}

}

TEST(KokkosViewTests, KokkosLayoutSetup)
{
	using Foo = OLatticeView<PMatrixView<PMatrixView<RComplexView<IndexOrderRight<1>>,4,IndexOrderRight<1>>,3,IndexOrderRight<1>>,IndexOrderRight<1>>;

	Foo testlat(20);
	auto dims=testlat.get_dims();

	Kokkos::LayoutLeft layout(1,1,1,1,1,1,1,1);

	for(int i=0; i < dims.size();i++) {
		layout.dimension[i] = dims[i];
	}
	for(int i=0; i < 8; ++i) {
		if( i < Foo::num_dims() ) {
			ASSERT_EQ(layout.dimension[i], dims[i]);
		}
		else {
			ASSERT_EQ(layout.dimension[i], 1);
		}
	}
}

TEST(KokkosViewTests, KokkosCheckoutGetView)
{

	// Horrible nested type corresponding to 6 indices
	using Foo = OLatticeView<PMatrixView<PMatrixView<RComplexView<IndexOrderRight<1>>,4,IndexOrderRight<1>>,3,IndexOrderRight<1>>,IndexOrderRight<1>>;

	// So this is a specification
	Foo testlat(20);

	// This gets the dimensions into an array
	auto dims=testlat.get_dims();

	// I can fill out the layout with the numbers from the array
	Kokkos::LayoutLeft layout{dims[0],dims[1],dims[2],dims[3],dims[4],dims[5],dims[6],dims[7]};

	// This converts the base type of 'float' to a float****** for the dimensions
	using ViewType = KokkosDimensionTrait<float,Foo::num_dims()>::Type_t;

	// Now how to create the view
	Kokkos::View<ViewType,decltype(layout)> my_view("a", layout);

	for(int i=0; i < Foo::num_dims(); ++i)
		ASSERT_EQ(my_view.extent(i), dims[i]);

	ASSERT_EQ( my_view.span(), 20*3*3*4*4*2);

}

