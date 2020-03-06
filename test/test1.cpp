#include "gtest/gtest.h"

#include "test.hpp"

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

#if 1
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
#endif

#if 1
TEST(Play, TestCreatePropViewLeft)
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
#endif
#if 1
using namespace Playground;
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
#endif

#if 1
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
#endif

TEST(Play, TestCreatePropViewRight)
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
