/*
 * test_add2.cpp
 *
 *  Created on: Mar 30, 2020
 *      Author: bjoo
 */



#include "gtest/gtest.h"

#include "binop_add.hpp"
#include "Kokkos_Core.hpp"
#include <type_traits>
using namespace Playground;

using TestMemSpace=Kokkos::CudaUVMSpace;
TEST(ExprTests, CreateRLeaf)
{
	using storage_type = typename Kokkos::View<float[1],TestMemSpace>;
	storage_type as("a");
	storage_type bs("b");
	storage_type cs("c");
	as(0)=0.5;
	bs(0)=0.6;
	cs(0)=1.2; // Over write later


	RScalar<float,storage_type> a(as);
	RScalar<float,storage_type> b(bs);
	RScalar<float,storage_type> c(cs);

  	// RLeaf<decltype(a)>  a_expr(a);
	RLeaf<decltype(a)> a_expr(a);

	// a_expr() should return a_expr's value which is a
	bool type_correct =  std::is_same< decltype(a_expr()), RScalar<float,storage_type,1> >::value ;
	ASSERT_TRUE( type_correct );

	b = a_expr();
	std::cout << "a_expr.value is " << b.elem() << "\n";
  	ASSERT_FLOAT_EQ( b.elem(), a.elem());

  	RLeaf<decltype(b)> b_expr(b);

  	auto sum = a_expr + b_expr;


  	auto sum2 = a_expr + sum;
  	bool sum2_check = std::is_same<decltype(sum2()), RScalarLocal<float>>::value ;
  	std::cout << " sum_check " << sum2_check << std::endl;

  	// View from RScalarLocal
  	c = sum2();

  	std::cout << "c = " << c.elem() << "\n";


}
