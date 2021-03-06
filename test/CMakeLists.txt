
# Boilerplate from Introduction to modern CMake

# Add the Googletest directory and mark its targets etc as advanced
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/googletest" "extern/googletest")
mark_as_advanced(
    BUILD_GTEST BUILD_SHARED_LIBS
    gtest_build_samples gtest_build_tests
    gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
)

# Do this to Keep IDE's happy according to introduction to modern cmake
set_target_properties(gtest PROPERTIES FOLDER extern)

add_library( testutils test_env.cpp )


target_link_libraries( testutils Kokkos::kokkos gtest )

# This macro takes the testname and atts an executable from the argumnets
# (ARGN is every argument after the listed one) which allows you to pass a list
# of sources after the testname
macro(package_add_test TESTNAME)
    # Make the executable
    add_executable(${TESTNAME} ${ARGN})

    # Add on the gtest libs
    target_link_libraries(${TESTNAME} testutils)

    # Add the test to CTest
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
    target_include_directories(${TESTNAME} PUBLIC ${PROJECT_SOURCE_DIR}/include
		PUBLIC ${PROJECT_BUILD_DIR}/include)
    target_include_directories(${TESTNAME} PUBLIC ${PROJECT_SOURCE_DIR}/extern/simd-math)
		
    # I am not 100% sure what tis is for but it is a property
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
  
endmacro()

# Here TESTNAME is test1
# and ARGN is test1.cpp
package_add_test(test1 test1.cpp)
package_add_test(test3 test3.cpp)
package_add_test(test_add2 test_add2.cpp)
package_add_test(test_add_olattice test_add_olattice.cpp)
package_add_test(bw_tests bw_tests.cpp) 
