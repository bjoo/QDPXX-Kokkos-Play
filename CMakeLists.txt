cmake_minimum_required(VERSION 3.13)
project(QDPXXKokkosPlay VERSION 0.1
	DESCRIPTION "QDPXX-Kokkos-Playpen"
        LANGUAGES CXX)


# Basic Includes
include(GNUInstallDirs)
include(CTest)
include(CMakePackageConfigHelpers)

# Set yes/no options 
# OFF is default value

# Remember to change the file names
#configure_file(include/myproj_config.h.cmake.in include/myproj_config.h)

set(CMAKE_CXX_STANDARD 14)


#Make libs here 
#add_subdirectory(lib)

#Make tests here
add_subdirectory(test)
add_subdirectory(extern/kokkos)

# Install the headers
#install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION ${CMAKE_INSTALL_PREFIX})

# Install the generated headers
#install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/myproj_config.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include/)

# Set some variables to export to the config file if not already set
#set(MyProject_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include)
#message(STATUS "Setting MyProject_INCLUDE_DIR: ${MyProject_INCLUDE_DIR}")

# Make the config file
#configure_package_config_file(MyProjectConfig.cmake.in MyProjectConfig.cmake INSTALL_DESTINATION lib/cmake/MyProject PATH_VARS MyProject_INCLUDE_DIR)

# Make the version file
#write_basic_package_version_file(
#    MyProjectVersion.cmake
#    VERSION ${PACKAGE_VERSION}
#    COMPATIBILITY AnyNewerVersion
#    )

#install(FILES ${CMAKE_CURRENT_BINARY_DIR}/MyProjectVersion.cmake ${CMAKE_CURRENT_BINARY_DIR}/MyProjectConfig.cmake DESTINATION lib/cmake/MyProject)


#install(EXPORT MyProjectTargets NAMESPACE MyProject:: DESTINATION lib/cmake/MyProject )

# Export stuff
# This remains a mystery
#export(PACKAGE MyProject)
