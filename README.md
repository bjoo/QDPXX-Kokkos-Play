QDPXXKokkosPlay
===============

Playpen for Kokkos/QDP things.

To build.

```mkdir ./workspace
   cd workspace
   mkdir src
   cd src
   git clone --recursive https://github.com/bjoo/QDPXXKokkosPlay.git 
   cd ../..
   # edit env.sh to set up your fave compiler like clang
   cp src/QDPXXKokkosPlay/env.sh .

   # This will do the building
   cp src/QDPXXKokkosPlay/build.sh .

   # Make sure you have a recent CMake available (3.13 at least)
   # Make a build directory
   mkdir build

   # Do the build
   ./build.sh
   
   # Run the tests
   cd ./build/tests
   ./test1
```
