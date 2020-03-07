QDPXXKokkosPlay
===============

Playpen for Kokkos/QDP things.

To build.

```
   # Create some workspace directory 
   mkdir ./workspace

   # Go into it
   cd workspace

   # Grab the source (use --recursive to get googletest and Kokkos too)
   mkdir src
   cd src
   git clone --recursive https://github.com/bjoo/QDPXXKokkosPlay.git 

   # Back up to workspace
   cd ../..

   # Grab edit env.sh and edit it set up your fave compiler like clang
   # Also make sure you have a recent CMake available -- set that up in env.sh too
   cp src/QDPXXKokkosPlay/env.sh .

   # This will do the building, so grab it too
   cp src/QDPXXKokkosPlay/build.sh .

   # Make a build directory
   mkdir build

   # Do the build
   ./build.sh
   
   # Run the tests
   cd ./build/tests
   ./test1
```
