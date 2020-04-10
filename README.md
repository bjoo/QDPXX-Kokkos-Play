QDPXX-Kokkos-Play
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
   git clone --recursive https://github.com/bjoo/QDPXX-Kokkos-Play.git 

   # Back up to workspace
   cd ../..

   # Grab edit env.sh and edit it set up your fave compiler like clang
   # Also make sure you have a recent CMake available -- set that up in env.sh too
   # e.g for summit
   ln -s ./src/QDPXX-Kokkos-Play/scripts/env_summit.sh ./env.sh
   ln -s ./src/QDPXX-Kokkos-Play/scripts/build_summit_nvcc.sh .

   # Do the build
   ./build_summit_nvcc.sh
   

   cd ./build/tests
   ./test1
   ./test3 
   ./test_add_olattice
   ./bw_tests

```

Build for HIP 
------------- 

Same process as above except instead of `env_summit.sh` link `env_tulip_mi60.sh` to `./env.sh` 
and link `build_hip_tulip.sh` instead of `build_summit_nvsocal.sh`. On batch queue based systems you may need to submit the tests in a job script, or grab an interactive shell.