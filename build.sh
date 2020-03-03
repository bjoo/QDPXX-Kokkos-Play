source ./env.sh
CXX=clang++

cmake \
	-DKokkos_ENABLE_OPENMP=ON \
	-DCMAKE_CXX_COMPILER=${CXX} \
	-DCMAKE_CXX_FLAGS="-g -O3" \
	..

make -j 8
