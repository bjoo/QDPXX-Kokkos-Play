source ./env.sh
CXX=clang++

TOP=`pwd`
SRCDIR=${TOP}/src
BUILDDIR=${TOP}/build

cd ${BUILDDIR}
cmake \
	-DKokkos_ENABLE_OPENMP=ON \
	-DCMAKE_CXX_COMPILER=g++\
	-DCMAKE_CXX_FLAGS="-g -O3 -march=native" \
	-G"Eclipse CDT4 - Unix Makefiles" \
    	-DCMAKE_ECLIPSE_MAKE_ARGUMENTS=-j8 \
    	-DCMAKE_ECLIPSE_VERSION=4.5.0 \
	${SRCDIR}/QDPXX-Kokkos-Play/

make -j 8
