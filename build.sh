source ./env.sh
CXX=clang++

TOP=`pwd`
SRCDIR=${TOP}/src
BUILDDIR=${TOP}/build

cd ${BUILDDIR}
cmake \
	-DKokkos_ENABLE_OPENMP=ON \
	-DCMAKE_CXX_COMPILER=${CXX} \
	-DCMAKE_CXX_FLAGS="-g -O3" \
	-G"Eclipse CDT4 - Unix Makefiles" \
    	-DCMAKE_ECLIPSE_MAKE_ARGUMENTS=-j8 \
    	-DCMAKE_ECLIPSE_VERSION=4.5.0 \
	${SRCDIR}/QDPXX-Kokkos-Play/

make -j 8
