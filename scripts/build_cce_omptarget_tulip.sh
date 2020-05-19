source ./env.sh

CXX=CC

TOP=`pwd`
SRCDIR=${TOP}/src
BUILDDIR=${TOP}/build
#export PATH=${SRCDIR}/QDPXX-Kokkos-Play/extern/kokkos/bin:$PATH

if [ -d ${BUILDDIR} ];

then
	rm -rf ${BUILDDIR}
fi

mkdir -p ${BUILDDIR}
pushd ${BUILDDIR}

cmake \
	-DKokkos_ENABLE_SERIAL=ON \
	-DKokkos_ENABLE_HIP=OFF \
	-DKokkos_ENABLE_OPENMPTARGET=ON \
	-DKokkos_ARCH_VEGA906=ON \
	-DKokkos_ENABLE_COMPLEX_ALIGN=ON \
	-DCMAKE_CXX_COMPILER=${CXX} \
	-DCMAKE_CXX_FLAGS="-g -O3 -fopenmp" \
	-DBUILD_GMOCK=OFF \
	${SRCDIR}/QDPXX-Kokkos-Play


make -j 4 VERBOSE=1
