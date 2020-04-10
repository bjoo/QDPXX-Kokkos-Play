source ./env.sh

CXX=hipcc

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
	-DKokkos_ENABLE_HIP=ON \
	-DKokkos_ARCH_VEGA906=ON \
	-DCMAKE_CXX_COMPILER=${CXX} \
	-DCMAKE_CXX_FLAGS="-O3" \
	-DBUILD_GMOCK=OFF \
	-DCMAKE_BUILD_TYPE=Release \
	${SRCDIR}/QDPXX-Kokkos-Play


make -j 4 VERBOSE=1
