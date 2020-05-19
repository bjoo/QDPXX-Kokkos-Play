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
	-DKokkos_ENABLE_COMPLEX_ALIGN=ON \
	-DCMAKE_CXX_COMPILER=${CXX} \
	-DCMAKE_CXX_FLAGS="-O3 --amdgpu-target=gfx906 -Wno-error=unused-command-line-argument" \
	-DBUILD_GMOCK=OFF \
	-DCMAKE_BUILD_TYPE=Debug \
	${SRCDIR}/QDPXX-Kokkos-Play


make -j 4 VERBOSE=1
