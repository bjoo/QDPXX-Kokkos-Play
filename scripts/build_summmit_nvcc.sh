source ./env.sh

CXX=nvcc_wrapper

TOP=`pwd`
SRCDIR=${TOP}/src
BUILDDIR=${TOP}/build
export PATH=${SRCDIR}/QDPXX-Kokkos-Play/extern/kokkos/bin:$PATH

if [ -d ${BUILDDIR} ];

then
	rm -rf ${BUILDDIR}
fi

mkdir -p ${BUILDDIR}
pushd ${BUILDDIR}

cmake \
	-DKokkos_ENABLE_SERIAL=ON \
	-DKokkos_ENABLE_CUDA=ON \
	-DKokkos_ARCH_VOLTA70=ON \
	-DKokkos_ENABLE_CUDA_LAMBDA=ON \
	-DKokkos_ENABLE_CUDA_CONSTEXPR=ON \
	-DCMAKE_CXX_COMPILER=${CXX} \
	-DCMAKE_CXX_FLAGS="-g -O3" \
	-DBUILD_GMOCK=OFF \
	${SRCDIR}/QDPXX-Kokkos-Play


make -j 4 VERBOSE=1
