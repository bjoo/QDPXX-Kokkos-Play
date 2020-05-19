module load PrgEnv-cray
module load rocm
module load cce
module load craype-accel-amd-gfx906
module load cmake

TOPDIR=`pwd`

BUILDDIR=${TOPDIR}/build
INSTALLDIR=${TOPDIR}/install
SRCDIR=${TOPDIR}/src

PK_CXX="CC"
PK_CC="cc"
PK_OMP_ENABLE=""
PK_CXXFLAGS=" -O3 -fopenmp"
#PK_CXXFLAGS_NVCC=${PK_CXXFLAGS}" --expt-extended-lambda --expt-relaxed-constexpr"
PK_CXXFLAGS_NVCC="${PK_CXXFLAGS}"
PK_CFLAGS=" -O3  "
MAKE="make -j 8"


