#include "gtest/gtest.h"

#include "Kokkos_Macros.hpp"
#include "Kokkos_Core.hpp"
#include "test3.hpp"
#include "binop_add_olattice.hpp"

using namespace Playground;

constexpr size_t L = 44;
constexpr size_t Lx = L;
constexpr size_t Ly = L;
constexpr size_t Lz = L;
constexpr size_t Lt = L;
constexpr size_t Lxh = Lx/2;

constexpr size_t Vcb = Lxh*Ly*Lz*Lt;

constexpr size_t n_repeats = 5;
constexpr size_t n_warms = 10;
constexpr size_t n_iters = 100;

#if defined(KOKKOS_ENABLE_CUDA)
using TestMemSpace=Kokkos::CudaSpace;
#elif defined(KOKKOS_ENABLE_HIP)
using TestMemSpace = Kokkos::Experimental::HIPSpace;
#elif defined(KOKKOS_ENABLE_OPENMP)
using TestMemSpace = Kokkos::HostSpace;
#elif defined(KOKKOS_ENABLE_OPENMPTARGET)
using TestMemSpace = Kokkos::Experimental::OpenMPTargetSpace;
#endif

void testSpinorAddBW(void)
{

  using LatticeFermion = OLattice<
    PVectorLocal<
      PVectorLocal<
	    RComplexLocal<float>,
	  3>,
    4>,
    TestMemSpace>;
  
  LatticeFermion x(Vcb);
  LatticeFermion y(Vcb);
  LatticeFermion z(Vcb);

  // Poor man's fill
  Kokkos::parallel_for(Vcb,KOKKOS_LAMBDA(const size_t site) {
      for(size_t spin=0; spin < 4; ++spin) {
	for(size_t color=0; color < 3; ++color) {
	  float r = static_cast<float>(0 + 2*(color + 3*(spin + 4*site)));
	  float i = static_cast<float>(1 + 2*(color + 3*(spin + 4*site)));
	  x.elem(site).elem(spin).elem(color).real() =r;
	  x.elem(site).elem(spin).elem(color).imag() = i;
	  
	  y.elem(site).elem(spin).elem(color).real() = 2.0f*r;
	  y.elem(site).elem(spin).elem(color).imag() = 2.0f*i;
	  
	  z.elem(site).elem(spin).elem(color).real() = 0.0f;
	  z.elem(site).elem(spin).elem(color).imag() = 0.0f;
	}
      }
    });
  Kokkos::fence();

  // Some expression
  std::cout << "Warmups " << std::endl << std::flush;

  for(int i=0; i < n_warms; ++i ) {
    evaluate( z, x + y );
  }

  std::cout << "Timing " << std::endl << std::flush;

  Kokkos::Array<double, n_repeats> timers;

  for(int rep=0; rep < n_repeats; ++rep) {

    Kokkos::Timer t;
    t.reset();

    for(int i=0; i < n_iters; ++i ) {
      evaluate( z, x + y );
    }
    timers[rep] = t.seconds();
  }

  double average = timers[0];
  for(int t=1; t < n_repeats; ++t) { 
    average+=timers[t];
  }
  average /= static_cast<double>(n_repeats);
  
  std::cout << "Average  Time=" << average << " sec" <<std::endl;
  
  for(int rfo=0; rfo < 2; ++rfo) {
    double bytes_in = n_iters*(2+rfo)*Vcb*(4*3*2)*sizeof(float);
    double bytes_out= n_iters*Vcb*(4*3*2)*sizeof(float);
    double gibytes_in = bytes_in / static_cast<double>(1000*1000*1000);
    double gibytes_out = bytes_out / static_cast<double>(1000*1000*1000);

    std::cout << "RFO=" << rfo << " Read BW=" << gibytes_in/average << " GB/s "
	      << " Write BW=" << gibytes_out/average << " GB/s" 
              << " Total BW=" << (gibytes_out + gibytes_in) / average << " GB/s " 
	      << std::endl;
  } // Other RFO projection
}

TEST(TestBW, TestSpinorAddBW)
{
  testSpinorAddBW();

}

#if defined(KOKKOS_ENABLE_CUDA)
template<typename T>
using simd_t = simd::simd<T, simd::simd_abi::cuda_warp<32>>;
using simd_float = simd_t<float>;
using simd_double= simd_t<double>;

#elif defined(KOKKOS_ENABLE_HIP)
template<typename T>
using simd_t = simd::simd<T, simd::simd_abi::hip_wavefront<64>>;
using simd_float = simd_t<float>;
using simd_double= simd_t<double>;
#elif defined(KOKKOS_ENABLE_OPENMP)
template<typename T>
using simd_t = simd::simd<T, simd::simd_abi::native>;
using simd_float = simd_t<float>;
using simd_double= simd_t<double>;
#elif defined(KOKKOS_ENABLE_OPENMPTARGET)
template<typename T>
using simd_t = simd::simd<T, simd::simd_abi::native>;
using simd_float = simd_t<float>;
using simd_double= simd_t<double>;
#endif
void testSpinorAddBWSIMD(void)
{

  using LatticeFermion = OLattice<
    PVectorLocal<
      PVectorLocal<
	    RComplexLocal <simd_float >,
	  3>,
    4>,
    TestMemSpace>;

  constexpr size_t N=simd_float::size();

  LatticeFermion x(Vcb/N);
  LatticeFermion y(Vcb/N);
  LatticeFermion z(Vcb/N);

  // Poor man's fill
  Kokkos::parallel_for(Kokkos::TeamPolicy<>(Vcb/N,1,simd_float::size()),KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
	  const int site = team.league_rank();

	  for(size_t spin=0; spin < 4; ++spin) {
    	  for(size_t color=0; color < 3; ++color) {
    		  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,N),
    		           [&] (const int ii) {
    			  float r = static_cast<float>(ii+N*(0 + 2*(color + 3*(spin + 4*site))));
    			  float i = static_cast<float>(ii+N*(1 + 2*(color + 3*(spin + 4*site))));
    			  x.elem(site).elem(spin).elem(color).real()[ii] =r;
    			  x.elem(site).elem(spin).elem(color).imag()[ii] = i;

    			  y.elem(site).elem(spin).elem(color).real()[ii] = 2.0f*r;
    			  y.elem(site).elem(spin).elem(color).imag()[ii] = 2.0f*i;

    			  z.elem(site).elem(spin).elem(color).real()[ii] = 0.0f;
    			  z.elem(site).elem(spin).elem(color).imag()[ii] = 0.0f;
    		  });
    	  }
	  }
    });
  Kokkos::fence();

  // Some expression
  std::cout << "Warmups " << std::endl << std::flush;

  for(int i=0; i < n_warms; ++i ) {
    evaluate( z, x + y );
  }


  std::cout << "Timing " << std::endl << std::flush;

  Kokkos::Array<double, n_repeats> timers;

  for(int rep=0; rep < n_repeats; ++rep) {

    Kokkos::Timer t;
    t.reset();

    for(int i=0; i < n_iters; ++i ) {
      evaluate( z, x + y );
    }
    timers[rep] = t.seconds();
  }

  double average = timers[0];
  for(int t=1; t < n_repeats; ++t) {
    average+=timers[t];
  }
  average /= static_cast<double>(n_repeats);

  std::cout << "Average  Time=" << average << " sec" <<std::endl;

  for(int rfo=0; rfo < 2; ++rfo) {
    double bytes_in = n_iters*(2+rfo)*Vcb*(4*3*2)*sizeof(float);
    double bytes_out= n_iters*Vcb*(4*3*2)*sizeof(float);
    double gibytes_in = bytes_in / static_cast<double>(1000*1000*1000);
    double gibytes_out = bytes_out / static_cast<double>(1000*1000*1000);

    std::cout << "RFO=" << rfo << " Read BW=" << gibytes_in/average << " GB/s "
	      << " Write BW=" << gibytes_out/average << " GB/s"
              << " Total BW=" << (gibytes_out + gibytes_in) / average << " GB/s "
	      << std::endl;
  } // Other RFO projection

}

TEST(TestBW, TestSpinorAddBWSIMD)
{
  testSpinorAddBWSIMD();

}
