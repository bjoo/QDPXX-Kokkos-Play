#include "gtest/gtest.h"

#include "Kokkos_Macros.hpp"
#include "Kokkos_Core.hpp"
#include "test3.hpp"
#include "binop_add_olattice.hpp"

using namespace Playground;

constexpr size_t Lx = 32;
constexpr size_t Ly = 32;
constexpr size_t Lz = 32;
constexpr size_t Lt = 32;
constexpr size_t Lxh = Lx/2;

constexpr size_t Vcb = Lxh*Ly*Lz*Lt;

constexpr size_t n_repeats = 3;
constexpr size_t n_warms = 5;
constexpr size_t n_iters = 100;

#if defined(KOKKOS_ENABLE_CUDA)
using TestMemSpace=Kokkos::CudaSpace;
#elif defined(KOKKOS_ENABLE_HIP)
using TestMemSpace = Kokkos::Experimental::HIPSpace;
#elif defined(KOKKOS_ENABLE_OPENMP)
using TestMemSpace = Kokkos::HostSpace;
#endif

void testSpinorAddBW(void)
{

  using storage=typename Kokkos::View<float*[4][3][2],TestMemSpace>;
  using LatticeFermion = OLattice<
    PVector<
      PVector<
	RComplex<float,storage,3,4>,
	storage,3,2>,
      storage,4,1 >,
    storage,0>;
  
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
