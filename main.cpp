#include <Kokkos_Core.hpp>
#include <config.hpp>
#include <timer.hpp>
#include <vector.hpp>

int
main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  {
    Vector<double, DeviceMemSpace> vec_1;
    vec_1.reinit(10);
    vec_1 = 10.0;
    Vector<double, DeviceMemSpace> vec_2(vec_1);

    std::cout << (vec_1 == vec_2) << std::endl;
  }
  Kokkos::finalize();

  return 0;
}
