#include <Kokkos_Core.hpp>
#include <config.hpp>
#include <data_out.hpp>
#include <dof_handler.hpp>
#include <timer.hpp>
#include <vector.hpp>

int
main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  {
    Triangulation<2> tria;
    tria.generate_hyper_cube();
    tria.refine_global(4);

    DataOut<2> data_out;
    data_out.write(tria, "test.vtu");
  }
  Kokkos::finalize();

  return 0;
}
