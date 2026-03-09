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

    Vector<double, HostMemSpace> solution(tria.n_cells());

    DataOut<2> data_out;
    data_out.clear();
    data_out.attach_triangulation(tria);
    data_out.set_time(0.0);
    data_out.set_cycle(0);

    data_out.add_data_vector(solution, "solution");

    data_out.write_vtu("test.vtu");
  }
  Kokkos::finalize();

  return 0;
}
