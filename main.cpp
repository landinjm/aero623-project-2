#include <Kokkos_Core.hpp>
#include <config.hpp>
#include <data_out.hpp>
#include <read_gri.hpp>
#include <timer.hpp>
#include <triangulation.hpp>
#include <vector.hpp>

int
main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  {
    GriReader<2> gri;
    Triangulation<2> tria;

    gri.read_gri("../grids/coarse_local_refinement_2.gri");
    gri.transfer_to_triangulation(tria);

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
