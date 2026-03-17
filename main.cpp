#include <Kokkos_Core.hpp>
#include <config.hpp>
#include <data_out.hpp>
#include <dof_handler.hpp>
#include <fe.hpp>
#include <read_gri.hpp>
#include <timer.hpp>
#include <triangulation.hpp>
#include <vector.hpp>

static constexpr unsigned int problem_degree = 1;

int
main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  {
    GriReader<2> gri;
    Triangulation<2> tria;
    FE_DGQLegendre<2, double> fe(problem_degree);
    QGaussSimplex<2, double> quad(problem_degree + 1);

    gri.read_gri("../grids/coarse_local_refinement_1.gri");
    gri.transfer_to_triangulation(tria);

    DoFHandler<2, double> dof_handler(tria, fe);

    std::cout << "Number of cells: " << tria.n_cells() << "\n";
    std::cout << "Number of DoFs: " << dof_handler.n_dofs() << "\n";

    Vector<double, HostMemSpace> solution(dof_handler.n_dofs());

    solution[0] = 1.0;
    solution[1] = 0.5;

    DataOut<2> data_out;
    data_out.clear();
    data_out.attach_dof_handler(dof_handler);
    data_out.set_time(0.0);
    data_out.set_cycle(0);

    data_out.add_data_vector(solution, "solution");

    data_out.write_vtu("test.vtu");
  }
  Kokkos::finalize();

  return 0;
}
