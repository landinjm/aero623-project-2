#include <Eigen/Dense>
#include <iostream>
#include <read_gri.hpp>
#include <solver.hpp>
#include <timer.hpp>

int
main(int argc, char** argv)
{
  // Read the grid
  Timer::instance().begin_section("read grid");
  GriReader<2> reader;
  reader.read_gri("../grids/coarse_local_refinement_1.gri");
  auto data = reader.get_data();
  Timer::instance().end_section("read grid");

  // Generate triangulation data structurs
  Timer::instance().begin_section("create triangulation");
  Triangulation<2, double> tria(reader.get_data());
  Timer::instance().end_section("create triangulation");

  // Free stream test
  Timer::instance().begin_section("freestream");
  Solver<2, 1, double> solver;
  solver.set_free_stream_initial_state(tria.get_elements());
  for (unsigned int i = 0; i < 100; ++i) {
    solver.compute_free_stream_residual(tria.get_interior_faces(),
                                        tria.get_boundary_faces(),
                                        tria.get_periodic_faces(),
                                        tria.get_elements());
    solver.compute_update_with_local_timestepping(tria.get_elements());
    std::cout << l2_norm(tria.get_elements().residual_density) << std::endl;
    std::cout << l2_norm(tria.get_elements().residual_momentum_x) << std::endl;
    std::cout << l2_norm(tria.get_elements().residual_momentum_y) << std::endl;
    std::cout << l2_norm(tria.get_elements().residual_energy) << std::endl;
  }

  Timer::instance().end_section("freestream");

  Timer::instance().summary();

  return 0;
}
