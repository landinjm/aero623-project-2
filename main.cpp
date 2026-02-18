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
  unsigned int n_iterations = 1000;
  std::vector<double> residual_history(n_iterations, 0);
  Solver<2, 1, double> solver;

  Timer::instance().begin_section("freestream - roe flux");
  std::cout << "\nFreestream Test - Roe Flux" << std::endl;
  solver.set_free_stream_initial_state(tria.get_elements());
  for (unsigned int i = 0; i < n_iterations; ++i) {
    solver.compute_free_stream_residual(tria.get_interior_faces(),
                                        tria.get_boundary_faces(),
                                        tria.get_periodic_faces(),
                                        tria.get_elements(),
                                        &flux_roe);

    solver.compute_update_with_local_timestepping(tria.get_elements());

    residual_history[i] = tria.get_elements().max_residual();
  }
  std::cout << "  Min/Max Residual " << min(residual_history) << "/"
            << max(residual_history) << std::endl;
  zero(residual_history);
  Timer::instance().end_section("freestream - roe flux");

  Timer::instance().begin_section("freestream - HLLE flux");
  std::cout << "\nFreestream Test - HLLE Flux" << std::endl;
  solver.set_free_stream_initial_state(tria.get_elements());
  for (unsigned int i = 0; i < n_iterations; ++i) {
    solver.compute_free_stream_residual(tria.get_interior_faces(),
                                        tria.get_boundary_faces(),
                                        tria.get_periodic_faces(),
                                        tria.get_elements(),
                                        &Flux_HLLE);

    solver.compute_update_with_local_timestepping(tria.get_elements());

    residual_history[i] = tria.get_elements().max_residual();
  }
  std::cout << "  Min/Max Residual " << min(residual_history) << "/"
            << max(residual_history) << std::endl;
  zero(residual_history);
  Timer::instance().end_section("freestream - HLLE flux");

  // Steady state
  solver.set_free_stream_initial_state(tria.get_elements());
  for (unsigned int i = 0; i < 10000; ++i) {
    solver.compute_residual(tria.get_interior_faces(),
                            tria.get_boundary_faces(),
                            tria.get_periodic_faces(),
                            tria.get_elements(),
                            &Flux_HLLE);
    solver.compute_update_with_local_timestepping(tria.get_elements());
    std::cout << "Residual: " << tria.get_elements().max_residual()
              << std::endl;
  }
  Timer::instance().summary();

  return 0;
}
