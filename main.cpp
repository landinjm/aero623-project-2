#include <forcecoeffs.hpp>
#include <iostream>
#include <read_gri.hpp>
#include <record.hpp>
#include <solver.hpp>
#include <timer.hpp>

template<unsigned int dim, unsigned int degree, typename RealType>
void
freestream_test(
  Triangulation<dim, degree, RealType> tria,
  std::string flux_function,
  const typename Solver<dim, degree, RealType>::FluxFunction& flux_func,
  unsigned int n_iterations = 100)
{
  std::vector<RealType> residual_history(n_iterations, 0);
  Solver<dim, degree, RealType> solver;
  Recorder<dim, degree, RealType> recorder;

  std::string timer = "Freestream - ";
  if constexpr (degree == 1) {
    timer += "1st Order - ";
  } else if constexpr (degree == 2) {
    timer += "2nd Order - ";
  }
  timer += flux_function;

  Timer::instance().begin_section(timer);
  std::cout << "\n" << timer << std::endl;
  solver.set_initial_state(tria.get_elements());
  for (unsigned int i = 0; i < n_iterations; ++i) {
    solver.compute_update_with_local_timestepping(tria.get_elements(),
                                                  tria.get_interior_faces(),
                                                  tria.get_boundary_faces(),
                                                  tria.get_periodic_faces(),
                                                  flux_func,
                                                  true);

    residual_history[i] = tria.get_elements().max_residual();
  }
  std::cout << "  Min/Max Residual " << min(residual_history) << "/"
            << max(residual_history) << std::endl;
  Timer::instance().end_section(timer);
}

int
main(int argc, char** argv)
{
  // Read the grid
  Timer::instance().begin_section("read grid");
  GriReader<2> reader;
  reader.read_gri("../grids/coarse_local_refinement_1.gri");
  auto data = reader.get_data();
  Timer::instance().end_section("read grid");

  // Generate triangulation data structures
  Timer::instance().begin_section("create triangulation");
  Triangulation<2, 1, double> tria_1(reader.get_data());
  Triangulation<2, 2, double> tria_2(reader.get_data());
  Timer::instance().end_section("create triangulation");

  // Free stream test
  freestream_test<2, 1, double>(tria_1, "Roe Flux", &flux_roe);
  freestream_test<2, 1, double>(tria_1, "HLLE Flux", &flux_hlle);
  freestream_test<2, 2, double>(tria_2, "Roe Flux", &flux_roe);
  freestream_test<2, 2, double>(tria_2, "HLLE Flux", &flux_hlle);

  // Cleanup
  Timer::instance().summary();

  return 0;
}
