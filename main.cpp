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

  using LocalSolver = Solver<dim, degree, RealType>;
  LocalSolver solver;
  typename LocalSolver::SolverConfig cfg{
    .time_integration = LocalSolver::TimeIntegration::LocalTimestepping,
    .is_freestream = true,
    .is_unsteady = false,
    .time = 0,
    .flux_func = &flux_func
  };
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
    solver.compute_update(tria.get_elements(),
                          tria.get_interior_faces(),
                          tria.get_boundary_faces(),
                          tria.get_periodic_faces(),
                          cfg);

    residual_history[i] = tria.get_elements().max_residual();
  }
  std::cout << "  Min/Max Residual " << min(residual_history) << "/"
            << max(residual_history) << std::endl;

  recorder.recordData(tria.get_elements(), "Freestream", "2ndOrder", "RoeFlux");

  Timer::instance().end_section(timer);
}

template<unsigned int dim, unsigned int degree, typename RealType>
void
steadystate_test(
  Triangulation<dim, degree, RealType> tria,
  std::string flux_function,
  const typename Solver<dim, degree, RealType>::FluxFunction& flux_func,
  unsigned int n_iterations = 20000,
  RealType rel_tol = 1.0e-5,
  bool use_ssp_rk2 = false)
{
  std::vector<RealType> residual_history;
  residual_history.reserve(n_iterations);

  using LocalSolver = Solver<dim, degree, RealType>;
  LocalSolver solver;
  typename LocalSolver::SolverConfig cfg{
    .time_integration = use_ssp_rk2
                          ? LocalSolver::TimeIntegration::SSPRK2
                          : LocalSolver::TimeIntegration::LocalTimestepping,
    .is_freestream = false,
    .is_unsteady = false,
    .time = 0,
    .flux_func = &flux_func
  };
  Recorder<dim, degree, RealType> recorder;

  std::string timer = "Steadystate - ";
  if constexpr (degree == 1) {
    timer += "1st Order - ";
  } else if constexpr (degree == 2) {
    timer += "2nd Order - ";
  }
  timer += flux_function;

  Timer::instance().begin_section(timer);
  std::cout << "\n" << timer << std::endl;
  solver.set_initial_state(tria.get_elements());
  RealType time = 0.0;
  for (unsigned int i = 0; i < n_iterations; ++i) {
    auto dt = solver.compute_update(tria.get_elements(),
                                    tria.get_interior_faces(),
                                    tria.get_boundary_faces(),
                                    tria.get_periodic_faces(),
                                    cfg);
    time += dt;
    auto residual = tria.get_elements().l1_residual();
    residual_history.push_back(residual);

    if (residual < residual_history.front() * rel_tol) {
      break;
    }
  }
  std::cout << "  Start/End Residual " << residual_history.front() << "/"
            << residual_history.back() << " in " << residual_history.size()
            << " Steps" << std::endl;
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

  // Steadystate test
  steadystate_test<2, 1, double>(tria_1, "Roe Flux", &flux_roe);
  steadystate_test<2, 1, double>(tria_1, "HLLE Flux", &flux_hlle);
  steadystate_test<2, 2, double>(tria_2, "Roe Flux", &flux_roe, 1000);
  steadystate_test<2, 2, double>(tria_2, "HLLE Flux", &flux_hlle, 1000);

  // Cleanup
  Timer::instance().summary();

  return 0;
}
