#include <forcecoeffs.hpp>
#include <iostream>
#include <read_gri.hpp>
#include <record.hpp>
#include <solver.hpp>
#include <timer.hpp>

template<unsigned int dim, unsigned int degree, typename RealType>
void
freestream_test(
  const MeshData& mesh_data,
  Triangulation<dim, degree, RealType>& tria,
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
    solver.compute_update(mesh_data,
                          tria.get_elements(),
                          tria.get_interior_faces(),
                          tria.get_boundary_faces(),
                          tria.get_periodic_faces(),
                          cfg);

    residual_history[i] = tria.get_elements().max_residual();
  }
  std::cout << "  Min/Max Residual " << min(residual_history) << "/"
            << max(residual_history) << std::endl;

  Timer::instance().end_section(timer);
}

template<unsigned int dim, unsigned int degree, typename RealType>
void
steadystate_test(
  const MeshData& mesh_data,
  Triangulation<dim, degree, RealType>& tria,
  std::string flux_function,
  const typename Solver<dim, degree, RealType>::FluxFunction& flux_func,
  unsigned int n_iterations = 20000,
  RealType rel_tol = 1.0e-5,
  bool use_global_timestep = false,
  std::function<void(ElementData<dim, degree, RealType>&)> initial_guess =
    nullptr)
{
  std::vector<RealType> residual_history;
  residual_history.reserve(n_iterations);

  using LocalSolver = Solver<dim, degree, RealType>;
  LocalSolver solver;
  typename LocalSolver::SolverConfig cfg{
    .time_integration = use_global_timestep
                          ? LocalSolver::TimeIntegration::RK3
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
  if (initial_guess) {
    initial_guess(tria.get_elements());
  }
  RealType time = 0.0;
  for (unsigned int i = 0; i < n_iterations; ++i) {
    auto dt = solver.compute_update(mesh_data,
                                    tria.get_elements(),
                                    tria.get_interior_faces(),
                                    tria.get_boundary_faces(),
                                    tria.get_periodic_faces(),
                                    cfg);
    time += dt;
    auto residual = tria.get_elements().l1_residual();
    residual_history.push_back(residual);
    if (i % 500 == 0)
      std::cout << "Step " << i << " Residual " << residual << "\n";
    if (residual < residual_history.front() * rel_tol) {
      break;
    }
  }
  std::cout << "  Start/End Residual " << residual_history.front() << "/"
            << residual_history.back() << " in " << residual_history.size()
            << " Steps" << std::endl;

  recorder.recordHist(
    residual_history, "Steadystate", "2ndOrder", "HLLE", "Residual");

  Timer::instance().end_section(timer);
}

int
main(int argc, char** argv)
{
  // Read the grid
  Timer::instance().begin_section("read grid");
  GriReader<2> reader;
  reader.read_gri("../grids/coarse_local_refinement.gri");
  auto data = reader.get_data();
  Timer::instance().end_section("read grid");

  // Generate triangulation data structures
  Timer::instance().begin_section("create triangulation");
  Triangulation<2, 1, double> tria_1(reader.get_data());
  Triangulation<2, 2, double> tria_2(reader.get_data());
  Timer::instance().end_section("create triangulation");

  // Free stream test
  freestream_test<2, 1, double>(data, tria_1, "Roe Flux", &flux_roe);
  freestream_test<2, 1, double>(data, tria_1, "HLLE Flux", &flux_hlle);
  freestream_test<2, 2, double>(data, tria_2, "Roe Flux", &flux_roe);
  freestream_test<2, 2, double>(data, tria_2, "HLLE Flux", &flux_hlle);

  // Steadystate test
  steadystate_test<2, 1, double>(data, tria_1, "Roe Flux", &flux_roe);

  auto& elements_1st = tria_1.get_elements();
  steadystate_test<2, 2, double>(
    data,
    tria_2,
    "Roe Flux",
    &flux_roe,
    10000,
    1.0e-5,
    true,
    [&elements_1st](auto& elements) {
      ASSERT(elements.size() == elements_1st.size());
      for (unsigned int i = 0; i < elements.size(); ++i) {
        elements.density[i] = elements_1st.density[i];
        elements.momentum_x[i] = elements_1st.momentum_x[i];
        elements.momentum_y[i] = elements_1st.momentum_y[i];
        elements.energy[i] = elements_1st.energy[i];
      }
    });

  return 0;

  steadystate_test<2, 1, double>(data, tria_1, "HLLE Flux", &flux_hlle);

  elements_1st = tria_1.get_elements();
  steadystate_test<2, 2, double>(
    data,
    tria_2,
    "HLLE Flux",
    &flux_hlle,
    1000,
    1.0e-5,
    true,
    [&elements_1st](auto& elements) {
      ASSERT(elements.size() == elements_1st.size());
      for (unsigned int i = 0; i < elements.size(); ++i) {
        elements.density[i] = elements_1st.density[i];
        elements.momentum_x[i] = elements_1st.momentum_x[i];
        elements.momentum_y[i] = elements_1st.momentum_y[i];
        elements.energy[i] = elements_1st.energy[i];
      }
    }); // Cleanup
  Timer::instance().summary();

  return 0;
}
