#include <Eigen/Dense>
#include <iostream>
#include <read_gri.hpp>
#include <record.hpp>
#include <solver.hpp>
#include <timer.hpp>
#include <forcecoeffs.hpp>

#include "flux.hpp"

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
  Triangulation<2, double> tria(reader.get_data());
  Timer::instance().end_section("create triangulation");

  // Recorder and solver instances
  Solver<2, 1, double> solver;
  Recorder<2, double> recorder;
  CalcForceCoeffs<2, double> CalcForceCoeffs;

  // Free stream test
  unsigned int n_iterations = 100;
  std::vector<double> freestream_residual_history(n_iterations, 0);
  {
    Timer::instance().begin_section("freestream - roe flux");
    std::cout << "\nFreestream Test - Roe Flux" << std::endl;
    solver.set_initial_state(tria.get_elements());
    for (unsigned int i = 0; i < n_iterations; ++i) {
      solver.compute_update_with_local_timestepping(tria.get_elements(),
                                                    tria.get_interior_faces(),
                                                    tria.get_boundary_faces(),
                                                    tria.get_periodic_faces(),
                                                    &flux_roe,
                                                    true);

      freestream_residual_history[i] = tria.get_elements().max_residual();
    }
    std::cout << "  Min/Max Residual " << min(freestream_residual_history)
              << "/" << max(freestream_residual_history) << std::endl;
    recorder.recordHist(
      freestream_residual_history, "freestream", "1st", "roe", "residual");
    zero(freestream_residual_history);
    Timer::instance().end_section("freestream - roe flux");
  }
  {
    Timer::instance().begin_section("freestream - HLLE flux");
    std::cout << "\nFreestream Test - HLLE Flux" << std::endl;
    solver.set_initial_state(tria.get_elements());
    for (unsigned int i = 0; i < n_iterations; ++i) {
      solver.compute_update_with_local_timestepping(tria.get_elements(),
                                                    tria.get_interior_faces(),
                                                    tria.get_boundary_faces(),
                                                    tria.get_periodic_faces(),
                                                    &Flux_HLLE,
                                                    true);

      freestream_residual_history[i] = tria.get_elements().max_residual();
    }
    std::cout << "  Min/Max Residual " << min(freestream_residual_history)
              << "/" << max(freestream_residual_history) << std::endl;
    recorder.recordHist(
      freestream_residual_history, "freestream", "1st", "HLLE", "residual");
    zero(freestream_residual_history);
    Timer::instance().end_section("freestream - HLLE flux");
  }

  // Steady state
  std::vector<double> steadystate_residual_history;
  double rel_tolerance = 1.0e-5;
  n_iterations = 100000;
  {
    Timer::instance().begin_section("steadystate - roe flux");
    std::cout << "\nSteadystate - Roe Flux" << std::endl;
    steadystate_residual_history.reserve(n_iterations);
    solver.set_initial_state(tria.get_elements());
    for (unsigned int i = 0; i < n_iterations; ++i) {
      solver.compute_update_with_local_timestepping(tria.get_elements(),
                                                    tria.get_interior_faces(),
                                                    tria.get_boundary_faces(),
                                                    tria.get_periodic_faces(),
                                                    &flux_roe);
      auto residual = tria.get_elements().l1_residual();
      steadystate_residual_history.push_back(residual);

      if (residual < steadystate_residual_history[0] * rel_tolerance) {
        break;
      }
    }
    std::cout << "  Start/End Residual " << steadystate_residual_history.front()
              << "/" << steadystate_residual_history.back() << " in "
              << steadystate_residual_history.size() << " Steps" << std::endl;
    recorder.recordHist(
      steadystate_residual_history, "steadystate", "1st", "roe", "residual");
    recorder.recordData(tria.get_elements(), "steadystate", "1st", "roe");
    steadystate_residual_history.clear();
    Timer::instance().end_section("steadystate - roe flux");
  }
  {
    Timer::instance().begin_section("steadystate - HLLE flux");
    std::cout << "\nSteadystate - HLLE Flux" << std::endl;
    steadystate_residual_history.reserve(n_iterations);
    solver.set_initial_state(tria.get_elements());
    for (unsigned int i = 0; i < n_iterations; ++i) {
      solver.compute_update_with_local_timestepping(tria.get_elements(),
                                                    tria.get_interior_faces(),
                                                    tria.get_boundary_faces(),
                                                    tria.get_periodic_faces(),
                                                    &Flux_HLLE);
      auto residual = tria.get_elements().l1_residual();
      steadystate_residual_history.push_back(residual);

      if (residual < steadystate_residual_history[0] * rel_tolerance) {
        break;
      }
    }
    std::cout << "  Start/End Residual " << steadystate_residual_history.front()
              << "/" << steadystate_residual_history.back() << " in "
              << steadystate_residual_history.size() << " Steps" << std::endl;
    recorder.recordHist(
      steadystate_residual_history, "steadystate", "1st", "HLLE", "residual");
    recorder.recordData(tria.get_elements(), "steadystate", "1st", "HLLE");
    steadystate_residual_history.clear();
    Timer::instance().end_section("steadystate - HLLE flux");
  }

  // Unsteady
  {
    Timer::instance().begin_section("unsteady - HLLE flux");
    std::cout << "\nUnsteady - HLLE Flux" << std::endl;

    // First, run to steady-state
    steadystate_residual_history.reserve(n_iterations);
    solver.set_initial_state(tria.get_elements());
    for (unsigned int i = 0; i < n_iterations; ++i) {
      solver.compute_update_with_local_timestepping(tria.get_elements(),
                                                    tria.get_interior_faces(),
                                                    tria.get_boundary_faces(),
                                                    tria.get_periodic_faces(),
                                                    &Flux_HLLE);
      auto residual = tria.get_elements().l1_residual();
      steadystate_residual_history.push_back(residual);

      if (residual < steadystate_residual_history[0] * rel_tolerance) {
        break;
      }
    }
    std::cout << "  Start/End Residual " << steadystate_residual_history.front()
              << "/" << steadystate_residual_history.back() << " in "
              << steadystate_residual_history.size() << " Steps" << std::endl;
    steadystate_residual_history.clear();

    // Now run for a bunch of timesteps
    double time = 0;
    for (unsigned int i = 0; i < 1000; ++i) {
      auto dt = solver.compute_update_with_ssp_rk2(tria.get_elements(),
                                                   tria.get_interior_faces(),
                                                   tria.get_boundary_faces(),
                                                   tria.get_periodic_faces(),
                                                   &Flux_HLLE,
                                                   time);
      time += dt;

      if (i % 100 == 0) {
        auto residual = tria.get_elements().l1_residual();
        std::cout << "  Time " << time << " Residual " << residual << "\n";
        recorder.recordDataHist(
          tria.get_elements(), "unsteady", "1st", "HLLE", std::to_string(time));
      }
    }

    Timer::instance().end_section("Unsteady - HLLE flux");
  }

  // Cleanup
  Timer::instance().summary();

  return 0;
}
