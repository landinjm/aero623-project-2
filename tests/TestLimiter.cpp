#include <cmath>
#include <gtest/gtest.h>
#include <read_gri.hpp>
#include <solver.hpp>
#include <triangulation.hpp>

// test to ensure a zero gradient returns the constant state
TEST(Gradient, ZeroGradient)
{
  GriReader<2> reader;
  reader.read_gri("../tests/test_2.gri");
  auto data = reader.get_data();
  Triangulation<2, 2, double> tria(data);
  auto& elem = tria.get_elements();

  // Set constant state
  set(elem.density, 1.0);
  set(elem.momentum_x, 2.0);
  set(elem.momentum_y, 3.0);
  set(elem.energy, 4.0);
  const Tensor<1, 4, double> constant_state = { 1.0, 2.0, 3.0, 4.0 };

  // Compute the gradients
  Solver<2, 2, double> solver;
  solver.zero_values(elem);
  solver.interior_face_gradient_prep(tria.get_interior_faces(), elem);
  solver.periodic_face_gradient_prep(tria.get_periodic_faces(), elem);
  solver.boundary_face_gradient_prep(
    tria.get_boundary_faces(), elem, true, constant_state);
  solver.finalize_gradient(elem);

  // Check that all gradients are zero
  for (unsigned int i = 0; i < elem.size(); ++i) {
    EXPECT_NEAR(elem.grad_x_density[i], 0.0, 1e-12);
    EXPECT_NEAR(elem.grad_y_density[i], 0.0, 1e-12);

    EXPECT_NEAR(elem.grad_x_momentum_x[i], 0.0, 1e-12);
    EXPECT_NEAR(elem.grad_y_momentum_x[i], 0.0, 1e-12);

    EXPECT_NEAR(elem.grad_x_momentum_y[i], 0.0, 1e-12);
    EXPECT_NEAR(elem.grad_y_momentum_y[i], 0.0, 1e-12);

    EXPECT_NEAR(elem.grad_x_energy[i], 0.0, 1e-12);
    EXPECT_NEAR(elem.grad_y_energy[i], 0.0, 1e-12);
  }
}

TEST(Gradient, LinearGradient)
{
  GriReader<2> reader;
  reader.read_gri("../tests/test_2.gri");
  auto data = reader.get_data();
  Triangulation<2, 2, double> tria(data);
  auto& elem = tria.get_elements();

  // Define linear field
  for (unsigned int i = 0; i < elem.size(); ++i) {
    elem.density[i] = 1.0 + elem.centroid_y[i];
    elem.momentum_x[i] = 1.0;
    elem.momentum_y[i] = 1.0;
    elem.energy[i] = 1.0;
  }

  // Compute the gradients
  Solver<2, 2, double> solver;
  solver.zero_values(elem);
  solver.interior_face_gradient_prep(tria.get_interior_faces(), elem);
  solver.periodic_face_gradient_prep(tria.get_periodic_faces(), elem);

  for (unsigned int i = 0; i < tria.get_boundary_faces().size(); ++i) {
    const auto e = tria.get_boundary_faces().elem[i];
    const auto n_x = tria.get_boundary_faces().normal_x[i];
    const auto n_y = tria.get_boundary_faces().normal_y[i];
    const auto area = tria.get_boundary_faces().face_area[i];
    const auto boundary_id = tria.get_boundary_faces().boundary_id[i];

    const auto f_y = tria.get_boundary_faces().centroid_y[i];

    const Tensor<1, 4, double> freestream_state = { 1.0 + f_y, 1.0, 1.0, 1.0 };

    const Tensor<1, 2, double> n = { n_x, n_y };
    const auto interior = solver.get_state(elem, e);
    const auto boundary = solver.get_boundary_state(
      interior, n, boundary_id, true, freestream_state);
    const auto avg = boundary;

    solver.accumulate_face_gradient(
      elem, e, avg[0], avg[1], avg[2], avg[3], area, n_x, n_y, 1.0);
  }
  solver.finalize_gradient(elem);

  for (unsigned int i = 0; i < elem.size(); ++i) {
    std::cout << i << " " << elem.centroid_x[i] << " " << elem.centroid_y[i]
              << std::endl;
    EXPECT_NEAR(elem.grad_x_density[i], 0.0, 1e-12);
    EXPECT_NEAR(elem.grad_y_density[i], 1.0, 1e-12);

    EXPECT_NEAR(elem.grad_x_momentum_x[i], 0.0, 1e-12);
    EXPECT_NEAR(elem.grad_y_momentum_x[i], 0.0, 1e-12);

    EXPECT_NEAR(elem.grad_x_momentum_y[i], 0.0, 1e-12);
    EXPECT_NEAR(elem.grad_y_momentum_y[i], 0.0, 1e-12);

    EXPECT_NEAR(elem.grad_x_energy[i], 0.0, 1e-12);
    EXPECT_NEAR(elem.grad_y_energy[i], 0.0, 1e-12);
  }
}
/*
TEST(Gradient, PeriodicLinearGradient)
{
  GriReader<2> reader;
  reader.read_gri("../tests/test_2.gri");
  auto data = reader.get_data();
  Triangulation<2, 2, double> tria(data);
  auto& elem = tria.get_elements();

  // Define linear field
  double x = 2.0;
  for (unsigned int i = 0; i < elem.size(); ++i) {
    elem.density[i] = 1.0 + (elem.centroid_x[i] - x / 2.0);
    elem.momentum_x[i] = 1.0;
    elem.momentum_y[i] = 1.0;
    elem.energy[i] = 1.0;
  }

  // Compute the gradients
  const Tensor<1, 4, double> freestream = { 1.0, 1.0, 1.0, 1.0 };

  Solver<2, 2, double> solver;
  solver.zero_values(elem);
  solver.interior_face_gradient_prep(tria.get_interior_faces(), elem);
  solver.periodic_face_gradient_prep(tria.get_periodic_faces(), elem);
  solver.boundary_face_gradient_prep(
    tria.get_boundary_faces(), elem, false, freestream);
  solver.finalize_gradient(elem);

  // Build set of boundary-adjacent elements to skip
  std::unordered_set<unsigned int> boundary_elems;
  for (unsigned int i = 0; i < tria.get_boundary_faces().size(); ++i) {
    boundary_elems.insert(tria.get_boundary_faces().elem[i]);
  }

  // Only check interior cells
  for (unsigned int i = 0; i < elem.size(); ++i) {
    if (boundary_elems.count(i))
      continue;
    EXPECT_NEAR(elem.grad_x_density[i], 1.0, 1e-12);
    EXPECT_NEAR(elem.grad_y_density[i], 0.0, 1e-12);

    EXPECT_NEAR(elem.grad_x_momentum_x[i], 0.0, 1e-12);
    EXPECT_NEAR(elem.grad_y_momentum_x[i], 0.0, 1e-12);

    EXPECT_NEAR(elem.grad_x_momentum_y[i], 0.0, 1e-12);
    EXPECT_NEAR(elem.grad_y_momentum_y[i], 0.0, 1e-12);

    EXPECT_NEAR(elem.grad_x_energy[i], 0.0, 1e-12);
    EXPECT_NEAR(elem.grad_y_energy[i], 0.0, 1e-12);
  }
}

// Test to ensure the limited gradient computes an arbitrary correct output
TEST(Limiter_BJ, CorrectOutput)
{
  std::array<Tensor<1, 2, double>, 4> L_in = {
    { { 5.0, 2.0 }, { 1.0, 3.0 }, { 4.0, 1.0 }, { 2.0, 6.0 } }
  };

  Tensor<1, 4, double> u_in = { 3, 1, 4, 2 };

  std::array<Tensor<1, 2, double>, 3> r_in = {
    { { 0, 1 }, { -0.5 * std::sqrt(3), -0.5 }, { 0.5 * std::sqrt(3), -0.5 } }
  };

  // Reasonable bounds around u_in
  Tensor<1, 4, double> umin = { 2, 0, 3, 1 };
  Tensor<1, 4, double> umax = { 5, 4, 6, 5 };

  std::array<Tensor<1, 2, double>, 4> L0 = L_in;

  Limiter_BJ(L0, u_in, umin, umax, r_in);

  // Ensure values are finite
  for (int eq = 0; eq < 4; ++eq) {
    EXPECT_TRUE(std::isfinite(L0[eq][0]));
    EXPECT_TRUE(std::isfinite(L0[eq][1]));
  }

  // Ensure limited slopes are not larger than original
  for (int eq = 0; eq < 4; ++eq) {
    EXPECT_LE(std::abs(L0[eq][0]), std::abs(L_in[eq][0]));
    EXPECT_LE(std::abs(L0[eq][1]), std::abs(L_in[eq][1]));
  }
}

// Test to ensure the limited gradient equates the 1st order when alpha =
// {0,0,0,0}
TEST(Limiter_BJ, ZeroGradient)
{
  std::array<Tensor<1, 2, double>, 4> L0 = {
    { { 1.0, 0.0 }, { 1.0, 0.0 }, { 1.0, 0.0 }, { 1.0, 0.0 } }
  };

  std::array<Tensor<1, 2, double>, 4> L0_expected = {
    { { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 } }
  };

  Tensor<1, 4, double> u_in = { 2, 2, 2, 2 };

  std::array<Tensor<1, 2, double>, 3> r_in = {
    { { 0, 1 }, { -0.5 * std::sqrt(3), -0.5 }, { 0.5 * std::sqrt(3), -0.5 } }
  };

  // Force alpha = 0
  Tensor<1, 4, double> umin = u_in;
  Tensor<1, 4, double> umax = u_in;

  Limiter_BJ(L0, u_in, umin, umax, r_in);

  for (int eq = 0; eq < 4; ++eq) {
    EXPECT_DOUBLE_EQ(L0[eq][0], L0_expected[eq][0]);
    EXPECT_DOUBLE_EQ(L0[eq][1], L0_expected[eq][1]);
  }
}

// Test to ensure the limited gradient equates the inputted gradient when alpha
// = {1,1,1,1}
TEST(Limiter_BJ, NoLimiting)
{
  std::array<Tensor<1, 2, double>, 4> L_in = {
    { { 500.0, 0.0 }, { 1.0, 0.0 }, { 1.0, 0.0 }, { 1.0, 0.0 } }
  };

  Tensor<1, 4, double> u_in = { 1, 2, 3, 4 };

  std::array<Tensor<1, 2, double>, 3> r_in = {
    { { 0, 1 }, { -0.5 * std::sqrt(3), -0.5 }, { 0.5 * std::sqrt(3), -0.5 } }
  };

  std::array<Tensor<1, 2, double>, 4> L0 = L_in;

  // Very wide bounds → alpha = 1
  Tensor<1, 4, double> umin = { -1e12, -1e12, -1e12, -1e12 };
  Tensor<1, 4, double> umax = { 1e12, 1e12, 1e12, 1e12 };

  Limiter_BJ(L0, u_in, umin, umax, r_in);

  for (int eq = 0; eq < 4; ++eq) {
    EXPECT_DOUBLE_EQ(L0[eq][0], L_in[eq][0]);
    EXPECT_DOUBLE_EQ(L0[eq][1], L_in[eq][1]);
  }
} */
