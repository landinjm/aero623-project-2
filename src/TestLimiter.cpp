#include "limiters.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <read_gri.hpp>
#include <triangulation.hpp>

// test to ensure a zero gradient returns the constant state
TEST(Gradient, ZeroGradient) {
  GriReader<2> reader;
  reader.read_gri("../tests/test_2.gri");
  Triangulation<2, double> tria(reader.get_data());

  auto &elem = tria.get_elements();

  // Set constant state
  for (unsigned int i = 0; i < elem.size(); ++i) {
    elem.density[i] = 1.0;
    elem.momentum_x[i] = 2.0;
    elem.momentum_y[i] = 3.0;
    elem.energy[i] = 4.0;
  }

  // Compute gradient for ONE element only
  const unsigned int e = 0;

  std::array<Tensor<1, 2, double>, 4> L0;
  ComputeGradients(tria, elem, e, L0);

  // Check that all gradients are zero
  for (int eq = 0; eq < 4; ++eq) {
    EXPECT_NEAR(L0[eq][0], 0.0, 1e-12);
    EXPECT_NEAR(L0[eq][1], 0.0, 1e-12);
  }
}

// Test to ensure the gradient computes correct output (single element)
TEST(Gradient, CorrectOutput) {
  GriReader<2> reader;
  reader.read_gri("../tests/test_2.gri");

  Triangulation<2, double> tria(reader.get_data());
  auto &elem = tria.get_elements();

  // 👇 ADD THIS BLOCK HERE
  const auto &interior = tria.get_interior_faces();
  double nx = interior.normal_x[0];
  double ny = interior.normal_y[0];
  double mag = std::sqrt(nx * nx + ny * ny);
  std::cout << "normal mag = " << mag
            << " face_area = " << interior.face_area[0] << std::endl;
  // 👆 END DEBUG BLOCK

  // Define linear field
  for (unsigned int i = 0; i < elem.size(); ++i) {
    double x = elem.centroid_x[i];
    double y = elem.centroid_y[i];

    elem.density[i] = x;        // ∂/∂x = 1, ∂/∂y = 0
    elem.momentum_x[i] = y;     // ∂/∂x = 0, ∂/∂y = 1
    elem.momentum_y[i] = x + y; // ∂/∂x = 1, ∂/∂y = 1
    elem.energy[i] = 2.0 * x;   // ∂/∂x = 2, ∂/∂y = 0
  }

  std::array<Tensor<1, 2, double>, 4> expected = {
      {{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {2.0, 0.0}}};

  // Only test ONE element
  const unsigned int e = 0;

  std::array<Tensor<1, 2, double>, 4> L0;
  ComputeGradients(tria, elem, e, L0);

  for (int i = 0; i < 4; i++) {
    EXPECT_NEAR(L0[i][0], expected[i][0], 1e-10);
    EXPECT_NEAR(L0[i][1], expected[i][1], 1e-10);
  }
}

// Test gradient correctness with periodic boundaries (single element)
TEST(Gradient, PeriodicBoundary) {
  GriReader<2> reader;
  reader.read_gri("../tests/test_2.gri");

  Triangulation<2, double> tria(reader.get_data());
  auto &elem = tria.get_elements();

  // Linear periodic field
  const double a = 1.0;
  const double b = 0.0;

  for (unsigned int i = 0; i < elem.size(); ++i) {
    double x = elem.centroid_x[i];
    double y = elem.centroid_y[i];

    double value = a * x + b * y;

    elem.density[i] = value;
    elem.momentum_x[i] = 2.0 * value;
    elem.momentum_y[i] = 3.0 * value;
    elem.energy[i] = 4.0 * value;
  }

  std::array<Tensor<1, 2, double>, 4> expected = {
      {{a, b}, {2.0 * a, 2.0 * b}, {3.0 * a, 3.0 * b}, {4.0 * a, 4.0 * b}}};

  // Test ONE representative element
  const unsigned int e = 0;

  std::array<Tensor<1, 2, double>, 4> L0;
  ComputeGradients(tria, elem, e, L0);

  for (int i = 0; i < 4; i++) {
    EXPECT_NEAR(L0[i][0], expected[i][0], 1e-10);
    EXPECT_NEAR(L0[i][1], expected[i][1], 1e-10);
  }
}

// Test to ensure the limited gradient computes an arbritrary correct output
TEST(Limiter_BJ, CorrectOutput) {
  std::array<Tensor<1, 2, double>, 4> L_in = {
      {{5.0, 2.0}, {1.0, 3.0}, {4.0, 1.0}, {2.0, 6.0}}};

  Tensor<1, 4, double> u_in = {3, 1, 4, 2};

  std::array<Tensor<1, 2, double>, 3> r_in = {
      {{0, 1}, {-0.5 * std::sqrt(3), -0.5}, {0.5 * std::sqrt(3), -0.5}}};

  // Reasonable bounds around u_in
  Tensor<1, 4, double> umin = {2, 0, 3, 1};
  Tensor<1, 4, double> umax = {5, 4, 6, 5};

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
TEST(Limiter_BJ, ZeroGradient) {
  std::array<Tensor<1, 2, double>, 4> L0 = {
      {{1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}}};

  std::array<Tensor<1, 2, double>, 4> L0_expected = {
      {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}};

  Tensor<1, 4, double> u_in = {2, 2, 2, 2};

  std::array<Tensor<1, 2, double>, 3> r_in = {
      {{0, 1}, {-0.5 * std::sqrt(3), -0.5}, {0.5 * std::sqrt(3), -0.5}}};

  // Force alpha = 0
  Tensor<1, 4, double> umin = u_in;
  Tensor<1, 4, double> umax = u_in;

  Limiter_BJ(L0, u_in, umin, umax, r_in);

  for (int eq = 0; eq < 4; ++eq) {
    EXPECT_DOUBLE_EQ(L0[eq][0], L0_expected[eq][0]);
    EXPECT_DOUBLE_EQ(L0[eq][1], L0_expected[eq][1]);
  }
}

// Test to ensure the limited gradient equates the inputed gradient when alpha =
// {1,1,1,1}
TEST(Limiter_BJ, NoLimiting) {
  std::array<Tensor<1, 2, double>, 4> L_in = {
      {{500.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}}};

  Tensor<1, 4, double> u_in = {1, 2, 3, 4};

  std::array<Tensor<1, 2, double>, 3> r_in = {
      {{0, 1}, {-0.5 * std::sqrt(3), -0.5}, {0.5 * std::sqrt(3), -0.5}}};

  std::array<Tensor<1, 2, double>, 4> L0 = L_in;

  // Very wide bounds → alpha = 1
  Tensor<1, 4, double> umin = {-1e12, -1e12, -1e12, -1e12};
  Tensor<1, 4, double> umax = {1e12, 1e12, 1e12, 1e12};

  Limiter_BJ(L0, u_in, umin, umax, r_in);

  for (int eq = 0; eq < 4; ++eq) {
    EXPECT_DOUBLE_EQ(L0[eq][0], L_in[eq][0]);
    EXPECT_DOUBLE_EQ(L0[eq][1], L_in[eq][1]);
  }
}
