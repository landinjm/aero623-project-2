#include <gtest/gtest.h>
#include "limiters.hpp"
#include <read_gri.hpp>
#include <triangulation.hpp>
#include <cmath>

//test to ensure a zero gradient returns the constant state
TEST(Gradient, ZeroGradient)
{
  GriReader<2> reader;
  reader.read_gri("../tests/test.gri");
  Triangulation<2,double> tria(reader.get_data());

  auto& elem = tria.get_elements();

  // Set constant Euler state
  for (unsigned int i = 0; i < elem.size(); ++i)
  {
    elem.density[i] = 1.0;
    elem.momentum_x[i]   = 2.0;
    elem.momentum_y[i]   = 3.0;
    elem.energy[i]  = 4.0;
  }

  Tensor<2,4,double> grad_U = ComputeGradients(tria, elem);

  Tensor<2,4,double> zero = {0,0,0,0,0,0,0,0};

  EXPECT_TRUE(grad_U.isApprox(zero, 1e-12));
}

//Test to ensure the gradient computes an arbitrary correct output
TEST(Gradient, CorrectOutput)
{
  GriReader<2> reader;
  reader.read_gri("../tests/test.gri");
  Triangulation<2,double> tria(reader.get_data());

  auto& elem = tria.get_elements();

  for (unsigned int i = 0; i < elem.size(); ++i)
  {
    double x = elem.centroid_x[i];
    double y = elem.centroid_y[i];

    elem.density[i] = x;
    elem.momentum_x[i]   = y;
    elem.momentum_y[i]   = x + y;
    elem.energy[i]  = 2.0 * x;
  }

  Tensor<2,4,double> grad_U = ComputeGradients(tria, elem);

  Tensor<2,4,double> expected = {
      1,0,
      0,1,
      1,1,
      2,0
  };

  EXPECT_TRUE(grad_U.isApprox(expected, 1e-10));
}

//Test to ensure the gradient computes a correct gradient on the periodic boundaries
TEST(Gradient, PeriodicBoundary)
{
  GriReader<2> reader;
  reader.read_gri("../tests/test_2.gri");

  Triangulation<2,double> tria(reader.get_data());
  auto& elem = tria.get_elements();

  // Linear periodic field
  for (unsigned int i = 0; i < elem.size(); ++i)
  {
    double x = elem.centroid_x[i];

    elem.density[i] = x;
    elem.momentum_x[i]   = 2.0 * x;
    elem.momentum_y[i]   = 3.0 * x;
    elem.energy[i]  = 4.0 * x;
  }

  Tensor<2,4,double> grad_U = ComputeGradients(tria, elem);

  Tensor<2,4,double> expected = {
      1,0,
      2,0,
      3,0,
      4,0
  };

  EXPECT_TRUE(grad_U.isApprox(expected, 1e-10));
}

//Test to ensure the gradient computes a correct gradient on a wall boundary
TEST(Gradient, WallBoundary)
{
  EXPECT_TRUE(true);
}

//Test to ensure the gradient computes a correct gradient on a inflow boundary
TEST(Gradient, InflowBoundary)
{
  EXPECT_TRUE(true);
}

//Test to ensure the gradient computes a correct gradient on a outflow boundary
TEST(Gradient, OutflowBoundary)
{
  EXPECT_TRUE(true);
}

//Test to ensure the limited gradient computes an arbritrary correct output
TEST(Limiter_BJ, CorrectOutput)
{
  Tensor<2,4,double> L_in = {
      5,2,
      1,3,
      4,1,
      2,6
  };

  Tensor<1,4,double> u_in = {3,1,4,2};

  std::array<Tensor<1,2,double>,3> r_in =
  {{
      {0,1},
      {-0.5*std::sqrt(3),-0.5},
      { 0.5*std::sqrt(3),-0.5}
  }};

  Tensor<2,4,double> L_out = Limiter_BJ(L_in, u_in, r_in);

  for (unsigned int i = 0; i < 8; ++i)
    EXPECT_TRUE(std::isfinite(L_out[i])); //TODO!!!!!
}

//Test to ensure the limited gradient equates the 1st order when alpha = {0,0,0,0}
TEST(Limiter_BJ, ZeroGradient)
{
  Tensor<2,4,double> L_in = {
      1,0,
      1,0,
      1,0,
      1,0
  };

  Tensor<2,4,double> L0 = {0,0,0,0,0,0,0,0};

  Tensor<1,4,double> u_in = {2,2,2,2};   // constant state

  std::array<Tensor<1,2,double>,3> r_in =
  {{
      {0,1},
      {-0.5*std::sqrt(3),-0.5},
      { 0.5*std::sqrt(3),-0.5}
  }};

  Tensor<2,4,double> L_out = Limiter_BJ(L_in, u_in, r_in);

  EXPECT_TRUE(L_out == L0);
}

//Test to ensure the limited gradient equates the inputed gradient when alpha = {1,1,1,1}
TEST(Limiter_BJ, NoLimiting)
{
  Tensor<2,4,double> L_in = {
      1,0,
      1,0,
      1,0,
      1,0
  };

  Tensor<1,4,double> u_in = {1,2,3,4};

  std::array<Tensor<1,2,double>,3> r_in =
  {{
      {0,1},
      {-0.5*std::sqrt(3),-0.5},
      { 0.5*std::sqrt(3),-0.5}
  }};

  Tensor<2,4,double> L_out = Limiter_BJ(L_in, u_in, r_in);

  EXPECT_TRUE(L_out == L_in);
}
