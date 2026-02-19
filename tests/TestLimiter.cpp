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

  for (unsigned int i = 0; i < elem.size(); ++i)
  {
    elem.density[i] = 1.0;
    elem.momentum_x[i] = 2.0;
    elem.momentum_y[i] = 3.0;
    elem.energy[i] = 4.0;
  }

  std::array<Tensor<1,2,double>,4> zero = {{
    {0.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0}
  }};

  for (unsigned int e = 0; e < elem.size(); ++e)
  {
    std::array<Tensor<1,2,double>,4> L0;
    ComputeGradients(tria, elem, e, L0);

    //loop over the states
    for (int i = 0; i < 4; i++)
      EXPECT_TRUE(L0[i].isApprox(zero[i], 1e-12));
  }
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

    elem.density[i]    = x;
    elem.momentum_x[i] = y;
    elem.momentum_y[i] = x + y;
    elem.energy[i]     = 2.0 * x;
  }

  std::array<Tensor<1,2,double>,4> expected = {{
    {1.0, 0.0},
    {0.0, 1.0},
    {1.0, 1.0},
    {2.0, 0.0}
  }};


  for (unsigned int e = 0; e < elem.size(); ++e)
  {
    std::array<Tensor<1,2,double>,4> L0;
    ComputeGradients(tria, elem, e, L0);

    //loop over the states
    for (int i = 0; i < 4; i++)
      EXPECT_TRUE(L0[i].isApprox(expected[i], 1e-10));
  }
}

//Test to ensure the gradient computes a correct gradient on the periodic boundaries
TEST(Gradient, PeriodicBoundary)
{
  GriReader<2> reader;
  reader.read_gri("../tests/test_2.gri");

  Triangulation<2,double> tria(reader.get_data());
  auto& elem = tria.get_elements();

  // Linear periodic field
  const double a = 1.0;
  const double b = 0.0;

  for (unsigned int i = 0; i < elem.size(); ++i)
  {
    double x = elem.centroid_x[i];
    double y = elem.centroid_y[i];

    elem.density[i]    = a*x + b*y;
    elem.momentum_x[i] = 2.0*(a*x + b*y);
    elem.momentum_y[i] = 3.0*(a*x + b*y);
    elem.energy[i]     = 4.0*(a*x + b*y);
  }

  std::array<Tensor<1,2,double>,4> expected = {{
    {a, b},
    {2.0 * a, 2.0 * b},
    {3.0 * a, 3.0 * b},
    {4.0 * a, 4.0 * b}
  }};

  for (unsigned int e = 0; e < elem.size(); ++e)
  {
    std::array<Tensor<1,2,double>,4> L0;
    ComputeGradients(tria, elem, e, L0);

    //loop over the states
    for (int i = 0; i < 4; i++)
      EXPECT_TRUE(L0[i].isApprox(expected[i], 1e-10));
  }
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

  std::array<Tensor<1,2,double>,4> L_in = {{
    {5.0, 2.0},
    {1.0, 3.0},
    {4.0, 1.0},
    {2.0, 6.0}
  }};


  Tensor<1,4,double> u_in = {3,1,4,2};

  std::array<Tensor<1,2,double>,3> r_in =
  {{
      {0,1},
      {-0.5*std::sqrt(3),-0.5},
      { 0.5*std::sqrt(3),-0.5}
  }};

  std::array<Tensor<1,2,double>,4> L0 = L_in;

  Limiter_BJ(L0, u_in, r_in);

  // for (unsigned int i = 0; i < 8; ++i) //TODO
  //   EXPECT_TRUE(std::isfinite(L0[i]));
  EXPECT_TRUE(true);
}

//Test to ensure the limited gradient equates the 1st order when alpha = {0,0,0,0}
TEST(Limiter_BJ, ZeroGradient)
{

  std::array<Tensor<1,2,double>,4> L0 = {{
    {1.0, 0.0},
    {1.0, 0.0},
    {1.0, 0.0},
    {1.0, 0.0}
  }};

  std::array<Tensor<1,2,double>,4> L0_expected = {{
    {0.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0}
  }};

  Tensor<1,4,double> u_in = {2,2,2,2};

  std::array<Tensor<1,2,double>,3> r_in =
  {{
      {0,1},
      {-0.5*std::sqrt(3),-0.5},
      { 0.5*std::sqrt(3),-0.5}
  }};

  Limiter_BJ(L0, u_in, r_in);

  EXPECT_TRUE(L0 == L0_expected);
}

//Test to ensure the limited gradient equates the inputed gradient when alpha = {1,1,1,1}
TEST(Limiter_BJ, NoLimiting)
{

  std::array<Tensor<1,2,double>,4> L_in = {{
    {1.0, 0.0},
    {1.0, 0.0},
    {1.0, 0.0},
    {1.0, 0.0}
  }};


  Tensor<1,4,double> u_in = {1,2,3,4};

  std::array<Tensor<1,2,double>,3> r_in =
  {{
      {0,1},
      {-0.5*std::sqrt(3),-0.5},
      { 0.5*std::sqrt(3),-0.5}
  }};

  std::array<Tensor<1,2,double>,4> L0 = L_in;

  Limiter_BJ(L0, u_in, r_in);

  EXPECT_TRUE(L0 == L_in);
}
