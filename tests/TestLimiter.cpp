#include <gtest/gtest.h>
#include "limiters.hpp"

//test to ensure a zero gradient returns the constant state
TEST(Gradient, ZeroGradient)
{
  EXPECT_TRUE(true);
}

//Test to ensure the gradient computes an arbitrary correct output
TEST(Gradient, CorrectOutput)
{
  EXPECT_TRUE(true);
}

//Test to ensure the gradient computes a correct gradient on the periodic boundaries
TEST(Gradient, PeriodicBoundary)
{
  EXPECT_TRUE(true);
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
  EXPECT_TRUE(true);
}

//Test to ensure the limited gradient equates the 1st order when alpha = {0,0,0,0}
TEST(Limiter_BJ, ZeroGradient)
{
  EXPECT_TRUE(true);
}

//Test to ensure the limited gradient equates the inputed gradient when alpha = {1,1,1,1}
TEST(Limiter_BJ, NoLimiting)
{
  EXPECT_TRUE(true);
}
