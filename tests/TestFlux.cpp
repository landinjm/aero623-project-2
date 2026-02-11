#include <gtest/gtest.h>
#include "flux.hpp"

//F_hat(u,u,n) = F(u)
TEST(HLLETest, Consistency)
{
  Tensor<1,4,double> U = //TODO
  Tensor<1,2,double> n = //TODO

  std::pair<Tensor<1,4,double>, double> fluxHLLE = Flux_HLLE(U, U, n);
  Tensor<1,4,double> fluxCell = Cell_Flux(U, n);
  EXPECT_TRUE(fluxCell == fluxHLLE[0]);

}

//F_hat(uL,uR,n) = -F_hat(uR,uL,n)
TEST(HLLETest, Conservation)
{
  //TODO
  EXPECT_TRUE(true);
}

//smag > a
TEST(HLLETest, SupersonicUpwinding)
{
  //TODO
  EXPECT_TRUE(true);
}

//
TEST(HLLETest, 1DEquality)
{
  //TODO
  EXPECT_TRUE(true);
}

//
TEST(HLLETest, CorrectOutput)
{
  //TODO
  EXPECT_TRUE(true);
}
