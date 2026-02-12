#include <gtest/gtest.h>
#include <iostream>
#include <cmath>
#include "flux.hpp"

TEST(CellFlux, CorrectOutput) {
  Tensor<1,4,double> U = {1,0,0,0};
  Tensor<1,2,double> n = {1,0};
  Tensor<1,4,double> fluxCell = Cell_Flux(U, n);
  Tensor<1,4,double> fluxActual = {0, 0, 0, 0};
  EXPECT_TRUE(fluxCell == fluxActual);

  U = {1,3,4,17.5};
  n = {1,1};
  fluxCell = Cell_Flux(U, n);
  fluxActual = {7, 23, 30, 136.5};
  EXPECT_TRUE(fluxCell == fluxActual);
  
}

//F_hat(u,u,n) = F(u)
TEST(HLLETest, Consistency)
{
  Tensor<1,4,double> U = {1,3,4,17.5};
  Tensor<1,2,double> n = {1,0};

  std::pair<Tensor<1,4,double>, double> fluxHLLE = Flux_HLLE(U, U, n);

  Tensor<1,4,double> fluxCell = Cell_Flux(U, n);
  EXPECT_TRUE(fluxCell == fluxHLLE.first);

}

//F_hat(uL,uR,n) = -F_hat(uR,uL,n)
TEST(HLLETest, Conservation)
{
  Tensor<1,4,double> UL = {1,3,4,17.5};
  Tensor<1,4,double> UR = {1,1,2,10};
  Tensor<1,2,double> n = {1,1};

  std::pair<Tensor<1,4,double>, double> fluxHLLE_1 = Flux_HLLE(UL, UR, n);
  std::pair<Tensor<1,4,double>, double> fluxHLLE_2 = Flux_HLLE(UR, UL, -1 * n);
  EXPECT_TRUE(fluxHLLE_1.first == -1 * fluxHLLE_2.first);
  EXPECT_TRUE(fluxHLLE_1.second == fluxHLLE_2.second);
}

//smag > a
TEST(HLLETest, SupersonicUpwinding)
{
  Tensor<1,4,double> UL = {1.225,735.0,0.0,473812.5};
  Tensor<1,4,double> UR = {0.9,450.0,0.0,237500.0};
  Tensor<1,2,double> n = {1,0};

  std::pair<Tensor<1,4,double>, double> fluxHLLE_1 = Flux_HLLE(UL, UR, n);
  Tensor<1,4,double> upwindFlux = Cell_Flux(UL, n);
  EXPECT_TRUE(fluxHLLE_1.first == upwindFlux);
}

//
TEST(HLLETest, CorrectOutput)
{
  Tensor<1,4,double> UL = {1, 0, 0, 12.5};
  Tensor<1,4,double> UR = {2, 0, 0, 12.5};
  Tensor<1,4,double> F_expected = {-1.322875656, 3, 4, 0};
  Tensor<1,2,double> n = {.6,.8};

  std::pair<Tensor<1,4,double>, double> fluxHLLE_1 = Flux_HLLE(UL, UR, n);
  Tensor<1,4,double> upwindFlux = Cell_Flux(UL, n);
  EXPECT_TRUE(fluxHLLE_1.first.isApprox(F_expected, 1e-6));
}
