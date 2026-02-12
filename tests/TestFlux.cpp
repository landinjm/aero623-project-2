#include <gtest/gtest.h>
#include <iostream>
#include "flux.hpp"
#include <cmath>
#include "tensor.hpp"

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

//
TEST(RoeTest, Consistency) {
    Tensor<1,4,double> U = {1.0, 1.0, 0.0, 3.0};
    Tensor<1,2,double> n = {1.0, 0.0};
    std::pair<Tensor<1,4,double>, double> fluxRoe = flux_roe(U, U, n);
    Tensor<1,4,double> fluxCell = Cell_Flux(U, n);
    EXPECT_TRUE(fluxRoe.first == fluxCell);
}

TEST(RoeTest, Conservation) {
    Tensor<1,4,double> UL = {1.0, 3.0, 4.0, 17.5};
    Tensor<1,4,double> UR = {1.0, 1.0, 2.0, 10.0};
    Tensor<1,2,double> n = {0.7071, 0.7071};
    std::pair<Tensor<1,4,double>, double> fluxRoe_1 = flux_roe(UL, UR, n);
    std::pair<Tensor<1,4,double>, double> fluxRoe_2 = flux_roe(UR, UL, -1.0 * n);
    EXPECT_TRUE(fluxRoe_1.first == -1.0 * fluxRoe_2.first);
    EXPECT_NEAR(fluxRoe_1.second, fluxRoe_2.second, 1e-10);
}

TEST(RoeTest, EntropyFixStagnation) {
    Tensor<1,4,double> U = {1.0, 0.001, 0.0, 2.5000005}; 
    Tensor<1,2,double> n = {1.0, 0.0};
    std::pair<Tensor<1,4,double>, double> fluxRoe = flux_roe(U, U, n);
    EXPECT_FALSE(std::isnan(fluxRoe.first[0]));
    EXPECT_GT(fluxRoe.second, 0.0);
}
