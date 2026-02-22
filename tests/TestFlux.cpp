#include <cmath>
#include <gtest/gtest.h>
#include <iostream>

#include "flux.hpp"
#include "tensor.hpp"

TEST(CellFlux, CorrectOutput)
{
  Tensor<1, 4, double> U = { 1, 0, 0, 0 };
  Tensor<1, 2, double> n = { 1, 0 };
  Tensor<1, 4, double> fluxCell = euler_flux(U, n);
  Tensor<1, 4, double> fluxActual = { 0, 0, 0, 0 };
  EXPECT_TRUE(fluxCell == fluxActual);

  U = { 1, 3, 4, 17.5 };
  n = { 1, 1 };
  fluxCell = euler_flux(U, n);
  fluxActual = { 7, 23, 30, 136.5 };
  EXPECT_TRUE(fluxCell == fluxActual);
}

// F_hat(u,u,n) = F(u)
TEST(HLLETest, Consistency)
{
  Tensor<1, 4, double> U = { 1, 3, 4, 17.5 };
  Tensor<1, 2, double> n = { 1, 0 };

  std::pair<Tensor<1, 4, double>, double> fluxHLLE = flux_hlle(U, U, n);

  Tensor<1, 4, double> fluxCell = euler_flux(U, n);
  EXPECT_TRUE(fluxCell == fluxHLLE.first);
}

// F_hat(uL,uR,n) = -F_hat(uR,uL,n)
TEST(HLLETest, Conservation)
{
  Tensor<1, 4, double> UL = { 1, 3, 4, 17.5 };
  Tensor<1, 4, double> UR = { 1, 1, 2, 10 };
  Tensor<1, 2, double> n = { 1, 1 };

  std::pair<Tensor<1, 4, double>, double> fluxHLLE_1 = flux_hlle(UL, UR, n);
  std::pair<Tensor<1, 4, double>, double> fluxHLLE_2 =
    flux_hlle(UR, UL, -1 * n);
  EXPECT_TRUE(fluxHLLE_1.first == -1 * fluxHLLE_2.first);
  EXPECT_TRUE(fluxHLLE_1.second == fluxHLLE_2.second);
}

// smag > a
TEST(HLLETest, SupersonicUpwinding)
{
  Tensor<1, 4, double> UL = { 1.225, 735.0, 0.0, 473812.5 };
  Tensor<1, 4, double> UR = { 0.9, 450.0, 0.0, 237500.0 };
  Tensor<1, 2, double> n = { 1, 0 };

  std::pair<Tensor<1, 4, double>, double> fluxHLLE_1 = flux_hlle(UL, UR, n);
  Tensor<1, 4, double> upwindFlux = euler_flux(UL, n);
  EXPECT_TRUE(fluxHLLE_1.first == upwindFlux);
}

//
TEST(HLLETest, CorrectOutput)
{
  Tensor<1, 4, double> UL = { 1, 0, 0, 12.5 };
  Tensor<1, 4, double> UR = { 2, 0, 0, 12.5 };
  Tensor<1, 4, double> F_expected = { -1.322875656, 3, 4, 0 };
  Tensor<1, 2, double> n = { .6, .8 };

  std::pair<Tensor<1, 4, double>, double> fluxHLLE_1 = flux_hlle(UL, UR, n);
  Tensor<1, 4, double> upwindFlux = euler_flux(UL, n);
  EXPECT_TRUE(fluxHLLE_1.first.isApprox(F_expected, 1e-6));
}

//
TEST(RoeTest, Consistency)
{
  Tensor<1, 4, double> U = { 1.0, 1.0, 0.0, 3.0 };
  Tensor<1, 2, double> n = { 1.0, 0.0 };
  std::pair<Tensor<1, 4, double>, double> fluxRoe = flux_roe(U, U, n);
  Tensor<1, 4, double> fluxCell = euler_flux(U, n);
  EXPECT_TRUE(fluxRoe.first == fluxCell);
}

TEST(RoeTest, Conservation)
{
  Tensor<1, 4, double> UL = { 1.0, 3.0, 4.0, 17.5 };
  Tensor<1, 4, double> UR = { 1.0, 1.0, 2.0, 10.0 };
  Tensor<1, 2, double> n = { 0.7071, 0.7071 };

  std::pair<Tensor<1, 4, double>, double> fluxRoe_1 = flux_roe(UL, UR, n);
  std::pair<Tensor<1, 4, double>, double> fluxRoe_2 =
    flux_roe(UR, UL, -1.0 * n);

  // Use isApprox or EXPECT_NEAR for the tensor components
  EXPECT_TRUE(fluxRoe_1.first.isApprox(-1.0 * fluxRoe_2.first, 1e-12));
  EXPECT_NEAR(fluxRoe_1.second, fluxRoe_2.second, 1e-10);
}

TEST(RoeTest, EntropyFixStagnation)
{
  Tensor<1, 4, double> U = { 1.0, 0.001, 0.0, 2.5000005 };
  Tensor<1, 2, double> n = { 1.0, 0.0 };
  std::pair<Tensor<1, 4, double>, double> fluxRoe = flux_roe(U, U, n);
  EXPECT_FALSE(std::isnan(fluxRoe.first[0]));
  EXPECT_GT(fluxRoe.second, 0.0);
}

TEST(RoeTest, SupersonicUpwinding)
{
  // High pressure/velocity flow in the +x direction
  Tensor<1, 4, double> UL = { 1.225, 735.0, 0.0, 473812.5 };
  Tensor<1, 4, double> UR = { 0.9, 450.0, 0.0, 237500.0 };
  Tensor<1, 2, double> n = { 1.0, 0.0 };

  // Note: Use 'flux_roe' (matching your function name from the snippet)
  std::pair<Tensor<1, 4, double>, double> fluxRoe = flux_roe(UL, UR, n);
  Tensor<1, 4, double> upwindFlux = euler_flux(UL, n);

  // For supersonic flow, the numerical flux must equal the upwind flux
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(fluxRoe.first[i], upwindFlux[i], 1e-6);
  }
}

TEST(RoeTest, CorrectOutput)
{
  // rhoL=1, rhoR=2, u=v=0, E=12.5 -> p=5.0
  Tensor<1, 4, double> UL = { 1.0, 0.0, 0.0, 12.5 };
  Tensor<1, 4, double> UR = { 2.0, 0.0, 0.0, 12.5 };
  Tensor<1, 2, double> n = { 0.6, 0.8 };

  // Expected: mass flux = 0, momentum flux = p * n, energy flux = 0
  // F_expected = {0, 5*0.6, 5*0.8, 0} = {0, 3, 4, 0}
  Tensor<1, 4, double> F_expected = { 0.0, 3.0, 4.0, 0.0 };

  std::pair<Tensor<1, 4, double>, double> fluxRoe = flux_roe(UL, UR, n);

  // Check flux values
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(fluxRoe.first[i], F_expected[i], 1e-6);
  }

  // Check max wave speed (should be roughly the sound speed c)
  // p=5, rho_avg~1.5, gamma=1.4 -> c ~ 2.16
  EXPECT_GT(fluxRoe.second, 2.0);
}
