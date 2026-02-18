#include <gtest/gtest.h>
#include <solver.hpp>

constexpr double tolerance = 1e-5;
constexpr double rho = 1.0;
constexpr double u = 0.5;
constexpr double v = 0.0;
constexpr double p = 1.0 / Parameters<double>::gamma;
constexpr double E =
  p / (Parameters<double>::gamma - 1.0) + 0.5 * rho * (u * u + v * v);
const Tensor<1, 4, double> interior_state{ rho, rho* u, rho* v, E };
const Tensor<1, 2, double> normal{ -1.0, 0.0 };

TEST(BoundaryFlux, inviscid_wall_2d)
{
  const auto result = inviscid_wall(interior_state, normal);

  const Tensor<1, 4, double> expected_flux{ 0.0, -0.7642857143, 0.0, 0.0 };
  const double expected_max_wavespeed = 1.034408043;

  for (unsigned int i = 0; i < 4; ++i) {
    EXPECT_NEAR(expected_flux[i], result.first[i], tolerance);
  }
  EXPECT_NEAR(expected_max_wavespeed, result.second, tolerance);
}

TEST(BoundaryFlux, subsonic_inflow_2d)
{
  const auto result = subsonic_inflow(interior_state, normal);

  const Tensor<1, 4, double> expected_flux{
    -0.3002405496, -0.6819079786, -0.1251375741, -0.7506013741
  };
  const double expected_max_wavespeed = 1.31967506;

  for (unsigned int i = 0; i < 4; ++i) {
    EXPECT_NEAR(expected_flux[i], result.first[i], tolerance);
  }
  EXPECT_NEAR(expected_max_wavespeed, result.second, tolerance);
}

TEST(BoundaryFlux, subsonic_outflow_2d)
{
  const auto result = subsonic_outflow(interior_state, normal);

  const Tensor<1, 4, double> expected_flux{
    -0.195025113, -0.5490710481, 0, -0.4464979747
  };
  const double expected_max_wavespeed = 1.201936795;

  for (unsigned int i = 0; i < 4; ++i) {
    EXPECT_NEAR(expected_flux[i], result.first[i], tolerance);
  }
  EXPECT_NEAR(expected_max_wavespeed, result.second, tolerance);
}
