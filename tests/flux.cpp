#include <config.hpp>
#include <flux.hpp>

#include <gtest/gtest.h>

using RealType = double;
static constexpr RealType tol = 1.0e-12;

using ResultViewHost = VectorViewTrait<RealType, HostMemSpace>::type;

template<typename function>
auto
run_on_device(int n_outputs, function f)
{
  ResultViewHost result("result", n_outputs);

  f(result);

  return result;
}

TEST(Flux, conservative_to_primitive_basic)
{
  auto result = run_on_device(3, [](auto& r) {
    RealType u, v, p;
    Flux<RealType>::conservative_to_primitive(1.0, 2.0, 3.0, 9.0, u, v, p);
    r(0) = u;
    r(1) = v;
    r(2) = p;
  });

  EXPECT_NEAR(result(0), 2.0, tol);
  EXPECT_NEAR(result(1), 3.0, tol);
  EXPECT_NEAR(result(2), 1.0, tol);
}

TEST(Flux, conservative_to_primitive_zero_velocity)
{
  auto result = run_on_device(3, [](auto& r) {
    RealType u, v, p;
    Flux<RealType>::conservative_to_primitive(1.0, 0.0, 0.0, 10.0, u, v, p);
    r(0) = u;
    r(1) = v;
    r(2) = p;
  });

  EXPECT_NEAR(result(0), 0.0, tol);
  EXPECT_NEAR(result(1), 0.0, tol);
  EXPECT_NEAR(result(2), 4.0, tol);
}

TEST(Flux, speed_of_sound_basic)
{
  auto result = run_on_device(
    1, [](auto& r) { r(0) = Flux<RealType>::speed_of_sound(1.0, 1.0); });

  EXPECT_NEAR(result(0), Kokkos::sqrt(1.4), tol);
}

TEST(Flux, max_wavespeed_aligned_normal_x)
{
  auto result = run_on_device(1, [](auto& r) {
    r(0) = Flux<RealType>::max_wavespeed(1.0, 2.0, 3.0, 1.0, 1.0, 0.0);
  });

  EXPECT_NEAR(result(0), 2.0 + Kokkos::sqrt(1.4), tol);
}

TEST(Flux, max_wavespeed_aligned_normal_x_negative_velocity)
{
  auto result = run_on_device(1, [](auto& r) {
    r(0) = Flux<RealType>::max_wavespeed(1.0, -2.0, 3.0, 1.0, 1.0, 0.0);
  });

  EXPECT_NEAR(result(0), 2.0 + Kokkos::sqrt(1.4), tol);
}

TEST(Flux, max_wavespeed_diagonal_normal)
{
  auto result = run_on_device(1, [](auto& r) {
    r(0) = Flux<RealType>::max_wavespeed(
      1.0, 1.0, 1.0, 1.0, 1.0 / Kokkos::sqrt(2.0), 1.0 / Kokkos::sqrt(2.0));
  });

  EXPECT_NEAR(result(0), Kokkos::sqrt(2.0) + Kokkos::sqrt(1.4), tol);
}

TEST(Flux, euler_flux_aligned_normal_x)
{
  auto result = run_on_device(4, [](auto& r) {
    RealType F_rho, F_rho_u, F_rho_v, F_rho_E;
    Flux<RealType>::euler_flux(
      1.0, 2.0, 3.0, 1.0, 9.0, 1.0, 0.0, F_rho, F_rho_u, F_rho_v, F_rho_E);
    r(0) = F_rho;
    r(1) = F_rho_u;
    r(2) = F_rho_v;
    r(3) = F_rho_E;
  });

  EXPECT_NEAR(result(0), 2.0, tol);
  EXPECT_NEAR(result(1), 5.0, tol);
  EXPECT_NEAR(result(2), 6.0, tol);
  EXPECT_NEAR(result(3), 20.0, tol);
}

TEST(Flux, euler_flux_aligned_normal_y)
{
  auto result = run_on_device(4, [](auto& r) {
    RealType F_rho, F_rho_u, F_rho_v, F_rho_E;
    Flux<RealType>::euler_flux(
      1.0, 2.0, 3.0, 1.0, 9.0, 0.0, 1.0, F_rho, F_rho_u, F_rho_v, F_rho_E);
    r(0) = F_rho;
    r(1) = F_rho_u;
    r(2) = F_rho_v;
    r(3) = F_rho_E;
  });

  EXPECT_NEAR(result(0), 3.0, tol);
  EXPECT_NEAR(result(1), 6.0, tol);
  EXPECT_NEAR(result(2), 10.0, tol);
  EXPECT_NEAR(result(3), 30.0, tol);
}

TEST(Flux, euler_flux_zero_normal_velocity)
{
  auto result = run_on_device(4, [](auto& r) {
    RealType F_rho, F_rho_u, F_rho_v, F_rho_E;
    Flux<RealType>::euler_flux(1.0,
                               -1.0 / Kokkos::sqrt(2.0),
                               1.0 / Kokkos::sqrt(2.0),
                               1.0,
                               9.0,
                               1.0 / Kokkos::sqrt(2.0),
                               1.0 / Kokkos::sqrt(2.0),
                               F_rho,
                               F_rho_u,
                               F_rho_v,
                               F_rho_E);
    r(0) = F_rho;
    r(1) = F_rho_u;
    r(2) = F_rho_v;
    r(3) = F_rho_E;
  });

  EXPECT_NEAR(result(0), 0.0, tol);
  EXPECT_NEAR(result(1), 1.0 / Kokkos::sqrt(2.0), tol);
  EXPECT_NEAR(result(2), 1.0 / Kokkos::sqrt(2.0), tol);
  EXPECT_NEAR(result(3), 0.0, tol);
}

TEST(Flux, roe_flux_consistency_x)
{
  // With equal left and right states, we recover the euler flux.
  auto result = run_on_device(5, [](auto& r) {
    RealType F_rho, F_rho_u, F_rho_v, F_rho_E, s_mag;
    Flux<RealType>::roe_flux(1.0,
                             2.0,
                             3.0,
                             9.0,
                             1.0,
                             2.0,
                             3.0,
                             9.0,
                             1.0,
                             0.0,
                             F_rho,
                             F_rho_u,
                             F_rho_v,
                             F_rho_E,
                             s_mag);
    r(0) = F_rho;
    r(1) = F_rho_u;
    r(2) = F_rho_v;
    r(3) = F_rho_E;
    r(4) = s_mag;
  });

  EXPECT_NEAR(result(0), 2.0, tol);
  EXPECT_NEAR(result(1), 5.0, tol);
  EXPECT_NEAR(result(2), 6.0, tol);
  EXPECT_NEAR(result(3), 20.0, tol);
  EXPECT_NEAR(result(4), 2.0 + Kokkos::sqrt(1.4), tol);
}

TEST(Flux, roe_flux_consistency_y)
{
  // With equal left and right states, we recover the euler flux.
  auto result = run_on_device(5, [](auto& r) {
    RealType F_rho, F_rho_u, F_rho_v, F_rho_E, s_mag;
    Flux<RealType>::roe_flux(1.0,
                             2.0,
                             3.0,
                             9.0,
                             1.0,
                             2.0,
                             3.0,
                             9.0,
                             0.0,
                             1.0,
                             F_rho,
                             F_rho_u,
                             F_rho_v,
                             F_rho_E,
                             s_mag);
    r(0) = F_rho;
    r(1) = F_rho_u;
    r(2) = F_rho_v;
    r(3) = F_rho_E;
    r(4) = s_mag;
  });

  EXPECT_NEAR(result(0), 3.0, tol);
  EXPECT_NEAR(result(1), 6.0, tol);
  EXPECT_NEAR(result(2), 10.0, tol);
  EXPECT_NEAR(result(3), 30.0, tol);
  EXPECT_NEAR(result(4), 3.0 + Kokkos::sqrt(1.4), tol);
}

TEST(Flux, roe_flux_conservation)
{
  // Flipping the sign on the normal vector should flip the sign on the flux
  auto result = run_on_device(10, [](auto& r) {
    RealType F_rho, F_rho_u, F_rho_v, F_rho_E, s_mag;
    Flux<RealType>::roe_flux(1.0,
                             2.0,
                             3.0,
                             9.0,
                             2.0,
                             2.0,
                             3.0,
                             9.0,
                             1.0 / Kokkos::sqrt(2.0),
                             1.0 / Kokkos::sqrt(2.0),
                             F_rho,
                             F_rho_u,
                             F_rho_v,
                             F_rho_E,
                             s_mag);
    r(0) = F_rho;
    r(1) = F_rho_u;
    r(2) = F_rho_v;
    r(3) = F_rho_E;
    r(4) = s_mag;

    Flux<RealType>::roe_flux(2.0,
                             2.0,
                             3.0,
                             9.0,
                             1.0,
                             2.0,
                             3.0,
                             9.0,
                             -1.0 / Kokkos::sqrt(2.0),
                             -1.0 / Kokkos::sqrt(2.0),
                             F_rho,
                             F_rho_u,
                             F_rho_v,
                             F_rho_E,
                             s_mag);
    r(5) = F_rho;
    r(6) = F_rho_u;
    r(7) = F_rho_v;
    r(8) = F_rho_E;
    r(9) = s_mag;
  });

  EXPECT_NEAR(result(0), -result(5), tol);
  EXPECT_NEAR(result(1), -result(6), tol);
  EXPECT_NEAR(result(2), -result(7), tol);
  EXPECT_NEAR(result(3), -result(8), tol);
  EXPECT_NEAR(result(4), result(9), tol);
}

TEST(Flux, roe_flux_basic)
{
  auto result = run_on_device(5, [](auto& r) {
    RealType F_rho, F_rho_u, F_rho_v, F_rho_E, s_mag;
    Flux<RealType>::roe_flux(1.0,
                             1.0,
                             0.0,
                             1.0,
                             1.0,
                             0.0,
                             1.0,
                             1.0,
                             1.0 / Kokkos::sqrt(2.0),
                             1.0 / Kokkos::sqrt(2.0),
                             F_rho,
                             F_rho_u,
                             F_rho_v,
                             F_rho_E,
                             s_mag);
    r(0) = F_rho;
    r(1) = F_rho_u;
    r(2) = F_rho_v;
    r(3) = F_rho_E;
    r(4) = s_mag;
  });

  EXPECT_NEAR(result(0), 0.707106781186547462, tol);
  EXPECT_NEAR(result(1), 0.848528137423856910, tol);
  EXPECT_NEAR(result(2), 0.141421356237309448, tol);
  EXPECT_NEAR(result(3), 0.848528137423856910, tol);
  EXPECT_NEAR(result(4), 1.323548181483444885, tol);
}

TEST(Flux, roe_flux_basic_2)
{
  auto result = run_on_device(5, [](auto& r) {
    RealType F_rho, F_rho_u, F_rho_v, F_rho_E, s_mag;
    Flux<RealType>::roe_flux(1.0,
                             0.1,
                             0.1,
                             1.0,
                             1.0,
                             0.2,
                             0.3,
                             1.0,
                             1.0 / Kokkos::sqrt(2.0),
                             1.0 / Kokkos::sqrt(2.0),
                             F_rho,
                             F_rho_u,
                             F_rho_v,
                             F_rho_E,
                             s_mag);
    r(0) = F_rho;
    r(1) = F_rho_u;
    r(2) = F_rho_v;
    r(3) = F_rho_E;
    r(4) = s_mag;
  });

  EXPECT_NEAR(result(0), 0.221736070823963460, tol);
  EXPECT_NEAR(result(1), 0.264412956326837201, tol);
  EXPECT_NEAR(result(2), 0.268428692056169871, tol);
  EXPECT_NEAR(result(3), 0.294033357867632128, tol);
  EXPECT_NEAR(result(4), 0.983354209194700735, tol);
}
