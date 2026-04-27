// #include <config.hpp>
// #include <flux.hpp>

// #include <gtest/gtest.h>

// using RealType = double;
// static constexpr RealType tol = 1.0e-12;

// using ResultViewHost = VectorViewTrait<RealType, HostMemSpace>::type;

// template<typename function>
// auto
// run_on_host(int n_outputs, function f)
// {
//   ResultViewHost result("result", n_outputs);
//   f(result);
//   return result;
// }

// TEST(Flux, 2d_conservative_to_primitive_basic)
// {
//   auto result = run_on_host(3, [](auto& r) {
//     Tensor<1, 2, RealType> rho_v{ 2.0, 3.0 };
//     Tensor<1, 2, RealType> v;
//     RealType p;

//     Flux<2, RealType>::conservative_to_primitive(1.0, rho_v, 9.0, v, p);
//     r(0) = v(0);
//     r(1) = v(1);
//     r(2) = p;
//   });

//   EXPECT_NEAR(result(0), 2.0, tol);
//   EXPECT_NEAR(result(1), 3.0, tol);
//   EXPECT_NEAR(result(2), 1.0, tol);
// }

// TEST(Flux, 2d_conservative_to_primitive_zero_velocity)
// {
//   auto result = run_on_host(3, [](auto& r) {
//     Tensor<1, 2, RealType> rho_v{ 0.0, 0.0 };
//     Tensor<1, 2, RealType> v;
//     RealType p;

//     Flux<2, RealType>::conservative_to_primitive(1.0, rho_v, 10.0, v, p);
//     r(0) = v(0);
//     r(1) = v(1);
//     r(2) = p;
//   });

//   EXPECT_NEAR(result(0), 0.0, tol);
//   EXPECT_NEAR(result(1), 0.0, tol);
//   EXPECT_NEAR(result(2), 4.0, tol);
// }

// TEST(Flux, 3d_conservative_to_primitive_basic)
// {
//   auto result = run_on_host(4, [](auto& r) {
//     Tensor<1, 3, RealType> rho_v{ 2.0, 3.0, 4.0 };
//     Tensor<1, 3, RealType> v;
//     RealType p;

//     Flux<3, RealType>::conservative_to_primitive(1.0, rho_v, 15.0, v, p);
//     r(0) = v(0);
//     r(1) = v(1);
//     r(2) = v(2);
//     r(3) = p;
//   });

//   EXPECT_NEAR(result(0), 2.0, tol);
//   EXPECT_NEAR(result(1), 3.0, tol);
//   EXPECT_NEAR(result(2), 4.0, tol);
//   EXPECT_NEAR(result(3), 0.2, tol);
// }

// TEST(Flux, 3d_conservative_to_primitive_zero_velocity)
// {
//   auto result = run_on_host(4, [](auto& r) {
//     Tensor<1, 3, RealType> rho_v{ 0.0, 0.0, 0.0 };
//     Tensor<1, 3, RealType> v;
//     RealType p;

//     Flux<3, RealType>::conservative_to_primitive(1.0, rho_v, 10.0, v, p);
//     r(0) = v(0);
//     r(1) = v(1);
//     r(2) = v(2);
//     r(3) = p;
//   });

//   EXPECT_NEAR(result(0), 0.0, tol);
//   EXPECT_NEAR(result(1), 0.0, tol);
//   EXPECT_NEAR(result(2), 0.0, tol);
//   EXPECT_NEAR(result(3), 4.0, tol);
// }

// TEST(Flux, 2d_speed_of_sound_basic)
// {
//   auto result = run_on_host(
//     1, [](auto& r) { r(0) = Flux<2, RealType>::speed_of_sound(1.0, 1.0); });

//   EXPECT_NEAR(result(0), Kokkos::sqrt(1.4), tol);
// }

// TEST(Flux, 3d_speed_of_sound_basic)
// {
//   auto result = run_on_host(
//     1, [](auto& r) { r(0) = Flux<3, RealType>::speed_of_sound(1.0, 1.0); });

//   EXPECT_NEAR(result(0), Kokkos::sqrt(1.4), tol);
// }

// TEST(Flux, 2d_max_wavespeed_aligned_normal_x)
// {
//   auto result = run_on_host(1, [](auto& r) {
//     Tensor<1, 2, RealType> v{ 2.0, 3.0 };
//     Tensor<1, 2, RealType> n{ 1.0, 0.0 };

//     r(0) = Flux<2, RealType>::max_wavespeed(1.0, v, 1.0, n);
//   });

//   EXPECT_NEAR(result(0), 2.0 + Kokkos::sqrt(1.4), tol);
// }

// TEST(Flux, 2d_max_wavespeed_aligned_normal_x_negative_velocity)
// {
//   auto result = run_on_host(1, [](auto& r) {
//     Tensor<1, 2, RealType> v{ 2.0, 3.0 };
//     Tensor<1, 2, RealType> n{ 1.0, 0.0 };

//     r(0) = Flux<2, RealType>::max_wavespeed(1.0, -v, 1.0, n);
//   });

//   EXPECT_NEAR(result(0), 2.0 + Kokkos::sqrt(1.4), tol);
// }

// TEST(Flux, 2d_max_wavespeed_diagonal_normal)
// {
//   auto result = run_on_host(1, [](auto& r) {
//     using Kokkos::sqrt;

//     Tensor<1, 2, RealType> v{ 1.0, 1.0 };
//     Tensor<1, 2, RealType> n{ 1.0 / sqrt(2.0), 1.0 / sqrt(2.0) };

//     r(0) = Flux<2, RealType>::max_wavespeed(1.0, v, 1.0, n);
//   });

//   EXPECT_NEAR(result(0), Kokkos::sqrt(2.0) + Kokkos::sqrt(1.4), tol);
// }

// TEST(Flux, 3d_max_wavespeed_aligned_normal_x)
// {
//   auto result = run_on_host(1, [](auto& r) {
//     Tensor<1, 3, RealType> v{ 2.0, 3.0, 4.0 };
//     Tensor<1, 3, RealType> n{ 1.0, 0.0, 0.0 };

//     r(0) = Flux<3, RealType>::max_wavespeed(1.0, v, 1.0, n);
//   });

//   EXPECT_NEAR(result(0), 2.0 + Kokkos::sqrt(1.4), tol);
// }

// TEST(Flux, 3d_max_wavespeed_aligned_normal_x_negative_velocity)
// {
//   auto result = run_on_host(1, [](auto& r) {
//     Tensor<1, 3, RealType> v{ 2.0, 3.0, 4.0 };
//     Tensor<1, 3, RealType> n{ 1.0, 0.0, 0.0 };

//     r(0) = Flux<3, RealType>::max_wavespeed(1.0, -v, 1.0, n);
//   });

//   EXPECT_NEAR(result(0), 2.0 + Kokkos::sqrt(1.4), tol);
// }

// TEST(Flux, 3d_max_wavespeed_diagonal_normal)
// {
//   auto result = run_on_host(1, [](auto& r) {
//     using Kokkos::sqrt;

//     Tensor<1, 3, RealType> v{ 1.0, 1.0, 1.0 };
//     Tensor<1, 3, RealType> n{ 1.0 / sqrt(3.0),
//                               1.0 / sqrt(3.0),
//                               1.0 / sqrt(3.0) };

//     r(0) = Flux<3, RealType>::max_wavespeed(1.0, v, 1.0, n);
//   });

//   EXPECT_NEAR(result(0), Kokkos::sqrt(3.0) + Kokkos::sqrt(1.4), tol);
// }

// TEST(Flux, 2d_euler_flux_aligned_normal_x)
// {
//   auto result = run_on_host(4, [](auto& r) {
//     Tensor<1, 2, RealType> v{ 2.0, 3.0 };
//     Tensor<1, 2, RealType> n{ 1.0, 0.0 };
//     RealType F_rho;
//     Tensor<1, 2, RealType> F_rho_v;
//     RealType F_rho_E;

//     Flux<2, RealType>::euler_flux(1.0, v, 1.0, 9.0, n, F_rho, F_rho_v, F_rho_E);
//     r(0) = F_rho;
//     r(1) = F_rho_v(0);
//     r(2) = F_rho_v(1);
//     r(3) = F_rho_E;
//   });

//   EXPECT_NEAR(result(0), 2.0, tol);
//   EXPECT_NEAR(result(1), 5.0, tol);
//   EXPECT_NEAR(result(2), 6.0, tol);
//   EXPECT_NEAR(result(3), 20.0, tol);
// }

// TEST(Flux, 2d_euler_flux_aligned_normal_y)
// {
//   auto result = run_on_host(4, [](auto& r) {
//     Tensor<1, 2, RealType> v{ 2.0, 3.0 };
//     Tensor<1, 2, RealType> n{ 0.0, 1.0 };
//     RealType F_rho;
//     Tensor<1, 2, RealType> F_rho_v;
//     RealType F_rho_E;

//     Flux<2, RealType>::euler_flux(1.0, v, 1.0, 9.0, n, F_rho, F_rho_v, F_rho_E);
//     r(0) = F_rho;
//     r(1) = F_rho_v(0);
//     r(2) = F_rho_v(1);
//     r(3) = F_rho_E;
//   });

//   EXPECT_NEAR(result(0), 3.0, tol);
//   EXPECT_NEAR(result(1), 6.0, tol);
//   EXPECT_NEAR(result(2), 10.0, tol);
//   EXPECT_NEAR(result(3), 30.0, tol);
// }

// TEST(Flux, 2d_euler_flux_zero_normal_velocity)
// {
//   auto result = run_on_host(4, [](auto& r) {
//     using Kokkos::sqrt;

//     Tensor<1, 2, RealType> v{ -1.0 / sqrt(2.0), 1.0 / sqrt(2.0) };
//     Tensor<1, 2, RealType> n{ 1.0 / sqrt(2.0), 1.0 / sqrt(2.0) };
//     RealType F_rho;
//     Tensor<1, 2, RealType> F_rho_v;
//     RealType F_rho_E;

//     Flux<2, RealType>::euler_flux(1.0, v, 3.4, 9.0, n, F_rho, F_rho_v, F_rho_E);
//     r(0) = F_rho;
//     r(1) = F_rho_v(0);
//     r(2) = F_rho_v(1);
//     r(3) = F_rho_E;
//   });

//   EXPECT_NEAR(result(0), 0.0, tol);
//   EXPECT_NEAR(result(1), 3.4 / Kokkos::sqrt(2.0), tol);
//   EXPECT_NEAR(result(2), 3.4 / Kokkos::sqrt(2.0), tol);
//   EXPECT_NEAR(result(3), 0.0, tol);
// }

// TEST(Flux, 3d_euler_flux_aligned_normal_x)
// {
//   auto result = run_on_host(5, [](auto& r) {
//     Tensor<1, 3, RealType> v{ 2.0, 3.0, 4.0 };
//     Tensor<1, 3, RealType> n{ 1.0, 0.0, 0.0 };
//     RealType F_rho;
//     Tensor<1, 3, RealType> F_rho_v;
//     RealType F_rho_E;

//     Flux<3, RealType>::euler_flux(
//       1.0, v, 1.4, 18.0, n, F_rho, F_rho_v, F_rho_E);
//     r(0) = F_rho;
//     r(1) = F_rho_v(0);
//     r(2) = F_rho_v(1);
//     r(3) = F_rho_v(2);
//     r(4) = F_rho_E;
//   });

//   EXPECT_NEAR(result(0), 2.0, tol);
//   EXPECT_NEAR(result(1), 5.4, tol);
//   EXPECT_NEAR(result(2), 6.0, tol);
//   EXPECT_NEAR(result(3), 8.0, tol);
//   EXPECT_NEAR(result(4), 38.8, tol);
// }

// TEST(Flux, 3d_euler_flux_aligned_normal_y)
// {
//   auto result = run_on_host(5, [](auto& r) {
//     Tensor<1, 3, RealType> v{ 2.0, 3.0, 4.0 };
//     Tensor<1, 3, RealType> n{ 0.0, 1.0, 0.0 };
//     RealType F_rho;
//     Tensor<1, 3, RealType> F_rho_v;
//     RealType F_rho_E;

//     Flux<3, RealType>::euler_flux(
//       1.0, v, 1.4, 18.0, n, F_rho, F_rho_v, F_rho_E);
//     r(0) = F_rho;
//     r(1) = F_rho_v(0);
//     r(2) = F_rho_v(1);
//     r(3) = F_rho_v(2);
//     r(4) = F_rho_E;
//   });

//   EXPECT_NEAR(result(0), 3.0, tol);
//   EXPECT_NEAR(result(1), 6.0, tol);
//   EXPECT_NEAR(result(2), 10.4, tol);
//   EXPECT_NEAR(result(3), 12.0, tol);
//   EXPECT_NEAR(result(4), 58.2, tol);
// }

// TEST(Flux, 3d_euler_flux_aligned_normal_z)
// {
//   auto result = run_on_host(5, [](auto& r) {
//     Tensor<1, 3, RealType> v{ 2.0, 3.0, 4.0 };
//     Tensor<1, 3, RealType> n{ 0.0, 0.0, 1.0 };
//     RealType F_rho;
//     Tensor<1, 3, RealType> F_rho_v;
//     RealType F_rho_E;

//     Flux<3, RealType>::euler_flux(
//       1.0, v, 1.4, 18.0, n, F_rho, F_rho_v, F_rho_E);
//     r(0) = F_rho;
//     r(1) = F_rho_v(0);
//     r(2) = F_rho_v(1);
//     r(3) = F_rho_v(2);
//     r(4) = F_rho_E;
//   });

//   EXPECT_NEAR(result(0), 4.0, tol);
//   EXPECT_NEAR(result(1), 8.0, tol);
//   EXPECT_NEAR(result(2), 12.0, tol);
//   EXPECT_NEAR(result(3), 17.4, tol);
//   EXPECT_NEAR(result(4), 77.6, tol);
// }

// TEST(Flux, 3d_euler_flux_zero_normal_velocity)
// {
//   auto result = run_on_host(5, [](auto& r) {
//     using Kokkos::sqrt;

//     Tensor<1, 3, RealType> v{ -1.0 / sqrt(2.0), 1.0 / sqrt(2.0), 0.0 };
//     Tensor<1, 3, RealType> n{ 1.0 / sqrt(3.0),
//                               1.0 / sqrt(3.0),
//                               1.0 / sqrt(3.0) };
//     RealType F_rho;
//     Tensor<1, 3, RealType> F_rho_v;
//     RealType F_rho_E;

//     Flux<3, RealType>::euler_flux(
//       1.0, v, 7.0, 18.0, n, F_rho, F_rho_v, F_rho_E);
//     r(0) = F_rho;
//     r(1) = F_rho_v(0);
//     r(2) = F_rho_v(1);
//     r(3) = F_rho_v(2);
//     r(4) = F_rho_E;
//   });

//   EXPECT_NEAR(result(0), 0.0, tol);
//   EXPECT_NEAR(result(1), 7.0 / Kokkos::sqrt(3.0), tol);
//   EXPECT_NEAR(result(2), 7.0 / Kokkos::sqrt(3.0), tol);
//   EXPECT_NEAR(result(3), 7.0 / Kokkos::sqrt(3.0), tol);
//   EXPECT_NEAR(result(4), 0.0, tol);
// }

// TEST(Flux, 2d_roe_flux_consistency_x)
// {
//   // With equal left and right states, we recover the euler flux.
//   auto result = run_on_host(5, [](auto& r) {
//     Tensor<1, 2, RealType> rho_v{ 2.0, 3.0 };
//     Tensor<1, 2, RealType> n{ 1.0, 0.0 };
//     Tensor<1, 2, RealType> F_rho_v;
//     RealType F_rho, F_rho_E, s_mag;

//     Flux<2, RealType>::roe_flux(
//       1.0, rho_v, 9.0, 1.0, rho_v, 9.0, n, F_rho, F_rho_v, F_rho_E, s_mag);
//     r(0) = F_rho;
//     r(1) = F_rho_v(0);
//     r(2) = F_rho_v(1);
//     r(3) = F_rho_E;
//     r(4) = s_mag;
//   });

//   EXPECT_NEAR(result(0), 2.0, tol);
//   EXPECT_NEAR(result(1), 5.0, tol);
//   EXPECT_NEAR(result(2), 6.0, tol);
//   EXPECT_NEAR(result(3), 20.0, tol);
//   EXPECT_NEAR(result(4), 2.0 + Kokkos::sqrt(1.4), tol);
// }

// TEST(Flux, 2d_roe_flux_consistency_y)
// {
//   // With equal left and right states, we recover the euler flux.
//   auto result = run_on_host(5, [](auto& r) {
//     Tensor<1, 2, RealType> rho_v{ 2.0, 3.0 };
//     Tensor<1, 2, RealType> n{ 0.0, 1.0 };
//     Tensor<1, 2, RealType> F_rho_v;
//     RealType F_rho, F_rho_E, s_mag;

//     Flux<2, RealType>::roe_flux(
//       1.0, rho_v, 9.0, 1.0, rho_v, 9.0, n, F_rho, F_rho_v, F_rho_E, s_mag);
//     r(0) = F_rho;
//     r(1) = F_rho_v(0);
//     r(2) = F_rho_v(1);
//     r(3) = F_rho_E;
//     r(4) = s_mag;
//   });

//   EXPECT_NEAR(result(0), 3.0, tol);
//   EXPECT_NEAR(result(1), 6.0, tol);
//   EXPECT_NEAR(result(2), 10.0, tol);
//   EXPECT_NEAR(result(3), 30.0, tol);
//   EXPECT_NEAR(result(4), 3.0 + Kokkos::sqrt(1.4), tol);
// }

// TEST(Flux, 2d_roe_flux_conservation)
// {
//   // Flipping the sign on the normal vector should flip the sign on the flux
//   auto result = run_on_host(10, [](auto& r) {
//     using Kokkos::sqrt;

//     Tensor<1, 2, RealType> rho_v{ 2.0, 3.0 };
//     Tensor<1, 2, RealType> n{ 1.0 / sqrt(2.0), 1.0 / sqrt(2.0) };
//     Tensor<1, 2, RealType> F_rho_v;
//     RealType F_rho, F_rho_E, s_mag;

//     Flux<2, RealType>::roe_flux(
//       1.0, rho_v, 9.0, 1.0, rho_v, 9.0, n, F_rho, F_rho_v, F_rho_E, s_mag);
//     r(0) = F_rho;
//     r(1) = F_rho_v(0);
//     r(2) = F_rho_v(1);
//     r(3) = F_rho_E;
//     r(4) = s_mag;

//     Flux<2, RealType>::roe_flux(
//       1.0, rho_v, 9.0, 1.0, rho_v, 9.0, -n, F_rho, F_rho_v, F_rho_E, s_mag);
//     r(5) = F_rho;
//     r(6) = F_rho_v(0);
//     r(7) = F_rho_v(1);
//     r(8) = F_rho_E;
//     r(9) = s_mag;
//   });

//   EXPECT_NEAR(result(0), -result(5), tol);
//   EXPECT_NEAR(result(1), -result(6), tol);
//   EXPECT_NEAR(result(2), -result(7), tol);
//   EXPECT_NEAR(result(3), -result(8), tol);
//   EXPECT_NEAR(result(4), result(9), tol);
// }

// TEST(Flux, 2d_roe_flux_basic)
// {
//   auto result = run_on_host(5, [](auto& r) {
//     using Kokkos::sqrt;

//     Tensor<1, 2, RealType> rho_v_L{ 1.0, 0.0 };
//     Tensor<1, 2, RealType> rho_v_R{ 0.0, 1.0 };
//     Tensor<1, 2, RealType> n{ 1.0 / sqrt(2.0), 1.0 / sqrt(2.0) };
//     Tensor<1, 2, RealType> F_rho_v;
//     RealType F_rho, F_rho_E, s_mag;

//     Flux<2, RealType>::roe_flux(
//       1.0, rho_v_L, 1.0, 1.0, rho_v_R, 1.0, n, F_rho, F_rho_v, F_rho_E, s_mag);
//     r(0) = F_rho;
//     r(1) = F_rho_v(0);
//     r(2) = F_rho_v(1);
//     r(3) = F_rho_E;
//     r(4) = s_mag;
//   });

//   EXPECT_NEAR(result(0), 0.707106781186547462, tol);
//   EXPECT_NEAR(result(1), 0.848528137423856910, tol);
//   EXPECT_NEAR(result(2), 0.141421356237309448, tol);
//   EXPECT_NEAR(result(3), 0.848528137423856910, tol);
//   EXPECT_NEAR(result(4), 1.323548181483444885, tol);
// }

// TEST(Flux, 2d_roe_flux_basic_2)
// {
//   auto result = run_on_host(5, [](auto& r) {
//     using Kokkos::sqrt;

//     Tensor<1, 2, RealType> rho_v_L{ 0.1, 0.1 };
//     Tensor<1, 2, RealType> rho_v_R{ 0.2, 0.3 };
//     Tensor<1, 2, RealType> n{ 1.0 / sqrt(2.0), 1.0 / sqrt(2.0) };
//     Tensor<1, 2, RealType> F_rho_v;
//     RealType F_rho, F_rho_E, s_mag;

//     Flux<2, RealType>::roe_flux(
//       1.0, rho_v_L, 1.0, 1.0, rho_v_R, 1.0, n, F_rho, F_rho_v, F_rho_E, s_mag);
//     r(0) = F_rho;
//     r(1) = F_rho_v(0);
//     r(2) = F_rho_v(1);
//     r(3) = F_rho_E;
//     r(4) = s_mag;
//   });

//   EXPECT_NEAR(result(0), 0.221736070823963460, tol);
//   EXPECT_NEAR(result(1), 0.264412956326837201, tol);
//   EXPECT_NEAR(result(2), 0.268428692056169871, tol);
//   EXPECT_NEAR(result(3), 0.294033357867632128, tol);
//   EXPECT_NEAR(result(4), 0.983354209194700735, tol);
// }

// TEST(Flux, 3d_roe_flux_consistency_x)
// {
//   // With equal left and right states, we recover the euler flux.
//   auto result = run_on_host(6, [](auto& r) {
//     Tensor<1, 3, RealType> rho_v{ 2.0, 3.0, 4.0 };
//     Tensor<1, 3, RealType> n{ 1.0, 0.0, 0.0 };
//     Tensor<1, 3, RealType> F_rho_v;
//     RealType F_rho, F_rho_E, s_mag;

//     Flux<3, RealType>::roe_flux(
//       1.0, rho_v, 18.0, 1.0, rho_v, 18.0, n, F_rho, F_rho_v, F_rho_E, s_mag);
//     r(0) = F_rho;
//     r(1) = F_rho_v(0);
//     r(2) = F_rho_v(1);
//     r(3) = F_rho_v(2);
//     r(4) = F_rho_E;
//     r(5) = s_mag;
//   });

//   EXPECT_NEAR(result(0), 2.0, tol);
//   EXPECT_NEAR(result(1), 5.4, tol);
//   EXPECT_NEAR(result(2), 6.0, tol);
//   EXPECT_NEAR(result(3), 8.0, tol);
//   EXPECT_NEAR(result(4), 38.8, tol);
//   EXPECT_NEAR(result(5), 3.4, tol);
// }

// TEST(Flux, 3d_roe_flux_consistency_y)
// {
//   // With equal left and right states, we recover the euler flux.
//   auto result = run_on_host(6, [](auto& r) {
//     Tensor<1, 3, RealType> rho_v{ 2.0, 3.0, 4.0 };
//     Tensor<1, 3, RealType> n{ 0.0, 1.0, 0.0 };
//     Tensor<1, 3, RealType> F_rho_v;
//     RealType F_rho, F_rho_E, s_mag;

//     Flux<3, RealType>::roe_flux(
//       1.0, rho_v, 18.0, 1.0, rho_v, 18.0, n, F_rho, F_rho_v, F_rho_E, s_mag);
//     r(0) = F_rho;
//     r(1) = F_rho_v(0);
//     r(2) = F_rho_v(1);
//     r(3) = F_rho_v(2);
//     r(4) = F_rho_E;
//     r(5) = s_mag;
//   });

//   EXPECT_NEAR(result(0), 3.0, tol);
//   EXPECT_NEAR(result(1), 6.0, tol);
//   EXPECT_NEAR(result(2), 10.4, tol);
//   EXPECT_NEAR(result(3), 12.0, tol);
//   EXPECT_NEAR(result(4), 58.2, tol);
//   EXPECT_NEAR(result(5), 4.4, tol);
// }

// TEST(Flux, 3d_roe_flux_consistency_z)
// {
//   // With equal left and right states, we recover the euler flux.
//   auto result = run_on_host(6, [](auto& r) {
//     Tensor<1, 3, RealType> rho_v{ 2.0, 3.0, 4.0 };
//     Tensor<1, 3, RealType> n{ 0.0, 0.0, 1.0 };
//     Tensor<1, 3, RealType> F_rho_v;
//     RealType F_rho, F_rho_E, s_mag;

//     Flux<3, RealType>::roe_flux(
//       1.0, rho_v, 18.0, 1.0, rho_v, 18.0, n, F_rho, F_rho_v, F_rho_E, s_mag);
//     r(0) = F_rho;
//     r(1) = F_rho_v(0);
//     r(2) = F_rho_v(1);
//     r(3) = F_rho_v(2);
//     r(4) = F_rho_E;
//     r(5) = s_mag;
//   });

//   EXPECT_NEAR(result(0), 4.0, tol);
//   EXPECT_NEAR(result(1), 8.0, tol);
//   EXPECT_NEAR(result(2), 12.0, tol);
//   EXPECT_NEAR(result(3), 17.4, tol);
//   EXPECT_NEAR(result(4), 77.6, tol);
//   EXPECT_NEAR(result(5), 5.4, tol);
// }

// TEST(Flux, 3d_roe_flux_conservation)
// {
//   // Flipping the sign on the normal vector should flip the sign on the flux
//   auto result = run_on_host(12, [](auto& r) {
//     using Kokkos::sqrt;

//     Tensor<1, 3, RealType> rho_v{ 2.0, 3.0, 4.0 };
//     Tensor<1, 3, RealType> n{ 1.0 / sqrt(3.0),
//                               1.0 / sqrt(3.0),
//                               1.0 / sqrt(3.0) };
//     Tensor<1, 3, RealType> F_rho_v;
//     RealType F_rho, F_rho_E, s_mag;

//     Flux<3, RealType>::roe_flux(
//       1.0, rho_v, 18.0, 1.0, rho_v, 18.0, n, F_rho, F_rho_v, F_rho_E, s_mag);
//     r(0) = F_rho;
//     r(1) = F_rho_v(0);
//     r(2) = F_rho_v(1);
//     r(3) = F_rho_v(2);
//     r(4) = F_rho_E;
//     r(5) = s_mag;

//     Flux<3, RealType>::roe_flux(
//       1.0, rho_v, 18.0, 1.0, rho_v, 18.0, -n, F_rho, F_rho_v, F_rho_E, s_mag);
//     r(6) = F_rho;
//     r(7) = F_rho_v(0);
//     r(8) = F_rho_v(1);
//     r(9) = F_rho_v(2);
//     r(10) = F_rho_E;
//     r(11) = s_mag;
//   });

//   EXPECT_NEAR(result(0), -result(6), tol);
//   EXPECT_NEAR(result(1), -result(7), tol);
//   EXPECT_NEAR(result(2), -result(8), tol);
//   EXPECT_NEAR(result(3), -result(9), tol);
//   EXPECT_NEAR(result(4), -result(10), tol);
//   EXPECT_NEAR(result(5), result(11), tol);
// }
