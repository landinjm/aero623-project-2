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

// =============================================================================
// 1. PRIMITIVE VARIABLE & PROPERTY TESTS
// =============================================================================

TEST(Flux, conservative_to_primitive_3d)
{
  auto result = run_on_device(4, [](auto& r) {
    RealType u, v, w, p;
    // State: rho=1.0, rhou=2.0, rhov=3.0, rhow=4.0, rhoE=20.0
    Flux<RealType>::conservative_to_primitive(1.0, 2.0, 3.0, 4.0, 20.0, u, v, w, p);
    r(0) = u; r(1) = v; r(2) = w; r(3) = p;
  });

  EXPECT_NEAR(result(0), 2.0, tol);
  EXPECT_NEAR(result(1), 3.0, tol);
  EXPECT_NEAR(result(2), 4.0, tol);
  // p = (gamma-1)*(E - 0.5*rho*(u^2+v^2+w^2)) = 0.4 * (20 - 0.5*1*(4+9+16)) = 2.2
  EXPECT_NEAR(result(3), 2.2, tol);
}

TEST(Flux, speed_of_sound_basic)
{
  auto result = run_on_device(1, [](auto& r) { 
    r(0) = Flux<RealType>::speed_of_sound(1.0, 1.0); 
  });
  EXPECT_NEAR(result(0), Kokkos::sqrt(1.4), tol);
}

// =============================================================================
// 2. EULER FLUX TESTS (3D)
// =============================================================================

TEST(Flux, euler_flux_3d_z_direction)
{
  auto result = run_on_device(5, [](auto& r) {
    RealType Frho, Frhou, Frhov, Frhow, FrhoE;
    // Normal in Z, velocity purely in Z
    Flux<RealType>::euler_flux(1.0, 0.0, 0.0, 2.0, 1.0, 10.0, 0.0, 0.0, 1.0,
                               Frho, Frhou, Frhov, Frhow, FrhoE);
    r(0) = Frho; r(1) = Frhow; r(2) = FrhoE; r(3) = Frhou; r(4) = Frhov;
  });

  EXPECT_NEAR(result(0), 2.0, tol);   // rho * w
  EXPECT_NEAR(result(1), 5.0, tol);   // rho * w * w + p = 1*4 + 1
  EXPECT_NEAR(result(2), 22.0, tol);  // (rhoE + p) * w = (10 + 1) * 2
  EXPECT_NEAR(result(3), 0.0, tol);   // rho * u * w
  EXPECT_NEAR(result(4), 0.0, tol);   // rho * v * w
}

// =============================================================================
// 3. BOUNDARY CONDITION TESTS (Group 4 Airfoil Requirements)
// =============================================================================

TEST(Flux, inviscid_wall_flux_3d)
{
  auto result = run_on_device(3, [](auto& r) {
    RealType Frho, Frhou, Frhov, Frhow, FrhoE, s;
    // Normal in X, flow heading into wall. 
    // Wall flux should return only pressure in the normal direction.
    Flux<RealType>::inviscid_wall_flux(1.0, 2.0, 0.0, 0.0, 10.0, 1.0, 0.0, 0.0, 
                                       Frho, Frhou, Frhov, Frhow, FrhoE, s);
    r(0) = Frho; r(1) = FrhoE; r(2) = Frhou;
  });

  EXPECT_NEAR(result(0), 0.0, tol); // Mass flux must be zero
  EXPECT_NEAR(result(1), 0.0, tol); // Energy flux must be zero
  // p_b = 0.4 * (10 - 0.5*1.0*(0^2)) = 4.0 (assuming u_t=0)
  EXPECT_NEAR(result(2), 4.0, tol); 
}

TEST(Flux, subsonic_inflow_test)
{
  auto result = run_on_device(1, [](auto& r) {
    RealType Frho, Frhou, Frhov, Frhow, FrhoE;
    // Current test uses n = [-1, 0, 0] and rho_u = 0.1 (velocity = +0.1)
    // The dot product v_n = -0.1. 
    Flux<RealType>::subsonic_inflow_flux(1.0, 0.1, 0.0, 0.0, 2.0, 
                                         -1.0, 0.0, 0.0, 
                                         Frho, Frhou, Frhov, Frhow, FrhoE); 
    r(0) = Frho;
  });
  // Change this to check that the flux is negative (outward normal, inward flow)
  EXPECT_LT(result(0), 0.0); 
}

// =============================================================================
// 4. ROE FLUX TESTS (3D CONSISTENCY & CONSERVATION)
// =============================================================================

TEST(Flux, roe_flux_3d_consistency)
{
  auto result = run_on_device(5, [](auto& r) {
    RealType Frho, Frhou, Frhov, Frhow, FrhoE, s;
    RealType rho=1.0, rhou=0.5, rhov=0.2, rhow=0.1, rhoE=10.0;
    RealType nx=0.0, ny=0.0, nz=1.0;

    Flux<RealType>::roe_flux(rho, rhou, rhov, rhow, rhoE,
                             rho, rhou, rhov, rhow, rhoE,
                             nx, ny, nz,
                             Frho, Frhou, Frhov, Frhow, FrhoE, s);
    r(0) = Frho; r(1) = Frhow; r(2) = FrhoE; r(3) = s;
  });

  EXPECT_NEAR(result(0), 0.1, tol);   // Match Euler: rho*w
  EXPECT_NEAR(result(1), 3.95, tol);  // Match Euler: rho*w*w + p
  EXPECT_GT(result(3), 0.0);          // s_mag must be positive
}

TEST(Flux, roe_flux_3d_conservation)
{
  auto result = run_on_device(2, [](auto& r) {
    RealType F1, F2, u2, u3, u4, u5, s;
    // Case 1: L to R
    Flux<RealType>::roe_flux(1.0, 0.1, 0.1, 0.1, 2.0, 
                             1.2, 0.2, 0.2, 0.2, 2.5, 
                             1.0, 0.0, 0.0, F1, u2, u3, u4, u5, s);
    // Case 2: Swapped L/R and flipped normal
    u2=0; u3=0; u4=0; u5=0; s=0; // Ensure s is also reset
  Flux<RealType>::roe_flux(1.2, 0.2, 0.2, 0.2, 2.5, 
                           1.0, 0.1, 0.1, 0.1, 2.0, 
                           -1.0, 0.0, 0.0, F2, u2, u3, u4, u5, s);
    r(0) = F1; r(1) = F2;
  });
  EXPECT_NEAR(result(0), -result(1), tol);
}

// =============================================================================
// 5. Unsteady runs
// =============================================================================

TEST(Flux, unsteady_inflow_stator_wake)
{
  auto result = run_on_device(2, [](auto& r) {
    RealType Frho1, Frho2, Frhou, Frhov, Frhow, FrhoE, s;
    
    RealType y_pos = 0.5;
    RealType t_freestream = 0.0;
    RealType t_wake = 0.1;

    // Call unsteady version (Matches 14-argument signature in flux.hpp)
    // 5 inputs, 3 normals, 2 space-time (y, t), 5 outputs (including s_mag)
    Flux<RealType>::unsteady_subsonic_inflow_flux(1.0, 0.2, 0.0, 0.0, 2.0, 
                                                  -1.0, 0.0, 0.0, 
                                                  y_pos, t_freestream,
                                                  Frho1, Frhou, Frhov, Frhow, FrhoE, s);
    
    Flux<RealType>::unsteady_subsonic_inflow_flux(1.0, 0.2, 0.0, 0.0, 2.0, 
                                                  -1.0, 0.0, 0.0, 
                                                  y_pos, t_wake,
                                                  Frho2, Frhou, Frhov, Frhow, FrhoE, s);
    r(0) = Frho1; 
    r(1) = Frho2;
  });

  EXPECT_NE(result(0), result(1));
  EXPECT_GT(Kokkos::abs(result(0)), Kokkos::abs(result(1))); 
}