#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <config.hpp>
#include <parameters.hpp>

/**
 * @brief 3D Flux class integrating unsteady stator-wake logic.
 */
template<typename RealType>
class Flux
{
public:
  // ===========================================================================
  // 1. CORE 3D PHYSICS
  // ===========================================================================

  KOKKOS_INLINE_FUNCTION static RealType pressure(RealType rho, RealType u, RealType v, RealType w, RealType rho_E)
  {
    ASSERT(rho > 0, "Density must be positive");
    return (Parameters<RealType>::gamma - RealType(1.0)) *
           (rho_E - RealType(0.5) * rho * (u * u + v * v + w * w));
  }

  KOKKOS_INLINE_FUNCTION static void conservative_to_primitive(
      RealType rho, RealType rho_u, RealType rho_v, RealType rho_w, RealType rho_E,
      RealType& u, RealType& v, RealType& w, RealType& p)
  {
    ASSERT(rho > 0, "Density must be positive");
    u = rho_u / rho;
    v = rho_v / rho;
    w = rho_w / rho;
    p = pressure(rho, u, v, w, rho_E);
    ASSERT(p > 0, "Pressure must be positive");
  }

  KOKKOS_INLINE_FUNCTION static RealType speed_of_sound(RealType rho, RealType p)
  {
    return Kokkos::sqrt(Parameters<RealType>::gamma * p / rho);
  }

  KOKKOS_INLINE_FUNCTION static void euler_flux(
      RealType rho, RealType u, RealType v, RealType w, RealType p, RealType rho_E,
      RealType n_x, RealType n_y, RealType n_z,
      RealType& F_rho, RealType& F_rho_u, RealType& F_rho_v, RealType& F_rho_w, RealType& F_rho_E)
  {
    const RealType v_n = u * n_x + v * n_y + w * n_z;
    F_rho   = rho * v_n;
    F_rho_u = rho * u * v_n + p * n_x;
    F_rho_v = rho * v * v_n + p * n_y;
    F_rho_w = rho * w * v_n + p * n_z;
    F_rho_E = (rho_E + p) * v_n;
  }

  // ===========================================================================
  // 2. UNSTEADY WAKE LOGIC (3D-UPDATED)
  // ===========================================================================

  KOKKOS_INLINE_FUNCTION
  static RealType wake_density(RealType y_rot, RealType t)
  {
    const RealType y_stator = y_rot + Parameters<RealType>::a_0 * t;
    const RealType eta_raw = y_stator / Parameters<RealType>::Delta_y;
    const RealType eta = eta_raw - Kokkos::floor(eta_raw) - RealType(0.5);
    return Parameters<RealType>::rho_0 *
           (RealType(1) -
            Parameters<RealType>::f_wake *
              Kokkos::exp(-eta * eta /
                          (RealType(2) * Parameters<RealType>::delta *
                           Parameters<RealType>::delta)));
  }

  KOKKOS_INLINE_FUNCTION
  static void unsteady_subsonic_inflow_flux(RealType rho_L, RealType rho_u_L, RealType rho_v_L, RealType rho_w_L, RealType rho_E_L,
                                            RealType n_x, RealType n_y, RealType n_z,
                                            RealType y_face, RealType t,
                                            RealType& F_rho, RealType& F_rho_u, RealType& F_rho_v, RealType& F_rho_w, RealType& F_rho_E,
                                            RealType& s_mag)
  {
    constexpr RealType gamma = Parameters<RealType>::gamma;
    constexpr RealType gm1 = gamma - RealType(1.0);
    constexpr RealType inv_gm1 = RealType(1.0) / gm1;
    
    const RealType rho_0 = wake_density(y_face, t);
    const RealType p_0 = rho_0 * Parameters<RealType>::a_0 * Parameters<RealType>::a_0 / gamma;
    const RealType RT_0 = p_0 / rho_0;

    // Primitive variables
    RealType u_L, v_L, w_L, p_L;
    conservative_to_primitive(rho_L, rho_u_L, rho_v_L, rho_w_L, rho_E_L, u_L, v_L, w_L, p_L);

    const RealType c_L = speed_of_sound(rho_L, p_L);
    const RealType v_n_L = u_L * n_x + v_L * n_y + w_L * n_z;

    // Outgoing Riemann invariant
    const RealType j_plus = v_n_L + RealType(2.0) * inv_gm1 * c_L;

    // Inflow direction (assumed 2D inflow, w_in = 0)
    const RealType nx_in = Parameters<RealType>::n_x_0();
    const RealType ny_in = Parameters<RealType>::n_y_0();
    const RealType nz_in = 0.0; 
    const RealType d_n = nx_in * n_x + ny_in * n_y + nz_in * n_z;

    // Quadratic solver for boundary Mach number
    const RealType A = gamma * RT_0 * d_n * d_n - gm1 / RealType(2.0) * j_plus * j_plus;
    const RealType B = RealType(4.0) * gamma * RT_0 * d_n * inv_gm1;
    const RealType C = RealType(4.0) * gamma * RT_0 * inv_gm1 * inv_gm1 - j_plus * j_plus;

    const RealType disc = Kokkos::sqrt(B * B - RealType(4.0) * A * C);
    const RealType M_b1 = (-B + disc) / (RealType(2.0) * A);
    const RealType M_b2 = (-B - disc) / (RealType(2.0) * A);

    RealType M_b = (M_b1 > 0 != M_b2 > 0) ? ((M_b1 > 0) ? M_b1 : M_b2) : 
                   ((Kokkos::abs(M_b1) < Kokkos::abs(M_b2)) ? M_b1 : M_b2);

    // Boundary state
    const RealType denom = RealType(1.0) + RealType(0.5) * gm1 * M_b * M_b;
    const RealType c_b = Kokkos::sqrt(gamma * RT_0 / denom);
    const RealType p_b = p_0 * Kokkos::pow(RealType(1.0) / denom, gamma * inv_gm1);
    const RealType rho_b = p_b / (RT_0 / denom);
    const RealType speed_b = M_b * c_b;
    
    const RealType u_b = speed_b * nx_in;
    const RealType v_b = speed_b * ny_in;
    const RealType w_b = speed_b * nz_in; // 0.0 for 2D-axial wake

    const RealType rho_E_b = p_b * inv_gm1 + RealType(0.5) * rho_b * (u_b * u_b + v_b * v_b + w_b * w_b);

    // Final 3D Flux Assembly
    euler_flux(rho_b, u_b, v_b, w_b, p_b, rho_E_b, n_x, n_y, n_z, 
               F_rho, F_rho_u, F_rho_v, F_rho_w, F_rho_E);

    s_mag = Kokkos::abs(u_b * n_x + v_b * n_y + w_b * n_z) + c_b;
  }

  KOKKOS_INLINE_FUNCTION static void roe_flux(RealType rho_L, RealType rho_u_L, RealType rho_v_L, RealType rho_w_L, RealType rho_E_L,
                                              RealType rho_R, RealType rho_u_R, RealType rho_v_R, RealType rho_w_R, RealType rho_E_R,
                                              RealType n_x, RealType n_y, RealType n_z,
                                              RealType& F_rho, RealType& F_rho_u, RealType& F_rho_v, RealType& F_rho_w, RealType& F_rho_E,
                                              RealType& s_mag)
  {
    RealType u_L, v_L, w_L, p_L, u_R, v_R, w_R, p_R;
    conservative_to_primitive(rho_L, rho_u_L, rho_v_L, rho_w_L, rho_E_L, u_L, v_L, w_L, p_L);
    conservative_to_primitive(rho_R, rho_u_R, rho_v_R, rho_w_R, rho_E_R, u_R, v_R, w_R, p_R);

    const RealType H_L = (rho_E_L + p_L) / rho_L;
    const RealType H_R = (rho_E_R + p_R) / rho_R;

    const RealType sL = Kokkos::sqrt(rho_L);
    const RealType sR = Kokkos::sqrt(rho_R);
    const RealType inv_den = 1.0 / (sL + sR);

    const RealType u = (sL * u_L + sR * u_R) * inv_den;
    const RealType v = (sL * v_L + sR * v_R) * inv_den;
    const RealType w = (sL * w_L + sR * w_R) * inv_den;
    const RealType H = (sL * H_L + sR * H_R) * inv_den;

    const RealType gm1 = Parameters<RealType>::gamma - 1.0;
    const RealType q2 = u * u + v * v + w * w;
    const RealType c = Kokkos::sqrt(gm1 * (H - 0.5 * q2));
    const RealType v_n = u * n_x + v * n_y + w * n_z;

    const RealType dp = p_R - p_L;
    const RealType drho = rho_R - rho_L;
    const RealType du = u_R - u_L;
    const RealType dv = v_R - v_L;
    const RealType dw = w_R - w_L;
    const RealType dv_n = du * n_x + dv * n_y + dw * n_z;

    // Alpha wave strengths - expanded from 2D logic
    const RealType a1 = 0.5 * (dp - rho_L * c * dv_n) / (c * c);
    const RealType a2 = drho - dp / (c * c);
    const RealType a3_u = du - dv_n * n_x; // Shear u
    const RealType a3_v = dv - dv_n * n_y; // Shear v
    const RealType a3_w = dw - dv_n * n_z; // Shear w
    const RealType a4 = 0.5 * (dp + rho_L * c * dv_n) / (c * c);

    auto efix = [&](RealType lam) {
      RealType eps = 0.1 * c;
      return (Kokkos::abs(lam) < eps) ? (lam * lam + eps * eps) / (2.0 * eps) : Kokkos::abs(lam);
    };

    const RealType l1 = efix(v_n - c);
    const RealType l2 = efix(v_n);
    const RealType l4 = efix(v_n + c);

    // Dissipation terms (consistent with your original 2D structure)
    const RealType d_rho = l1 * a1 + l2 * a2 + l4 * a4;
    const RealType d_rhou = l1 * a1 * (u - c * n_x) + l2 * (a2 * u + rho_L * a3_u) + l4 * a4 * (u + c * n_x);
    const RealType d_rhov = l1 * a1 * (v - c * n_y) + l2 * (a2 * v + rho_L * a3_v) + l4 * a4 * (v + c * n_y);
    const RealType d_rhow = l1 * a1 * (w - c * n_z) + l2 * (a2 * w + rho_L * a3_w) + l4 * a4 * (w + c * n_z);
    const RealType d_rhoE = l1 * a1 * (H - c * v_n) + l2 * (a2 * 0.5 * q2 + rho_L * (u * a3_u + v * a3_v + w * a3_w)) + l4 * a4 * (H + c * v_n);

    RealType FL[5], FR[5];
    euler_flux(rho_L, u_L, v_L, w_L, p_L, rho_E_L, n_x, n_y, n_z, FL[0], FL[1], FL[2], FL[3], FL[4]);
    euler_flux(rho_R, u_R, v_R, w_R, p_R, rho_E_R, n_x, n_y, n_z, FR[0], FR[1], FR[2], FR[3], FR[4]);

    F_rho = 0.5 * (FL[0] + FR[0] - d_rho);
    F_rho_u = 0.5 * (FL[1] + FR[1] - d_rhou);
    F_rho_v = 0.5 * (FL[2] + FR[2] - d_rhov);
    F_rho_w = 0.5 * (FL[3] + FR[3] - d_rhow);
    F_rho_E = 0.5 * (FL[4] + FR[4] - d_rhoE);
    s_mag = Kokkos::abs(v_n) + c;
  }

  // Updated to accept time-varying rho_0_in for unsteady stator wake
  KOKKOS_INLINE_FUNCTION static void subsonic_inflow_flux(RealType rho_L, RealType rho_u_L, RealType rho_v_L, RealType rho_w_L, RealType rho_E_L,
                                                         RealType n_x, RealType n_y, RealType n_z, RealType rho_0_in,
                                                         RealType& F_rho, RealType& F_rho_u, RealType& F_rho_v, RealType& F_rho_w, RealType& F_rho_E,
                                                         RealType& s_mag)
  {
    const RealType p_0 = Parameters<RealType>::p_0;
    RealType u_L, v_L, w_L, p_L;
    conservative_to_primitive(rho_L, rho_u_L, rho_v_L, rho_w_L, rho_E_L, u_L, v_L, w_L, p_L);
    
    // Simple isentropic boundary state
    const RealType p_b = p_0 * Kokkos::pow(rho_0_in / Parameters<RealType>::rho_0, Parameters<RealType>::gamma);
    euler_flux(rho_0_in, u_L, v_L, w_L, p_b, rho_E_L, n_x, n_y, n_z, F_rho, F_rho_u, F_rho_v, F_rho_w, F_rho_E);
    s_mag = Kokkos::abs(u_L * n_x + v_L * n_y + w_L * n_z) + speed_of_sound(rho_0_in, p_b);
  }

  KOKKOS_INLINE_FUNCTION static void subsonic_outflow_flux(RealType rho_L, RealType rho_u_L, RealType rho_v_L, RealType rho_w_L, RealType rho_E_L,
                                                         RealType n_x, RealType n_y, RealType n_z,
                                                         RealType& F_rho, RealType& F_rho_u, RealType& F_rho_v, RealType& F_rho_w, RealType& F_rho_E,
                                                         RealType& s_mag)
  {
    const RealType p_out = Parameters<RealType>::p_out;
    RealType u_L, v_L, w_L, p_L;
    conservative_to_primitive(rho_L, rho_u_L, rho_v_L, rho_w_L, rho_E_L, u_L, v_L, w_L, p_L);
    euler_flux(rho_L, u_L, v_L, w_L, p_out, rho_E_L, n_x, n_y, n_z, F_rho, F_rho_u, F_rho_v, F_rho_w, F_rho_E);
    s_mag = Kokkos::abs(u_L * n_x + v_L * n_y + w_L * n_z) + speed_of_sound(rho_L, p_out);
  }

  KOKKOS_INLINE_FUNCTION static void inviscid_wall_flux(RealType rho, RealType rho_u, RealType rho_v, RealType rho_w, RealType rho_E,
                                                        RealType n_x, RealType n_y, RealType n_z,
                                                        RealType& F_rho, RealType& F_rho_u, RealType& F_rho_v, RealType& F_rho_w, RealType& F_rho_E,
                                                        RealType& s_mag)
  {
    RealType u, v, w, p;
    conservative_to_primitive(rho, rho_u, rho_v, rho_w, rho_E, u, v, w, p);
    // Only pressure acts on the wall; velocity-based fluxes are zero
    F_rho = 0.0;
    F_rho_u = p * n_x;
    F_rho_v = p * n_y;
    F_rho_w = p * n_z;
    F_rho_E = 0.0;
    s_mag = speed_of_sound(rho, p);
  }
};

