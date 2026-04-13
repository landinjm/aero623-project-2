#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <config.hpp>
#include <parameters.hpp>

/**
 * @brief Advanced 3D Flux class.
 * Fully 3D Riemann Invariant BCs with numerical safety guards.
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
  // 2. UNSTEADY WAKE & ADVANCED INFLOW (RIEMANN INVARIANT)
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
              Kokkos::exp(-eta * eta / (RealType(2) * Parameters<RealType>::delta * Parameters<RealType>::delta)));
  }

  KOKKOS_INLINE_FUNCTION
  static void solve_riemann_inflow(RealType rho_L, RealType u_L, RealType v_L, RealType w_L, RealType p_L,
                                   RealType n_x, RealType n_y, RealType n_z,
                                   RealType rho_0, RealType p_0,
                                   RealType& rho_b, RealType& u_b, RealType& v_b, RealType& w_b, RealType& p_b, RealType& c_b)
  {
    constexpr RealType gamma = Parameters<RealType>::gamma;
    constexpr RealType gm1 = gamma - RealType(1.0);
    constexpr RealType inv_gm1 = RealType(1.0) / gm1;
    const RealType RT_0 = p_0 / rho_0;

    const RealType c_L = speed_of_sound(rho_L, p_L);
    const RealType v_n_L = u_L * n_x + v_L * n_y + w_L * n_z;
    const RealType j_plus = v_n_L + RealType(2.0) * inv_gm1 * c_L;

    const RealType nx_in = Parameters<RealType>::n_x_0();
    const RealType ny_in = Parameters<RealType>::n_y_0();
    const RealType nz_in = 0.0; 
    const RealType d_n = nx_in * n_x + ny_in * n_y + nz_in * n_z; 

    const RealType A = gamma * RT_0 * d_n * d_n - gm1 / RealType(2.0) * j_plus * j_plus;
    const RealType B = RealType(4.0) * gamma * RT_0 * d_n * inv_gm1;
    const RealType C = RealType(4.0) * gamma * RT_0 * inv_gm1 * inv_gm1 - j_plus * j_plus;

    const RealType disc = Kokkos::fmax(0.0, B * B - RealType(4.0) * A * C);
    const RealType sqrt_disc = Kokkos::sqrt(disc);
    
    // Safety check for A near zero to avoid NaN
    const RealType M_b1 = (Kokkos::abs(A) > 1e-12) ? (-B + sqrt_disc) / (RealType(2.0) * A) : -C/B;
    const RealType M_b2 = (Kokkos::abs(A) > 1e-12) ? (-B - sqrt_disc) / (RealType(2.0) * A) : -C/B;
    RealType M_b = ((M_b1 > 0) != (M_b2 > 0)) ? ((M_b1 > 0) ? M_b1 : M_b2) : 
                   ((Kokkos::abs(M_b1) < Kokkos::abs(M_b2)) ? M_b1 : M_b2);

    const RealType denom = RealType(1.0) + RealType(0.5) * gm1 * M_b * M_b;
    c_b = Kokkos::sqrt(gamma * RT_0 / denom);
    p_b = p_0 * Kokkos::pow(RealType(1.0) / denom, gamma * inv_gm1);
    rho_b = p_b / (RT_0 / denom);
    
    const RealType speed_b = M_b * c_b;
    u_b = speed_b * nx_in;
    v_b = speed_b * ny_in;
    w_b = speed_b * nz_in; 
  }

  KOKKOS_INLINE_FUNCTION
  static void unsteady_subsonic_inflow_flux(RealType rho_L, RealType rho_u_L, RealType rho_v_L, RealType rho_w_L, RealType rho_E_L,
                                            RealType n_x, RealType n_y, RealType n_z,
                                            RealType y_face, RealType t,
                                            RealType& F_rho, RealType& F_rho_u, RealType& F_rho_v, RealType& F_rho_w, RealType& F_rho_E,
                                            RealType& s_mag)
  {
    const RealType rho_0 = wake_density(y_face, t);
    const RealType p_0 = rho_0 * Parameters<RealType>::a_0 * Parameters<RealType>::a_0 / Parameters<RealType>::gamma;
    
    RealType u_L, v_L, w_L, p_L, rho_b, u_b, v_b, w_b, p_b, c_b;
    conservative_to_primitive(rho_L, rho_u_L, rho_v_L, rho_w_L, rho_E_L, u_L, v_L, w_L, p_L);
    
    solve_riemann_inflow(rho_L, u_L, v_L, w_L, p_L, n_x, n_y, n_z, rho_0, p_0, rho_b, u_b, v_b, w_b, p_b, c_b);

    const RealType rho_E_b = p_b / (Parameters<RealType>::gamma - 1.0) + RealType(0.5) * rho_b * (u_b * u_b + v_b * v_b + w_b * w_b);
    euler_flux(rho_b, u_b, v_b, w_b, p_b, rho_E_b, n_x, n_y, n_z, F_rho, F_rho_u, F_rho_v, F_rho_w, F_rho_E);
    s_mag = Kokkos::abs(u_b * n_x + v_b * n_y + w_b * n_z) + c_b;
  }

  KOKKOS_INLINE_FUNCTION static void subsonic_inflow_flux(
      RealType rho_in, RealType u_in, RealType v_in, RealType w_in, RealType p_in, // Input state
      RealType n_x, RealType n_y, RealType n_z,                                  // Normal
      RealType& f_rho, RealType& f_rhou, RealType& f_rhov, RealType& f_rhow, RealType& f_rhoE)
  {
      // FIX: Ensure 'rho_in' is used to compute the ghost state, not a constant from Parameters
      RealType rho_g = rho_in; 
      RealType u_g   = u_in;
      RealType v_g   = v_in;
      RealType w_g   = w_in;
      RealType p_g   = p_in;

      // Recalculate Energy based on the SPECIFIC rho_in provided
      RealType rhoE_g = p_g / (Parameters<RealType>::gamma - 1.0) + 
                        0.5 * rho_g * (u_g*u_g + v_g*v_g + w_g*w_g);

      // Compute flux using these ghost values...
      RealType vn = u_g*n_x + v_g*n_y + w_g*n_z;
      f_rho  = rho_g * vn;
      f_rhou = rho_g * u_g * vn + p_g * n_x;
      f_rhov = rho_g * v_g * vn + p_g * n_y;
      f_rhow = rho_g * w_g * vn + p_g * n_z;
      f_rhoE = (rhoE_g + p_g) * vn;
  }

  // ===========================================================================
  // 3. ADVANCED OUTFLOW (ENTROPY PRESERVATION)
  // ===========================================================================

  KOKKOS_INLINE_FUNCTION
  static void subsonic_outflow_flux(RealType rho_L, RealType rho_u_L, RealType rho_v_L, RealType rho_w_L, RealType rho_E_L,
                                   RealType n_x, RealType n_y, RealType n_z,
                                   RealType& F_rho, RealType& F_rho_u, RealType& F_rho_v, RealType& F_rho_w, RealType& F_rho_E,
                                   RealType& s_mag)
  {
    constexpr RealType gamma = Parameters<RealType>::gamma;
    constexpr RealType gm1 = gamma - RealType(1.0);
    constexpr RealType inv_gm1 = RealType(1.0) / gm1;
    const RealType p_out = Parameters<RealType>::p_out;

    RealType u_L, v_L, w_L, p_L;
    conservative_to_primitive(rho_L, rho_u_L, rho_v_L, rho_w_L, rho_E_L, u_L, v_L, w_L, p_L);
    const RealType c_L = speed_of_sound(rho_L, p_L);
    const RealType v_n_L = u_L * n_x + v_L * n_y + w_L * n_z;

    const RealType entropy_L = p_L / Kokkos::pow(rho_L, gamma);
    const RealType rho_b = Kokkos::pow(p_out / entropy_L, RealType(1.0) / gamma);
    const RealType c_b = Kokkos::sqrt(gamma * p_out / rho_b);

    const RealType j_plus = v_n_L + RealType(2.0) * inv_gm1 * c_L;
    const RealType v_n_b = j_plus - RealType(2.0) * inv_gm1 * c_b;

    const RealType u_b = u_L + (v_n_b - v_n_L) * n_x;
    const RealType v_b = v_L + (v_n_b - v_n_L) * n_y;
    const RealType w_b = w_L + (v_n_b - v_n_L) * n_z;

    const RealType rho_E_b = p_out * inv_gm1 + RealType(0.5) * rho_b * (u_b * u_b + v_b * v_b + w_b * w_b);

    euler_flux(rho_b, u_b, v_b, w_b, p_out, rho_E_b, n_x, n_y, n_z, F_rho, F_rho_u, F_rho_v, F_rho_w, F_rho_E);
    s_mag = Kokkos::abs(v_n_b) + c_b;
  }

  // ===========================================================================
  // 4. INVISCID WALL (3D TANGENTIAL CORRECTION)
  // ===========================================================================

  KOKKOS_INLINE_FUNCTION
  static void inviscid_wall_flux(RealType rho_L, RealType rho_u_L, RealType rho_v_L, RealType rho_w_L, RealType rho_E_L,
                                 RealType n_x, RealType n_y, RealType n_z,
                                 RealType& F_rho, RealType& F_rho_u, RealType& F_rho_v, RealType& F_rho_w, RealType& F_rho_E,
                                 RealType& s_mag)
  {
    constexpr RealType gm1 = Parameters<RealType>::gamma - 1.0;
    
    RealType u_L, v_L, w_L, p_L;
    conservative_to_primitive(rho_L, rho_u_L, rho_v_L, rho_w_L, rho_E_L, u_L, v_L, w_L, p_L);

    const RealType v_n_L = u_L * n_x + v_L * n_y + w_L * n_z;
    const RealType u_t = u_L - v_n_L * n_x;
    const RealType v_t = v_L - v_n_L * n_y;
    const RealType w_t = w_L - v_n_L * n_z;

    RealType p_b = gm1 * (rho_E_L - RealType(0.5) * rho_L * (u_t * u_t + v_t * v_t + w_t * w_t));
    p_b = Kokkos::fmax(p_b, RealType(1e-6));

    F_rho   = 0.0;
    F_rho_u = p_b * n_x;
    F_rho_v = p_b * n_y;
    F_rho_w = p_b * n_z;
    F_rho_E = 0.0;
    
    s_mag = speed_of_sound(rho_L, p_b);
  }

  // ===========================================================================
  // 5. ROE FLUX (3D SHEAR WAVES)
  // ===========================================================================

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

    const RealType rho_roe = sL * sR; // Use geometric mean for acoustic wave scaling
    const RealType a1 = 0.5 * (dp - rho_roe * c * dv_n) / (c * c);
    const RealType a2 = drho - dp / (c * c);
    const RealType a3_u = du - dv_n * n_x; 
    const RealType a3_v = dv - dv_n * n_y; 
    const RealType a3_w = dw - dv_n * n_z; 
    const RealType a4 = 0.5 * (dp + rho_roe * c * dv_n) / (c * c);

    const RealType eps = 0.1 * c;
    auto efix = [&](RealType lam) {
      return (Kokkos::abs(lam) < eps) ? (lam * lam + eps * eps) / (2.0 * eps) : Kokkos::abs(lam);
    };

    const RealType l1 = efix(v_n - c);
    const RealType l2 = efix(v_n);
    const RealType l4 = efix(v_n + c);

    const RealType d_rho = l1 * a1 + l2 * a2 + l4 * a4;
    const RealType d_rhou = l1 * a1 * (u - c * n_x) + l2 * (a2 * u + rho_roe * a3_u) + l4 * a4 * (u + c * n_x);
    const RealType d_rhov = l1 * a1 * (v - c * n_y) + l2 * (a2 * v + rho_roe * a3_v) + l4 * a4 * (v + c * n_y);
    const RealType d_rhow = l1 * a1 * (w - c * n_z) + l2 * (a2 * w + rho_roe * a3_w) + l4 * a4 * (w + c * n_z);
    const RealType d_rhoE = l1 * a1 * (H - c * v_n) + l2 * (a2 * 0.5 * q2 + rho_roe * (u * a3_u + v * a3_v + w * a3_w)) + l4 * a4 * (H + c * v_n);

    RealType FL_rho, FL_rhou, FL_rhov, FL_rhow, FL_rhoE;
    RealType FR_rho, FR_rhou, FR_rhov, FR_rhow, FR_rhoE;
    euler_flux(rho_L, u_L, v_L, w_L, p_L, rho_E_L, n_x, n_y, n_z, FL_rho, FL_rhou, FL_rhov, FL_rhow, FL_rhoE);
    euler_flux(rho_R, u_R, v_R, w_R, p_R, rho_E_R, n_x, n_y, n_z, FR_rho, FR_rhou, FR_rhov, FR_rhow, FR_rhoE);

    F_rho = 0.5 * (FL_rho + FR_rho - d_rho);
    F_rho_u = 0.5 * (FL_rhou + FR_rhou - d_rhou);
    F_rho_v = 0.5 * (FL_rhov + FR_rhov - d_rhov);
    F_rho_w = 0.5 * (FL_rhow + FR_rhow - d_rhow);
    F_rho_E = 0.5 * (FL_rhoE + FR_rhoE - d_rhoE);
    s_mag = Kokkos::abs(v_n) + c;
  }
};