#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <config.hpp>
#include <parameters.hpp>

/**
 * @brief Class that contains all functions for flux evaluation.
 */
template<typename RealType>
class Flux
{
public:
  KOKKOS_INLINE_FUNCTION static RealType pressure(RealType rho,
                                                  RealType u,
                                                  RealType v,
                                                  RealType rho_E)
  {
    ASSERT(rho > 0, "Density must be positive");
    ASSERT(rho_E > 0, "Total energy must be positive");

    return (Parameters<RealType>::gamma - RealType(1.0)) *
           (rho_E - RealType(0.5) * rho * (u * u + v * v));
  }

  KOKKOS_INLINE_FUNCTION static void conservative_to_primitive(RealType rho,
                                                               RealType rho_u,
                                                               RealType rho_v,
                                                               RealType rho_E,
                                                               RealType& u,
                                                               RealType& v,
                                                               RealType& p)
  {
    ASSERT(rho > 0, "Density must be positive");
    ASSERT(rho_E > 0, "Total energy must be positive");

    u = rho_u / rho;
    v = rho_v / rho;
    p = pressure(rho, u, v, rho_E);

    ASSERT(p > 0, "Pressure must be positive");
  }

  KOKKOS_INLINE_FUNCTION
  static RealType speed_of_sound(RealType rho, RealType p)
  {
    ASSERT(rho > 0, "Density must be positive");
    ASSERT(p > 0, "Pressure must be positive");
    return Kokkos::sqrt(Parameters<RealType>::gamma * p / rho);
  }

  KOKKOS_INLINE_FUNCTION
  static RealType max_wavespeed(RealType rho,
                                RealType u,
                                RealType v,
                                RealType p,
                                RealType n_x,
                                RealType n_y)
  {
    ASSERT(rho > 0, "Density must be positive");
    ASSERT(p > 0, "Pressure must be positive");
    ASSERT(Kokkos::abs(n_x * n_x + n_y * n_y - 1.0) < 1e-10,
           "Normal must be a unit vector");
    return Kokkos::abs(u * n_x + v * n_y) + speed_of_sound(rho, p);
  }

  KOKKOS_INLINE_FUNCTION
  static void euler_flux(RealType rho,
                         RealType u,
                         RealType v,
                         RealType p,
                         RealType rho_E,
                         RealType n_x,
                         RealType n_y,
                         RealType& F_rho,
                         RealType& F_rho_u,
                         RealType& F_rho_v,
                         RealType& F_rho_E)
  {
    ASSERT(rho > 0, "Density must be positive");
    ASSERT(p > 0, "Pressure must be positive");
    ASSERT(rho_E > 0, "Total energy must be positive");
    ASSERT(Kokkos::abs(n_x * n_x + n_y * n_y - 1.0) < 1e-10,
           "Normal must be a unit vector");

    // Normal velocity
    const auto v_n = u * n_x + v * n_y;

    F_rho = rho * v_n;
    F_rho_u = rho * u * v_n + p * n_x;
    F_rho_v = rho * v * v_n + p * n_y;
    F_rho_E = (rho_E + p) * v_n;
  }

  KOKKOS_INLINE_FUNCTION
  static void roe_flux(RealType rho_L,
                       RealType rho_u_L,
                       RealType rho_v_L,
                       RealType rho_E_L,
                       RealType rho_R,
                       RealType rho_u_R,
                       RealType rho_v_R,
                       RealType rho_E_R,
                       RealType n_x,
                       RealType n_y,
                       RealType& F_rho,
                       RealType& F_rho_u,
                       RealType& F_rho_v,
                       RealType& F_rho_E,
                       RealType& s_mag)
  {
    ASSERT(rho_L > 0, "Density must be positive");
    ASSERT(rho_E_L > 0, "Total energy must be positive");
    ASSERT(rho_R > 0, "Density must be positive");
    ASSERT(rho_E_R > 0, "Total energy must be positive");
    ASSERT(Kokkos::abs(n_x * n_x + n_y * n_y - 1.0) < 1e-10,
           "Normal must be a unit vector");

    // Convert to primitive state
    RealType u_L, v_L, p_L;
    RealType u_R, v_R, p_R;
    conservative_to_primitive(rho_L, rho_u_L, rho_v_L, rho_E_L, u_L, v_L, p_L);
    conservative_to_primitive(rho_R, rho_u_R, rho_v_R, rho_E_R, u_R, v_R, p_R);

    const RealType H_L = (rho_E_L + p_L) / rho_L;
    const RealType H_R = (rho_E_R + p_R) / rho_R;

    // Fluxes
    RealType FL_rho, FL_rho_u, FL_rho_v, FL_rho_E;
    RealType FR_rho, FR_rho_u, FR_rho_v, FR_rho_E;
    euler_flux(rho_L,
               u_L,
               v_L,
               p_L,
               rho_E_L,
               n_x,
               n_y,
               FL_rho,
               FL_rho_u,
               FL_rho_v,
               FL_rho_E);
    euler_flux(rho_R,
               u_R,
               v_R,
               p_R,
               rho_E_R,
               n_x,
               n_y,
               FR_rho,
               FR_rho_u,
               FR_rho_v,
               FR_rho_E);

    // Jumps
    const RealType jump_p = p_R - p_L;
    const RealType jump_u = u_R - u_L;
    const RealType jump_v = v_R - v_L;
    const RealType jump_rho = rho_R - rho_L;
    const RealType jump_v_n = jump_u * n_x + jump_v * n_y;
    const RealType jump_v_t = jump_u * -n_y + jump_v * n_x;

    // Grab the Roe averages
    const RealType sqrt_rho_L = Kokkos::sqrt(rho_L);
    const RealType sqrt_rho_R = Kokkos::sqrt(rho_R);
    const RealType denom = RealType(1.0) / (sqrt_rho_L + sqrt_rho_R);

    const RealType u_roe = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) * denom;
    const RealType v_roe = (sqrt_rho_L * v_L + sqrt_rho_R * v_R) * denom;
    const RealType H_roe = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) * denom;

    const RealType c_roe_sq =
      (Parameters<RealType>::gamma - RealType(1.0)) *
      (H_roe - RealType(0.5) * (u_roe * u_roe + v_roe * v_roe));
    ASSERT(c_roe_sq > 0,
           "Roe-averaged speed of sound squared must be positive");
    const RealType c_roe = Kokkos::sqrt(c_roe_sq);

    const RealType v_n_roe = u_roe * n_x + v_roe * n_y;

    // Wave strengths
    const RealType rho_roe = Kokkos::sqrt(rho_L * rho_R);
    const RealType inv_c2 = RealType(1.0) / c_roe_sq;
    const RealType alpha_1 =
      RealType(0.5) * inv_c2 * (jump_p - rho_roe * c_roe * jump_v_n);
    const RealType alpha_2 = jump_rho - jump_p * inv_c2;
    const RealType alpha_3 = rho_roe * jump_v_t;
    const RealType alpha_4 =
      RealType(0.5) * inv_c2 * (jump_p + rho_roe * c_roe * jump_v_n);

    // Eigenvalues with entropy fix
    auto entropy_fix = [](RealType lambda, RealType eps) -> RealType {
      if (Kokkos::abs(lambda) < eps) {
        return (lambda * lambda + eps * eps) / (RealType(2.0) * eps);
      }
      return Kokkos::abs(lambda);
    };

    const RealType eps = RealType(0.1) * c_roe;
    const RealType lambda_1 = entropy_fix(v_n_roe - c_roe, eps);
    const RealType lambda_2 = entropy_fix(v_n_roe, eps);
    const RealType lambda_3 = lambda_2;
    const RealType lambda_4 = entropy_fix(v_n_roe + c_roe, eps);

    // Dissipation
    const RealType diss_rho =
      lambda_1 * alpha_1 + lambda_2 * alpha_2 + lambda_4 * alpha_4;

    const RealType diss_rho_u =
      lambda_1 * alpha_1 * (u_roe - c_roe * n_x) + lambda_2 * alpha_2 * u_roe +
      lambda_3 * alpha_3 * -n_y + lambda_4 * alpha_4 * (u_roe + c_roe * n_x);

    const RealType diss_rho_v =
      lambda_1 * alpha_1 * (v_roe - c_roe * n_y) + lambda_2 * alpha_2 * v_roe +
      lambda_3 * alpha_3 * n_x + lambda_4 * alpha_4 * (v_roe + c_roe * n_y);

    const RealType diss_rho_E =
      lambda_1 * alpha_1 * (H_roe - c_roe * v_n_roe) +
      lambda_2 * alpha_2 * RealType(0.5) * (u_roe * u_roe + v_roe * v_roe) +
      lambda_3 * alpha_3 * (-n_y * u_roe + n_x * v_roe) +
      lambda_4 * alpha_4 * (H_roe + c_roe * v_n_roe);

    // Corrected flux
    F_rho = RealType(0.5) * (FL_rho + FR_rho - diss_rho);
    F_rho_u = RealType(0.5) * (FL_rho_u + FR_rho_u - diss_rho_u);
    F_rho_v = RealType(0.5) * (FL_rho_v + FR_rho_v - diss_rho_v);
    F_rho_E = RealType(0.5) * (FL_rho_E + FR_rho_E - diss_rho_E);

    // Max wavespeed for CFL condition
    s_mag = Kokkos::abs(v_n_roe) + c_roe;
  }
};
