#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <config.hpp>
#include <parameters.hpp>
#include <tensor.hpp>

/**
 * @brief Flux evaluation class.
 */
template<unsigned int dim, typename RealType>
class Flux
{
public:
  KOKKOS_INLINE_FUNCTION static RealType pressure(
    const RealType rho,
    const Tensor<1, dim, RealType>& v,
    const RealType rho_E)
  {
    ASSERT(rho > 0, "Density must be positive");
    ASSERT(rho_E > 0, "Total energy must be positive");

    return (Parameters<RealType>::gamma - RealType(1)) *
           (rho_E - RealType(0.5) * rho * v.norm_square());
  }

  KOKKOS_INLINE_FUNCTION static void conservative_to_primitive(
    const RealType rho,
    const Tensor<1, dim, RealType>& rho_v,
    const RealType rho_E,
    Tensor<1, dim, RealType>& v,
    RealType& p)
  {
    ASSERT(rho > 0, "Density must be positive");
    ASSERT(rho_E > 0, "Total energy must be positive");

    v = rho_v / rho;
    p = pressure(rho, v, rho_E);

    ASSERT(p > 0, "Pressure must be positive");
  }

  KOKKOS_INLINE_FUNCTION static RealType speed_of_sound(const RealType rho,
                                                        const RealType p)
  {
    using Kokkos::sqrt;

    ASSERT(rho > 0, "Density must be positive");
    ASSERT(p > 0, "Pressure must be positive");

    return sqrt(Parameters<RealType>::gamma * p / rho);
  }

  KOKKOS_INLINE_FUNCTION static RealType max_wavespeed(
    const RealType rho,
    const Tensor<1, dim, RealType>& v,
    const RealType p,
    const Tensor<1, dim, RealType>& n)
  {
    using Kokkos::abs;

    ASSERT(rho > 0, "Density must be positive");
    ASSERT(p > 0, "Pressure must be positive");
    ASSERT(abs(n.norm() - RealType(1)) <
             10.0 * std::numeric_limits<RealType>::epsilon(),
           "Normal must be a unit vector");

    return abs(dot(v, n)) + speed_of_sound(rho, p);
  }

  KOKKOS_INLINE_FUNCTION static void euler_flux(
    const RealType rho,
    const Tensor<1, dim, RealType>& v,
    const RealType p,
    const RealType rho_E,
    const Tensor<1, dim, RealType>& n,
    RealType& F_rho,
    Tensor<1, dim, RealType>& F_rho_v,
    RealType& F_rho_E)
  {
    using Kokkos::abs;

    ASSERT(rho > 0, "Density must be positive");
    ASSERT(p > 0, "Pressure must be positive");
    ASSERT(abs(n.norm() - RealType(1)) <
             10.0 * std::numeric_limits<RealType>::epsilon(),
           "Normal must be a unit vector");
    ASSERT(abs(p - pressure(rho, v, rho_E)) <
             10.0 * std::numeric_limits<RealType>::epsilon(),
           "Pressure state inconsistent");

    const RealType v_n = dot(v, n);
    F_rho = rho * v_n;
    F_rho_v = rho * v_n * v + p * n;
    F_rho_E = (rho_E + p) * v_n;
  }

  KOKKOS_INLINE_FUNCTION static void euler_flux(
    const RealType rho,
    const Tensor<1, dim, RealType>& v,
    const RealType p,
    const RealType rho_E,
    Tensor<1, dim, RealType>& F_rho,
    Tensor<2, dim, RealType>& F_rho_v,
    Tensor<1, dim, RealType>& F_rho_E)
  {
    using Kokkos::abs;

    ASSERT(rho > 0, "Density must be positive");
    ASSERT(p > 0, "Pressure must be positive");
    ASSERT(abs(p - pressure(rho, v, rho_E)) <
             10.0 * std::numeric_limits<RealType>::epsilon(),
           "Pressure state inconsistent");

    F_rho = rho * v;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        F_rho_v(i, j) = rho * v[i] * v[j] + (i == j ? p : RealType(0));
    F_rho_E = (rho_E + p) * v;
  }

  KOKKOS_INLINE_FUNCTION static void roe_flux(
    const RealType rho_L,
    const Tensor<1, dim, RealType>& rho_v_L,
    const RealType rho_E_L,
    const RealType rho_R,
    const Tensor<1, dim, RealType>& rho_v_R,
    const RealType rho_E_R,
    const Tensor<1, dim, RealType>& n,
    RealType& F_rho,
    Tensor<1, dim, RealType>& F_rho_v,
    RealType& F_rho_E,
    RealType& s_mag)
  {
    using Kokkos::abs;
    using Kokkos::sqrt;

    ASSERT(rho_L > 0, "Density must be positive");
    ASSERT(rho_E_L > 0, "Total energy must be positive");
    ASSERT(rho_R > 0, "Density must be positive");
    ASSERT(rho_E_R > 0, "Total energy must be positive");
    ASSERT(abs(n.norm() - RealType(1)) <
             10.0 * std::numeric_limits<RealType>::epsilon(),
           "Normal must be a unit vector");

    // Convert conservative state to primitive
    Tensor<1, dim, RealType> v_L, v_R;
    RealType p_L, p_R;
    conservative_to_primitive(rho_L, rho_v_L, rho_E_L, v_L, p_L);
    conservative_to_primitive(rho_R, rho_v_R, rho_E_R, v_R, p_R);

    // Grab enthalpies
    const RealType H_L = (rho_E_L + p_L) / rho_L;
    const RealType H_R = (rho_E_R + p_R) / rho_R;

    // Grab the Roe averages
    const RealType sqrt_rho_L = sqrt(rho_L);
    const RealType sqrt_rho_R = sqrt(rho_R);
    const RealType denom = RealType(1) / (sqrt_rho_L + sqrt_rho_R);

    const Tensor<1, dim, RealType> v_roe =
      (sqrt_rho_L * v_L + sqrt_rho_R * v_R) * denom;
    const RealType H_roe = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) * denom;

    const RealType c_roe_sq = (Parameters<RealType>::gamma - RealType(1)) *
                              (H_roe - RealType(0.5) * v_roe.norm_square());
    ASSERT(c_roe_sq > 0,
           "Roe-averaged speed of sound squared must be positive");
    const RealType c_roe = sqrt(c_roe_sq);

    const RealType v_n_roe = dot(v_roe, n);

    // Jumps
    const RealType jump_p = p_R - p_L;
    const Tensor<1, dim, RealType> jump_v = v_R - v_L;
    const RealType jump_rho = rho_R - rho_L;
    const RealType jump_v_n = dot(jump_v, n);
    const Tensor<1, dim, RealType> jump_v_t = jump_v - jump_v_n * n;

    // Wave strengths
    const RealType rho_roe = sqrt_rho_L * sqrt_rho_R;
    const RealType inv_c_roe_sq = RealType(1) / c_roe_sq;
    const RealType alpha_1 =
      RealType(0.5) * inv_c_roe_sq * (jump_p - rho_roe * c_roe * jump_v_n);
    const RealType alpha_2 = jump_rho - jump_p * inv_c_roe_sq;
    const Tensor<1, dim, RealType> alpha_3 = rho_roe * jump_v_t;
    const RealType alpha_4 =
      RealType(0.5) * inv_c_roe_sq * (jump_p + rho_roe * c_roe * jump_v_n);

    // Eigenvalues with entropy fix
    auto entropy_fix = [](RealType lambda, RealType eps) -> RealType {
      if (abs(lambda) < eps) {
        return (lambda * lambda + eps * eps) / (RealType(2) * eps);
      }
      return abs(lambda);
    };

    const RealType eps = RealType(0.1) * c_roe;
    const RealType lambda_1 = entropy_fix(v_n_roe - c_roe, eps);
    const RealType lambda_2 = entropy_fix(v_n_roe, eps);
    const RealType lambda_3 = lambda_2;
    const RealType lambda_4 = entropy_fix(v_n_roe + c_roe, eps);

    // Dissipation
    const RealType diss_rho =
      lambda_1 * alpha_1 + lambda_2 * alpha_2 + lambda_4 * alpha_4;
    const Tensor<1, dim, RealType> diss_rho_v =
      lambda_1 * alpha_1 * (v_roe - c_roe * n) + lambda_2 * alpha_2 * v_roe +
      lambda_3 * alpha_3 + lambda_4 * alpha_4 * (v_roe + c_roe * n);
    const RealType diss_rho_E =
      lambda_1 * alpha_1 * (H_roe - c_roe * v_n_roe) +
      lambda_2 * alpha_2 * RealType(0.5) * v_roe.norm_square() +
      lambda_3 * dot(alpha_3, v_roe) +
      lambda_4 * alpha_4 * (H_roe + c_roe * v_n_roe);

    // Fluxes
    RealType F_rho_L, F_rho_E_L, F_rho_R, F_rho_E_R;
    Tensor<1, dim, RealType> F_rho_v_L, F_rho_v_R;
    euler_flux(rho_L, v_L, p_L, rho_E_L, n, F_rho_L, F_rho_v_L, F_rho_E_L);
    euler_flux(rho_R, v_R, p_R, rho_E_R, n, F_rho_R, F_rho_v_R, F_rho_E_R);

    // Corrected flux
    F_rho = RealType(0.5) * (F_rho_L + F_rho_R - diss_rho);
    F_rho_v = RealType(0.5) * (F_rho_v_L + F_rho_v_R - diss_rho_v);
    F_rho_E = RealType(0.5) * (F_rho_E_L + F_rho_E_R - diss_rho_E);

    // Max wavespeed for CFL condition
    s_mag = abs(v_n_roe) + c_roe;
  }

  KOKKOS_INLINE_FUNCTION
  static void inviscid_wall_flux(const RealType rho_L,
                                 const Tensor<1, dim, RealType>& rho_v_L,
                                 const RealType rho_E_L,
                                 const Tensor<1, dim, RealType>& n,
                                 RealType& F_rho,
                                 Tensor<1, dim, RealType>& F_rho_v,
                                 RealType& F_rho_E,
                                 RealType& s_mag)
  {
    using Kokkos::abs;
    using Kokkos::sqrt;

    ASSERT(rho_L > 0, "Density must be positive");
    ASSERT(rho_E_L > 0, "Total energy must be positive");
    ASSERT(abs(n.norm() - RealType(1)) <
             10.0 * std::numeric_limits<RealType>::epsilon(),
           "Normal must be a unit vector");

    // Convert conservative state to primitive
    Tensor<1, dim, RealType> v_L;
    RealType p_L;
    conservative_to_primitive(rho_L, rho_v_L, rho_E_L, v_L, p_L);

    // Remove normal component
    const RealType v_n = dot(v_L, n);
    const Tensor<1, dim, RealType> v_t = v_L - v_n * n;

    // Wall pressure
    const RealType p_b = pressure(rho_L, v_t, rho_E_L);

    // Flux terms
    F_rho = RealType(0);
    F_rho_v = p_b * n;
    F_rho_E = RealType(0);

    // Wavespeed
    const RealType c_b = sqrt(Parameters<RealType>::gamma * p_b / rho_L);
    s_mag = c_b;
  }

  KOKKOS_INLINE_FUNCTION static void subsonic_inflow_flux(
    const RealType rho_L,
    const Tensor<1, dim, RealType>& rho_v_L,
    const RealType rho_E_L,
    const Tensor<1, dim, RealType>& n,
    RealType& F_rho,
    Tensor<1, dim, RealType>& F_rho_v,
    RealType& F_rho_E,
    RealType& s_mag)
  {
    using Kokkos::abs;
    using Kokkos::pow;
    using Kokkos::sqrt;

    constexpr RealType gamma = Parameters<RealType>::gamma;
    constexpr RealType gm1 = gamma - RealType(1);
    constexpr RealType inv_gm1 = RealType(1) / gm1;
    constexpr RealType p_0 = Parameters<RealType>::p_0;
    constexpr RealType rho_0 = Parameters<RealType>::rho_0;
    constexpr RealType RT_0 =
      Parameters<RealType>::p_0 / Parameters<RealType>::rho_0;

    ASSERT(rho_L > 0, "Density must be positive");
    ASSERT(rho_E_L > 0, "Total energy must be positive");
    ASSERT(abs(n.norm() - RealType(1)) <
             10.0 * std::numeric_limits<RealType>::epsilon(),
           "Normal must be a unit vector");

    // Convert conservative state to primitive
    Tensor<1, dim, RealType> v_L;
    RealType p_L;
    conservative_to_primitive(rho_L, rho_v_L, rho_E_L, v_L, p_L);

    // Interior speed of sound and normal velocity
    const RealType c_L = speed_of_sound(rho_L, p_L);
    const RealType v_n_L = dot(v_L, n);

    // Outgoing Riemann invariant
    const RealType j_plus = v_n_L + RealType(2) * inv_gm1 * c_L;

    // Inflow direction
    const Tensor<1, dim, RealType> n_in =
      Parameters<RealType>::template n_0<dim>();
    const RealType d_n = dot(n, n_in);

    // Boundary Mach number
    const RealType A =
      gamma * RT_0 * d_n * d_n - gm1 / RealType(2) * j_plus * j_plus;
    const RealType B = RealType(4) * gamma * RT_0 * d_n * inv_gm1;
    const RealType C =
      RealType(4) * gamma * RT_0 * inv_gm1 * inv_gm1 - j_plus * j_plus;

    const RealType disc = sqrt(B * B - RealType(4) * A * C);
    const RealType M_b1 = (-B + disc) / (RealType(2) * A);
    const RealType M_b2 = (-B - disc) / (RealType(2) * A);

    // Select physical root — take positive root if roots have opposite signs,
    // otherwise take root closest to zero
    RealType M_b;
    if ((M_b1 > 0) != (M_b2 > 0)) {
      M_b = (M_b1 > 0) ? M_b1 : M_b2;
    } else {
      M_b = (Kokkos::abs(M_b1) < Kokkos::abs(M_b2)) ? M_b1 : M_b2;
    }

    // Boundary speed of sound and isentropic state
    const RealType denom = RealType(1) + RealType(0.5) * gm1 * M_b * M_b;
    const RealType c_b = sqrt(gamma * RT_0 / denom);
    const RealType p_b = p_0 * pow(RealType(1) / denom, gamma * inv_gm1);
    const RealType rho_b = p_b / (RT_0 / denom);

    // Boundary velocity
    const RealType speed_b = M_b * c_b;
    const Tensor<1, dim, RealType> v_b = speed_b * n_in;

    // Boundary enthalpy
    const RealType rho_E_b =
      p_b * inv_gm1 + RealType(0.5) * rho_b * v_b.norm_square();
    const RealType H_b = (rho_E_b + p_b) / rho_b;

    // Compute flux
    const RealType v_n_b = dot(v_b, n);
    F_rho = rho_b * v_n_b;
    F_rho_v = rho_b * outer(v_b, v_b) * n + p_b * n;
    F_rho_E = rho_b * H_b * v_n_b;

    s_mag = abs(v_n_b) + c_b;
  }

  KOKKOS_INLINE_FUNCTION static void subsonic_outflow_flux(
    const RealType rho_L,
    const Tensor<1, dim, RealType>& rho_v_L,
    const RealType rho_E_L,
    const Tensor<1, dim, RealType>& n,
    RealType& F_rho,
    Tensor<1, dim, RealType>& F_rho_v,
    RealType& F_rho_E,
    RealType& s_mag)
  {
    using Kokkos::abs;
    using Kokkos::pow;
    using Kokkos::sqrt;

    constexpr RealType gamma = Parameters<RealType>::gamma;
    constexpr RealType gm1 = gamma - RealType(1);
    constexpr RealType inv_gm1 = RealType(1) / gm1;
    constexpr RealType p_out = Parameters<RealType>::p_out;

    ASSERT(rho_L > 0, "Density must be positive");
    ASSERT(rho_E_L > 0, "Total energy must be positive");
    ASSERT(abs(n.norm() - RealType(1)) <
             10.0 * std::numeric_limits<RealType>::epsilon(),
           "Normal must be a unit vector");

    // Convert conservative state to primitive
    Tensor<1, dim, RealType> v_L;
    RealType p_L;
    conservative_to_primitive(rho_L, rho_v_L, rho_E_L, v_L, p_L);

    // Interior speed of sound and normal velocity
    const RealType c_L = speed_of_sound(rho_L, p_L);
    const RealType v_n_L = dot(v_L, n);

    // Entropy
    const RealType entropy = p_L / pow(rho_L, gamma);

    // Boundary density
    const RealType rho_b = pow(p_out / entropy, RealType(1) / gamma);
    const RealType c_b = sqrt(gamma * p_out / rho_b);

    // Outgoing Riemann invariant
    const RealType j_plus = v_n_L + RealType(2) * inv_gm1 * c_L;
    const RealType v_n_b = j_plus - RealType(2) * inv_gm1 * c_b;

    // Reconstruct full velocity
    const Tensor<1, dim, RealType> v_b = v_L - v_n_L * n + v_n_b * n;

    // Boundary energy and enthalpy
    const RealType rho_E_b =
      p_out * inv_gm1 + RealType(0.5) * rho_b * v_b.norm_square();
    const RealType H_b = (rho_E_b + p_out) / rho_b;

    // Compute flux
    F_rho = rho_b * v_n_b;
    F_rho_v = rho_b * outer(v_b, v_b) * n + p_out * n;
    F_rho_E = rho_b * H_b * v_n_b;

    s_mag = abs(v_n_b) + c_b;
  }
};
