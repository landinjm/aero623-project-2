#pragma once

#include <flux.hpp>
#include <functional>
#include <libassert/assert.hpp>
#include <numeric>
#include <parameters.hpp>
#include <tensor.hpp>
#include <triangulation.hpp>
#include <utilities.hpp>

template<typename RealType>
inline RealType
unsteady_inlet_density(RealType t, RealType y)
{
  const auto y_stator = y + Parameters<RealType>::a_0 * t;

  const auto eta = y_stator / Parameters<RealType>::Delta_y -
                   std::floor(y_stator / Parameters<RealType>::Delta_y) - 0.5;

  const auto rho = Parameters<RealType>::rho_0 *
                   (1.0 - Parameters<RealType>::f_wake *
                            std::exp(-eta * eta /
                                     (2.0 * Parameters<RealType>::delta *
                                      Parameters<RealType>::delta)));

  return rho;
}

template<unsigned int dim, typename RealType>
inline Tensor<1, 2 + dim, RealType>
inviscid_wall_state(const Tensor<1, 2 + dim, RealType>& interior_state,
                    const Tensor<1, dim, RealType>& normal)
{
  static_assert(dim == 2, "Only 2D is supported");

  // Grab values
  const auto rho = interior_state[0];
  const auto rho_and_u = interior_state[1];
  const auto rho_and_v = interior_state[2];
  const auto rho_and_E = interior_state[3];

  const auto n_x = normal[0];
  const auto n_y = normal[1];

  // Compute the tangential boundary velocity
  const auto dot = rho_and_u * n_x + rho_and_v * n_y;
  const auto rho_and_u_b = rho_and_u - dot * n_x;
  const auto rho_and_v_b = rho_and_v - dot * n_y;

  return Tensor<1, 2 + dim, RealType>{
    rho, rho_and_u_b, rho_and_v_b, rho_and_E
  };
}

template<unsigned int dim, typename RealType>
inline std::pair<Tensor<1, 2 + dim, RealType>, RealType>
inviscid_wall(const Tensor<1, 2 + dim, RealType>& interior_state,
              const Tensor<1, dim, RealType>& normal)
{
  static_assert(dim == 2, "Only 2D is supported");

  // Grab the boundary state
  const auto boundary_state = inviscid_wall_state(interior_state, normal);

  // Construct boundary flux
  const auto boundary_flux = euler_flux(boundary_state, normal);

  // Compute the max wavespeed
  const auto smag = max_wavespeed(boundary_state, normal);

  return { boundary_flux, smag };
}

template<unsigned int dim, typename RealType>
inline Tensor<1, 2 + dim, RealType>
subsonic_inflow_state(const Tensor<1, 2 + dim, RealType>& interior_state,
                      const Tensor<1, dim, RealType>& normal)
{
  static_assert(dim == 2, "Only 2D is supported");

  // Grab values
  const auto rho = interior_state[0];
  const auto rho_and_u = interior_state[1];
  const auto rho_and_v = interior_state[2];
  const auto rho_and_E = interior_state[3];

  const auto n_x = normal[0];
  const auto n_y = normal[1];

  // Grab the velocity and normal velocity
  const auto u = rho_and_u / rho;
  const auto v = rho_and_v / rho;
  const auto velocity_normal = u * n_x + v * n_y;

  // Construct the Riemann invariant
  const auto j_plus = velocity_normal + 2.0 * speed_of_sound(interior_state) /
                                          (Parameters<RealType>::gamma - 1.0);

  // Grab the inflow direction
  const auto d_n =
    Parameters<RealType>::n_x_0 * n_x + Parameters<RealType>::n_y_0 * n_y;

  // Solve for the boundary mach number. Since this is a quadratic equation and
  // we only want real roots, we'll do the following
  const auto a =
    Parameters<RealType>::gamma * Parameters<RealType>::T_0_and_R * d_n * d_n -
    (Parameters<RealType>::gamma - 1.0) / 2.0 * j_plus * j_plus;
  const auto b = 4.0 * Parameters<RealType>::gamma *
                 Parameters<RealType>::T_0_and_R * d_n /
                 (Parameters<RealType>::gamma - 1.0);
  const auto c = 4.0 * Parameters<RealType>::gamma *
                   Parameters<RealType>::T_0_and_R /
                   (Parameters<RealType>::gamma - 1.0) /
                   (Parameters<RealType>::gamma - 1.0) -
                 j_plus * j_plus;

  const auto discriminant = b * b - 4.0 * a * c;
  const auto sqrt_discriminant =
    (discriminant >= 0.0) ? std::sqrt(discriminant) : 0.0;

  const auto mach_root_1 = (-b + sqrt_discriminant) / (2.0 * a);
  const auto mach_root_2 = (-b - sqrt_discriminant) / (2.0 * a);

  // Select the correct Mach number root. We to select a positive if one is
  // negative and the smaller positive if both are positive.
  auto M_b = (mach_root_1 >= 0 && mach_root_2 >= 0)
               ? std::min(mach_root_1, mach_root_2)
               : std::max(mach_root_1, mach_root_2);
  DEBUG_ASSERT(M_b >= 0.0,
               "With the following a = " + std::to_string(a) +
                 " , b = " + std::to_string(b) + " , c = " + std::to_string(c) +
                 " b^2-4ac = " + std::to_string(discriminant));

  // Compute the boundary speed of sound
  const auto denom =
    1.0 + 0.5 * (Parameters<RealType>::gamma - 1.0) * M_b * M_b;
  const auto c_b = std::sqrt(
    (Parameters<RealType>::gamma * Parameters<RealType>::T_0_and_R) / denom);

  // Compute the boundary pressure
  const auto p_b =
    Parameters<RealType>::p_0 *
    std::pow(1.0 / denom,
             Parameters<RealType>::gamma / (Parameters<RealType>::gamma - 1.0));
  DEBUG_ASSERT(p_b >= 0, "Pressure must be positive");

  // Compute the boundary density
  const auto rho_b = p_b * denom / Parameters<RealType>::T_0_and_R;
  DEBUG_ASSERT(rho_b > 0.0, "Density must be positive");

  // Compute the boundary velocity
  const auto u_b = M_b * c_b * Parameters<RealType>::n_x_0;
  const auto v_b = M_b * c_b * Parameters<RealType>::n_y_0;

  // Compute the boundary energy & enthalpy
  const auto energy_b = p_b / (Parameters<RealType>::gamma - 1.0) +
                        0.5 * rho_b * (u_b * u_b + v_b * v_b);

  return Tensor<1, 2 + dim, RealType>{
    rho_b, rho_b * u_b, rho_b * v_b, energy_b
  };
}

template<unsigned int dim, typename RealType>
inline std::pair<Tensor<1, 2 + dim, RealType>, RealType>
subsonic_inflow(const Tensor<1, 2 + dim, RealType>& interior_state,
                const Tensor<1, dim, RealType>& normal)
{
  static_assert(dim == 2, "Only 2D is supported");

  // Grab the boundary state
  const auto boundary_state = subsonic_inflow_state(interior_state, normal);

  // Construct boundary flux
  const auto boundary_flux = euler_flux(boundary_state, normal);

  // Compute the max wavespeed
  const auto smag = max_wavespeed(boundary_state, normal);

  return { boundary_flux, smag };
}

template<unsigned int dim, typename RealType>
inline Tensor<1, 2 + dim, RealType>
unsteady_subsonic_inflow_state(
  const Tensor<1, 2 + dim, RealType>& interior_state,
  const Tensor<1, dim, RealType>& normal,
  RealType y_face,
  RealType t)
{
  // Grab values
  const auto rho = interior_state[0];
  const auto rho_and_u = interior_state[1];
  const auto rho_and_v = interior_state[2];
  const auto rho_and_E = interior_state[3];

  const auto n_x = normal[0];
  const auto n_y = normal[1];

  // Grab the velocity and normal velocity
  const auto u = rho_and_u / rho;
  const auto v = rho_and_v / rho;
  const auto velocity_normal = u * n_x + v * n_y;

  // Compute the inlet stagnation density and stagnation temperature
  const auto rho_0 = unsteady_inlet_density(t, y_face);
  DEBUG_ASSERT(rho_0 > 0, "Density must be positive");
  const auto p_0 = rho_0 * Parameters<RealType>::a_0 *
                   Parameters<RealType>::a_0 / Parameters<RealType>::gamma;
  const auto T_0_and_R = p_0 / rho_0;

  // Construct the Riemann invariant
  const auto j_plus = velocity_normal + 2.0 * speed_of_sound(interior_state) /
                                          (Parameters<RealType>::gamma - 1.0);

  // Grab the inflow direction
  const auto d_n =
    Parameters<RealType>::n_x_0 * n_x + Parameters<RealType>::n_y_0 * n_y;

  // Solve for the boundary mach number. Since this is a quadratic equation and
  // we only want real roots, we'll do the following
  const auto a = Parameters<RealType>::gamma * T_0_and_R * d_n * d_n -
                 (Parameters<RealType>::gamma - 1.0) / 2.0 * j_plus * j_plus;
  const auto b = 4.0 * Parameters<RealType>::gamma * T_0_and_R * d_n /
                 (Parameters<RealType>::gamma - 1.0);
  const auto c = 4.0 * Parameters<RealType>::gamma * T_0_and_R /
                   (Parameters<RealType>::gamma - 1.0) /
                   (Parameters<RealType>::gamma - 1.0) -
                 j_plus * j_plus;

  const auto discriminant = b * b - 4.0 * a * c;
  const auto sqrt_discriminant =
    (discriminant >= 0.0) ? std::sqrt(discriminant) : 0.0;

  const auto mach_root_1 = (-b + sqrt_discriminant) / (2.0 * a);
  const auto mach_root_2 = (-b - sqrt_discriminant) / (2.0 * a);

  // Select the correct Mach number root. We to select a positive if one is
  // negative and the smaller positive if both are positive.
  auto M_b = (mach_root_1 >= 0 && mach_root_2 >= 0)
               ? std::min(mach_root_1, mach_root_2)
               : std::max(mach_root_1, mach_root_2);
  DEBUG_ASSERT(M_b >= 0.0,
               "With the following a = " + std::to_string(a) +
                 " , b = " + std::to_string(b) + " , c = " + std::to_string(c) +
                 " b^2-4ac = " + std::to_string(discriminant));

  // Compute the boundary speed of sound
  const auto denom =
    1.0 + 0.5 * (Parameters<RealType>::gamma - 1.0) * M_b * M_b;
  const auto c_b = std::sqrt((Parameters<RealType>::gamma * T_0_and_R) / denom);

  // Compute the boundary pressure
  const auto p_b = p_0 * std::pow(1.0 / denom,
                                  Parameters<RealType>::gamma /
                                    (Parameters<RealType>::gamma - 1.0));
  DEBUG_ASSERT(p_b >= 0, "Pressure must be positive");

  // Compute the boundary density
  const auto rho_b = p_b * denom / T_0_and_R;
  DEBUG_ASSERT(rho_b > 0.0, "Density must be positive");

  // Compute the boundary velocity
  const auto u_b = M_b * c_b * Parameters<RealType>::n_x_0;
  const auto v_b = M_b * c_b * Parameters<RealType>::n_y_0;

  // Compute the boundary energy & enthalpy
  const auto energy_b = p_b / (Parameters<RealType>::gamma - 1.0) +
                        0.5 * rho_b * (u_b * u_b + v_b * v_b);

  return Tensor<1, 2 + dim, RealType>{
    rho_b, rho_b * u_b, rho_b * v_b, energy_b
  };
}

template<unsigned int dim, typename RealType>
inline std::pair<Tensor<1, 2 + dim, RealType>, RealType>
unsteady_subsonic_inflow(const Tensor<1, 2 + dim, RealType>& interior_state,
                         const Tensor<1, dim, RealType>& normal,
                         RealType y_face,
                         RealType t)
{
  static_assert(dim == 2, "Only 2D is supported");

  // Grab the boundary state
  const auto boundary_state =
    unsteady_subsonic_inflow_state(interior_state, normal, y_face, t);

  // Construct boundary flux
  const auto boundary_flux = euler_flux(boundary_state, normal);

  // Compute the max wavespeed
  const auto smag = max_wavespeed(boundary_state, normal);

  return { boundary_flux, smag };
}

template<unsigned int dim, typename RealType>
inline Tensor<1, 2 + dim, RealType>
subsonic_outflow_state(const Tensor<1, 2 + dim, RealType>& interior_state,
                       const Tensor<1, dim, RealType>& normal)
{
  // Grab values
  const auto rho = interior_state[0];
  const auto rho_and_u = interior_state[1];
  const auto rho_and_v = interior_state[2];
  const auto rho_and_E = interior_state[3];

  const auto n_x = normal[0];
  const auto n_y = normal[1];

  // Grab the velocity and normal velocity
  const auto u = rho_and_u / rho;
  const auto v = rho_and_v / rho;
  const auto velocity_normal = u * n_x + v * n_y;

  // Construct the Riemann invariant
  const auto j_plus = velocity_normal + 2.0 * speed_of_sound(interior_state) /
                                          (Parameters<RealType>::gamma - 1.0);

  // Compute the entropy
  const auto entropy =
    pressure(interior_state) / std::pow(rho, Parameters<RealType>::gamma);

  // Compute the boundary density
  const auto rho_b = std::pow(Parameters<RealType>::p_out / entropy,
                              1.0 / Parameters<RealType>::gamma);
  DEBUG_ASSERT(rho_b > 0.0, "Density must be positive");

  // Compute the boundary speed of sound
  const auto c_b = std::sqrt(Parameters<RealType>::gamma *
                             Parameters<RealType>::p_out / rho_b);

  // Compute the boundary normal velocity magnitude
  const auto u_n_b = j_plus - 2.0 * c_b / (Parameters<RealType>::gamma - 1.0);

  // Compute the boundary outflow velocity
  const auto u_b = u - velocity_normal * n_x + u_n_b * n_x;
  const auto v_b = v - velocity_normal * n_y + u_n_b * n_y;

  // Compute the boundary energy & enthalpy
  const auto energy_b =
    Parameters<RealType>::p_out / (Parameters<RealType>::gamma - 1.0) +
    0.5 * rho_b * (u_b * u_b + v_b * v_b);

  return Tensor<1, 2 + dim, RealType>{
    rho_b, rho_b * u_b, rho_b * v_b, energy_b
  };
}

template<unsigned int dim, typename RealType>
inline std::pair<Tensor<1, 2 + dim, RealType>, RealType>
subsonic_outflow(const Tensor<1, 2 + dim, RealType>& interior_state,
                 const Tensor<1, dim, RealType>& normal)
{
  static_assert(dim == 2, "Only 2D is supported");

  // Grab the boundary state
  const auto boundary_state = subsonic_outflow_state(interior_state, normal);

  // Construct boundary flux
  const auto boundary_flux = euler_flux(boundary_state, normal);

  // Compute the max wavespeed
  const auto smag = max_wavespeed(boundary_state, normal);

  return { boundary_flux, smag };
}

template<unsigned int dim, unsigned int degree, typename RealType>
class Solver
{
public:
  /**
   * @brief Flux function.
   */
  using FluxFunction =
    std::function<std::pair<Tensor<1, 4, RealType>, RealType>(
      const Tensor<1, 4, RealType>&,
      const Tensor<1, 4, RealType>&,
      const Tensor<1, 2, RealType>&)>;

  /**
   * @brief Boundary flux function.
   */
  using BoundaryFluxFunction = std::pair<Tensor<1, 4, RealType>, RealType> (*)(
    const Tensor<1, 4, RealType>&,
    const Tensor<1, 2, RealType>&);

  Solver() = default;

  enum class TimeIntegration
  {
    LocalTimestepping,
    RK3
  };

  struct SolverConfig
  {
    TimeIntegration time_integration = TimeIntegration::LocalTimestepping;
    bool is_freestream = false;
    bool is_unsteady = false;
    bool use_limiter = false;
    RealType time = 0;
    const FluxFunction* flux_func = &flux_roe;
  };

  /**
   * @brief Set the initial condition according the inflow condition and a
   * given mach number.
   */
  void set_initial_state(ElementData<dim, degree, RealType>& element_scratch,
                         RealType M = 0.1) const
  {
    // Compute the speed of sound
    const auto denom = 1.0 + 0.5 * (Parameters<RealType>::gamma - 1.0) * M * M;
    const auto c = std::sqrt(
      (Parameters<RealType>::gamma * Parameters<RealType>::T_0_and_R) / denom);

    // Compute the pressure
    const auto p = Parameters<RealType>::p_0 *
                   std::pow(1.0 / denom,
                            Parameters<RealType>::gamma /
                              (Parameters<RealType>::gamma - 1.0));
    DEBUG_ASSERT(p >= 0, "Pressure must be positive");

    // Compute the density
    const auto rho = p * denom / Parameters<RealType>::T_0_and_R;
    DEBUG_ASSERT(rho > 0.0, "Density must be positive");

    // Compute the velocity
    const auto u = M * c * Parameters<RealType>::n_x_0;
    const auto v = M * c * Parameters<RealType>::n_y_0;

    // Compute the energy
    const auto E =
      p / (Parameters<RealType>::gamma - 1.0) + 0.5 * rho * (u * u + v * v);

    // Set the state everywhere
    set(element_scratch.density, rho);
    set(element_scratch.momentum_x, rho * u);
    set(element_scratch.momentum_y, rho * v);
    set(element_scratch.energy, E);
  }

  /**
   * @brief Get the initial condition according the inflow condition and a
   * given mach number.
   */
  Tensor<1, 4, RealType> get_initial_state(RealType M = 0.1) const
  {
    // Compute the speed of sound
    const auto denom = 1.0 + 0.5 * (Parameters<RealType>::gamma - 1.0) * M * M;
    const auto c = std::sqrt(
      (Parameters<RealType>::gamma * Parameters<RealType>::T_0_and_R) / denom);

    // Compute the pressure
    const auto p = Parameters<RealType>::p_0 *
                   std::pow(1.0 / denom,
                            Parameters<RealType>::gamma /
                              (Parameters<RealType>::gamma - 1.0));
    DEBUG_ASSERT(p >= 0, "Pressure must be positive");

    // Compute the density
    const auto rho = p * denom / Parameters<RealType>::T_0_and_R;
    DEBUG_ASSERT(rho > 0.0, "Density must be positive");

    // Compute the velocity
    const auto u = M * c * Parameters<RealType>::n_x_0;
    const auto v = M * c * Parameters<RealType>::n_y_0;

    // Compute the energy
    const auto E =
      p / (Parameters<RealType>::gamma - 1.0) + 0.5 * rho * (u * u + v * v);

    return Tensor<1, 2 + dim, RealType>{ rho, rho * u, rho * v, E };
  }

  void compute_residual(
    const MeshData& mesh_data,
    const InteriorFaceData<dim, RealType>& interior_face_scratch,
    const BoundaryFaceData<dim, RealType>& boundary_face_scratch,
    const PeriodicFaceData<dim, RealType>& periodic_face_scratch,
    ElementData<dim, degree, RealType>& element_scratch,
    const SolverConfig& cfg) const
  {
    zero_values(element_scratch);

    if constexpr (degree == 2) {
      gradient_prep(interior_face_scratch,
                    boundary_face_scratch,
                    periodic_face_scratch,
                    element_scratch,
                    cfg.is_freestream,
                    cfg.is_unsteady,
                    cfg.time);
      if (cfg.use_limiter) {
        apply_barth_jespersen_limiter(mesh_data,
                                      interior_face_scratch,
                                      boundary_face_scratch,
                                      periodic_face_scratch,
                                      element_scratch,
                                      cfg.is_freestream);
      }
    }

    interior_face_residual(
      interior_face_scratch, element_scratch, *cfg.flux_func);
    periodic_face_residual(
      periodic_face_scratch, element_scratch, *cfg.flux_func);
    boundary_face_residual(boundary_face_scratch,
                           element_scratch,
                           *cfg.flux_func,
                           cfg.is_freestream,
                           cfg.is_unsteady,
                           cfg.time);
    optimal_time(element_scratch);
  };

  RealType compute_update(
    const MeshData& mesh_data,
    ElementData<dim, degree, RealType>& element_scratch,
    const InteriorFaceData<dim, RealType>& interior_face_scratch,
    const BoundaryFaceData<dim, RealType>& boundary_face_scratch,
    const PeriodicFaceData<dim, RealType>& periodic_face_scratch,
    const SolverConfig& cfg) const
  {
    if (cfg.time_integration == TimeIntegration::LocalTimestepping) {
      return compute_update_local(mesh_data,
                                  element_scratch,
                                  interior_face_scratch,
                                  boundary_face_scratch,
                                  periodic_face_scratch,
                                  cfg);
    } else {
      return compute_update_rk3(mesh_data,
                                element_scratch,
                                interior_face_scratch,
                                boundary_face_scratch,
                                periodic_face_scratch,
                                cfg);
    }
  }

  RealType compute_update_local(
    const MeshData& mesh_data,
    ElementData<dim, degree, RealType>& element_scratch,
    const InteriorFaceData<dim, RealType>& interior_face_scratch,
    const BoundaryFaceData<dim, RealType>& boundary_face_scratch,
    const PeriodicFaceData<dim, RealType>& periodic_face_scratch,
    const SolverConfig& cfg) const
  {
    compute_residual(mesh_data,
                     interior_face_scratch,
                     boundary_face_scratch,
                     periodic_face_scratch,
                     element_scratch,
                     cfg);

    for (unsigned int i = 0; i < element_scratch.size(); ++i) {
      const auto dt_inv_area =
        element_scratch.optimal_timestep[i] * element_scratch.inv_area[i];
      element_scratch.density[i] -=
        element_scratch.residual_density[i] * dt_inv_area;
      element_scratch.momentum_x[i] -=
        element_scratch.residual_momentum_x[i] * dt_inv_area;
      element_scratch.momentum_y[i] -=
        element_scratch.residual_momentum_y[i] * dt_inv_area;
      element_scratch.energy[i] -=
        element_scratch.residual_energy[i] * dt_inv_area;
    }

    return min(element_scratch.optimal_timestep);
  }

  RealType compute_update_rk3(
    const MeshData& mesh_data,
    ElementData<dim, degree, RealType>& element_scratch,
    const InteriorFaceData<dim, RealType>& interior_face_scratch,
    const BoundaryFaceData<dim, RealType>& boundary_face_scratch,
    const PeriodicFaceData<dim, RealType>& periodic_face_scratch,
    const SolverConfig& cfg) const
  {
    // Stage 1
    compute_residual(mesh_data,
                     interior_face_scratch,
                     boundary_face_scratch,
                     periodic_face_scratch,
                     element_scratch,
                     cfg);

    const auto dt = min(element_scratch.optimal_timestep);
    auto tmp = element_scratch;

    for (unsigned int i = 0; i < tmp.size(); ++i) {
      const auto dt_inv_area = dt * tmp.inv_area[i];
      tmp.density[i] -= tmp.residual_density[i] * dt_inv_area;
      tmp.momentum_x[i] -= tmp.residual_momentum_x[i] * dt_inv_area;
      tmp.momentum_y[i] -= tmp.residual_momentum_y[i] * dt_inv_area;
      tmp.energy[i] -= tmp.residual_energy[i] * dt_inv_area;
    }

    // Stage 2
    compute_residual(mesh_data,
                     interior_face_scratch,
                     boundary_face_scratch,
                     periodic_face_scratch,
                     tmp,
                     cfg);

    auto tmp_2 = element_scratch;

    for (unsigned int i = 0; i < element_scratch.size(); ++i) {
      const auto dt_inv_area = dt * tmp.inv_area[i];
      tmp_2.density[i] =
        0.75 * element_scratch.density[i] +
        0.25 * (tmp.density[i] - tmp.residual_density[i] * dt_inv_area);
      tmp_2.momentum_x[i] =
        0.75 * element_scratch.momentum_x[i] +
        0.25 * (tmp.momentum_x[i] - tmp.residual_momentum_x[i] * dt_inv_area);
      tmp_2.momentum_y[i] =
        0.75 * element_scratch.momentum_y[i] +
        0.25 * (tmp.momentum_y[i] - tmp.residual_momentum_y[i] * dt_inv_area);
      tmp_2.energy[i] =
        0.75 * element_scratch.energy[i] +
        0.25 * (tmp.energy[i] - tmp.residual_energy[i] * dt_inv_area);
    }

    // Stage 3
    compute_residual(mesh_data,
                     interior_face_scratch,
                     boundary_face_scratch,
                     periodic_face_scratch,
                     tmp_2,
                     cfg);

    for (unsigned int i = 0; i < element_scratch.size(); ++i) {
      const auto dt_inv_area = 0.5 * dt * tmp_2.inv_area[i];
      element_scratch.density[i] =
        (1.0 / 3.0) * element_scratch.density[i] +
        (2.0 / 3.0) *
          (tmp_2.density[i] - tmp_2.residual_density[i] * dt_inv_area);
      element_scratch.momentum_x[i] =
        (1.0 / 3.0) * element_scratch.momentum_x[i] +
        (2.0 / 3.0) *
          (tmp_2.momentum_x[i] - tmp_2.residual_momentum_x[i] * dt_inv_area);
      element_scratch.momentum_y[i] =
        (1.0 / 3.0) * element_scratch.momentum_y[i] +
        (2.0 / 3.0) *
          (tmp_2.momentum_y[i] - tmp_2.residual_momentum_y[i] * dt_inv_area);
      element_scratch.energy[i] =
        (1.0 / 3.0) * element_scratch.energy[i] +
        (2.0 / 3.0) *
          (tmp_2.energy[i] - tmp_2.residual_energy[i] * dt_inv_area);
    }

    return dt;
  }

  /**
   * @brief Construct state from element index.
   */
  inline Tensor<1, 4, RealType> get_state(
    const ElementData<dim, degree, RealType>& element_scratch,
    unsigned int element_index) const
  {
    return Tensor<1, 4, RealType>{ element_scratch.density[element_index],
                                   element_scratch.momentum_x[element_index],
                                   element_scratch.momentum_y[element_index],
                                   element_scratch.energy[element_index] };
  }

  /**
   * @brief Compute the state value at some boundary face.
   */
  inline Tensor<1, 4, RealType> get_face_state(
    const ElementData<dim, degree, RealType>& element_scratch,
    unsigned int element_index,
    RealType x,
    RealType y) const
  {
    // Grab the difference in position between the face midpoint and the cell
    // centroid.
    const auto dx = x - element_scratch.centroid_x[element_index];
    const auto dy = y - element_scratch.centroid_y[element_index];

    // Grab the state in the center
    auto state = get_state(element_scratch, element_index);

    // Add to the state based on the gradient
    if constexpr (degree == 2) {
      state[0] += element_scratch.grad_x_density[element_index] * dx +
                  element_scratch.grad_y_density[element_index] * dy;
      state[1] += element_scratch.grad_x_momentum_x[element_index] * dx +
                  element_scratch.grad_y_momentum_x[element_index] * dy;
      state[2] += element_scratch.grad_x_momentum_y[element_index] * dx +
                  element_scratch.grad_y_momentum_y[element_index] * dy;
      state[3] += element_scratch.grad_x_energy[element_index] * dx +
                  element_scratch.grad_y_energy[element_index] * dy;
    }

    return state;
  }

  /**
   * @brief Zero out residuals and optimal timestep.
   */
  void zero_values(ElementData<dim, degree, RealType>& element_scratch) const
  {
    zero(element_scratch.residual_density);
    zero(element_scratch.residual_momentum_x);
    zero(element_scratch.residual_momentum_y);
    zero(element_scratch.residual_energy);
    zero(element_scratch.optimal_timestep);

    if constexpr (degree == 2) {
      zero(element_scratch.grad_x_density);
      zero(element_scratch.grad_y_density);
      zero(element_scratch.grad_x_momentum_x);
      zero(element_scratch.grad_y_momentum_x);
      zero(element_scratch.grad_x_momentum_y);
      zero(element_scratch.grad_y_momentum_y);
      zero(element_scratch.grad_x_energy);
      zero(element_scratch.grad_y_energy);
    }
  }

  /**
   * @brief Compute the optimal timestep at each element based on CFL.
   */
  void optimal_time(ElementData<dim, degree, RealType>& element_scratch) const
  {
    for (unsigned int i = 0; i < element_scratch.size(); ++i) {
      element_scratch.optimal_timestep[i] =
        Parameters<RealType>::cfl_max * RealType(2.0) *
        element_scratch.area[i] / element_scratch.optimal_timestep[i];
      DEBUG_ASSERT(element_scratch.optimal_timestep[i] >= 0.0);
    }
  }

  Tensor<1, 4, RealType> get_boundary_state(
    const Tensor<1, 4, RealType>& interior_state,
    const Tensor<1, 2, RealType>& n,
    unsigned int boundary_id,
    bool is_freestream,
    const Tensor<1, 4, RealType>& freestream_state) const
  {
    if (is_freestream) {
      return freestream_state;
    }
    switch (boundary_id) {
      case 4:
        return subsonic_inflow_state(interior_state, n);
      case 5:
        return subsonic_outflow_state(interior_state, n);
      case 6:
      case 7:
        return inviscid_wall_state(interior_state, n);
      default:
        DEBUG_ASSERT(false, "Unknown boundary");
        return {};
    }
  }

  /**
   * @brief Compute the residual over the interior faces.
   */
  void interior_face_residual(
    const InteriorFaceData<dim, RealType>& interior_face_scratch,
    ElementData<dim, degree, RealType>& element_scratch,
    const FluxFunction& flux_func) const
  {
    for (unsigned int i = 0; i < interior_face_scratch.size(); ++i) {
      // Construct the state vectors and normals on the fly
      const auto e_l = interior_face_scratch.elem_l[i];
      const auto e_r = interior_face_scratch.elem_r[i];
      const auto n_x = interior_face_scratch.normal_x[i];
      const auto n_y = interior_face_scratch.normal_y[i];
      const auto area = interior_face_scratch.face_area[i];

      Tensor<1, 4, RealType> u_l;
      Tensor<1, 4, RealType> u_r;
      if constexpr (degree == 2) {
        const auto f_x = interior_face_scratch.centroid_x[i];
        const auto f_y = interior_face_scratch.centroid_y[i];

        u_l = get_face_state(element_scratch, e_l, f_x, f_y);
        u_r = get_face_state(element_scratch, e_r, f_x, f_y);
      } else {
        u_l = get_state(element_scratch, e_l);
        u_r = get_state(element_scratch, e_r);
      }

      const Tensor<1, 2, RealType> n = { n_x, n_y };

      const auto result = flux_func(u_l, u_r, n);
      const auto flux = result.first;
      const auto max_wavespeed = result.second;
      DEBUG_ASSERT(max_wavespeed > 0);

      // Add the residual to the elements
      element_scratch.residual_density[e_l] += flux[0] * area;
      element_scratch.residual_momentum_x[e_l] += flux[1] * area;
      element_scratch.residual_momentum_y[e_l] += flux[2] * area;
      element_scratch.residual_energy[e_l] += flux[3] * area;

      element_scratch.residual_density[e_r] -= flux[0] * area;
      element_scratch.residual_momentum_x[e_r] -= flux[1] * area;
      element_scratch.residual_momentum_y[e_r] -= flux[2] * area;
      element_scratch.residual_energy[e_r] -= flux[3] * area;

      // Add the edge-weighted wave speed to the optimal timestep. We'll
      // compute the actual timestep later.
      element_scratch.optimal_timestep[e_r] += max_wavespeed * area;
      element_scratch.optimal_timestep[e_l] += max_wavespeed * area;
    }
  }

  /**
   * @brief Compute the residual over the periodic faces.
   */
  void periodic_face_residual(
    const PeriodicFaceData<dim, RealType>& periodic_face_scratch,
    ElementData<dim, degree, RealType>& element_scratch,
    const FluxFunction& flux_func) const
  {
    for (unsigned int i = 0; i < periodic_face_scratch.size(); ++i) {
      // Construct the state vectors and normals on the fly
      const auto e_l = periodic_face_scratch.elem_l[i];
      const auto e_r = periodic_face_scratch.elem_r[i];
      const auto n_x = periodic_face_scratch.normal_x[i];
      const auto n_y = periodic_face_scratch.normal_y[i];
      const auto area = periodic_face_scratch.face_area[i];

      Tensor<1, 4, RealType> u_l;
      Tensor<1, 4, RealType> u_r;
      if constexpr (degree == 2) {
        // Make sure to adjust for the translation of periodicity
        const auto f_x = periodic_face_scratch.centroid_x[i];
        const auto f_y = periodic_face_scratch.centroid_y[i];
        const auto trans_x = periodic_face_scratch.translation_x[i];
        const auto trans_y = periodic_face_scratch.translation_y[i];

        u_l = get_face_state(element_scratch, e_l, f_x, f_y);
        u_r =
          get_face_state(element_scratch, e_r, f_x + trans_x, f_y + trans_y);
      } else {
        u_l = get_state(element_scratch, e_l);
        u_r = get_state(element_scratch, e_r);
      }

      const Tensor<1, 2, RealType> n = { n_x, n_y };

      const auto result = flux_func(u_l, u_r, n);
      const auto flux = result.first;
      const auto max_wavespeed = result.second;
      DEBUG_ASSERT(max_wavespeed > 0);

      // Add the residual to the elements
      element_scratch.residual_density[e_l] += flux[0] * area;
      element_scratch.residual_momentum_x[e_l] += flux[1] * area;
      element_scratch.residual_momentum_y[e_l] += flux[2] * area;
      element_scratch.residual_energy[e_l] += flux[3] * area;

      element_scratch.residual_density[e_r] -= flux[0] * area;
      element_scratch.residual_momentum_x[e_r] -= flux[1] * area;
      element_scratch.residual_momentum_y[e_r] -= flux[2] * area;
      element_scratch.residual_energy[e_r] -= flux[3] * area;

      // Add the edge-weighted wave speed to the optimal timestep. We'll
      // compute the actual timestep later.
      element_scratch.optimal_timestep[e_r] += max_wavespeed * area;
      element_scratch.optimal_timestep[e_l] += max_wavespeed * area;
    }
  }

  /**
   * @brief Compute the residual over the boundary faces.
   */
  void boundary_face_residual(
    const BoundaryFaceData<dim, RealType>& boundary_face_scratch,
    ElementData<dim, degree, RealType>& element_scratch,
    const FluxFunction& flux_func,
    bool is_freestream,
    bool is_unsteady = false,
    RealType time = 0) const
  {
    for (unsigned int i = 0; i < boundary_face_scratch.size(); ++i) {
      // Construct the state vectors and normals on the fly
      const auto e = boundary_face_scratch.elem[i];
      const auto n_x = boundary_face_scratch.normal_x[i];
      const auto n_y = boundary_face_scratch.normal_y[i];
      const auto area = boundary_face_scratch.face_area[i];
      const auto boundary_id = boundary_face_scratch.boundary_id[i];

      Tensor<1, 4, RealType> u;
      if constexpr (degree == 2) {
        const auto f_x = boundary_face_scratch.centroid_x[i];
        const auto f_y = boundary_face_scratch.centroid_y[i];

        u = get_face_state(element_scratch, e, f_x, f_y);

        if (is_freestream) {
          auto zero_grad_diff = u - get_state(element_scratch, e);
          DEBUG_ASSERT(std::abs(zero_grad_diff[0]) < 1e-8);
          DEBUG_ASSERT(std::abs(zero_grad_diff[1]) < 1e-8);
          DEBUG_ASSERT(std::abs(zero_grad_diff[2]) < 1e-8);
          DEBUG_ASSERT(std::abs(zero_grad_diff[3]) < 1e-8);
        }
      } else {
        u = get_state(element_scratch, e);
      }

      const Tensor<1, 2, RealType> n = { n_x, n_y };

      // Check that we don't accidentally have periodic boundaries
      DEBUG_ASSERT(boundary_id != 0 && boundary_id != 1 && boundary_id != 2 &&
                   boundary_id != 3);

      Tensor<1, 4, RealType> flux;
      RealType max_wavespeed;

      if (is_freestream) {
        const auto result = flux_func(u, u, n);
        flux = result.first;
        max_wavespeed = result.second;
      } else {
        BoundaryFluxFunction boundary_flux_func = nullptr;
        switch (boundary_id) {
            // InletSide
          case 4:
            boundary_flux_func = &subsonic_inflow<dim, RealType>;
            break;
            // OutletSide
          case 5:
            boundary_flux_func = &subsonic_outflow<dim, RealType>;
            break;
            // BladeTop
          case 6:
            // BladeBottom
          case 7:
            boundary_flux_func = &inviscid_wall<dim, RealType>;
            break;
          default:
            DEBUG_ASSERT(false, "How did we get here? Probably hardcoding");
        }
        DEBUG_ASSERT(boundary_flux_func != nullptr);

        auto result = boundary_flux_func(u, n);
        flux = result.first;
        max_wavespeed = result.second;

        if (is_unsteady && boundary_id == 4) {
          result = unsteady_subsonic_inflow(
            u, n, time, boundary_face_scratch.centroid_y[i]);
          flux = result.first;
          max_wavespeed = result.second;
        }
      }

      DEBUG_ASSERT(max_wavespeed > 0);

      // Add the residual to the elements
      element_scratch.residual_density[e] += flux[0] * area;
      element_scratch.residual_momentum_x[e] += flux[1] * area;
      element_scratch.residual_momentum_y[e] += flux[2] * area;
      element_scratch.residual_energy[e] += flux[3] * area;

      // Add the edge-weighted wave speed to the optimal timestep. We'll
      // compute the actual timestep later.
      element_scratch.optimal_timestep[e] += max_wavespeed * area;
    }
  }

  /**
   * @brief Compute the gradient
   */
  void gradient_prep(
    const InteriorFaceData<dim, RealType>& interior_face_scratch,
    const BoundaryFaceData<dim, RealType>& boundary_face_scratch,
    const PeriodicFaceData<dim, RealType>& periodic_face_scratch,
    ElementData<dim, degree, RealType>& element_scratch,
    bool is_freestream,
    bool is_unsteady,
    RealType time = 0) const
  {
    const auto n_elements = element_scratch.size();

    // Loop over interior faces
    for (unsigned int i = 0; i < interior_face_scratch.size(); ++i) {
      const auto e_l = interior_face_scratch.elem_l[i];
      const auto e_r = interior_face_scratch.elem_r[i];

      const auto dx =
        element_scratch.centroid_x[e_r] - element_scratch.centroid_x[e_l];
      const auto dy =
        element_scratch.centroid_y[e_r] - element_scratch.centroid_y[e_l];

      const auto u_l = get_state(element_scratch, e_l);
      const auto u_r = get_state(element_scratch, e_r);
      const auto du = u_r - u_l;

      element_scratch.A11[e_l] += dx * dx;
      element_scratch.A12[e_l] += dx * dy;
      element_scratch.A22[e_l] += dy * dy;

      element_scratch.A11[e_r] += dx * dx;
      element_scratch.A12[e_r] += dx * dy;
      element_scratch.A22[e_r] += dy * dy;

      element_scratch.b1_density[e_l] += dx * du[0];
      element_scratch.b2_density[e_l] += dy * du[0];
      element_scratch.b1_momentum_x[e_l] += dx * du[1];
      element_scratch.b2_momentum_x[e_l] += dy * du[1];
      element_scratch.b1_momentum_y[e_l] += dx * du[2];
      element_scratch.b2_momentum_y[e_l] += dy * du[2];
      element_scratch.b1_energy[e_l] += dx * du[3];
      element_scratch.b2_energy[e_l] += dy * du[3];

      element_scratch.b1_density[e_r] -= dx * du[0];
      element_scratch.b2_density[e_r] -= dy * du[0];
      element_scratch.b1_momentum_x[e_r] -= dx * du[1];
      element_scratch.b2_momentum_x[e_r] -= dy * du[1];
      element_scratch.b1_momentum_y[e_r] -= dx * du[2];
      element_scratch.b2_momentum_y[e_r] -= dy * du[2];
      element_scratch.b1_energy[e_r] -= dx * du[3];
      element_scratch.b2_energy[e_r] -= dy * du[3];
    }

    // Loop over periodic faces
    for (unsigned int i = 0; i < periodic_face_scratch.size(); ++i) {
      const auto e_l = periodic_face_scratch.elem_l[i];
      const auto e_r = periodic_face_scratch.elem_r[i];

      const auto dx = element_scratch.centroid_x[e_r] +
                      periodic_face_scratch.translation_x[i] -
                      element_scratch.centroid_x[e_l];
      const auto dy = element_scratch.centroid_y[e_r] +
                      periodic_face_scratch.translation_y[i] -
                      element_scratch.centroid_y[e_l];

      const auto u_l = get_state(element_scratch, e_l);
      const auto u_r = get_state(element_scratch, e_r);
      const auto du = u_r - u_l;

      element_scratch.A11[e_l] += dx * dx;
      element_scratch.A12[e_l] += dx * dy;
      element_scratch.A22[e_l] += dy * dy;

      element_scratch.A11[e_r] += dx * dx;
      element_scratch.A12[e_r] += dx * dy;
      element_scratch.A22[e_r] += dy * dy;

      element_scratch.b1_density[e_l] += dx * du[0];
      element_scratch.b2_density[e_l] += dy * du[0];
      element_scratch.b1_momentum_x[e_l] += dx * du[1];
      element_scratch.b2_momentum_x[e_l] += dy * du[1];
      element_scratch.b1_momentum_y[e_l] += dx * du[2];
      element_scratch.b2_momentum_y[e_l] += dy * du[2];
      element_scratch.b1_energy[e_l] += dx * du[3];
      element_scratch.b2_energy[e_l] += dy * du[3];

      element_scratch.b1_density[e_r] -= dx * du[0];
      element_scratch.b2_density[e_r] -= dy * du[0];
      element_scratch.b1_momentum_x[e_r] -= dx * du[1];
      element_scratch.b2_momentum_x[e_r] -= dy * du[1];
      element_scratch.b1_momentum_y[e_r] -= dx * du[2];
      element_scratch.b2_momentum_y[e_r] -= dy * du[2];
      element_scratch.b1_energy[e_r] -= dx * du[3];
      element_scratch.b2_energy[e_r] -= dy * du[3];
    }

    // Loop over boundary faces
    for (unsigned int i = 0; i < boundary_face_scratch.size(); ++i) {
      const auto e = boundary_face_scratch.elem[i];
      const auto boundary_id = boundary_face_scratch.boundary_id[i];
      const auto n_x = boundary_face_scratch.normal_x[i];
      const auto n_y = boundary_face_scratch.normal_y[i];
      const auto u = get_state(element_scratch, e);
      const Tensor<1, 2, RealType> n = { n_x, n_y };

      // Check that we don't accidentally have periodic boundaries
      DEBUG_ASSERT(boundary_id != 0 && boundary_id != 1 && boundary_id != 2 &&
                   boundary_id != 3);

      const auto dx =
        boundary_face_scratch.centroid_x[i] - element_scratch.centroid_x[e];
      const auto dy =
        boundary_face_scratch.centroid_y[i] - element_scratch.centroid_y[e];

      const auto interior = get_state(element_scratch, e);
      auto boundary = get_boundary_state(
        interior, n, boundary_id, is_freestream, get_initial_state());

      if (is_unsteady && boundary_id == 4) {
        boundary = unsteady_subsonic_inflow_state(
          u, n, time, boundary_face_scratch.centroid_y[i]);
      }

      const auto du = boundary - interior;

      element_scratch.A11[e] += dx * dx;
      element_scratch.A12[e] += dx * dy;
      element_scratch.A22[e] += dy * dy;

      element_scratch.b1_density[e] += dx * du[0];
      element_scratch.b2_density[e] += dy * du[0];
      element_scratch.b1_momentum_x[e] += dx * du[1];
      element_scratch.b2_momentum_x[e] += dy * du[1];
      element_scratch.b1_momentum_y[e] += dx * du[2];
      element_scratch.b2_momentum_y[e] += dy * du[2];
      element_scratch.b1_energy[e] += dx * du[3];
      element_scratch.b2_energy[e] += dy * du[3];
    }

    // Solve the linear system
    for (unsigned int i = 0; i < n_elements; ++i) {

      const auto local_A11 = element_scratch.A11[i];
      const auto local_A12 = element_scratch.A12[i];
      const auto local_A22 = element_scratch.A22[i];

      const auto det = local_A11 * local_A22 - local_A12 * local_A12;

      if (std::abs(det) < 1e-12) {
        element_scratch.grad_x_density[i] = 0.0;
        element_scratch.grad_y_density[i] = 0.0;
        element_scratch.grad_x_momentum_x[i] = 0.0;
        element_scratch.grad_y_momentum_x[i] = 0.0;
        element_scratch.grad_x_momentum_y[i] = 0.0;
        element_scratch.grad_y_momentum_y[i] = 0.0;
        element_scratch.grad_x_energy[i] = 0.0;
        element_scratch.grad_y_energy[i] = 0.0;
        continue;
      }

      element_scratch.grad_x_density[i] =
        (local_A22 * element_scratch.b1_density[i] -
         local_A12 * element_scratch.b2_density[i]) /
        det;

      element_scratch.grad_y_density[i] =
        (local_A11 * element_scratch.b2_density[i] -
         local_A12 * element_scratch.b1_density[i]) /
        det;

      element_scratch.grad_x_momentum_x[i] =
        (local_A22 * element_scratch.b1_momentum_x[i] -
         local_A12 * element_scratch.b2_momentum_x[i]) /
        det;

      element_scratch.grad_y_momentum_x[i] =
        (local_A11 * element_scratch.b2_momentum_x[i] -
         local_A12 * element_scratch.b1_momentum_x[i]) /
        det;

      element_scratch.grad_x_momentum_y[i] =
        (local_A22 * element_scratch.b1_momentum_y[i] -
         local_A12 * element_scratch.b2_momentum_y[i]) /
        det;

      element_scratch.grad_y_momentum_y[i] =
        (local_A11 * element_scratch.b2_momentum_y[i] -
         local_A12 * element_scratch.b1_momentum_y[i]) /
        det;

      element_scratch.grad_x_energy[i] =
        (local_A22 * element_scratch.b1_energy[i] -
         local_A12 * element_scratch.b2_energy[i]) /
        det;

      element_scratch.grad_y_energy[i] =
        (local_A11 * element_scratch.b2_energy[i] -
         local_A12 * element_scratch.b1_energy[i]) /
        det;
    }
  }

  /**
   * @brief Barth Jespersen Limiter
   */
  void apply_barth_jespersen_limiter(
    const MeshData& mesh_data,
    const InteriorFaceData<dim, RealType>& interior_face_scratch,
    const BoundaryFaceData<dim, RealType>& boundary_face_scratch,
    const PeriodicFaceData<dim, RealType>& periodic_face_scratch,
    ElementData<dim, degree, RealType>& element_scratch,
    bool is_freestream) const
  {
    const auto n_elements = element_scratch.size();
    set(element_scratch.alpha, 1.0);

    // Create vectors for the min and max states for each cells and its
    // neighbors
    auto min_density = element_scratch.density;
    auto max_density = element_scratch.density;
    auto min_momentum_x = element_scratch.momentum_x;
    auto max_momentum_x = element_scratch.momentum_x;
    auto min_momentum_y = element_scratch.momentum_y;
    auto max_momentum_y = element_scratch.momentum_y;
    auto min_energy = element_scratch.energy;
    auto max_energy = element_scratch.energy;

    // Loop over interior faces
    for (unsigned int i = 0; i < interior_face_scratch.size(); ++i) {
      const auto e_l = interior_face_scratch.elem_l[i];
      const auto e_r = interior_face_scratch.elem_r[i];

      min_density[e_l] =
        std::min(min_density[e_l], element_scratch.density[e_r]);
      max_density[e_l] =
        std::max(max_density[e_l], element_scratch.density[e_r]);
      min_density[e_r] =
        std::min(min_density[e_r], element_scratch.density[e_l]);
      max_density[e_r] =
        std::max(max_density[e_r], element_scratch.density[e_l]);

      min_momentum_x[e_l] =
        std::min(min_momentum_x[e_l], element_scratch.momentum_x[e_r]);
      max_momentum_x[e_l] =
        std::max(max_momentum_x[e_l], element_scratch.momentum_x[e_r]);
      min_momentum_x[e_r] =
        std::min(min_momentum_x[e_r], element_scratch.momentum_x[e_l]);
      max_momentum_x[e_r] =
        std::max(max_momentum_x[e_r], element_scratch.momentum_x[e_l]);

      min_momentum_y[e_l] =
        std::min(min_momentum_y[e_l], element_scratch.momentum_y[e_r]);
      max_momentum_y[e_l] =
        std::max(max_momentum_y[e_l], element_scratch.momentum_y[e_r]);
      min_momentum_y[e_r] =
        std::min(min_momentum_y[e_r], element_scratch.momentum_y[e_l]);
      max_momentum_y[e_r] =
        std::max(max_momentum_y[e_r], element_scratch.momentum_y[e_l]);

      min_energy[e_l] = std::min(min_energy[e_l], element_scratch.energy[e_r]);
      max_energy[e_l] = std::max(max_energy[e_l], element_scratch.energy[e_r]);
      min_energy[e_r] = std::min(min_energy[e_r], element_scratch.energy[e_l]);
      max_energy[e_r] = std::max(max_energy[e_r], element_scratch.energy[e_l]);
    }

    // Loop over periodic faces
    for (unsigned int i = 0; i < periodic_face_scratch.size(); ++i) {
      const auto e_l = periodic_face_scratch.elem_l[i];
      const auto e_r = periodic_face_scratch.elem_r[i];

      min_density[e_l] =
        std::min(min_density[e_l], element_scratch.density[e_r]);
      max_density[e_l] =
        std::max(max_density[e_l], element_scratch.density[e_r]);
      min_density[e_r] =
        std::min(min_density[e_r], element_scratch.density[e_l]);
      max_density[e_r] =
        std::max(max_density[e_r], element_scratch.density[e_l]);

      min_momentum_x[e_l] =
        std::min(min_momentum_x[e_l], element_scratch.momentum_x[e_r]);
      max_momentum_x[e_l] =
        std::max(max_momentum_x[e_l], element_scratch.momentum_x[e_r]);
      min_momentum_x[e_r] =
        std::min(min_momentum_x[e_r], element_scratch.momentum_x[e_l]);
      max_momentum_x[e_r] =
        std::max(max_momentum_x[e_r], element_scratch.momentum_x[e_l]);

      min_momentum_y[e_l] =
        std::min(min_momentum_y[e_l], element_scratch.momentum_y[e_r]);
      max_momentum_y[e_l] =
        std::max(max_momentum_y[e_l], element_scratch.momentum_y[e_r]);
      min_momentum_y[e_r] =
        std::min(min_momentum_y[e_r], element_scratch.momentum_y[e_l]);
      max_momentum_y[e_r] =
        std::max(max_momentum_y[e_r], element_scratch.momentum_y[e_l]);

      min_energy[e_l] = std::min(min_energy[e_l], element_scratch.energy[e_r]);
      max_energy[e_l] = std::max(max_energy[e_l], element_scratch.energy[e_r]);
      min_energy[e_r] = std::min(min_energy[e_r], element_scratch.energy[e_l]);
      max_energy[e_r] = std::max(max_energy[e_r], element_scratch.energy[e_l]);
    }

    for (unsigned int i = 0; i < n_elements; ++i) {
      // Grab the three vectors from cell centroid to nodes
      const auto n_1 = mesh_data.node_1[i];
      const auto n_2 = mesh_data.node_2[i];
      const auto n_3 = mesh_data.node_3[i];

      const auto r_1_x = mesh_data.x[n_1] - element_scratch.centroid_x[i];
      const auto r_1_y = mesh_data.y[n_1] - element_scratch.centroid_y[i];
      const auto r_2_x = mesh_data.x[n_2] - element_scratch.centroid_x[i];
      const auto r_2_y = mesh_data.y[n_2] - element_scratch.centroid_y[i];
      const auto r_3_x = mesh_data.x[n_3] - element_scratch.centroid_x[i];
      const auto r_3_y = mesh_data.y[n_3] - element_scratch.centroid_y[i];

      // Compute the neighbor states
      const auto density_1 = element_scratch.density[i] +
                             r_1_x * element_scratch.grad_x_density[i] +
                             r_1_y * element_scratch.grad_y_density[i];
      const auto density_2 = element_scratch.density[i] +
                             r_2_x * element_scratch.grad_x_density[i] +
                             r_2_y * element_scratch.grad_y_density[i];
      const auto density_3 = element_scratch.density[i] +
                             r_3_x * element_scratch.grad_x_density[i] +
                             r_3_y * element_scratch.grad_y_density[i];

      const auto momentum_x_1 = element_scratch.momentum_x[i] +
                                r_1_x * element_scratch.grad_x_momentum_x[i] +
                                r_1_y * element_scratch.grad_y_momentum_x[i];
      const auto momentum_x_2 = element_scratch.momentum_x[i] +
                                r_2_x * element_scratch.grad_x_momentum_x[i] +
                                r_2_y * element_scratch.grad_y_momentum_x[i];
      const auto momentum_x_3 = element_scratch.momentum_x[i] +
                                r_3_x * element_scratch.grad_x_momentum_x[i] +
                                r_3_y * element_scratch.grad_y_momentum_x[i];

      const auto momentum_y_1 = element_scratch.momentum_y[i] +
                                r_1_x * element_scratch.grad_x_momentum_y[i] +
                                r_1_y * element_scratch.grad_y_momentum_y[i];
      const auto momentum_y_2 = element_scratch.momentum_y[i] +
                                r_2_x * element_scratch.grad_x_momentum_y[i] +
                                r_2_y * element_scratch.grad_y_momentum_y[i];
      const auto momentum_y_3 = element_scratch.momentum_y[i] +
                                r_3_x * element_scratch.grad_x_momentum_y[i] +
                                r_3_y * element_scratch.grad_y_momentum_y[i];

      const auto energy_1 = element_scratch.energy[i] +
                            r_1_x * element_scratch.grad_x_energy[i] +
                            r_1_y * element_scratch.grad_y_energy[i];
      const auto energy_2 = element_scratch.energy[i] +
                            r_2_x * element_scratch.grad_x_energy[i] +
                            r_2_y * element_scratch.grad_y_energy[i];
      const auto energy_3 = element_scratch.energy[i] +
                            r_3_x * element_scratch.grad_x_energy[i] +
                            r_3_y * element_scratch.grad_y_energy[i];

      // Compute the alpha for each of neighbor states
      auto limit = [](RealType q_node,
                      RealType q_cell,
                      RealType q_min,
                      RealType q_max) -> RealType {
        const auto dq = q_node - q_cell;
        if (dq > 1.0e-8) {
          return std::min(RealType(1), (q_max - q_cell) / dq);
        } else if (dq < -1.0e-8) {
          return std::min(RealType(1), (q_min - q_cell) / dq);
        }
        return RealType(1);
      };

      element_scratch.alpha[i] = std::min(element_scratch.alpha[i],
                                          limit(density_1,
                                                element_scratch.density[i],
                                                min_density[i],
                                                max_density[i]));
      element_scratch.alpha[i] = std::min(element_scratch.alpha[i],
                                          limit(density_2,
                                                element_scratch.density[i],
                                                min_density[i],
                                                max_density[i]));
      element_scratch.alpha[i] = std::min(element_scratch.alpha[i],
                                          limit(density_3,
                                                element_scratch.density[i],
                                                min_density[i],
                                                max_density[i]));

      element_scratch.alpha[i] = std::min(element_scratch.alpha[i],
                                          limit(momentum_x_1,
                                                element_scratch.momentum_x[i],
                                                min_momentum_x[i],
                                                max_momentum_x[i]));
      element_scratch.alpha[i] = std::min(element_scratch.alpha[i],
                                          limit(momentum_x_2,
                                                element_scratch.momentum_x[i],
                                                min_momentum_x[i],
                                                max_momentum_x[i]));
      element_scratch.alpha[i] = std::min(element_scratch.alpha[i],
                                          limit(momentum_x_3,
                                                element_scratch.momentum_x[i],
                                                min_momentum_x[i],
                                                max_momentum_x[i]));

      element_scratch.alpha[i] = std::min(element_scratch.alpha[i],
                                          limit(momentum_y_1,
                                                element_scratch.momentum_y[i],
                                                min_momentum_y[i],
                                                max_momentum_y[i]));
      element_scratch.alpha[i] = std::min(element_scratch.alpha[i],
                                          limit(momentum_y_2,
                                                element_scratch.momentum_y[i],
                                                min_momentum_y[i],
                                                max_momentum_y[i]));
      element_scratch.alpha[i] = std::min(element_scratch.alpha[i],
                                          limit(momentum_y_3,
                                                element_scratch.momentum_y[i],
                                                min_momentum_y[i],
                                                max_momentum_y[i]));

      element_scratch.alpha[i] = std::min(
        element_scratch.alpha[i],
        limit(
          energy_1, element_scratch.energy[i], min_energy[i], max_energy[i]));
      element_scratch.alpha[i] = std::min(
        element_scratch.alpha[i],
        limit(
          energy_2, element_scratch.energy[i], min_energy[i], max_energy[i]));
      element_scratch.alpha[i] = std::min(
        element_scratch.alpha[i],
        limit(
          energy_3, element_scratch.energy[i], min_energy[i], max_energy[i]));
    }

    // Apply limiter
    for (unsigned int i = 0; i < n_elements; ++i) {
      element_scratch.grad_x_density[i] *= element_scratch.alpha[i];
      element_scratch.grad_y_density[i] *= element_scratch.alpha[i];
      element_scratch.grad_x_momentum_x[i] *= element_scratch.alpha[i];
      element_scratch.grad_y_momentum_x[i] *= element_scratch.alpha[i];
      element_scratch.grad_x_momentum_y[i] *= element_scratch.alpha[i];
      element_scratch.grad_y_momentum_y[i] *= element_scratch.alpha[i];
      element_scratch.grad_x_energy[i] *= element_scratch.alpha[i];
      element_scratch.grad_y_energy[i] *= element_scratch.alpha[i];
    }
  }
};
