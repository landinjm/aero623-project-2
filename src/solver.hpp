#pragma once

#include <flux.hpp>
#include <functional>
#include <libassert/assert.hpp>
#include <numbers>
#include <parameters.hpp>
#include <tensor.hpp>
#include <triangulation.hpp>
#include <utilities.hpp>

#include "limiters.hpp"

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
inline std::pair<Tensor<1, 2 + dim, RealType>, RealType>
inviscid_wall(const Tensor<1, 2 + dim, RealType>& interior_state,
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
  const auto u_b = (rho_and_u - dot * n_x) / rho;
  const auto v_b = (rho_and_v - dot * n_y) / rho;

  // Compute boundary pressure
  const auto p_b = (Parameters<RealType>::gamma - 1.0) *
                   (rho_and_E - 0.5 * rho * (u_b * u_b + v_b * v_b));
  DEBUG_ASSERT(p_b >= 0, "Pressure must be positive");

  // Construct boundary flux
  const auto boundary_flux = Tensor<1, 2 + dim, RealType>{
    RealType(0.0), p_b * n_x, p_b * n_y, RealType(0.0)
  };

  // Compute the max wavespeed
  const auto smag = max_wavespeed(u_b, v_b, n_x, n_y, p_b, rho);

  return { boundary_flux, smag };
}

template<unsigned int dim, typename RealType>
inline std::pair<Tensor<1, 2 + dim, RealType>, RealType>
subsonic_inflow(const Tensor<1, 2 + dim, RealType>& interior_state,
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

  // Construct the boundary state
  const auto state_b =
    Tensor<1, 2 + dim, RealType>{ rho_b, rho_b * u_b, rho_b * v_b, energy_b };

  // Compute the boundary flux
  const auto boundary_flux = euler_flux<dim, RealType>(state_b, normal);

  // Compute the max wavespeed
  const auto smag = max_wavespeed(u_b, v_b, n_x, n_y, p_b, rho_b);

  return { boundary_flux, smag };
}

template<unsigned int dim, typename RealType>
inline std::pair<Tensor<1, 2 + dim, RealType>, RealType>
unsteady_subsonic_inflow(const Tensor<1, 2 + dim, RealType>& interior_state,
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

  // Construct the boundary state
  const auto state_b =
    Tensor<1, 2 + dim, RealType>{ rho_b, rho_b * u_b, rho_b * v_b, energy_b };

  // Compute the boundary flux
  const auto boundary_flux = euler_flux<dim, RealType>(state_b, normal);

  // Compute the max wavespeed
  const auto smag = max_wavespeed(u_b, v_b, n_x, n_y, p_b, rho_b);

  return { boundary_flux, smag };
}

template<unsigned int dim, typename RealType>
inline std::pair<Tensor<1, 2 + dim, RealType>, RealType>
subsonic_outflow(const Tensor<1, 2 + dim, RealType>& interior_state,
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

  // Construct the boundary state
  const auto state_b =
    Tensor<1, 2 + dim, RealType>{ rho_b, rho_b * u_b, rho_b * v_b, energy_b };

  // Compute the boundary flux
  const auto boundary_flux = euler_flux<dim, RealType>(state_b, normal);

  // Compute the max wavespeed
  const auto smag =
    max_wavespeed(u_b, v_b, n_x, n_y, Parameters<RealType>::p_out, rho_b);

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

  void compute_free_stream_residual(
    const InteriorFaceData<dim, RealType>& interior_face_scratch,
    const BoundaryFaceData<dim, RealType>& boundary_face_scratch,
    const PeriodicFaceData<dim, RealType>& periodic_face_scratch,
    ElementData<dim, degree, RealType>& element_scratch,
    const FluxFunction& flux_func = &flux_roe) const
  {
    zero_values(element_scratch);

    if constexpr (degree == 2) {
      interior_face_gradient_prep(interior_face_scratch, element_scratch);
      periodic_face_gradient_prep(periodic_face_scratch, element_scratch);
      boundary_face_gradient_prep(boundary_face_scratch, element_scratch, true);
      finalize_gradient(element_scratch);
    }

    interior_face_residual(interior_face_scratch, element_scratch, flux_func);
    periodic_face_residual(periodic_face_scratch, element_scratch, flux_func);
    boundary_face_residual(
      boundary_face_scratch, element_scratch, flux_func, true);
    optimal_time(element_scratch);
  };

  void compute_residual(
    const InteriorFaceData<dim, RealType>& interior_face_scratch,
    const BoundaryFaceData<dim, RealType>& boundary_face_scratch,
    const PeriodicFaceData<dim, RealType>& periodic_face_scratch,
    ElementData<dim, degree, RealType>& element_scratch,
    const FluxFunction& flux_func = &flux_roe,
    RealType time = 0,
    bool is_unsteady = false) const
  {
    zero_values(element_scratch);

    if constexpr (degree == 2) {
      interior_face_gradient_prep(interior_face_scratch, element_scratch);
      periodic_face_gradient_prep(periodic_face_scratch, element_scratch);
      boundary_face_gradient_prep(
        boundary_face_scratch, element_scratch, false);
      finalize_gradient(element_scratch);
    }

    interior_face_residual(interior_face_scratch, element_scratch, flux_func);
    periodic_face_residual(periodic_face_scratch, element_scratch, flux_func);
    boundary_face_residual(boundary_face_scratch,
                           element_scratch,
                           flux_func,
                           false,
                           is_unsteady,
                           time);
    optimal_time(element_scratch);
  };

  void compute_update_with_local_timestepping(
    ElementData<dim, degree, RealType>& element_scratch,
    const InteriorFaceData<dim, RealType>& interior_face_scratch,
    const BoundaryFaceData<dim, RealType>& boundary_face_scratch,
    const PeriodicFaceData<dim, RealType>& periodic_face_scratch,
    const FluxFunction& flux_func = &flux_roe,
    bool is_freestream = false) const
  {
    if (is_freestream) {
      compute_free_stream_residual(interior_face_scratch,
                                   boundary_face_scratch,
                                   periodic_face_scratch,
                                   element_scratch,
                                   flux_func);
    } else {
      compute_residual(interior_face_scratch,
                       boundary_face_scratch,
                       periodic_face_scratch,
                       element_scratch,
                       flux_func);
    }

    for (unsigned int i = 0; i < element_scratch.size(); ++i) {
      const auto dt = element_scratch.optimal_timestep[i];
      element_scratch.density[i] -=
        element_scratch.residual_density[i] * dt * element_scratch.inv_area[i];
      element_scratch.momentum_x[i] -= element_scratch.residual_momentum_x[i] *
                                       dt * element_scratch.inv_area[i];
      element_scratch.momentum_y[i] -= element_scratch.residual_momentum_y[i] *
                                       dt * element_scratch.inv_area[i];
      element_scratch.energy[i] -=
        element_scratch.residual_energy[i] * dt * element_scratch.inv_area[i];
    }
  }

  void compute_update_with_local_timestepping(
    ElementData<dim, degree, RealType>& element_scratch) const
  {
    for (unsigned int i = 0; i < element_scratch.size(); ++i) {
      const auto dt = element_scratch.optimal_timestep[i];
      const auto inv_area = element_scratch.inv_area[i];
      element_scratch.density[i] -= element_scratch.residual_density[i] * dt;
      element_scratch.momentum_x[i] -=
        element_scratch.residual_momentum_x[i] * dt;
      element_scratch.momentum_y[i] -=
        element_scratch.residual_momentum_y[i] * dt;
      element_scratch.energy[i] -= element_scratch.residual_energy[i] * dt;
    }
  }

  RealType compute_update_with_ssp_rk2(
    ElementData<dim, degree, RealType>& element_scratch,
    const InteriorFaceData<dim, RealType>& interior_face_scratch,
    const BoundaryFaceData<dim, RealType>& boundary_face_scratch,
    const PeriodicFaceData<dim, RealType>& periodic_face_scratch,
    const FluxFunction& flux_func = &flux_roe,
    RealType time = 0) const
  {
    compute_residual(interior_face_scratch,
                     boundary_face_scratch,
                     periodic_face_scratch,
                     element_scratch,
                     flux_func,
                     time,
                     true);

    // First determine the maximum timestep
    auto dt = min(element_scratch.optimal_timestep);

    // Copy the element data into a tmp vector
    auto tmp = element_scratch;

    // First stage
    for (unsigned int i = 0; i < tmp.size(); ++i) {
      tmp.density[i] -= tmp.residual_density[i] * dt * tmp.inv_area[i];
      tmp.momentum_x[i] -= tmp.residual_momentum_x[i] * dt * tmp.inv_area[i];
      ;
      tmp.momentum_y[i] -= tmp.residual_momentum_y[i] * dt * tmp.inv_area[i];
      ;
      tmp.energy[i] -= tmp.residual_energy[i] * dt * tmp.inv_area[i];
      ;
    }

    // Second stage
    compute_residual(interior_face_scratch,
                     boundary_face_scratch,
                     periodic_face_scratch,
                     tmp,
                     flux_func,
                     time,
                     true);
    for (unsigned int i = 0; i < element_scratch.size(); ++i) {
      element_scratch.density[i] =
        0.5 * element_scratch.density[i] + 0.5 * tmp.density[i] -
        0.5 * tmp.residual_density[i] * dt * tmp.inv_area[i];
      ;
      element_scratch.momentum_x[i] =
        0.5 * element_scratch.momentum_x[i] + 0.5 * tmp.momentum_x[i] -
        0.5 * tmp.residual_momentum_x[i] * dt * tmp.inv_area[i];
      ;
      element_scratch.momentum_y[i] =
        0.5 * element_scratch.momentum_y[i] + 0.5 * tmp.momentum_y[i] -
        0.5 * tmp.residual_momentum_y[i] * dt * tmp.inv_area[i];
      ;
      element_scratch.energy[i] =
        0.5 * element_scratch.energy[i] + 0.5 * tmp.energy[i] -
        0.5 * tmp.residual_energy[i] * dt * tmp.inv_area[i];
      ;
    }

    // Return the timestep
    return dt;
  }

private:
  /**
   * @brief Contruct state from element index.
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
   * @brief Compute the gradient over the interior faces.
   */
  void interior_face_gradient_prep(
    const InteriorFaceData<dim, RealType>& interior_face_scratch,
    ElementData<dim, degree, RealType>& element_scratch) const
  {
    for (unsigned int i = 0; i < interior_face_scratch.size(); ++i) {
      const auto e_l = interior_face_scratch.elem_l[i];
      const auto e_r = interior_face_scratch.elem_r[i];
      const auto n_x = interior_face_scratch.normal_x[i];
      const auto n_y = interior_face_scratch.normal_y[i];
      const auto area = interior_face_scratch.face_area[i];

      const auto avg_density =
        0.5 * element_scratch.density[e_l] + 0.5 * element_scratch.density[e_r];
      const auto avg_momentum_x = 0.5 * element_scratch.momentum_x[e_l] +
                                  0.5 * element_scratch.momentum_x[e_r];
      const auto avg_momentum_y = 0.5 * element_scratch.momentum_y[e_l] +
                                  0.5 * element_scratch.momentum_y[e_r];
      const auto avg_energy =
        0.5 * element_scratch.energy[e_l] + 0.5 * element_scratch.energy[e_r];

      const auto weight_density_x = avg_density * area * n_x;
      const auto weight_density_y = avg_density * area * n_y;
      const auto weight_momentum_x_x = avg_momentum_x * area * n_x;
      const auto weight_momentum_x_y = avg_momentum_x * area * n_y;
      const auto weight_momentum_y_x = avg_momentum_y * area * n_x;
      const auto weight_momentum_y_y = avg_momentum_y * area * n_y;
      const auto weight_energy_x = avg_energy * area * n_x;
      const auto weight_energy_y = avg_energy * area * n_y;

      element_scratch.grad_x_density[e_l] += weight_density_x;
      element_scratch.grad_y_density[e_l] += weight_density_y;
      element_scratch.grad_x_momentum_x[e_l] += weight_momentum_x_x;
      element_scratch.grad_y_momentum_x[e_l] += weight_momentum_x_y;
      element_scratch.grad_x_momentum_y[e_l] += weight_momentum_y_x;
      element_scratch.grad_y_momentum_y[e_l] += weight_momentum_y_y;
      element_scratch.grad_x_energy[e_l] += weight_energy_x;
      element_scratch.grad_y_energy[e_l] += weight_energy_y;

      element_scratch.grad_x_density[e_r] -= weight_density_x;
      element_scratch.grad_y_density[e_r] -= weight_density_y;
      element_scratch.grad_x_momentum_x[e_r] -= weight_momentum_x_x;
      element_scratch.grad_y_momentum_x[e_r] -= weight_momentum_x_y;
      element_scratch.grad_x_momentum_y[e_r] -= weight_momentum_y_x;
      element_scratch.grad_y_momentum_y[e_r] -= weight_momentum_y_y;
      element_scratch.grad_x_energy[e_r] -= weight_energy_x;
      element_scratch.grad_y_energy[e_r] -= weight_energy_y;
    }
  }

  /**
   * @brief Compute the gradient over the periodic faces.
   */
  void periodic_face_gradient_prep(
    const PeriodicFaceData<dim, RealType>& periodic_face_scratch,
    ElementData<dim, degree, RealType>& element_scratch) const
  {
    for (unsigned int i = 0; i < periodic_face_scratch.size(); ++i) {
      const auto e_l = periodic_face_scratch.elem_l[i];
      const auto e_r = periodic_face_scratch.elem_r[i];
      const auto n_x = periodic_face_scratch.normal_x[i];
      const auto n_y = periodic_face_scratch.normal_y[i];
      const auto area = periodic_face_scratch.face_area[i];

      const auto avg_density =
        0.5 * element_scratch.density[e_l] + 0.5 * element_scratch.density[e_r];
      const auto avg_momentum_x = 0.5 * element_scratch.momentum_x[e_l] +
                                  0.5 * element_scratch.momentum_x[e_r];
      const auto avg_momentum_y = 0.5 * element_scratch.momentum_y[e_l] +
                                  0.5 * element_scratch.momentum_y[e_r];
      const auto avg_energy =
        0.5 * element_scratch.energy[e_l] + 0.5 * element_scratch.energy[e_r];

      const auto weight_density_x = avg_density * area * n_x;
      const auto weight_density_y = avg_density * area * n_y;
      const auto weight_momentum_x_x = avg_momentum_x * area * n_x;
      const auto weight_momentum_x_y = avg_momentum_x * area * n_y;
      const auto weight_momentum_y_x = avg_momentum_y * area * n_x;
      const auto weight_momentum_y_y = avg_momentum_y * area * n_y;
      const auto weight_energy_x = avg_energy * area * n_x;
      const auto weight_energy_y = avg_energy * area * n_y;

      element_scratch.grad_x_density[e_l] += weight_density_x;
      element_scratch.grad_y_density[e_l] += weight_density_y;
      element_scratch.grad_x_momentum_x[e_l] += weight_momentum_x_x;
      element_scratch.grad_y_momentum_x[e_l] += weight_momentum_x_y;
      element_scratch.grad_x_momentum_y[e_l] += weight_momentum_y_x;
      element_scratch.grad_y_momentum_y[e_l] += weight_momentum_y_y;
      element_scratch.grad_x_energy[e_l] += weight_energy_x;
      element_scratch.grad_y_energy[e_l] += weight_energy_y;

      element_scratch.grad_x_density[e_r] -= weight_density_x;
      element_scratch.grad_y_density[e_r] -= weight_density_y;
      element_scratch.grad_x_momentum_x[e_r] -= weight_momentum_x_x;
      element_scratch.grad_y_momentum_x[e_r] -= weight_momentum_x_y;
      element_scratch.grad_x_momentum_y[e_r] -= weight_momentum_y_x;
      element_scratch.grad_y_momentum_y[e_r] -= weight_momentum_y_y;
      element_scratch.grad_x_energy[e_r] -= weight_energy_x;
      element_scratch.grad_y_energy[e_r] -= weight_energy_y;
    }
  }

  /**
   * @brief Compute the gradient over the boundary faces.
   */
  void boundary_face_gradient_prep(
    const BoundaryFaceData<dim, RealType>& boundary_face_scratch,
    ElementData<dim, degree, RealType>& element_scratch,
    bool is_freestream) const
  {
    for (unsigned int i = 0; i < boundary_face_scratch.size(); ++i) {
      const auto e = boundary_face_scratch.elem[i];
      const auto n_x = boundary_face_scratch.normal_x[i];
      const auto n_y = boundary_face_scratch.normal_y[i];
      const auto area = boundary_face_scratch.face_area[i];

      const auto avg_density = element_scratch.density[e];
      const auto avg_momentum_x = element_scratch.momentum_x[e];
      const auto avg_momentum_y = element_scratch.momentum_y[e];
      const auto avg_energy = element_scratch.energy[e];

      const auto weight_density_x = avg_density * area * n_x;
      const auto weight_density_y = avg_density * area * n_y;
      const auto weight_momentum_x_x = avg_momentum_x * area * n_x;
      const auto weight_momentum_x_y = avg_momentum_x * area * n_y;
      const auto weight_momentum_y_x = avg_momentum_y * area * n_x;
      const auto weight_momentum_y_y = avg_momentum_y * area * n_y;
      const auto weight_energy_x = avg_energy * area * n_x;
      const auto weight_energy_y = avg_energy * area * n_y;

      element_scratch.grad_x_density[e] += weight_density_x;
      element_scratch.grad_y_density[e] += weight_density_y;
      element_scratch.grad_x_momentum_x[e] += weight_momentum_x_x;
      element_scratch.grad_y_momentum_x[e] += weight_momentum_x_y;
      element_scratch.grad_x_momentum_y[e] += weight_momentum_y_x;
      element_scratch.grad_y_momentum_y[e] += weight_momentum_y_y;
      element_scratch.grad_x_energy[e] += weight_energy_x;
      element_scratch.grad_y_energy[e] += weight_energy_y;
    }
  }

  /**
   * @brief Finalize gradient calculation
   */
  void finalize_gradient(
    ElementData<dim, degree, RealType>& element_scratch) const
  {
    for (unsigned int i = 0; i < element_scratch.size(); ++i) {
      const auto inv_area = element_scratch.inv_area[i];
      element_scratch.grad_x_density[i] *= inv_area;
      element_scratch.grad_y_density[i] *= inv_area;
      element_scratch.grad_x_momentum_x[i] *= inv_area;
      element_scratch.grad_y_momentum_x[i] *= inv_area;
      element_scratch.grad_x_momentum_y[i] *= inv_area;
      element_scratch.grad_y_momentum_y[i] *= inv_area;
      element_scratch.grad_x_energy[i] *= inv_area;
      element_scratch.grad_y_energy[i] *= inv_area;
    }
  }
};
