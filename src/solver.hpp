#pragma once

#include <flux.hpp>
#include <functional>
#include <libassert/assert.hpp>
#include <parameters.hpp>
#include <tensor.hpp>
#include <triangulation.hpp>
#include <utilities.hpp>

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
  const auto M_b = (mach_root_1 >= 0 && mach_root_2 >= 0)
                     ? std::min(mach_root_1, mach_root_2)
                     : std::max(mach_root_1, mach_root_2);
  DEBUG_ASSERT(M_b >= 0.0);

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

  void set_free_stream_initial_state(
    ElementData<dim, RealType>& element_scratch) const
  {
    set(element_scratch.density,
        Parameters<RealType>::p_0 / Parameters<RealType>::T_0_and_R);
    set(element_scratch.momentum_x, 1.0);
    set(element_scratch.momentum_y, 1.0);
    set(element_scratch.energy,
        Parameters<RealType>::p_0 /
          (Parameters<RealType>::gamma - RealType(1.0)));
  }

  void compute_free_stream_residual(
    const InteriorFaceData<dim, RealType>& interior_face_scratch,
    const BoundaryFaceData<dim, RealType>& boundary_face_scratch,
    const PeriodicFaceData<dim, RealType>& periodic_face_scratch,
    ElementData<dim, RealType>& element_scratch,
    const FluxFunction& flux_func = &flux_roe) const
  {
    // Zero out the residuals and optimal timestep
    zero(element_scratch.residual_density);
    zero(element_scratch.residual_momentum_x);
    zero(element_scratch.residual_momentum_y);
    zero(element_scratch.residual_energy);
    zero(element_scratch.optimal_timestep);

    // Loop over interior faces
    for (unsigned int i = 0; i < interior_face_scratch.size(); ++i) {
      // Construct the state vectors and normals on the fly
      const auto e_l = interior_face_scratch.elem_l[i];
      const auto e_r = interior_face_scratch.elem_r[i];
      const auto n_x = interior_face_scratch.normal_x[i];
      const auto n_y = interior_face_scratch.normal_y[i];
      const auto area = interior_face_scratch.face_area[i];

      const Tensor<1, 4, RealType> u_l = { element_scratch.density[e_l],
                                           element_scratch.momentum_x[e_l],
                                           element_scratch.momentum_y[e_l],
                                           element_scratch.energy[e_l] };
      const Tensor<1, 4, RealType> u_r = { element_scratch.density[e_r],
                                           element_scratch.momentum_x[e_r],
                                           element_scratch.momentum_y[e_r],
                                           element_scratch.energy[e_r] };
      const Tensor<1, 2, RealType> n = { n_x, n_y };

      const auto result = flux_func(u_l, u_r, n);
      const auto flux = result.first;
      const auto max_wavespeed = result.second;

      // Add the residual to the elements
      element_scratch.residual_density[e_l] += flux[0] * area;
      element_scratch.residual_momentum_x[e_l] += flux[1] * area;
      element_scratch.residual_momentum_y[e_l] += flux[2] * area;
      element_scratch.residual_energy[e_l] += flux[3] * area;

      element_scratch.residual_density[e_r] -= flux[0] * area;
      element_scratch.residual_momentum_x[e_r] -= flux[1] * area;
      element_scratch.residual_momentum_y[e_r] -= flux[2] * area;
      element_scratch.residual_energy[e_r] -= flux[3] * area;

      // Add the edge-weighted wave speed to the optimal timestep. We'll compute
      // the actual timestep later.
      element_scratch.optimal_timestep[e_r] += max_wavespeed * area;
      element_scratch.optimal_timestep[e_l] += max_wavespeed * area;
    }

    // Loop over boundary faces. Note that we call flux_roe with an identical
    // state rather than calling actual boundary conditions.
    for (unsigned int i = 0; i < boundary_face_scratch.size(); ++i) {
      // Construct the state vectors and normals on the fly
      const auto e = boundary_face_scratch.elem[i];
      const auto n_x = boundary_face_scratch.normal_x[i];
      const auto n_y = boundary_face_scratch.normal_y[i];
      const auto area = boundary_face_scratch.face_area[i];

      const Tensor<1, 4, RealType> u = { element_scratch.density[e],
                                         element_scratch.momentum_x[e],
                                         element_scratch.momentum_y[e],
                                         element_scratch.energy[e] };
      const Tensor<1, 2, RealType> n = { n_x, n_y };

      const auto result = flux_func(u, u, n);
      const auto flux = result.first;
      const auto max_wavespeed = result.second;

      // Add the residual to the elements
      element_scratch.residual_density[e] += flux[0] * area;
      element_scratch.residual_momentum_x[e] += flux[1] * area;
      element_scratch.residual_momentum_y[e] += flux[2] * area;
      element_scratch.residual_energy[e] += flux[3] * area;

      // Add the edge-weighted wave speed to the optimal timestep. We'll compute
      // the actual timestep later.
      element_scratch.optimal_timestep[e] += max_wavespeed * area;
    }

    // Loop over periodic faces
    for (unsigned int i = 0; i < periodic_face_scratch.size(); ++i) {
      // Construct the state vectors and normals on the fly
      const auto e_l = periodic_face_scratch.elem_l[i];
      const auto e_r = periodic_face_scratch.elem_r[i];
      const auto n_x = periodic_face_scratch.normal_x[i];
      const auto n_y = periodic_face_scratch.normal_y[i];
      const auto area = periodic_face_scratch.face_area[i];

      const Tensor<1, 4, RealType> u_l = { element_scratch.density[e_l],
                                           element_scratch.momentum_x[e_l],
                                           element_scratch.momentum_y[e_l],
                                           element_scratch.energy[e_l] };
      const Tensor<1, 4, RealType> u_r = { element_scratch.density[e_r],
                                           element_scratch.momentum_x[e_r],
                                           element_scratch.momentum_y[e_r],
                                           element_scratch.energy[e_r] };
      const Tensor<1, 2, RealType> n = { n_x, n_y };

      const auto result = flux_func(u_l, u_r, n);
      const auto flux = result.first;
      const auto max_wavespeed = result.second;

      // Add the residual to the elements
      element_scratch.residual_density[e_l] += flux[0] * area;
      element_scratch.residual_momentum_x[e_l] += flux[1] * area;
      element_scratch.residual_momentum_y[e_l] += flux[2] * area;
      element_scratch.residual_energy[e_l] += flux[3] * area;

      element_scratch.residual_density[e_r] -= flux[0] * area;
      element_scratch.residual_momentum_x[e_r] -= flux[1] * area;
      element_scratch.residual_momentum_y[e_r] -= flux[2] * area;
      element_scratch.residual_energy[e_r] -= flux[3] * area;

      // Add the edge-weighted wave speed to the optimal timestep. We'll compute
      // the actual timestep later.
      element_scratch.optimal_timestep[e_r] += max_wavespeed * area;
      element_scratch.optimal_timestep[e_l] += max_wavespeed * area;
    }

    // Compute the optimal timestep
    for (unsigned int i = 0; i < element_scratch.size(); ++i) {
      element_scratch.optimal_timestep[i] = Parameters<RealType>::cfl_max *
                                            RealType(2.0) /
                                            element_scratch.optimal_timestep[i];
    }
  };

  void compute_residual(
    const InteriorFaceData<dim, RealType>& interior_face_scratch,
    const BoundaryFaceData<dim, RealType>& boundary_face_scratch,
    const PeriodicFaceData<dim, RealType>& periodic_face_scratch,
    ElementData<dim, RealType>& element_scratch,
    const FluxFunction& flux_func = &flux_roe) const
  {
    // Zero out the residuals and optimal timestep
    zero(element_scratch.residual_density);
    zero(element_scratch.residual_momentum_x);
    zero(element_scratch.residual_momentum_y);
    zero(element_scratch.residual_energy);
    zero(element_scratch.optimal_timestep);

    // Loop over interior faces
    for (unsigned int i = 0; i < interior_face_scratch.size(); ++i) {
      // Construct the state vectors and normals on the fly
      const auto e_l = interior_face_scratch.elem_l[i];
      const auto e_r = interior_face_scratch.elem_r[i];
      const auto n_x = interior_face_scratch.normal_x[i];
      const auto n_y = interior_face_scratch.normal_y[i];
      const auto area = interior_face_scratch.face_area[i];

      const Tensor<1, 4, RealType> u_l = { element_scratch.density[e_l],
                                           element_scratch.momentum_x[e_l],
                                           element_scratch.momentum_y[e_l],
                                           element_scratch.energy[e_l] };
      const Tensor<1, 4, RealType> u_r = { element_scratch.density[e_r],
                                           element_scratch.momentum_x[e_r],
                                           element_scratch.momentum_y[e_r],
                                           element_scratch.energy[e_r] };
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

      // Add the edge-weighted wave speed to the optimal timestep. We'll compute
      // the actual timestep later.
      element_scratch.optimal_timestep[e_r] += max_wavespeed * area;
      element_scratch.optimal_timestep[e_l] += max_wavespeed * area;
    }

    // Loop over boundary faces.
    for (unsigned int i = 0; i < boundary_face_scratch.size(); ++i) {
      // Construct the state vectors and normals on the fly
      const auto e = boundary_face_scratch.elem[i];
      const auto n_x = boundary_face_scratch.normal_x[i];
      const auto n_y = boundary_face_scratch.normal_y[i];
      const auto area = boundary_face_scratch.face_area[i];
      const auto boundary_id = boundary_face_scratch.boundary_id[i];

      const Tensor<1, 4, RealType> u = { element_scratch.density[e],
                                         element_scratch.momentum_x[e],
                                         element_scratch.momentum_y[e],
                                         element_scratch.energy[e] };
      const Tensor<1, 2, RealType> n = { n_x, n_y };

      // Check that we don't accidentally have periodic boundaries
      DEBUG_ASSERT(boundary_id != 0 && boundary_id != 1 && boundary_id != 2 &&
                   boundary_id != 3);

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
          ASSERT(false, "How did we get here? Probably hardcoding");
      }
      ASSERT(boundary_flux_func != nullptr);

      const auto result = boundary_flux_func(u, n);
      const auto flux = result.first;
      const auto max_wavespeed = result.second;
      DEBUG_ASSERT(max_wavespeed > 0);

      // Add the residual to the elements
      element_scratch.residual_density[e] += flux[0] * area;
      element_scratch.residual_momentum_x[e] += flux[1] * area;
      element_scratch.residual_momentum_y[e] += flux[2] * area;
      element_scratch.residual_energy[e] += flux[3] * area;

      // Add the edge-weighted wave speed to the optimal timestep. We'll compute
      // the actual timestep later.
      element_scratch.optimal_timestep[e] += max_wavespeed * area;
    }

    // Loop over periodic faces
    for (unsigned int i = 0; i < periodic_face_scratch.size(); ++i) {
      // Construct the state vectors and normals on the fly
      const auto e_l = periodic_face_scratch.elem_l[i];
      const auto e_r = periodic_face_scratch.elem_r[i];
      const auto n_x = periodic_face_scratch.normal_x[i];
      const auto n_y = periodic_face_scratch.normal_y[i];
      const auto area = periodic_face_scratch.face_area[i];

      const Tensor<1, 4, RealType> u_l = { element_scratch.density[e_l],
                                           element_scratch.momentum_x[e_l],
                                           element_scratch.momentum_y[e_l],
                                           element_scratch.energy[e_l] };
      const Tensor<1, 4, RealType> u_r = { element_scratch.density[e_r],
                                           element_scratch.momentum_x[e_r],
                                           element_scratch.momentum_y[e_r],
                                           element_scratch.energy[e_r] };
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

      // Add the edge-weighted wave speed to the optimal timestep. We'll compute
      // the actual timestep later.
      element_scratch.optimal_timestep[e_r] += max_wavespeed * area;
      element_scratch.optimal_timestep[e_l] += max_wavespeed * area;
    }

    // Compute the optimal timestep
    for (unsigned int i = 0; i < element_scratch.size(); ++i) {
      element_scratch.optimal_timestep[i] =
        Parameters<RealType>::cfl_max * RealType(2.0) *
        element_scratch.area[i] / element_scratch.optimal_timestep[i];
      DEBUG_ASSERT(element_scratch.optimal_timestep[i] >= 0.0);
    }
  };

  void compute_update_with_local_timestepping(
    ElementData<dim, RealType>& element_scratch) const
  {
    for (unsigned int i = 0; i < element_scratch.size(); ++i) {
      const auto dt = 1e-4; // element_scratch.optimal_timestep[i];
      const auto inv_area = element_scratch.inv_area[i];
      element_scratch.density[i] -=
        element_scratch.residual_density[i] * dt * inv_area;
      element_scratch.momentum_x[i] -=
        element_scratch.residual_momentum_x[i] * dt * inv_area;
      element_scratch.momentum_y[i] -=
        element_scratch.residual_momentum_y[i] * dt * inv_area;
      element_scratch.energy[i] -=
        element_scratch.residual_energy[i] * dt * inv_area;

      ASSERT(element_scratch.density[i] > 0.0, "Density must be positive");
    }
  }
};
