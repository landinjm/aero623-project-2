#pragma once

#include <flux.hpp>
#include <parameters.hpp>
#include <read_gri.hpp>
#include <tensor.hpp>

template<unsigned int dim, typename RealType>
inline Tensor<1, 2 + dim, RealType>
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
  const auto v_b_x = (rho_and_u - dot * n_x) / rho;
  const auto v_b_y = (rho_and_v - dot * n_y) / rho;

  // Compute boundary pressure
  const auto p_b = (Parameters<RealType>::gamma - 1.0) *
                   (rho_and_E - 0.5 * rho * (v_b_x * v_b_x + v_b_y * v_b_y));

  return Tensor<1, 2 + dim, RealType>{
    RealType(0.0), p_b * n_x, p_b * n_y, RealType(0.0)
  };
}

template<unsigned int dim, typename RealType>
inline Tensor<1, 2 + dim, RealType>
inflow(const Tensor<1, 2 + dim, RealType>& interior_state,
       const Tensor<1, dim, RealType>& normal)
{
  // Grab values
  const auto rho = interior_state[0];
  const auto rho_and_u = interior_state[1];
  const auto rho_and_v = interior_state[2];
  const auto rho_and_E = interior_state[3];

  const auto n_x = normal[0];
  const auto n_y = normal[1];

  //

  return Tensor<1, 2 + dim, RealType>{
    rho, rho_and_u * n_x, rho_and_v * n_y, rho_and_E
  };
}

template<unsigned int dim, typename RealType>
inline Tensor<1, 2 + dim, RealType>
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

  return Tensor<1, 2 + dim, RealType>{
    rho, rho_and_u * n_x, rho_and_v * n_y, rho_and_E
  };
}

template<unsigned int dim, typename RealType>
inline Tensor<1, 2 + dim, RealType>
supersonic_outflow(const Tensor<1, 2 + dim, RealType>& interior_state,
                   const Tensor<1, dim, RealType>& normal)
{
  // Grab values
  const auto rho = interior_state[0];
  const auto rho_and_u = interior_state[1];
  const auto rho_and_v = interior_state[2];
  const auto rho_and_E = interior_state[3];

  const auto n_x = normal[0];
  const auto n_y = normal[1];

  return Tensor<1, 2 + dim, RealType>{
    rho, rho_and_u * n_x, rho_and_v * n_y, rho_and_E
  };
}

template<unsigned int dim, unsigned int degree, typename RealType>
class Solver
{
public:
  Solver() = default;

  void set_free_stream_initial_state(
    ElementData<dim, RealType>& element_scratch,
    RealType mach = 0.1) const
  {
    for (unsigned int i = 0; i < element_scratch.size(); ++i) {
      element_scratch.density[i] =
        Parameters<RealType>::p_0 / Parameters<RealType>::T_0_and_R;
      element_scratch.momentum_x[i] = 0.0;
      element_scratch.momentum_y[i] = 0.0;
      element_scratch.energy[i] = Parameters<RealType>::p_0 /
                                  (Parameters<RealType>::gamma - RealType(1.0));
    }
  }

  void compute_free_stream_residual(
    const InteriorFaceData<dim, RealType>& interior_face_scratch,
    const BoundaryFaceData<dim, RealType>& boundary_face_scratch,
    const PeriodicFaceData<dim, RealType>& periodic_face_scratch,
    ElementData<dim, RealType>& element_scratch) const
  {

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

      const auto flux = flux_roe(u_l, u_r, n);

      // Add the residual to the elements
      element_scratch.residual_density[e_l] += flux.first[0] * area;
      element_scratch.residual_momentum_x[e_l] += flux.first[1] * area;
      element_scratch.residual_momentum_y[e_l] += flux.first[2] * area;
      element_scratch.residual_energy[e_l] += flux.first[3] * area;

      element_scratch.residual_density[e_r] -= flux.first[0] * area;
      element_scratch.residual_momentum_x[e_r] -= flux.first[1] * area;
      element_scratch.residual_momentum_y[e_r] -= flux.first[2] * area;
      element_scratch.residual_energy[e_r] -= flux.first[3] * area;
    }

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

      const auto flux = flux_roe(u, u, n);

      // Add the residual to the elements
      element_scratch.residual_density[e] += flux.first[0] * area;
      element_scratch.residual_momentum_x[e] += flux.first[1] * area;
      element_scratch.residual_momentum_y[e] += flux.first[2] * area;
      element_scratch.residual_energy[e] += flux.first[3] * area;
    }

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

      const auto flux = flux_roe(u_l, u_r, n);

      // Add the residual to the elements
      element_scratch.residual_density[e_l] += flux.first[0] * area;
      element_scratch.residual_momentum_x[e_l] += flux.first[1] * area;
      element_scratch.residual_momentum_y[e_l] += flux.first[2] * area;
      element_scratch.residual_energy[e_l] += flux.first[3] * area;

      element_scratch.residual_density[e_r] -= flux.first[0] * area;
      element_scratch.residual_momentum_x[e_r] -= flux.first[1] * area;
      element_scratch.residual_momentum_y[e_r] -= flux.first[2] * area;
      element_scratch.residual_energy[e_r] -= flux.first[3] * area;
    }
  };

private:
};
