#pragma once

#include <Kokkos_Core.hpp>
#include <config.hpp>
#include <dof_handler.hpp>
#include <fe.hpp>
#include <triangulation.hpp>

template<unsigned int dim, unsigned int degree, typename RealType>
class CellIntegratorAdvection
{
public:
  static_assert(dim == 2);
  static_assert(degree < 4);

  static constexpr unsigned int n_dofs_per_cell =
    FE_DGLagrangeSimplex<dim, RealType>::n_dofs_per_cell(degree);

  static constexpr unsigned int n_q_points =
    QGaussSimplex<dim, RealType>::n_q_points(degree + 1);

  static constexpr unsigned int n_q_points_face =
    QGaussSimplex<dim - 1, RealType>::n_q_points(degree + 1);

  static constexpr unsigned int n_faces_per_cell =
    SimplexTopology<dim>::faces_per_cell;

  Kokkos::View<RealType**, Layout, DeviceMemSpace> JxW_;  // [cell, q]
  Kokkos::View<RealType***, Layout, DeviceMemSpace> phi_; // [cell, dof, q]
  Kokkos::View<RealType****, Layout, DeviceMemSpace>
    grad_phi_; // [cell, dof, q, dim]

  Kokkos::View<RealType***, Layout, DeviceMemSpace>
    face_JxW_; // [cell, face, q]
  Kokkos::View<RealType****, Layout, DeviceMemSpace>
    face_phi_; // [cell, face, dof, q]
  Kokkos::View<RealType****, Layout, DeviceMemSpace>
    face_normal_; // [cell, face, q, dim]

  Kokkos::View<uint32_t***, Layout, DeviceMemSpace>
    neighbor_indices_; // [cell, face, dof]

  CellIntegratorAdvection(const DoFHandler<dim, RealType>& dof_handler,
                          FEValues<dim, RealType>& fe_values,
                          FEFaceValues<dim, RealType>& fe_face_values)
  {
    ASSERT(fe_values.n_q_points() == n_q_points, "Degree mismatch");
    ASSERT(fe_face_values.n_q_points() == n_q_points_face, "Degree mismatch");
    ASSERT(dof_handler.n_dofs_per_cell() == n_dofs_per_cell, "Degree mismatch");
  }

  void precompute_geometry(const DoFHandler<dim, RealType>& dof_handler,
                           FEValues<dim, RealType>& fe_values,
                           FEFaceValues<dim, RealType>& fe_face_values)
  {
    const auto n_cells = dof_handler.n_cells();

    // Initialize the data
    JxW_ = Kokkos::View<RealType**, Layout, DeviceMemSpace>(
      "JxW", n_cells, n_q_points);
    phi_ = Kokkos::View<RealType***, Layout, DeviceMemSpace>(
      "phi", n_cells, n_dofs_per_cell, n_q_points);
    grad_phi_ = Kokkos::View<RealType****, Layout, DeviceMemSpace>(
      "grad_phi", n_cells, n_dofs_per_cell, n_q_points, dim);

    face_JxW_ = Kokkos::View<RealType***, Layout, DeviceMemSpace>(
      "face_JxW", n_cells, n_faces_per_cell, n_q_points_face);
    face_phi_ = Kokkos::View<RealType****, Layout, DeviceMemSpace>(
      "face_phi", n_cells, n_faces_per_cell, n_dofs_per_cell, n_q_points_face);
    face_normal_ = Kokkos::View<RealType****, Layout, DeviceMemSpace>(
      "face_normal", n_cells, n_faces_per_cell, n_q_points_face, dim);

    neighbor_indices_ = Kokkos::View<uint32_t***, Layout, DeviceMemSpace>(
      "neigh_idx", n_cells, n_faces_per_cell, n_dofs_per_cell);

    // Fill these on the host and copy over to device
    auto JxW_h = Kokkos::create_mirror_view(JxW_);
    auto phi_h = Kokkos::create_mirror_view(phi_);
    auto grad_phi_h = Kokkos::create_mirror_view(grad_phi_);
    auto face_JxW_h = Kokkos::create_mirror_view(face_JxW_);
    auto face_phi_h = Kokkos::create_mirror_view(face_phi_);
    auto face_normal_h = Kokkos::create_mirror_view(face_normal_);
    auto neighbor_indices_h = Kokkos::create_mirror_view(neighbor_indices_);
  }
};
