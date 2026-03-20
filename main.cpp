#include <Kokkos_Core.hpp>
#include <config.hpp>
#include <cstdint>
#include <data_out.hpp>
#include <dof_handler.hpp>
#include <fe.hpp>
#include <flux.hpp>
#include <matrix.hpp>
#include <parameters.hpp>
#include <read_gri.hpp>
#include <solve.hpp>
#include <stdexcept>
#include <timer.hpp>
#include <triangulation.hpp>
#include <vector.hpp>

enum BoundaryId
{
  inflow = 10,
  wall = 10,
};

template<unsigned int dim, typename RealType>
class EulerSolver
{
public:
  using VecDevice = Vector<RealType, DeviceMemSpace>;
  using VecHost = Vector<RealType, HostMemSpace>;

  EulerSolver(const DoFHandler<dim, RealType>& dof_handler,
              FEValues<dim, RealType>& fe_values,
              FEFaceValues<dim, RealType>& fe_face_values,
              unsigned int degree)
    : dof_handler_(dof_handler)
    , fe_values_(fe_values)
    , fe_face_values_(fe_face_values)
    , n_dofs_(dof_handler.n_dofs())
    , degree_(degree)
  {
    // Allocate state vectors and those that have the same dofs
    rho_ = VecDevice(n_dofs_);
    rho_u_ = VecDevice(n_dofs_);
    rho_v_ = VecDevice(n_dofs_);
    rho_E_ = VecDevice(n_dofs_);

    rho_old_ = VecDevice(n_dofs_);
    rho_u_old_ = VecDevice(n_dofs_);
    rho_v_old_ = VecDevice(n_dofs_);
    rho_E_old_ = VecDevice(n_dofs_);

    res_rho_ = VecDevice(n_dofs_);
    res_rho_u_ = VecDevice(n_dofs_);
    res_rho_v_ = VecDevice(n_dofs_);
    res_rho_E_ = VecDevice(n_dofs_);

    dt_ = VecDevice(n_dofs_);

    // Precompute geometries
    precompute_geometry();
  }

  template<typename InitFunc>
  void set_initial_condition(InitFunc&& f)
  {
    constexpr RealType gamma = Parameters<RealType>::gamma;
    constexpr RealType gm1 = gamma - RealType(1);

    auto rho_h = Kokkos::create_mirror_view(rho_.view());
    auto rho_u_h = Kokkos::create_mirror_view(rho_u_.view());
    auto rho_v_h = Kokkos::create_mirror_view(rho_v_.view());
    auto rho_E_h = Kokkos::create_mirror_view(rho_E_.view());

    std::vector<uint32_t> dof_ids;
    for (auto cell : dof_handler_.active_cell_range()) {
      cell.get_dof_indices(dof_ids);
      const auto ctr = cell.tria_cell.center();

      auto [rho0, u0, v0, p0] = f(ctr(0), ctr(1));
      const RealType rhoE0 =
        p0 / gm1 + RealType(0.5) * rho0 * (u0 * u0 + v0 * v0);

      rho_h(dof_ids[0]) = rho0;
      rho_u_h(dof_ids[0]) = rho0 * u0;
      rho_v_h(dof_ids[0]) = rho0 * v0;
      rho_E_h(dof_ids[0]) = rhoE0;
    }

    Kokkos::deep_copy(rho_.view(), rho_h);
    Kokkos::deep_copy(rho_u_.view(), rho_u_h);
    Kokkos::deep_copy(rho_v_.view(), rho_v_h);
    Kokkos::deep_copy(rho_E_.view(), rho_E_h);
  }

  const unsigned int degree_;
  const DoFHandler<dim, RealType>& dof_handler_;
  FEValues<dim, RealType>& fe_values_;
  FEFaceValues<dim, RealType>& fe_face_values_;

  uint32_t n_dofs_;
  uint32_t n_cells_;
  uint32_t n_interior_faces_;
  uint32_t n_periodic_faces_;
  uint32_t n_boundary_faces_;

  // Cell geometry
  Kokkos::View<RealType**, Layout, DeviceMemSpace> JxW_; // [n_cells, n_q]
  Kokkos::View<RealType***, Layout, DeviceMemSpace>
    q_point_; // [n_cells, n_q, dim]
  Kokkos::View<RealType***, Layout, DeviceMemSpace>
    phi_; // [n_cells, n_dofs_per_cell, n_q]
  Kokkos::View<RealType****, Layout, DeviceMemSpace>
    grad_phi_; // [n_cells, n_dofs_per_cell, n_q, dim]
  Kokkos::View<uint32_t**, Layout, DeviceMemSpace>
    cell_dofs_; // [n_cells, n_dofs_per_cell]

  // NOTE: Do we want to precompute cell area here?

  // Interior face geometry
  Kokkos::View<RealType**, Layout, DeviceMemSpace> interior_JxW_; // [face, q]
  Kokkos::View<RealType***, Layout, DeviceMemSpace>
    interior_q_point_; // [face, n_q, dim]
  Kokkos::View<RealType***, Layout, DeviceMemSpace>
    interior_normal_; // [face, q, dim]
  Kokkos::View<RealType***, Layout, DeviceMemSpace>
    interior_phi_L_; // [face, q, dof]
  Kokkos::View<RealType***, Layout, DeviceMemSpace>
    interior_phi_R_; // [face, q, dof]
  Kokkos::View<RealType****, Layout, DeviceMemSpace>
    interior_grad_phi_L_; // [face, q, dof, dim]
  Kokkos::View<RealType****, Layout, DeviceMemSpace>
    interior_grad_phi_R_; // [face, q, dof, dim]
  Kokkos::View<uint32_t**, Layout, DeviceMemSpace>
    interior_dofs_L_; // [face, dof]
  Kokkos::View<uint32_t**, Layout, DeviceMemSpace>
    interior_dofs_R_; // [face, dof]

  // Periodic face geometry - same structure as interior, just different
  // connectivity
  Kokkos::View<RealType**, Layout, DeviceMemSpace> periodic_JxW_; // [face, q]
  Kokkos::View<RealType***, Layout, DeviceMemSpace>
    periodic_q_point_; // [face, q, dim]
  Kokkos::View<RealType***, Layout, DeviceMemSpace>
    periodic_normal_; // [face, q, dim]
  Kokkos::View<RealType***, Layout, DeviceMemSpace>
    periodic_phi_L_; // [face, q, dof]
  Kokkos::View<RealType***, Layout, DeviceMemSpace>
    periodic_phi_R_; // [face, q, dof]
  Kokkos::View<RealType****, Layout, DeviceMemSpace>
    periodic_grad_phi_L_; // [face, q, dof, dim]
  Kokkos::View<RealType****, Layout, DeviceMemSpace>
    periodic_grad_phi_R_; // [face, q, dof, dim]
  Kokkos::View<uint32_t**, Layout, DeviceMemSpace>
    periodic_dofs_L_; // [face, dof]
  Kokkos::View<uint32_t**, Layout, DeviceMemSpace>
    periodic_dofs_R_; // [face, dof]

  // Boundary face geometry - one side only, plus BC tag and physical coords for
  // BC evaluation
  Kokkos::View<RealType**, Layout, DeviceMemSpace> boundary_JxW_; // [face, q]
  Kokkos::View<RealType***, Layout, DeviceMemSpace>
    boundary_q_point_; // [face, q, dim]
  Kokkos::View<RealType***, Layout, DeviceMemSpace>
    boundary_normal_; // [face, q, dim]
  Kokkos::View<RealType***, Layout, DeviceMemSpace>
    boundary_phi_; // [face, q, dof]
  Kokkos::View<RealType****, Layout, DeviceMemSpace>
    boundary_grad_phi_; // [face, q, dof, dim]
  Kokkos::View<uint32_t**, Layout, DeviceMemSpace>
    boundary_dofs_; // [face, dof]
  Kokkos::View<uint32_t*, Layout, DeviceMemSpace>
    boundary_id_; // [face] BC type

  // Current state - u^n
  VecDevice rho_, rho_u_, rho_v_, rho_E_;

  // Old state - u^n-1
  VecDevice rho_old_, rho_u_old_, rho_v_old_, rho_E_old_;

  // State residuals
  VecDevice res_rho_, res_rho_u_, res_rho_v_, res_rho_E_;

  // Local timesteps
  VecDevice dt_;

  void zero_residuals()
  {
    res_rho_ = RealType(0);
    res_rho_u_ = RealType(0);
    res_rho_v_ = RealType(0);
    res_rho_E_ = RealType(0);
  }

  void precompute_geometry()
  {
    const auto n_cells = dof_handler_.n_cells();
    const auto n_dofs_per_cell = dof_handler_.n_dofs_per_cell();
    const auto n_q_points = fe_values_.n_q_points();
    const auto n_q_points_face = fe_face_values_.n_q_points();

    std::cout << "Number of cells " << n_cells << std::endl;
    std::cout << "Number of DoFs per cell " << n_dofs_per_cell << std::endl;
    std::cout << "Number of q points " << n_q_points << std::endl;
    std::cout << "Number of q points face " << n_q_points_face << std::endl;
    std::cout << std::endl;

    // Count the number of faces
    n_cells_ = n_cells;
    n_interior_faces_ = 0;
    n_periodic_faces_ = 0;
    n_boundary_faces_ = 0;

    for (auto cell : dof_handler_.active_cell_range()) {
      for (unsigned int lf = 0; lf < SimplexTopology<dim>::faces_per_cell;
           ++lf) {
        auto face = cell.face(lf);
        if (face.at_boundary()) {
          n_boundary_faces_++;
        } else if (face.is_periodic()) {
          // Only count if this face's index is less than its periodic partner
          if (face.index < face.periodic_neighbor_index())
            n_periodic_faces_++;
        } else {
          // Only count from the owner cell's perspective
          if (face.owner_index() == cell.index())
            n_interior_faces_++;
        }
      }
    }

    std::cout << "Number of interior faces " << n_interior_faces_ << std::endl;
    std::cout << "Number of periodic faces " << n_periodic_faces_ << std::endl;
    std::cout << "Number of boundary faces " << n_boundary_faces_ << std::endl;
    std::cout << std::endl;

    // Allocate the cell geometry and copy to device
    auto JxW_h = Kokkos::View<RealType**, Layout, HostMemSpace>(
      "JxW_h", n_cells, n_q_points);
    auto q_point_h = Kokkos::View<RealType***, Layout, HostMemSpace>(
      "q_point_h", n_cells, n_q_points, dim);
    auto phi_h = Kokkos::View<RealType***, Layout, HostMemSpace>(
      "phi_h", n_cells, n_dofs_per_cell, n_q_points);
    auto grad_phi_h = Kokkos::View<RealType****, Layout, HostMemSpace>(
      "grad_phi_h", n_cells, n_dofs_per_cell, n_q_points, dim);
    auto cell_dofs_h = Kokkos::View<uint32_t**, Layout, HostMemSpace>(
      "cell_dofs_h", n_cells, n_dofs_per_cell);

    std::vector<uint32_t> dof_indices;
    for (auto cell : dof_handler_.active_cell_range()) {
      const auto k = cell.index();
      fe_values_.reinit(cell);
      cell.get_dof_indices(dof_indices);

      for (unsigned int q = 0; q < n_q_points; ++q) {
        JxW_h(k, q) = fe_values_.JxW(q);

        auto p = fe_values_.q_point(q);
        for (unsigned int d = 0; d < dim; ++d)
          q_point_h(k, q, d) = p(d);

        for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
          phi_h(k, i, q) = fe_values_.shape_value(i, q);
          auto grad = fe_values_.shape_gradient(i, q);
          for (unsigned int d = 0; d < dim; ++d)
            grad_phi_h(k, i, q, d) = grad(d);
        }
      }

      for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
        cell_dofs_h(k, i) = dof_indices[i];
    }

    JxW_ = Kokkos::View<RealType**, Layout, DeviceMemSpace>(
      "JxW", n_cells, n_q_points);
    q_point_ = Kokkos::View<RealType***, Layout, DeviceMemSpace>(
      "q_point", n_cells, n_q_points, dim);
    phi_ = Kokkos::View<RealType***, Layout, DeviceMemSpace>(
      "phi", n_cells, n_dofs_per_cell, n_q_points);
    grad_phi_ = Kokkos::View<RealType****, Layout, DeviceMemSpace>(
      "grad_phi", n_cells, n_dofs_per_cell, n_q_points, dim);
    cell_dofs_ = Kokkos::View<uint32_t**, Layout, DeviceMemSpace>(
      "cell_dofs", n_cells, n_dofs_per_cell);

    Kokkos::deep_copy(JxW_, JxW_h);
    Kokkos::deep_copy(q_point_, q_point_h);
    Kokkos::deep_copy(phi_, phi_h);
    Kokkos::deep_copy(grad_phi_, grad_phi_h);
    Kokkos::deep_copy(cell_dofs_, cell_dofs_h);

    // Allocate the face geometry and copy to device
    auto interior_JxW_h = Kokkos::View<RealType**, Layout, HostMemSpace>(
      "interior_JxW_h", n_interior_faces_, n_q_points_face);
    auto interior_q_point_h = Kokkos::View<RealType***, Layout, HostMemSpace>(
      "interior_q_point_h", n_interior_faces_, n_q_points_face, dim);
    auto interior_normal_h = Kokkos::View<RealType***, Layout, HostMemSpace>(
      "interior_normal_h", n_interior_faces_, n_q_points_face, dim);
    auto interior_phi_L_h = Kokkos::View<RealType***, Layout, HostMemSpace>(
      "interior_phi_L_h", n_interior_faces_, n_q_points_face, n_dofs_per_cell);
    auto interior_phi_R_h = Kokkos::View<RealType***, Layout, HostMemSpace>(
      "interior_phi_R_h", n_interior_faces_, n_q_points_face, n_dofs_per_cell);
    auto interior_dofs_L_h = Kokkos::View<uint32_t**, Layout, HostMemSpace>(
      "interior_dofs_L_h", n_interior_faces_, n_dofs_per_cell);
    auto interior_dofs_R_h = Kokkos::View<uint32_t**, Layout, HostMemSpace>(
      "interior_dofs_R_h", n_interior_faces_, n_dofs_per_cell);

    auto periodic_JxW_h = Kokkos::View<RealType**, Layout, HostMemSpace>(
      "periodic_JxW_h", n_periodic_faces_, n_q_points_face);
    auto periodic_q_point_h = Kokkos::View<RealType***, Layout, HostMemSpace>(
      "periodic_q_point_h", n_periodic_faces_, n_q_points_face, dim);
    auto periodic_normal_h = Kokkos::View<RealType***, Layout, HostMemSpace>(
      "periodic_normal_h", n_periodic_faces_, n_q_points_face, dim);
    auto periodic_phi_L_h = Kokkos::View<RealType***, Layout, HostMemSpace>(
      "periodic_phi_L_h", n_periodic_faces_, n_q_points_face, n_dofs_per_cell);
    auto periodic_phi_R_h = Kokkos::View<RealType***, Layout, HostMemSpace>(
      "periodic_phi_R_h", n_periodic_faces_, n_q_points_face, n_dofs_per_cell);
    auto periodic_dofs_L_h = Kokkos::View<uint32_t**, Layout, HostMemSpace>(
      "periodic_dofs_L_h", n_periodic_faces_, n_dofs_per_cell);
    auto periodic_dofs_R_h = Kokkos::View<uint32_t**, Layout, HostMemSpace>(
      "periodic_dofs_R_h", n_periodic_faces_, n_dofs_per_cell);

    auto boundary_JxW_h = Kokkos::View<RealType**, Layout, HostMemSpace>(
      "boundary_JxW_h", n_boundary_faces_, n_q_points_face);
    auto boundary_q_point_h = Kokkos::View<RealType***, Layout, HostMemSpace>(
      "boundary_q_point_h", n_boundary_faces_, n_q_points_face, dim);
    auto boundary_normal_h = Kokkos::View<RealType***, Layout, HostMemSpace>(
      "boundary_normal_h", n_boundary_faces_, n_q_points_face, dim);
    auto boundary_phi_h = Kokkos::View<RealType***, Layout, HostMemSpace>(
      "boundary_phi_h", n_boundary_faces_, n_q_points_face, n_dofs_per_cell);
    auto boundary_dofs_h = Kokkos::View<uint32_t**, Layout, HostMemSpace>(
      "boundary_dofs_h", n_boundary_faces_, n_dofs_per_cell);
    auto boundary_id_h = Kokkos::View<uint32_t*, Layout, HostMemSpace>(
      "boundary_id_h", n_boundary_faces_);

    uint32_t interior_face_idx = 0;
    uint32_t periodic_face_idx = 0;
    uint32_t boundary_face_idx = 0;

    std::vector<uint32_t> neighbor_dof_indices;

    for (auto cell : dof_handler_.active_cell_range()) {
      cell.get_dof_indices(dof_indices);
      for (unsigned int lf = 0; lf < SimplexTopology<dim>::faces_per_cell;
           ++lf) {
        fe_face_values_.reinit(cell, lf);
        auto face = cell.face(lf);

        if (face.at_boundary()) {
          // Boundary: only one cell owns this face
          const uint32_t f = boundary_face_idx++;
          boundary_id_h(f) = face.boundary_id();
          for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
            boundary_dofs_h(f, i) = dof_indices[i];
          for (unsigned int q = 0; q < n_q_points_face; ++q) {
            boundary_JxW_h(f, q) = fe_face_values_.JxW(q);
            auto p = fe_face_values_.q_point(q);
            auto n = fe_face_values_.normal(q);
            for (unsigned int d = 0; d < dim; ++d) {
              boundary_q_point_h(f, q, d) = p(d);
              boundary_normal_h(f, q, d) = n(d);
            }
            for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
              boundary_phi_h(f, q, i) = fe_face_values_.shape_value(i, q);
          }

        } else if (face.is_periodic()) {
          // Only process from the owner cell
          if (face.owner_index() == cell.index()) {
            const uint32_t f = periodic_face_idx++;
            auto neighbor_cell = cell.periodic_neighbor(lf);
            neighbor_cell.get_dof_indices(neighbor_dof_indices);

            for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
              periodic_dofs_L_h(f, i) = dof_indices[i];
              periodic_dofs_R_h(f, i) = neighbor_dof_indices[i];
            }

            // Left side (owner cell)
            for (unsigned int q = 0; q < n_q_points_face; ++q) {
              periodic_JxW_h(f, q) = fe_face_values_.JxW(q);
              auto p = fe_face_values_.q_point(q);
              auto n = fe_face_values_.normal(q);
              for (unsigned int d = 0; d < dim; ++d) {
                periodic_q_point_h(f, q, d) = p(d);
                periodic_normal_h(f, q, d) = n(d);
              }
              for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
                periodic_phi_L_h(f, q, i) = fe_face_values_.shape_value(i, q);
            }

            // Right side (periodic neighbor)
            unsigned int neighbor_lf = face.periodic_neighbor_face_index();
            fe_face_values_.reinit(neighbor_cell, neighbor_lf);
            for (unsigned int q = 0; q < n_q_points_face; ++q)
              for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
                periodic_phi_R_h(f, q, i) = fe_face_values_.shape_value(i, q);
          }

        } else {
          // Only process from the owner cell
          if (face.owner_index() == cell.index()) {
            const uint32_t f = interior_face_idx++;
            auto neighbor_cell = cell.neighbor(lf);
            neighbor_cell.get_dof_indices(neighbor_dof_indices);

            for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
              interior_dofs_L_h(f, i) = dof_indices[i];
              interior_dofs_R_h(f, i) = neighbor_dof_indices[i];
            }

            // Left side (owner cell)
            for (unsigned int q = 0; q < n_q_points_face; ++q) {
              interior_JxW_h(f, q) = fe_face_values_.JxW(q);
              auto p = fe_face_values_.q_point(q);
              auto n = fe_face_values_.normal(q);
              for (unsigned int d = 0; d < dim; ++d) {
                interior_q_point_h(f, q, d) = p(d);
                interior_normal_h(f, q, d) = n(d);
              }
              for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
                interior_phi_L_h(f, q, i) = fe_face_values_.shape_value(i, q);
            }

            // Right side (neighbor cell)
            unsigned int neighbor_lf = cell.neighbor_face_index(lf);
            fe_face_values_.reinit(neighbor_cell, neighbor_lf);
            for (unsigned int q = 0; q < n_q_points_face; ++q)
              for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
                interior_phi_R_h(f, q, i) = fe_face_values_.shape_value(i, q);
          }
        }
      }
    }

    interior_JxW_ = Kokkos::View<RealType**, Layout, DeviceMemSpace>(
      "interior_JxW", n_interior_faces_, n_q_points_face);
    interior_q_point_ = Kokkos::View<RealType***, Layout, DeviceMemSpace>(
      "interior_q_point", n_interior_faces_, n_q_points_face, dim);
    interior_normal_ = Kokkos::View<RealType***, Layout, DeviceMemSpace>(
      "interior_normal", n_interior_faces_, n_q_points_face, dim);
    interior_phi_L_ = Kokkos::View<RealType***, Layout, DeviceMemSpace>(
      "interior_phi_L", n_interior_faces_, n_q_points_face, n_dofs_per_cell);
    interior_phi_R_ = Kokkos::View<RealType***, Layout, DeviceMemSpace>(
      "interior_phi_R", n_interior_faces_, n_q_points_face, n_dofs_per_cell);
    interior_dofs_L_ = Kokkos::View<uint32_t**, Layout, DeviceMemSpace>(
      "interior_dofs_L", n_interior_faces_, n_dofs_per_cell);
    interior_dofs_R_ = Kokkos::View<uint32_t**, Layout, DeviceMemSpace>(
      "interior_dofs_R", n_interior_faces_, n_dofs_per_cell);

    periodic_JxW_ = Kokkos::View<RealType**, Layout, DeviceMemSpace>(
      "periodic_JxW", n_periodic_faces_, n_q_points_face);
    periodic_q_point_ = Kokkos::View<RealType***, Layout, DeviceMemSpace>(
      "periodic_q_point", n_periodic_faces_, n_q_points_face, dim);
    periodic_normal_ = Kokkos::View<RealType***, Layout, DeviceMemSpace>(
      "periodic_normal", n_periodic_faces_, n_q_points_face, dim);
    periodic_phi_L_ = Kokkos::View<RealType***, Layout, DeviceMemSpace>(
      "periodic_phi_L", n_periodic_faces_, n_q_points_face, n_dofs_per_cell);
    periodic_phi_R_ = Kokkos::View<RealType***, Layout, DeviceMemSpace>(
      "periodic_phi_R", n_periodic_faces_, n_q_points_face, n_dofs_per_cell);
    periodic_dofs_L_ = Kokkos::View<uint32_t**, Layout, DeviceMemSpace>(
      "periodic_dofs_L", n_periodic_faces_, n_dofs_per_cell);
    periodic_dofs_R_ = Kokkos::View<uint32_t**, Layout, DeviceMemSpace>(
      "periodic_dofs_R", n_periodic_faces_, n_dofs_per_cell);

    boundary_JxW_ = Kokkos::View<RealType**, Layout, DeviceMemSpace>(
      "boundary_JxW", n_boundary_faces_, n_q_points_face);
    boundary_q_point_ = Kokkos::View<RealType***, Layout, DeviceMemSpace>(
      "boundary_q_point", n_boundary_faces_, n_q_points_face, dim);
    boundary_normal_ = Kokkos::View<RealType***, Layout, DeviceMemSpace>(
      "boundary_normal", n_boundary_faces_, n_q_points_face, dim);
    boundary_phi_ = Kokkos::View<RealType***, Layout, DeviceMemSpace>(
      "boundary_phi", n_boundary_faces_, n_q_points_face, n_dofs_per_cell);
    boundary_dofs_ = Kokkos::View<uint32_t**, Layout, DeviceMemSpace>(
      "boundary_dofs", n_boundary_faces_, n_dofs_per_cell);
    boundary_id_ = Kokkos::View<uint32_t*, Layout, DeviceMemSpace>(
      "boundary_id", n_boundary_faces_);

    Kokkos::deep_copy(interior_JxW_, interior_JxW_h);
    Kokkos::deep_copy(interior_q_point_, interior_q_point_h);
    Kokkos::deep_copy(interior_normal_, interior_normal_h);
    Kokkos::deep_copy(interior_phi_L_, interior_phi_L_h);
    Kokkos::deep_copy(interior_phi_R_, interior_phi_R_h);
    Kokkos::deep_copy(interior_dofs_L_, interior_dofs_L_h);
    Kokkos::deep_copy(interior_dofs_R_, interior_dofs_R_h);

    Kokkos::deep_copy(periodic_JxW_, periodic_JxW_h);
    Kokkos::deep_copy(periodic_q_point_, periodic_q_point_h);
    Kokkos::deep_copy(periodic_normal_, periodic_normal_h);
    Kokkos::deep_copy(periodic_phi_L_, periodic_phi_L_h);
    Kokkos::deep_copy(periodic_phi_R_, periodic_phi_R_h);
    Kokkos::deep_copy(periodic_dofs_L_, periodic_dofs_L_h);
    Kokkos::deep_copy(periodic_dofs_R_, periodic_dofs_R_h);

    Kokkos::deep_copy(boundary_JxW_, boundary_JxW_h);
    Kokkos::deep_copy(boundary_q_point_, boundary_q_point_h);
    Kokkos::deep_copy(boundary_normal_, boundary_normal_h);
    Kokkos::deep_copy(boundary_phi_, boundary_phi_h);
    Kokkos::deep_copy(boundary_dofs_, boundary_dofs_h);
    Kokkos::deep_copy(boundary_id_, boundary_id_h);
  }

  void print_geometry_debug() const
  {
    std::cout << "=== Geometry Debug ===" << std::endl;
    std::cout << "n_cells=" << n_cells_ << " n_interior=" << n_interior_faces_
              << " n_periodic=" << n_periodic_faces_
              << " n_boundary=" << n_boundary_faces_ << std::endl;

    // Cell DOFs
    std::cout << "\n--- Cell DOFs ---" << std::endl;
    auto cell_dofs_h = Kokkos::create_mirror_view(cell_dofs_);
    Kokkos::deep_copy(cell_dofs_h, cell_dofs_);
    for (unsigned int k = 0; k < n_cells_; ++k) {
      std::cout << "Cell " << k << " dofs: ";
      for (unsigned int i = 0; i < dof_handler_.n_dofs_per_cell(); ++i)
        std::cout << cell_dofs_h(k, i) << " ";
      std::cout << std::endl;
    }

    // Interior faces
    std::cout << "\n--- Interior Faces ---" << std::endl;
    auto int_dofs_L_h = Kokkos::create_mirror_view(interior_dofs_L_);
    auto int_dofs_R_h = Kokkos::create_mirror_view(interior_dofs_R_);
    auto int_normal_h = Kokkos::create_mirror_view(interior_normal_);
    auto int_JxW_h = Kokkos::create_mirror_view(interior_JxW_);
    auto int_qpt_h = Kokkos::create_mirror_view(interior_q_point_);
    Kokkos::deep_copy(int_dofs_L_h, interior_dofs_L_);
    Kokkos::deep_copy(int_dofs_R_h, interior_dofs_R_);
    Kokkos::deep_copy(int_normal_h, interior_normal_);
    Kokkos::deep_copy(int_JxW_h, interior_JxW_);
    Kokkos::deep_copy(int_qpt_h, interior_q_point_);

    for (unsigned int f = 0; f < n_interior_faces_; ++f) {
      std::cout << "Interior face " << f << std::endl;
      std::cout << "  dofs_L: ";
      for (unsigned int i = 0; i < dof_handler_.n_dofs_per_cell(); ++i)
        std::cout << int_dofs_L_h(f, i) << " ";
      std::cout << std::endl;
      std::cout << "  dofs_R: ";
      for (unsigned int i = 0; i < dof_handler_.n_dofs_per_cell(); ++i)
        std::cout << int_dofs_R_h(f, i) << " ";
      std::cout << std::endl;
      for (unsigned int q = 0; q < interior_JxW_.extent(1); ++q) {
        std::cout << "  q=" << q << " JxW=" << int_JxW_h(f, q) << " p=("
                  << int_qpt_h(f, q, 0) << "," << int_qpt_h(f, q, 1) << ")"
                  << " n=(" << int_normal_h(f, q, 0) << ","
                  << int_normal_h(f, q, 1) << ")" << std::endl;
      }
    }

    // Periodic faces
    std::cout << "\n--- Periodic Faces ---" << std::endl;
    auto per_dofs_L_h = Kokkos::create_mirror_view(periodic_dofs_L_);
    auto per_dofs_R_h = Kokkos::create_mirror_view(periodic_dofs_R_);
    auto per_normal_h = Kokkos::create_mirror_view(periodic_normal_);
    auto per_JxW_h = Kokkos::create_mirror_view(periodic_JxW_);
    auto per_qpt_h = Kokkos::create_mirror_view(periodic_q_point_);
    Kokkos::deep_copy(per_dofs_L_h, periodic_dofs_L_);
    Kokkos::deep_copy(per_dofs_R_h, periodic_dofs_R_);
    Kokkos::deep_copy(per_normal_h, periodic_normal_);
    Kokkos::deep_copy(per_JxW_h, periodic_JxW_);
    Kokkos::deep_copy(per_qpt_h, periodic_q_point_);

    for (unsigned int f = 0; f < n_periodic_faces_; ++f) {
      std::cout << "Periodic face " << f << std::endl;
      std::cout << "  dofs_L: ";
      for (unsigned int i = 0; i < dof_handler_.n_dofs_per_cell(); ++i)
        std::cout << per_dofs_L_h(f, i) << " ";
      std::cout << std::endl;
      std::cout << "  dofs_R: ";
      for (unsigned int i = 0; i < dof_handler_.n_dofs_per_cell(); ++i)
        std::cout << per_dofs_R_h(f, i) << " ";
      std::cout << std::endl;
      for (unsigned int q = 0; q < periodic_JxW_.extent(1); ++q) {
        std::cout << "  q=" << q << " JxW=" << per_JxW_h(f, q) << " p=("
                  << per_qpt_h(f, q, 0) << "," << per_qpt_h(f, q, 1) << ")"
                  << " n=(" << per_normal_h(f, q, 0) << ","
                  << per_normal_h(f, q, 1) << ")" << std::endl;
      }
    }

    // Boundary faces
    std::cout << "\n--- Boundary Faces ---" << std::endl;
    auto bnd_dofs_h = Kokkos::create_mirror_view(boundary_dofs_);
    auto bnd_normal_h = Kokkos::create_mirror_view(boundary_normal_);
    auto bnd_JxW_h = Kokkos::create_mirror_view(boundary_JxW_);
    auto bnd_qpt_h = Kokkos::create_mirror_view(boundary_q_point_);
    auto bnd_id_h = Kokkos::create_mirror_view(boundary_id_);
    Kokkos::deep_copy(bnd_dofs_h, boundary_dofs_);
    Kokkos::deep_copy(bnd_normal_h, boundary_normal_);
    Kokkos::deep_copy(bnd_JxW_h, boundary_JxW_);
    Kokkos::deep_copy(bnd_qpt_h, boundary_q_point_);
    Kokkos::deep_copy(bnd_id_h, boundary_id_);

    for (unsigned int f = 0; f < n_boundary_faces_; ++f) {
      std::cout << "Boundary face " << f << " id=" << bnd_id_h(f) << std::endl;
      std::cout << "  dofs: ";
      for (unsigned int i = 0; i < dof_handler_.n_dofs_per_cell(); ++i)
        std::cout << bnd_dofs_h(f, i) << " ";
      std::cout << std::endl;
      for (unsigned int q = 0; q < boundary_JxW_.extent(1); ++q) {
        std::cout << "  q=" << q << " JxW=" << bnd_JxW_h(f, q) << " p=("
                  << bnd_qpt_h(f, q, 0) << "," << bnd_qpt_h(f, q, 1) << ")"
                  << " n=(" << bnd_normal_h(f, q, 0) << ","
                  << bnd_normal_h(f, q, 1) << ")" << std::endl;
      }
    }
  }

  void compute_volume_residual()
  {
    const auto n_dofs_per_cell = dof_handler_.n_dofs_per_cell();
    const auto n_q_points = fe_values_.n_q_points();

    auto phi = phi_;
    auto grad_phi = grad_phi_;
    auto JxW = JxW_;
    auto indices = cell_dofs_;

    auto d_rho = rho_.view();
    auto d_rho_u = rho_u_.view();
    auto d_rho_v = rho_v_.view();
    auto d_rho_E = rho_E_.view();

    auto d_res_rho = res_rho_.view();
    auto d_res_rho_u = res_rho_u_.view();
    auto d_res_rho_v = res_rho_v_.view();
    auto d_res_rho_E = res_rho_E_.view();

    Kokkos::parallel_for(
      "volume_residual",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
        { 0, 0 }, { (int)n_cells_, (int)n_dofs_per_cell }),
      KOKKOS_LAMBDA(int k, int i) {
        RealType local_res_rho = 0;
        RealType local_res_rho_u = 0;
        RealType local_res_rho_v = 0;
        RealType local_res_rho_E = 0;

        for (unsigned int q = 0; q < n_q_points; ++q) {
          // Reconstruct conservative state at quad point
          RealType rho = 0;
          RealType rho_u = 0;
          RealType rho_v = 0;
          RealType rho_E = 0;
          for (unsigned int j = 0; j < n_dofs_per_cell; ++j) {
            const RealType phi_j = phi(k, j, q);
            const uint32_t dof_j = indices(k, j);
            rho += d_rho(dof_j) * phi_j;
            rho_u += d_rho_u(dof_j) * phi_j;
            rho_v += d_rho_v(dof_j) * phi_j;
            rho_E += d_rho_E(dof_j) * phi_j;
          }

          // Compute primitive variables
          RealType u = 0, v = 0, p = 0;
          Flux<RealType>::conservative_to_primitive(
            rho, rho_u, rho_v, rho_E, u, v, p);

          // x-direction flux
          RealType Fx_rho, Fx_rho_u, Fx_rho_v, Fx_rho_E;
          Flux<RealType>::euler_flux(rho,
                                     u,
                                     v,
                                     p,
                                     rho_E,
                                     RealType(1),
                                     RealType(0),
                                     Fx_rho,
                                     Fx_rho_u,
                                     Fx_rho_v,
                                     Fx_rho_E);

          // y-direction flux
          RealType Fy_rho, Fy_rho_u, Fy_rho_v, Fy_rho_E;
          Flux<RealType>::euler_flux(rho,
                                     u,
                                     v,
                                     p,
                                     rho_E,
                                     RealType(0),
                                     RealType(1),
                                     Fy_rho,
                                     Fy_rho_u,
                                     Fy_rho_v,
                                     Fy_rho_E);

          // Accumulate: R_i -= (∇φ_i · F) * JxW
          const RealType jxw = JxW(k, q);
          const RealType dphi_dx = grad_phi(k, i, q, 0);
          const RealType dphi_dy = grad_phi(k, i, q, 1);

          local_res_rho -= (dphi_dx * Fx_rho + dphi_dy * Fy_rho) * jxw;
          local_res_rho_u -= (dphi_dx * Fx_rho_u + dphi_dy * Fy_rho_u) * jxw;
          local_res_rho_v -= (dphi_dx * Fx_rho_v + dphi_dy * Fy_rho_v) * jxw;
          local_res_rho_E -= (dphi_dx * Fx_rho_E + dphi_dy * Fy_rho_E) * jxw;
        }

        // Scatter to global residual
        const uint32_t dof_i = indices(k, i);
        Kokkos::atomic_add(&d_res_rho(dof_i), local_res_rho);
        Kokkos::atomic_add(&d_res_rho_u(dof_i), local_res_rho_u);
        Kokkos::atomic_add(&d_res_rho_v(dof_i), local_res_rho_v);
        Kokkos::atomic_add(&d_res_rho_E(dof_i), local_res_rho_E);
      });
    Kokkos::fence();
  }

  void compute_face_residual()
  {
    compute_interior_face_residual();
    compute_periodic_face_residual();
    compute_boundary_face_residual();
  }

  void compute_interior_face_residual()
  {
    const auto n_dofs_per_cell = dof_handler_.n_dofs_per_cell();
    const auto n_q_face = fe_face_values_.n_q_points();

    auto phi_L = interior_phi_L_;
    auto phi_R = interior_phi_R_;
    auto JxW = interior_JxW_;
    auto normal = interior_normal_;
    auto dofs_L = interior_dofs_L_;
    auto dofs_R = interior_dofs_R_;

    auto d_rho = rho_.view();
    auto d_rho_u = rho_u_.view();
    auto d_rho_v = rho_v_.view();
    auto d_rho_E = rho_E_.view();

    auto d_res_rho = res_rho_.view();
    auto d_res_rho_u = res_rho_u_.view();
    auto d_res_rho_v = res_rho_v_.view();
    auto d_res_rho_E = res_rho_E_.view();

    Kokkos::parallel_for(
      "interior_face_residual", n_interior_faces_, KOKKOS_LAMBDA(int f) {
        for (unsigned int q = 0; q < n_q_face; ++q) {
          const RealType nx = normal(f, q, 0);
          const RealType ny = normal(f, q, 1);
          const RealType jxw = JxW(f, q);

          // Reconstruct left state
          RealType rho_L = 0, rho_u_L = 0, rho_v_L = 0, rho_E_L = 0;
          for (unsigned int j = 0; j < n_dofs_per_cell; ++j) {
            const RealType phi_j = phi_L(f, q, j);
            const uint32_t dof_j = dofs_L(f, j);
            rho_L += d_rho(dof_j) * phi_j;
            rho_u_L += d_rho_u(dof_j) * phi_j;
            rho_v_L += d_rho_v(dof_j) * phi_j;
            rho_E_L += d_rho_E(dof_j) * phi_j;
          }

          // Reconstruct right state
          RealType rho_R = 0, rho_u_R = 0, rho_v_R = 0, rho_E_R = 0;
          for (unsigned int j = 0; j < n_dofs_per_cell; ++j) {
            const RealType phi_j = phi_R(f, q, j);
            const uint32_t dof_j = dofs_R(f, j);
            rho_R += d_rho(dof_j) * phi_j;
            rho_u_R += d_rho_u(dof_j) * phi_j;
            rho_v_R += d_rho_v(dof_j) * phi_j;
            rho_E_R += d_rho_E(dof_j) * phi_j;
          }

          // Numerical flux
          RealType flux_rho, flux_rho_u, flux_rho_v, flux_rho_E, smag;
          Flux<RealType>::roe_flux(rho_L,
                                   rho_u_L,
                                   rho_v_L,
                                   rho_E_L,
                                   rho_R,
                                   rho_u_R,
                                   rho_v_R,
                                   rho_E_R,
                                   nx,
                                   ny,
                                   flux_rho,
                                   flux_rho_u,
                                   flux_rho_v,
                                   flux_rho_E,
                                   smag);

          // Scatter to left cell (+flux) and right cell (-flux)
          for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
            const RealType phi_i_L = phi_L(f, q, i) * jxw;
            const RealType phi_i_R = phi_R(f, q, i) * jxw;
            const uint32_t dof_i_L = dofs_L(f, i);
            const uint32_t dof_i_R = dofs_R(f, i);

            Kokkos::atomic_add(&d_res_rho(dof_i_L), phi_i_L * flux_rho);
            Kokkos::atomic_add(&d_res_rho_u(dof_i_L), phi_i_L * flux_rho_u);
            Kokkos::atomic_add(&d_res_rho_v(dof_i_L), phi_i_L * flux_rho_v);
            Kokkos::atomic_add(&d_res_rho_E(dof_i_L), phi_i_L * flux_rho_E);

            Kokkos::atomic_add(&d_res_rho(dof_i_R), -phi_i_R * flux_rho);
            Kokkos::atomic_add(&d_res_rho_u(dof_i_R), -phi_i_R * flux_rho_u);
            Kokkos::atomic_add(&d_res_rho_v(dof_i_R), -phi_i_R * flux_rho_v);
            Kokkos::atomic_add(&d_res_rho_E(dof_i_R), -phi_i_R * flux_rho_E);
          }
        }
      });
    Kokkos::fence();
  }

  void compute_periodic_face_residual()
  {
    // Identical structure to interior — just uses periodic_* views
    const auto n_dofs_per_cell = dof_handler_.n_dofs_per_cell();
    const auto n_q_face = fe_face_values_.n_q_points();

    auto phi_L = periodic_phi_L_;
    auto phi_R = periodic_phi_R_;
    auto JxW = periodic_JxW_;
    auto normal = periodic_normal_;
    auto dofs_L = periodic_dofs_L_;
    auto dofs_R = periodic_dofs_R_;

    auto d_rho = rho_.view();
    auto d_rho_u = rho_u_.view();
    auto d_rho_v = rho_v_.view();
    auto d_rho_E = rho_E_.view();

    auto d_res_rho = res_rho_.view();
    auto d_res_rho_u = res_rho_u_.view();
    auto d_res_rho_v = res_rho_v_.view();
    auto d_res_rho_E = res_rho_E_.view();

    Kokkos::parallel_for(
      "periodic_face_residual", n_periodic_faces_, KOKKOS_LAMBDA(int f) {
        for (unsigned int q = 0; q < n_q_face; ++q) {
          const RealType nx = normal(f, q, 0);
          const RealType ny = normal(f, q, 1);
          const RealType jxw = JxW(f, q);

          // Reconstruct left state
          RealType rho_L = 0, rho_u_L = 0, rho_v_L = 0, rho_E_L = 0;
          for (unsigned int j = 0; j < n_dofs_per_cell; ++j) {
            const RealType phi_j = phi_L(f, q, j);
            const uint32_t dof_j = dofs_L(f, j);
            rho_L += d_rho(dof_j) * phi_j;
            rho_u_L += d_rho_u(dof_j) * phi_j;
            rho_v_L += d_rho_v(dof_j) * phi_j;
            rho_E_L += d_rho_E(dof_j) * phi_j;
          }

          // Reconstruct right state
          RealType rho_R = 0, rho_u_R = 0, rho_v_R = 0, rho_E_R = 0;
          for (unsigned int j = 0; j < n_dofs_per_cell; ++j) {
            const RealType phi_j = phi_R(f, q, j);
            const uint32_t dof_j = dofs_R(f, j);
            rho_R += d_rho(dof_j) * phi_j;
            rho_u_R += d_rho_u(dof_j) * phi_j;
            rho_v_R += d_rho_v(dof_j) * phi_j;
            rho_E_R += d_rho_E(dof_j) * phi_j;
          }

          // Numerical flux
          RealType flux_rho, flux_rho_u, flux_rho_v, flux_rho_E, smag;
          Flux<RealType>::roe_flux(rho_L,
                                   rho_u_L,
                                   rho_v_L,
                                   rho_E_L,
                                   rho_R,
                                   rho_u_R,
                                   rho_v_R,
                                   rho_E_R,
                                   nx,
                                   ny,
                                   flux_rho,
                                   flux_rho_u,
                                   flux_rho_v,
                                   flux_rho_E,
                                   smag);

          // Scatter to left cell (+flux) and right cell (-flux)
          for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
            const RealType phi_i_L = phi_L(f, q, i) * jxw;
            const RealType phi_i_R = phi_R(f, q, i) * jxw;
            const uint32_t dof_i_L = dofs_L(f, i);
            const uint32_t dof_i_R = dofs_R(f, i);

            Kokkos::atomic_add(&d_res_rho(dof_i_L), phi_i_L * flux_rho);
            Kokkos::atomic_add(&d_res_rho_u(dof_i_L), phi_i_L * flux_rho_u);
            Kokkos::atomic_add(&d_res_rho_v(dof_i_L), phi_i_L * flux_rho_v);
            Kokkos::atomic_add(&d_res_rho_E(dof_i_L), phi_i_L * flux_rho_E);

            Kokkos::atomic_add(&d_res_rho(dof_i_R), -phi_i_R * flux_rho);
            Kokkos::atomic_add(&d_res_rho_u(dof_i_R), -phi_i_R * flux_rho_u);
            Kokkos::atomic_add(&d_res_rho_v(dof_i_R), -phi_i_R * flux_rho_v);
            Kokkos::atomic_add(&d_res_rho_E(dof_i_R), -phi_i_R * flux_rho_E);
          }
        }
      });
    Kokkos::fence();
  }

  void compute_boundary_face_residual()
  {
    const auto n_dofs_per_cell = dof_handler_.n_dofs_per_cell();
    const auto n_q_face = fe_face_values_.n_q_points();

    auto phi = boundary_phi_;
    auto JxW = boundary_JxW_;
    auto normal = boundary_normal_;
    auto q_point = boundary_q_point_;
    auto dofs = boundary_dofs_;
    auto ids = boundary_id_;

    auto d_rho = rho_.view();
    auto d_rho_u = rho_u_.view();
    auto d_rho_v = rho_v_.view();
    auto d_rho_E = rho_E_.view();

    auto d_res_rho = res_rho_.view();
    auto d_res_rho_u = res_rho_u_.view();
    auto d_res_rho_v = res_rho_v_.view();
    auto d_res_rho_E = res_rho_E_.view();

    Kokkos::parallel_for(
      "boundary_face_residual", n_boundary_faces_, KOKKOS_LAMBDA(int f) {
        const uint32_t id = ids(f);

        for (unsigned int q = 0; q < n_q_face; ++q) {
          const RealType nx = normal(f, q, 0);
          const RealType ny = normal(f, q, 1);
          const RealType jxw = JxW(f, q);

          // Reconstruct interior state
          RealType rho_L = 0, rho_u_L = 0, rho_v_L = 0, rho_E_L = 0;
          for (unsigned int j = 0; j < n_dofs_per_cell; ++j) {
            const RealType phi_j = phi(f, q, j);
            const uint32_t dof_j = dofs(f, j);
            rho_L += d_rho(dof_j) * phi_j;
            rho_u_L += d_rho_u(dof_j) * phi_j;
            rho_v_L += d_rho_v(dof_j) * phi_j;
            rho_E_L += d_rho_E(dof_j) * phi_j;
          }

          // For freestream, exterior state is the same
          RealType rho_R = rho_L;
          RealType rho_u_R = rho_u_L;
          RealType rho_v_R = rho_v_L;
          RealType rho_E_R = rho_E_L;

          // Numerical flux
          RealType flux_rho, flux_rho_u, flux_rho_v, flux_rho_E, smag;
          Flux<RealType>::roe_flux(rho_L,
                                   rho_u_L,
                                   rho_v_L,
                                   rho_E_L,
                                   rho_R,
                                   rho_u_R,
                                   rho_v_R,
                                   rho_E_R,
                                   nx,
                                   ny,
                                   flux_rho,
                                   flux_rho_u,
                                   flux_rho_v,
                                   flux_rho_E,
                                   smag);

          // Scatter to interior cell only (no neighbor)
          for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
            const RealType phi_i = phi(f, q, i) * jxw;
            const uint32_t dof_i = dofs(f, i);

            Kokkos::atomic_add(&d_res_rho(dof_i), phi_i * flux_rho);
            Kokkos::atomic_add(&d_res_rho_u(dof_i), phi_i * flux_rho_u);
            Kokkos::atomic_add(&d_res_rho_v(dof_i), phi_i * flux_rho_v);
            Kokkos::atomic_add(&d_res_rho_E(dof_i), phi_i * flux_rho_E);
          }
        }
      });
    Kokkos::fence();
  }

  RealType residual_norm() const
  {
    auto d_res_rho = res_rho_.view();
    auto d_res_rho_u = res_rho_u_.view();
    auto d_res_rho_v = res_rho_v_.view();
    auto d_res_rho_E = res_rho_E_.view();

    RealType norm = RealType(0);
    Kokkos::parallel_reduce(
      "res_norm",
      Kokkos::RangePolicy<>(0, n_dofs_),
      KOKKOS_LAMBDA(int i, RealType& local) {
        local += d_res_rho(i) * d_res_rho(i) + d_res_rho_u(i) * d_res_rho_u(i) +
                 d_res_rho_v(i) * d_res_rho_v(i) +
                 d_res_rho_E(i) * d_res_rho_E(i);
      },
      norm);
    return Kokkos::sqrt(norm);
  }
};

static constexpr unsigned int problem_degree = 0;

int
main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  {
    GriReader<2> gri;
    Triangulation<2> tria;
    FE_DGQLegendre<2, double> fe(problem_degree);
    QGaussSimplex<2, double> quad(problem_degree + 1);
    QGaussSimplex<1, double> face_quad(problem_degree + 1);

    gri.read_gri("../tests/test_2.gri");
    gri.transfer_to_triangulation(tria);
    if (!tria.verify_mesh()) {
      std::runtime_error("Verify mesh failed");
    }

    DoFHandler<2, double> dof_handler(tria, fe);

    // Create the FEValues objects
    FEValues<2, double> fe_values(fe, quad);
    FEFaceValues<2, double> fe_face_values(fe, face_quad);

    EulerSolver<2, double> solver(
      dof_handler, fe_values, fe_face_values, problem_degree);

    // Set uniform freestream state
    solver.set_initial_condition([&](double x, double y) {
      return std::make_tuple(1.0, 1.0, 0.0, 1.0 / Parameters<double>::gamma);
    });

    // Compute one residual
    solver.zero_residuals();
    solver.compute_volume_residual();
    solver.compute_face_residual();
    std::cout << "Residual " << solver.residual_norm() << std::endl;
  }

  Kokkos::finalize();
  return 0;
}
