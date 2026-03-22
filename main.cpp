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

struct BoundaryId
{
  static constexpr uint32_t InviscidWall = 1;
  static constexpr uint32_t SubsonicInflow = 2;
  static constexpr uint32_t SubsonicOutflow = 3;
  static constexpr uint32_t Freestream = 4;
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

  void set_state_from_host(const VecHost& rho_h,
                           const VecHost& rho_u_h,
                           const VecHost& rho_v_h,
                           const VecHost& rho_E_h)
  {
    Kokkos::deep_copy(rho_.view(), rho_h.view());
    Kokkos::deep_copy(rho_u_.view(), rho_u_h.view());
    Kokkos::deep_copy(rho_v_.view(), rho_v_h.view());
    Kokkos::deep_copy(rho_E_.view(), rho_E_h.view());
  }

  void copy_state_to_host(VecHost& rho_h,
                          VecHost& rho_u_h,
                          VecHost& rho_v_h,
                          VecHost& rho_E_h) const
  {
    Kokkos::deep_copy(rho_h.view(), rho_.view());
    Kokkos::deep_copy(rho_u_h.view(), rho_u_.view());
    Kokkos::deep_copy(rho_v_h.view(), rho_v_.view());
    Kokkos::deep_copy(rho_E_h.view(), rho_E_.view());
  }

  void write_solution(const std::string& filename, unsigned int cycle = 0)
  {
    constexpr RealType gamma = Parameters<RealType>::gamma;
    constexpr RealType gm1 = gamma - RealType(1);

    // Pull state to host
    VecHost rho_h(n_dofs_);
    VecHost rho_u_h(n_dofs_);
    VecHost rho_v_h(n_dofs_);
    VecHost rho_E_h(n_dofs_);

    Kokkos::deep_copy(rho_h.view(), rho_.view());
    Kokkos::deep_copy(rho_u_h.view(), rho_u_.view());
    Kokkos::deep_copy(rho_v_h.view(), rho_v_.view());
    Kokkos::deep_copy(rho_E_h.view(), rho_E_.view());

    // Pull residuals to host
    VecHost res_rho_h(n_dofs_);
    VecHost res_rho_u_h(n_dofs_);
    VecHost res_rho_v_h(n_dofs_);
    VecHost res_rho_E_h(n_dofs_);

    Kokkos::deep_copy(res_rho_h.view(), res_rho_.view());
    Kokkos::deep_copy(res_rho_u_h.view(), res_rho_u_.view());
    Kokkos::deep_copy(res_rho_v_h.view(), res_rho_v_.view());
    Kokkos::deep_copy(res_rho_E_h.view(), res_rho_E_.view());

    // Compute derived quantities
    VecHost u_h(n_dofs_);
    VecHost v_h(n_dofs_);
    VecHost p_h(n_dofs_);
    VecHost mach_h(n_dofs_);
    VecHost entropy_h(n_dofs_);

    for (unsigned int i = 0; i < n_dofs_; ++i) {
      const RealType rho = rho_h[i];
      const RealType rho_u = rho_u_h[i];
      const RealType rho_v = rho_v_h[i];
      const RealType rho_E = rho_E_h[i];

      const RealType u = rho_u / rho;
      const RealType v = rho_v / rho;
      const RealType p = gm1 * (rho_E - RealType(0.5) * rho * (u * u + v * v));
      const RealType a = std::sqrt(gamma * p / rho);

      u_h[i] = u;
      v_h[i] = v;
      p_h[i] = p;
      mach_h[i] = std::sqrt(u * u + v * v) / a;
      entropy_h[i] = p / std::pow(rho, gamma);
    }

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler_);
    data_out.set_cycle(cycle);

    data_out.add_data_vector(rho_h, "density");
    data_out.add_data_vector(rho_u_h, "momentum_x");
    data_out.add_data_vector(rho_v_h, "momentum_y");
    data_out.add_data_vector(rho_E_h, "total_energy");
    data_out.add_data_vector(u_h, "velocity_x");
    data_out.add_data_vector(v_h, "velocity_y");
    data_out.add_data_vector(p_h, "pressure");
    data_out.add_data_vector(mach_h, "mach");
    data_out.add_data_vector(entropy_h, "entropy");
    data_out.add_data_vector(res_rho_h, "res_density");
    data_out.add_data_vector(res_rho_u_h, "res_momentum_x");
    data_out.add_data_vector(res_rho_v_h, "res_momentum_y");
    data_out.add_data_vector(res_rho_E_h, "res_total_energy");

    data_out.write_vtu(filename);
    data_out.clear();
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

      for (auto dof : dof_ids) {
        rho_h(dof) = rho0;
        rho_u_h(dof) = rho0 * u0;
        rho_v_h(dof) = rho0 * v0;
        rho_E_h(dof) = rhoE0;
      }
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
  Kokkos::View<RealType*, Layout, DeviceMemSpace> cell_area_; // [n_cells]

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

  // Inverted mass matrix
  Kokkos::View<RealType***, Layout, DeviceMemSpace>
    invm_; // [n_cells, n_dofs_per_cell, n_dofs_per_cell]

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
    auto cell_area_h =
      Kokkos::View<RealType*, Layout, HostMemSpace>("cell_area_h", n_cells);

    uint32_t k = 0;
    std::vector<uint32_t> dof_indices;
    for (auto cell : dof_handler_.active_cell_range()) {
      fe_values_.reinit(cell);
      cell.get_dof_indices(dof_indices);

      RealType area = 0;
      for (unsigned int q = 0; q < n_q_points; ++q) {
        JxW_h(k, q) = fe_values_.JxW(q);
        area += fe_values_.JxW(q);

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
      cell_area_h(k) = area;

      for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
        cell_dofs_h(k, i) = dof_indices[i];
      }
      ++k;
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
    cell_area_ =
      Kokkos::View<RealType*, Layout, DeviceMemSpace>("cell_area", n_cells);

    Kokkos::deep_copy(JxW_, JxW_h);
    Kokkos::deep_copy(q_point_, q_point_h);
    Kokkos::deep_copy(phi_, phi_h);
    Kokkos::deep_copy(grad_phi_, grad_phi_h);
    Kokkos::deep_copy(cell_dofs_, cell_dofs_h);
    Kokkos::deep_copy(cell_area_, cell_area_h);

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

    // Create and invert the mass matrix
    MassMatrix<2, double> invm(fe_values_);
    invm.assemble(dof_handler_);
    invm.invert();
    invm.check_inverse();
    invm_ = invm.device_inverse();
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

  void solve(unsigned int max_iter = 50000,
             RealType cfl = Parameters<RealType>::cfl_max,
             unsigned int write_interval = 1000)
  {
    std::cout << "Starting solve with " << max_iter << " iterations"
              << std::endl;

    // Compute initial residual norm for normalization
    zero_residuals();
    compute_volume_residual();
    compute_face_residual();
    const RealType res0 = residual_norm();
    std::cout << "Initial residual: " << res0 << std::endl;

    // Write initial solution
    write_solution("solution_0000.vtu", 0);

    for (unsigned int iter = 0; iter < max_iter; ++iter) {
      // Store u^n
      Kokkos::deep_copy(rho_old_.view(), rho_.view());
      Kokkos::deep_copy(rho_u_old_.view(), rho_u_.view());
      Kokkos::deep_copy(rho_v_old_.view(), rho_v_.view());
      Kokkos::deep_copy(rho_E_old_.view(), rho_E_.view());

      // Compute local timestep once per iteration
      compute_local_dt(cfl);

      // --- SSP-RK3 Stage 1: u^(1) = u^n + dt*R(u^n) ---
      zero_residuals();
      compute_volume_residual();
      compute_face_residual();
      update(RealType(0.0), RealType(1.0));

      // --- SSP-RK3 Stage 2: u^(2) = 3/4*u^n + 1/4*(u^(1) + dt*R(u^(1))) ---
      zero_residuals();
      compute_volume_residual();
      compute_face_residual();
      update(RealType(0.75), RealType(0.25));

      // --- SSP-RK3 Stage 3: u^(n+1) = 1/3*u^n + 2/3*(u^(2) + dt*R(u^(2))) ---
      zero_residuals();
      compute_volume_residual();
      compute_face_residual();
      update(RealType(1.0 / 3.0), RealType(2.0 / 3.0));

      // Compute residual for convergence check
      if (iter % write_interval == 0 || iter == max_iter - 1) {
        zero_residuals();
        compute_volume_residual();
        compute_face_residual();
        const RealType res = residual_norm();
        const RealType rel_res = (res0 > RealType(0)) ? res / res0 : res;

        std::cout << "Iter " << std::setw(6) << iter
                  << "  abs_res=" << std::scientific << std::setprecision(6)
                  << res << "  rel_res=" << rel_res << std::endl;

        // Write solution
        std::ostringstream fname;
        fname << "solution_" << std::setw(4) << std::setfill('0') << iter
              << ".vtu";
        write_solution(fname.str(), iter);

        // Check for NaN/divergence
        if (std::isnan(res) || std::isinf(res)) {
          std::cout << "Solution diverged at iter " << iter << std::endl;
          break;
        }

        // Convergence check
        if (rel_res < Parameters<RealType>::convergence_tol) {
          std::cout << "Converged at iter " << iter << " rel_res=" << rel_res
                    << std::endl;
          write_solution("solution_final.vtu", iter);
          break;
        }
      }
    }

    write_solution("solution_final.vtu", max_iter);
  }

  void test_freestream_preservation(unsigned int n_steps = 10)
  {
    // Store initial state
    auto rho_init = Kokkos::create_mirror_view(rho_.view());
    auto rho_u_init = Kokkos::create_mirror_view(rho_u_.view());
    auto rho_v_init = Kokkos::create_mirror_view(rho_v_.view());
    auto rho_E_init = Kokkos::create_mirror_view(rho_E_.view());
    Kokkos::deep_copy(rho_init, rho_.view());
    Kokkos::deep_copy(rho_u_init, rho_u_.view());
    Kokkos::deep_copy(rho_v_init, rho_v_.view());
    Kokkos::deep_copy(rho_E_init, rho_E_.view());

    // Run n_steps of SSP-RK3
    for (unsigned int step = 0; step < n_steps; ++step) {
      // Store u^n
      Kokkos::deep_copy(rho_old_.view(), rho_.view());
      Kokkos::deep_copy(rho_u_old_.view(), rho_u_.view());
      Kokkos::deep_copy(rho_v_old_.view(), rho_v_.view());
      Kokkos::deep_copy(rho_E_old_.view(), rho_E_.view());

      compute_local_dt(Parameters<RealType>::cfl_max);

      // Stage 1
      zero_residuals();
      compute_volume_residual();
      compute_face_residual();
      update(RealType(0.0), RealType(1.0));

      // Stage 2
      zero_residuals();
      compute_volume_residual();
      compute_face_residual();
      update(RealType(0.75), RealType(0.25));

      // Stage 3
      zero_residuals();
      compute_volume_residual();
      compute_face_residual();
      update(RealType(1.0 / 3.0), RealType(2.0 / 3.0));

      // Check residual is still zero after this step
      zero_residuals();
      compute_volume_residual();
      compute_face_residual();

      auto res_rho_h = Kokkos::create_mirror_view(res_rho_.view());
      auto res_rho_u_h = Kokkos::create_mirror_view(res_rho_u_.view());
      auto res_rho_v_h = Kokkos::create_mirror_view(res_rho_v_.view());
      auto res_rho_E_h = Kokkos::create_mirror_view(res_rho_E_.view());
      Kokkos::deep_copy(res_rho_h, res_rho_.view());
      Kokkos::deep_copy(res_rho_u_h, res_rho_u_.view());
      Kokkos::deep_copy(res_rho_v_h, res_rho_v_.view());
      Kokkos::deep_copy(res_rho_E_h, res_rho_E_.view());

      RealType max_res = 0;
      for (unsigned int i = 0; i < n_dofs_; ++i) {
        max_res = Kokkos::max(max_res, Kokkos::abs(res_rho_h(i)));
        max_res = Kokkos::max(max_res, Kokkos::abs(res_rho_u_h(i)));
        max_res = Kokkos::max(max_res, Kokkos::abs(res_rho_v_h(i)));
        max_res = Kokkos::max(max_res, Kokkos::abs(res_rho_E_h(i)));
      }
      std::cout << "Step " << step << " max_residual=" << max_res << std::endl;

      // Also check state hasn't changed from initial
      auto rho_h = Kokkos::create_mirror_view(rho_.view());
      auto rho_u_h = Kokkos::create_mirror_view(rho_u_.view());
      auto rho_v_h = Kokkos::create_mirror_view(rho_v_.view());
      auto rho_E_h = Kokkos::create_mirror_view(rho_E_.view());
      Kokkos::deep_copy(rho_h, rho_.view());
      Kokkos::deep_copy(rho_u_h, rho_u_.view());
      Kokkos::deep_copy(rho_v_h, rho_v_.view());
      Kokkos::deep_copy(rho_E_h, rho_E_.view());

      RealType max_state_change = 0;
      for (unsigned int i = 0; i < n_dofs_; ++i) {
        max_state_change =
          Kokkos::max(max_state_change, Kokkos::abs(rho_h(i) - rho_init(i)));
        max_state_change = Kokkos::max(max_state_change,
                                       Kokkos::abs(rho_u_h(i) - rho_u_init(i)));
        max_state_change = Kokkos::max(max_state_change,
                                       Kokkos::abs(rho_v_h(i) - rho_v_init(i)));
        max_state_change = Kokkos::max(max_state_change,
                                       Kokkos::abs(rho_E_h(i) - rho_E_init(i)));
      }
      std::cout << "Step " << step << " max_state_change=" << max_state_change
                << std::endl;
    }
  }

  void compute_local_dt(RealType cfl)
  {
    auto d_rho = rho_.view();
    auto d_rho_u = rho_u_.view();
    auto d_rho_v = rho_v_.view();
    auto d_rho_E = rho_E_.view();
    auto d_dt = dt_.view();
    auto indices = cell_dofs_;
    auto areas = cell_area_;

    const auto n_dofs_per_cell = dof_handler_.n_dofs_per_cell();
    const RealType gamma = Parameters<RealType>::gamma;
    const RealType p_order = static_cast<RealType>(degree_);
    const RealType dg_scaling =
      RealType(1) / (RealType(2) * p_order + RealType(1));

    Kokkos::parallel_for(
      "compute_local_dt", n_cells_, KOKKOS_LAMBDA(int k) {
        // Cell-averaged state
        RealType rho_avg = 0, rhou_avg = 0, rhov_avg = 0, rhoE_avg = 0;
        for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
          const uint32_t idx = indices(k, i);
          rho_avg += d_rho(idx);
          rhou_avg += d_rho_u(idx);
          rhov_avg += d_rho_v(idx);
          rhoE_avg += d_rho_E(idx);
        }
        rho_avg /= n_dofs_per_cell;
        rhou_avg /= n_dofs_per_cell;
        rhov_avg /= n_dofs_per_cell;
        rhoE_avg /= n_dofs_per_cell;

        // Velocity and speed of sound
        const RealType u = rhou_avg / rho_avg;
        const RealType v = rhov_avg / rho_avg;
        const RealType vel_mag = Kokkos::sqrt(u * u + v * v);
        const RealType p =
          (gamma - RealType(1)) *
          (rhoE_avg - RealType(0.5) * rho_avg * (u * u + v * v));
        const RealType a = Kokkos::sqrt(gamma * p / rho_avg);

        // Characteristic length from precomputed area
        const RealType h = Kokkos::sqrt(areas(k));
        const RealType dt = cfl * dg_scaling * h / (vel_mag + a);

        for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
          d_dt(indices(k, i)) = dt;
      });
    Kokkos::fence();
  }

  void compute_volume_residual()
  {
    const auto n_dofs_per_cell = dof_handler_.n_dofs_per_cell();
    const auto n_q_points = fe_values_.n_q_points();

    const auto phi = phi_;
    const auto grad_phi = grad_phi_;
    const auto JxW = JxW_;
    const auto indices = cell_dofs_;

    const auto d_rho = rho_.view();
    const auto d_rho_u = rho_u_.view();
    const auto d_rho_v = rho_v_.view();
    const auto d_rho_E = rho_E_.view();

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

  void update(RealType alpha, RealType beta)
  {
    const auto ndpc = dof_handler_.n_dofs_per_cell();

    auto d_rho = rho_.view();
    auto d_rho_u = rho_u_.view();
    auto d_rho_v = rho_v_.view();
    auto d_rho_E = rho_E_.view();
    auto d_rho_old = rho_old_.view();
    auto d_rho_u_old = rho_u_old_.view();
    auto d_rho_v_old = rho_v_old_.view();
    auto d_rho_E_old = rho_E_old_.view();
    auto d_res_rho = res_rho_.view();
    auto d_res_rho_u = res_rho_u_.view();
    auto d_res_rho_v = res_rho_v_.view();
    auto d_res_rho_E = res_rho_E_.view();
    auto d_dt = dt_.view();
    auto indices = cell_dofs_;
    auto invm = invm_;

    Kokkos::parallel_for(
      "update_rk_stage",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 },
                                             { (int)n_cells_, (int)ndpc }),
      KOKKOS_LAMBDA(int k, int i) {
        const uint32_t dof_i = indices(k, i);
        const RealType dt = d_dt(dof_i);

        // Apply M^{-1} R: result_i = sum_j Minv[k,i,j] * R[j]
        RealType Minv_R_rho = 0;
        RealType Minv_R_rho_u = 0;
        RealType Minv_R_rho_v = 0;
        RealType Minv_R_rho_E = 0;
        for (unsigned int j = 0; j < ndpc; ++j) {
          const RealType Minv_ij = invm(k, i, j);
          const uint32_t dof_j = indices(k, j);
          Minv_R_rho += Minv_ij * d_res_rho(dof_j);
          Minv_R_rho_u += Minv_ij * d_res_rho_u(dof_j);
          Minv_R_rho_v += Minv_ij * d_res_rho_v(dof_j);
          Minv_R_rho_E += Minv_ij * d_res_rho_E(dof_j);
        }

        // SSP-RK3: u_new = alpha*u_old + beta*(u_curr - dt * M^{-1} R)
        d_rho(dof_i) =
          alpha * d_rho_old(dof_i) + beta * (d_rho(dof_i) - dt * Minv_R_rho);
        d_rho_u(dof_i) = alpha * d_rho_u_old(dof_i) +
                         beta * (d_rho_u(dof_i) - dt * Minv_R_rho_u);
        d_rho_v(dof_i) = alpha * d_rho_v_old(dof_i) +
                         beta * (d_rho_v(dof_i) - dt * Minv_R_rho_v);
        d_rho_E(dof_i) = alpha * d_rho_E_old(dof_i) +
                         beta * (d_rho_E(dof_i) - dt * Minv_R_rho_E);
      });
    Kokkos::fence();
  }

  void check_initial_condition_validity()
  {
    constexpr RealType gamma = Parameters<RealType>::gamma;
    constexpr RealType gm1 = gamma - RealType(1);

    auto rho_h = Kokkos::create_mirror_view(rho_.view());
    auto rho_u_h = Kokkos::create_mirror_view(rho_u_.view());
    auto rho_v_h = Kokkos::create_mirror_view(rho_v_.view());
    auto rho_E_h = Kokkos::create_mirror_view(rho_E_.view());
    Kokkos::deep_copy(rho_h, rho_.view());
    Kokkos::deep_copy(rho_u_h, rho_u_.view());
    Kokkos::deep_copy(rho_v_h, rho_v_.view());
    Kokkos::deep_copy(rho_E_h, rho_E_.view());

    int n_bad = 0;
    for (unsigned int i = 0; i < n_dofs_; ++i) {
      const RealType rho = rho_h(i);
      const RealType rho_u = rho_u_h(i);
      const RealType rho_v = rho_v_h(i);
      const RealType rho_E = rho_E_h(i);
      const RealType u = rho_u / rho;
      const RealType v = rho_v / rho;
      const RealType p = gm1 * (rho_E - RealType(0.5) * rho * (u * u + v * v));

      if (rho <= 0 || p <= 0 || rho_E <= 0) {
        std::cout << "Bad DOF " << i << " rho=" << rho << " p=" << p
                  << " rho_E=" << rho_E << std::endl;
        n_bad++;
      }
    }
    std::cout << "Total bad DOFs: " << n_bad << "/" << n_dofs_ << std::endl;
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

          // Numerical flux
          RealType flux_rho, flux_rho_u, flux_rho_v, flux_rho_E, smag;
          switch (id) {
            case BoundaryId::InviscidWall: {
              Flux<RealType>::inviscid_wall_flux(rho_L,
                                                 rho_u_L,
                                                 rho_v_L,
                                                 rho_E_L,
                                                 nx,
                                                 ny,
                                                 flux_rho,
                                                 flux_rho_u,
                                                 flux_rho_v,
                                                 flux_rho_E,
                                                 smag);
              break;
            }
            case BoundaryId::SubsonicInflow: {
              Flux<RealType>::subsonic_inflow_flux(rho_L,
                                                   rho_u_L,
                                                   rho_v_L,
                                                   rho_E_L,
                                                   nx,
                                                   ny,
                                                   flux_rho,
                                                   flux_rho_u,
                                                   flux_rho_v,
                                                   flux_rho_E,
                                                   smag);
              break;
            }
            case BoundaryId::SubsonicOutflow: {
              Flux<RealType>::subsonic_outflow_flux(rho_L,
                                                    rho_u_L,
                                                    rho_v_L,
                                                    rho_E_L,
                                                    nx,
                                                    ny,
                                                    flux_rho,
                                                    flux_rho_u,
                                                    flux_rho_v,
                                                    flux_rho_E,
                                                    smag);
              break;
            }
            default: {
              // Freestream / do-nothing
              Flux<RealType>::roe_flux(rho_L,
                                       rho_u_L,
                                       rho_v_L,
                                       rho_E_L,
                                       rho_L,
                                       rho_u_L,
                                       rho_v_L,
                                       rho_E_L,
                                       nx,
                                       ny,
                                       flux_rho,
                                       flux_rho_u,
                                       flux_rho_v,
                                       flux_rho_E,
                                       smag);
              break;
            }
          }

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

template<unsigned int dim, typename RealType>
void
interpolate_solution(
  const DoFHandler<dim, RealType>& dof_handler_lo,
  const DoFHandler<dim, RealType>& dof_handler_hi,
  const FE_DGQLegendre<dim, RealType>& fe_lo,
  const FE_DGQLegendre<dim, RealType>& fe_hi,
  const Kokkos::View<RealType*, Layout, HostMemSpace>& rho_lo,
  const Kokkos::View<RealType*, Layout, HostMemSpace>& rhou_lo,
  const Kokkos::View<RealType*, Layout, HostMemSpace>& rhov_lo,
  const Kokkos::View<RealType*, Layout, HostMemSpace>& rhoE_lo,
  Kokkos::View<RealType*, Layout, HostMemSpace>& rho_hi,
  Kokkos::View<RealType*, Layout, HostMemSpace>& rhou_hi,
  Kokkos::View<RealType*, Layout, HostMemSpace>& rhov_hi,
  Kokkos::View<RealType*, Layout, HostMemSpace>& rhoE_hi)
{
  const auto n_dofs_lo = fe_lo.n_dofs();
  const auto n_dofs_hi = fe_hi.n_dofs();

  std::vector<uint32_t> dof_indices_lo, dof_indices_hi;

  // Assumes cells are in the same order in both dof handlers
  // i.e. same mesh, different polynomial degree
  for (auto cell : dof_handler_hi.active_cell_range()) {
    const auto k = cell.index();
    cell.get_dof_indices(dof_indices_hi);

    // Get low-order dof indices for same cell
    auto cell_lo = dof_handler_lo.cell(k);
    cell_lo.get_dof_indices(dof_indices_lo);

    // For each high-order node, evaluate the low-order polynomial
    for (unsigned int i = 0; i < n_dofs_hi; ++i) {
      // Get the reference coordinate of high-order node i
      const auto xi = fe_hi.node(i);

      // Evaluate low-order basis functions at this point
      RealType rho_val = 0, rhou_val = 0, rhov_val = 0, rhoE_val = 0;
      for (unsigned int j = 0; j < n_dofs_lo; ++j) {
        const RealType phi_j = fe_lo.shape_value(j, xi);
        const uint32_t dof_j = dof_indices_lo[j];
        rho_val += rho_lo(dof_j) * phi_j;
        rhou_val += rhou_lo(dof_j) * phi_j;
        rhov_val += rhov_lo(dof_j) * phi_j;
        rhoE_val += rhoE_lo(dof_j) * phi_j;
      }

      const uint32_t dof_i = dof_indices_hi[i];
      rho_hi(dof_i) = rho_val;
      rhou_hi(dof_i) = rhou_val;
      rhov_hi(dof_i) = rhov_val;
      rhoE_hi(dof_i) = rhoE_val;
    }
  }
}

static constexpr unsigned int lo_degree = 0;
static constexpr unsigned int hi_degree = 1;

int
main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  {
    GriReader<2> gri;
    Triangulation<2> tria;
    gri.read_gri("../grids/coarse_local_refinement_1.gri");
    gri.transfer_to_triangulation(tria);
    if (!tria.verify_mesh()) {
      std::runtime_error("Verify mesh failed");
    }

    // Renumber boundary ids so they match above.
    std::unordered_map<uint32_t, uint32_t> id_map = {
      { 7, BoundaryId::InviscidWall },
      { 8, BoundaryId::InviscidWall },
      { 5, BoundaryId::SubsonicInflow },
      { 6, BoundaryId::SubsonicOutflow },
    };
    tria.remap_boundary_ids(id_map);

    // Low-order setup
    FE_DGQLegendre<2, double> fe_lo(lo_degree);
    QGaussSimplex<2, double> quad_lo(2 * lo_degree + 1);
    QGaussSimplex<1, double> face_quad_lo(2 * lo_degree + 1);
    DoFHandler<2, double> dof_handler_lo(tria, fe_lo);
    FEValues<2, double> fe_values_lo(fe_lo, quad_lo);
    FEFaceValues<2, double> fe_face_values_lo(fe_lo, face_quad_lo);

    EulerSolver<2, double> solver_lo(
      dof_handler_lo, fe_values_lo, fe_face_values_lo, lo_degree);

    auto ic = [](double x, double y) {
      constexpr double gamma = Parameters<double>::gamma;
      constexpr double gm1 = gamma - 1.0;
      constexpr double p_0 = Parameters<double>::p_0;
      constexpr double rho_0 = Parameters<double>::rho_0;
      constexpr double a_0 = Parameters<double>::a_0;
      constexpr double p_out = Parameters<double>::p_out;

      const double p = p_out;
      const double ratio = p / p_0;
      const double rho = rho_0 * std::pow(ratio, 1.0 / gamma);
      const double a = std::sqrt(gamma * p / rho);
      const double V = std::sqrt(2.0 / gm1 * (a_0 * a_0 - a * a));
      const double u = V * Parameters<double>::n_x_0();
      const double v = V * Parameters<double>::n_y_0();
      return std::make_tuple(rho, u, v, p);
    };

    solver_lo.set_initial_condition(ic);
    solver_lo.check_initial_condition_validity();
    solver_lo.solve();
    solver_lo.write_solution("solution_lo.vtu");

    // Extract low-order state to host
    const auto n_dofs_lo = dof_handler_lo.n_dofs();
    Vector<double, HostMemSpace> rho_h(n_dofs_lo), rhou_h(n_dofs_lo),
      rhov_h(n_dofs_lo), rhoE_h(n_dofs_lo);
    solver_lo.copy_state_to_host(rho_h, rhou_h, rhov_h, rhoE_h);

    // High-order setup
    FE_DGQLegendre<2, double> fe_hi(hi_degree);
    QGaussSimplex<2, double> quad_hi(2 * hi_degree + 1);
    QGaussSimplex<1, double> face_quad_hi(2 * hi_degree + 1);
    DoFHandler<2, double> dof_handler_hi(tria, fe_hi);
    FEValues<2, double> fe_values_hi(fe_hi, quad_hi);
    FEFaceValues<2, double> fe_face_values_hi(fe_hi, face_quad_hi);

    // Interpolate
    const auto n_dofs_hi = dof_handler_hi.n_dofs();
    Vector<double, HostMemSpace> rho_hi(n_dofs_hi), rhou_hi(n_dofs_hi),
      rhov_hi(n_dofs_hi), rhoE_hi(n_dofs_hi);

    interpolate_solution(dof_handler_lo,
                         dof_handler_hi,
                         fe_lo,
                         fe_hi,
                         rho_h.view(),
                         rhou_h.view(),
                         rhov_h.view(),
                         rhoE_h.view(),
                         rho_hi.view(),
                         rhou_hi.view(),
                         rhov_hi.view(),
                         rhoE_hi.view());

    // Inject into high-order solver and write
    EulerSolver<2, double> solver_hi(
      dof_handler_hi, fe_values_hi, fe_face_values_hi, hi_degree);
    solver_hi.set_state_from_host(rho_hi, rhou_hi, rhov_hi, rhoE_hi);
    solver_hi.write_solution("solution_hi_interpolated.vtu");
    solver_hi.solve();
    solver_hi.write_solution("solution_hi.vtu");
  }

  Kokkos::finalize();
  return 0;
}
