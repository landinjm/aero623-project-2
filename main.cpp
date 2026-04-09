#include <Kokkos_Core.hpp>
#include <config.hpp>
#include <cstdint>
#include <data_out.hpp>
#include <dof_handler.hpp>
#include <fe.hpp>
#include <flux.hpp>
#include <iterator>
#include <matrix.hpp>
#include <parameters.hpp>
#include <read_gri.hpp>
#include <solve.hpp>
#include <stdexcept>
#include <timer.hpp>
#include <triangulation.hpp>
#include <vector.hpp>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <tuple>

struct BoundaryId
{
  static constexpr uint32_t InviscidWall = 1;
  static constexpr uint32_t SubsonicInflow = 2;
  static constexpr uint32_t SubsonicOutflow = 3;
  static constexpr uint32_t UnsteadySubsonicInflow = 4;
  static constexpr uint32_t Freestream = 5;
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
    // Allocate state vectors for 3D
    rho_   = VecDevice(n_dofs_);
    rho_u_ = VecDevice(n_dofs_);
    rho_v_ = VecDevice(n_dofs_);
    rho_w_ = VecDevice(n_dofs_); 
    rho_E_ = VecDevice(n_dofs_);

    rho_old_   = VecDevice(n_dofs_);
    rho_u_old_ = VecDevice(n_dofs_);
    rho_v_old_ = VecDevice(n_dofs_);
    rho_w_old_ = VecDevice(n_dofs_); 
    rho_E_old_ = VecDevice(n_dofs_);

    res_rho_   = VecDevice(n_dofs_);
    res_rho_u_ = VecDevice(n_dofs_);
    res_rho_v_ = VecDevice(n_dofs_);
    res_rho_w_ = VecDevice(n_dofs_); 
    res_rho_E_ = VecDevice(n_dofs_);

    dt_ = VecDevice(n_dofs_);

    // Precompute geometries
    precompute_geometry();
  }

  void set_state_from_host(const VecHost& rho_h,
                           const VecHost& rho_u_h,
                           const VecHost& rho_v_h,
                           const VecHost& rho_w_h,
                           const VecHost& rho_E_h)
  {
    Kokkos::deep_copy(rho_.view(), rho_h.view());
    Kokkos::deep_copy(rho_u_.view(), rho_u_h.view());
    Kokkos::deep_copy(rho_v_.view(), rho_v_h.view());
    Kokkos::deep_copy(rho_w_.view(), rho_w_h.view()); 
    Kokkos::deep_copy(rho_E_.view(), rho_E_h.view());
  }

  void copy_state_to_host(VecHost& rho_h,
                          VecHost& rho_u_h,
                          VecHost& rho_v_h,
                          VecHost& rho_w_h,
                          VecHost& rho_E_h) const
  {
    Kokkos::deep_copy(rho_h.view(), rho_.view());
    Kokkos::deep_copy(rho_u_h.view(), rho_u_.view());
    Kokkos::deep_copy(rho_v_h.view(), rho_v_.view());
    Kokkos::deep_copy(rho_w_h.view(), rho_w_.view()); 
    Kokkos::deep_copy(rho_E_h.view(), rho_E_.view());
  }

  void write_solution(const std::string& filename,
                      unsigned int cycle = 0,
                      double time = 0.0)
  {
    constexpr RealType gamma = Parameters<RealType>::gamma;
    constexpr RealType gm1 = gamma - RealType(1);

    // Pull state to host
    VecHost rho_h(n_dofs_), rho_u_h(n_dofs_), rho_v_h(n_dofs_), rho_w_h(n_dofs_), rho_E_h(n_dofs_);
    copy_state_to_host(rho_h, rho_u_h, rho_v_h, rho_w_h, rho_E_h); 

    // Pull residuals to host
    VecHost res_rho_h(n_dofs_), res_rho_u_h(n_dofs_), res_rho_v_h(n_dofs_), res_rho_w_h(n_dofs_), res_rho_E_h(n_dofs_);
    Kokkos::deep_copy(res_rho_h.view(), res_rho_.view());
    Kokkos::deep_copy(res_rho_u_h.view(), res_rho_u_.view());
    Kokkos::deep_copy(res_rho_v_h.view(), res_rho_v_.view());
    Kokkos::deep_copy(res_rho_w_h.view(), res_rho_w_.view()); 
    Kokkos::deep_copy(res_rho_E_h.view(), res_rho_E_.view());

    VecHost u_h(n_dofs_), v_h(n_dofs_), w_h(n_dofs_), p_h(n_dofs_), mach_h(n_dofs_), entropy_h(n_dofs_);

    for (unsigned int i = 0; i < n_dofs_; ++i) {
      const RealType rho = rho_h[i];
      const RealType u = rho_u_h[i] / rho;
      const RealType v = rho_v_h[i] / rho;
      const RealType w = rho_w_h[i] / rho; 
      const RealType p = gm1 * (rho_E_h[i] - RealType(0.5) * rho * (u*u + v*v + w*w)); 
      const RealType a = std::sqrt(gamma * p / rho);

      u_h[i] = u; v_h[i] = v; w_h[i] = w; p_h[i] = p;
      mach_h[i] = std::sqrt(u*u + v*v + w*w) / a;
      entropy_h[i] = p / std::pow(rho, gamma);
    }

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler_);
    data_out.set_cycle(cycle);
    data_out.set_time(time);

    data_out.add_data_vector(rho_h, "density");
    data_out.add_data_vector(rho_u_h, "momentum_x");
    data_out.add_data_vector(rho_v_h, "momentum_y");
    data_out.add_data_vector(rho_w_h, "momentum_z");
    data_out.add_data_vector(rho_E_h, "total_energy");
    data_out.add_data_vector(u_h, "velocity_x");
    data_out.add_data_vector(v_h, "velocity_y");
    data_out.add_data_vector(w_h, "velocity_z");
    data_out.add_data_vector(p_h, "pressure");
    data_out.add_data_vector(mach_h, "mach");
    data_out.add_data_vector(entropy_h, "entropy");
    
    data_out.add_data_vector(res_rho_h, "res_density");
    data_out.add_data_vector(res_rho_u_h, "res_momentum_x");
    data_out.add_data_vector(res_rho_v_h, "res_momentum_y");
    data_out.add_data_vector(res_rho_w_h, "res_momentum_z"); 
    data_out.add_data_vector(res_rho_E_h, "res_total_energy");

    data_out.write_vtu(filename);
    data_out.clear();
  }

  template<typename InitFunc>
  void set_initial_condition(InitFunc&& f)
  {
    constexpr RealType gm1 = Parameters<RealType>::gamma - RealType(1.0);
    
    // Create host mirrors
    auto rho_h = Kokkos::create_mirror_view(rho_.view());
    auto rho_u_h = Kokkos::create_mirror_view(rho_u_.view());
    auto rho_v_h = Kokkos::create_mirror_view(rho_v_.view());
    auto rho_w_h = Kokkos::create_mirror_view(rho_w_.view()); 
    auto rho_E_h = Kokkos::create_mirror_view(rho_E_.view());

    std::vector<uint32_t> dof_ids;
    for (auto cell : dof_handler_.active_cell_range()) {
      cell.get_dof_indices(dof_ids);
      const auto ctr = cell.tria_cell.center();
      
      auto [rho0, u0, v0, w0, p0] = f(ctr(0), ctr(1), (dim == 3 ? ctr(2) : 0.0));
      const RealType rhoE0 = p0 / gm1 + 0.5 * rho0 * (u0*u0 + v0*v0 + w0*w0);

      for (auto dof : dof_ids) {
        rho_h(dof) = rho0; 
        rho_u_h(dof) = rho0 * u0;
        rho_v_h(dof) = rho0 * v0; 
        rho_w_h(dof) = rho0 * w0;
        rho_E_h(dof) = rhoE0;
      }
    }
    
    // Deep copy directly instead of passing through set_state_from_host wrapper
    Kokkos::deep_copy(rho_.view(), rho_h);
    Kokkos::deep_copy(rho_u_.view(), rho_u_h);
    Kokkos::deep_copy(rho_v_.view(), rho_v_h);
    Kokkos::deep_copy(rho_w_.view(), rho_w_h);
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

  // Cell geometry views
  Kokkos::View<RealType**, Layout, DeviceMemSpace> JxW_; 
  Kokkos::View<RealType***, Layout, DeviceMemSpace> q_point_; 
  Kokkos::View<RealType***, Layout, DeviceMemSpace> phi_; 
  Kokkos::View<RealType****, Layout, DeviceMemSpace> grad_phi_; 
  Kokkos::View<uint32_t**, Layout, DeviceMemSpace> cell_dofs_; 
  Kokkos::View<RealType*, Layout, DeviceMemSpace> cell_area_; 

  // Interior face geometry views
  Kokkos::View<RealType**, Layout, DeviceMemSpace> interior_JxW_;
  Kokkos::View<RealType***, Layout, DeviceMemSpace> interior_q_point_;
  Kokkos::View<RealType***, Layout, DeviceMemSpace> interior_normal_;
  Kokkos::View<RealType***, Layout, DeviceMemSpace> interior_phi_L_;
  Kokkos::View<RealType***, Layout, DeviceMemSpace> interior_phi_R_;
  Kokkos::View<uint32_t**, Layout, DeviceMemSpace> interior_dofs_L_;
  Kokkos::View<uint32_t**, Layout, DeviceMemSpace> interior_dofs_R_;

  // Periodic face geometry views
  Kokkos::View<RealType**, Layout, DeviceMemSpace> periodic_JxW_;
  Kokkos::View<RealType***, Layout, DeviceMemSpace> periodic_q_point_;
  Kokkos::View<RealType***, Layout, DeviceMemSpace> periodic_normal_;
  Kokkos::View<RealType***, Layout, DeviceMemSpace> periodic_phi_L_;
  Kokkos::View<RealType***, Layout, DeviceMemSpace> periodic_phi_R_;
  Kokkos::View<uint32_t**, Layout, DeviceMemSpace> periodic_dofs_L_;
  Kokkos::View<uint32_t**, Layout, DeviceMemSpace> periodic_dofs_R_;

  // Boundary face geometry views
  Kokkos::View<RealType**, Layout, DeviceMemSpace> boundary_JxW_;
  Kokkos::View<RealType***, Layout, DeviceMemSpace> boundary_q_point_;
  Kokkos::View<RealType***, Layout, DeviceMemSpace> boundary_normal_;
  Kokkos::View<RealType***, Layout, DeviceMemSpace> boundary_phi_;
  Kokkos::View<uint32_t**, Layout, DeviceMemSpace> boundary_dofs_;
  Kokkos::View<uint32_t*, Layout, DeviceMemSpace> boundary_id_; 

  // Inverted mass matrix
  Kokkos::View<RealType***, Layout, DeviceMemSpace> invm_; 

  // Current state
  VecDevice rho_, rho_u_, rho_v_, rho_w_, rho_E_;
  // Old state
  VecDevice rho_old_, rho_u_old_, rho_v_old_, rho_w_old_, rho_E_old_;
  // State residuals
  VecDevice res_rho_, res_rho_u_, res_rho_v_, res_rho_w_, res_rho_E_;
  // Local timesteps
  VecDevice dt_;

  void zero_residuals()
  {
    Kokkos::deep_copy(res_rho_.view(), 0.0);
    Kokkos::deep_copy(res_rho_u_.view(), 0.0);
    Kokkos::deep_copy(res_rho_v_.view(), 0.0);
    Kokkos::deep_copy(res_rho_w_.view(), 0.0);
    Kokkos::deep_copy(res_rho_E_.view(), 0.0);
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

    n_cells_ = n_cells;
    n_interior_faces_ = 0;
    n_periodic_faces_ = 0;
    n_boundary_faces_ = 0;

    for (auto cell : dof_handler_.active_cell_range()) {
      for (unsigned int lf = 0; lf < SimplexTopology<dim>::faces_per_cell; ++lf) {
        auto face = cell.face(lf);
        if (face.at_boundary()) {
          n_boundary_faces_++;
        } else if (face.is_periodic()) {
          if (face.index < face.periodic_neighbor_index()) n_periodic_faces_++;
        } else {
          if (face.owner_index() == cell.index()) n_interior_faces_++;
        }
      }
    }

    std::cout << "Number of interior faces " << n_interior_faces_ << std::endl;
    std::cout << "Number of periodic faces " << n_periodic_faces_ << std::endl;
    std::cout << "Number of boundary faces " << n_boundary_faces_ << std::endl;
    std::cout << std::endl;

    auto JxW_h = Kokkos::View<RealType**, Layout, HostMemSpace>("JxW_h", n_cells, n_q_points);
    auto q_point_h = Kokkos::View<RealType***, Layout, HostMemSpace>("q_point_h", n_cells, n_q_points, dim);
    auto phi_h = Kokkos::View<RealType***, Layout, HostMemSpace>("phi_h", n_cells, n_dofs_per_cell, n_q_points);
    auto grad_phi_h = Kokkos::View<RealType****, Layout, HostMemSpace>("grad_phi_h", n_cells, n_dofs_per_cell, n_q_points, dim);
    auto cell_dofs_h = Kokkos::View<uint32_t**, Layout, HostMemSpace>("cell_dofs_h", n_cells, n_dofs_per_cell);
    auto cell_area_h = Kokkos::View<RealType*, Layout, HostMemSpace>("cell_area_h", n_cells);

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
        for (unsigned int d = 0; d < dim; ++d) q_point_h(k, q, d) = p(d);

        for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
          phi_h(k, i, q) = fe_values_.shape_value(i, q);
          auto grad = fe_values_.shape_gradient(i, q);
          for (unsigned int d = 0; d < dim; ++d) grad_phi_h(k, i, q, d) = grad(d);
        }
      }
      cell_area_h(k) = area;

      for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
        cell_dofs_h(k, i) = dof_indices[i];
      }
      ++k;
    }

    JxW_ = Kokkos::View<RealType**, Layout, DeviceMemSpace>("JxW", n_cells, n_q_points);
    q_point_ = Kokkos::View<RealType***, Layout, DeviceMemSpace>("q_point", n_cells, n_q_points, dim);
    phi_ = Kokkos::View<RealType***, Layout, DeviceMemSpace>("phi", n_cells, n_dofs_per_cell, n_q_points);
    grad_phi_ = Kokkos::View<RealType****, Layout, DeviceMemSpace>("grad_phi", n_cells, n_dofs_per_cell, n_q_points, dim);
    cell_dofs_ = Kokkos::View<uint32_t**, Layout, DeviceMemSpace>("cell_dofs", n_cells, n_dofs_per_cell);
    cell_area_ = Kokkos::View<RealType*, Layout, DeviceMemSpace>("cell_area", n_cells);

    Kokkos::deep_copy(JxW_, JxW_h);
    Kokkos::deep_copy(q_point_, q_point_h);
    Kokkos::deep_copy(phi_, phi_h);
    Kokkos::deep_copy(grad_phi_, grad_phi_h);
    Kokkos::deep_copy(cell_dofs_, cell_dofs_h);
    Kokkos::deep_copy(cell_area_, cell_area_h);

    auto interior_JxW_h = Kokkos::View<RealType**, Layout, HostMemSpace>("interior_JxW_h", n_interior_faces_, n_q_points_face);
    auto interior_q_point_h = Kokkos::View<RealType***, Layout, HostMemSpace>("interior_q_point_h", n_interior_faces_, n_q_points_face, dim);
    auto interior_normal_h = Kokkos::View<RealType***, Layout, HostMemSpace>("interior_normal_h", n_interior_faces_, n_q_points_face, dim);
    auto interior_phi_L_h = Kokkos::View<RealType***, Layout, HostMemSpace>("interior_phi_L_h", n_interior_faces_, n_q_points_face, n_dofs_per_cell);
    auto interior_phi_R_h = Kokkos::View<RealType***, Layout, HostMemSpace>("interior_phi_R_h", n_interior_faces_, n_q_points_face, n_dofs_per_cell);
    auto interior_dofs_L_h = Kokkos::View<uint32_t**, Layout, HostMemSpace>("interior_dofs_L_h", n_interior_faces_, n_dofs_per_cell);
    auto interior_dofs_R_h = Kokkos::View<uint32_t**, Layout, HostMemSpace>("interior_dofs_R_h", n_interior_faces_, n_dofs_per_cell);

    auto periodic_JxW_h = Kokkos::View<RealType**, Layout, HostMemSpace>("periodic_JxW_h", n_periodic_faces_, n_q_points_face);
    auto periodic_q_point_h = Kokkos::View<RealType***, Layout, HostMemSpace>("periodic_q_point_h", n_periodic_faces_, n_q_points_face, dim);
    auto periodic_normal_h = Kokkos::View<RealType***, Layout, HostMemSpace>("periodic_normal_h", n_periodic_faces_, n_q_points_face, dim);
    auto periodic_phi_L_h = Kokkos::View<RealType***, Layout, HostMemSpace>("periodic_phi_L_h", n_periodic_faces_, n_q_points_face, n_dofs_per_cell);
    auto periodic_phi_R_h = Kokkos::View<RealType***, Layout, HostMemSpace>("periodic_phi_R_h", n_periodic_faces_, n_q_points_face, n_dofs_per_cell);
    auto periodic_dofs_L_h = Kokkos::View<uint32_t**, Layout, HostMemSpace>("periodic_dofs_L_h", n_periodic_faces_, n_dofs_per_cell);
    auto periodic_dofs_R_h = Kokkos::View<uint32_t**, Layout, HostMemSpace>("periodic_dofs_R_h", n_periodic_faces_, n_dofs_per_cell);

    auto boundary_JxW_h = Kokkos::View<RealType**, Layout, HostMemSpace>("boundary_JxW_h", n_boundary_faces_, n_q_points_face);
    auto boundary_q_point_h = Kokkos::View<RealType***, Layout, HostMemSpace>("boundary_q_point_h", n_boundary_faces_, n_q_points_face, dim);
    auto boundary_normal_h = Kokkos::View<RealType***, Layout, HostMemSpace>("boundary_normal_h", n_boundary_faces_, n_q_points_face, dim);
    auto boundary_phi_h = Kokkos::View<RealType***, Layout, HostMemSpace>("boundary_phi_h", n_boundary_faces_, n_q_points_face, n_dofs_per_cell);
    auto boundary_dofs_h = Kokkos::View<uint32_t**, Layout, HostMemSpace>("boundary_dofs_h", n_boundary_faces_, n_dofs_per_cell);
    auto boundary_id_h = Kokkos::View<uint32_t*, Layout, HostMemSpace>("boundary_id_h", n_boundary_faces_);

    uint32_t interior_face_idx = 0;
    uint32_t periodic_face_idx = 0;
    uint32_t boundary_face_idx = 0;
    std::vector<uint32_t> neighbor_dof_indices;

    for (auto cell : dof_handler_.active_cell_range()) {
      cell.get_dof_indices(dof_indices);
      for (unsigned int lf = 0; lf < SimplexTopology<dim>::faces_per_cell; ++lf) {
        fe_face_values_.reinit(cell, lf);
        auto face = cell.face(lf);

        if (face.at_boundary()) {
          const uint32_t f = boundary_face_idx++;
          boundary_id_h(f) = face.boundary_id();
          for (unsigned int i = 0; i < n_dofs_per_cell; ++i) boundary_dofs_h(f, i) = dof_indices[i];
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
          if (face.owner_index() == cell.index()) {
            const uint32_t f = periodic_face_idx++;
            auto neighbor_cell = cell.periodic_neighbor(lf);
            neighbor_cell.get_dof_indices(neighbor_dof_indices);
            for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
              periodic_dofs_L_h(f, i) = dof_indices[i];
              periodic_dofs_R_h(f, i) = neighbor_dof_indices[i];
            }
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
            unsigned int neighbor_lf = face.periodic_neighbor_face_index();
            fe_face_values_.reinit(neighbor_cell, neighbor_lf);
            for (unsigned int q = 0; q < n_q_points_face; ++q) {
              unsigned int q_r = n_q_points_face - 1 - q;
              for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
                periodic_phi_R_h(f, q, i) = fe_face_values_.shape_value(i, q_r);
              }
            }
          }
        } else {
          if (face.owner_index() == cell.index()) {
            const uint32_t f = interior_face_idx++;
            auto neighbor_cell = cell.neighbor(lf);
            neighbor_cell.get_dof_indices(neighbor_dof_indices);
            for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
              interior_dofs_L_h(f, i) = dof_indices[i];
              interior_dofs_R_h(f, i) = neighbor_dof_indices[i];
            }
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
            unsigned int neighbor_lf = cell.neighbor_face_index(lf);
            fe_face_values_.reinit(neighbor_cell, neighbor_lf);
            for (unsigned int q = 0; q < n_q_points_face; ++q) {
              unsigned int q_r = n_q_points_face - 1 - q;
              for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
                interior_phi_R_h(f, q, i) = fe_face_values_.shape_value(i, q_r);
              }
            }
          }
        }
      }
    }

    interior_JxW_ = Kokkos::View<RealType**, Layout, DeviceMemSpace>("interior_JxW", n_interior_faces_, n_q_points_face);
    interior_q_point_ = Kokkos::View<RealType***, Layout, DeviceMemSpace>("interior_q_point", n_interior_faces_, n_q_points_face, dim);
    interior_normal_ = Kokkos::View<RealType***, Layout, DeviceMemSpace>("interior_normal", n_interior_faces_, n_q_points_face, dim);
    interior_phi_L_ = Kokkos::View<RealType***, Layout, DeviceMemSpace>("interior_phi_L", n_interior_faces_, n_q_points_face, n_dofs_per_cell);
    interior_phi_R_ = Kokkos::View<RealType***, Layout, DeviceMemSpace>("interior_phi_R", n_interior_faces_, n_q_points_face, n_dofs_per_cell);
    interior_dofs_L_ = Kokkos::View<uint32_t**, Layout, DeviceMemSpace>("interior_dofs_L", n_interior_faces_, n_dofs_per_cell);
    interior_dofs_R_ = Kokkos::View<uint32_t**, Layout, DeviceMemSpace>("interior_dofs_R", n_interior_faces_, n_dofs_per_cell);

    periodic_JxW_ = Kokkos::View<RealType**, Layout, DeviceMemSpace>("periodic_JxW", n_periodic_faces_, n_q_points_face);
    periodic_q_point_ = Kokkos::View<RealType***, Layout, DeviceMemSpace>("periodic_q_point", n_periodic_faces_, n_q_points_face, dim);
    periodic_normal_ = Kokkos::View<RealType***, Layout, DeviceMemSpace>("periodic_normal", n_periodic_faces_, n_q_points_face, dim);
    periodic_phi_L_ = Kokkos::View<RealType***, Layout, DeviceMemSpace>("periodic_phi_L", n_periodic_faces_, n_q_points_face, n_dofs_per_cell);
    periodic_phi_R_ = Kokkos::View<RealType***, Layout, DeviceMemSpace>("periodic_phi_R", n_periodic_faces_, n_q_points_face, n_dofs_per_cell);
    periodic_dofs_L_ = Kokkos::View<uint32_t**, Layout, DeviceMemSpace>("periodic_dofs_L", n_periodic_faces_, n_dofs_per_cell);
    periodic_dofs_R_ = Kokkos::View<uint32_t**, Layout, DeviceMemSpace>("periodic_dofs_R", n_periodic_faces_, n_dofs_per_cell);

    boundary_JxW_ = Kokkos::View<RealType**, Layout, DeviceMemSpace>("boundary_JxW", n_boundary_faces_, n_q_points_face);
    boundary_q_point_ = Kokkos::View<RealType***, Layout, DeviceMemSpace>("boundary_q_point", n_boundary_faces_, n_q_points_face, dim);
    boundary_normal_ = Kokkos::View<RealType***, Layout, DeviceMemSpace>("boundary_normal", n_boundary_faces_, n_q_points_face, dim);
    boundary_phi_ = Kokkos::View<RealType***, Layout, DeviceMemSpace>("boundary_phi", n_boundary_faces_, n_q_points_face, n_dofs_per_cell);
    boundary_dofs_ = Kokkos::View<uint32_t**, Layout, DeviceMemSpace>("boundary_dofs", n_boundary_faces_, n_dofs_per_cell);
    boundary_id_ = Kokkos::View<uint32_t*, Layout, DeviceMemSpace>("boundary_id", n_boundary_faces_);

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
    MassMatrix<dim, double> invm(fe_values_);
    invm.assemble(dof_handler_);
    invm.invert();
    invm.check_inverse();
    invm_ = invm.device_inverse();
  }

  void compute_volume_residual()
  {
    auto cell_dofs = cell_dofs_;
    auto phi = phi_;
    auto grad_phi = grad_phi_;
    auto JxW = JxW_;
    
    auto rho = rho_.view();
    auto rho_u = rho_u_.view();
    auto rho_v = rho_v_.view();
    auto rho_w = rho_w_.view();
    auto rho_E = rho_E_.view();
    
    auto res_rho = res_rho_.view();
    auto res_rho_u = res_rho_u_.view();
    auto res_rho_v = res_rho_v_.view();
    auto res_rho_w = res_rho_w_.view();
    auto res_rho_E = res_rho_E_.view();

    const unsigned int n_q_points = fe_values_.n_q_points();
    const unsigned int n_dofs_per_cell = dof_handler_.n_dofs_per_cell();

    Kokkos::parallel_for("VolumeResidual", n_cells_, KOKKOS_LAMBDA(const uint32_t c) {
      for (unsigned int q = 0; q < n_q_points; ++q) {
        RealType r = 0, ru = 0, rv = 0, rw = 0, rE = 0;
        for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
          const uint32_t dof = cell_dofs(c, i);
          const RealType p = phi(c, i, q);
          r  += rho(dof) * p;
          ru += rho_u(dof) * p;
          rv += rho_v(dof) * p;
          rw += rho_w(dof) * p;
          rE += rho_E(dof) * p;
        }

        RealType u, v, w, p_val;
        Flux<RealType>::conservative_to_primitive(r, ru, rv, rw, rE, u, v, w, p_val);

        RealType Fx_rho, Fx_rhou, Fx_rhov, Fx_rhow, Fx_rhoE;
        RealType Fy_rho, Fy_rhou, Fy_rhov, Fy_rhow, Fy_rhoE;
        RealType Fz_rho, Fz_rhou, Fz_rhov, Fz_rhow, Fz_rhoE;

        Flux<RealType>::euler_flux(r, u, v, w, p_val, rE, 1.0, 0.0, 0.0, Fx_rho, Fx_rhou, Fx_rhov, Fx_rhow, Fx_rhoE);
        Flux<RealType>::euler_flux(r, u, v, w, p_val, rE, 0.0, 1.0, 0.0, Fy_rho, Fy_rhou, Fy_rhov, Fy_rhow, Fy_rhoE);
        Flux<RealType>::euler_flux(r, u, v, w, p_val, rE, 0.0, 0.0, 1.0, Fz_rho, Fz_rhou, Fz_rhov, Fz_rhow, Fz_rhoE);

        const RealType jxw = JxW(c, q);
        for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
          const uint32_t dof = cell_dofs(c, i);
          const RealType dphi_dx = grad_phi(c, i, q, 0);
          const RealType dphi_dy = grad_phi(c, i, q, 1);
          const RealType dphi_dz = grad_phi(c, i, q, 2);

          Kokkos::atomic_add(&res_rho(dof),   jxw * (Fx_rho * dphi_dx   + Fy_rho * dphi_dy   + Fz_rho * dphi_dz));
          Kokkos::atomic_add(&res_rho_u(dof), jxw * (Fx_rhou * dphi_dx  + Fy_rhou * dphi_dy  + Fz_rhou * dphi_dz));
          Kokkos::atomic_add(&res_rho_v(dof), jxw * (Fx_rhov * dphi_dx  + Fy_rhov * dphi_dy  + Fz_rhov * dphi_dz));
          Kokkos::atomic_add(&res_rho_w(dof), jxw * (Fx_rhow * dphi_dx  + Fy_rhow * dphi_dy  + Fz_rhow * dphi_dz));
          Kokkos::atomic_add(&res_rho_E(dof), jxw * (Fx_rhoE * dphi_dx  + Fy_rhoE * dphi_dy  + Fz_rhoE * dphi_dz));
        }
      }
    });
  }

  void compute_interior_face_residual()
  {
    auto dofs_L = interior_dofs_L_; auto dofs_R = interior_dofs_R_;
    auto phi_L = interior_phi_L_;   auto phi_R = interior_phi_R_;
    auto normal = interior_normal_; auto JxW = interior_JxW_;
    
    auto rho = rho_.view(); auto rho_u = rho_u_.view(); auto rho_v = rho_v_.view(); auto rho_w = rho_w_.view(); auto rho_E = rho_E_.view();
    auto res_rho = res_rho_.view(); auto res_rho_u = res_rho_u_.view(); auto res_rho_v = res_rho_v_.view(); auto res_rho_w = res_rho_w_.view(); auto res_rho_E = res_rho_E_.view();

    const unsigned int n_q_points_face = fe_face_values_.n_q_points();
    const unsigned int n_dofs_per_cell = dof_handler_.n_dofs_per_cell();

    Kokkos::parallel_for("InteriorFaceResidual", n_interior_faces_, KOKKOS_LAMBDA(const uint32_t f) {
      for (unsigned int q = 0; q < n_q_points_face; ++q) {
        RealType rL=0, ruL=0, rvL=0, rwL=0, rEL=0;
        RealType rR=0, ruR=0, rvR=0, rwR=0, rER=0;
        
        for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
          const uint32_t dL = dofs_L(f, i); const RealType pL = phi_L(f, q, i);
          rL += rho(dL)*pL; ruL += rho_u(dL)*pL; rvL += rho_v(dL)*pL; rwL += rho_w(dL)*pL; rEL += rho_E(dL)*pL;

          const uint32_t dR = dofs_R(f, i); const RealType pR = phi_R(f, q, i);
          rR += rho(dR)*pR; ruR += rho_u(dR)*pR; rvR += rho_v(dR)*pR; rwR += rho_w(dR)*pR; rER += rho_E(dR)*pR;
        }

        const RealType nx = normal(f, q, 0); const RealType ny = normal(f, q, 1); const RealType nz = normal(f, q, 2);
        RealType F_r, F_ru, F_rv, F_rw, F_rE, s_mag;

        Flux<RealType>::roe_flux(rL, ruL, rvL, rwL, rEL, rR, ruR, rvR, rwR, rER, nx, ny, nz, F_r, F_ru, F_rv, F_rw, F_rE, s_mag);

        const RealType jxw = JxW(f, q);
        for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
          const uint32_t dL = dofs_L(f, i); const RealType pL = phi_L(f, q, i);
          Kokkos::atomic_sub(&res_rho(dL), jxw * F_r * pL);
          Kokkos::atomic_sub(&res_rho_u(dL), jxw * F_ru * pL);
          Kokkos::atomic_sub(&res_rho_v(dL), jxw * F_rv * pL);
          Kokkos::atomic_sub(&res_rho_w(dL), jxw * F_rw * pL);
          Kokkos::atomic_sub(&res_rho_E(dL), jxw * F_rE * pL);

          const uint32_t dR = dofs_R(f, i); const RealType pR = phi_R(f, q, i);
          Kokkos::atomic_add(&res_rho(dR), jxw * F_r * pR);
          Kokkos::atomic_add(&res_rho_u(dR), jxw * F_ru * pR);
          Kokkos::atomic_add(&res_rho_v(dR), jxw * F_rv * pR);
          Kokkos::atomic_add(&res_rho_w(dR), jxw * F_rw * pR);
          Kokkos::atomic_add(&res_rho_E(dR), jxw * F_rE * pR);
        }
      }
    });
  }

  void compute_periodic_face_residual()
  {
    auto dofs_L = periodic_dofs_L_; auto dofs_R = periodic_dofs_R_;
    auto phi_L = periodic_phi_L_;   auto phi_R = periodic_phi_R_;
    auto normal = periodic_normal_; auto JxW = periodic_JxW_;
    
    auto rho = rho_.view(); auto rho_u = rho_u_.view(); auto rho_v = rho_v_.view(); auto rho_w = rho_w_.view(); auto rho_E = rho_E_.view();
    auto res_rho = res_rho_.view(); auto res_rho_u = res_rho_u_.view(); auto res_rho_v = res_rho_v_.view(); auto res_rho_w = res_rho_w_.view(); auto res_rho_E = res_rho_E_.view();

    const unsigned int n_q_points_face = fe_face_values_.n_q_points();
    const unsigned int n_dofs_per_cell = dof_handler_.n_dofs_per_cell();

    Kokkos::parallel_for("PeriodicFaceResidual", n_periodic_faces_, KOKKOS_LAMBDA(const uint32_t f) {
      for (unsigned int q = 0; q < n_q_points_face; ++q) {
        RealType rL=0, ruL=0, rvL=0, rwL=0, rEL=0;
        RealType rR=0, ruR=0, rvR=0, rwR=0, rER=0;
        
        for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
          const uint32_t dL = dofs_L(f, i); const RealType pL = phi_L(f, q, i);
          rL += rho(dL)*pL; ruL += rho_u(dL)*pL; rvL += rho_v(dL)*pL; rwL += rho_w(dL)*pL; rEL += rho_E(dL)*pL;

          const uint32_t dR = dofs_R(f, i); const RealType pR = phi_R(f, q, i);
          rR += rho(dR)*pR; ruR += rho_u(dR)*pR; rvR += rho_v(dR)*pR; rwR += rho_w(dR)*pR; rER += rho_E(dR)*pR;
        }

        const RealType nx = normal(f, q, 0); const RealType ny = normal(f, q, 1); const RealType nz = normal(f, q, 2);
        RealType F_r, F_ru, F_rv, F_rw, F_rE, s_mag;

        Flux<RealType>::roe_flux(rL, ruL, rvL, rwL, rEL, rR, ruR, rvR, rwR, rER, nx, ny, nz, F_r, F_ru, F_rv, F_rw, F_rE, s_mag);

        const RealType jxw = JxW(f, q);
        for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
          const uint32_t dL = dofs_L(f, i); const RealType pL = phi_L(f, q, i);
          Kokkos::atomic_sub(&res_rho(dL), jxw * F_r * pL);
          Kokkos::atomic_sub(&res_rho_u(dL), jxw * F_ru * pL);
          Kokkos::atomic_sub(&res_rho_v(dL), jxw * F_rv * pL);
          Kokkos::atomic_sub(&res_rho_w(dL), jxw * F_rw * pL);
          Kokkos::atomic_sub(&res_rho_E(dL), jxw * F_rE * pL);

          const uint32_t dR = dofs_R(f, i); const RealType pR = phi_R(f, q, i);
          Kokkos::atomic_add(&res_rho(dR), jxw * F_r * pR);
          Kokkos::atomic_add(&res_rho_u(dR), jxw * F_ru * pR);
          Kokkos::atomic_add(&res_rho_v(dR), jxw * F_rv * pR);
          Kokkos::atomic_add(&res_rho_w(dR), jxw * F_rw * pR);
          Kokkos::atomic_add(&res_rho_E(dR), jxw * F_rE * pR);
        }
      }
    });
  }

  void compute_boundary_face_residual(RealType time)
  {
    auto dofs = boundary_dofs_; auto phi = boundary_phi_; auto b_id = boundary_id_;
    auto normal = boundary_normal_; auto JxW = boundary_JxW_; auto qp = boundary_q_point_;
    
    auto rho = rho_.view(); auto rho_u = rho_u_.view(); auto rho_v = rho_v_.view(); auto rho_w = rho_w_.view(); auto rho_E = rho_E_.view();
    auto res_rho = res_rho_.view(); auto res_rho_u = res_rho_u_.view(); auto res_rho_v = res_rho_v_.view(); auto res_rho_w = res_rho_w_.view(); auto res_rho_E = res_rho_E_.view();

    const unsigned int n_q_points_face = fe_face_values_.n_q_points();
    const unsigned int n_dofs_per_cell = dof_handler_.n_dofs_per_cell();

    Kokkos::parallel_for("BoundaryFaceResidual", n_boundary_faces_, KOKKOS_LAMBDA(const uint32_t f) {
      const uint32_t id = b_id(f);
      for (unsigned int q = 0; q < n_q_points_face; ++q) {
        RealType rL=0, ruL=0, rvL=0, rwL=0, rEL=0;
        
        for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
          const uint32_t dL = dofs(f, i); const RealType pL = phi(f, q, i);
          rL += rho(dL)*pL; ruL += rho_u(dL)*pL; rvL += rho_v(dL)*pL; rwL += rho_w(dL)*pL; rEL += rho_E(dL)*pL;
        }

        const RealType nx = normal(f, q, 0); const RealType ny = normal(f, q, 1); const RealType nz = normal(f, q, 2);
        RealType F_r, F_ru, F_rv, F_rw, F_rE, s_mag;

        if (id == BoundaryId::InviscidWall) {
          Flux<RealType>::inviscid_wall_flux(rL, ruL, rvL, rwL, rEL, nx, ny, nz, F_r, F_ru, F_rv, F_rw, F_rE, s_mag);
        } else if (id == BoundaryId::SubsonicOutflow) {
          Flux<RealType>::subsonic_outflow_flux(rL, ruL, rvL, rwL, rEL, nx, ny, nz, F_r, F_ru, F_rv, F_rw, F_rE, s_mag);
        } else if (id == BoundaryId::UnsteadySubsonicInflow || id == BoundaryId::SubsonicInflow) {
          const RealType y_face = qp(f, q, 1);
          Flux<RealType>::unsteady_subsonic_inflow_flux(rL, ruL, rvL, rwL, rEL, nx, ny, nz, y_face, time, F_r, F_ru, F_rv, F_rw, F_rE, s_mag);
        } else {
          Flux<RealType>::inviscid_wall_flux(rL, ruL, rvL, rwL, rEL, nx, ny, nz, F_r, F_ru, F_rv, F_rw, F_rE, s_mag);
        }

        const RealType jxw = JxW(f, q);
        for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
          const uint32_t dL = dofs(f, i); const RealType pL = phi(f, q, i);
          Kokkos::atomic_sub(&res_rho(dL), jxw * F_r * pL);
          Kokkos::atomic_sub(&res_rho_u(dL), jxw * F_ru * pL);
          Kokkos::atomic_sub(&res_rho_v(dL), jxw * F_rv * pL);
          Kokkos::atomic_sub(&res_rho_w(dL), jxw * F_rw * pL);
          Kokkos::atomic_sub(&res_rho_E(dL), jxw * F_rE * pL);
        }
      }
    });
  }

  void compute_face_residual(RealType time = 0.0)
  {
    compute_interior_face_residual();
    compute_periodic_face_residual();
    compute_boundary_face_residual(time);
  }

  void apply_inverse_mass_and_update(RealType c1, RealType c2, RealType dt_scale)
  {
    auto cell_dofs = cell_dofs_;
    auto invm = invm_;
    
    auto rho = rho_.view(); auto rho_old = rho_old_.view();
    auto rho_u = rho_u_.view(); auto rho_u_old = rho_u_old_.view();
    auto rho_v = rho_v_.view(); auto rho_v_old = rho_v_old_.view();
    auto rho_w = rho_w_.view(); auto rho_w_old = rho_w_old_.view();
    auto rho_E = rho_E_.view(); auto rho_E_old = rho_E_old_.view();
    
    auto res_rho = res_rho_.view();
    auto res_rho_u = res_rho_u_.view();
    auto res_rho_v = res_rho_v_.view();
    auto res_rho_w = res_rho_w_.view();
    auto res_rho_E = res_rho_E_.view();

    const unsigned int n_dofs_per_cell = dof_handler_.n_dofs_per_cell();

    Kokkos::parallel_for("Update", n_cells_, KOKKOS_LAMBDA(const uint32_t c) {
      for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
        RealType drho = 0, drhou = 0, drhov = 0, drhow = 0, drhoE = 0;
        
        for (unsigned int j = 0; j < n_dofs_per_cell; ++j) {
          const uint32_t dof_j = cell_dofs(c, j);
          const RealType m_inv = invm(c, i, j);
          drho  += m_inv * res_rho(dof_j);
          drhou += m_inv * res_rho_u(dof_j);
          drhov += m_inv * res_rho_v(dof_j);
          drhow += m_inv * res_rho_w(dof_j);
          drhoE += m_inv * res_rho_E(dof_j);
        }
        
        const uint32_t dof_i = cell_dofs(c, i);
        rho(dof_i)   = c1 * rho(dof_i)   + c2 * rho_old(dof_i)   + dt_scale * drho;
        rho_u(dof_i) = c1 * rho_u(dof_i) + c2 * rho_u_old(dof_i) + dt_scale * drhou;
        rho_v(dof_i) = c1 * rho_v(dof_i) + c2 * rho_v_old(dof_i) + dt_scale * drhov;
        rho_w(dof_i) = c1 * rho_w(dof_i) + c2 * rho_w_old(dof_i) + dt_scale * drhow;
        rho_E(dof_i) = c1 * rho_E(dof_i) + c2 * rho_E_old(dof_i) + dt_scale * drhoE;
      }
    });
  }

  void ssp_rk3_step(RealType cfl)
  {
    // A fixed global DT is assumed for steady state RK3. Replace with compute_global_dt() if you have it.
    const RealType dt = 1e-4; 
    
    Kokkos::deep_copy(rho_old_.view(), rho_.view());
    Kokkos::deep_copy(rho_u_old_.view(), rho_u_.view());
    Kokkos::deep_copy(rho_v_old_.view(), rho_v_.view());
    Kokkos::deep_copy(rho_w_old_.view(), rho_w_.view());
    Kokkos::deep_copy(rho_E_old_.view(), rho_E_.view());

    // RK3 Stage 1
    zero_residuals();
    compute_volume_residual();
    compute_face_residual();
    apply_inverse_mass_and_update(0.0, 1.0, dt);

    // RK3 Stage 2
    zero_residuals();
    compute_volume_residual();
    compute_face_residual();
    apply_inverse_mass_and_update(0.25, 0.75, 0.25 * dt);

    // RK3 Stage 3
    zero_residuals();
    compute_volume_residual();
    compute_face_residual();
    apply_inverse_mass_and_update(2.0/3.0, 1.0/3.0, (2.0/3.0) * dt);
  }

  RealType residual_norm()
  {
    RealType norm_sq = 0;
    auto res_rho = res_rho_.view();
    Kokkos::parallel_reduce("Norm", n_dofs_, KOKKOS_LAMBDA(const uint32_t i, RealType& lsum) {
      lsum += res_rho(i) * res_rho(i);
    }, norm_sq);
    return std::sqrt(norm_sq);
  }

  double solve_steady_state(unsigned int max_iter = 100000, RealType cfl = Parameters<RealType>::cfl_max, unsigned int write_interval = 1000, bool use_abs_tol = false, double abs_tol = 1e-5)
  {
    const std::string deg_str = "p" + std::to_string(degree_);
    std::cout << "Starting steady-state solve with " << max_iter << " iterations" << std::endl;
    
    zero_residuals();
    compute_volume_residual();
    compute_face_residual();
    const RealType res0 = residual_norm();
    std::cout << "Initial residual: " << res0 << std::endl;

    for (unsigned int iter = 0; iter < max_iter; ++iter) {
      ssp_rk3_step(cfl);
      
      zero_residuals();
      compute_volume_residual();
      compute_face_residual();
      
      const RealType res = residual_norm();
      if (std::isnan(res) || std::isinf(res)) {
        std::cout << "Solution diverged!" << std::endl;
        break;
      }
      
      if (iter % write_interval == 0) {
        std::cout << "Iter " << iter << " Res: " << res << std::endl;
      }

      if (use_abs_tol && res < abs_tol) {
        std::cout << "Converged to absolute tolerance." << std::endl;
        break;
      }
    }
    return 0.0;
  }

  // ---------------------------------------------------------------------------
  // compute_local_dt: CFL-based per-cell timestep (ported from 2D)
  // ---------------------------------------------------------------------------
  void compute_local_dt(RealType cfl)
  {
    auto d_rho   = rho_.view();
    auto d_rho_u = rho_u_.view();
    auto d_rho_v = rho_v_.view();
    auto d_rho_w = rho_w_.view();
    auto d_rho_E = rho_E_.view();
    auto d_dt    = dt_.view();
    auto indices = cell_dofs_;
    auto areas   = cell_area_;

    const auto     n_dofs_per_cell = dof_handler_.n_dofs_per_cell();
    const RealType gamma      = Parameters<RealType>::gamma;
    const RealType p_order    = static_cast<RealType>(degree_);
    const RealType dg_scaling = RealType(1) / (RealType(2) * p_order + RealType(1));

    Kokkos::parallel_for(
      "compute_local_dt", n_cells_, KOKKOS_LAMBDA(int k) {
        RealType rho_avg = 0, rhou_avg = 0, rhov_avg = 0, rhow_avg = 0, rhoE_avg = 0;
        for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
          const uint32_t idx = indices(k, i);
          rho_avg  += d_rho(idx);
          rhou_avg += d_rho_u(idx);
          rhov_avg += d_rho_v(idx);
          rhow_avg += d_rho_w(idx);
          rhoE_avg += d_rho_E(idx);
        }
        rho_avg  /= n_dofs_per_cell;
        rhou_avg /= n_dofs_per_cell;
        rhov_avg /= n_dofs_per_cell;
        rhow_avg /= n_dofs_per_cell;
        rhoE_avg /= n_dofs_per_cell;

        const RealType u       = rhou_avg / rho_avg;
        const RealType v       = rhov_avg / rho_avg;
        const RealType w       = rhow_avg / rho_avg;
        const RealType vel_mag = Kokkos::sqrt(u*u + v*v + w*w);
        const RealType p       = (gamma - RealType(1)) *
                                 (rhoE_avg - RealType(0.5) * rho_avg * (u*u + v*v + w*w));
        const RealType a       = Kokkos::sqrt(gamma * p / rho_avg);

        // Use cube-root of volume as characteristic length for 3D
        const RealType h  = Kokkos::pow(areas(k), RealType(1)/RealType(3));
        const RealType dt = cfl * dg_scaling * h / (vel_mag + a);

        for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
          d_dt(indices(k, i)) = dt;
      });
    Kokkos::fence();
  }

  // ---------------------------------------------------------------------------
  // compute_global_dt: minimum CFL timestep across all cells (ported from 2D)
  // ---------------------------------------------------------------------------
  void compute_global_dt(RealType cfl)
  {
    compute_local_dt(cfl);

    auto d_dt = dt_.view();
    RealType dt_min = RealType(0);
    Kokkos::parallel_reduce(
      "find_min_dt",
      Kokkos::RangePolicy<>(0, n_dofs_),
      KOKKOS_LAMBDA(int i, RealType& local_min) {
        local_min = Kokkos::min(local_min, d_dt(i));
      },
      Kokkos::Min<RealType>(dt_min));
    Kokkos::fence();

    Kokkos::parallel_for(
      "fill_global_dt",
      Kokkos::RangePolicy<>(0, n_dofs_),
      KOKKOS_LAMBDA(int i) { d_dt(i) = dt_min; });
    Kokkos::fence();
  }

  // ---------------------------------------------------------------------------
  // test_freestream_preservation (ported from 2D, extended to 5 fields)
  // ---------------------------------------------------------------------------
  void test_freestream_preservation(unsigned int n_steps = 10)
  {
    auto rho_init   = Kokkos::create_mirror_view(rho_.view());
    auto rho_u_init = Kokkos::create_mirror_view(rho_u_.view());
    auto rho_v_init = Kokkos::create_mirror_view(rho_v_.view());
    auto rho_w_init = Kokkos::create_mirror_view(rho_w_.view());
    auto rho_E_init = Kokkos::create_mirror_view(rho_E_.view());
    Kokkos::deep_copy(rho_init,   rho_.view());
    Kokkos::deep_copy(rho_u_init, rho_u_.view());
    Kokkos::deep_copy(rho_v_init, rho_v_.view());
    Kokkos::deep_copy(rho_w_init, rho_w_.view());
    Kokkos::deep_copy(rho_E_init, rho_E_.view());

    for (unsigned int step = 0; step < n_steps; ++step) {
      ssp_rk3_step(Parameters<RealType>::cfl_max);

      zero_residuals();
      compute_volume_residual();
      compute_face_residual();

      auto res_rho_h   = Kokkos::create_mirror_view(res_rho_.view());
      auto res_rho_u_h = Kokkos::create_mirror_view(res_rho_u_.view());
      auto res_rho_v_h = Kokkos::create_mirror_view(res_rho_v_.view());
      auto res_rho_w_h = Kokkos::create_mirror_view(res_rho_w_.view());
      auto res_rho_E_h = Kokkos::create_mirror_view(res_rho_E_.view());
      Kokkos::deep_copy(res_rho_h,   res_rho_.view());
      Kokkos::deep_copy(res_rho_u_h, res_rho_u_.view());
      Kokkos::deep_copy(res_rho_v_h, res_rho_v_.view());
      Kokkos::deep_copy(res_rho_w_h, res_rho_w_.view());
      Kokkos::deep_copy(res_rho_E_h, res_rho_E_.view());

      RealType max_res = 0;
      for (unsigned int i = 0; i < n_dofs_; ++i) {
        max_res = Kokkos::max(max_res, Kokkos::abs(res_rho_h(i)));
        max_res = Kokkos::max(max_res, Kokkos::abs(res_rho_u_h(i)));
        max_res = Kokkos::max(max_res, Kokkos::abs(res_rho_v_h(i)));
        max_res = Kokkos::max(max_res, Kokkos::abs(res_rho_w_h(i)));
        max_res = Kokkos::max(max_res, Kokkos::abs(res_rho_E_h(i)));
      }
      std::cout << "Step " << step << " max_residual=" << max_res << std::endl;

      auto rho_h   = Kokkos::create_mirror_view(rho_.view());
      auto rho_u_h = Kokkos::create_mirror_view(rho_u_.view());
      auto rho_v_h = Kokkos::create_mirror_view(rho_v_.view());
      auto rho_w_h = Kokkos::create_mirror_view(rho_w_.view());
      auto rho_E_h = Kokkos::create_mirror_view(rho_E_.view());
      Kokkos::deep_copy(rho_h,   rho_.view());
      Kokkos::deep_copy(rho_u_h, rho_u_.view());
      Kokkos::deep_copy(rho_v_h, rho_v_.view());
      Kokkos::deep_copy(rho_w_h, rho_w_.view());
      Kokkos::deep_copy(rho_E_h, rho_E_.view());

      RealType max_state_change = 0;
      for (unsigned int i = 0; i < n_dofs_; ++i) {
        max_state_change = Kokkos::max(max_state_change, Kokkos::abs(rho_h(i)   - rho_init(i)));
        max_state_change = Kokkos::max(max_state_change, Kokkos::abs(rho_u_h(i) - rho_u_init(i)));
        max_state_change = Kokkos::max(max_state_change, Kokkos::abs(rho_v_h(i) - rho_v_init(i)));
        max_state_change = Kokkos::max(max_state_change, Kokkos::abs(rho_w_h(i) - rho_w_init(i)));
        max_state_change = Kokkos::max(max_state_change, Kokkos::abs(rho_E_h(i) - rho_E_init(i)));
      }
      std::cout << "Step " << step << " max_state_change=" << max_state_change << std::endl;
    }
  }

  // ---------------------------------------------------------------------------
  // solve_unsteady: time-marching SSP-RK3 with global dt (ported from 2D)
  // ---------------------------------------------------------------------------
  double solve_unsteady(RealType t_end,
                        RealType cfl = Parameters<RealType>::cfl_max,
                        unsigned int write_interval = 100)
  {
    const std::string deg_str = "p" + std::to_string(degree_);
    std::cout << "Starting unsteady solve, t_end=" << t_end << std::endl;

    RealType     t    = 0.0;
    unsigned int iter = 0;

    write_solution("solution_" + deg_str + "_t0000.vtu", 0);

    while (t < t_end) {
      Kokkos::deep_copy(rho_old_.view(),   rho_.view());
      Kokkos::deep_copy(rho_u_old_.view(), rho_u_.view());
      Kokkos::deep_copy(rho_v_old_.view(), rho_v_.view());
      Kokkos::deep_copy(rho_w_old_.view(), rho_w_.view());
      Kokkos::deep_copy(rho_E_old_.view(), rho_E_.view());

      compute_global_dt(cfl);

      auto dt_h = Kokkos::create_mirror_view(dt_.view());
      Kokkos::deep_copy(dt_h, dt_.view());
      const RealType dt = Kokkos::min(dt_h(0), t_end - t);

      if (dt < dt_h(0)) {
        auto d_dt = dt_.view();
        Kokkos::parallel_for(
          "fill_clamped_dt",
          Kokkos::RangePolicy<>(0, n_dofs_),
          KOKKOS_LAMBDA(int i) { d_dt(i) = dt; });
        Kokkos::fence();
      }

      // SSP-RK3 Stage 1
      zero_residuals();
      compute_volume_residual();
      compute_face_residual(t);
      apply_inverse_mass_and_update(0.0, 1.0, dt);

      // SSP-RK3 Stage 2
      zero_residuals();
      compute_volume_residual();
      compute_face_residual(t + dt);
      apply_inverse_mass_and_update(0.75, 0.25, 0.25 * dt);

      // SSP-RK3 Stage 3
      zero_residuals();
      compute_volume_residual();
      compute_face_residual(t + 0.5 * dt);
      apply_inverse_mass_and_update(1.0/3.0, 2.0/3.0, (2.0/3.0) * dt);

      t += dt;
      iter++;

      if (iter % write_interval == 0) {
        std::ostringstream fname;
        fname << "solution_" << deg_str << "_"
              << std::setw(8) << std::setfill('0') << iter << ".vtu";
        write_solution(fname.str(), iter, t);
        std::cout << "t=" << std::scientific << std::setprecision(4) << t
                  << "  iter=" << iter << "  dt=" << dt << std::endl;
      }

      if (std::isnan(t) || std::isinf(t)) {
        std::cout << "Solution diverged at t=" << t << std::endl;
        break;
      }
    }

    std::cout << "Finished at t=" << t << " iter=" << iter << std::endl;
    write_solution("solution_" + deg_str + "_final.vtu", iter, t);
    return t;
  }
};

// -----------------------------------------------------------------------------
// interpolate_solution: p-multigrid prolongation for 3D (5 fields)
// Ported from 2D version; evaluates low-order polynomial at high-order nodes.
// -----------------------------------------------------------------------------
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
  const Kokkos::View<RealType*, Layout, HostMemSpace>& rhow_lo,
  const Kokkos::View<RealType*, Layout, HostMemSpace>& rhoE_lo,
  Kokkos::View<RealType*, Layout, HostMemSpace>& rho_hi,
  Kokkos::View<RealType*, Layout, HostMemSpace>& rhou_hi,
  Kokkos::View<RealType*, Layout, HostMemSpace>& rhov_hi,
  Kokkos::View<RealType*, Layout, HostMemSpace>& rhow_hi,
  Kokkos::View<RealType*, Layout, HostMemSpace>& rhoE_hi)
{
  const auto n_dofs_lo = fe_lo.n_dofs();
  const auto n_dofs_hi = fe_hi.n_dofs();

  std::vector<uint32_t> dof_indices_lo, dof_indices_hi;

  for (auto cell : dof_handler_hi.active_cell_range()) {
    const auto k = cell.index();
    cell.get_dof_indices(dof_indices_hi);

    auto cell_lo = dof_handler_lo.cell(k);
    cell_lo.get_dof_indices(dof_indices_lo);

    for (unsigned int i = 0; i < n_dofs_hi; ++i) {
      const auto xi = fe_hi.node(i);

      RealType rho_val = 0, rhou_val = 0, rhov_val = 0, rhow_val = 0, rhoE_val = 0;
      for (unsigned int j = 0; j < n_dofs_lo; ++j) {
        const RealType phi_j = fe_lo.shape_value(j, xi);
        const uint32_t dof_j = dof_indices_lo[j];
        rho_val  += rho_lo(dof_j)  * phi_j;
        rhou_val += rhou_lo(dof_j) * phi_j;
        rhov_val += rhov_lo(dof_j) * phi_j;
        rhow_val += rhow_lo(dof_j) * phi_j;
        rhoE_val += rhoE_lo(dof_j) * phi_j;
      }

      const uint32_t dof_i = dof_indices_hi[i];
      rho_hi(dof_i)  = rho_val;
      rhou_hi(dof_i) = rhou_val;
      rhov_hi(dof_i) = rhov_val;
      rhow_hi(dof_i) = rhow_val;
      rhoE_hi(dof_i) = rhoE_val;
    }
  }
}

// -----------------------------------------------------------------------------
// MAIN ROUTINE
// -----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  Kokkos::initialize(argc, argv);
  {
    Triangulation<3> tria; // 3D Triangulation
    
    // Replace with your actual mesh file path
    read_gri(tria, "mesh.gri");

    // 3D Initial Condition (Returns 5-Tuple)
    auto ic = [](double x, double y, double z) {
      double rho = 1.0;
      double u = 1.0;
      double v = 0.0;
      double w = 0.0;
      double p = 1.0;
      return std::make_tuple(rho, u, v, w, p);
    };

    double abs_tol = 1e-6;

    // Helper lambda for p-multigrid prolongation
    auto do_interpolate = [&](unsigned int deg_lo, unsigned int deg_hi,
                              const Vector<double, HostMemSpace>& rho_in,
                              const Vector<double, HostMemSpace>& rhou_in,
                              const Vector<double, HostMemSpace>& rhov_in,
                              const Vector<double, HostMemSpace>& rhow_in,
                              const Vector<double, HostMemSpace>& rhoE_in,
                              Vector<double, HostMemSpace>& rho_out,
                              Vector<double, HostMemSpace>& rhou_out,
                              Vector<double, HostMemSpace>& rhov_out,
                              Vector<double, HostMemSpace>& rhow_out,
                              Vector<double, HostMemSpace>& rhoE_out) {
      FE_DGQLegendre<3, double> fe_l(deg_lo), fe_h(deg_hi);
      DoFHandler<3, double> dh_l(tria, fe_l), dh_h(tria, fe_h);
      interpolate_solution(dh_l, dh_h, fe_l, fe_h,
                           rho_in.view(),  rhou_in.view(),  rhov_in.view(),
                           rhow_in.view(), rhoE_in.view(),
                           rho_out.view(), rhou_out.view(), rhov_out.view(),
                           rhow_out.view(), rhoE_out.view());
    };

    // --- Degree 0 Solver ---
    FE_DGQLegendre<3, double> fe0(0);
    QGaussSimplex<3, double> q0(2);
    QGaussSimplex<2, double> fq0(2); // Face Quadrature is 2D in a 3D geometry
    DoFHandler<3, double> dh0(tria, fe0);
    FEValues<3, double> fev0(fe0, q0);
    FEFaceValues<3, double> ffev0(fe0, fq0);
    
    Vector<double, Kokkos::HostSpace> rho0(dh0.n_dofs()), rhou0(dh0.n_dofs()), rhov0(dh0.n_dofs()), rhow0(dh0.n_dofs()), rhoE0(dh0.n_dofs());
    
    EulerSolver<3, double> s0(dh0, fev0, ffev0, 0);
    s0.set_initial_condition(ic);
    s0.solve_steady_state(10000, 0.5, 1000, true, abs_tol);
    s0.write_solution("solution_steady_state_p0.vtu");
    s0.copy_state_to_host(rho0, rhou0, rhov0, rhow0, rhoE0);

    // --- Degree 1 Solver ---
    FE_DGQLegendre<3, double> fe1(1);
    QGaussSimplex<3, double> q1(3);
    QGaussSimplex<2, double> fq1(3);
    DoFHandler<3, double> dh1(tria, fe1);
    FEValues<3, double> fev1(fe1, q1);
    FEFaceValues<3, double> ffev1(fe1, fq1);

    Vector<double, Kokkos::HostSpace> rho1(dh1.n_dofs()), rhou1(dh1.n_dofs()), rhov1(dh1.n_dofs()), rhow1(dh1.n_dofs()), rhoE1(dh1.n_dofs());
    do_interpolate(0, 1, rho0, rhou0, rhov0, rhow0, rhoE0, rho1, rhou1, rhov1, rhow1, rhoE1);
    
    EulerSolver<3, double> s1(dh1, fev1, ffev1, 1);
    s1.set_state_from_host(rho1, rhou1, rhov1, rhow1, rhoE1);
    s1.solve_steady_state(10000, 0.5, 1000, true, abs_tol);
    s1.write_solution("solution_steady_state_p1.vtu");
    s1.copy_state_to_host(rho1, rhou1, rhov1, rhow1, rhoE1);

    // --- Degree 2 Solver ---
    FE_DGQLegendre<3, double> fe2(2);
    QGaussSimplex<3, double> q2(4);
    QGaussSimplex<2, double> fq2(4);
    DoFHandler<3, double> dh2(tria, fe2);
    FEValues<3, double> fev2(fe2, q2);
    FEFaceValues<3, double> ffev2(fe2, fq2);

    Vector<double, Kokkos::HostSpace> rho2(dh2.n_dofs()), rhou2(dh2.n_dofs()), rhov2(dh2.n_dofs()), rhow2(dh2.n_dofs()), rhoE2(dh2.n_dofs());
    do_interpolate(1, 2, rho1, rhou1, rhov1, rhow1, rhoE1, rho2, rhou2, rhov2, rhow2, rhoE2);
    
    EulerSolver<3, double> s2(dh2, fev2, ffev2, 2);
    s2.set_state_from_host(rho2, rhou2, rhov2, rhow2, rhoE2);
    s2.solve_steady_state(10000, 0.5, 1000, true, abs_tol);
    s2.write_solution("solution_steady_state_p2.vtu");
    s2.copy_state_to_host(rho2, rhou2, rhov2, rhow2, rhoE2);

    // --- Degree 3 Solver ---
    FE_DGQLegendre<3, double> fe3(3);
    QGaussSimplex<3, double> q3(7);
    QGaussSimplex<2, double> fq3(7);
    DoFHandler<3, double> dh3(tria, fe3);
    FEValues<3, double> fev3(fe3, q3);
    FEFaceValues<3, double> ffev3(fe3, fq3);

    Vector<double, Kokkos::HostSpace> rho3(dh3.n_dofs()), rhou3(dh3.n_dofs()), rhov3(dh3.n_dofs()), rhow3(dh3.n_dofs()), rhoE3(dh3.n_dofs());
    do_interpolate(2, 3, rho2, rhou2, rhov2, rhow2, rhoE2, rho3, rhou3, rhov3, rhow3, rhoE3);
    
    EulerSolver<3, double> s3(dh3, fev3, ffev3, 3);
    s3.set_state_from_host(rho3, rhou3, rhov3, rhow3, rhoE3);
    s3.solve_steady_state(10000, 0.5, 1000, true, abs_tol);
    s3.write_solution("solution_steady_state_p3.vtu");
  }
  Kokkos::finalize();
  return 0;
}