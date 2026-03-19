#include <Kokkos_Core.hpp>
#include <config.hpp>
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

    // 1. Allocate Current State Vectors
    rho_ = VecDevice(n_dofs_);
    rho_u_ = VecDevice(n_dofs_);
    rho_v_ = VecDevice(n_dofs_);
    rho_E_ = VecDevice(n_dofs_);

    // 2. Allocate "Old" State Vectors for SSP-RK3 Stages
    // These store u^n to perform the convex combinations required by the
    // scheme.
    rho_old_ = VecDevice(n_dofs_);
    rho_u_old_ = VecDevice(n_dofs_);
    rho_v_old_ = VecDevice(n_dofs_);
    rho_E_old_ = VecDevice(n_dofs_);

    // 3. Allocate Residual Vectors
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
    // TODO: Fix this to use import with the vector function rather than mirrors
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

  void solve(unsigned int max_iter = 10000,
             RealType cfl = Parameters<RealType>::cfl_max)
  {
    for (unsigned int iter = 0; iter < max_iter; ++iter) {
      // 1. Store u^n (the values at the start of the entire RK3 cycle)
      Kokkos::deep_copy(rho_old_.view(), rho_.view());
      Kokkos::deep_copy(rho_u_old_.view(), rho_u_.view());
      Kokkos::deep_copy(rho_v_old_.view(), rho_v_.view());
      Kokkos::deep_copy(rho_E_old_.view(), rho_E_.view());

      // Compute Local Time Step once per iteration
      compute_local_dt(cfl);

      // --- Stage 1 ---
      zero_residuals();
      compute_volume_residual();
      compute_face_residual();
      update(RealType(0.0), RealType(1.0));

      // --- Stage 2 ---
      zero_residuals();
      compute_volume_residual();
      compute_face_residual();
      update(RealType(0.75), RealType(0.25));

      // --- Stage 3 ---
      zero_residuals();
      compute_volume_residual();
      compute_face_residual();
      update(RealType(1.0 / 3.0), RealType(2.0 / 3.0));

      // Convergence check [cite: 18, 19]
      if (iter % 100 == 0) {
        std::cout << "Iteration " << iter << " Residual: " << residual_norm()
                  << std::endl;
      }
    }
  }

  const unsigned int degree_;
  const DoFHandler<dim, RealType>& dof_handler_;
  FEValues<dim, RealType>& fe_values_;
  FEFaceValues<dim, RealType>& fe_face_values_;

  uint32_t n_dofs_;

  Kokkos::View<RealType**, Layout, DeviceMemSpace> JxW_;
  Kokkos::View<RealType***, Layout, DeviceMemSpace> phi_;
  Kokkos::View<RealType****, Layout, DeviceMemSpace> grad_phi_;

  // Current State (u_current)
  VecDevice rho_, rho_u_, rho_v_, rho_E_;

  // Storage for start-of-step state (u^n)
  VecDevice rho_old_, rho_u_old_, rho_v_old_, rho_E_old_;

  // Residuals and Timesteps
  VecDevice res_rho_, res_rho_u_, res_rho_v_, res_rho_E_;
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
    // TODO: Add face stuff here

    const auto n_cells = dof_handler_.n_cells();
    const auto n_dofs_per_cell = dof_handler_.n_dofs_per_cell();
    const auto n_q_points = fe_values_.n_q_points();

    // Initialize the data
    JxW_ = Kokkos::View<RealType**, Layout, DeviceMemSpace>(
      "phi", n_cells, n_q_points);
    phi_ = Kokkos::View<RealType***, Layout, DeviceMemSpace>(
      "phi", n_cells, n_dofs_per_cell, n_q_points);
    grad_phi_ = Kokkos::View<RealType****, Layout, DeviceMemSpace>(
      "grad_phi", n_cells, n_dofs_per_cell, n_q_points, dim);

    // Fill these on the host and copy over to device
    auto JxW_h = Kokkos::create_mirror_view(JxW_);
    auto phi_h = Kokkos::create_mirror_view(phi_);
    auto grad_phi_h = Kokkos::create_mirror_view(grad_phi_);

    for (auto cell : dof_handler_.active_cell_range()) {
      const auto k = cell.index();
      fe_values_.reinit(cell);

      for (unsigned int q = 0; q < n_q_points; ++q) {
        // JxW_h(k, q) = fe_values_.jxw(q);

        for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
          // phi_h(k, i, q) = fe_values_.shape_value(i, q);

          for (unsigned int d = 0; d < dim; ++d)
            continue;
          // grad_phi_h(k, i, q, d) = fe_values_.shape_gradient(i, q, d);
        }
      }
    }

    Kokkos::deep_copy(JxW_, JxW_h);
    Kokkos::deep_copy(phi_, phi_h);
    Kokkos::deep_copy(grad_phi_, grad_phi_h);
  }

  void compute_local_dt(RealType cfl)
  {
    auto d_rho = rho_.view();
    auto d_rho_u = rho_u_.view();
    auto d_rho_v = rho_v_.view();
    auto d_rho_E = rho_E_.view();
    auto d_dt = dt_.view();

    auto indices = dof_handler_.cell_dof_indices();
    auto JxW = JxW_; // Use precomputed Jacobian weights to find area

    const auto n_cells = dof_handler_.n_cells();
    const auto n_dofs_per_cell = dof_handler_.n_dofs_per_cell();
    const auto n_q_points = fe_values_.n_q_points();
    const RealType gamma = Parameters<RealType>::gamma;

    // Problem degree is used to scale the DG time step for stability
    // Typically dt_dg ~ dt_fv / (2p + 1)
    const RealType p_order = static_cast<RealType>(degree_);
    const RealType dg_scaling =
      RealType(1.0) / (RealType(2.0) * p_order + RealType(1.0));

    Kokkos::parallel_for(
      "compute_local_dt", n_cells, KOKKOS_LAMBDA(int k) {
        // 1. Compute Cell-Averaged State
        // We use the arithmetic mean of the DoFs as a simple approximation for
        // the cell state
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

        // 2. Compute Velocity and Speed of Sound
        const RealType u = rhou_avg / rho_avg;
        const RealType v = rhov_avg / rho_avg;
        const RealType vel_mag = Kokkos::sqrt(u * u + v * v);

        const RealType p =
          (gamma - 1.0) * (rhoE_avg - 0.5 * rho_avg * (u * u + v * v));
        const RealType a = Kokkos::sqrt(gamma * p / rho_avg);

        // 3. Determine Characteristic Length h
        // A robust estimate for triangles is h = sqrt(Area).
        // Area is the integral of JxW over the element.
        RealType area = 0;
        for (unsigned int q = 0; q < n_q_points; ++q) {
          area += JxW(k, q);
        }
        const RealType h = Kokkos::sqrt(area);

        // 4. Calculate Local dt
        // The wave speed is |u| + a.
        const RealType dt_cell = cfl * dg_scaling * (h / (vel_mag + a));

        // 5. Assign dt to all DoFs in this cell
        for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
          d_dt(indices(k, i)) = dt_cell;
        }
      });
    Kokkos::fence();
  }

  void compute_volume_residual()
  {
    const auto n_cells = dof_handler_.n_cells();
    const auto n_dofs_per_cell = dof_handler_.n_dofs_per_cell();
    const auto n_q_points = fe_values_.n_q_points();

    auto phi = phi_;
    auto grad_phi = grad_phi_;
    auto JxW = JxW_;
    auto indices = dof_handler_.cell_dof_indices();

    auto d_rho = rho_.view();
    auto d_rho_u = rho_u_.view();
    auto d_rho_v = rho_v_.view();
    auto d_rho_E = rho_E_.view();
    auto d_res_rho = res_rho_.view();
    auto d_res_rho_u = res_rho_u_.view();
    auto d_res_rho_v = res_rho_v_.view();
    auto d_res_rho_E = res_rho_E_.view();

    Kokkos::parallel_for(
      "volume_residual_test",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
        { 0, 0 }, { (int)n_cells, (int)n_dofs_per_cell }),
      KOKKOS_LAMBDA(int k, int i) {
        RealType local_residual_rho = 0;
        RealType local_residual_rho_u = 0;
        RealType local_residual_rho_v = 0;
        RealType local_residual_rho_E = 0;

        // Loop over quad points
        for (unsigned int q = 0; q < n_q_points; ++q) {
          // Resconstruct the conservative state
          RealType local_rho = 0;
          RealType local_rho_u = 0;
          RealType local_rho_v = 0;
          RealType local_rho_E = 0;
          for (unsigned int j = 0; j < n_dofs_per_cell; ++j) {
            const RealType phi_j = phi(k, j, q);
            const uint32_t dof_j = indices(k, j);
            local_rho += d_rho(dof_j) * phi_j;
            local_rho_u += d_rho_u(dof_j) * phi_j;
            local_rho_v += d_rho_v(dof_j) * phi_j;
            local_rho_E += d_rho_E(dof_j) * phi_j;
          }

          // Compute the primitive variables
          RealType local_u = 0, local_v = 0, local_p = 0;
          Flux<RealType>::conservative_to_primitive(local_rho,
                                                    local_rho_u,
                                                    local_rho_v,
                                                    local_rho_E,
                                                    local_u,
                                                    local_v,
                                                    local_p);

          // x-flux
          RealType Fx_rho, Fx_rho_u, Fx_rho_v, Fx_rho_E;
          Flux<RealType>::euler_flux(local_rho,
                                     local_u,
                                     local_v,
                                     local_p,
                                     local_rho_E,
                                     RealType(1),
                                     RealType(0),
                                     Fx_rho,
                                     Fx_rho_u,
                                     Fx_rho_v,
                                     Fx_rho_E);

          // y-flux
          RealType Fy_rho, Fy_rho_u, Fy_rho_v, Fy_rho_E;
          Flux<RealType>::euler_flux(local_rho,
                                     local_u,
                                     local_v,
                                     local_p,
                                     local_rho_E,
                                     RealType(0),
                                     RealType(1),
                                     Fy_rho,
                                     Fy_rho_u,
                                     Fy_rho_v,
                                     Fy_rho_E);

          // Residual weak form
          const RealType jxw = JxW(k, q);
          const RealType dphi_dx = grad_phi(k, i, q, 0);
          const RealType dphi_dy = grad_phi(k, i, q, 1);

          local_residual_rho -= (dphi_dx * Fx_rho + dphi_dy * Fy_rho) * jxw;
          local_residual_rho_u -=
            (dphi_dx * Fx_rho_u + dphi_dy * Fy_rho_u) * jxw;
          local_residual_rho_v -=
            (dphi_dx * Fx_rho_v + dphi_dy * Fy_rho_v) * jxw;
          local_residual_rho_E -=
            (dphi_dx * Fx_rho_E + dphi_dy * Fy_rho_E) * jxw;
        }

        // Scatter local residual contributions
        const uint32_t dof_i = indices(k, i);
        Kokkos::atomic_add(&d_res_rho(dof_i), local_residual_rho);
        Kokkos::atomic_add(&d_res_rho_u(dof_i), local_residual_rho_u);
        Kokkos::atomic_add(&d_res_rho_v(dof_i), local_residual_rho_v);
        Kokkos::atomic_add(&d_res_rho_E(dof_i), local_residual_rho_E);
      });
    Kokkos::fence();
  }

  void compute_face_residual() {}

  void update(RealType alpha, RealType beta)
  {
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

    Kokkos::parallel_for(
      "update_rk_stage", n_dofs_, KOKKOS_LAMBDA(int i) {
        const RealType dt = d_dt(i);
        // SSP-RK3 Convex Combination: u_new = alpha*u_old + beta*(u_curr +
        // dt*R)
        d_rho(i) = alpha * d_rho_old(i) + beta * (d_rho(i) + dt * d_res_rho(i));
        d_rho_u(i) =
          alpha * d_rho_u_old(i) + beta * (d_rho_u(i) + dt * d_res_rho_u(i));
        d_rho_v(i) =
          alpha * d_rho_v_old(i) + beta * (d_rho_v(i) + dt * d_res_rho_v(i));
        d_rho_E(i) =
          alpha * d_rho_E_old(i) + beta * (d_rho_E(i) + dt * d_res_rho_E(i));
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

static constexpr unsigned int problem_degree = 3;

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
    const unsigned int n_dofs = dof_handler.n_dofs();

    std::cout << "Number of cells: " << tria.n_cells() << "\n";
    std::cout << "Number of DoFs: " << n_dofs << "\n";

    // Create the FEValues objects
    FEValues<2, double> fe_values(fe, quad);
    FEFaceValues<2, double> fe_face_values(fe, face_quad);
  }

  Kokkos::finalize();
  return 0;
}
