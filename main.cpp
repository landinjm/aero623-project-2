#include <Kokkos_Core.hpp>
#include <config.hpp>
#include <data_out.hpp>
#include <dof_handler.hpp>
#include <fe.hpp>
#include <flux.hpp>
#include <parameters.hpp>
#include <read_gri.hpp>
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
              FEFaceValues<dim, RealType>& fe_face_values)
    : dof_handler_(dof_handler)
    , fe_values_(fe_values)
    , fe_face_values_(fe_face_values)
    , n_dofs_(dof_handler.n_dofs())
  {

    // Allocate the vectors for the conservative state and its residual along
    // with the timestep
    rho_ = VecDevice(n_dofs_);
    rho_u_ = VecDevice(n_dofs_);
    rho_v_ = VecDevice(n_dofs_);
    rho_E_ = VecDevice(n_dofs_);

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
             RealType tol = RealType(1e-10),
             RealType cfl = Parameters<RealType>::cfl_max)
  {
    for (unsigned int iter = 0; iter < max_iter; ++iter) {
      zero_residuals();
      compute_volume_residual();
      compute_face_residual();
      compute_local_dt(cfl);
      update();

      const RealType res_norm = residual_norm();
      if (iter % 100 == 0)
        std::cout << "iter " << iter << "  ||R|| = " << res_norm << "\n";
      /*
            if (res_norm < tol) {
              std::cout << "Converged at iter " << iter << "  ||R|| = " <<
         res_norm
                        << "\n";
              return;
            }*/
    }
    std::cout << "Max iterations reached\n";
  }

private:
  const DoFHandler<dim, RealType>& dof_handler_;
  FEValues<dim, RealType>& fe_values_;
  FEFaceValues<dim, RealType>& fe_face_values_;

  uint32_t n_dofs_;

  Kokkos::View<RealType**, Layout, DeviceMemSpace> JxW_;
  Kokkos::View<RealType***, Layout, DeviceMemSpace> phi_;
  Kokkos::View<RealType****, Layout, DeviceMemSpace> grad_phi_;

  VecDevice rho_, rho_u_, rho_v_, rho_E_;
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
        JxW_h(k, q) = fe_values_.jxw(q);

        for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
          phi_h(k, i, q) = fe_values_.shape_value(i, q);

          for (unsigned int d = 0; d < dim; ++d)
            grad_phi_h(k, i, q, d) = fe_values_.shape_gradient(i, q, d);
        }
      }
    }

    Kokkos::deep_copy(JxW_, JxW_h);
    Kokkos::deep_copy(phi_, phi_h);
    Kokkos::deep_copy(grad_phi_, grad_phi_h);
  }

  void compute_local_dt(RealType cfl) {}

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

  void update()
  {
    // TODO: Add a different time-stepping here
    auto d_rho = rho_.view();
    auto d_rho_u = rho_u_.view();
    auto d_rho_v = rho_v_.view();
    auto d_rho_E = rho_E_.view();
    auto d_res_rho = res_rho_.view();
    auto d_res_rho_u = res_rho_u_.view();
    auto d_res_rho_v = res_rho_v_.view();
    auto d_res_rho_E = res_rho_E_.view();
    auto d_dt = dt_.view();

    const uint32_t n_dofs = dof_handler_.n_dofs();

    Kokkos::parallel_for(
      "update", Kokkos::RangePolicy<>(0, n_dofs), KOKKOS_LAMBDA(int i) {
        const RealType dt = d_dt(i);
        d_rho(i) += dt * d_res_rho(i);
        d_rho_u(i) += dt * d_res_rho_u(i);
        d_rho_v(i) += dt * d_res_rho_v(i);
        d_rho_E(i) += dt * d_res_rho_E(i);
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

    gri.read_gri("../grids/coarse_local_refinement_2.gri");
    gri.transfer_to_triangulation(tria);

    DoFHandler<2, double> dof_handler(tria, fe);
    const unsigned int n_dofs = dof_handler.n_dofs();

    std::cout << "Number of cells: " << tria.n_cells() << "\n";
    std::cout << "Number of DoFs: " << n_dofs << "\n";

    // Create the FEValues objects
    FEValues<2, double> fe_values(fe, quad);
    FEFaceValues<2, double> fe_face_values(fe, face_quad);

    // Create solver
    EulerSolver<2, double> solver(dof_handler, fe_values, fe_face_values);

    solver.set_initial_condition([](double x, double y) {
      // freestream init
      const double gamma = Parameters<double>::gamma;
      const double rho = Parameters<double>::rho_0;
      const double mach = 0.5;
      const double a = Parameters<double>::a_0;
      const double u = mach * a * Parameters<double>::n_x_0();
      const double v = mach * a * Parameters<double>::n_y_0();
      const double p = Parameters<double>::p_0 *
                       std::pow(1.0 / (1.0 + 0.2 * mach * mach), gamma / 0.4);
      return std::make_tuple(rho, u, v, p);
    });

    solver.solve(10000, 1e-10, Parameters<double>::cfl_max);
  }

  Kokkos::finalize();

  return 0;
}
