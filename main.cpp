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

static constexpr unsigned int problem_degree = 1;

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

    gri.read_gri("../grids/coarse_local_refinement_1.gri");
    gri.transfer_to_triangulation(tria);

    DoFHandler<2, double> dof_handler(tria, fe);
    const unsigned int n_dofs = dof_handler.n_dofs();

    std::cout << "Number of cells: " << tria.n_cells() << "\n";
    std::cout << "Number of DoFs: " << n_dofs << "\n";

    // Freestream state
    double gamma = Parameters<double>::gamma;
    double rho0 = Parameters<double>::rho_0;
    double a0 = Parameters<double>::a_0;
    double p0 = Parameters<double>::p_0;

    double mach = 0.5;
    double u0 = mach * a0 * Parameters<double>::n_x_0();
    double v0 = mach * a0 * Parameters<double>::n_y_0();

    double gm1 = gamma - 1.0;
    double T_rat = 1.0 / (1.0 + 0.5 * gm1 * mach * mach);
    double p_fs = p0 * Kokkos::pow(T_rat, gamma / gm1);
    double rho_fs = rho0 * Kokkos::pow(T_rat, 1.0 / gm1);
    double rhoE0 = p_fs / gm1 + 0.5 * rho_fs * (u0 * u0 + v0 * v0);

    // ------------------------------------------------------------------
    // Initialize solution to uniform freestream
    // ------------------------------------------------------------------
    Vector<double, DeviceMemSpace> rho(n_dofs);
    Vector<double, DeviceMemSpace> rho_u(n_dofs);
    Vector<double, DeviceMemSpace> rho_v(n_dofs);
    Vector<double, DeviceMemSpace> rho_E(n_dofs);

    {
      auto rho_h = Kokkos::create_mirror_view(rho.view());
      auto rho_u_h = Kokkos::create_mirror_view(rho_u.view());
      auto rho_v_h = Kokkos::create_mirror_view(rho_v.view());
      auto rho_E_h = Kokkos::create_mirror_view(rho_E.view());

      std::vector<uint32_t> dof_ids;
      for (auto cell : dof_handler.active_cell_range()) {
        cell.get_dof_indices(dof_ids);
        // DOF 0 is cell average for Legendre basis — set freestream value
        // Higher DOFs remain zero (uniform field has no variation)
        rho_h(dof_ids[0]) = rho0;
        rho_u_h(dof_ids[0]) = rho0 * u0;
        rho_v_h(dof_ids[0]) = rho0 * v0;
        rho_E_h(dof_ids[0]) = rhoE0;
      }

      Kokkos::deep_copy(rho.view(), rho_h);
      Kokkos::deep_copy(rho_u.view(), rho_u_h);
      Kokkos::deep_copy(rho_v.view(), rho_v_h);
      Kokkos::deep_copy(rho_E.view(), rho_E_h);
    }

    // ------------------------------------------------------------------
    // Compute residual with periodic / no boundary treatment —
    // only interior faces contribute
    // ------------------------------------------------------------------
    Vector<double, DeviceMemSpace> res_rho(n_dofs);
    Vector<double, DeviceMemSpace> res_rho_u(n_dofs);
    Vector<double, DeviceMemSpace> res_rho_v(n_dofs);
    Vector<double, DeviceMemSpace> res_rho_E(n_dofs);

    Kokkos::deep_copy(res_rho.view(), 0.0);
    Kokkos::deep_copy(res_rho_u.view(), 0.0);
    Kokkos::deep_copy(res_rho_v.view(), 0.0);
    Kokkos::deep_copy(res_rho_E.view(), 0.0);

    FEValues<2, double> fe_values(fe, quad);
    FEFaceValues<2, double> fe_face_values(fe, face_quad);

    std::vector<uint32_t> dof_ids, nbr_dof_ids;

    // ---- Volume terms ------------------------------------------------
    for (auto cell : dof_handler.active_cell_range()) {
      cell.get_dof_indices(dof_ids);
      fe_values.reinit(cell);
      auto fev = fe_values.device_proxy();

      Kokkos::View<uint32_t*, DeviceMemSpace> ids("ids", fe.n_dofs());
      {
        auto ids_h = Kokkos::create_mirror_view(ids);
        for (unsigned int i = 0; i < fe.n_dofs(); ++i)
          ids_h(i) = dof_ids[i];
        Kokkos::deep_copy(ids, ids_h);
      }

      auto d_rho = rho.view();
      auto d_rho_u = rho_u.view();
      auto d_rho_v = rho_v.view();
      auto d_rho_E = rho_E.view();
      auto d_res_rho = res_rho.view();
      auto d_res_rho_u = res_rho_u.view();
      auto d_res_rho_v = res_rho_v.view();
      auto d_res_rho_E = res_rho_E.view();

      Kokkos::parallel_for(
        "volume_freestream",
        Kokkos::RangePolicy<>(0, fe.n_dofs()),
        KOKKOS_LAMBDA(int i) {
          double R_rho = 0, R_rho_u = 0, R_rho_v = 0, R_rho_E = 0;

          for (unsigned int q = 0; q < fev.n_q; ++q) {
            double r = 0, ru = 0, rv = 0, rE = 0;
            for (unsigned int j = 0; j < fev.n_dofs; ++j) {
              const double phi_j = fev.shape_value(j, q);
              r += d_rho(ids(j)) * phi_j;
              ru += d_rho_u(ids(j)) * phi_j;
              rv += d_rho_v(ids(j)) * phi_j;
              rE += d_rho_E(ids(j)) * phi_j;
            }

            double u, v, p;
            Flux<double>::conservative_to_primitive(r, ru, rv, rE, u, v, p);

            double Fx_rho, Fx_rho_u, Fx_rho_v, Fx_rho_E;
            Flux<double>::euler_flux(
              r, u, v, p, rE, 1.0, 0.0, Fx_rho, Fx_rho_u, Fx_rho_v, Fx_rho_E);
            double Fy_rho, Fy_rho_u, Fy_rho_v, Fy_rho_E;
            Flux<double>::euler_flux(
              r, u, v, p, rE, 0.0, 1.0, Fy_rho, Fy_rho_u, Fy_rho_v, Fy_rho_E);

            const double jxw = fev.jxw(q);
            const double dphi_dx = fev.shape_gradient(i, q, 0);
            const double dphi_dy = fev.shape_gradient(i, q, 1);

            R_rho -= (dphi_dx * Fx_rho + dphi_dy * Fy_rho) * jxw;
            R_rho_u -= (dphi_dx * Fx_rho_u + dphi_dy * Fy_rho_u) * jxw;
            R_rho_v -= (dphi_dx * Fx_rho_v + dphi_dy * Fy_rho_v) * jxw;
            R_rho_E -= (dphi_dx * Fx_rho_E + dphi_dy * Fy_rho_E) * jxw;
          }

          Kokkos::atomic_add(&d_res_rho(ids(i)), R_rho);
          Kokkos::atomic_add(&d_res_rho_u(ids(i)), R_rho_u);
          Kokkos::atomic_add(&d_res_rho_v(ids(i)), R_rho_v);
          Kokkos::atomic_add(&d_res_rho_E(ids(i)), R_rho_E);
        });
      Kokkos::fence();
    }

    // ---- Interior face terms only — skip boundary faces -------------
    for (auto cell : dof_handler.active_cell_range()) {
      cell.get_dof_indices(dof_ids);

      for (unsigned int face_no = 0; face_no < 3; ++face_no) {

        // Skip boundary faces entirely for freestream test
        if (cell.face_at_boundary(face_no))
          continue;

        // Only process each interior face once
        CellIndexType nbr_idx = cell.neighbor_index(face_no);
        if (nbr_idx < cell.index())
          continue;

        dof_handler.cell(nbr_idx).get_dof_indices(nbr_dof_ids);
        fe_face_values.reinit(cell.tria_cell, face_no);
        auto ffv = fe_face_values.device_proxy();

        Kokkos::View<uint32_t*, DeviceMemSpace> ids("ids", fe.n_dofs());
        Kokkos::View<uint32_t*, DeviceMemSpace> nbr("nbr", fe.n_dofs());
        {
          auto ids_h = Kokkos::create_mirror_view(ids);
          auto nbr_h = Kokkos::create_mirror_view(nbr);
          for (unsigned int i = 0; i < fe.n_dofs(); ++i) {
            ids_h(i) = dof_ids[i];
            nbr_h(i) = nbr_dof_ids[i];
          }
          Kokkos::deep_copy(ids, ids_h);
          Kokkos::deep_copy(nbr, nbr_h);
        }

        auto d_rho = rho.view();
        auto d_rho_u = rho_u.view();
        auto d_rho_v = rho_v.view();
        auto d_rho_E = rho_E.view();
        auto d_res_rho = res_rho.view();
        auto d_res_rho_u = res_rho_u.view();
        auto d_res_rho_v = res_rho_v.view();
        auto d_res_rho_E = res_rho_E.view();

        Kokkos::parallel_for(
          "face_freestream",
          Kokkos::RangePolicy<>(0, fe.n_dofs()),
          KOKKOS_LAMBDA(int i) {
            double R_rho = 0, R_rho_u = 0, R_rho_v = 0, R_rho_E = 0;
            double Rn_rho = 0, Rn_rho_u = 0, Rn_rho_v = 0, Rn_rho_E = 0;

            for (unsigned int q = 0; q < ffv.n_q; ++q) {
              double rL = 0, ruL = 0, rvL = 0, rEL = 0;
              double rR = 0, ruR = 0, rvR = 0, rER = 0;
              for (unsigned int j = 0; j < ffv.n_dofs; ++j) {
                const double phi_j = ffv.shape_value(j, q);
                rL += d_rho(ids(j)) * phi_j;
                ruL += d_rho_u(ids(j)) * phi_j;
                rvL += d_rho_v(ids(j)) * phi_j;
                rEL += d_rho_E(ids(j)) * phi_j;
                rR += d_rho(nbr(j)) * phi_j;
                ruR += d_rho_u(nbr(j)) * phi_j;
                rvR += d_rho_v(nbr(j)) * phi_j;
                rER += d_rho_E(nbr(j)) * phi_j;
              }

              const double nx = ffv.normal(q, 0);
              const double ny = ffv.normal(q, 1);

              double F_rho, F_rho_u, F_rho_v, F_rho_E, s_mag;
              Flux<double>::roe_flux(rL,
                                     ruL,
                                     rvL,
                                     rEL,
                                     rR,
                                     ruR,
                                     rvR,
                                     rER,
                                     nx,
                                     ny,
                                     F_rho,
                                     F_rho_u,
                                     F_rho_v,
                                     F_rho_E,
                                     s_mag);

              const double phi_i = ffv.shape_value(i, q);
              const double jxw = ffv.jxw(q);

              R_rho += phi_i * F_rho * jxw;
              R_rho_u += phi_i * F_rho_u * jxw;
              R_rho_v += phi_i * F_rho_v * jxw;
              R_rho_E += phi_i * F_rho_E * jxw;

              Rn_rho -= phi_i * F_rho * jxw;
              Rn_rho_u -= phi_i * F_rho_u * jxw;
              Rn_rho_v -= phi_i * F_rho_v * jxw;
              Rn_rho_E -= phi_i * F_rho_E * jxw;
            }

            Kokkos::atomic_add(&d_res_rho(ids(i)), R_rho);
            Kokkos::atomic_add(&d_res_rho_u(ids(i)), R_rho_u);
            Kokkos::atomic_add(&d_res_rho_v(ids(i)), R_rho_v);
            Kokkos::atomic_add(&d_res_rho_E(ids(i)), R_rho_E);
            Kokkos::atomic_add(&d_res_rho(nbr(i)), Rn_rho);
            Kokkos::atomic_add(&d_res_rho_u(nbr(i)), Rn_rho_u);
            Kokkos::atomic_add(&d_res_rho_v(nbr(i)), Rn_rho_v);
            Kokkos::atomic_add(&d_res_rho_E(nbr(i)), Rn_rho_E);
          });
        Kokkos::fence();
      }
    }

    // ------------------------------------------------------------------
    // Check: residual should be zero to machine precision
    // ------------------------------------------------------------------
    {
      auto res_rho_h = Kokkos::create_mirror_view(res_rho.view());
      auto res_rho_u_h = Kokkos::create_mirror_view(res_rho_u.view());
      auto res_rho_v_h = Kokkos::create_mirror_view(res_rho_v.view());
      auto res_rho_E_h = Kokkos::create_mirror_view(res_rho_E.view());
      Kokkos::deep_copy(res_rho_h, res_rho.view());
      Kokkos::deep_copy(res_rho_u_h, res_rho_u.view());
      Kokkos::deep_copy(res_rho_v_h, res_rho_v.view());
      Kokkos::deep_copy(res_rho_E_h, res_rho_E.view());
    }
  }

  Kokkos::finalize();

  return 0;
}
