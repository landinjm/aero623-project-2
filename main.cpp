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

    // Initialize conservative state
    Vector<double, DeviceMemSpace> rho(n_dofs);
    Vector<double, DeviceMemSpace> rho_u(n_dofs);
    Vector<double, DeviceMemSpace> rho_v(n_dofs);
    Vector<double, DeviceMemSpace> rho_E(n_dofs);

    // Initialize the conservative state on the host
    Vector<double, HostMemSpace> rho_h(n_dofs);
    Vector<double, HostMemSpace> rho_u_h(n_dofs);
    Vector<double, HostMemSpace> rho_v_h(n_dofs);
    Vector<double, HostMemSpace> rho_E_h(n_dofs);

    // Freestream state initial condition
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

    rho_h = rho0;
    rho_u_h = rho0 * u0;
    rho_v_h = rho0 * v0;
    rho_E_h = rhoE0;

    // Output the initial condition
    DataOut<2> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(rho_h, "rho");
    data_out.add_data_vector(rho_u_h, "rho_u");
    data_out.add_data_vector(rho_v_h, "rho_v");
    data_out.add_data_vector(rho_E_h, "rho_E");
    data_out.write_vtu("test.vtu");

    // Import initial condition to device
    rho.import(rho_h, VectorOperation::insert);
    rho_u.import(rho_u_h, VectorOperation::insert);
    rho_v.import(rho_v_h, VectorOperation::insert);
    rho_E.import(rho_E_h, VectorOperation::insert);

    // Initialize the residual vectors
    Vector<double, DeviceMemSpace> res_rho(n_dofs);
    Vector<double, DeviceMemSpace> res_rho_u(n_dofs);
    Vector<double, DeviceMemSpace> res_rho_v(n_dofs);
    Vector<double, DeviceMemSpace> res_rho_E(n_dofs);

    // Create the FEValues objects
    FEValues<2, double> fe_values(fe, quad);
    FEFaceValues<2, double> fe_face_values(fe, face_quad);
  }

  Kokkos::finalize();

  return 0;
}
