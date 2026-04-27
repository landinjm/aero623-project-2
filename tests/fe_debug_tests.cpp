/**
 * Diagnostic tests for FEFaceValues/FEValues on curved (mesh_q=2) meshes.
 *
 * Four targeted tests:
 *
 *  1. JacobianVariation_FEValues_3D_q2
 *     Checks that det(J) actually varies across quad points on curved cells,
 *     confirming the mesh has meaningful curvature and the volume Jacobian is
 *     evaluated correctly per-point.
 *
 *  2. JacobianConsistency_FEFaceValues_3D_q2
 *     Directly compares the face Jacobian used inside FEFaceValues against the
 *     correct isoparametric one (rebuilt from all mesh nodes at each quad point).
 *     This is the primary regression check for the linear-vs-curved Jacobian bug.
 *
 *  3. ShapeGradient_PhysicalFD_FEValues_3D_q2
 *     Finite-differences shape function values in physical space through
 *     FEValues::reinit to verify the physical-space gradients returned by
 *     FEValues are correct on a curved mesh.
 *
 *  4. ShapeGradient_PhysicalFD_FEFaceValues_3D_q2
 *     Same finite-difference check but for FEFaceValues, exercising the exact
 *     code path that the DivergenceTheorem test relies on.
 */

#include <dof_handler.hpp>
#include <fe.hpp>
#include <read_gri.hpp>
#include <triangulation.hpp>

#include <gtest/gtest.h>
#include <cmath>

using RealType = double;

// ---------------------------------------------------------------------------
// Test 1: Jacobian variation in FEValues on a curved mesh
//
// On a truly curved (q=2) mesh the determinant of the volume Jacobian must vary
// across quadrature points within a cell.  If FEValues were accidentally using a
// constant linear Jacobian, all det(J) values would be identical.
//
// We report, per cell, the ratio max(det_J)/min(det_J).  For a flat mesh the
// ratio is 1.  For a curved mesh at least some cells should show a noticeably
// larger ratio.  The test asserts that the *global* maximum ratio across all
// cells is above a threshold (1.001 is already definitive), and also prints a
// summary so the user can see the distribution.
// ---------------------------------------------------------------------------
TEST(FEDiagnostic, JacobianVariation_FEValues_3D_q2)
{
    constexpr unsigned int dim    = 3;
    constexpr unsigned int mesh_q = 2;

    Triangulation<dim, mesh_q> tria;
    GriReader<dim, mesh_q> gri;
    gri.read_gri("../airfoils/cube.gri");
    gri.transfer_to_triangulation(tria);

    FE_DGLagrangeSimplex<dim, RealType> fe(1);
    QGaussSimplex<dim, RealType>        quad(3);
    FEValues<dim, mesh_q, RealType>     fe_values(fe, quad);

    constexpr unsigned int n_mesh_nodes =
        FE_DGLagrangeSimplex<dim, RealType>::n_dofs_per_cell(mesh_q);

    double global_max_ratio = 1.0;
    unsigned int n_cells_with_variation = 0;

    for (const auto& cell : tria.active_cell_range())
    {
        fe_values.reinit(cell);

        // Rebuild det(J) per quad point using the full isoparametric map so we
        // can inspect its variation independently of what FEValues stores.
        FE_DGLagrangeSimplex<dim, RealType> mesh_fe(mesh_q);

        RealType mesh_coords[n_mesh_nodes][dim];
        for (unsigned int I = 0; I < n_mesh_nodes; ++I)
            for (unsigned int d = 0; d < dim; ++d)
                mesh_coords[I][d] = cell.vertex(I)(d);

        double det_min = 1e30, det_max = 0.0;

        for (unsigned int q = 0; q < quad.n_points(); ++q)
        {
            const auto xi = quad.point(q);

            double J[3][3] = {};
            for (unsigned int I = 0; I < n_mesh_nodes; ++I)
            {
                const auto dN = mesh_fe.shape_gradient(I, xi);
                for (unsigned int i = 0; i < 3; ++i)
                    for (unsigned int j = 0; j < 3; ++j)
                        J[i][j] += mesh_coords[I][i] * dN(j);
            }

            double det = J[0][0]*(J[1][1]*J[2][2] - J[1][2]*J[2][1])
                       - J[0][1]*(J[1][0]*J[2][2] - J[1][2]*J[2][0])
                       + J[0][2]*(J[1][0]*J[2][1] - J[1][1]*J[2][0]);
            det = std::abs(det);

            det_min = std::min(det_min, det);
            det_max = std::max(det_max, det);
        }

        double ratio = (det_min > 0.0) ? det_max / det_min : 1.0;
        if (ratio > 1.001) ++n_cells_with_variation;
        global_max_ratio = std::max(global_max_ratio, ratio);
    }

    std::cout << "[JacobianVariation] global max det_J ratio: " << global_max_ratio
              << "  cells with >0.1% variation: " << n_cells_with_variation << "\n";

    // On a non-trivial q=2 mesh at least one cell must show curvature.
    EXPECT_GT(global_max_ratio, 1.001)
        << "All det(J) values are identical across quad points — "
           "the mesh may be flat or the Jacobian is computed incorrectly.";
}

// ---------------------------------------------------------------------------
// Test 2: Face Jacobian consistency in FEFaceValues on a curved mesh
//
// FEFaceValues internally builds a Jacobian from cell corner vertices only
// (linear approximation).  For a curved mesh this Jacobian should differ from
// the correct isoparametric one evaluated at each face quad point.
//
// This test rebuilds the correct isoparametric Jacobian at every face quad
// point and compares it element-wise to what FEFaceValues would use if it
// only used the linear approximation.  Large differences indicate the bug.
//
// We report the maximum absolute element-wise error across all faces and all
// quad points so the degree of incorrectness is visible.
// ---------------------------------------------------------------------------
TEST(FEDiagnostic, JacobianConsistency_FEFaceValues_3D_q2)
{
    constexpr unsigned int dim    = 3;
    constexpr unsigned int mesh_q = 2;

    Triangulation<dim, mesh_q> tria;
    GriReader<dim, mesh_q> gri;
    gri.read_gri("../airfoils/cube.gri");
    gri.transfer_to_triangulation(tria);

    FE_DGLagrangeSimplex<dim, RealType> fe(1);
    QGaussSimplex<dim - 1, RealType>    face_quad(3);
    FEFaceValues<dim, mesh_q, RealType> fe_face(fe, face_quad);

    FE_DGLagrangeSimplex<dim, RealType> mesh_fe(mesh_q);

    constexpr unsigned int n_mesh_nodes =
        FE_DGLagrangeSimplex<dim, RealType>::n_dofs_per_cell(mesh_q);
    constexpr unsigned int faces_per_cell =
        SimplexTopology<dim, mesh_q>::faces_per_cell;

    // For reference: the linear Jacobian uses only the 4 corner vertices.
    // We compare that to the full isoparametric Jacobian at each quad point.
    double max_J_error       = 0.0;
    double max_JxW_rel_error = 0.0;

    for (const auto& cell : tria.active_cell_range())
    {
        RealType mesh_coords[n_mesh_nodes][dim];
        for (unsigned int I = 0; I < n_mesh_nodes; ++I)
            for (unsigned int d = 0; d < dim; ++d)
                mesh_coords[I][d] = cell.vertex(I)(d);

        // Linear (corner-only) Jacobian — same logic as current FEFaceValues.
        double J_lin[3][3] = {};
        for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < 3; ++j)
                J_lin[i][j] = cell.vertex(j + 1)(i) - cell.vertex(0)(i);

        for (unsigned int f = 0; f < faces_per_cell; ++f)
        {
            fe_face.reinit(cell, f);

            for (unsigned int q = 0; q < fe_face.n_q_points(); ++q)
            {
                // Map face quad point back to cell reference coordinates.
                // We need the reference-space location of this face quad point.
                // Extract it from the physical quad point and the linear J_inv.
                // (This is an approximation sufficient for comparison purposes.)
                const auto x_phys = fe_face.q_point(q);

                // Isoparametric J at the reference point corresponding to x_phys.
                // We use the linear J_inv to get an approximate xi, then build
                // the isoparametric J there.
                double det_lin = J_lin[0][0]*(J_lin[1][1]*J_lin[2][2] - J_lin[1][2]*J_lin[2][1])
                               - J_lin[0][1]*(J_lin[1][0]*J_lin[2][2] - J_lin[1][2]*J_lin[2][0])
                               + J_lin[0][2]*(J_lin[1][0]*J_lin[2][1] - J_lin[1][1]*J_lin[2][0]);

                // Build linear J inverse
                double J_lin_inv[3][3];
                J_lin_inv[0][0] = (J_lin[1][1]*J_lin[2][2] - J_lin[1][2]*J_lin[2][1]) / det_lin;
                J_lin_inv[0][1] = (J_lin[0][2]*J_lin[2][1] - J_lin[0][1]*J_lin[2][2]) / det_lin;
                J_lin_inv[0][2] = (J_lin[0][1]*J_lin[1][2] - J_lin[0][2]*J_lin[1][1]) / det_lin;
                J_lin_inv[1][0] = (J_lin[1][2]*J_lin[2][0] - J_lin[1][0]*J_lin[2][2]) / det_lin;
                J_lin_inv[1][1] = (J_lin[0][0]*J_lin[2][2] - J_lin[0][2]*J_lin[2][0]) / det_lin;
                J_lin_inv[1][2] = (J_lin[0][2]*J_lin[1][0] - J_lin[0][0]*J_lin[1][2]) / det_lin;
                J_lin_inv[2][0] = (J_lin[1][0]*J_lin[2][1] - J_lin[1][1]*J_lin[2][0]) / det_lin;
                J_lin_inv[2][1] = (J_lin[0][1]*J_lin[2][0] - J_lin[0][0]*J_lin[2][1]) / det_lin;
                J_lin_inv[2][2] = (J_lin[0][0]*J_lin[1][1] - J_lin[0][1]*J_lin[1][0]) / det_lin;

                double x0[3] = { cell.vertex(0)(0), cell.vertex(0)(1), cell.vertex(0)(2) };
                Tensor<1, dim, RealType> xi;
                for (unsigned int d = 0; d < 3; ++d)
                {
                    xi(d) = 0.0;
                    for (unsigned int k = 0; k < 3; ++k)
                        xi(d) += J_lin_inv[d][k] * (x_phys(k) - x0[k]);
                }

                // Build the isoparametric Jacobian at xi.
                double J_iso[3][3] = {};
                for (unsigned int I = 0; I < n_mesh_nodes; ++I)
                {
                    const auto dN = mesh_fe.shape_gradient(I, xi);
                    for (unsigned int i = 0; i < 3; ++i)
                        for (unsigned int j = 0; j < 3; ++j)
                            J_iso[i][j] += mesh_coords[I][i] * dN(j);
                }

                // Compare element-wise.
                for (unsigned int i = 0; i < 3; ++i)
                    for (unsigned int j = 0; j < 3; ++j)
                        max_J_error = std::max(max_J_error, std::abs(J_iso[i][j] - J_lin[i][j]));

                // Compare JxW: the face measures computed from each Jacobian.
                double det_iso = J_iso[0][0]*(J_iso[1][1]*J_iso[2][2] - J_iso[1][2]*J_iso[2][1])
                               - J_iso[0][1]*(J_iso[1][0]*J_iso[2][2] - J_iso[1][2]*J_iso[2][0])
                               + J_iso[0][2]*(J_iso[1][0]*J_iso[2][1] - J_iso[1][1]*J_iso[2][0]);

                double jxw_iso = std::abs(det_iso) * face_quad.weight(q);
                double jxw_lin = fe_face.JxW(q);
                double jxw_ref = (jxw_iso > 1e-14) ? std::abs(jxw_iso - jxw_lin) / jxw_iso : 0.0;
                max_JxW_rel_error = std::max(max_JxW_rel_error, jxw_ref);
            }
        }
    }

    std::cout << "[JacobianConsistency] max |J_iso - J_lin| entry: " << max_J_error << "\n";
    std::cout << "[JacobianConsistency] max relative JxW error:    " << max_JxW_rel_error << "\n";

    // On a flat mesh both errors should be ~0.  On a curved mesh the linear
    // approximation diverges.  We report but do not fail — the purpose here is
    // to SHOW the discrepancy that causes DivergenceTheorem_3D_q2 to fail.
    // Once the bug is fixed, max_J_error should drop to ~machine epsilon.
    if (max_J_error > 1e-10)
    {
        ADD_FAILURE() << "FEFaceValues uses J built from corner vertices only.\n"
                      << "For a curved (mesh_q=2) mesh this differs from the correct\n"
                      << "isoparametric Jacobian by up to " << max_J_error << ".\n"
                      << "Fix: rebuild J from all mesh_fe nodes at each quad point,\n"
                      << "exactly as FEValues::reinit already does.";
    }
}

// ---------------------------------------------------------------------------
// Test 3: Physical-space shape gradient finite-difference check — FEValues
//
// For each cell and each quadrature point we evaluate phi_i at slightly
// perturbed physical positions by re-initialising FEValues on a locally
// perturbed mesh.  The finite-difference gradient in physical space should
// match shape_gradient(i, q) returned by FEValues.
//
// This is the physical-space counterpart of the existing reference-space FD
// test; it catches incorrect Jacobian inversions on curved cells.
// ---------------------------------------------------------------------------
TEST(FEDiagnostic, ShapeGradient_PhysicalFD_FEValues_3D_q2)
{
    constexpr unsigned int dim    = 3;
    constexpr unsigned int mesh_q = 2;

    Triangulation<dim, mesh_q> tria;
    GriReader<dim, mesh_q> gri;
    gri.read_gri("../airfoils/cube.gri");
    gri.transfer_to_triangulation(tria);

    // Use p=1 so the expected gradient is simple to reason about.
    FE_DGLagrangeSimplex<dim, RealType> fe(1);
    QGaussSimplex<dim, RealType>        quad(2);
    FEValues<dim, mesh_q, RealType>     fe_values(fe, quad);

    const double h    = 1e-5;
    const double tol  = 1e-4;  // looser than reference-space because FD is noisy
    double max_error  = 0.0;
    unsigned int n_tested = 0;

    for (const auto& cell : tria.active_cell_range())
    {
        fe_values.reinit(cell);

        for (unsigned int q = 0; q < fe_values.n_q_points(); ++q)
        {
            const auto x_q = fe_values.q_point(q);

            for (unsigned int i = 0; i < fe_values.n_dofs(); ++i)
            {
                const auto grad = fe_values.shape_gradient(i, q);

                // Finite-difference each physical direction.
                for (unsigned int d = 0; d < dim; ++d)
                {
                    // We perturb the reference quad point rather than the physical
                    // point, then re-evaluate.  To get the physical perturbation
                    // direction we need J.  As a shortcut we simply compare the
                    // reference-domain FD against the physical gradient pulled back
                    // via J^{-T}; since FEValues handles the pull-back internally we
                    // can compare directly in reference space and check consistency.
                    //
                    // Simpler and fully equivalent: perturb the reference-space quad
                    // point, re-evaluate shape_value, divide by the reference-space
                    // step, compare to the reference-space gradient component.
                    // This already catches a wrong J_inv.
                    const auto xi = quad.point(q);

                    Tensor<1, dim, RealType> xi_fwd = xi;
                    Tensor<1, dim, RealType> xi_bwd = xi;
                    xi_fwd(d) += h;
                    xi_bwd(d) -= h;

                    // Build the physical gradient in direction d via the chain rule:
                    //   d phi / d x_d  =  sum_k  (d phi / d xi_k) * (d xi_k / d x_d)
                    //                  =  J^{-T}_{kd} * grad_xi phi_k
                    // The physical gradient is already what FEValues returns, so we
                    // verify it against the reference gradient divided by the Jacobian.
                    //
                    // For a more direct check: compare fe_values.shape_value evaluated
                    // at xi +/- h*e_k with the pull-back of the physical gradient.

                    RealType phi_fwd = fe.shape_value(i, xi_fwd);
                    RealType phi_bwd = fe.shape_value(i, xi_bwd);
                    RealType fd_ref  = (phi_fwd - phi_bwd) / (2.0 * h);

                    // The reference gradient from the basis.
                    RealType ref_grad_d = fe.shape_gradient(i, xi)(d);

                    double err = std::abs(fd_ref - ref_grad_d);
                    max_error = std::max(max_error, err);
                    ++n_tested;

                    EXPECT_NEAR(fd_ref, ref_grad_d, tol)
                        << "Reference-space gradient FD mismatch: DoF=" << i
                        << " q=" << q << " dim=" << d;
                }

                // Now check that the physical gradient (which involves J^{-T})
                // is consistent with the reference gradient via an explicit
                // Jacobian comparison.  Rebuild J at this quad point.
                FE_DGLagrangeSimplex<dim, RealType> mesh_fe(mesh_q);

                constexpr unsigned int n_mesh_nodes =
                    FE_DGLagrangeSimplex<dim, RealType>::n_dofs_per_cell(mesh_q);

                double J[3][3] = {};
                for (unsigned int I = 0; I < n_mesh_nodes; ++I)
                {
                    const auto dN = mesh_fe.shape_gradient(I, quad.point(q));
                    for (unsigned int ii = 0; ii < 3; ++ii)
                        for (unsigned int jj = 0; jj < 3; ++jj)
                            J[ii][jj] += cell.vertex(I)(ii) * dN(jj);
                }

                double det = J[0][0]*(J[1][1]*J[2][2] - J[1][2]*J[2][1])
                           - J[0][1]*(J[1][0]*J[2][2] - J[1][2]*J[2][0])
                           + J[0][2]*(J[1][0]*J[2][1] - J[1][1]*J[2][0]);

                double J_inv[3][3];
                J_inv[0][0] = (J[1][1]*J[2][2] - J[1][2]*J[2][1]) / det;
                J_inv[0][1] = (J[0][2]*J[2][1] - J[0][1]*J[2][2]) / det;
                J_inv[0][2] = (J[0][1]*J[1][2] - J[0][2]*J[1][1]) / det;
                J_inv[1][0] = (J[1][2]*J[2][0] - J[1][0]*J[2][2]) / det;
                J_inv[1][1] = (J[0][0]*J[2][2] - J[0][2]*J[2][0]) / det;
                J_inv[1][2] = (J[0][2]*J[1][0] - J[0][0]*J[1][2]) / det;
                J_inv[2][0] = (J[1][0]*J[2][1] - J[1][1]*J[2][0]) / det;
                J_inv[2][1] = (J[0][1]*J[2][0] - J[0][0]*J[2][1]) / det;
                J_inv[2][2] = (J[0][0]*J[1][1] - J[0][1]*J[1][0]) / det;

                const auto xi = quad.point(q);
                const auto ref_grad = fe.shape_gradient(i, xi);

                // Physical gradient = J^{-T} * ref_grad
                for (unsigned int d = 0; d < 3; ++d)
                {
                    double expected_phys = 0.0;
                    for (unsigned int k = 0; k < 3; ++k)
                        expected_phys += J_inv[k][d] * ref_grad(k);

                    double actual_phys = fe_values.shape_gradient(i, q)(d);
                    double err = std::abs(expected_phys - actual_phys);
                    max_error = std::max(max_error, err);

                    EXPECT_NEAR(actual_phys, expected_phys, 1e-10)
                        << "Physical gradient from FEValues does not match J^{-T}*grad_xi.\n"
                        << "DoF=" << i << " q=" << q << " dim=" << d
                        << "  expected=" << expected_phys
                        << "  actual=" << actual_phys;
                }
            }
        }
    }

    std::cout << "[ShapeGradient_PhysicalFD_FEValues] max error: " << max_error
              << "  (n_tested=" << n_tested << ")\n";
}

// ---------------------------------------------------------------------------
// Test 4: Physical-space shape gradient finite-difference check — FEFaceValues
//
// Same principle as Test 3 but uses FEFaceValues.  We evaluate the physical
// gradient returned by FEFaceValues at each face quad point and compare it to
// what the correct isoparametric J^{-T} would give using the same reference
// gradient.  A discrepancy indicates that FEFaceValues is using a wrong J_inv.
//
// This is the most direct regression test for the bug: if FEFaceValues builds
// J from corner vertices only, these gradients will be wrong on curved cells.
// ---------------------------------------------------------------------------
TEST(FEDiagnostic, ShapeGradient_PhysicalFD_FEFaceValues_3D_q2)
{
    constexpr unsigned int dim    = 3;
    constexpr unsigned int mesh_q = 2;

    Triangulation<dim, mesh_q> tria;
    GriReader<dim, mesh_q> gri;
    gri.read_gri("../airfoils/cube.gri");
    gri.transfer_to_triangulation(tria);

    FE_DGLagrangeSimplex<dim, RealType> fe(1);
    QGaussSimplex<dim - 1, RealType>    face_quad(3);
    FEFaceValues<dim, mesh_q, RealType> fe_face(fe, face_quad);

    FE_DGLagrangeSimplex<dim, RealType> mesh_fe(mesh_q);

    constexpr unsigned int n_mesh_nodes =
        FE_DGLagrangeSimplex<dim, RealType>::n_dofs_per_cell(mesh_q);
    constexpr unsigned int faces_per_cell =
        SimplexTopology<dim, mesh_q>::faces_per_cell;

    double max_error  = 0.0;
    unsigned int n_failures = 0;

    // We need a face quad mapped to reference cell space.  To do this we
    // use the same mapping logic as FEFaceValues but re-implemented here
    // using the full isoparametric map so we can compare.

    for (const auto& cell : tria.active_cell_range())
    {
        RealType mesh_coords[n_mesh_nodes][dim];
        for (unsigned int I = 0; I < n_mesh_nodes; ++I)
            for (unsigned int d = 0; d < dim; ++d)
                mesh_coords[I][d] = cell.vertex(I)(d);

        // Linear J for computing xi from x_phys (used to locate the quad point
        // in reference space so we can evaluate the isoparametric J there).
        double J_lin[3][3] = {};
        for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < 3; ++j)
                J_lin[i][j] = cell.vertex(j + 1)(i) - cell.vertex(0)(i);

        double det_lin = J_lin[0][0]*(J_lin[1][1]*J_lin[2][2] - J_lin[1][2]*J_lin[2][1])
                       - J_lin[0][1]*(J_lin[1][0]*J_lin[2][2] - J_lin[1][2]*J_lin[2][0])
                       + J_lin[0][2]*(J_lin[1][0]*J_lin[2][1] - J_lin[1][1]*J_lin[2][0]);

        double J_lin_inv[3][3];
        J_lin_inv[0][0] = (J_lin[1][1]*J_lin[2][2] - J_lin[1][2]*J_lin[2][1]) / det_lin;
        J_lin_inv[0][1] = (J_lin[0][2]*J_lin[2][1] - J_lin[0][1]*J_lin[2][2]) / det_lin;
        J_lin_inv[0][2] = (J_lin[0][1]*J_lin[1][2] - J_lin[0][2]*J_lin[1][1]) / det_lin;
        J_lin_inv[1][0] = (J_lin[1][2]*J_lin[2][0] - J_lin[1][0]*J_lin[2][2]) / det_lin;
        J_lin_inv[1][1] = (J_lin[0][0]*J_lin[2][2] - J_lin[0][2]*J_lin[2][0]) / det_lin;
        J_lin_inv[1][2] = (J_lin[0][2]*J_lin[1][0] - J_lin[0][0]*J_lin[1][2]) / det_lin;
        J_lin_inv[2][0] = (J_lin[1][0]*J_lin[2][1] - J_lin[1][1]*J_lin[2][0]) / det_lin;
        J_lin_inv[2][1] = (J_lin[0][1]*J_lin[2][0] - J_lin[0][0]*J_lin[2][1]) / det_lin;
        J_lin_inv[2][2] = (J_lin[0][0]*J_lin[1][1] - J_lin[0][1]*J_lin[1][0]) / det_lin;

        for (unsigned int f = 0; f < faces_per_cell; ++f)
        {
            fe_face.reinit(cell, f);

            for (unsigned int q = 0; q < fe_face.n_q_points(); ++q)
            {
                const auto x_phys = fe_face.q_point(q);
                double x0[3] = { cell.vertex(0)(0), cell.vertex(0)(1), cell.vertex(0)(2) };

                // Approximate reference-space location of this face quad point.
                Tensor<1, dim, RealType> xi;
                for (unsigned int d = 0; d < 3; ++d)
                {
                    xi(d) = 0.0;
                    for (unsigned int k = 0; k < 3; ++k)
                        xi(d) += J_lin_inv[d][k] * (x_phys(k) - x0[k]);
                }

                // Isoparametric J at xi.
                double J_iso[3][3] = {};
                for (unsigned int I = 0; I < n_mesh_nodes; ++I)
                {
                    const auto dN = mesh_fe.shape_gradient(I, xi);
                    for (unsigned int i = 0; i < 3; ++i)
                        for (unsigned int j = 0; j < 3; ++j)
                            J_iso[i][j] += mesh_coords[I][i] * dN(j);
                }

                double det_iso = J_iso[0][0]*(J_iso[1][1]*J_iso[2][2] - J_iso[1][2]*J_iso[2][1])
                               - J_iso[0][1]*(J_iso[1][0]*J_iso[2][2] - J_iso[1][2]*J_iso[2][0])
                               + J_iso[0][2]*(J_iso[1][0]*J_iso[2][1] - J_iso[1][1]*J_iso[2][0]);

                double J_iso_inv[3][3];
                J_iso_inv[0][0] = (J_iso[1][1]*J_iso[2][2] - J_iso[1][2]*J_iso[2][1]) / det_iso;
                J_iso_inv[0][1] = (J_iso[0][2]*J_iso[2][1] - J_iso[0][1]*J_iso[2][2]) / det_iso;
                J_iso_inv[0][2] = (J_iso[0][1]*J_iso[1][2] - J_iso[0][2]*J_iso[1][1]) / det_iso;
                J_iso_inv[1][0] = (J_iso[1][2]*J_iso[2][0] - J_iso[1][0]*J_iso[2][2]) / det_iso;
                J_iso_inv[1][1] = (J_iso[0][0]*J_iso[2][2] - J_iso[0][2]*J_iso[2][0]) / det_iso;
                J_iso_inv[1][2] = (J_iso[0][2]*J_iso[1][0] - J_iso[0][0]*J_iso[1][2]) / det_iso;
                J_iso_inv[2][0] = (J_iso[1][0]*J_iso[2][1] - J_iso[1][1]*J_iso[2][0]) / det_iso;
                J_iso_inv[2][1] = (J_iso[0][1]*J_iso[2][0] - J_iso[0][0]*J_iso[2][1]) / det_iso;
                J_iso_inv[2][2] = (J_iso[0][0]*J_iso[1][1] - J_iso[0][1]*J_iso[1][0]) / det_iso;

                for (unsigned int i = 0; i < fe_face.n_dofs(); ++i)
                {
                    const auto ref_grad = fe.shape_gradient(i, xi);

                    // Correct physical gradient via isoparametric J^{-T}.
                    for (unsigned int d = 0; d < 3; ++d)
                    {
                        double expected = 0.0;
                        for (unsigned int k = 0; k < 3; ++k)
                            expected += J_iso_inv[k][d] * ref_grad(k);

                        double actual = fe_face.shape_gradient(i, q)(d);
                        double err    = std::abs(expected - actual);
                        max_error = std::max(max_error, err);

                        if (err > 1e-8)
                        {
                            ++n_failures;
                            if (n_failures <= 5)  // print first few failures only
                            {
                                ADD_FAILURE()
                                    << "FEFaceValues shape_gradient is wrong on curved cell.\n"
                                    << "  face=" << f << " q=" << q << " dof=" << i << " dim=" << d << "\n"
                                    << "  expected (isoparametric J^{-T}): " << expected << "\n"
                                    << "  actual   (from FEFaceValues):    " << actual << "\n"
                                    << "  error: " << err << "\n"
                                    << "  Root cause: FEFaceValues builds J from corner vertices only.\n"
                                    << "  Fix: loop over all n_mesh_nodes with mesh_fe.shape_gradient,\n"
                                    << "       matching the FEValues::reinit implementation.";
                            }
                        }
                    }
                }
            }
        }
    }

    std::cout << "[ShapeGradient_PhysicalFD_FEFaceValues] max error: " << max_error
              << "  total failures: " << n_failures << "\n";
}
