#include <dof_handler.hpp>
#include <read_gri.hpp>
#include <triangulation.hpp>
#include <fe.hpp>
#include <gtest/gtest.h>

using RealType = double;
static constexpr RealType tol = 1.0e-12;

TEST(Curved_Grid, read_grid_correctness) {
    unsigned int problem_degree = 2;

    GriReader<2> gri;
    Triangulation<2> tria;
    FE_DGQLegendre<2, double> fe(problem_degree);
    QGaussSimplex<2, double> quad(problem_degree + 1);
    QGaussSimplex<1, double> face_quad(problem_degree + 1);

    gri.read_gri("../grids/Uniform2K_curved.gri");
    gri.transfer_to_triangulation(tria);

    DoFHandler<2, double> dof_handler(tria, fe);

    FEValues<2, double> fe_values(fe, quad);
    FEFaceValues<2, double> fe_face_values(fe, face_quad);

    const MeshData& mesh = gri.get_data();

    EXPECT_EQ(mesh.n_nodes, 1778);
    EXPECT_EQ(mesh.n_elements, 1999);
    EXPECT_EQ(mesh.n_boundary_groups, 6);
    EXPECT_EQ(mesh.n_periodic_groups, 2);

}