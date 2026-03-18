#include <gtest/gtest.h>
#include <read_gri.hpp>
#include <triangulation.hpp>

TEST(ReadGri, basic)
{
  GriReader<2> gri;
  gri.read_gri("../tests/test.gri");
  auto data = gri.data();

  EXPECT_EQ(data.n_nodes, 6);
  EXPECT_EQ(data.n_elements, 4);
  EXPECT_EQ(data.n_boundary_groups, 4);
  EXPECT_EQ(data.n_periodic_groups, 1);
  EXPECT_EQ(data.n_boundary_faces(), 6);
}

TEST(ReadGri, basic_2)
{
  GriReader<2> gri;
  gri.read_gri("../tests/test_2.gri");
  auto data = gri.data();

  EXPECT_EQ(data.n_nodes, 12);
  EXPECT_EQ(data.n_elements, 12);
  EXPECT_EQ(data.n_boundary_groups, 4);
  EXPECT_EQ(data.n_periodic_groups, 1);
  EXPECT_EQ(data.n_boundary_faces(), 10);
}

TEST(ReadGri, basic_3)
{
  GriReader<2> gri;
  gri.read_gri("../tests/test_3.gri");
  auto data = gri.data();

  EXPECT_EQ(data.n_nodes, 6);
  EXPECT_EQ(data.n_elements, 4);
  EXPECT_EQ(data.n_boundary_groups, 4);
  EXPECT_EQ(data.n_periodic_groups, 1);
  EXPECT_EQ(data.n_boundary_faces(), 6);
}

TEST(ReadGri, basic_4)
{
  GriReader<2> gri;
  gri.read_gri("../tests/test_4.gri");
  auto data = gri.data();

  EXPECT_EQ(data.n_nodes, 15);
  EXPECT_EQ(data.n_elements, 16);
  EXPECT_EQ(data.n_boundary_groups, 4);
  EXPECT_EQ(data.n_periodic_groups, 1);
  EXPECT_EQ(data.n_boundary_faces(), 12);
}

TEST(ReadGri, basic_5)
{
  GriReader<2> gri;
  gri.read_gri("../tests/test_5.gri");
  auto data = gri.data();

  EXPECT_EQ(data.n_nodes, 45);
  EXPECT_EQ(data.n_elements, 64);
  EXPECT_EQ(data.n_boundary_groups, 4);
  EXPECT_EQ(data.n_periodic_groups, 1);
  EXPECT_EQ(data.n_boundary_faces(), 24);
}

TEST(TriangulationImport, basic)
{
  Triangulation<2> tria;
  GriReader<2> gri;
  gri.read_gri("../tests/test.gri");
  gri.transfer_to_triangulation(tria);

  EXPECT_EQ(tria.n_vertices(), 6);
  EXPECT_EQ(tria.n_cells(), 4);
  EXPECT_EQ(tria.n_boundary_faces(), 4);
  EXPECT_EQ(tria.n_periodic_faces(), 2);
  EXPECT_TRUE(tria.verify_mesh());
}

TEST(TriangulationImport, basic_2)
{
  Triangulation<2> tria;
  GriReader<2> gri;
  gri.read_gri("../tests/test_2.gri");
  gri.transfer_to_triangulation(tria);

  EXPECT_EQ(tria.n_vertices(), 12);
  EXPECT_EQ(tria.n_cells(), 12);
  EXPECT_EQ(tria.n_boundary_faces(), 6);
  EXPECT_EQ(tria.n_periodic_faces(), 4);
  EXPECT_TRUE(tria.verify_mesh());
}

TEST(TriangulationImport, basic_3)
{
  Triangulation<2> tria;
  GriReader<2> gri;
  gri.read_gri("../tests/test_3.gri");
  gri.transfer_to_triangulation(tria);

  EXPECT_EQ(tria.n_vertices(), 6);
  EXPECT_EQ(tria.n_cells(), 4);
  EXPECT_EQ(tria.n_boundary_faces(), 2);
  EXPECT_EQ(tria.n_periodic_faces(), 4);
  EXPECT_TRUE(tria.verify_mesh());
}

TEST(TriangulationImport, basic_4)
{
  Triangulation<2> tria;
  GriReader<2> gri;
  gri.read_gri("../tests/test_4.gri");
  gri.transfer_to_triangulation(tria);

  EXPECT_EQ(tria.n_vertices(), 15);
  EXPECT_EQ(tria.n_cells(), 16);
  EXPECT_EQ(tria.n_boundary_faces(), 8);
  EXPECT_EQ(tria.n_periodic_faces(), 4);
  EXPECT_TRUE(tria.verify_mesh());
}

TEST(TriangulationImport, basic_5)
{
  Triangulation<2> tria;
  GriReader<2> gri;
  gri.read_gri("../tests/test_5.gri");
  gri.transfer_to_triangulation(tria);

  EXPECT_EQ(tria.n_vertices(), 45);
  EXPECT_EQ(tria.n_cells(), 64);
  EXPECT_EQ(tria.n_boundary_faces(), 16);
  EXPECT_EQ(tria.n_periodic_faces(), 8);
  EXPECT_TRUE(tria.verify_mesh());
}
