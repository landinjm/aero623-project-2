#include <read_gri.hpp>
#include <triangulation.hpp>

#include <gtest/gtest.h>

TEST(ReadGri3D, BasicParsing)
{
  GriReader<3> reader;

  // Make sure path is correct relative to build directory
  reader.read_gri("../tests/test3D.gri");

  MeshData data = reader.data();

  EXPECT_EQ(data.n_nodes, 12);
  EXPECT_EQ(data.n_elements, 10);
  EXPECT_EQ(data.n_boundary_groups, 6);
  EXPECT_EQ(data.n_periodic_groups, 1);

  // Check first node
  EXPECT_DOUBLE_EQ(data.x[0], 0.0);
  EXPECT_DOUBLE_EQ(data.y[0], 0.0);
  EXPECT_DOUBLE_EQ(data.z[0], 0.0);

  // Check first element (remember 0-based indexing)
  EXPECT_EQ(data.node_1[0], 0);
  EXPECT_EQ(data.node_2[0], 1);
  EXPECT_EQ(data.node_3[0], 7);
  EXPECT_EQ(data.node_4[0], 4);
}

TEST(ReadGri3D, TransferToTriangulation)
{
  GriReader<3> reader;
  reader.read_gri("../tests/test3D.gri");

  Triangulation<3> tria;

  EXPECT_NO_THROW({ reader.transfer_to_triangulation(tria); });
}
