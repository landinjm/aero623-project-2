#include <gtest/gtest.h>
#include <limits>
#include <read_gri.hpp>
#include <triangulation.hpp>

TEST(ReadGri, test_mesh_1)
{
  GriReader<2> reader;
  reader.read_gri("../tests/test.gri");

  Triangulation<2, double> mesh(reader.get_data());

  auto error = mesh_verification(mesh.get_interior_faces(),
                                 mesh.get_boundary_faces(),
                                 mesh.get_periodic_faces(),
                                 mesh.get_elements());
  EXPECT_NEAR(error, 0, std::numeric_limits<double>::epsilon());
}

TEST(ReadGri, test_mesh_2)
{
  GriReader<2> reader;
  reader.read_gri("../tests/test_2.gri");

  Triangulation<2, double> mesh(reader.get_data());

  auto error = mesh_verification(mesh.get_interior_faces(),
                                 mesh.get_boundary_faces(),
                                 mesh.get_periodic_faces(),
                                 mesh.get_elements());
  EXPECT_NEAR(error, 0, std::numeric_limits<double>::epsilon());
}

TEST(ReadGri, test_mesh_3)
{
  GriReader<2> reader;
  reader.read_gri("../tests/test_3.gri");

  Triangulation<2, double> mesh(reader.get_data());

  auto error = mesh_verification(mesh.get_interior_faces(),
                                 mesh.get_boundary_faces(),
                                 mesh.get_periodic_faces(),
                                 mesh.get_elements());
  EXPECT_NEAR(error, 0, std::numeric_limits<double>::epsilon());
}

TEST(ReadGri, test_mesh_4)
{
  GriReader<2> reader;
  reader.read_gri("../tests/test_4.gri");

  Triangulation<2, double> mesh(reader.get_data());

  auto error = mesh_verification(mesh.get_interior_faces(),
                                 mesh.get_boundary_faces(),
                                 mesh.get_periodic_faces(),
                                 mesh.get_elements());
  EXPECT_NEAR(error, 0, std::numeric_limits<double>::epsilon());
}

TEST(ReadGri, test_mesh_5)
{
  GriReader<2> reader;
  reader.read_gri("../tests/test_5.gri");

  Triangulation<2, double> mesh(reader.get_data());

  auto error = mesh_verification(mesh.get_interior_faces(),
                                 mesh.get_boundary_faces(),
                                 mesh.get_periodic_faces(),
                                 mesh.get_elements());
  EXPECT_NEAR(error, 0, std::numeric_limits<double>::epsilon());
}
