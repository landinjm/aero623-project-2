#include <Eigen/Dense>
#include <iostream>
#include <read_gri.hpp>
#include <solver.hpp>

int
main(int argc, char** argv)
{
  // Create a matrix and print it
  Eigen::MatrixXd m(2, 2);
  m(0, 0) = 3;
  m(1, 0) = 2.5;
  m(0, 1) = -1;
  m(1, 1) = m(1, 0) + m(0, 1);
  std::cout << m << std::endl;

  GriReader<2> reader;
  reader.read_gri("../grids/coarse_local_refinement_1.gri");
  auto data = reader.get_data();

  Mesh<2, double> mesh(reader.get_data());

  Solver<2, 1, double> solver;
  solver.set_free_stream_initial_state(mesh.get_elements());
  solver.compute_free_stream_residual(mesh.get_interior_faces(),
                                      mesh.get_boundary_faces(),
                                      mesh.get_periodic_faces(),
                                      mesh.get_elements());
  std::cout << l2_norm(mesh.get_elements().residual_density) << std::endl;
  std::cout << l2_norm(mesh.get_elements().residual_momentum_x) << std::endl;
  std::cout << l2_norm(mesh.get_elements().residual_momentum_y) << std::endl;
  std::cout << l2_norm(mesh.get_elements().residual_energy) << std::endl;

  return 0;
}
