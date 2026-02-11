#include <Eigen/Dense>
#include <iostream>
#include <read_gri.hpp>

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

  Mesh<2> mesh;

  return 0;
}
