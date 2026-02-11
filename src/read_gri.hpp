#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <libassert/assert.hpp>
#include <string>
#include <vector>

struct InteriorFaceData
{
  std::vector<unsigned int> elem_l;
  std::vector<unsigned int> face_l;
  std::vector<unsigned int> elem_r;
  std::vector<unsigned int> face_r;

  std::vector<double> normal_x;
  std::vector<double> normal_y;

  std::vector<double> face_area;

  std::size_t size() const { return elem_l.size(); }

  void reserve(std::size_t n)
  {
    elem_l.reserve(n);
    face_l.reserve(n);
    elem_r.reserve(n);
    face_r.reserve(n);
    normal_x.reserve(n);
    normal_y.reserve(n);
    face_area.reserve(n);
  }
};

struct BoundaryFaceData
{
  std::vector<unsigned int> elem;
  std::vector<unsigned int> face;

  std::vector<double> normal_x;
  std::vector<double> normal_y;

  std::vector<double> face_area;

  std::size_t size() const { return elem.size(); }

  void reserve(std::size_t n)
  {
    elem.reserve(n);
    face.reserve(n);
    normal_x.reserve(n);
    normal_y.reserve(n);
    face_area.reserve(n);
  }
};

struct ElementData
{
  std::vector<double> area;
  std::vector<double> inv_area;

  std::vector<double> density;
  std::vector<double> momentum_x;
  std::vector<double> momentum_y;
  std::vector<double> energy;

  std::size_t size() const { return area.size(); }

  void reserve(std::size_t n)
  {
    area.reserve(n);
    inv_area.reserve(n);
    density.reserve(n);
    momentum_x.reserve(n);
    momentum_y.reserve(n);
    energy.reserve(n);
  }
};

/**
 * @brief For mesh verification, we simply take the sum of the normal vectors
 * mutliplied with boundary area at each element. The l2-norm of this error is
 * returned at the end.
 */
template<unsigned int dim>
double
mesh_verification(const InteriorFaceData& interior_face_scratch,
                  const std::vector<BoundaryFaceData>& boundary_face_scratches,
                  const ElementData& element_scratch)
{
  std::vector<std::array<double, dim>> elem_sum(element_scratch.size());

  for (unsigned int i = 0; i < interior_face_scratch.size(); ++i) {
    const auto elem_l = interior_face_scratch.elem_l[i];
    const auto elem_r = interior_face_scratch.elem_r[i];

    elem_sum[elem_l][0] +=
      interior_face_scratch.normal_x[i] * interior_face_scratch.face_area[i];
    elem_sum[elem_l][1] +=
      interior_face_scratch.normal_y[i] * interior_face_scratch.face_area[i];
    elem_sum[elem_r][0] -=
      interior_face_scratch.normal_x[i] * interior_face_scratch.face_area[i];
    elem_sum[elem_r][1] -=
      interior_face_scratch.normal_y[i] * interior_face_scratch.face_area[i];
  }

  for (const auto boundary_face_scratch : boundary_face_scratches) {
    for (unsigned int i = 0; i < boundary_face_scratch.size(); ++i) {
      const auto elem = boundary_face_scratch.elem[i];
      elem_sum[elem][0] +=
        boundary_face_scratch.normal_x[i] * boundary_face_scratch.face_area[i];
      elem_sum[elem][1] +=
        boundary_face_scratch.normal_y[i] * boundary_face_scratch.face_area[i];
    }
  }

  double error = 0.0;
  for (unsigned int i = 0; i < elem_sum.size(); ++i) {
    error += elem_sum[i][0] * elem_sum[i][0] + elem_sum[i][1] * elem_sum[i][1];
  }
  return std::sqrt(error);
}

/**
 * @brief Mesh class.
 */
template<unsigned int dim>
class Mesh
{
public:
  static_assert(dim == 2 || dim == 3, "Only 2D and 3D meshes are supported");

  Mesh()
  {
    read_gri("../tests/test.gri");
    reinit();
  }

  void reinit() {}

  void read_gri(const std::string& filename);

  void print() const;

private:
  unsigned int n_nodes;
  unsigned int n_elements;
  unsigned int n_boundary_groups;
  unsigned int n_periodic_groups;

  std::vector<std::array<unsigned int, dim + 1>> elements;
  std::vector<std::pair<unsigned int, unsigned int>> periodic_node_pairs;
  std::vector<std::array<double, dim>> node_positions;
  std::vector<unsigned int> boundary_group_n_faces;
  std::vector<std::string> boundary_group_names;
  std::vector<unsigned int> boundary_faces;
  std::vector<unsigned int> periodic_group_n_nodes;
};

template<unsigned int dim>
void
Mesh<dim>::read_gri(const std::string& filename)
{
  // Filestream
  std::ifstream in(filename);
  ASSERT(in, "Unable to open file " + filename);

  // Grab global information
  int dummy;
  in >> n_nodes >> n_elements >> dummy;
  ASSERT(n_nodes > 0, "Number of nodes must be greater than zero.");
  ASSERT(n_elements > 0, "Number of elements must be greater than zero.");
  ASSERT(n_nodes > n_elements,
         "Number of elements must be greater than number of nodes");

  // Grab nodes
  node_positions.resize(n_nodes);
  for (unsigned int i = 0; i < n_nodes; ++i) {
    std::array<double, dim> pos;
    in >> pos[0] >> pos[1];
    if constexpr (dim == 3) {
      in >> pos[2];
    }
    node_positions[i] = pos;
  }

  // Grab boundaries
  in >> n_boundary_groups;
  boundary_group_n_faces.resize(n_boundary_groups);
  boundary_group_names.resize(n_boundary_groups);
  for (unsigned int i = 0; i < n_boundary_groups; ++i) {
    in >> boundary_group_n_faces[i] >> dummy;
    ASSERT(boundary_group_n_faces[i] > 0,
           "Boundary groups must have at least one face");
    ASSERT(dummy == dim,
           "Boundary groups nodes must be equal to the dimension");
    std::getline(in >> std::ws, boundary_group_names[i]);
    for (unsigned int j = 0; j < boundary_group_n_faces[i]; ++j) {
      for (unsigned int k = 0; k < dim; ++k) {
        in >> dummy;
        boundary_faces.emplace_back(dummy);
      }
    }
  }

  // Grab elements
  // TODO: Assumes only one element group and linear elements (dim + 1 nodes)
  int n_element_in_group;
  int element_group_order;
  std::string element_basis;
  in >> n_element_in_group >> element_group_order;
  std::getline(in >> std::ws, element_basis);
  ASSERT(n_element_in_group == n_elements,
         "Multiple element groups are unsupported");
  ASSERT(element_group_order == 1, "Only first order elements are supported");

  elements.resize(n_elements);
  for (unsigned int i = 0; i < n_elements; ++i) {
    std::array<unsigned int, dim + 1> nodes;
    in >> nodes[0] >> nodes[1] >> nodes[2];
    if constexpr (dim == 3) {
      in >> nodes[3];
    }
    elements[i] = nodes;
  }

  // Periodic groups
  std::string periodic_group;
  in >> n_periodic_groups;
  std::getline(in >> std::ws, periodic_group);
  ASSERT(periodic_group == "PeriodicGroup",
         "Invalid .gri file. Periodic groups must contain PeriodicGroup "
         "after the number of periodic groups.");
  periodic_group_n_nodes.resize(n_periodic_groups);
  for (unsigned int i = 0; i < n_periodic_groups; ++i) {
    std::string periodic_group_type;
    in >> periodic_group_n_nodes[i];
    std::getline(in >> std::ws, periodic_group_type);
    ASSERT(periodic_group_type == "Translational",
           "Invalid periodicity type " + periodic_group_type);
    for (unsigned int j = 0; j < periodic_group_n_nodes[i]; ++j) {
      std::pair<unsigned int, unsigned int> periodic_pair;
      in >> periodic_pair.first >> periodic_pair.second;
      periodic_node_pairs.emplace_back(periodic_pair);
    }
  }
}

template<unsigned int dim>
void
Mesh<dim>::print() const
{
  std::cout << "Number of nodes: " << n_nodes << "\n"
            << "Number of elements: " << n_elements << "\n"
            << "Number of boundary groups: " << n_boundary_groups << "\n"
            << std::endl;

  std::cout << "Node ids - positions" << std::endl;
  for (unsigned int i = 0; i < n_nodes; ++i) {
    std::cout << i << " - " << node_positions[i][0] << " "
              << node_positions[i][1];
    if constexpr (dim == 3) {
      std::cout << " " << node_positions[i][2];
    }
    std::cout << "\n";
  }
  std::cout << std::endl;

  std::cout << "Elements ids - nodes" << std::endl;
  for (unsigned int i = 0; i < n_elements; ++i) {
    std::cout << i << " - " << elements[i][0] << " " << elements[i][1] << " "
              << elements[i][2];
    if constexpr (dim == 3) {
      std::cout << " " << node_positions[i][3];
    }
    std::cout << "\n";
  }
  std::cout << std::endl;

  std::cout << "Boundary group id - faces, nodes, and title" << std::endl;
  unsigned int linear_index = 0;
  for (unsigned int i = 0; i < n_boundary_groups; ++i) {
    std::cout << i << " - " << boundary_group_n_faces[i] << " " << dim << " "
              << boundary_group_names[i] << "\n";
    for (unsigned int j = 0; j < boundary_group_n_faces[i]; ++j) {
      std::cout << "    ";
      for (unsigned int k = 0; k < dim; ++k) {
        std::cout << boundary_faces[linear_index] << " ";
        linear_index++;
      }
      std::cout << "\n";
    }
  }
  std::cout << std::endl;

  std::cout << "Periodic group id - nodes and type" << std::endl;
  linear_index = 0;
  for (unsigned int i = 0; i < n_periodic_groups; ++i) {
    std::cout << i << " - " << periodic_group_n_nodes[i] << " "
              << "Translational" << "\n";
    for (unsigned int j = 0; j < periodic_group_n_nodes[i]; ++j) {
      std::cout << "    " << periodic_node_pairs[linear_index].first << " "
                << periodic_node_pairs[linear_index].second << "\n";
      linear_index++;
    }
  }
  std::cout << std::endl;
}
