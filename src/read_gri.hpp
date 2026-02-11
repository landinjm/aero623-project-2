#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <libassert/assert.hpp>
#include <span>
#include <string>
#include <vector>

/**
 * @brief Types of periodicity
 */
enum PeriodicityType
{
  Translational,
  Rotational
};

/**
 * @brief Invalid id
 */
static constexpr unsigned int invalid_id = static_cast<unsigned int>(-1);

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

  /**
   * @brief Get the span of the elements. Used for range-based loops
   */
  std::span<const std::array<unsigned int, dim + 1>> get_elements() const
  {
    return std::span(elements);
  }

  /**
   * @brief Get the span of the nodes. Used for range-based loops
   */
  std::span<const std::array<double, dim>> get_nodes() const
  {
    return std::span(node_positions);
  }

  void reinit() { n_interior_faces = n_elements * (dim + 1); }

  void read_gri(const std::string& filename);

  void print() const;

private:
  unsigned int n_nodes = invalid_id;
  unsigned int n_elements = invalid_id;
  unsigned int n_boundary_groups = invalid_id;
  unsigned int n_element_groups = invalid_id;
  unsigned int n_periodic_groups = invalid_id;

  std::vector<std::array<double, dim>> node_positions;

  std::vector<std::array<unsigned int, dim + 1>> elements;

  std::vector<unsigned int> boundary_group_n_faces;
  std::vector<unsigned int> boundary_group_n_nodes;
  std::vector<std::string> boundary_group_names;

  std::vector<unsigned int> boundary_faces;

  std::vector<unsigned int> periodic_group_n_nodes;
  std::vector<PeriodicityType> periodic_group_type;

  std::vector<std::pair<unsigned int, unsigned int>> periodic_node_pairs;

  unsigned int n_interior_faces;

  std::vector<std::array<unsigned int, 4>> I2E;
  std::vector<std::array<unsigned int, 3>> B2E;
  std::vector<std::array<double, dim>> In;
  std::vector<std::array<double, dim>> Bn;
  std::vector<double> Area;
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
  boundary_group_n_nodes.resize(n_boundary_groups);
  boundary_group_names.resize(n_boundary_groups);
  for (unsigned int i = 0; i < n_boundary_groups; ++i) {
    in >> boundary_group_n_faces[i] >> boundary_group_n_nodes[i];
    ASSERT(boundary_group_n_faces[i] > 0,
           "Boundary groups must have at least one face");
    ASSERT(boundary_group_n_nodes[i] > 1,
           "Boundary groups must have at least two nodes");
    std::getline(in >> std::ws, boundary_group_names[i]);
    for (unsigned int j = 0; j < boundary_group_n_faces[i]; ++j) {
      for (unsigned int k = 0; k < boundary_group_n_nodes[i]; ++k) {
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
  periodic_group_type.resize(n_periodic_groups);
  for (unsigned int i = 0; i < n_periodic_groups; ++i) {
    std::string periodic_group_type;
    in >> periodic_group_n_nodes[i];
    std::getline(in >> std::ws, periodic_group_type);
    if (periodic_group_type == "Translational") {
      periodic_group_type[i] = PeriodicityType::Translational;
    } else if (periodic_group_type == "Rotational") {
      periodic_group_type[i] = PeriodicityType::Rotational;
    } else {
      ASSERT(false, "Unknown periodicity type " + periodic_group_type);
    }
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
    std::cout << i << " - " << boundary_group_n_faces[i] << " "
              << boundary_group_n_nodes[i] << " " << boundary_group_names[i]
              << "\n";
    for (unsigned int j = 0; j < boundary_group_n_faces[i]; ++j) {
      std::cout << "    ";
      for (unsigned int k = 0; k < boundary_group_n_nodes[i]; ++k) {
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
              << periodic_group_type[i] << "\n";
    for (unsigned int j = 0; j < periodic_group_n_nodes[i]; ++j) {
      std::cout << "    " << periodic_node_pairs[linear_index].first << " "
                << periodic_node_pairs[linear_index].second << "\n";
      linear_index++;
    }
  }
  std::cout << std::endl;
}

/**
 *  @brief Run a mesh verification check
 */
