#include <array>
#include <cmath>
#include <cstddef>
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

struct MeshData
{
  unsigned int n_nodes;
  unsigned int n_elements;
  unsigned int n_boundary_groups;
  unsigned int n_periodic_groups;

  std::vector<double> x;
  std::vector<double> y;

  std::vector<unsigned int> node_1;
  std::vector<unsigned int> node_2;
  std::vector<unsigned int> node_3;

  std::vector<unsigned int> boundary_group_n_faces;
  std::vector<std::string> boundary_group_names;

  std::vector<unsigned int> boundary_node_1;
  std::vector<unsigned int> boundary_node_2;

  std::vector<unsigned int> periodic_group_n_nodes;

  std::vector<unsigned int> periodic_node_1;
  std::vector<unsigned int> periodic_node_2;

  void set_n_nodes(std::size_t n)
  {
    n_nodes = n;
    x.resize(n);
    y.resize(n);
  }
  void set_n_elements(std::size_t n)
  {
    n_elements = n;
    node_1.resize(n);
    node_2.resize(n);
    node_3.resize(n);
  }

  void set_n_boundary_groups(std::size_t n)
  {
    n_boundary_groups = n;
    boundary_group_n_faces.resize(n);
    boundary_group_names.resize(n);
  }

  void set_n_periodic_groups(std::size_t n)
  {
    n_periodic_groups = n;
    periodic_group_n_nodes.resize(n);
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
  static_assert(dim == 2, "Only 2D meshes are supported");

  Mesh()
  {
    read_gri("../tests/test.gri");
    print();
    reinit();
  }

  void reinit() {}

  void read_gri(const std::string& filename);

  void print() const;

private:
  MeshData data;
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
  in >> dummy;
  data.set_n_nodes(dummy);
  in >> dummy;
  data.set_n_elements(dummy);
  in >> dummy;

  ASSERT(data.n_nodes > 0, "Number of nodes must be greater than zero.");
  ASSERT(data.n_elements > 0, "Number of elements must be greater than zero.");
  ASSERT(data.n_nodes > data.n_elements,
         "Number of elements must be greater than number of nodes");
  ASSERT(dummy == dim, "File has different dimension than mesh.");

  // Grab nodes
  for (unsigned int i = 0; i < data.n_nodes; ++i) {
    in >> data.x[i] >> data.y[i];
  }

  // Grab boundaries
  in >> dummy;
  data.set_n_boundary_groups(dummy);

  for (unsigned int i = 0; i < data.n_boundary_groups; ++i) {
    in >> data.boundary_group_n_faces[i];

    ASSERT(data.boundary_group_n_faces[i] > 0,
           "Boundary groups must have at least one face");

    in >> dummy;

    ASSERT(dummy == dim,
           "Boundary groups nodes must be equal to the dimension");

    std::getline(in >> std::ws, data.boundary_group_names[i]);

    for (unsigned int j = 0; j < data.boundary_group_n_faces[i]; ++j) {
      in >> dummy;
      data.boundary_node_1.emplace_back(dummy);
      in >> dummy;
      data.boundary_node_2.emplace_back(dummy);
    }
  }

  // Grab elements
  // TODO: Assumes only one element group and linear elements (dim + 1 nodes)
  int n_element_in_group;
  int element_group_order;
  std::string element_basis;
  in >> n_element_in_group >> element_group_order;
  std::getline(in >> std::ws, element_basis);

  ASSERT(n_element_in_group == data.n_elements,
         "Multiple element groups are unsupported");
  ASSERT(element_group_order == 1, "Only first order elements are supported");

  for (unsigned int i = 0; i < data.n_elements; ++i) {
    in >> data.node_1[i] >> data.node_2[i] >> data.node_3[i];
  }

  // Periodic groups
  in >> dummy;
  data.set_n_periodic_groups(dummy);

  std::string periodic_group;
  std::getline(in >> std::ws, periodic_group);

  ASSERT(periodic_group == "PeriodicGroup",
         "Invalid .gri file. Periodic groups must contain PeriodicGroup "
         "after the number of periodic groups.");

  for (unsigned int i = 0; i < data.n_periodic_groups; ++i) {
    in >> data.periodic_group_n_nodes[i];

    std::string periodic_group_type;
    std::getline(in >> std::ws, periodic_group_type);

    ASSERT(periodic_group_type == "Translational",
           "Invalid periodicity type " + periodic_group_type);

    for (unsigned int j = 0; j < data.periodic_group_n_nodes[i]; ++j) {
      in >> dummy;
      data.periodic_node_1.emplace_back(dummy);
      in >> dummy;
      data.periodic_node_2.emplace_back(dummy);
    }
  }
}

template<unsigned int dim>
void
Mesh<dim>::print() const
{
}
