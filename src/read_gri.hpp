#pragma once

#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <libassert/assert.hpp>
#include <string>
#include <vector>

template<unsigned int dim>
class Triangulation;

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

template<unsigned int dim>
class GriReader
{
public:
  static_assert(dim == 2, "Only 2D meshes are supported");

  GriReader() = default;

  void read_gri(const std::string& filename);

  void transfer_to_triangulation(Triangulation<dim>& tria) const
  {
    using Topo = typename Triangulation<dim>::Topo;
    auto& raw = tria.raw;

    raw.vertices.resize(data.n_nodes);
    for (unsigned int i = 0; i < data.n_nodes; ++i)
      raw.vertices[i] = { data.x[i], data.y[i] };

    raw.cell_verts.resize(data.n_elements);
    for (unsigned int i = 0; i < data.n_elements; ++i)
      raw.cell_verts[i] = { data.node_1[i], data.node_2[i], data.node_3[i] };

    raw.cell_flags.assign(data.n_elements, CellFlags::Active);

    tria.build_face_connectivity();

    unsigned int face_offset = 0;
    for (unsigned int g = 0; g < data.n_boundary_groups; ++g) {
      unsigned int n_faces = data.boundary_group_n_faces[g];
      for (unsigned int f = 0; f < n_faces; ++f) {
        // Find the face in raw that matches these two boundary nodes
        std::array<VertexIndexType, 2> key = {
          data.boundary_node_1[face_offset + f],
          data.boundary_node_2[face_offset + f]
        };
        std::sort(key.begin(), key.end());

        for (unsigned int fi = 0; fi < raw.face_verts.size(); ++fi) {
          auto fv = raw.face_verts[fi];
          std::sort(fv.begin(), fv.end());
          if (fv == key) {
            raw.boundary_ids[fi] = static_cast<BoundaryIdType>(g);
            break;
          }
        }
      }
      face_offset += n_faces;
    }

    tria.commit();
  }

private:
  MeshData data;
};

template<unsigned int dim>
void
GriReader<dim>::read_gri(const std::string& filename)
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

    // NOTE: The files assumes indexing from 1 so adjust for that
    for (unsigned int j = 0; j < data.boundary_group_n_faces[i]; ++j) {
      in >> dummy;
      data.boundary_node_1.emplace_back(dummy - 1);
      in >> dummy;
      data.boundary_node_2.emplace_back(dummy - 1);
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

  // NOTE: The files assumes indexing from 1 so adjust for that
  for (unsigned int i = 0; i < data.n_elements; ++i) {
    in >> dummy;
    data.node_1[i] = dummy - 1;
    in >> dummy;
    data.node_2[i] = dummy - 1;
    in >> dummy;
    data.node_3[i] = dummy - 1;
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

    // NOTE: The files assumes indexing from 1 so adjust for that
    for (unsigned int j = 0; j < data.periodic_group_n_nodes[i]; ++j) {
      in >> dummy;
      data.periodic_node_1.emplace_back(dummy - 1);
      in >> dummy;
      data.periodic_node_2.emplace_back(dummy - 1);
    }
  }
}
