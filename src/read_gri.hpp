#pragma once

#include <cmath>
#include <config.hpp>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <triangulation.hpp>
#include <vector>

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

  unsigned int n_boundary_faces() const
  {
    unsigned int result = 0;
    for (auto val : boundary_group_n_faces) {
      result += val;
    }
    return result;
  }

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

  MeshData data() const { return data_; }

  void transfer_to_triangulation(Triangulation<dim>& tria) const
  {
    // Grab the number of cells
    const unsigned int n_cells = data_.n_elements;

    // Build a face map
    using EdgeKey = std::pair<unsigned int, unsigned int>;

    struct RawFace
    {
      // Global vertex indices
      uint v0, v1;

      // Neighboring global cell indices. The first index is always the owner
      // and the second is the neighbor.
      CellIndexType cells[2] = { CellIndexType(-1), CellIndexType(-1) };

      // Whether the face is at a boundary. Note that this is false for periodic
      // faces.
      bool is_boundary = false;
      BoundaryIdType boundary_id = 0;

      // Whether the face is periodic
      bool is_periodic = false;

      // Neighbor global face index
      unsigned int neighbor = unsigned(-1);
    };

    std::map<EdgeKey, FaceIndexType> edge_to_face;
    std::vector<RawFace> raw_faces;

    // Each local face id corresponds to an opposite vertex with the same id.
    // For example face 0: v1-v2, face 1: v2-v0, and face 2:v0-v1
    constexpr unsigned int edge_v0[3] = { 1, 2, 0 };
    constexpr unsigned int edge_v1[3] = { 2, 0, 1 };
    constexpr unsigned int edge_v2[3] = { 0, 1, 2 };

    // Allocate an array for the total number of faces
    std::vector<std::array<FaceIndexType, 3>> cell_face_ids(n_cells);

    // For each cell fill out the face information.
    for (unsigned int c = 0; c < n_cells; ++c) {
      const unsigned int vertex_indices[3] = { data_.node_1[c],
                                               data_.node_2[c],
                                               data_.node_3[c] };

      for (unsigned int f = 0; f < 3; ++f) {
        // Grab the vertex indices from the local face number and make a key.
      }
    }

    for (unsigned int c = 0; c < n_cells; ++c) {
      const unsigned int verts[3] = { data_.node_1[c],
                                      data_.node_2[c],
                                      data_.node_3[c] };

      for (unsigned int lf = 0; lf < 3; ++lf) {
        const unsigned int a = verts[edge_v0[lf]];
        const unsigned int b = verts[edge_v1[lf]];
        const EdgeKey key = { std::min(a, b), std::max(a, b) };

        auto it = edge_to_face.find(key);
        if (it == edge_to_face.end()) {
          const FaceIndexType fi = static_cast<FaceIndexType>(raw_faces.size());
          edge_to_face[key] = fi;
          cell_face_ids[c][lf] = fi;
          RawFace rf;
          rf.v0 = a;
          rf.v1 = b;
          rf.cells[0] = static_cast<CellIndexType>(c);
          raw_faces.push_back(rf);
        } else {
          const FaceIndexType fi = it->second;
          cell_face_ids[c][lf] = fi;
          raw_faces[fi].cells[1] = static_cast<CellIndexType>(c);
        }
      }
    }

    {
      unsigned int offset = 0;
      for (unsigned int g = 0; g < data_.n_boundary_groups; ++g) {
        const BoundaryIdType bid = static_cast<BoundaryIdType>(g + 1);
        for (unsigned int i = 0; i < data_.boundary_group_n_faces[g]; ++i) {
          const unsigned int a = data_.boundary_node_1[offset + i];
          const unsigned int b = data_.boundary_node_2[offset + i];
          const EdgeKey key = { std::min(a, b), std::max(a, b) };
          auto it = edge_to_face.find(key);
          ASSERT(it != edge_to_face.end(),
                 "Boundary edge not found in cell connectivity");
          raw_faces[it->second].is_boundary = true;
          raw_faces[it->second].boundary_id = bid;
        }
        offset += data_.boundary_group_n_faces[g];
      }
    }

    const unsigned int nf = static_cast<unsigned int>(raw_faces.size());
    const unsigned int nv = data_.n_nodes;

    tria.internal_reinit(n_cells, nf, nv);

    // Fill in the vertices
    for (unsigned int v = 0; v < data_.n_nodes; ++v) {
      tria.vertices(v, 0) = data_.x[v];
      tria.vertices(v, 1) = data_.y[v];
    }

    // Fill in the cells
    for (unsigned int c = 0; c < data_.n_elements; ++c) {
      tria.cell_vertices(c, 0) = data_.node_1[c];
      tria.cell_vertices(c, 1) = data_.node_2[c];
      tria.cell_vertices(c, 2) = data_.node_3[c];
      tria.cell_faces(c, 0) = cell_face_ids[c][0];
      tria.cell_faces(c, 1) = cell_face_ids[c][1];
      tria.cell_faces(c, 2) = cell_face_ids[c][2];
    }

    // Fill in the faces
    for (unsigned int f = 0; f < nf; ++f) {
      tria.face_vertices(f, 0) = raw_faces[f].v0;
      tria.face_vertices(f, 1) = raw_faces[f].v1;
      tria.face_cells(f, 0) = raw_faces[f].cells[0];
      tria.face_cells(f, 1) = raw_faces[f].cells[1];
      tria.boundary_ids(f) = raw_faces[f].boundary_id;

      uint8_t flags = 0;
      if (raw_faces[f].is_boundary)
        flags |= FaceFlags::Boundary;
      tria.face_flags(f) = flags;
    }

    // Handle periodicity
    if (data_.n_periodic_groups > 0) {
      std::unordered_map<unsigned int, unsigned int> periodic_node_map;
      unsigned int poffset = 0;
      for (unsigned int g = 0; g < data_.n_periodic_groups; ++g) {
        for (unsigned int i = 0; i < data_.periodic_group_n_nodes[g]; ++i) {
          const unsigned int n1 = data_.periodic_node_1[poffset + i];
          const unsigned int n2 = data_.periodic_node_2[poffset + i];
          periodic_node_map[n1] = n2;
          periodic_node_map[n2] = n1;
        }
        poffset += data_.periodic_group_n_nodes[g];
      }

      for (unsigned int fi = 0; fi < nf; ++fi) {
        if (!(tria.face_flags(fi) & FaceFlags::Boundary))
          continue;
        if (tria.face_flags(fi) & FaceFlags::Periodic)
          continue;

        const unsigned int a = tria.face_vertices(fi, 0);
        const unsigned int b = tria.face_vertices(fi, 1);

        auto it_a = periodic_node_map.find(a);
        auto it_b = periodic_node_map.find(b);
        if (it_a == periodic_node_map.end() || it_b == periodic_node_map.end())
          continue;

        const EdgeKey partner_key = { std::min(it_a->second, it_b->second),
                                      std::max(it_a->second, it_b->second) };
        auto it_face = edge_to_face.find(partner_key);
        if (it_face == edge_to_face.end())
          continue;

        const FaceIndexType fj = it_face->second;
        if (fi == fj)
          continue;

        tria.face_flags(fi) = FaceFlags::Periodic;
        tria.face_flags(fj) = FaceFlags::Periodic;
        tria.periodic_face_neighbor(fi) = static_cast<FaceIndexType>(fj);
        tria.periodic_face_neighbor(fj) = static_cast<FaceIndexType>(fi);

        for (unsigned int d = 0; d < 2; ++d) {
          const double ci = 0.5 * (tria.vertices(a, d) + tria.vertices(b, d));
          const double cj = 0.5 * (tria.vertices(it_a->second, d) +
                                   tria.vertices(it_b->second, d));
          tria.periodic_face_offset(fi, d) = cj - ci;
          tria.periodic_face_offset(fj, d) = ci - cj;
        }
      }
    }
  }

private:
  MeshData data_;
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
  data_.set_n_nodes(dummy);
  in >> dummy;
  data_.set_n_elements(dummy);
  in >> dummy;

  ASSERT(data_.n_nodes > 0, "Number of nodes must be greater than zero.");
  ASSERT(data_.n_elements > 0, "Number of elements must be greater than zero.");
  ASSERT(dummy == dim, "File has different dimension than mesh.");

  // Grab nodes
  for (unsigned int i = 0; i < data_.n_nodes; ++i) {
    in >> data_.x[i] >> data_.y[i];
  }

  // Grab boundaries
  in >> dummy;
  data_.set_n_boundary_groups(dummy);

  for (unsigned int i = 0; i < data_.n_boundary_groups; ++i) {
    in >> data_.boundary_group_n_faces[i];

    ASSERT(data_.boundary_group_n_faces[i] > 0,
           "Boundary groups must have at least one face");

    in >> dummy;

    ASSERT(dummy == dim,
           "Boundary groups nodes must be equal to the dimension");

    std::getline(in >> std::ws, data_.boundary_group_names[i]);

    // NOTE: The files assumes indexing from 1 so adjust for that
    for (unsigned int j = 0; j < data_.boundary_group_n_faces[i]; ++j) {
      in >> dummy;
      data_.boundary_node_1.emplace_back(dummy - 1);
      in >> dummy;
      data_.boundary_node_2.emplace_back(dummy - 1);
    }
  }

  // Grab elements
  // TODO: Assumes only one element group and linear elements (dim + 1 nodes)
  int n_element_in_group;
  int element_group_order;
  std::string element_basis;
  in >> n_element_in_group >> element_group_order;
  std::getline(in >> std::ws, element_basis);

  ASSERT(n_element_in_group == data_.n_elements,
         "Multiple element groups are unsupported");
  ASSERT(element_group_order == 1, "Only first order elements are supported");

  // NOTE: The files assumes indexing from 1 so adjust for that
  for (unsigned int i = 0; i < data_.n_elements; ++i) {
    in >> dummy;
    data_.node_1[i] = dummy - 1;
    in >> dummy;
    data_.node_2[i] = dummy - 1;
    in >> dummy;
    data_.node_3[i] = dummy - 1;
  }

  // Periodic groups
  in >> dummy;
  data_.set_n_periodic_groups(dummy);

  std::string periodic_group;
  std::getline(in >> std::ws, periodic_group);

  ASSERT(periodic_group == "PeriodicGroup",
         "Invalid .gri file. Periodic groups must contain PeriodicGroup "
         "after the number of periodic groups.");

  for (unsigned int i = 0; i < data_.n_periodic_groups; ++i) {
    in >> data_.periodic_group_n_nodes[i];

    std::string periodic_group_type;
    std::getline(in >> std::ws, periodic_group_type);

    ASSERT(periodic_group_type == "Translational",
           "Invalid periodicity type " + periodic_group_type);

    // NOTE: The files assumes indexing from 1 so adjust for that
    for (unsigned int j = 0; j < data_.periodic_group_n_nodes[i]; ++j) {
      in >> dummy;
      data_.periodic_node_1.emplace_back(dummy - 1);
      in >> dummy;
      data_.periodic_node_2.emplace_back(dummy - 1);
    }
  }
}
