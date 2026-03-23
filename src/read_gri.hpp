#pragma once

#include <cmath>
#include <config.hpp>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
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

  // Corner nodes (all elements)
  std::vector<unsigned int> node_1;
  std::vector<unsigned int> node_2;
  std::vector<unsigned int> node_3;

  // Mid-edge nodes (P2 elements only, UINT_MAX for P1)
  std::vector<unsigned int> node_4; // between node_1 and node_2
  std::vector<unsigned int> node_5; // between node_2 and node_3
  std::vector<unsigned int> node_6; // between node_3 and node_1
  std::vector<unsigned int> element_order; // 1 or 2

  std::vector<unsigned int> boundary_group_n_faces;
  std::vector<std::string>  boundary_group_names;
  std::vector<unsigned int> boundary_group_n_nodes; // 2 or 3 per group

  std::vector<unsigned int> boundary_node_1;
  std::vector<unsigned int> boundary_node_2;
  std::vector<unsigned int> boundary_node_3; // UINT_MAX for linear faces

  std::vector<unsigned int> periodic_group_n_nodes;
  std::vector<unsigned int> periodic_node_1;
  std::vector<unsigned int> periodic_node_2;

  unsigned int n_boundary_faces() const
  {
    unsigned int result = 0;
    for (auto val : boundary_group_n_faces)
      result += val;
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
    element_order.resize(n);
    node_1.resize(n);
    node_2.resize(n);
    node_3.resize(n);
    node_4.resize(n, std::numeric_limits<unsigned int>::max());
    node_5.resize(n, std::numeric_limits<unsigned int>::max());
    node_6.resize(n, std::numeric_limits<unsigned int>::max());
  }

  void set_n_boundary_groups(std::size_t n)
  {
    n_boundary_groups = n;
    boundary_group_n_faces.resize(n);
    boundary_group_n_nodes.resize(n);
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

  MeshData get_data() const { return data_; }

  void check_counter_clockwise() const
  {
    for (unsigned int i = 0; i < data_.n_elements; ++i) {
      const double ax = data_.x[data_.node_1[i]], ay = data_.y[data_.node_1[i]];
      const double bx = data_.x[data_.node_2[i]], by = data_.y[data_.node_2[i]];
      const double cx = data_.x[data_.node_3[i]], cy = data_.y[data_.node_3[i]];
      const double cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
      ASSERT(cross > 0.0,
             "Element " + std::to_string(i) + " is not counter-clockwise");
    }
  }

  void transfer_to_triangulation(Triangulation<dim>& tria) const
  {
    check_counter_clockwise();

    const unsigned int n_cells = data_.n_elements;

    using EdgeKey = std::pair<unsigned int, unsigned int>;

    struct RawFace
    {
      unsigned int v0, v1;
      CellIndexType cells[2] = { CellIndexType(-1), CellIndexType(-1) };
      bool is_boundary = false;
      BoundaryIdType boundary_id = 0;
      bool is_periodic = false;
      unsigned int neighbor = unsigned(-1);
    };

    std::map<EdgeKey, FaceIndexType> edge_to_face;
    std::vector<RawFace> raw_faces;

    constexpr unsigned int edge_v0[3] = { 1, 2, 0 };
    constexpr unsigned int edge_v1[3] = { 2, 0, 1 };

    std::vector<std::array<FaceIndexType, 3>> cell_face_ids(n_cells);

    // Build face connectivity from corner nodes only
    for (unsigned int c = 0; c < n_cells; ++c) {
      const unsigned int vertex_indices[3] = { data_.node_1[c],
                                               data_.node_2[c],
                                               data_.node_3[c] };
      for (unsigned int f = 0; f < 3; ++f) {
        const unsigned int a = vertex_indices[edge_v0[f]];
        const unsigned int b = vertex_indices[edge_v1[f]];
        const EdgeKey key = { std::min(a, b), std::max(a, b) };

        auto it = edge_to_face.find(key);
        if (it == edge_to_face.end()) {
          const FaceIndexType face_index =
            static_cast<FaceIndexType>(raw_faces.size());
          cell_face_ids[c][f] = face_index;
          edge_to_face[key] = face_index;
          RawFace rf;
          rf.v0 = a;
          rf.v1 = b;
          rf.cells[0] = static_cast<CellIndexType>(c);
          raw_faces.push_back(rf);
        } else {
          const FaceIndexType face_index = it->second;
          cell_face_ids[c][f] = face_index;
          raw_faces[face_index].cells[1] = static_cast<CellIndexType>(c);
        }
      }
    }

    // Build set of midpoint nodes so we can exclude them from periodic matching
    std::map<unsigned int, bool> midpoint_nodes;
    for (unsigned int c = 0; c < data_.n_elements; ++c) {
      if (data_.element_order[c] == 2) {
        midpoint_nodes[data_.node_4[c]] = true;
        midpoint_nodes[data_.node_5[c]] = true;
        midpoint_nodes[data_.node_6[c]] = true;
      }
    }

    // Build periodic node map — corner nodes only
    std::map<unsigned int, unsigned int> periodic_nodes;
    for (unsigned int i = 0; i < data_.periodic_node_1.size(); ++i) {
      const unsigned int n1 = data_.periodic_node_1[i];
      const unsigned int n2 = data_.periodic_node_2[i];
      if (!midpoint_nodes.count(n1) && !midpoint_nodes.count(n2)) {
        periodic_nodes[n1] = n2;
        periodic_nodes[n2] = n1;
      }
    }

    // Detect periodic faces. If both corner nodes of a boundary face have
    // periodic partners, and those partners form a valid edge, it's periodic.
    for (auto& rf : raw_faces) {
      if (rf.cells[1] != CellIndexType(-1)) continue;

      const auto a = rf.v0;
      const auto b = rf.v1;

      if (periodic_nodes.count(a) && periodic_nodes.count(b)) {
        const auto c = periodic_nodes[a];
        const auto d = periodic_nodes[b];
        const EdgeKey key = { std::min(c, d), std::max(c, d) };

        // If the partner edge doesn't exist in the mesh, these two nodes are
        // periodic individually but not as a face pair — skip.
        if (!edge_to_face.count(key)) continue;

        const FaceIndexType face_index = edge_to_face[key];
        rf.neighbor = face_index;
        rf.cells[1] = raw_faces[face_index].cells[0];
        rf.is_periodic = true;
      }
    }

    // Mark boundary faces
    {
      unsigned int offset = 0;
      for (unsigned int g = 0; g < data_.n_boundary_groups; ++g) {
        const BoundaryIdType bid = static_cast<BoundaryIdType>(g + 1);
        for (unsigned int i = 0; i < data_.boundary_group_n_faces[g]; ++i) {
          const unsigned int a = data_.boundary_node_1[offset + i];
          const unsigned int b = data_.boundary_node_2[offset + i];

          // Skip midpoint-to-anything entries in curved boundary groups
          if (midpoint_nodes.count(a) || midpoint_nodes.count(b)) continue;

          const EdgeKey key = { std::min(a, b), std::max(a, b) };
          auto it = edge_to_face.find(key);
          ASSERT(it != edge_to_face.end(),
                 "Boundary edge not found in cell connectivity");

          if (!raw_faces[it->second].is_periodic) {
            raw_faces[it->second].is_boundary = true;
            raw_faces[it->second].boundary_id = bid;
          }
        }
        offset += data_.boundary_group_n_faces[g];
      }
    }

    const unsigned int n_faces    = static_cast<unsigned int>(raw_faces.size());
    const unsigned int n_vertices = data_.n_nodes;

    tria.internal_reinit(n_cells, n_faces, n_vertices);

    for (unsigned int v = 0; v < data_.n_nodes; ++v) {
      tria.vertices(v, 0) = data_.x[v];
      tria.vertices(v, 1) = data_.y[v];
    }

    for (unsigned int c = 0; c < data_.n_elements; ++c) {
      tria.cell_vertices(c, 0) = data_.node_1[c];
      tria.cell_vertices(c, 1) = data_.node_2[c];
      tria.cell_vertices(c, 2) = data_.node_3[c];
      tria.cell_faces(c, 0) = cell_face_ids[c][0];
      tria.cell_faces(c, 1) = cell_face_ids[c][1];
      tria.cell_faces(c, 2) = cell_face_ids[c][2];

      // Geometry order and nodes (P1 elements leave nodes 4-6 as UINT_MAX)
      tria.cell_geometry_order(c) = static_cast<uint8_t>(data_.element_order[c]);
      tria.cell_geometry_nodes(c, 0) = data_.node_1[c];
      tria.cell_geometry_nodes(c, 1) = data_.node_2[c];
      tria.cell_geometry_nodes(c, 2) = data_.node_3[c];
      tria.cell_geometry_nodes(c, 3) = data_.node_4[c];
      tria.cell_geometry_nodes(c, 4) = data_.node_5[c];
      tria.cell_geometry_nodes(c, 5) = data_.node_6[c];
    }

    for (unsigned int f = 0; f < n_faces; ++f) {
      tria.face_vertices(f, 0) = raw_faces[f].v0;
      tria.face_vertices(f, 1) = raw_faces[f].v1;

      // Default to linear face geometry
      tria.face_geometry_order(f) = uint8_t(1);
      tria.face_geometry_nodes(f, 0) = raw_faces[f].v0;
      tria.face_geometry_nodes(f, 1) = VertexIndexType(-1);
      tria.face_geometry_nodes(f, 2) = raw_faces[f].v1;

      tria.face_cells(f, 0) = raw_faces[f].cells[0];
      tria.face_cells(f, 1) = raw_faces[f].cells[1];
      tria.boundary_ids(f)  = raw_faces[f].boundary_id;

      tria.periodic_face_neighbor(f) =
        raw_faces[f].is_periodic
          ? static_cast<FaceIndexType>(raw_faces[f].neighbor)
          : FaceIndexType(-1);

      if (raw_faces[f].is_periodic) {
        const unsigned int neighbor = raw_faces[f].neighbor;
        const double mx0 = 0.5 * (data_.x[raw_faces[f].v0] + data_.x[raw_faces[f].v1]);
        const double my0 = 0.5 * (data_.y[raw_faces[f].v0] + data_.y[raw_faces[f].v1]);
        const double mx1 = 0.5 * (data_.x[raw_faces[neighbor].v0] + data_.x[raw_faces[neighbor].v1]);
        const double my1 = 0.5 * (data_.y[raw_faces[neighbor].v0] + data_.y[raw_faces[neighbor].v1]);
        tria.periodic_face_offset(f, 0) = mx1 - mx0;
        tria.periodic_face_offset(f, 1) = my1 - my0;
      } else {
        tria.periodic_face_offset(f, 0) = 0.0;
        tria.periodic_face_offset(f, 1) = 0.0;
      }

      uint32_t flags = 0;
      if (raw_faces[f].is_boundary) flags |= FaceFlags::Boundary;
      if (raw_faces[f].is_periodic) flags |= FaceFlags::Periodic;
      tria.face_flags(f) = flags;
    }

    // Overwrite face geometry for curved boundary faces (3-node groups)
    {
      unsigned int offset = 0;
      for (unsigned int g = 0; g < data_.n_boundary_groups; ++g) {
        for (unsigned int i = 0; i < data_.boundary_group_n_faces[g]; ++i) {
          if (data_.boundary_group_n_nodes[g] == 3) {
            const unsigned int a   = data_.boundary_node_1[offset + i];
            const unsigned int b   = data_.boundary_node_2[offset + i];
            const unsigned int mid = data_.boundary_node_3[offset + i];

            // Skip midpoint-to-midpoint entries
            if (midpoint_nodes.count(a) || midpoint_nodes.count(b)) continue;

            const EdgeKey key = { std::min(a, b), std::max(a, b) };
            auto it = edge_to_face.find(key);
            ASSERT(it != edge_to_face.end(),
                   "Boundary edge not found when assigning curved face geometry");

            const FaceIndexType f = it->second;
            tria.face_geometry_order(f) = uint8_t(2);

            if (tria.face_vertices(f, 0) == a && tria.face_vertices(f, 1) == b) {
              tria.face_geometry_nodes(f, 0) = a;
              tria.face_geometry_nodes(f, 1) = mid;
              tria.face_geometry_nodes(f, 2) = b;
            } else if (tria.face_vertices(f, 0) == b && tria.face_vertices(f, 1) == a) {
              tria.face_geometry_nodes(f, 0) = b;
              tria.face_geometry_nodes(f, 1) = mid;
              tria.face_geometry_nodes(f, 2) = a;
            } else {
              ASSERT(false, "Face endpoint mismatch when assigning curved face geometry");
            }
          }
        }
        offset += data_.boundary_group_n_faces[g];
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
  std::ifstream in(filename);
  ASSERT(in, "Unable to open file " + filename);

  int dummy;
  in >> dummy; data_.set_n_nodes(dummy);
  in >> dummy; data_.set_n_elements(dummy);
  in >> dummy;

  ASSERT(data_.n_nodes > 0,    "Number of nodes must be greater than zero.");
  ASSERT(data_.n_elements > 0, "Number of elements must be greater than zero.");
  ASSERT(dummy == dim,         "File has different dimension than mesh.");

  for (unsigned int i = 0; i < data_.n_nodes; ++i)
    in >> data_.x[i] >> data_.y[i];

  in >> dummy;
  data_.set_n_boundary_groups(dummy);

  for (unsigned int i = 0; i < data_.n_boundary_groups; ++i) {
    in >> data_.boundary_group_n_faces[i];
    ASSERT(data_.boundary_group_n_faces[i] > 0,
           "Boundary groups must have at least one face");

    in >> dummy;
    data_.boundary_group_n_nodes[i] = dummy;
    ASSERT(dummy == 2 || dummy == 3,
           "Boundary groups must have 2 or 3 nodes per face");

    std::getline(in >> std::ws, data_.boundary_group_names[i]);

    for (unsigned int j = 0; j < data_.boundary_group_n_faces[i]; ++j) {
      in >> dummy; data_.boundary_node_1.emplace_back(dummy - 1);
      in >> dummy; data_.boundary_node_2.emplace_back(dummy - 1);

      if (data_.boundary_group_n_nodes[i] == 3) {
        in >> dummy; data_.boundary_node_3.emplace_back(dummy - 1);
      } else {
        data_.boundary_node_3.emplace_back(std::numeric_limits<unsigned int>::max());
      }
    }
  }

  // Element groups — support multiple groups and mixed P1/P2
  unsigned int elem_counter = 0;
  while (elem_counter < data_.n_elements) {
    int n_in_group, order;
    std::string basis;
    in >> n_in_group >> order;
    std::getline(in >> std::ws, basis);

    ASSERT(basis == "TriLagrange", "Only TriLagrange elements are supported");
    ASSERT(order == 1 || order == 2, "Only P1 and P2 elements are supported");

    for (int i = 0; i < n_in_group; ++i) {
      ASSERT(elem_counter < data_.n_elements,
             "Read more elements than header specifies");

      data_.element_order[elem_counter] = order;

      if (order == 1) {
        in >> dummy; data_.node_1[elem_counter] = dummy - 1;
        in >> dummy; data_.node_2[elem_counter] = dummy - 1;
        in >> dummy; data_.node_3[elem_counter] = dummy - 1;
      } else {
        // P2 node ordering in .gri: n1, n4(mid 1-2), n2, n6(mid 3-1), n5(mid 2-3), n3
        in >> dummy; data_.node_1[elem_counter] = dummy - 1;
        in >> dummy; data_.node_4[elem_counter] = dummy - 1;
        in >> dummy; data_.node_2[elem_counter] = dummy - 1;
        in >> dummy; data_.node_6[elem_counter] = dummy - 1;
        in >> dummy; data_.node_5[elem_counter] = dummy - 1;
        in >> dummy; data_.node_3[elem_counter] = dummy - 1;
      }
      ++elem_counter;
    }
  }
  ASSERT(elem_counter == data_.n_elements,
         "Did not read expected number of elements");

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

    for (unsigned int j = 0; j < data_.periodic_group_n_nodes[i]; ++j) {
      in >> dummy; data_.periodic_node_1.emplace_back(dummy - 1);
      in >> dummy; data_.periodic_node_2.emplace_back(dummy - 1);
    }
  }
}