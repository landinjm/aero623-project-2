#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <config.hpp>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <triangulation.hpp>
#include <unordered_map>
#include <vector>

struct MeshData {
  unsigned int n_nodes;
  unsigned int n_elements;
  unsigned int n_boundary_groups;
  unsigned int n_periodic_groups;

  std::vector<double> x;
  std::vector<double> y;

  // Element connectivity
  std::vector<unsigned int> element_order; // 1 or 2

  std::vector<unsigned int> node_1;
  std::vector<unsigned int> node_2;
  std::vector<unsigned int> node_3;
  std::vector<unsigned int> node_4;
  std::vector<unsigned int> node_5;
  std::vector<unsigned int> node_6;

  // Boundary data
  std::vector<unsigned int> boundary_group_n_faces;
  std::vector<std::string> boundary_group_names;
  std::vector<unsigned int> boundary_group_n_nodes; // 2 or 3 for each group
  std::vector<unsigned int> boundary_node_1;
  std::vector<unsigned int> boundary_node_2;
  std::vector<unsigned int> boundary_node_3; // UINT_MAX for linear faces

  // Periodic data
  std::vector<unsigned int> periodic_group_n_nodes;
  std::vector<unsigned int> periodic_node_1;
  std::vector<unsigned int> periodic_node_2;

  void set_n_nodes(std::size_t n) {
    n_nodes = n;
    x.resize(n);
    y.resize(n);
  }
  

  void set_n_elements(std::size_t n) {
    n_elements = n;
    element_order.resize(n);

    node_1.resize(n);
    node_2.resize(n);
    node_3.resize(n);
    node_4.resize(n, std::numeric_limits<unsigned int>::max());
    node_5.resize(n, std::numeric_limits<unsigned int>::max());
    node_6.resize(n, std::numeric_limits<unsigned int>::max());
  }

  void set_n_boundary_groups(std::size_t n) {
    n_boundary_groups = n;
    boundary_group_n_faces.resize(n);
    boundary_group_n_nodes.resize(n);
    boundary_group_names.resize(n);
  }

  void set_n_periodic_groups(std::size_t n) {
    n_periodic_groups = n;
    periodic_group_n_nodes.resize(n);
  }

  // NEW FUNCTION -------------------
  unsigned int n_boundary_faces() const {
    unsigned int result = 0;
    for (auto val : boundary_group_n_faces) {
      result += val;
    }
    return result;
  }
};

template <unsigned int dim> class GriReader {
public:
  static_assert(dim == 2, "Only 2D meshes are supported");

  GriReader() = default;

  void read_gri(const std::string &filename);

  const MeshData &get_data() const { return data; }

  void check_counter_clockwise() const {
    for (unsigned int i = 0; i < data.n_elements; ++i) {
        const double ax = data.x[data.node_1[i]];
        const double ay = data.y[data.node_1[i]];

        const double bx = data.x[data.node_2[i]];
        const double by = data.y[data.node_2[i]];

        const double cx = data.x[data.node_3[i]];
        const double cy = data.y[data.node_3[i]];
      // z-component of (B-A) x (C-A)
      const double cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);

      ASSERT(cross > 0.0,
             "Element " + std::to_string(i) + " is not counter-clockwise");
    }
  }
  void transfer_to_triangulation(Triangulation<dim> &tria) const {
    // Check orientation first
    check_counter_clockwise();

    const unsigned int n_cells = data.n_elements;

    using EdgeKey = std::pair<unsigned int, unsigned int>;

    struct RawFace {
      unsigned int v0, v1;
      CellIndexType cells[2] = {CellIndexType(-1), CellIndexType(-1)};

      bool is_boundary = false;
      BoundaryIdType boundary_id = 0;

      bool is_periodic = false;
      unsigned int neighbor = unsigned(-1);
    };

    std::map<EdgeKey, FaceIndexType> edge_to_face;
    std::vector<RawFace> raw_faces;

    constexpr unsigned int edge_v0[3] = {1, 2, 0};
    constexpr unsigned int edge_v1[3] = {2, 0, 1};

    std::vector<std::array<FaceIndexType, 3>> cell_face_ids(n_cells);

    // -------------------------------
    // Build face connectivity from corner nodes
    // -------------------------------
    for (unsigned int c = 0; c < n_cells; ++c) {
      const unsigned int vertex_indices[3] = {data.node_1[c], data.node_2[c],
                                              data.node_3[c]};

      for (unsigned int f = 0; f < 3; ++f) {
        const unsigned int a = vertex_indices[edge_v0[f]];
        const unsigned int b = vertex_indices[edge_v1[f]];
        const EdgeKey key = {std::min(a, b), std::max(a, b)};

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

    // -------------------------------
    // Build periodic node lookup
    // -------------------------------
    std::map<unsigned int, unsigned int> periodic_nodes;
    for (unsigned int i = 0; i < data.periodic_node_1.size(); ++i) {
      periodic_nodes[data.periodic_node_1[i]] = data.periodic_node_2[i];
      periodic_nodes[data.periodic_node_2[i]] = data.periodic_node_1[i];
    }

    // -------------------------------
    // Detect periodic faces
    // -------------------------------
    for (auto &rf : raw_faces) {
      if (rf.cells[1] != CellIndexType(-1)) {
        continue;
      }

      const auto a = rf.v0;
      const auto b = rf.v1;

      if (periodic_nodes.contains(a) && periodic_nodes.contains(b)) {
        const auto c = periodic_nodes[a];
        const auto d = periodic_nodes[b];
        const EdgeKey key = {std::min(c, d), std::max(c, d)};

        ASSERT(edge_to_face.contains(key), "Periodic edge from " + 
          std::to_string(c) + " and " + std::to_string(d) +  " match not found");

        const FaceIndexType face_index = edge_to_face[key];
        rf.neighbor = face_index;
        rf.cells[1] = raw_faces[face_index].cells[0];
        rf.is_periodic = true;
      }
    }

    // -------------------------------
    // Mark boundary faces from boundary groups
    // -------------------------------
    {
      unsigned int offset = 0;

      for (unsigned int g = 0; g < data.n_boundary_groups; ++g) {
        const BoundaryIdType bid = static_cast<BoundaryIdType>(g + 1);

        for (unsigned int i = 0; i < data.boundary_group_n_faces[g]; ++i) {
          const unsigned int a = data.boundary_node_1[offset + i];
          const unsigned int b = data.boundary_node_2[offset + i];

          const EdgeKey key = {std::min(a, b), std::max(a, b)};
          auto it = edge_to_face.find(key);

          ASSERT(it != edge_to_face.end(),
                 "Boundary edge not found in cell connectivity");

          if (!raw_faces[it->second].is_periodic) {
            raw_faces[it->second].is_boundary = true;
            raw_faces[it->second].boundary_id = bid;
          }
        }

        offset += data.boundary_group_n_faces[g];
      }
    }
    const unsigned int n_faces = static_cast<unsigned int>(raw_faces.size());
    const unsigned int n_vertices = data.n_nodes;

    tria.internal_reinit(n_cells, n_faces, n_vertices);

    // -------------------------------
    // Fill vertex coordinates
    // -------------------------------
    for (unsigned int v = 0; v < data.n_nodes; ++v) {
      tria.vertices(v, 0) = data.x[v];
      tria.vertices(v, 1) = data.y[v];
    }

    // -------------------------------
    // Fill cell topology + geometry
    // -------------------------------
    for (unsigned int c = 0; c < data.n_elements; ++c) {
      // Topology
      tria.cell_vertices(c, 0) = data.node_1[c];
      tria.cell_vertices(c, 1) = data.node_2[c];
      tria.cell_vertices(c, 2) = data.node_3[c];

      tria.cell_faces(c, 0) = cell_face_ids[c][0];
      tria.cell_faces(c, 1) = cell_face_ids[c][1];
      tria.cell_faces(c, 2) = cell_face_ids[c][2];

      // Geometry
      tria.cell_geometry_order(c) = static_cast<uint8_t>(data.element_order[c]);

      tria.cell_geometry_nodes(c, 0) = data.node_1[c];
      tria.cell_geometry_nodes(c, 1) = data.node_2[c];
      tria.cell_geometry_nodes(c, 2) = data.node_3[c];
      tria.cell_geometry_nodes(c, 3) = data.node_4[c];
      tria.cell_geometry_nodes(c, 4) = data.node_5[c];
      tria.cell_geometry_nodes(c, 5) = data.node_6[c];
    }

    // -------------------------------
    // Fill faces
    // -------------------------------
    for (unsigned int f = 0; f < n_faces; ++f) {
      tria.face_vertices(f, 0) = raw_faces[f].v0;
      tria.face_vertices(f, 1) = raw_faces[f].v1;

      // Default face geometry: linear
      tria.face_geometry_order(f) = uint8_t(1);
      tria.face_geometry_nodes(f, 0) = raw_faces[f].v0;
      tria.face_geometry_nodes(f, 1) = VertexIndexType(-1);
      tria.face_geometry_nodes(f, 2) = raw_faces[f].v1;

      tria.face_cells(f, 0) = raw_faces[f].cells[0];
      tria.face_cells(f, 1) = raw_faces[f].cells[1];

      tria.boundary_ids(f) = raw_faces[f].boundary_id;

      tria.periodic_face_neighbor(f) =
          raw_faces[f].is_periodic
              ? static_cast<FaceIndexType>(raw_faces[f].neighbor)
              : FaceIndexType(-1);

      if (raw_faces[f].is_periodic) {
        const unsigned int neighbor = raw_faces[f].neighbor;

        const double mx0 =
            0.5 * (data.x[raw_faces[f].v0] + data.x[raw_faces[f].v1]);
        const double my0 =
            0.5 * (data.y[raw_faces[f].v0] + data.y[raw_faces[f].v1]);

        const double mx1 = 0.5 * (data.x[raw_faces[neighbor].v0] +
                                  data.x[raw_faces[neighbor].v1]);
        const double my1 = 0.5 * (data.y[raw_faces[neighbor].v0] +
                                  data.y[raw_faces[neighbor].v1]);

        tria.periodic_face_offset(f, 0) = mx1 - mx0;
        tria.periodic_face_offset(f, 1) = my1 - my0;
      } else {
        tria.periodic_face_offset(f, 0) = 0.0;
        tria.periodic_face_offset(f, 1) = 0.0;
      }

      uint32_t flags = 0;
      if (raw_faces[f].is_boundary)
        flags |= FaceFlags::Boundary;
      if (raw_faces[f].is_periodic)
        flags |= FaceFlags::Periodic;

      tria.face_flags(f) = flags;
    }

    // ---------------------------------------
    // Overwrite curved boundary face geometry
    // for groups that have 3 nodes per face
    // ---------------------------------------
    {
      unsigned int offset = 0;

      for (unsigned int g = 0; g < data.n_boundary_groups; ++g) {
        for (unsigned int i = 0; i < data.boundary_group_n_faces[g]; ++i) {
          const unsigned int a = data.boundary_node_1[offset + i];
          const unsigned int b = data.boundary_node_2[offset + i];

          const EdgeKey key = {std::min(a, b), std::max(a, b)};
          auto it = edge_to_face.find(key);

          ASSERT(it != edge_to_face.end(),
                 "Boundary edge not found when assigning face geometry");

          const FaceIndexType f = it->second;

          if (data.boundary_group_n_nodes[g] == 3) {
            const unsigned int mid = data.boundary_node_3[offset + i];

            tria.face_geometry_order(f) = uint8_t(2);

            // Keep face geometry ordering consistent with stored face endpoints
            if (tria.face_vertices(f, 0) == a &&
                tria.face_vertices(f, 1) == b) {
              tria.face_geometry_nodes(f, 0) = a;
              tria.face_geometry_nodes(f, 1) = mid;
              tria.face_geometry_nodes(f, 2) = b;
            } else if (tria.face_vertices(f, 0) == b &&
                       tria.face_vertices(f, 1) == a) {
              tria.face_geometry_nodes(f, 0) = b;
              tria.face_geometry_nodes(f, 1) = mid;
              tria.face_geometry_nodes(f, 2) = a;
            } else {
              ASSERT(
                  false,
                  "Face endpoint mismatch when assigning curved face geometry");
            }
          }
        }

        offset += data.boundary_group_n_faces[g];
      }
    }
  }
  
private: 
  MeshData data;
};

template <unsigned int dim>
void GriReader<dim>::read_gri(const std::string &filename) {
  std::ifstream in(filename);
  assert(in && "Unable to open file ");
  int dummy;

  // Global header
  in >> dummy;
  data.set_n_nodes(dummy);

  in >> dummy;
  data.set_n_elements(dummy);

  in >> dummy;
  assert(dummy == dim && "File has different dimension than mesh.");

  assert(data.n_nodes > 0 && "Number of nodes must be greater than zero.");
  assert(data.n_elements > 0 &&
         "Number of elements must be greater than zero.");

  // Grab nodes
  for (unsigned int i = 0; i < data.n_nodes; ++i) {
    in >> data.x[i] >> data.y[i];
  }

  // Boundary groups
  in >> dummy;
  data.set_n_boundary_groups(dummy);

  for (unsigned int i = 0; i < data.n_boundary_groups; ++i) {
    in >> data.boundary_group_n_faces[i];

    assert(data.boundary_group_n_faces[i] > 0 &&
           "Boundary groups must have at least one face");

    in >> dummy;
    data.boundary_group_n_nodes[i] = dummy;

    assert((dummy == 2 || dummy == 3) &&
           "Boundary groups must have 2 or 3 nodes per face");

    std::getline(in >> std::ws, data.boundary_group_names[i]);

    for (unsigned int j = 0; j < data.boundary_group_n_faces[i]; ++j) {
      in >> dummy;
      data.boundary_node_1.emplace_back(dummy - 1);

      in >> dummy;
      data.boundary_node_2.emplace_back(dummy - 1);

      if (data.boundary_group_n_nodes[i] == 3) {
        in >> dummy;
        data.boundary_node_3.emplace_back(dummy - 1);
      } else {
        data.boundary_node_3.emplace_back(
            std::numeric_limits<unsigned int>::max());
      }
    }
  }

  // Element groups
  unsigned int elem_counter = 0;

  while (elem_counter < data.n_elements) {
    int n_element_in_group;
    int element_group_order;
    std::string element_basis;

    in >> n_element_in_group >> element_group_order;
    std::getline(in >> std::ws, element_basis);

    assert(element_basis == "TriLagrange" &&
           "Only TriLagrange elements are supported");
    assert((element_group_order == 1 || element_group_order == 2) &&
           "Only first and second order elements are supported");

    for (int i = 0; i < n_element_in_group; ++i) {
      assert(elem_counter < data.n_elements &&
             "Read more elements than header specifies");

      data.element_order[elem_counter] = element_group_order;

      if (element_group_order == 1) {
        in >> dummy;
        data.node_1[elem_counter] = dummy - 1;

        in >> dummy;
        data.node_2[elem_counter] = dummy - 1;

        in >> dummy;
        data.node_3[elem_counter] = dummy - 1;

      } else { //read in the curved nodes
        in >> dummy;
        data.node_1[elem_counter] = dummy - 1;

        in >> dummy;
        data.node_4[elem_counter] = dummy - 1;

        in >> dummy;
        data.node_2[elem_counter] = dummy - 1;

        in >> dummy;
        data.node_6[elem_counter] = dummy - 1;

        in >> dummy;
        data.node_5[elem_counter] = dummy - 1;

        in >> dummy;
        data.node_3[elem_counter] = dummy - 1;
      }

      ++elem_counter;
    }
  }
  assert(elem_counter == data.n_elements &&
         "Did not read expected number of elements");

  // Periodic groups
  in >> dummy;
  data.set_n_periodic_groups(dummy);

  std::string periodic_group;
  std::getline(in >> std::ws, periodic_group);

  assert(periodic_group == "PeriodicGroup" &&
         "Invalid .gri file. Periodic groups must contain PeriodicGroup "
         "after the number of periodic groups.");

  for (unsigned int i = 0; i < data.n_periodic_groups; ++i) {
    in >> data.periodic_group_n_nodes[i];

    std::string periodic_group_type;
    std::getline(in >> std::ws, periodic_group_type);

    assert(periodic_group_type == "Translational" &&
           "Invalid periodicity type");

    for (unsigned int j = 0; j < data.periodic_group_n_nodes[i]; ++j) {
      in >> dummy;
      data.periodic_node_1.emplace_back(dummy - 1);

      in >> dummy;
      data.periodic_node_2.emplace_back(dummy - 1);
    }
  }

  // -----------------------------
  // DEBUG: element order summary
  // -----------------------------
  unsigned int n_p1 = 0, n_p2 = 0;

  for (unsigned int i = 0; i < data.n_elements; ++i) {
    if (data.element_order[i] == 1)
      ++n_p1;
    else if (data.element_order[i] == 2)
      ++n_p2;
  }

  std::cout << "\nMesh Read Summary:\n";
  std::cout << "  Nodes: " << data.n_nodes << "\n";
  std::cout << "  Elements: " << data.n_elements << "\n";
  std::cout << "    P1 elements: " << n_p1 << "\n";
  std::cout << "    P2 elements: " << n_p2 << "\n";
  std::cout << "  Boundary groups: " << data.n_boundary_groups << "\n";
  std::cout << "  Periodic groups: " << data.n_periodic_groups << "\n";

  unsigned int n_linear_faces = 0;
  unsigned int n_curved_faces = 0;

  for (unsigned int g = 0; g < data.n_boundary_groups; ++g) {
    for (unsigned int f = 0; f < data.boundary_group_n_faces[g]; ++f) {
      if (data.boundary_group_n_nodes[g] == 2)
        ++n_linear_faces;
      if (data.boundary_group_n_nodes[g] == 3)
        ++n_curved_faces;
    }
  }

  std::cout << "  Boundary faces:\n";
  std::cout << "    Linear (2-node): " << n_linear_faces << "\n";
  std::cout << "    Curved (3-node): " << n_curved_faces << "\n";
}