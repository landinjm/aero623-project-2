#pragma once

#include <cmath>
#include <config.hpp>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <triangulation.hpp>
#include <vector>

/**
 * Simple struct to store mesh data in.
 *
 * NOTE: This will allocate memory as if the mesh were 3D even if 2D.
 */
struct MeshData
{
  unsigned int n_nodes;
  unsigned int n_elements;
  unsigned int n_boundary_groups;
  unsigned int n_periodic_groups;
  unsigned int q_order;

  // 2D node coordinates
  std::vector<double> x;
  std::vector<double> y;
  // 3D node coordinates (z unused in 2D)
  std::vector<double> z;

  // Triangle connectivity (2D)
  std::vector<unsigned int> node_1;
  std::vector<unsigned int> node_2;
  std::vector<unsigned int> node_3;
  // Tetrahedron connectivity (3D, 4th node)
  std::vector<unsigned int> node_4;
  // Higher Order Connectivity (3D q=2)
  std::vector<unsigned int> node_5;
  std::vector<unsigned int> node_6; // up to here for 2D q=2
  std::vector<unsigned int> node_7;
  std::vector<unsigned int> node_8;
  std::vector<unsigned int> node_9;
  std::vector<unsigned int> node_10;

  std::vector<unsigned int> boundary_group_n_faces;
  std::vector<unsigned int> boundary_group_n_nodes;
  std::vector<std::string> boundary_group_names;

  // 2D boundary edges
  std::vector<unsigned int> boundary_node_1;
  std::vector<unsigned int> boundary_node_2;
  // 3D boundary triangles (3rd node)
  std::vector<unsigned int> boundary_node_3; // up to here for 2D q=2
  // Higher Order Connectivity (3D q=2)
  std::vector<unsigned int> boundary_node_4;
  std::vector<unsigned int> boundary_node_5;
  std::vector<unsigned int> boundary_node_6;

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
    z.resize(n, 0.0);
  }
  void set_n_elements(std::size_t n)
  {
    n_elements = n;
    node_1.resize(n);
    node_2.resize(n);
    node_3.resize(n);
    node_4.resize(n, 0);
    node_5.resize(n, 0);
    node_6.resize(n, 0);
    node_7.resize(n, 0);
    node_8.resize(n, 0);
    node_9.resize(n, 0);
    node_10.resize(n, 0);
  }

  void set_n_boundary_groups(std::size_t n)
  {
    n_boundary_groups = n;
    boundary_group_n_faces.resize(n);
    boundary_group_names.resize(n);
    boundary_group_n_nodes.resize(n);
  }

  void set_n_periodic_groups(std::size_t n)
  {
    n_periodic_groups = n;
    periodic_group_n_nodes.resize(n);
  }
};

template<unsigned int dim, unsigned int q>
class GriReader
{
public:
  static_assert(dim == 2 || dim == 3, "Only 2D and 3D meshes are supported");
  static_assert(q == 1 || q == 2, "Only q1 and q2 meshes are supported");

  GriReader() = default;

  void read_gri(const std::string& filename);

  MeshData data() const { return data_; }

  void check_counter_clockwise() const
  {
    // Only meaningful in 2D
    if constexpr (dim == 2) {
      for (unsigned int i = 0; i < data_.n_elements; ++i) {
        const double ax = data_.x[data_.node_1[i]],
                     ay = data_.y[data_.node_1[i]];
        const double bx = data_.x[data_.node_2[i]],
                     by = data_.y[data_.node_2[i]];
        const double cx = data_.x[data_.node_3[i]],
                     cy = data_.y[data_.node_3[i]];

        // z-component of AB x AC
        const double cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
        ASSERT(cross > 0.0,
               "Element " + std::to_string(i) + " is not counter-clockwise");
      }
    }
  }

  void transfer_to_triangulation(Triangulation<dim, q>& tria) const
  {
    if constexpr (dim == 2) {
      transfer_to_triangulation_2d(tria);
    }
    if constexpr (dim == 3) {
      if constexpr (q == 1) {
        transfer_to_triangulation_3d(tria);
      }
      if constexpr (q == 2) {
        transfer_to_triangulation_3d_q2(tria);
      }
    }
  }

private:
  MeshData data_;

  void transfer_to_triangulation_2d(Triangulation<2, 1>& tria) const
  {
    // Check that things are counter clockwise first
    check_counter_clockwise();

    const unsigned int n_cells = data_.n_elements;

    using EdgeKey = std::pair<unsigned int, unsigned int>;

    struct RawFace
    {
      uint v0, v1;
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

    std::map<unsigned int, unsigned int> periodic_nodes;
    for (unsigned int i = 0; i < data_.periodic_node_1.size(); ++i) {
      periodic_nodes[data_.periodic_node_1[i]] = data_.periodic_node_2[i];
      periodic_nodes[data_.periodic_node_2[i]] = data_.periodic_node_1[i];
    }

    for (auto& rf : raw_faces) {
      if (rf.cells[1] != CellIndexType(-1))
        continue;

      const auto a = rf.v0, b = rf.v1;

      if (periodic_nodes.contains(a) && periodic_nodes.contains(b)) {
        const auto c = periodic_nodes[a];
        const auto d = periodic_nodes[b];
        const EdgeKey key = { std::min(c, d), std::max(c, d) };
        ASSERT(edge_to_face.contains(key), "IDK");
        const FaceIndexType face_index = edge_to_face[key];

        rf.neighbor = face_index;
        rf.cells[1] = raw_faces[face_index].cells[0];
        rf.is_periodic = true;
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
          if (!raw_faces[it->second].is_periodic) {
            raw_faces[it->second].is_boundary = true;
            raw_faces[it->second].boundary_id = bid;
          }
        }
        offset += data_.boundary_group_n_faces[g];
      }
    }

    const unsigned int n_faces = static_cast<unsigned int>(raw_faces.size());
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
    }

    for (unsigned int f = 0; f < n_faces; ++f) {
      tria.face_vertices(f, 0) = raw_faces[f].v0;
      tria.face_vertices(f, 1) = raw_faces[f].v1;
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
          0.5 * (data_.x[raw_faces[f].v0] + data_.x[raw_faces[f].v1]);
        const double my0 =
          0.5 * (data_.y[raw_faces[f].v0] + data_.y[raw_faces[f].v1]);
        const double mx1 = 0.5 * (data_.x[raw_faces[neighbor].v0] +
                                  data_.x[raw_faces[neighbor].v1]);
        const double my1 = 0.5 * (data_.y[raw_faces[neighbor].v0] +
                                  data_.y[raw_faces[neighbor].v1]);
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
  }

  void transfer_to_triangulation_3d(Triangulation<3, 1>& tria) const
  {
    // Grab the number of cells
    const unsigned int n_cells = data_.n_elements;

    // Build a face map
    using TriKey = std::tuple<unsigned int, unsigned int, unsigned int>;

    struct RawFace
    {
      // Global vertex indices
      uint v0, v1, v2;

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

    std::map<TriKey, FaceIndexType> tri_to_face;
    std::vector<RawFace> raw_faces;

    // Each tet has 4 local faces, each opposite the vertex with the same index.
    // Local face k uses the 3 vertices that are NOT vertex k.
    // face 0: v1,v2,v3   face 1: v0,v2,v3   face 2: v0,v1,v3   face 3: v0,v1,v2
    constexpr unsigned int face_verts[4][3] = {
      { 1, 2, 3 }, { 0, 2, 3 }, { 0, 1, 3 }, { 0, 1, 2 }
    };

    std::vector<std::array<FaceIndexType, 4>> cell_face_ids(n_cells);

    for (unsigned int c = 0; c < n_cells; ++c) {
      const unsigned int vi[4] = {
        data_.node_1[c], data_.node_2[c], data_.node_3[c], data_.node_4[c]
      };

      for (unsigned int lf = 0; lf < 4; ++lf) {
        unsigned int a = vi[face_verts[lf][0]];
        unsigned int b = vi[face_verts[lf][1]];
        unsigned int cc = vi[face_verts[lf][2]];

        // Sort for canonical key
        unsigned int s[3] = { a, b, cc };
        if (s[0] > s[1])
          std::swap(s[0], s[1]);
        if (s[1] > s[2])
          std::swap(s[1], s[2]);
        if (s[0] > s[1])
          std::swap(s[0], s[1]);
        const TriKey key = { s[0], s[1], s[2] };

        auto it = tri_to_face.find(key);
        if (it == tri_to_face.end()) {
          const FaceIndexType face_index =
            static_cast<FaceIndexType>(raw_faces.size());
          cell_face_ids[c][lf] = face_index;
          tri_to_face[key] = face_index;
          RawFace rf;
          rf.v0 = a;
          rf.v1 = b;
          rf.v2 = cc;
          rf.cells[0] = static_cast<CellIndexType>(c);
          raw_faces.push_back(rf);
        } else {
          const FaceIndexType face_index = it->second;
          cell_face_ids[c][lf] = face_index;
          raw_faces[face_index].cells[1] = static_cast<CellIndexType>(c);
        }
      }
    }

    // Handle periodic nodes (same logic as 2D)
    std::map<unsigned int, unsigned int> periodic_nodes;
    for (unsigned int i = 0; i < data_.periodic_node_1.size(); ++i) {
      periodic_nodes[data_.periodic_node_1[i]] = data_.periodic_node_2[i];
      periodic_nodes[data_.periodic_node_2[i]] = data_.periodic_node_1[i];
    }

    for (auto& rf : raw_faces) {
      if (rf.cells[1] != CellIndexType(-1))
        continue;

      const auto a = rf.v0, b = rf.v1, cc = rf.v2;

      if (periodic_nodes.contains(a) && periodic_nodes.contains(b) &&
          periodic_nodes.contains(cc)) {
        unsigned int pa = periodic_nodes[a];
        unsigned int pb = periodic_nodes[b];
        unsigned int pc = periodic_nodes[cc];
        unsigned int s[3] = { pa, pb, pc };
        if (s[0] > s[1])
          std::swap(s[0], s[1]);
        if (s[1] > s[2])
          std::swap(s[1], s[2]);
        if (s[0] > s[1])
          std::swap(s[0], s[1]);
        const TriKey key = { s[0], s[1], s[2] };
        auto it = tri_to_face.find(key);
        if (it != tri_to_face.end()) {
          const FaceIndexType face_index = it->second;
          rf.neighbor = face_index;
          rf.cells[1] = raw_faces[face_index].cells[0];
          rf.is_periodic = true;
        }
      }
    }

    // Mark boundary faces using the 3-node boundary triangles
    {
      unsigned int offset = 0;
      for (unsigned int g = 0; g < data_.n_boundary_groups; ++g) {
        const BoundaryIdType bid = static_cast<BoundaryIdType>(g + 1);
        for (unsigned int i = 0; i < data_.boundary_group_n_faces[g]; ++i) {
          const unsigned int a = data_.boundary_node_1[offset + i];
          const unsigned int b = data_.boundary_node_2[offset + i];
          const unsigned int cc = data_.boundary_node_3[offset + i];
          unsigned int s[3] = { a, b, cc };
          if (s[0] > s[1])
            std::swap(s[0], s[1]);
          if (s[1] > s[2])
            std::swap(s[1], s[2]);
          if (s[0] > s[1])
            std::swap(s[0], s[1]);
          const TriKey key = { s[0], s[1], s[2] };
          auto it = tri_to_face.find(key);
          ASSERT(it != tri_to_face.end(),
                 "Boundary triangle not found in cell connectivity");
          if (!raw_faces[it->second].is_periodic) {
            raw_faces[it->second].is_boundary = true;
            raw_faces[it->second].boundary_id = bid;
          }
        }
        offset += data_.boundary_group_n_faces[g];
      }
    }

    const unsigned int n_faces = static_cast<unsigned int>(raw_faces.size());
    const unsigned int n_vertices = data_.n_nodes;

    tria.internal_reinit(n_cells, n_faces, n_vertices);

    for (unsigned int v = 0; v < data_.n_nodes; ++v) {
      tria.vertices(v, 0) = data_.x[v];
      tria.vertices(v, 1) = data_.y[v];
      tria.vertices(v, 2) = data_.z[v];
    }

    for (unsigned int c = 0; c < data_.n_elements; ++c) {
      tria.cell_vertices(c, 0) = data_.node_1[c];
      tria.cell_vertices(c, 1) = data_.node_2[c];
      tria.cell_vertices(c, 2) = data_.node_3[c];
      tria.cell_vertices(c, 3) = data_.node_4[c];
      tria.cell_faces(c, 0) = cell_face_ids[c][0];
      tria.cell_faces(c, 1) = cell_face_ids[c][1];
      tria.cell_faces(c, 2) = cell_face_ids[c][2];
      tria.cell_faces(c, 3) = cell_face_ids[c][3];
    }

    for (unsigned int f = 0; f < n_faces; ++f) {
      tria.face_vertices(f, 0) = raw_faces[f].v0;
      tria.face_vertices(f, 1) = raw_faces[f].v1;
      tria.face_vertices(f, 2) = raw_faces[f].v2;
      tria.face_cells(f, 0) = raw_faces[f].cells[0];
      tria.face_cells(f, 1) = raw_faces[f].cells[1];
      tria.boundary_ids(f) = raw_faces[f].boundary_id;

      tria.periodic_face_neighbor(f) =
        raw_faces[f].is_periodic
          ? static_cast<FaceIndexType>(raw_faces[f].neighbor)
          : FaceIndexType(-1);

      if (raw_faces[f].is_periodic) {
        const unsigned int neighbor = raw_faces[f].neighbor;
        // Offset = centroid of neighbor face minus centroid of this face
        for (unsigned int d = 0; d < 3; ++d) {
          const double* coord = (d == 0   ? data_.x.data()
                                 : d == 1 ? data_.y.data()
                                          : data_.z.data());
          const double mc0 = (coord[raw_faces[f].v0] + coord[raw_faces[f].v1] +
                              coord[raw_faces[f].v2]) /
                             3.0;
          const double mc1 =
            (coord[raw_faces[neighbor].v0] + coord[raw_faces[neighbor].v1] +
             coord[raw_faces[neighbor].v2]) /
            3.0;
          tria.periodic_face_offset(f, d) = mc1 - mc0;
        }
      } else {
        for (unsigned int d = 0; d < 3; ++d)
          tria.periodic_face_offset(f, d) = 0.0;
      }

      uint32_t flags = 0;
      if (raw_faces[f].is_boundary)
        flags |= FaceFlags::Boundary;
      if (raw_faces[f].is_periodic)
        flags |= FaceFlags::Periodic;
      tria.face_flags(f) = flags;
    }
  }

  void transfer_to_triangulation_3d_q2(Triangulation<3, 2>& tria) const
  {
    // Grab the number of cells
    const unsigned int n_cells = data_.n_elements;

    // A face in 3D is a triangle, identified by a sorted triple of vertex
    // indices
    using TriKey = std::tuple<unsigned int, unsigned int, unsigned int>;

    struct RawFace
    {
      uint v0, v1, v2, v3, v4, v5;

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

    std::map<TriKey, FaceIndexType> tri_to_face;
    std::vector<RawFace> raw_faces;

    // Each tet has 4 local faces, each opposite the vertex with the same index.
    // Local face k uses the 3 vertices that are NOT vertex k.
    // face 0: v1,v5,v2,v9,v3,v8    face 1: v0,v6,v2,v9,v3,v7 <- all go vert,
    // mp, vert, face 2: v0,v4,v1,v8,v3,v7    face 3: v0,v4,v1,v5,v2,v6 mp,
    // vert, mp
    constexpr unsigned int face_verts[4][6] = { { 1, 5, 2, 9, 3, 8 },
                                                { 0, 6, 2, 9, 3, 7 },
                                                { 0, 4, 1, 8, 3, 7 },
                                                { 0, 4, 1, 5, 2, 6 } };

    std::vector<std::array<FaceIndexType, 4>> cell_face_ids(n_cells);

    for (unsigned int c = 0; c < n_cells; ++c) {
      const unsigned int vi[10] = { data_.node_1[c], data_.node_2[c],
                                    data_.node_3[c], data_.node_4[c],
                                    data_.node_5[c], data_.node_6[c],
                                    data_.node_7[c], data_.node_8[c],
                                    data_.node_9[c], data_.node_10[c] };

      for (unsigned int lf = 0; lf < 4; ++lf) {
        unsigned int a = vi[face_verts[lf][0]];  // vert 1
        unsigned int b = vi[face_verts[lf][2]];  // vert 2
        unsigned int cc = vi[face_verts[lf][4]]; // vert 3
        unsigned int dd = vi[face_verts[lf][1]]; // mp 1
        unsigned int ee = vi[face_verts[lf][3]]; // mp 2
        unsigned int ff = vi[face_verts[lf][5]]; // mp 3

        // Sort for canonical key (shouldn't need to change for q = 2)
        unsigned int s[3] = { a, b, cc };
        if (s[0] > s[1])
          std::swap(s[0], s[1]);
        if (s[1] > s[2])
          std::swap(s[1], s[2]);
        if (s[0] > s[1])
          std::swap(s[0], s[1]);
        const TriKey key = { s[0], s[1], s[2] };

        auto it = tri_to_face.find(key);
        if (it == tri_to_face.end()) {
          const FaceIndexType face_index =
            static_cast<FaceIndexType>(raw_faces.size());
          cell_face_ids[c][lf] = face_index;
          tri_to_face[key] = face_index;
          RawFace rf;
          rf.v0 = a;
          rf.v1 = b;
          rf.v2 = cc;
          rf.v3 = dd;
          rf.v4 = ee;
          rf.v5 = ff;
          rf.cells[0] = static_cast<CellIndexType>(c);
          raw_faces.push_back(rf);
        } else {
          const FaceIndexType face_index = it->second;
          cell_face_ids[c][lf] = face_index;
          raw_faces[face_index].cells[1] = static_cast<CellIndexType>(c);
        }
      }
    }

    // Handle periodic nodes (same logic as 2D) (same logic for q = 1)
    std::map<unsigned int, unsigned int> periodic_nodes;
    for (unsigned int i = 0; i < data_.periodic_node_1.size(); ++i) {
      periodic_nodes[data_.periodic_node_1[i]] = data_.periodic_node_2[i];
      periodic_nodes[data_.periodic_node_2[i]] = data_.periodic_node_1[i];
    }

    for (auto& rf : raw_faces) {
      if (rf.cells[1] != CellIndexType(-1))
        continue;

      const auto a = rf.v0, b = rf.v1, cc = rf.v2;

      if (periodic_nodes.contains(a) && periodic_nodes.contains(b) &&
          periodic_nodes.contains(cc)) {
        unsigned int pa = periodic_nodes[a];
        unsigned int pb = periodic_nodes[b];
        unsigned int pc = periodic_nodes[cc];
        unsigned int s[3] = { pa, pb, pc };
        if (s[0] > s[1])
          std::swap(s[0], s[1]);
        if (s[1] > s[2])
          std::swap(s[1], s[2]);
        if (s[0] > s[1])
          std::swap(s[0], s[1]);
        const TriKey key = { s[0], s[1], s[2] };
        auto it = tri_to_face.find(key);
        if (it != tri_to_face.end()) {
          const FaceIndexType face_index = it->second;
          rf.neighbor = face_index;
          rf.cells[1] = raw_faces[face_index].cells[0];
          rf.is_periodic = true;
        }
      }
    }

    // Mark boundary faces using the 3-node boundary triangles (same logic for q
    // = 1)
    {
      unsigned int offset = 0;
      for (unsigned int g = 0; g < data_.n_boundary_groups; ++g) {
        const BoundaryIdType bid = static_cast<BoundaryIdType>(g + 1);
        for (unsigned int i = 0; i < data_.boundary_group_n_faces[g]; ++i) {
          const unsigned int a = data_.boundary_node_1[offset + i];
          const unsigned int b = data_.boundary_node_2[offset + i];
          const unsigned int cc = data_.boundary_node_3[offset + i];
          unsigned int s[3] = { a, b, cc };
          if (s[0] > s[1])
            std::swap(s[0], s[1]);
          if (s[1] > s[2])
            std::swap(s[1], s[2]);
          if (s[0] > s[1])
            std::swap(s[0], s[1]);
          const TriKey key = { s[0], s[1], s[2] };
          auto it = tri_to_face.find(key);
          ASSERT(it != tri_to_face.end(),
                 "Boundary triangle not found in cell connectivity");
          if (!raw_faces[it->second].is_periodic) {
            raw_faces[it->second].is_boundary = true;
            raw_faces[it->second].boundary_id = bid;
          }
        }
        offset += data_.boundary_group_n_faces[g];
      }
    }

    const unsigned int n_faces = static_cast<unsigned int>(raw_faces.size());
    const unsigned int n_vertices = data_.n_nodes;

    tria.internal_reinit(n_cells, n_faces, n_vertices);

    for (unsigned int v = 0; v < data_.n_nodes; ++v) {
      tria.vertices(v, 0) = data_.x[v];
      tria.vertices(v, 1) = data_.y[v];
      tria.vertices(v, 2) = data_.z[v];
    }

    for (unsigned int c = 0; c < data_.n_elements; ++c) {
      tria.cell_vertices(c, 0) = data_.node_1[c];
      tria.cell_vertices(c, 1) = data_.node_2[c];
      tria.cell_vertices(c, 2) = data_.node_3[c];
      tria.cell_vertices(c, 3) = data_.node_4[c];
      tria.cell_vertices(c, 4) = data_.node_5[c];
      tria.cell_vertices(c, 5) = data_.node_6[c];
      tria.cell_vertices(c, 6) = data_.node_7[c];
      tria.cell_vertices(c, 7) = data_.node_8[c];
      tria.cell_vertices(c, 8) = data_.node_9[c];
      tria.cell_vertices(c, 9) = data_.node_10[c];
      tria.cell_faces(c, 0) = cell_face_ids[c][0];
      tria.cell_faces(c, 1) = cell_face_ids[c][1];
      tria.cell_faces(c, 2) = cell_face_ids[c][2];
      tria.cell_faces(c, 3) = cell_face_ids[c][3];
    }

    for (unsigned int f = 0; f < n_faces; ++f) {
      tria.face_vertices(f, 0) = raw_faces[f].v0;
      tria.face_vertices(f, 1) = raw_faces[f].v1;
      tria.face_vertices(f, 2) = raw_faces[f].v2;
      tria.face_vertices(f, 3) = raw_faces[f].v3;
      tria.face_vertices(f, 4) = raw_faces[f].v4;
      tria.face_vertices(f, 5) = raw_faces[f].v5;
      tria.face_cells(f, 0) = raw_faces[f].cells[0];
      tria.face_cells(f, 1) = raw_faces[f].cells[1];
      tria.boundary_ids(f) = raw_faces[f].boundary_id;

      tria.periodic_face_neighbor(f) =
        raw_faces[f].is_periodic
          ? static_cast<FaceIndexType>(raw_faces[f].neighbor)
          : FaceIndexType(-1);

      if (raw_faces[f].is_periodic) {
        const unsigned int neighbor = raw_faces[f].neighbor;
        // Offset = centroid of neighbor face minus centroid of this face
        for (unsigned int d = 0; d < 3; ++d) {
          const double* coord = (d == 0   ? data_.x.data()
                                 : d == 1 ? data_.y.data()
                                          : data_.z.data());
          const double mc0 = (coord[raw_faces[f].v0] + coord[raw_faces[f].v1] +
                              coord[raw_faces[f].v2] + coord[raw_faces[f].v3] +
                              coord[raw_faces[f].v4] + coord[raw_faces[f].v5]) /
                             6.0;
          const double mc1 =
            (coord[raw_faces[neighbor].v0] + coord[raw_faces[neighbor].v1] +
             coord[raw_faces[neighbor].v2] + coord[raw_faces[neighbor].v3] +
             coord[raw_faces[neighbor].v4] + coord[raw_faces[neighbor].v5]) /
            6.0;
          tria.periodic_face_offset(f, d) = mc1 - mc0;
        }
      } else {
        for (unsigned int d = 0; d < 3; ++d)
          tria.periodic_face_offset(f, d) = 0.0;
      }

      uint32_t flags = 0;
      if (raw_faces[f].is_boundary)
        flags |= FaceFlags::Boundary;
      if (raw_faces[f].is_periodic)
        flags |= FaceFlags::Periodic;
      tria.face_flags(f) = flags;
    }
  }
};

// ---------------------------------------------------------------------------
// read_gri implementation
// ---------------------------------------------------------------------------
template<unsigned int dim, unsigned int q>
void
GriReader<dim, q>::read_gri(const std::string& filename)
{
  std::ifstream in(filename);
  ASSERT(in, "Unable to open file " + filename);

  int dummy;
  in >> dummy;
  data_.set_n_nodes(dummy);
  in >> dummy;
  data_.set_n_elements(dummy);
  in >> dummy;

  ASSERT(data_.n_nodes > 0, "Number of nodes must be greater than zero.");
  ASSERT(data_.n_elements > 0, "Number of elements must be greater than zero.");
  ASSERT(dummy == (int)dim, "File has different dimension than mesh.");

  // Node coordinates
  for (unsigned int i = 0; i < data_.n_nodes; ++i) {
    in >> data_.x[i] >> data_.y[i];
    if constexpr (dim == 3)
      in >> data_.z[i];
  }

  // Boundary groups
  in >> dummy;
  data_.set_n_boundary_groups(dummy);

  std::cout << "n boundaries " << data_.n_boundary_groups << std::endl;

  for (unsigned int i = 0; i < data_.n_boundary_groups; ++i) {
    in >> data_.boundary_group_n_faces[i];
    //std::cout << "Number of boundary group faces: " << std::to_string(data_.boundary_group_n_faces[i]) << std::endl;
    ASSERT(data_.boundary_group_n_faces[i] > 0,
           "Boundary groups must have at least one face");
    in >> data_.boundary_group_n_nodes[i]; // nodes per boundary face
                                           // (dim for 2D edges, dim-1+1=dim for
                                           // 3D tris)

    std::getline(in >> std::ws, data_.boundary_group_names[i]);

    // NOTE: The files assume indexing from 1 so adjust for that
    for (unsigned int j = 0; j < data_.boundary_group_n_faces[i]; ++j) {
      in >> dummy;
      data_.boundary_node_1.emplace_back(dummy - 1);
      in >> dummy;
      data_.boundary_node_2.emplace_back(dummy - 1);
      if (data_.boundary_group_n_nodes[i] >= 3) {
        in >> dummy;
        data_.boundary_node_3.emplace_back(dummy - 1);
        if (data_.boundary_group_n_nodes[i] >= 6) {
          in >> dummy;
          data_.boundary_node_4.emplace_back(dummy - 1);
          in >> dummy;
          data_.boundary_node_5.emplace_back(dummy - 1);
          in >> dummy;
          data_.boundary_node_6.emplace_back(dummy - 1);
        }
      }
    }
  }

  // Elements (one group assumed, linear simplices)
  int n_element_in_group;
  std::string element_basis;
  in >> n_element_in_group >> data_.q_order;
  std::getline(in >> std::ws, element_basis);

  ASSERT(n_element_in_group == (int)data_.n_elements,
         "Multiple element groups are unsupported");
  ASSERT(data_.q_order == 1 || data_.q_order == 2,
         "Only first & 2nd order elements are supported");

  // NOTE: The files assume indexing from 1 so adjust for that
  for (unsigned int i = 0; i < data_.n_elements; ++i) {
    in >> dummy; data_.node_1[i] = dummy - 1;
    in >> dummy; data_.node_2[i] = dummy - 1;
    in >> dummy; data_.node_3[i] = dummy - 1;

    if constexpr (dim == 3) {
      in >> dummy; data_.node_4[i] = dummy - 1;
      if (data_.q_order == 2) {
        in >> dummy; data_.node_5[i] = dummy - 1;
        in >> dummy; data_.node_6[i] = dummy - 1;
        in >> dummy; data_.node_7[i] = dummy - 1;
        in >> dummy; data_.node_8[i] = dummy - 1;
        in >> dummy; data_.node_9[i] = dummy - 1;
        in >> dummy; data_.node_10[i] = dummy - 1;
      }
    } else if constexpr (dim == 2) {  // <-- separate branch, not nested
      if (data_.q_order == 2) {
        in >> dummy; data_.node_4[i] = dummy - 1;
        in >> dummy; data_.node_5[i] = dummy - 1;
        in >> dummy; data_.node_6[i] = dummy - 1;
      }
    }
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

    // NOTE: The files assume indexing from 1 so adjust for that
    for (unsigned int j = 0; j < data_.periodic_group_n_nodes[i]; ++j) {
      in >> dummy;
      data_.periodic_node_1.emplace_back(dummy - 1);
      in >> dummy;
      data_.periodic_node_2.emplace_back(dummy - 1);
    }
  }
}

// ---------------------------------------------------------------------------
// Free function: read_gri(tria, filename)
// This is the signature used by main.cpp.
// ---------------------------------------------------------------------------
template<unsigned int dim, unsigned int q>
void
read_gri(Triangulation<dim, q>& tria, const std::string& filename)
{
  GriReader<dim, q> reader;
  reader.read_gri(filename);
  reader.transfer_to_triangulation(tria);
}
