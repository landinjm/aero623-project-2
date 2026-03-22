#pragma once

#include <Kokkos_Core.hpp>
#include <cassert>
#include <cmath>
#include <config.hpp>
#include <cstdint>
#include <stdexcept>
#include <tensor.hpp>

template <unsigned int dim> class GriReader;

template <unsigned int dim> class Triangulation;

using LocalIndexType = uint32_t;
using CellIndexType = uint32_t;
using FaceIndexType = int32_t;
using VertexIndexType = uint32_t;
using BoundaryIdType = uint32_t;
using NeighborIndexType = int32_t;

constexpr NeighborIndexType NO_NEIGHBOR = -1;

template <unsigned int dim> class FaceAccessor;

/**
 * @brief Bit flags for face types.
 */
namespace FaceFlags {
constexpr BoundaryIdType Boundary = 1 << 0;
constexpr BoundaryIdType Interior = 1 << 1;
constexpr BoundaryIdType Periodic = 1 << 2;
} // namespace FaceFlags

/**
 * @brief Mesh topology constants for simplex elements.
 */
template <int dim> struct SimplexTopology;

template <> struct SimplexTopology<1> {
  static constexpr LocalIndexType verts_per_cell = 2;
  static constexpr LocalIndexType faces_per_cell = 2;
  static constexpr LocalIndexType verts_per_face = 1;
  static constexpr LocalIndexType face_verts[2][1] = {{0}, {1}};
};

template <> struct SimplexTopology<2> {
  static constexpr LocalIndexType verts_per_cell = 3;
  static constexpr LocalIndexType faces_per_cell = 3;
  static constexpr LocalIndexType verts_per_face = 2;
  static constexpr LocalIndexType face_verts[3][2] = {{1, 2}, {0, 2}, {0, 1}};
};

template <> struct SimplexTopology<3> {
  static constexpr LocalIndexType verts_per_cell = 4;
  static constexpr LocalIndexType faces_per_cell = 4;
  static constexpr LocalIndexType verts_per_face = 3;
  static constexpr LocalIndexType face_verts[4][3] = {
      {1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}};
};

/**
 * @brief Cell accessor for cell loops.
 */
template <unsigned int dim> struct CellAccessor {
public:
  using Topo = SimplexTopology<dim>;

  CellIndexType index;
  const Triangulation<dim> *tria;

  static constexpr LocalIndexType n_vertices() { return Topo::verts_per_cell; }
  static constexpr LocalIndexType n_faces() { return Topo::faces_per_cell; }

  /**
   * @brief Grab the global vertex index from the local one.
   */
  VertexIndexType vertex_index(LocalIndexType local_v) const {
    return tria->cell_vertices(index, local_v);
  }

  /**
   * @brief Grab the vertex coordinate from the local index.
   */
  Tensor<1, dim, double> vertex(LocalIndexType local_v) const {
    Tensor<1, dim, double> p;
    const auto global_v = vertex_index(local_v);
    for (unsigned int d = 0; d < dim; ++d) {
      p(d) = tria->vertices(global_v, d);
    }
    return p;
  }

  //--------- NEW STUFF ----------------------------
  /**
   * @brief Grab the geometry order for this cell.
   */
  uint8_t geometry_order() const { return tria->cell_geometry_order(index); }

  /**
   * @brief Grab the global geometry node index from the local one.
   *
   * Local geometry node numbering is:
   *   0,1,2 = corner nodes
   *   3,4,5 = midside nodes for q=2 geometry
   */
  VertexIndexType geometry_node_index(LocalIndexType local_g) const {
    ASSERT(local_g < 6, "Invalid local geometry node");
    return tria->cell_geometry_nodes(index, local_g);
  }

  /**
   * @brief Grab the geometry node coordinate from the local geometry index.
   */
  Tensor<1, dim, double> geometry_node(LocalIndexType local_g) const {
    Tensor<1, dim, double> p;
    const auto global_v = geometry_node_index(local_g);
    for (unsigned int d = 0; d < dim; ++d) {
      p(d) = tria->vertices(global_v, d);
    }
    return p;
  }
  // ------------------------------------------------

  /**
   * @brief Grab the global face index from the local one.
   */
  FaceIndexType face_index(LocalIndexType local_f) const {
    return tria->cell_faces(index, local_f);
  }

  /**
   * @brief Grab the face accessor from the local index.
   */
  FaceAccessor<dim> face(LocalIndexType local_f) const {
    return tria->get_face(face_index(local_f));
  }

  bool face_at_boundary(LocalIndexType local_f) const {
    return tria->face_flags(face_index(local_f)) & FaceFlags::Boundary;
  }

  bool face_is_periodic(LocalIndexType local_f) const {
    return tria->face_flags(face_index(local_f)) & FaceFlags::Periodic;
  }

  BoundaryIdType face_boundary_id(LocalIndexType local_f) const {
    return tria->boundary_ids(face_index(local_f));
  }

  /**
   * @brief Index of the neighboring cell across some local face.
   *
   * For the local face, (f, 0) is the owner and (f, 1) is the neighbor if it
   * exists.
   */
  CellIndexType neighbor_index(LocalIndexType local_f) const {
    const auto global_f = face_index(local_f);

    if (face_is_periodic(local_f)) {
      const auto neighbor = tria->periodic_face_neighbor(global_f);
      ASSERT(neighbor != NO_NEIGHBOR, "Perioidc face has no partner");
      // If all is good, return the owner cell of the neighbor
      return tria->face_cells(neighbor, 0);
    }

    ASSERT(!face_at_boundary(local_f),
           "Cannot get the neighbor of a non-periodic boundary face");

    const auto c0 = tria->face_cells(global_f, 0);
    const auto c1 = tria->face_cells(global_f, 1);
    return (c0 == index) ? c1 : c0;
  }

  /**
   * @brief Grab the cell accessor of the neighbor.
   */
  CellAccessor<dim> neighbor(LocalIndexType local_f) const {
    return tria->get_cell(neighbor_index(local_f));
  }

  Tensor<1, dim, double> center() const {
    Tensor<1, dim, double> c;
    for (LocalIndexType v = 0; v < Topo::verts_per_cell; ++v) {
      const auto p = vertex(v);
      c += p;
    }
    return c / static_cast<double>(Topo::verts_per_cell);
  }

  double measure() const {
    if constexpr (dim == 2) {
      const auto v0 = vertex(0);
      const auto v1 = vertex(1);
      const auto v2 = vertex(2);
      const double a0 = v1[0] - v0[0], a1 = v1[1] - v0[1];
      const double b0 = v2[0] - v0[0], b1 = v2[1] - v0[1];
      return 0.5 * Kokkos::abs(a0 * b1 - a1 * b0);
    }
    return 0.0;
  }

  unsigned int n_geometry_nodes() const {
    return (geometry_order() == 2) ? 6 : 3;
  }

};

/**
 * @brief Cell accessor for face loops.
 */
template <unsigned int dim> struct FaceAccessor {
  using Topo = SimplexTopology<dim>;

  FaceIndexType index;
  const Triangulation<dim> *tria;

  static constexpr uint8_t n_vertices() { return Topo::verts_per_face; }

  /**
   * @brief Grab the global vertex index from the local one.
   */
  VertexIndexType vertex_index(LocalIndexType local_v) const {
    ASSERT(local_v < Topo::verts_per_face, "Invalid local vertex");
    return tria->face_vertices(index, local_v);
  }

  /**
   * @brief Grab the vertex coordinate from the local index.
   */
  Tensor<1, dim, double> vertex(LocalIndexType local_v) const {
    Tensor<1, dim, double> p;
    const auto global_v = vertex_index(local_v);
    for (unsigned int d = 0; d < dim; ++d) {
      p(d) = tria->vertices(global_v, d);
    }
    return p;
  }

  //--------- NEW STUFF ----------------------------
  /**
   * @brief Grab the geometry order for this face.
   */
  uint8_t geometry_order() const { return tria->face_geometry_order(index); }

  /**
   * @brief Grab the global geometry node index from the local one.
   *
   * Local face geometry node numbering is:
   *   0 = first corner node
   *   1 = midside node for q=2 face geometry
   *   2 = second corner node
   *
   * For q=1 faces, entry 1 can be an invalid sentinel.
   */
  VertexIndexType geometry_node_index(LocalIndexType local_g) const {
    ASSERT(local_g < 3, "Invalid local face geometry node");
    return tria->face_geometry_nodes(index, local_g);
  }

  /**
   * @brief Grab the face geometry node coordinate from the local geometry
   * index.
   */
  Tensor<1, dim, double> geometry_node(LocalIndexType local_g) const {
    Tensor<1, dim, double> p;
    const auto global_v = geometry_node_index(local_g);
    ASSERT(global_v != VertexIndexType(-1), "Invalid face geometry node");
    for (unsigned int d = 0; d < dim; ++d) {
      p(d) = tria->vertices(global_v, d);
    }
    return p;
  }

  unsigned int n_geometry_nodes() const {
    return (geometry_order() == 2) ? 6 : 3;
  }

  // ------------------------------------------------

  /**
   * @brief Grab the owner cell of this face
   */
  CellIndexType owner_index() const {
    const auto global_cell = tria->face_cells(index, 0);

    // If the face is periodic we must determine an owner so we don't double
    // count. For this we assume the owner is the minimum global cell index.
    if (is_periodic()) {
      const auto neighbor = tria->face_cells(index, 1);

      return std::min(global_cell, neighbor);
    }

    return global_cell;
  }

  /**
   * @brief Grab the neighbor cell of this face
   */
  CellIndexType neighbor_index() const {
    ASSERT(!at_boundary(), "Cannot get neighbor of a boundary face");
    return tria->face_cells(index, 1);
  }

  CellAccessor<dim> owner() const { return tria->get_cell(owner_index()); }

  CellAccessor<dim> neighbor() const {
    return tria->get_cell(neighbor_index());
  }

  bool at_boundary() const {
    return tria->face_flags(index) & FaceFlags::Boundary;
  }

  bool is_periodic() const {
    return tria->face_flags(index) & FaceFlags::Periodic;
  }

  BoundaryIdType boundary_id() const { return tria->boundary_ids(index); }

  FaceIndexType periodic_neighbor_index() const {
    ASSERT(is_periodic(), "Face is not periodic");
    return tria->periodic_face_neighbor(index);
  }

  FaceAccessor<dim> periodic_neighbor() const {
    return tria->get_face(periodic_neighbor_index());
  }

  Tensor<1, dim, double> periodic_offset() const {
    ASSERT(is_periodic(), "Face is not periodic");
    Tensor<1, dim, double> p;
    for (unsigned int d = 0; d < dim; ++d) {
      p(d) = tria->periodic_face_offset(index, d);
    }
    return p;
  }

  LocalIndexType periodic_neighbor_face_index() const {
    ASSERT(is_periodic(), "Face is not periodic");
    const auto neighbor_face_idx = periodic_neighbor_index();
    const auto neighbor_cell_idx = tria->face_cells(neighbor_face_idx, 0);
    const auto neighbor_cell = tria->get_cell(neighbor_cell_idx);

    for (LocalIndexType lf = 0; lf < Topo::faces_per_cell; ++lf)
      if (neighbor_cell.face_index(lf) == neighbor_face_idx)
        return lf;

    ASSERT(false, "Could not find periodic neighbor face index");
    return 0;
  }

  Tensor<1, dim, double> center() const {
    Tensor<1, dim, double> c;
    for (LocalIndexType v = 0; v < Topo::verts_per_face; ++v) {
      c += vertex(v);
    }
    return c / static_cast<double>(Topo::verts_per_face);
  }

  double measure() const {
    if constexpr (dim == 2) {
      const auto v0 = vertex(0);
      const auto v1 = vertex(1);
      const auto e = v1 - v0;
      return Kokkos::sqrt(e(0) * e(0) + e(1) * e(1));
    }
    return 0.0;
  }

  Tensor<1, dim, double> normal(CellIndexType cell) const {
    Tensor<1, dim, double> n;

    if constexpr (dim == 2) {
      const auto v0 = vertex(0);
      const auto v1 = vertex(1);
      const auto e = v1 - v0;

      const double len = Kokkos::sqrt(e(0) * e(0) + e(1) * e(1));
      // Rotate tangent 90 degrees to get a candidate normal
      n(0) = e(1) / len;
      n(1) = -e(0) / len;
    }

    // Orient away from owner cell center
    auto owner_center = tria->get_cell(owner_index()).center();

    // If periodic subtract the displacement if not the owner cell
    if (is_periodic() && (owner_index() != cell)) {
      owner_center -= periodic_offset();
    }
    const auto face_center = center();
    const auto d = face_center - owner_center;

    double dot = 0.0;
    for (unsigned int i = 0; i < dim; ++i) {
      dot += d(i) * n(i);
    }

    if (dot < 0.0) {
      for (unsigned int i = 0; i < dim; ++i) {
        n(i) = -n(i);
      }
    }

    if (cell != owner_index()) {
      return -n;
    }

    return n;
  }
};

template <unsigned int dim> class Triangulation {
public:
  using Topo = SimplexTopology<dim>;

  friend struct CellAccessor<dim>;
  friend struct FaceAccessor<dim>;
  friend class GriReader<dim>;

  template <typename T>
  using HostMat = typename MatrixViewTrait<T, HostMemSpace>::type;

  template <typename T>
  using HostVec = typename VectorViewTrait<T, HostMemSpace>::type;

  size_t n_vertices() const { return vertices.extent(0); };

  size_t n_cells() const { return cell_vertices.extent(0); };

  size_t n_faces() const { return face_vertices.extent(0); };

  size_t n_boundary_faces() const {
    size_t count = 0;
    auto flags = face_flags;
    Kokkos::parallel_reduce(
        "n_boundary", Kokkos::RangePolicy<HostExecSpace>(0, n_faces()),
        [=](const size_t f, size_t &lc) {
          if (flags(f) & FaceFlags::Boundary) {
            ++lc;
          }
        },
        count);
    return count;
  }

  size_t n_periodic_faces() const {
    size_t count = 0;
    auto flags = face_flags;
    Kokkos::parallel_reduce(
        "n_periodic", Kokkos::RangePolicy<HostExecSpace>(0, n_faces()),
        [=](const size_t f, size_t &lc) {
          if (flags(f) & FaceFlags::Periodic) {
            ++lc;
          }
        },
        count);
    return count;
  }

  CellAccessor<dim> get_cell(CellIndexType c) const { return {c, this}; }

  FaceAccessor<dim> get_face(FaceIndexType f) const { return {f, this}; }

  struct ActiveCellRange {
    const Triangulation *tria;

    struct Iterator {
      const Triangulation *tria;
      size_t idx;

      void advance() {
        while (idx < tria->n_cells() && !true) {
          ++idx;
        }
      }

      Iterator &operator++() {
        ++idx;
        advance();
        return *this;
      }
      bool operator!=(const Iterator &o) const { return idx != o.idx; }
      CellAccessor<dim> operator*() const { return tria->get_cell(idx); }
    };

    Iterator begin() const {
      Iterator it{tria, 0};
      it.advance();
      return it;
    }
    Iterator end() const { return {tria, tria->n_cells()}; }
  };

  ActiveCellRange active_cell_range() const { return {this}; }

  struct BoundaryFaceRange {
    const Triangulation *tria;

    struct Iterator {
      const Triangulation *tria;
      size_t idx;

      void advance() {
        while (idx < tria->n_faces() &&
               !(tria->face_flags(idx) & FaceFlags::Boundary)) {
          ++idx;
        }
      }

      Iterator &operator++() {
        ++idx;
        advance();
        return *this;
      }
      bool operator!=(const Iterator &o) const { return idx != o.idx; }
      FaceAccessor<dim> operator*() const { return tria->get_face(idx); }
    };

    Iterator begin() const {
      Iterator it{tria, 0};
      it.advance();
      return it;
    }
    Iterator end() const { return {tria, tria->n_faces()}; }
  };

  BoundaryFaceRange boundary_face_range() const { return {this}; }

  struct PeriodicFaceRange {
    const Triangulation *tria;

    struct Iterator {
      const Triangulation *tria;
      size_t idx;

      void advance() {
        while (idx < tria->n_faces() &&
               !(tria->face_flags(idx) & FaceFlags::Periodic)) {
          ++idx;
        }
      }

      Iterator &operator++() {
        ++idx;
        advance();
        return *this;
      }
      bool operator!=(const Iterator &o) const { return idx != o.idx; }
      FaceAccessor<dim> operator*() const { return tria->get_face(idx); }
    };

    Iterator begin() const {
      Iterator it{tria, 0};
      it.advance();
      return it;
    }
    Iterator end() const { return {tria, tria->n_faces()}; }
  };

  PeriodicFaceRange periodic_face_range() const { return {this}; }

  bool verify_mesh() const {
    constexpr double tol = 1e-12;
    bool passed = true;

    for (auto cell : active_cell_range()) {
      Tensor<1, dim, double> sum;
      for (LocalIndexType lf = 0; lf < Topo::faces_per_cell; ++lf) {
        const auto face = cell.face(lf);

        const auto n = face.normal(cell.index);
        const double len = face.measure();
        sum += n * len;
      }

      if (sum.norm() > tol) {
        passed = false;
      }
    }

    return passed;
  }

  void
  remap_boundary_ids(const std::unordered_map<uint32_t, uint32_t> &id_map) {
    for (size_t f = 0; f < n_faces(); ++f) {
      if (face_flags(f) & FaceFlags::Boundary) {
        auto it = id_map.find(boundary_ids(f));
        if (it != id_map.end())
          boundary_ids(f) = it->second;
      }
    }
  }

  // TODO: Undo this
  // private:

  /**
   * @brief Vertex positions.
   *
   * Array size is n_nodes, dim
   */
  HostMat<double> vertices;

  /**
   * @brief Global vertex indices for each cell.
   *
   * Array size is n_cells, Topo<dim>::verts_per_cell
   */
  HostMat<VertexIndexType> cell_vertices;

  // ----------- NEW STUFF -------------------
  /**
   * @brief Geometry order for each cell.
   *
   * Array size is n_cells
   *
   * 1 = linear geometry
   * 2 = quadratic geometry
   */
  HostVec<uint8_t> cell_geometry_order;

  /**
   * @brief Geometry node indices for each cell.
   *
   * Array size is n_cells, 6
   *
   * Entries 0,1,2 are the corner nodes.
   * Entries 3,4,5 are midside nodes for q=2 geometry.
   * For q=1 cells, the last three entries can be invalid sentinels.
   */
  HostMat<VertexIndexType> cell_geometry_nodes;
  // ----------------------------------------

  /**
   * @brief Global faces indices for each cell.
   *
   * Array size is n_cells, Topo<dim>::faces_per_cell
   */
  HostMat<FaceIndexType> cell_faces;

  /**
   * @brief Global vertex indices for each face.
   *
   * Array size is n_faces, Topo<dim>::verts_per_face
   */
  HostMat<VertexIndexType> face_vertices;

  // ----------- NEW STUFF -------------------
  /**
   * @brief Geometry order for each face.
   *
   * Array size is n_faces
   *
   * 1 = linear face geometry
   * 2 = quadratic face geometry
   */
  HostVec<uint8_t> face_geometry_order;

  /**
   * @brief Geometry node indices for each face.
   *
   * Array size is n_faces, 3
   *
   * Entries are stored as:
   *   0 = first corner node
   *   1 = midside node for q=2 face geometry, or invalid sentinel for q=1
   *   2 = second corner node
   */
  HostMat<VertexIndexType> face_geometry_nodes;
  // ----------------------------------------

  /**
   * @brief Global cell indices of cell neighbors for each face.
   *
   * Array size is n_faces, 2
   *
   * When the face is a boundary, the neighbor is NO_NEIGHBOR = -1. Note that
   * the NO_NEIGHBOR will always occur in the 2nd index.
   */
  HostMat<NeighborIndexType> face_cells;

  /**
   * @brief Face flags for each face.
   *
   * Array size is n_faces
   */
  HostVec<BoundaryIdType> face_flags;

  /**
   * @brief Boundary ID for each face.
   *
   * Array size is n_faces
   */
  HostVec<BoundaryIdType> boundary_ids;

  /**
   * @brief Global face index for the periodic neighbor face.
   *
   * Array size is n_faces
   *
   * When there is no matching periodic face, the neighbor is NO_NEIGHBOR = -1
   */
  HostVec<FaceIndexType> periodic_face_neighbor;

  /**
   * @brief Geometric displacement for each periodic face pair.
   *
   * Array size is n_faces, dim
   */
  HostMat<double> periodic_face_offset;

  void internal_reinit(unsigned int n_cells, unsigned int n_faces,
                       unsigned int n_nodes) {
    Kokkos::resize(vertices, n_nodes, dim);
    Kokkos::resize(cell_vertices, n_cells, Topo::verts_per_cell);
    // NEW STUFF ------------------
    Kokkos::resize(cell_geometry_order, n_cells);
    Kokkos::resize(cell_geometry_nodes, n_cells, 6);
    Kokkos::resize(face_geometry_order, n_faces);
    Kokkos::resize(face_geometry_nodes, n_faces, 3);
    // ----------------------------
    Kokkos::resize(cell_faces, n_cells, Topo::faces_per_cell);
    Kokkos::resize(face_vertices, n_faces, Topo::verts_per_face);
    Kokkos::resize(face_cells, n_faces, 2);
    Kokkos::resize(face_flags, n_faces);
    Kokkos::resize(boundary_ids, n_faces);
    Kokkos::resize(periodic_face_neighbor, n_faces);
    Kokkos::resize(periodic_face_offset, n_faces, dim);

    Kokkos::deep_copy(face_flags, BoundaryIdType(0));
    Kokkos::deep_copy(boundary_ids, BoundaryIdType(0));
    Kokkos::deep_copy(periodic_face_neighbor, FaceIndexType(-1));
    Kokkos::deep_copy(periodic_face_offset, double(0));
    // NEW STUFF ------------------
    Kokkos::deep_copy(cell_geometry_order, uint8_t(1));
    Kokkos::deep_copy(cell_geometry_nodes, VertexIndexType(-1));
    Kokkos::deep_copy(face_geometry_order, uint8_t(1));
    Kokkos::deep_copy(face_geometry_nodes, VertexIndexType(-1));
    // ----------------------------
  }
};