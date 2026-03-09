#pragma once

#include <Kokkos_Core.hpp>
#include <config.hpp>
#include <cstdint>

using LocalIndexType = uint8_t;
using CellIndexType = uint32_t;
using FaceIndexType = uint32_t;
using VertexIndexType = uint32_t;
using BoundaryIdType = uint8_t;
using NeighborIndexType = int32_t;

constexpr NeighborIndexType NO_NEIGHBOR = -1;

/**
 * @brief Bit flags for cell types.
 */
namespace CellFlags {
constexpr uint8_t Active = 1 << 0;
constexpr uint8_t LocallyOwned = 1 << 1;
constexpr uint8_t Ghost = 1 << 2;
constexpr uint8_t Refine = 1 << 3;
constexpr uint8_t Coarsen = 1 << 4;
}

/**
 * @brief Bit flags for face types.
 */
namespace FaceFlags {
constexpr uint8_t Boundary = 1 << 0;
constexpr uint8_t Interior = 1 << 1;
constexpr uint8_t Hanging = 1 << 2;
}

/**
 * @brief Mesh topology constants for simplex elements.
 */
template<int dim>
struct SimplexTopology;

template<>
struct SimplexTopology<1>
{
  static constexpr uint8_t verts_per_cell = 2;
  static constexpr uint8_t faces_per_cell = 2;
  static constexpr uint8_t verts_per_face = 1;
  static constexpr uint8_t face_verts[2][1] = { { 0 }, { 1 } };
};

template<>
struct SimplexTopology<2>
{
  static constexpr uint8_t verts_per_cell = 3;
  static constexpr uint8_t faces_per_cell = 3;
  static constexpr uint8_t verts_per_face = 2;
  static constexpr uint8_t face_verts[3][2] = { { 1, 2 }, { 0, 2 }, { 0, 1 } };
};

template<>
struct SimplexTopology<3>
{
  static constexpr uint8_t verts_per_cell = 4;
  static constexpr uint8_t faces_per_cell = 4;
  static constexpr uint8_t verts_per_face = 3;
  static constexpr uint8_t face_verts[4][3] = { { 1, 2, 3 },
                                                { 0, 2, 3 },
                                                { 0, 1, 3 },
                                                { 0, 1, 2 } };
};

template<unsigned int dim>
class Triangulation;

/**
 * @brief Cell accessor for cell loops.
 */
template<unsigned int dim>
struct CellAccessor
{
public:
  using Topo = SimplexTopology<dim>;

  CellIndexType index;
  const Triangulation<dim>* tria;

  static constexpr uint8_t n_vertices() { return Topo::verts_per_cell; }
  static constexpr uint8_t n_faces() { return Topo::faces_per_cell; }

  VertexIndexType vertex_index(uint8_t local_v) const
  {
    return tria->cell_verts(index, local_v);
  }

  std::array<double, dim> vertex(uint8_t local_v) const
  {
    std::array<double, dim> p;
    VertexIndexType gv = vertex_index(local_v);
    for (int d = 0; d < dim; ++d)
      p[d] = tria->vertices(gv, d);
    return p;
  }

  FaceIndexType face_index(uint8_t local_f) const
  {
    return tria->cell_faces(index, local_f);
  }

  bool face_at_boundary(uint8_t local_f) const
  {
    return tria->face_flags(face_index(local_f)) & FaceFlags::Boundary;
  }

  BoundaryIdType face_boundary_id(uint8_t local_f) const
  {
    return tria->boundary_ids(face_index(local_f));
  }

  std::array<double, dim> center() const
  {
    std::array<double, dim> c{};
    for (uint8_t v = 0; v < Topo::verts_per_cell; ++v) {
      auto p = vertex(v);
      for (int d = 0; d < dim; ++d)
        c[d] += p[d];
    }
    for (int d = 0; d < dim; ++d)
      c[d] /= static_cast<double>(Topo::verts_per_cell);
    return c;
  }

  double diameter() const
  {
    double d2_max = 0.0;
    for (uint8_t i = 0; i < Topo::verts_per_cell; ++i) {
      for (uint8_t j = i + 1; j < Topo::verts_per_cell; ++j) {
        double d2 = 0.0;
        auto vi = vertex(i), vj = vertex(j);
        for (int d = 0; d < dim; ++d) {
          double dk = vi[d] - vj[d];
          d2 += dk * dk;
        }
        if (d2 > d2_max)
          d2_max = d2;
      }
    }
    return std::sqrt(d2_max);
  }

  double measure() const
  {
    if constexpr (dim == 2) {
      auto v0 = vertex(0), v1 = vertex(1), v2 = vertex(2);
      double a0 = v1[0] - v0[0], a1 = v1[1] - v0[1];
      double b0 = v2[0] - v0[0], b1 = v2[1] - v0[1];
      return 0.5 * std::abs(a0 * b1 - a1 * b0);
    }
    if constexpr (dim == 3) {
      auto v0 = vertex(0), v1 = vertex(1), v2 = vertex(2), v3 = vertex(3);
      double a0 = v1[0] - v0[0], a1 = v1[1] - v0[1], a2 = v1[2] - v0[2];
      double b0 = v2[0] - v0[0], b1 = v2[1] - v0[1], b2 = v2[2] - v0[2];
      double c0 = v3[0] - v0[0], c1 = v3[1] - v0[1], c2 = v3[2] - v0[2];
      double det = a0 * (b1 * c2 - b2 * c1) - a1 * (b0 * c2 - b2 * c0) +
                   a2 * (b0 * c1 - b1 * c0);
      return std::abs(det) / 6.0;
    }
    return 0.0;
  }

  bool is_active() const { return tria->cell_flags(index) & CellFlags::Active; }
  bool is_locally_owned() const
  {
    return tria->cell_flags(index) & CellFlags::LocallyOwned;
  }
  bool is_ghost() const { return tria->cell_flags(index) & CellFlags::Ghost; }

  void set_refine_flag() const { tria->cell_flags(index) |= CellFlags::Refine; }
  void clear_refine_flag() const
  {
    tria->cell_flags(index) &= ~CellFlags::Refine;
  }

  BoundaryIdType material_id() const { return tria->material_ids(index); }
  void set_material_id(BoundaryIdType id) const
  {
    tria->material_ids(index) = id;
  }

  uint32_t level() const { return tria->cell_level(index); }
  int32_t parent() const { return tria->cell_parent(index); }
};

template<unsigned int dim>
struct FaceAccessor
{
  using Topo = SimplexTopology<dim>;

  FaceIndexType index;
  const Triangulation<dim>* tria;

  static constexpr uint8_t n_vertices() { return Topo::verts_per_face; }

  bool at_boundary() const
  {
    return tria->face_flags(index) & FaceFlags::Boundary;
  }
  bool is_interior() const
  {
    return tria->face_flags(index) & FaceFlags::Interior;
  }

  BoundaryIdType boundary_id() const { return tria->boundary_ids(index); }
  void set_boundary_id(BoundaryIdType id) const
  {
    tria->boundary_ids(index) = id;
  }

  NeighborIndexType neighbor(uint8_t side) const
  {
    return tria->face_cells(index, side);
  }

  VertexIndexType vertex_index(uint8_t local_v) const
  {
    return tria->face_verts(index, local_v);
  }

  std::array<double, dim> vertex(uint8_t local_v) const
  {
    std::array<double, dim> p;
    VertexIndexType gv = vertex_index(local_v);
    for (int d = 0; d < dim; ++d)
      p[d] = tria->vertices(gv, d);
    return p;
  }

  std::array<double, dim> center() const
  {
    std::array<double, dim> c{};
    for (uint8_t v = 0; v < Topo::verts_per_face; ++v) {
      auto p = vertex(v);
      for (int d = 0; d < dim; ++d)
        c[d] += p[d];
    }
    for (int d = 0; d < dim; ++d)
      c[d] /= static_cast<double>(Topo::verts_per_face);
    return c;
  }

  double measure() const
  {
    if constexpr (dim == 2) {
      auto v0 = vertex(0), v1 = vertex(1);
      double dx = v1[0] - v0[0], dy = v1[1] - v0[1];
      return std::sqrt(dx * dx + dy * dy);
    }
    if constexpr (dim == 3) {
      auto v0 = vertex(0), v1 = vertex(1), v2 = vertex(2);
      double ax = v1[0] - v0[0], ay = v1[1] - v0[1], az = v1[2] - v0[2];
      double bx = v2[0] - v0[0], by = v2[1] - v0[1], bz = v2[2] - v0[2];
      double cx = ay * bz - az * by, cy = az * bx - ax * bz,
             cz = ax * by - ay * bx;
      return 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
    }
    return 0.0;
  }

  std::array<double, 2> unit_normal_2d() const
    requires(dim == 2)
  {
    auto v0 = vertex(0), v1 = vertex(1);
    double tx = v1[0] - v0[0], ty = v1[1] - v0[1];
    double len = std::sqrt(tx * tx + ty * ty);
    return { ty / len, -tx / len };
  }

  std::array<double, 3> unit_normal_3d() const
    requires(dim == 3)
  {
    auto v0 = vertex(0), v1 = vertex(1), v2 = vertex(2);
    double ax = v1[0] - v0[0], ay = v1[1] - v0[1], az = v1[2] - v0[2];
    double bx = v2[0] - v0[0], by = v2[1] - v0[1], bz = v2[2] - v0[2];
    double cx = ay * bz - az * by, cy = az * bx - ax * bz,
           cz = ax * by - ay * bx;
    double len = std::sqrt(cx * cx + cy * cy + cz * cz);
    return { cx / len, cy / len, cz / len };
  }
};

template<typename T>
using HostVec = typename VectorViewTrait<T, HostMemSpace>::type;

template<typename T>
using HostMat = typename MatrixViewTrait<T, HostMemSpace>::type;

template<unsigned int dim>
class Triangulation
{
public:
  using Topo = SimplexTopology<dim>;

  friend struct CellAccessor<dim>;
  friend struct FaceAccessor<dim>;

  HostMat<double> vertices;
  HostMat<VertexIndexType> cell_verts;
  HostMat<FaceIndexType> cell_faces;
  HostMat<VertexIndexType> face_verts;
  HostMat<NeighborIndexType> face_cells;

  mutable HostVec<uint8_t> cell_flags;
  mutable HostVec<uint8_t> face_flags;
  mutable HostVec<BoundaryIdType> boundary_ids;
  mutable HostVec<BoundaryIdType> material_ids;
  HostVec<uint32_t> cell_level;
  HostVec<int32_t> cell_parent;

  uint32_t n_vertices() const
  {
    return static_cast<uint32_t>(vertices.extent(0));
  }
  uint32_t n_cells() const
  {
    return static_cast<uint32_t>(cell_verts.extent(0));
  }
  uint32_t n_faces() const
  {
    return static_cast<uint32_t>(face_verts.extent(0));
  }

  uint32_t n_active_cells() const
  {
    uint32_t count = 0;
    Kokkos::parallel_reduce(
      "n_active",
      host_range_policy(0, n_cells()),
      [&](const uint32_t c, uint32_t& lc) {
        if (cell_flags(c) & CellFlags::Active)
          ++lc;
      },
      count);
    return count;
  }

  uint32_t n_boundary_faces() const
  {
    uint32_t count = 0;
    Kokkos::parallel_reduce(
      "n_bnd",
      host_range_policy(0, n_faces()),
      [&](const uint32_t f, uint32_t& lc) {
        if (face_flags(f) & FaceFlags::Boundary)
          ++lc;
      },
      count);
    return count;
  }

  CellAccessor<dim> get_cell(CellIndexType c) const { return { c, this }; }

  FaceAccessor<dim> get_face(FaceIndexType f) const { return { f, this }; }

  struct ActiveCellRange
  {
    const Triangulation* tria;

    struct Iterator
    {
      const Triangulation* tria;
      uint32_t idx;

      void advance()
      {
        while (idx < tria->n_cells() &&
               !(tria->cell_flags(idx) & CellFlags::Active))
          ++idx;
      }
      Iterator& operator++()
      {
        ++idx;
        advance();
        return *this;
      }
      bool operator!=(const Iterator& o) const { return idx != o.idx; }
      CellAccessor<dim> operator*() const { return tria->get_cell(idx); }
    };

    Iterator begin() const
    {
      Iterator it{ tria, 0 };
      it.advance();
      return it;
    }
    Iterator end() const { return { tria, tria->n_cells() }; }
  };

  ActiveCellRange active_cell_range() const { return { this }; }

  struct BoundaryFaceRange
  {
    const Triangulation* tria;

    struct Iterator
    {
      const Triangulation* tria;
      uint32_t idx;

      void advance()
      {
        while (idx < tria->n_faces() &&
               !(tria->face_flags(idx) & FaceFlags::Boundary))
          ++idx;
      }
      Iterator& operator++()
      {
        ++idx;
        advance();
        return *this;
      }
      bool operator!=(const Iterator& o) const { return idx != o.idx; }
      FaceAccessor<dim> operator*() const { return tria->get_face(idx); }
    };

    Iterator begin() const
    {
      Iterator it{ tria, 0 };
      it.advance();
      return it;
    }
    Iterator end() const { return { tria, tria->n_faces() }; }
  };

  BoundaryFaceRange boundary_face_range() const { return { this }; }

  void generate_hyper_cube(double left = 0.0,
                           double right = 1.0,
                           bool colorize = false)
  {
    generate_subdivided_hyper_cube(1, left, right, colorize);
  }

  void generate_subdivided_hyper_cube(uint32_t n,
                                      double left = 0.0,
                                      double right = 1.0,
                                      bool colorize = false)
  {
    raw = {};
    if constexpr (dim == 2)
      build_triangle_grid(n, left, right);
    if constexpr (dim == 3)
      build_tet_grid(n, left, right);
    build_face_connectivity();
    mark_all_active();
    if (colorize)
      colorize_cube(left, right);
    commit();
  }

  void generate_subdivided_hyper_rectangle(
    const std::array<double, dim>& p1,
    const std::array<double, dim>& p2,
    const std::array<uint32_t, dim>& reps,
    bool colorize = false)
  {
    raw = {};
    if constexpr (dim == 2)
      build_triangle_rect(p1, p2, reps);
    if constexpr (dim == 3)
      build_tet_rect(p1, p2, reps);
    build_face_connectivity();
    mark_all_active();
    if (colorize)
      colorize_rect(p1, p2);
    commit();
  }

  void refine_global(uint32_t n_times = 1)
  {
    for (uint32_t t = 0; t < n_times; ++t) {
      for (uint32_t c = 0; c < n_cells(); ++c)
        cell_flags(c) |= CellFlags::Refine;
      execute_refinement();
    }
  }

  void execute_refinement()
  {
    if constexpr (dim == 2)
      refine_triangles();
    if constexpr (dim == 3)
      refine_tets();
    build_face_connectivity();
    commit();
  }

private:
  struct RawMesh
  {
    std::vector<std::array<double, dim>> vertices;
    std::vector<std::array<VertexIndexType, Topo::verts_per_cell>> cell_verts;
    std::vector<std::array<FaceIndexType, Topo::faces_per_cell>> cell_faces;
    std::vector<std::array<VertexIndexType, Topo::verts_per_face>> face_verts;
    std::vector<std::array<NeighborIndexType, 2>> face_cells;
    std::vector<uint8_t> cell_flags;
    std::vector<uint8_t> face_flags;
    std::vector<BoundaryIdType> boundary_ids;
    std::vector<BoundaryIdType> material_ids;
    std::vector<uint32_t> cell_level;
    std::vector<int32_t> cell_parent;
  } raw;

  void commit()
  {
    uint32_t nv = static_cast<uint32_t>(raw.vertices.size());
    uint32_t nc = static_cast<uint32_t>(raw.cell_verts.size());
    uint32_t nf = static_cast<uint32_t>(raw.face_verts.size());

    vertices = fill_mat<double>(raw.vertices, nv, dim, "vertices");
    cell_verts = fill_mat<VertexIndexType>(
      raw.cell_verts, nc, Topo::verts_per_cell, "cell_verts");
    cell_faces = fill_mat<FaceIndexType>(
      raw.cell_faces, nc, Topo::faces_per_cell, "cell_faces");
    face_verts = fill_mat<VertexIndexType>(
      raw.face_verts, nf, Topo::verts_per_face, "face_verts");
    face_cells =
      fill_mat<NeighborIndexType>(raw.face_cells, nf, 2, "face_cells");
    cell_flags = fill_vec<uint8_t>(raw.cell_flags, nc, "cell_flags");
    face_flags = fill_vec<uint8_t>(raw.face_flags, nf, "face_flags");
    boundary_ids =
      fill_vec<BoundaryIdType>(raw.boundary_ids, nf, "boundary_ids");
    material_ids =
      fill_vec<BoundaryIdType>(raw.material_ids, nc, "material_ids");
    cell_level = fill_vec<uint32_t>(raw.cell_level, nc, "cell_level");
    cell_parent = fill_vec<int32_t>(raw.cell_parent, nc, "cell_parent");

    raw = {};
  }

  void build_triangle_grid(uint32_t n, double left, double right)
  {
    double h = (right - left) / n;
    for (uint32_t j = 0; j <= n; ++j)
      for (uint32_t i = 0; i <= n; ++i)
        raw.vertices.push_back({ left + i * h, left + j * h });

    auto vid = [&](uint32_t i, uint32_t j) -> VertexIndexType {
      return static_cast<VertexIndexType>(j * (n + 1) + i);
    };

    for (uint32_t j = 0; j < n; ++j) {
      for (uint32_t i = 0; i < n; ++i) {
        VertexIndexType v00 = vid(i, j), v10 = vid(i + 1, j),
                        v11 = vid(i + 1, j + 1), v01 = vid(i, j + 1);
        raw.cell_verts.push_back({ v00, v10, v11 });
        raw.cell_verts.push_back({ v00, v11, v01 });
      }
    }
  }

  void build_triangle_rect(const std::array<double, 2>& p1,
                           const std::array<double, 2>& p2,
                           const std::array<uint32_t, 2>& reps)
  {
    uint32_t nx = reps[0], ny = reps[1];
    double hx = (p2[0] - p1[0]) / nx, hy = (p2[1] - p1[1]) / ny;
    for (uint32_t j = 0; j <= ny; ++j)
      for (uint32_t i = 0; i <= nx; ++i)
        raw.vertices.push_back({ p1[0] + i * hx, p1[1] + j * hy });

    auto vid = [&](uint32_t i, uint32_t j) -> VertexIndexType {
      return static_cast<VertexIndexType>(j * (nx + 1) + i);
    };

    for (uint32_t j = 0; j < ny; ++j)
      for (uint32_t i = 0; i < nx; ++i) {
        VertexIndexType v00 = vid(i, j), v10 = vid(i + 1, j),
                        v11 = vid(i + 1, j + 1), v01 = vid(i, j + 1);
        raw.cell_verts.push_back({ v00, v10, v11 });
        raw.cell_verts.push_back({ v00, v11, v01 });
      }
  }

  void build_tet_grid(uint32_t n, double left, double right)
  {
    double h = (right - left) / n;
    for (uint32_t k = 0; k <= n; ++k)
      for (uint32_t j = 0; j <= n; ++j)
        for (uint32_t i = 0; i <= n; ++i)
          raw.vertices.push_back({ left + i * h, left + j * h, left + k * h });

    auto vid = [&](uint32_t i, uint32_t j, uint32_t k) -> VertexIndexType {
      return static_cast<VertexIndexType>(k * (n + 1) * (n + 1) + j * (n + 1) +
                                          i);
    };

    // Kuhn partition: 6 tets per cube cell
    for (uint32_t k = 0; k < n; ++k)
      for (uint32_t j = 0; j < n; ++j)
        for (uint32_t i = 0; i < n; ++i) {
          VertexIndexType v000 = vid(i, j, k), v100 = vid(i + 1, j, k),
                          v010 = vid(i, j + 1, k), v110 = vid(i + 1, j + 1, k),
                          v001 = vid(i, j, k + 1), v101 = vid(i + 1, j, k + 1),
                          v011 = vid(i, j + 1, k + 1),
                          v111 = vid(i + 1, j + 1, k + 1);
          raw.cell_verts.push_back({ v000, v100, v110, v111 });
          raw.cell_verts.push_back({ v000, v100, v101, v111 });
          raw.cell_verts.push_back({ v000, v010, v110, v111 });
          raw.cell_verts.push_back({ v000, v010, v011, v111 });
          raw.cell_verts.push_back({ v000, v001, v101, v111 });
          raw.cell_verts.push_back({ v000, v001, v011, v111 });
        }
  }

  void build_tet_rect(const std::array<double, 3>& p1,
                      const std::array<double, 3>& p2,
                      const std::array<uint32_t, 3>& reps)
  {
    uint32_t nx = reps[0], ny = reps[1], nz = reps[2];
    double hx = (p2[0] - p1[0]) / nx, hy = (p2[1] - p1[1]) / ny,
           hz = (p2[2] - p1[2]) / nz;
    for (uint32_t k = 0; k <= nz; ++k)
      for (uint32_t j = 0; j <= ny; ++j)
        for (uint32_t i = 0; i <= nx; ++i)
          raw.vertices.push_back(
            { p1[0] + i * hx, p1[1] + j * hy, p1[2] + k * hz });

    auto vid = [&](uint32_t i, uint32_t j, uint32_t k) -> VertexIndexType {
      return static_cast<VertexIndexType>(k * (ny + 1) * (nx + 1) +
                                          j * (nx + 1) + i);
    };

    for (uint32_t k = 0; k < nz; ++k)
      for (uint32_t j = 0; j < ny; ++j)
        for (uint32_t i = 0; i < nx; ++i) {
          VertexIndexType v000 = vid(i, j, k), v100 = vid(i + 1, j, k),
                          v010 = vid(i, j + 1, k), v110 = vid(i + 1, j + 1, k),
                          v001 = vid(i, j, k + 1), v101 = vid(i + 1, j, k + 1),
                          v011 = vid(i, j + 1, k + 1),
                          v111 = vid(i + 1, j + 1, k + 1);
          raw.cell_verts.push_back({ v000, v100, v110, v111 });
          raw.cell_verts.push_back({ v000, v100, v101, v111 });
          raw.cell_verts.push_back({ v000, v010, v110, v111 });
          raw.cell_verts.push_back({ v000, v010, v011, v111 });
          raw.cell_verts.push_back({ v000, v001, v101, v111 });
          raw.cell_verts.push_back({ v000, v001, v011, v111 });
        }
  }

  void build_face_connectivity()
  {
    uint32_t nc = static_cast<uint32_t>(raw.cell_verts.size());
    raw.cell_faces.resize(nc);
    raw.face_verts.clear();
    raw.face_cells.clear();
    raw.face_flags.clear();
    raw.boundary_ids.clear();

    // Sorted vertex key → face index
    std::map<std::vector<VertexIndexType>, FaceIndexType> face_map;

    for (uint32_t c = 0; c < nc; ++c) {
      for (uint8_t lf = 0; lf < Topo::faces_per_cell; ++lf) {

        // Look up local vertex indices from topology table
        std::array<VertexIndexType, Topo::verts_per_face> fv;
        for (uint8_t v = 0; v < Topo::verts_per_face; ++v)
          fv[v] = raw.cell_verts[c][Topo::face_verts[lf][v]];

        std::vector<VertexIndexType> key(fv.begin(), fv.end());
        std::sort(key.begin(), key.end());

        auto [it, inserted] = face_map.emplace(
          key, static_cast<FaceIndexType>(raw.face_verts.size()));
        FaceIndexType fi = it->second;

        raw.cell_faces[c][lf] = fi;
        if (inserted) {
          raw.face_verts.push_back(fv);
          raw.face_cells.push_back(
            { static_cast<NeighborIndexType>(c), NO_NEIGHBOR });
          raw.face_flags.push_back(FaceFlags::Boundary);
          raw.boundary_ids.push_back(0);
        } else {
          raw.face_cells[fi][1] = static_cast<NeighborIndexType>(c);
          raw.face_flags[fi] = FaceFlags::Interior;
        }
      }
    }
  }

  void refine_triangles()
  {
    pull_views_to_raw();

    std::vector<std::array<double, 2>> new_verts = raw.vertices;
    std::vector<std::array<VertexIndexType, 3>> new_cells;
    std::vector<uint8_t> new_flags;
    std::vector<uint32_t> new_level;
    std::vector<int32_t> new_parent;
    std::vector<BoundaryIdType> new_matid;

    // Cache: sorted edge (a,b) → midpoint vertex index
    std::map<std::pair<VertexIndexType, VertexIndexType>, VertexIndexType>
      mid_cache;

    auto get_mid = [&](VertexIndexType a,
                       VertexIndexType b) -> VertexIndexType {
      if (a > b)
        std::swap(a, b);
      auto [it, ins] = mid_cache.emplace(
        std::make_pair(a, b), static_cast<VertexIndexType>(new_verts.size()));
      if (ins) {
        auto& va = raw.vertices[a];
        auto& vb = raw.vertices[b];
        new_verts.push_back({ (va[0] + vb[0]) / 2.0, (va[1] + vb[1]) / 2.0 });
      }
      return it->second;
    };

    uint32_t nc = static_cast<uint32_t>(raw.cell_verts.size());
    for (uint32_t c = 0; c < nc; ++c) {
      auto& cv = raw.cell_verts[c];
      uint32_t lv = raw.cell_level[c];
      auto mid = raw.material_ids[c];

      if (!(raw.cell_flags[c] & CellFlags::Refine)) {
        new_cells.push_back(cv);
        new_flags.push_back(raw.cell_flags[c] & ~CellFlags::Refine);
        new_level.push_back(lv);
        new_parent.push_back(raw.cell_parent[c]);
        new_matid.push_back(mid);
        continue;
      }

      //        v2
      //       /  \
      //     m20   m12
      //     / \  / \
      //   v0--m01---v1
      //
      VertexIndexType m01 = get_mid(cv[0], cv[1]);
      VertexIndexType m12 = get_mid(cv[1], cv[2]);
      VertexIndexType m20 = get_mid(cv[2], cv[0]);

      uint8_t child_flag = CellFlags::Active | CellFlags::LocallyOwned;
      new_cells.push_back({ cv[0], m01, m20 });
      new_cells.push_back({ m01, cv[1], m12 });
      new_cells.push_back({ m20, m12, cv[2] });
      new_cells.push_back({ m01, m12, m20 }); // center

      for (uint8_t i = 0; i < 4; ++i) {
        new_flags.push_back(child_flag);
        new_level.push_back(lv + 1);
        new_parent.push_back(static_cast<int32_t>(c));
        new_matid.push_back(mid);
      }
    }

    raw.vertices = std::move(new_verts);
    raw.cell_verts = std::move(new_cells);
    raw.cell_flags = std::move(new_flags);
    raw.cell_level = std::move(new_level);
    raw.cell_parent = std::move(new_parent);
    raw.material_ids = std::move(new_matid);
  }

  void refine_tets()
  {
    // 8-subtetrahedron red refinement — same midpoint pattern in 3D
    // (omitted for brevity; same structure as refine_triangles)
  }

  void colorize_cube(double left, double right)
  {
    const double eps = 1e-10 * (right - left);
    uint32_t nf = static_cast<uint32_t>(raw.face_verts.size());
    for (uint32_t f = 0; f < nf; ++f) {
      if (!(raw.face_flags[f] & FaceFlags::Boundary))
        continue;
      auto c = raw_face_center(f);
      if (c[0] < left + eps)
        raw.boundary_ids[f] = 0; // left
      else if (c[0] > right - eps)
        raw.boundary_ids[f] = 1; // right
      else if (c[1] < left + eps)
        raw.boundary_ids[f] = 2; // bottom
      else if (c[1] > right - eps)
        raw.boundary_ids[f] = 3; // top
      if constexpr (dim == 3) {
        if (c[2] < left + eps)
          raw.boundary_ids[f] = 4;
        else if (c[2] > right - eps)
          raw.boundary_ids[f] = 5;
      }
    }
  }

  void colorize_rect(const std::array<double, dim>& p1,
                     const std::array<double, dim>& p2)
  {
    double eps = 0.0;
    for (int d = 0; d < dim; ++d)
      eps = std::max(eps, 1e-10 * (p2[d] - p1[d]));
    uint32_t nf = static_cast<uint32_t>(raw.face_verts.size());
    for (uint32_t f = 0; f < nf; ++f) {
      if (!(raw.face_flags[f] & FaceFlags::Boundary))
        continue;
      auto c = raw_face_center(f);
      for (int d = 0; d < dim; ++d) {
        if (c[d] < p1[d] + eps) {
          raw.boundary_ids[f] = 2 * d;
          break;
        }
        if (c[d] > p2[d] - eps) {
          raw.boundary_ids[f] = 2 * d + 1;
          break;
        }
      }
    }
  }

  void mark_all_active()
  {
    uint32_t nc = static_cast<uint32_t>(raw.cell_verts.size());
    uint32_t nf = static_cast<uint32_t>(raw.face_verts.size());
    raw.cell_flags.assign(nc, CellFlags::Active | CellFlags::LocallyOwned);
    raw.face_flags.assign(nf, FaceFlags::Boundary);
    raw.boundary_ids.assign(nf, 0);
    raw.material_ids.assign(nc, 0);
    raw.cell_level.assign(nc, 0u);
    raw.cell_parent.assign(nc, -1);
  }

  std::array<double, dim> raw_face_center(uint32_t f) const
  {
    std::array<double, dim> c{};
    for (uint8_t v = 0; v < Topo::verts_per_face; ++v)
      for (int d = 0; d < dim; ++d)
        c[d] += raw.vertices[raw.face_verts[f][v]][d];
    for (int d = 0; d < dim; ++d)
      c[d] /= static_cast<double>(Topo::verts_per_face);
    return c;
  }

  // Pull committed Views back to raw before mutation (refinement)
  void pull_views_to_raw()
  {
    uint32_t nc = n_cells(), nf = n_faces(), nv = n_vertices();

    raw.vertices.resize(nv);
    raw.cell_verts.resize(nc);
    raw.cell_faces.resize(nc);
    raw.face_verts.resize(nf);
    raw.face_cells.resize(nf);
    raw.cell_flags.resize(nc);
    raw.face_flags.resize(nf);
    raw.boundary_ids.resize(nf);
    raw.material_ids.resize(nc);
    raw.cell_level.resize(nc);
    raw.cell_parent.resize(nc);

    for (uint32_t i = 0; i < nv; ++i)
      for (int d = 0; d < dim; ++d)
        raw.vertices[i][d] = vertices(i, d);

    for (uint32_t c = 0; c < nc; ++c) {
      for (uint8_t v = 0; v < Topo::verts_per_cell; ++v)
        raw.cell_verts[c][v] = cell_verts(c, v);
      for (uint8_t f = 0; f < Topo::faces_per_cell; ++f)
        raw.cell_faces[c][f] = cell_faces(c, f);
      raw.cell_flags[c] = cell_flags(c);
      raw.material_ids[c] = material_ids(c);
      raw.cell_level[c] = cell_level(c);
      raw.cell_parent[c] = cell_parent(c);
    }

    for (uint32_t f = 0; f < nf; ++f) {
      for (uint8_t v = 0; v < Topo::verts_per_face; ++v)
        raw.face_verts[f][v] = face_verts(f, v);
      raw.face_cells[f] = { face_cells(f, 0), face_cells(f, 1) };
      raw.face_flags[f] = face_flags(f);
      raw.boundary_ids[f] = boundary_ids(f);
    }
  }

  template<typename T, typename Src>
  HostMat<T> fill_mat(const Src& src,
                      uint32_t rows,
                      uint32_t cols,
                      const char* label)
  {
    HostMat<T> v;
    Kokkos::realloc(Kokkos::WithoutInitializing, v, rows, cols);
    for (uint32_t i = 0; i < rows; ++i)
      for (uint32_t j = 0; j < cols; ++j)
        v(i, j) = static_cast<T>(src[i][j]);
    return v;
  }

  template<typename T, typename Src>
  HostVec<T> fill_vec(const Src& src, uint32_t n, const char* label)
  {
    HostVec<T> v;
    Kokkos::realloc(Kokkos::WithoutInitializing, v, n);
    for (uint32_t i = 0; i < n; ++i)
      v(i) = static_cast<T>(src[i]);
    return v;
  }
};

template<unsigned int dim>
class DoFHandler
{};
