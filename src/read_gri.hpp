#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <libassert/assert.hpp>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

template<unsigned int dim, typename RealType>
struct ElementData
{
  std::vector<RealType> area;
  std::vector<RealType> inv_area;

  std::vector<RealType> centroid_x;
  std::vector<RealType> centroid_y;
  std::vector<RealType> centroid_z;

  std::vector<RealType> density;
  std::vector<RealType> momentum_x;
  std::vector<RealType> momentum_y;
  std::vector<RealType> momentum_z;
  std::vector<RealType> energy;

  std::vector<RealType> residual_density;
  std::vector<RealType> residual_momentum_x;
  std::vector<RealType> residual_momentum_y;
  std::vector<RealType> residual_momentum_z;
  std::vector<RealType> residual_energy;

  std::size_t size() const { return area.size(); }

  void resize(std::size_t n)
  {
    area.resize(n);
    inv_area.resize(n);

    centroid_x.resize(n);
    centroid_y.resize(n);
    if constexpr (dim == 3) {
      centroid_z.resize(n);
    }

    density.resize(n);
    momentum_x.resize(n);
    momentum_y.resize(n);
    if constexpr (dim == 3) {
      momentum_z.resize(n);
    }
    energy.resize(n);

    residual_density.resize(n);
    residual_momentum_x.resize(n);
    residual_momentum_y.resize(n);
    if constexpr (dim == 3) {
      residual_momentum_z.resize(n);
    }
    residual_energy.resize(n);
  }
};

template<unsigned int dim, typename RealType>
struct InteriorFaceData
{
  std::vector<unsigned int> elem_l;
  std::vector<unsigned int> face_l;
  std::vector<unsigned int> elem_r;
  std::vector<unsigned int> face_r;

  std::vector<RealType> normal_x;
  std::vector<RealType> normal_y;
  std::vector<RealType> normal_z;

  std::vector<RealType> face_area;

  std::vector<RealType> centroid_x;
  std::vector<RealType> centroid_y;
  std::vector<RealType> centroid_z;

  std::size_t size() const { return elem_l.size(); }

  void resize(std::size_t n)
  {
    elem_l.resize(n);
    elem_r.resize(n);
    face_l.resize(n);
    face_r.resize(n);

    normal_x.resize(n);
    normal_y.resize(n);
    if constexpr (dim == 3) {
      normal_z.resize(n);
    }

    face_area.resize(n);

    centroid_x.resize(n);
    centroid_y.resize(n);
    if constexpr (dim == 3) {
      centroid_z.resize(n);
    }
  }
};

template<unsigned int dim, typename RealType>
struct BoundaryFaceData
{
  std::vector<unsigned int> elem;
  std::vector<unsigned int> face;

  std::vector<unsigned int> boundary_id;

  std::vector<RealType> normal_x;
  std::vector<RealType> normal_y;
  std::vector<RealType> normal_z;

  std::vector<RealType> face_area;

  std::vector<RealType> centroid_x;
  std::vector<RealType> centroid_y;
  std::vector<RealType> centroid_z;

  std::size_t size() const { return elem.size(); }

  void resize(std::size_t n)
  {
    elem.resize(n);
    face.resize(n);

    boundary_id.resize(n);

    normal_x.resize(n);
    normal_y.resize(n);
    if constexpr (dim == 3) {
      normal_z.resize(n);
    }

    face_area.resize(n);

    centroid_x.resize(n);
    centroid_y.resize(n);
    if constexpr (dim == 3) {
      centroid_z.resize(n);
    }
  }
};

template<unsigned int dim, typename RealType>
struct PeriodicFaceData
{
  std::vector<unsigned int> elem_l;
  std::vector<unsigned int> face_l;
  std::vector<unsigned int> elem_r;
  std::vector<unsigned int> face_r;

  std::vector<RealType> normal_x;
  std::vector<RealType> normal_y;
  std::vector<RealType> normal_z;

  std::vector<RealType> face_area;

  std::vector<RealType> centroid_x;
  std::vector<RealType> centroid_y;
  std::vector<RealType> centroid_z;

  std::vector<RealType> translation_x;
  std::vector<RealType> translation_y;
  std::vector<RealType> translation_z;

  std::size_t size() const { return elem_l.size(); }

  void resize(std::size_t n)
  {
    elem_l.resize(n);
    elem_r.resize(n);
    face_l.resize(n);
    face_r.resize(n);

    normal_x.resize(n);
    normal_y.resize(n);
    if constexpr (dim == 3) {
      normal_z.resize(n);
    }

    face_area.resize(n);

    centroid_x.resize(n);
    centroid_y.resize(n);
    if constexpr (dim == 3) {
      centroid_z.resize(n);
    }

    translation_x.resize(n);
    translation_y.resize(n);
    if constexpr (dim == 3) {
      translation_z.resize(n);
    }
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
 * multiplied with boundary area at each element. The l2-norm of this error is
 * returned at the end.
 */
template<unsigned int dim, typename RealType>
RealType
mesh_verification(const InteriorFaceData<dim, RealType>& interior_face_scratch,
                  const BoundaryFaceData<dim, RealType>& boundary_face_scratch,
                  const PeriodicFaceData<dim, RealType>& periodic_face_scratch,
                  const ElementData<dim, RealType>& element_scratch)
{
  std::vector<std::array<RealType, dim>> elem_sum(element_scratch.size());

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

  for (unsigned int i = 0; i < boundary_face_scratch.size(); ++i) {
    const auto elem = boundary_face_scratch.elem[i];
    elem_sum[elem][0] +=
      boundary_face_scratch.normal_x[i] * boundary_face_scratch.face_area[i];
    elem_sum[elem][1] +=
      boundary_face_scratch.normal_y[i] * boundary_face_scratch.face_area[i];
  }

  for (unsigned int i = 0; i < periodic_face_scratch.size(); ++i) {
    const auto elem_l = periodic_face_scratch.elem_l[i];
    const auto elem_r = periodic_face_scratch.elem_r[i];

    elem_sum[elem_l][0] +=
      periodic_face_scratch.normal_x[i] * periodic_face_scratch.face_area[i];
    elem_sum[elem_l][1] +=
      periodic_face_scratch.normal_y[i] * periodic_face_scratch.face_area[i];
    elem_sum[elem_r][0] -=
      periodic_face_scratch.normal_x[i] * periodic_face_scratch.face_area[i];
    elem_sum[elem_r][1] -=
      periodic_face_scratch.normal_y[i] * periodic_face_scratch.face_area[i];
  }

  RealType error = 0.0;
  for (unsigned int i = 0; i < elem_sum.size(); ++i) {
    for (unsigned int d = 0; d < dim; ++d) {
      error += elem_sum[i][d] * elem_sum[i][d];
    }
  }
  return std::sqrt(error);
}

template<unsigned int dim, typename RealType>
class Mesh
{
public:
  Mesh(const MeshData& data)
  {
    static_assert(dim == 2, "Only 2D is supported.");

    for (const auto& n_faces : data.boundary_group_n_faces) {
      _n_boundary_faces += n_faces;
    }
    // TODO: Should add a check here to make sure we don't do any unintentional
    // rounding
    _n_interior_faces = (data.n_elements * (dim + 1) - _n_boundary_faces) / 2;
    for (const auto& n_nodes : data.periodic_group_n_nodes) {
      // TODO: This doesn't generalize to higher dimensions
      _n_periodic_faces += n_nodes - 1;
    }
    _n_boundary_faces -= 2 * _n_periodic_faces;

    std::cout << "Number of interior faces: " << _n_interior_faces << std::endl;
    std::cout << "Number of boundary faces: " << _n_boundary_faces << std::endl;
    std::cout << "Number of periodic faces: " << _n_periodic_faces << std::endl;

    // Compute element data
    compute_element_data(data);

    // Compute the true interior face data (excluding periodic)
    compute_interior_face_data(data);

    // Compute the boundary face data
    compute_boundary_face_data(data);

    // Compute the periodic face data
    compute_periodic_face_data(data);
  }

  const ElementData<dim, RealType>& get_elements() const
  {
    return element_data;
  }

  const InteriorFaceData<dim, RealType>& get_interior_faces() const
  {
    return interior_face_data;
  }

  const BoundaryFaceData<dim, RealType>& get_boundary_faces() const
  {
    return boundary_face_data;
  }

  const PeriodicFaceData<dim, RealType>& get_periodic_faces() const
  {
    return periodic_face_data;
  }

  std::size_t n_elements() const { return element_data.size(); }

  std::size_t n_interior_faces() const { return interior_face_data.size(); }

  std::size_t n_boundary_faces() const { return boundary_face_data.size(); }

  std::size_t n_periodic_faces() const { return periodic_face_data.size(); }

private:
  void compute_element_data(const MeshData& data)
  {
    // Resize
    element_data.resize(data.n_elements);

#pragma omp simd
    for (unsigned int i = 0; i < data.n_elements; ++i) {
      unsigned int v0 = data.node_1[i];
      unsigned int v1 = data.node_2[i];
      unsigned int v2 = data.node_3[i];

      // Get vertex coordinates
      RealType x0 = data.x[v0];
      RealType y0 = data.y[v0];
      RealType x1 = data.x[v1];
      RealType y1 = data.y[v1];
      RealType x2 = data.x[v2];
      RealType y2 = data.y[v2];

      // Compute area using cross product formula
      RealType dx1 = x1 - x0;
      RealType dy1 = y1 - y0;
      RealType dx2 = x2 - x0;
      RealType dy2 = y2 - y0;

      RealType area = RealType(0.5) * std::abs(dx1 * dy2 - dx2 * dy1);

      element_data.area[i] = area;
      element_data.inv_area[i] = RealType(1.0) / area;

      // Compute centroid
      element_data.centroid_x[i] = (x0 + x1 + x2) / RealType(3.0);
      element_data.centroid_y[i] = (y0 + y1 + y2) / RealType(3.0);
    }
  }

  void compute_interior_face_data(const MeshData& data)
  {
    // Resize
    interior_face_data.resize(_n_interior_faces);

    // Map for edge information. Vertex ids are the key and the data is the
    // element and local face.
    using EdgeKey = std::pair<unsigned int, unsigned int>;
    struct EdgeHasher
    {
      std::size_t operator()(const EdgeKey& edge) const
      {
        return std::hash<unsigned int>{}(edge.first) ^
               (std::hash<unsigned int>{}(edge.second) << 1);
      }
    };
    struct EdgeInfo
    {
      unsigned int element;
      unsigned int local_face;
    };
    std::unordered_map<EdgeKey, EdgeInfo, EdgeHasher> edge_map;

    // Helper function for sorting edge (smaller index first)
    auto make_edge = [](unsigned int v0, unsigned int v1) -> EdgeKey {
      return (v0 < v1) ? EdgeKey{ v0, v1 } : EdgeKey{ v1, v0 };
    };

    // Helper function to get the vertices of a local face for an element
    auto get_face_vertices = [&data](unsigned int elem, unsigned int local_face)
      -> std::pair<unsigned int, unsigned int> {
      unsigned int v0 = data.node_1[elem];
      unsigned int v1 = data.node_2[elem];
      unsigned int v2 = data.node_3[elem];

      // Assumes counter-clockwise ordering
      // Local face 0: edge v1-v2
      // Local face 1: edge v2-v0
      // Local face 2: edge v0-v1
      if (local_face == 0) {
        return { v1, v2 };
      }
      if (local_face == 1) {
        return { v2, v0 };
      }
      return { v0, v1 };
    };

    // Temporary storage for interior faces
    std::vector<unsigned int> temp_elem_l, temp_face_l;
    std::vector<unsigned int> temp_elem_r, temp_face_r;
    std::vector<unsigned int> temp_v0, temp_v1;

    // First find the interior edges
    for (unsigned int i = 0; i < data.n_elements; ++i) {
      unsigned int v0 = data.node_1[i];
      unsigned int v1 = data.node_2[i];
      unsigned int v2 = data.node_3[i];

      // Three edges of the triangle ordered by the local convention
      std::array<EdgeKey, 3> edges = { make_edge(v1, v2),
                                       make_edge(v2, v0),
                                       make_edge(v0, v1) };

      for (unsigned int j = 0; j < 3; ++j) {
        auto it = edge_map.find(edges[j]);

        // Store the edge if we haven't already seen it. If we have, it's an
        // interior face.
        if (it == edge_map.end()) {
          edge_map[edges[j]] = EdgeInfo{ i, j };
        } else {
          EdgeInfo& info = it->second;

          temp_elem_l.push_back(info.element);
          temp_face_l.push_back(info.local_face);
          temp_elem_r.push_back(i);
          temp_face_r.push_back(j);
          temp_v0.push_back(edges[j].first);
          temp_v1.push_back(edges[j].second);

          edge_map.erase(it);
        }
      }
    }

    // Check that the logic works out
    ASSERT(temp_elem_l.size() == _n_interior_faces);

    // Move data
    interior_face_data.elem_l = std::move(temp_elem_l);
    interior_face_data.face_l = std::move(temp_face_l);
    interior_face_data.elem_r = std::move(temp_elem_r);
    interior_face_data.face_r = std::move(temp_face_r);

    // Compute the other geometry information
#pragma omp simd
    for (unsigned int i = 0; i < _n_interior_faces; ++i) {
      unsigned int v0 = temp_v0[i];
      unsigned int v1 = temp_v1[i];

      // Get vertex coordinates
      RealType x0 = data.x[v0];
      RealType y0 = data.y[v0];
      RealType x1 = data.x[v1];
      RealType y1 = data.y[v1];

      // Compute the centroid
      interior_face_data.centroid_x[i] = RealType(0.5) * (x0 + x1);
      interior_face_data.centroid_y[i] = RealType(0.5) * (y0 + y1);

      // Compute the edge vector
      RealType dx = x1 - x0;
      RealType dy = y1 - y0;

      // Compute edge length
      RealType length = std::sqrt(dx * dx + dy * dy);
      interior_face_data.face_area[i] = length;

      // Compute the normal vector so that it points from left to right element
      RealType nx = dy / length;
      RealType ny = -dx / length;
      unsigned int e_l = interior_face_data.elem_l[i];
      unsigned int e_r = interior_face_data.elem_r[i];
      RealType cx_l = element_data.centroid_x[e_l];
      RealType cy_l = element_data.centroid_y[e_l];
      RealType cx_r = element_data.centroid_x[e_r];
      RealType cy_r = element_data.centroid_y[e_r];
      RealType dx_lr = cx_r - cx_l;
      RealType dy_lr = cy_r - cy_l;
      RealType dot = nx * dx_lr + ny * dy_lr;
      if (dot < RealType(0.0)) {
        nx = -nx;
        ny = -ny;
      }
      interior_face_data.normal_x[i] = nx;
      interior_face_data.normal_y[i] = ny;
    }
  }

  void compute_boundary_face_data(const MeshData& data)
  {
    // Resize
    boundary_face_data.resize(_n_boundary_faces);

    // Map for edge information. Vertex ids are the key and the data is the
    // element and local face.
    using EdgeKey = std::pair<unsigned int, unsigned int>;
    struct EdgeHasher
    {
      std::size_t operator()(const EdgeKey& edge) const
      {
        return std::hash<unsigned int>{}(edge.first) ^
               (std::hash<unsigned int>{}(edge.second) << 1);
      }
    };
    struct EdgeInfo
    {
      unsigned int element;
      unsigned int local_face;
    };
    std::unordered_map<EdgeKey, EdgeInfo, EdgeHasher> edge_map;

    // Helper function for sorting edge (smaller index first)
    auto make_edge = [](unsigned int v0, unsigned int v1) -> EdgeKey {
      return (v0 < v1) ? EdgeKey{ v0, v1 } : EdgeKey{ v1, v0 };
    };

    // Create a map of the edges
    for (unsigned int i = 0; i < data.n_elements; ++i) {
      unsigned int v0 = data.node_1[i];
      unsigned int v1 = data.node_2[i];
      unsigned int v2 = data.node_3[i];

      std::array<EdgeKey, 3> edges = { make_edge(v1, v2),
                                       make_edge(v2, v0),
                                       make_edge(v0, v1) };

      for (unsigned int j = 0; j < 3; ++j) {
        edge_map[edges[j]] = EdgeInfo{ i, j };
      }
    }

    // Build a set of periodic edges to exclude
    std::unordered_set<EdgeKey, EdgeHasher> periodic_edges;

    if (data.n_periodic_groups > 0) {
      // Create the periodic node mapping
      std::unordered_map<unsigned int, unsigned int> periodic_pair;
      for (unsigned int i = 0; i < data.periodic_node_1.size(); ++i) {
        periodic_pair[data.periodic_node_1[i]] = data.periodic_node_2[i];
      }

      // Identify periodic edges
      std::size_t linear_index = 0;
      for (unsigned int i = 0; i < data.n_boundary_groups; ++i) {
        for (unsigned int j = 0; j < data.boundary_group_n_faces[i]; ++j) {
          unsigned int v0 = data.boundary_node_1[linear_index];
          unsigned int v1 = data.boundary_node_2[linear_index];

          EdgeKey edge = make_edge(v0, v1);

          // Check if both vertices have periodic partners
          auto it0 = periodic_pair.find(v0);
          auto it1 = periodic_pair.find(v1);

          if (it0 != periodic_pair.end() && it1 != periodic_pair.end()) {
            unsigned int partner_v0 = it0->second;
            unsigned int partner_v1 = it1->second;

            EdgeKey partner_edge = make_edge(partner_v0, partner_v1);

            periodic_edges.insert(edge);
            periodic_edges.insert(partner_edge);
          }

          linear_index++;
        }
      }

      // Temporary storage for boundary faces
      std::vector<unsigned int> temp_elem;
      std::vector<unsigned int> temp_face;
      std::vector<unsigned int> temp_boundary_id;
      std::vector<unsigned int> temp_v0, temp_v1;

      linear_index = 0;
      for (unsigned int i = 0; i < data.n_boundary_groups; ++i) {
        for (unsigned int j = 0; j < data.boundary_group_n_faces[i]; ++j) {
          unsigned int v0 = data.boundary_node_1[linear_index];
          unsigned int v1 = data.boundary_node_2[linear_index];

          EdgeKey edge = make_edge(v0, v1);

          // Skip if this is a periodic edge
          if (periodic_edges.count(edge)) {
            linear_index++;
            continue;
          }

          // Add the data for the edge
          auto it = edge_map.find(edge);
          if (it != edge_map.end()) {
            EdgeInfo& info = it->second;

            temp_elem.push_back(info.element);
            temp_face.push_back(info.local_face);
            temp_boundary_id.push_back(i);
            temp_v0.push_back(v0);
            temp_v1.push_back(v1);
          }

          linear_index++;
        }
      }

      // Check that the logic works out
      ASSERT(temp_elem.size() == _n_boundary_faces);

      // Move data
      boundary_face_data.elem = std::move(temp_elem);
      boundary_face_data.face = std::move(temp_face);
      boundary_face_data.boundary_id = std::move(temp_boundary_id);

      // Compute the other geometry information
#pragma omp simd
      for (unsigned int i = 0; i < _n_boundary_faces; ++i) {
        unsigned int v0 = temp_v0[i];
        unsigned int v1 = temp_v1[i];

        // Get vertex coordinates
        RealType x0 = data.x[v0];
        RealType y0 = data.y[v0];
        RealType x1 = data.x[v1];
        RealType y1 = data.y[v1];

        // Compute the centroid
        boundary_face_data.centroid_x[i] = RealType(0.5) * (x0 + x1);
        boundary_face_data.centroid_y[i] = RealType(0.5) * (y0 + y1);

        // Compute the edge vector
        RealType dx = x1 - x0;
        RealType dy = y1 - y0;

        // Compute edge length
        RealType length = std::sqrt(dx * dx + dy * dy);
        boundary_face_data.face_area[i] = length;

        // Compute the normal vector so that it points outward
        RealType nx = dy / length;
        RealType ny = -dx / length;
        unsigned int e = boundary_face_data.elem[i];
        RealType cx = element_data.centroid_x[e];
        RealType cy = element_data.centroid_y[e];
        RealType dx_to_face = boundary_face_data.centroid_x[i] - cx;
        RealType dy_to_face = boundary_face_data.centroid_y[i] - cy;
        RealType dot = nx * dx_to_face + ny * dy_to_face;
        if (dot < RealType(0)) {
          nx = -nx;
          ny = -ny;
        }
        boundary_face_data.normal_x[i] = nx;
        boundary_face_data.normal_y[i] = ny;
      }
    }
  }

  void compute_periodic_face_data(const MeshData& data)
  {
    // Resize
    periodic_face_data.resize(_n_periodic_faces);

    if (_n_periodic_faces == 0) {
      return;
    }

    // Map for edge information. Vertex ids are the key and the data is the
    // element and local face.
    using EdgeKey = std::pair<unsigned int, unsigned int>;
    struct EdgeHasher
    {
      std::size_t operator()(const EdgeKey& edge) const
      {
        return std::hash<unsigned int>{}(edge.first) ^
               (std::hash<unsigned int>{}(edge.second) << 1);
      }
    };
    struct EdgeInfo
    {
      unsigned int element;
      unsigned int local_face;
    };
    std::unordered_map<EdgeKey, EdgeInfo, EdgeHasher> edge_map;

    // Helper function for sorting edge (smaller index first)
    auto make_edge = [](unsigned int v0, unsigned int v1) -> EdgeKey {
      return (v0 < v1) ? EdgeKey{ v0, v1 } : EdgeKey{ v1, v0 };
    };

    // Create a map of the edges
    for (unsigned int i = 0; i < data.n_elements; ++i) {
      unsigned int v0 = data.node_1[i];
      unsigned int v1 = data.node_2[i];
      unsigned int v2 = data.node_3[i];

      std::array<EdgeKey, 3> edges = { make_edge(v1, v2),
                                       make_edge(v2, v0),
                                       make_edge(v0, v1) };

      for (unsigned int j = 0; j < 3; ++j) {
        edge_map[edges[j]] = EdgeInfo{ i, j };
      }
    }

    // Create the periodic node mapping
    std::unordered_map<unsigned int, unsigned int> periodic_pair;
    for (unsigned int i = 0; i < data.periodic_node_1.size(); ++i) {
      periodic_pair[data.periodic_node_1[i]] = data.periodic_node_2[i];
    }

    // Temporary storage for periodic faces
    std::vector<unsigned int> temp_elem_l, temp_face_l;
    std::vector<unsigned int> temp_elem_r, temp_face_r;
    std::vector<unsigned int> temp_v0, temp_v1;
    std::vector<unsigned int> temp_partner_v0, temp_partner_v1;

    // Track the edges we've visited
    std::unordered_set<EdgeKey, EdgeHasher> visited_edges;
    std::size_t linear_index = 0;
    for (unsigned int i = 0; i < data.n_boundary_groups; ++i) {
      for (unsigned int j = 0; j < data.boundary_group_n_faces[i]; ++j) {
        unsigned int v0 = data.boundary_node_1[linear_index];
        unsigned int v1 = data.boundary_node_2[linear_index];

        EdgeKey edge = make_edge(v0, v1);

        // Skip if already processed
        if (visited_edges.count(edge)) {
          linear_index++;
          continue;
        }

        // Check if both vertices have periodic partners
        auto it0 = periodic_pair.find(v0);
        auto it1 = periodic_pair.find(v1);

        if (it0 != periodic_pair.end() && it1 != periodic_pair.end()) {
          unsigned int partner_v0 = it0->second;
          unsigned int partner_v1 = it1->second;

          EdgeKey partner_edge = make_edge(partner_v0, partner_v1);

          // Find element info for both edges
          auto edge_it = edge_map.find(edge);
          auto partner_it = edge_map.find(partner_edge);

          if (edge_it != edge_map.end() && partner_it != edge_map.end()) {
            temp_elem_l.push_back(edge_it->second.element);
            temp_face_l.push_back(edge_it->second.local_face);
            temp_elem_r.push_back(partner_it->second.element);
            temp_face_r.push_back(partner_it->second.local_face);
            temp_v0.push_back(v0);
            temp_v1.push_back(v1);
            temp_partner_v0.push_back(partner_v0);
            temp_partner_v1.push_back(partner_v1);

            // Mark both edges as visited
            visited_edges.insert(edge);
            visited_edges.insert(partner_edge);
          }
        }

        linear_index++;
      }
    }

    // Check that the logic works out
    ASSERT(temp_elem_l.size() == _n_periodic_faces);

    // Move data
    periodic_face_data.elem_l = std::move(temp_elem_l);
    periodic_face_data.face_l = std::move(temp_face_l);
    periodic_face_data.elem_r = std::move(temp_elem_r);
    periodic_face_data.face_r = std::move(temp_face_r);

    // Compute the other geometry information
#pragma omp simd
    for (unsigned int i = 0; i < _n_periodic_faces; ++i) {
      unsigned int v0 = temp_v0[i];
      unsigned int v1 = temp_v1[i];
      unsigned int v0_partner = temp_partner_v0[i];
      unsigned int v1_partner = temp_partner_v1[i];

      // Get vertex coordinates
      RealType x0 = data.x[v0];
      RealType y0 = data.y[v0];
      RealType x1 = data.x[v1];
      RealType y1 = data.y[v1];
      RealType x0_partner = data.x[v0_partner];
      RealType y0_partner = data.y[v0_partner];
      RealType x1_partner = data.x[v1_partner];
      RealType y1_partner = data.y[v1_partner];

      // Compute the centroid
      periodic_face_data.centroid_x[i] = RealType(0.5) * (x0 + x1);
      periodic_face_data.centroid_y[i] = RealType(0.5) * (y0 + y1);

      // Compute the translation vector from the two centroids
      RealType partner_cx = RealType(0.5) * (x0_partner + x1_partner);
      RealType partner_cy = RealType(0.5) * (y0_partner + y1_partner);
      periodic_face_data.translation_x[i] =
        partner_cx - periodic_face_data.centroid_x[i];
      periodic_face_data.translation_y[i] =
        partner_cy - periodic_face_data.centroid_y[i];

      // Compute the edge vector
      RealType dx = x1 - x0;
      RealType dy = y1 - y0;

      // Compute edge length
      RealType length = std::sqrt(dx * dx + dy * dy);
      periodic_face_data.face_area[i] = length;

      // Compute the normal vector so that it points from left to right element
      RealType nx = dy / length;
      RealType ny = -dx / length;
      unsigned int e_l = periodic_face_data.elem_l[i];
      RealType cx_l = element_data.centroid_x[e_l];
      RealType cy_l = element_data.centroid_y[e_l];
      RealType dx_to_face = periodic_face_data.centroid_x[i] - cx_l;
      RealType dy_to_face = periodic_face_data.centroid_y[i] - cy_l;
      RealType dot = nx * dx_to_face + ny * dy_to_face;
      if (dot < RealType(0)) {
        nx = -nx;
        ny = -ny;
      }
      periodic_face_data.normal_x[i] = nx;
      periodic_face_data.normal_y[i] = ny;
    }
  }

  unsigned int _n_boundary_faces = 0;
  unsigned int _n_interior_faces = 0;
  unsigned int _n_periodic_faces = 0;

  ElementData<dim, RealType> element_data;
  InteriorFaceData<dim, RealType> interior_face_data;
  BoundaryFaceData<dim, RealType> boundary_face_data;
  PeriodicFaceData<dim, RealType> periodic_face_data;
};

/**
 * @brief For mesh verification, we simply take the sum of the normal vectors
 * multiplied with boundary area at each element. The l2-norm of this error is
 * returned at the end.
 */
template<unsigned int dim, typename RealType>
RealType
mesh_verification()
{
}

template<unsigned int dim>
class GriReader
{
public:
  static_assert(dim == 2, "Only 2D meshes are supported");

  GriReader() = default;

  void read_gri(const std::string& filename);

  const MeshData& get_data() { return data; }

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
