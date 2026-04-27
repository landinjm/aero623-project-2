#pragma once

#include <Kokkos_Core.hpp>
#include <dof_handler.hpp>
#include <fstream>
#include <stdexcept>
#include <string>
#include <triangulation.hpp>
#include <vector.hpp>
#include <vector>

template<unsigned int dim, unsigned int q>
class DataOut
{
public:
  using Topo = SimplexTopology<dim, q>;

  void attach_dof_handler(const DoFHandler<dim, q, double>& dh)
  {
    dh_ = &dh;
    tria_ = &dh.tria();
  }

  void set_time(double t)
  {
    time_ = t;
    has_time_ = true;
  }
  void set_cycle(unsigned int c) { cycle_ = c; }

  void clear()
  {
    entries_.clear();
    tria_ = nullptr;
    dh_ = nullptr;
    has_time_ = false;
    time_ = 0.0;
    cycle_ = 0;
  }

  template<typename RealType>
  void add_data_vector(const Vector<RealType, HostMemSpace>& vec,
                       const std::string& name)
  {
    ASSERT(dh_ != nullptr,
           "attach_dof_handler must be called before add_dof_data");
    ASSERT(vec.size() == dh_->n_dofs(), "Vector length must match n_dofs");
    entries_.emplace_back(name, vec);
  }

  void write_vtu(const std::string& filename) const
  {
    ASSERT(dh_ != nullptr, "No DoFHandler attached");
    ASSERT(!entries_.empty(), "No data vectors added");

    std::ofstream out(filename);
    ASSERT(out.is_open(), "Could not open output file");

    const auto n_cells = dh_->n_cells();
    const auto n_dofs_per_cell = dh_->n_dofs_per_cell();
    const auto p = dh_->fe().degree();
    const auto verts_per_cell = Topo::verts_per_cell;

    std::vector<uint32_t> local_dof_indices;

    write_header(out);

    // Special case for p = 0. These have cell average values that must be
    // distributed to the vertices
    if (p == 0) {
      // Use the corresponding VTK_CELL_TYPE for triangles and tetrahedrals
      const uint8_t cell_type = (verts_per_cell == 3   ? 5
                                 : verts_per_cell == 4 ? 10
                                                       : 0);

      // Grab the number of points and fill them out. Use a 0.0 placeholder for
      // the extraneous dimensions
      const uint32_t n_points = n_cells * verts_per_cell;
      out << "    <Piece NumberOfPoints=\"" << n_points << "\" NumberOfCells=\""
          << n_cells << "\">\n";
      out << "      <Points>\n"
          << "        <DataArray type=\"Float64\""
          << " NumberOfComponents=\"3\" format=\"ascii\">\n";
      for (auto cell : dh_->active_cell_range()) {
        for (unsigned int v = 0; v < verts_per_cell; ++v) {
          auto x = cell.vertex(v);
          for (unsigned int d = 0; d < dim; ++d)
            out << x(d) << " ";
          for (unsigned int d = dim; d < 3; ++d)
            out << "0.0 ";
          out << "\n";
        }
      }
      out << "        </DataArray>\n"
          << "      </Points>\n";

      // Fill out the cell connectivities
      out << "      <Cells>\n";
      out << "        <DataArray type=\"UInt32\" Name=\"connectivity\""
          << " format=\"ascii\">\n";
      for (uint32_t k = 0; k < n_cells; ++k) {
        for (uint32_t v = 0; v < verts_per_cell; ++v)
          out << k * verts_per_cell + v << " ";
        out << "\n";
      }
      out << "        </DataArray>\n";
      out << "        <DataArray type=\"UInt32\" Name=\"offsets\""
          << " format=\"ascii\">\n";
      for (uint32_t k = 0; k < n_cells; ++k)
        out << (k + 1) * verts_per_cell << " ";
      out << "\n        </DataArray>\n";
      out << "        <DataArray type=\"UInt8\" Name=\"types\""
          << " format=\"ascii\">\n";

      for (uint32_t k = 0; k < n_cells; ++k)
        out << static_cast<unsigned int>(cell_type) << " ";
      out << "\n        </DataArray>\n";
      out << "      </Cells>\n";

      // Fill out the data to
      out << "      <CellData>\n";
      for (const auto& e : entries_) {
        out << "        <DataArray type=\"Float64\" Name=\"" << e.name
            << "\" format=\"ascii\">\n";
        for (auto cell : dh_->active_cell_range()) {
          cell.get_dof_indices(local_dof_indices);
          out << e.data[local_dof_indices[0]] << "\n";
        }
        out << "        </DataArray>\n";
      }
      out << "      </CellData>\n";
    }
    // For higher degrees, we will subdivide into p^dim linear sub-triangles
    else {
      // Use the corresponding VTK_CELL_TYPE for triangles and tetrahedrals
      const uint8_t cell_type = (verts_per_cell == 3   ? 69
                                 : verts_per_cell == 4 ? 71
                                                       : 0);

      // Compute the vtk ordering
      const auto vtk_perm = compute_vtk_permutation(dh_->fe());

      QGaussSimplex<dim, double> quad(p + 1);
      FEValues<dim, q, double> fe_values(dh_->fe(), quad);

      // Grab the number of points and fill them out. Use a 0.0 placeholder for
      // the extraneous dimensions
      const uint32_t n_points = n_cells * n_dofs_per_cell;
      out << "    <Piece NumberOfPoints=\"" << n_points << "\" NumberOfCells=\""
          << n_cells << "\">\n";
      out << "      <Points>\n"
          << "        <DataArray type=\"Float64\""
          << " NumberOfComponents=\"3\" format=\"ascii\">\n";
      for (auto cell : dh_->active_cell_range()) {
        for (unsigned int v = 0; v < n_dofs_per_cell; ++v) {
          auto xi = dh_->fe().node(vtk_perm[v]);
          for (unsigned int d = 0; d < dim; ++d) {
            double xd = cell.vertex(0)(d);
            for (unsigned int k = 0; k < dim; ++k)
              xd += (cell.vertex(k + 1)(d) - cell.vertex(0)(d)) * xi(k);
            out << xd << " ";
          }

          for (unsigned int d = dim; d < 3; ++d)
            out << "0.0 ";
          out << "\n";
        }
      }
      out << "        </DataArray>\n"
          << "      </Points>\n";

      // Fill out the cell connectivities
      out << "      <Cells>\n";
      out << "        <DataArray type=\"UInt32\" Name=\"connectivity\""
          << " format=\"ascii\">\n";
      for (uint32_t k = 0; k < n_cells; ++k) {
        for (uint32_t v = 0; v < n_dofs_per_cell; ++v)
          out << k * n_dofs_per_cell + v << " ";
        out << "\n";
      }
      out << "        </DataArray>\n";
      out << "        <DataArray type=\"UInt32\" Name=\"offsets\""
          << " format=\"ascii\">\n";
      for (uint32_t k = 0; k < n_cells; ++k)
        out << (k + 1) * n_dofs_per_cell << " ";
      out << "\n        </DataArray>\n";
      out << "        <DataArray type=\"UInt8\" Name=\"types\""
          << " format=\"ascii\">\n";

      for (uint32_t k = 0; k < n_cells; ++k)
        out << static_cast<unsigned int>(cell_type) << " ";
      out << "\n        </DataArray>\n";
      out << "      </Cells>\n";

      // Fill out the data to
      out << "      <PointData>\n";
      for (const auto& e : entries_) {
        out << "        <DataArray type=\"Float64\" Name=\"" << e.name
            << "\" format=\"ascii\">\n";
        for (auto cell : dh_->active_cell_range()) {
          cell.get_dof_indices(local_dof_indices);
          for (unsigned int v = 0; v < n_dofs_per_cell; ++v) {
            out << e.data[local_dof_indices[vtk_perm[v]]] << "\n";
          }
        }
        out << "        </DataArray>\n";
      }
      out << "      </PointData>\n";
    }

    write_piece_close(out);
    write_footer(out);
  }

private:
  struct DataEntry
  {
    std::string name;
    Vector<double, HostMemSpace> data;
  };

  const Triangulation<dim, q>* tria_ = nullptr;
  const DoFHandler<dim, q, double>* dh_ = nullptr;

  std::vector<DataEntry> entries_;
  double time_ = 0.0;
  unsigned int cycle_ = 0;
  bool has_time_ = false;

  void write_header(std::ofstream& out) const
  {
    out << "<?xml version=\"1.0\"?>\n"
        << "<VTKFile type=\"UnstructuredGrid\" version=\"2.2\""
        << " byte_order=\"LittleEndian\">\n"
        << "  <UnstructuredGrid>\n";
    if (has_time_) {
      out << "    <FieldData>\n"
          << "      <DataArray type=\"Float64\" Name=\"TimeValue\""
          << " NumberOfTuples=\"1\" format=\"ascii\">\n"
          << "        " << time_ << "\n"
          << "      </DataArray>\n"
          << "    </FieldData>\n";
    }
  }

  void write_footer(std::ofstream& out) const
  {
    out << "  </UnstructuredGrid>\n"
        << "</VTKFile>\n";
  }

  void write_piece_close(std::ofstream& out) const { out << "    </Piece>\n"; }
};

// In your VTK writer class or alongside FE_DGLagrangeSimplex,
// compute a permutation from your FE node ordering → VTK Lagrange ordering.
//
// VTK Lagrange simplex node ordering (arbitrary degree):
//   2D (triangle, type 69):
//     - dim+1 corners first, in VTK vertex order
//     - then edge midpoints, then interior nodes
//   3D (tet, type 71): similar
//
// Rather than hardcoding, we match by coordinate: for each VTK-expected
// support point location, find the index in your FE that has that location.

template<unsigned int dim, typename RealType>
std::vector<unsigned int>
compute_vtk_permutation(const FE_DGLagrangeSimplex<dim, RealType>& fe)
{
  const unsigned int n = fe.n_dofs();
  const unsigned int p = fe.degree();

  // Build VTK's expected node list in order, as (i0,i1[,i2]) multi-indices.
  // VTK Lagrange triangle: corners, then edge nodes, then interior.
  // We reconstruct VTK's ordering by enumerating the same equally-spaced
  // grid but in VTK's prescribed sequence.

  // VTK corner ordering on reference simplex:
  // 2D: (0,0), (p,0), (0,p)
  // 3D: (0,0,0), (p,0,0), (0,p,0), (0,0,p)
  // These are in VTK's vertex index order.

  // We'll build the list of (i0,i1[,i2]) that VTK expects, then look up
  // each one in your FE's node list.

  // Helper: find your FE node index whose coordinate matches a given
  // reference point (within tolerance).
  auto find_node =
    [&](const std::array<RealType, dim>& target) -> unsigned int {
    for (unsigned int i = 0; i < n; ++i) {
      auto xi = fe.node(i);
      bool match = true;
      for (unsigned int d = 0; d < dim; ++d) {
        if (std::abs(xi(d) - target[d]) > 1e-10) {
          match = false;
          break;
        }
      }
      if (match)
        return i;
    }
    ASSERT(false, "VTK permutation: node not found");
    return n;
  };

  // Build VTK's node sequence as reference coordinates.
  // VTK Lagrange triangle/tet uses the same equally-spaced grid as your FE,
  // but orders them as: corners → edges → faces → interior (each
  // sub-entity scanned in a specific order).
  //
  // We enumerate them explicitly below.
  std::vector<std::array<RealType, dim>> vtk_order;
  vtk_order.reserve(n);

  if constexpr (dim == 2) {
    // VTK corners (in VTK vertex order for a triangle)
    vtk_order.push_back({ RealType(0), RealType(0) }); // v0
    vtk_order.push_back({ RealType(1), RealType(0) }); // v1  (i0=p)
    vtk_order.push_back({ RealType(0), RealType(1) }); // v2  (i1=p)

    if (p >= 2) {
      // Edge v0→v1: i1=0, i0=1..p-1
      for (unsigned int i0 = 1; i0 < p; ++i0)
        vtk_order.push_back({ RealType(i0) / p, RealType(0) });

      // Edge v1→v2: i0+i1=p, i0=p-1..1 (i.e. i1 increasing)
      for (unsigned int i1 = 1; i1 < p; ++i1)
        vtk_order.push_back({ RealType(p - i1) / p, RealType(i1) / p });

      // Edge v0→v2: i0=0, i1=1..p-1
      for (unsigned int i1 = 1; i1 < p; ++i1)
        vtk_order.push_back({ RealType(0), RealType(i1) / p });
    }

    // Interior nodes: same row-major scan as your FE but skip boundary
    if (p >= 3) {
      for (unsigned int i1 = 1; i1 < p; ++i1)
        for (unsigned int i0 = 1; i0 < p - i1; ++i0)
          vtk_order.push_back({ RealType(i0) / p, RealType(i1) / p });
    }
  } else if constexpr (dim == 3) {
    // VTK corners for tet
    vtk_order.push_back({ RealType(0), RealType(0), RealType(0) }); // v0
    vtk_order.push_back({ RealType(1), RealType(0), RealType(0) }); // v1
    vtk_order.push_back({ RealType(0), RealType(1), RealType(0) }); // v2
    vtk_order.push_back({ RealType(0), RealType(0), RealType(1) }); // v3

    if (p >= 2) {
      // Edge v0→v1: i1=0,i2=0, i0=1..p-1
      for (unsigned int i0 = 1; i0 < p; ++i0)
        vtk_order.push_back({ RealType(i0) / p, RealType(0), RealType(0) });

      // Edge v1→v2: i0+i1=p, i2=0, i1=1..p-1
      for (unsigned int i1 = 1; i1 < p; ++i1)
        vtk_order.push_back(
          { RealType(p - i1) / p, RealType(i1) / p, RealType(0) });

      // Edge v0→v2: i0=0, i2=0, i1=1..p-1
      for (unsigned int i1 = 1; i1 < p; ++i1)
        vtk_order.push_back({ RealType(0), RealType(i1) / p, RealType(0) });

      // Edge v0→v3: i0=0, i1=0, i2=1..p-1
      for (unsigned int i2 = 1; i2 < p; ++i2)
        vtk_order.push_back({ RealType(0), RealType(0), RealType(i2) / p });

      // Edge v1→v3: i0+i2=p, i1=0, i2=1..p-1
      for (unsigned int i2 = 1; i2 < p; ++i2)
        vtk_order.push_back(
          { RealType(p - i2) / p, RealType(0), RealType(i2) / p });

      // Edge v2→v3: i1+i2=p, i0=0, i2=1..p-1
      for (unsigned int i2 = 1; i2 < p; ++i2)
        vtk_order.push_back(
          { RealType(0), RealType(p - i2) / p, RealType(i2) / p });
    }

    if (p >= 3) {
      // Face 0 (i1=0): scan i0=1..p-1, i2=1..p-1-i0
      for (unsigned int i0 = 1; i0 < p; ++i0)
        for (unsigned int i2 = 1; i2 < p - i0; ++i2)
          vtk_order.push_back(
            { RealType(i0) / p, RealType(0), RealType(i2) / p });

      // Face 1 (i0+i1+i2=p): scan i0=1..p-1, i1=1..p-1-i0
      for (unsigned int i0 = 1; i0 < p; ++i0)
        for (unsigned int i1 = 1; i1 < p - i0; ++i1)
          vtk_order.push_back(
            { RealType(i0) / p, RealType(i1) / p, RealType(p - i0 - i1) / p });

      // Face 2 (i0=0): scan i1=1..p-1, i2=1..p-1-i1
      for (unsigned int i1 = 1; i1 < p; ++i1)
        for (unsigned int i2 = 1; i2 < p - i1; ++i2)
          vtk_order.push_back(
            { RealType(0), RealType(i1) / p, RealType(i2) / p });

      // Face 3 (i2=0): scan i0=1..p-1, i1=1..p-1-i0
      for (unsigned int i0 = 1; i0 < p; ++i0)
        for (unsigned int i1 = 1; i1 < p - i0; ++i1)
          vtk_order.push_back(
            { RealType(i0) / p, RealType(i1) / p, RealType(0) });
    }

    // Volume interior
    if (p >= 4) {
      for (unsigned int i2 = 1; i2 < p; ++i2)
        for (unsigned int i1 = 1; i1 < p - i2; ++i1)
          for (unsigned int i0 = 1; i0 < p - i1 - i2; ++i0)
            vtk_order.push_back(
              { RealType(i0) / p, RealType(i1) / p, RealType(i2) / p });
    }
  }

  ASSERT(vtk_order.size() == n, "VTK node count mismatch");

  // Build permutation: perm[vtk_pos] = your_fe_index
  std::vector<unsigned int> perm(n);
  for (unsigned int v = 0; v < n; ++v)
    perm[v] = find_node(vtk_order[v]);

  return perm;
}
