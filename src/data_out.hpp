#pragma once

#include <Kokkos_Core.hpp>
#include <dof_handler.hpp>
#include <fstream>
#include <stdexcept>
#include <string>
#include <triangulation.hpp>
#include <vector.hpp>
#include <vector>

template<unsigned int dim>
class DataOut
{
public:
  using Topo = SimplexTopology<dim>;

  void attach_dof_handler(const DoFHandler<dim, double>& dh)
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
    ASSERT(tria_ != nullptr, "No Triangulation attached");
    ASSERT(!entries_.empty(), "No data vectors added");

    std::ofstream out(filename);
    ASSERT(out.is_open(), "Could not open output file");

    const uint32_t n_cells = dh_->n_cells();
    const uint32_t ndpc = dh_->n_dofs_per_cell();
    // For degree 0: write 3 vertices per cell as points
    const uint32_t verts_per_cell = Topo::verts_per_cell;
    const uint32_t n_points = n_cells * verts_per_cell;

    write_header(out);
    out << "    <Piece NumberOfPoints=\"" << n_points << "\" NumberOfCells=\""
        << n_cells << "\">\n";

    // Points: use actual cell vertices
    out << "      <Points>\n"
        << "        <DataArray type=\"Float64\""
        << " NumberOfComponents=\"3\" format=\"ascii\">\n";
    std::vector<uint32_t> local_dof_indices;
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

    // Cells: one VTK_TRIANGLE per DG cell
    out << "      <Cells>\n";

    out << "        <DataArray type=\"UInt32\" Name=\"connectivity\""
        << " format=\"ascii\">\n";
    for (uint32_t k = 0; k < n_cells; ++k)
      out << k * verts_per_cell + 0 << " " << k * verts_per_cell + 1 << " "
          << k * verts_per_cell + 2 << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"UInt32\" Name=\"offsets\""
        << " format=\"ascii\">\n";
    for (uint32_t k = 0; k < n_cells; ++k)
      out << (k + 1) * verts_per_cell << " ";
    out << "\n        </DataArray>\n";

    out << "        <DataArray type=\"UInt8\" Name=\"types\""
        << " format=\"ascii\">\n";
    for (uint32_t k = 0; k < n_cells; ++k)
      out << "5 ";
    out << "\n        </DataArray>\n";

    out << "      </Cells>\n";

    // Cell data: one value per cell (constant per element for degree 0)
    out << "      <CellData>\n";
    for (const auto& e : entries_) {
      out << "        <DataArray type=\"Float64\" Name=\"" << e.name
          << "\" format=\"ascii\">\n";
      for (auto cell : dh_->active_cell_range()) {
        cell.get_dof_indices(local_dof_indices);
        // degree 0: one DOF per cell
        out << e.data[local_dof_indices[0]] << "\n";
      }
      out << "        </DataArray>\n";
    }
    out << "      </CellData>\n";

    write_piece_close(out);
    write_footer(out);
  }

private:
  struct DataEntry
  {
    std::string name;
    Vector<double, HostMemSpace> data;
  };

  const Triangulation<dim>* tria_ = nullptr;
  const DoFHandler<dim, double>* dh_ = nullptr;

  std::vector<DataEntry> entries_;
  double time_ = 0.0;
  unsigned int cycle_ = 0;
  bool has_time_ = false;

  void write_dg_points(std::ofstream& out, uint32_t ndpc) const
  {
    const auto& fe = dh_->fe();

    out << "      <Points>\n"
        << "        <DataArray type=\"Float64\""
        << " NumberOfComponents=\"3\" format=\"ascii\">\n";

    std::vector<uint32_t> local_dof_indices;
    for (auto cell : dh_->active_cell_range()) {
      cell.get_dof_indices(local_dof_indices);
      auto jac = cell_jacobian(cell);

      for (uint32_t i = 0; i < ndpc; ++i) {
        auto x_phys = map_to_physical(cell, jac, fe.node(i));
        for (unsigned int d = 0; d < dim; ++d)
          out << x_phys(d) << " ";
        for (unsigned int d = dim; d < 3; ++d)
          out << "0.0 ";
        out << "\n";
      }
    }

    out << "        </DataArray>\n"
        << "      </Points>\n";
  }

  void write_dg_cells(std::ofstream& out, uint32_t ndpc, uint32_t n_cells) const
  {
    out << "      <Cells>\n";

    out << "        <DataArray type=\"UInt32\" Name=\"connectivity\""
        << " format=\"ascii\">\n";
    for (uint32_t k = 0; k < n_cells; ++k) {
      for (uint32_t i = 0; i < ndpc; ++i)
        out << k * ndpc + i << " ";
      out << "\n";
    }
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"UInt32\" Name=\"offsets\""
        << " format=\"ascii\">\n";
    for (uint32_t k = 0; k < n_cells; ++k)
      out << (k + 1) * ndpc << " ";
    out << "\n        </DataArray>\n";

    out << "        <DataArray type=\"UInt8\" Name=\"types\""
        << " format=\"ascii\">\n";
    const int vtype = static_cast<int>(vtk_cell_type_for_dofs(ndpc));
    for (uint32_t k = 0; k < n_cells; ++k)
      out << vtype << " ";
    out << "\n        </DataArray>\n";

    out << "      </Cells>\n";
  }

  void write_dg_point_data(std::ofstream& out, uint32_t ndpc) const
  {
    out << "      <PointData>\n";

    std::vector<uint32_t> local_dof_indices;
    for (const auto& e : entries_) {
      out << "        <DataArray type=\"Float64\" Name=\"" << e.name
          << "\" format=\"ascii\">\n";
      for (auto cell : dh_->active_cell_range()) {
        cell.get_dof_indices(local_dof_indices);
        for (uint32_t i = 0; i < ndpc; ++i)
          out << e.data[local_dof_indices[i]] << " ";
        out << "\n";
      }
      out << "        </DataArray>\n";
    }

    out << "      </PointData>\n";
  }

  template<typename CellAcc>
  std::array<Tensor<1, dim, double>, dim> cell_jacobian(
    const CellAcc& cell) const
  {
    std::array<Tensor<1, dim, double>, dim> jac;
    auto x0 = cell.vertex(0);
    for (unsigned int d = 0; d < dim; ++d) {
      auto xd = cell.vertex(d + 1);
      for (unsigned int d2 = 0; d2 < dim; ++d2)
        jac[d](d2) = xd[d2] - x0[d2];
    }
    return jac;
  }

  template<typename CellAcc>
  Tensor<1, dim, double> map_to_physical(
    const CellAcc& cell,
    const std::array<Tensor<1, dim, double>, dim>& jac,
    const Tensor<1, dim, double>& x_ref) const
  {
    auto x0 = cell.vertex(0);
    Tensor<1, dim, double> x_phys;
    for (unsigned int d = 0; d < dim; ++d)
      x_phys(d) = x0[d];
    for (unsigned int d2 = 0; d2 < dim; ++d2)
      for (unsigned int d = 0; d < dim; ++d)
        x_phys(d) += jac[d2](d) * x_ref(d2);
    return x_phys;
  }

  static uint8_t vtk_cell_type_for_dofs(uint32_t ndpc)
  {
    if constexpr (dim == 1)
      return 3;
    if constexpr (dim == 2) {
      switch (ndpc) {
        case 3:
          return 5; // VTK_TRIANGLE
        case 6:
          return 22; // VTK_QUADRATIC_TRIANGLE
        case 10:
          return 69; // VTK_LAGRANGE_TRIANGLE (VTK 9+)
        default:
          return 7; // VTK_POLYGON
      }
    }
    if constexpr (dim == 3)
      return 10;
    return 7;
  }

  void write_header(std::ofstream& out) const
  {
    out << "<?xml version=\"1.0\"?>\n"
        << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\""
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
