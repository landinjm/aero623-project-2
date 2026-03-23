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
    ASSERT(!entries_.empty(), "No data vectors added");

    std::ofstream out(filename);
    ASSERT(out.is_open(), "Could not open output file");

    const uint32_t n_cells = dh_->n_cells();
    const uint32_t ndpc = dh_->n_dofs_per_cell();
    const uint32_t p = dh_->fe().degree();
    const uint32_t verts_per_cell = Topo::verts_per_cell;

    std::vector<uint32_t> local_dof_indices;

    write_header(out);

    if (p == 0) {
      // Degree 0: one DOF per cell, use vertices for geometry and CellData
      const uint32_t n_points = n_cells * verts_per_cell;
      out << "    <Piece NumberOfPoints=\"" << n_points << "\" NumberOfCells=\""
          << n_cells << "\">\n";

      // Points
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

      // Cells
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

      // Cell data
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

    } else {
      // Degree >= 1: subdivide into p^2 linear sub-triangles
      std::vector<std::array<double, dim>> ref_points;
      for (unsigned int j = 0; j <= p; ++j)
        for (unsigned int i = 0; i <= p - j; ++i)
          ref_points.push_back(
            { static_cast<double>(i) / p, static_cast<double>(j) / p });

      const uint32_t n_ref_points = ref_points.size();

      auto idx = [&](unsigned int i, unsigned int j) -> uint32_t {
        uint32_t start = j * (p + 1) - j * (j - 1) / 2;
        return start + i;
      };

      std::vector<std::array<uint32_t, 3>> ref_tris;
      for (unsigned int j = 0; j < p; ++j) {
        for (unsigned int i = 0; i < p - j; ++i) {
          ref_tris.push_back({ idx(i, j), idx(i + 1, j), idx(i, j + 1) });
          if (i + 1 + j + 1 <= p)
            ref_tris.push_back(
              { idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1) });
        }
      }

      const uint32_t n_sub_tris = ref_tris.size();
      const uint32_t total_points = n_cells * n_ref_points;
      const uint32_t total_cells = n_cells * n_sub_tris;

      out << "    <Piece NumberOfPoints=\"" << total_points
          << "\" NumberOfCells=\"" << total_cells << "\">\n";

      // Points
      out << "      <Points>\n"
          << "        <DataArray type=\"Float64\""
          << " NumberOfComponents=\"3\" format=\"ascii\">\n";
      for (auto cell : dh_->active_cell_range()) {
        double x0[dim], J[dim][dim];
        for (unsigned int d = 0; d < dim; ++d)
          x0[d] = cell.vertex(0)(d);
        for (unsigned int d = 0; d < dim; ++d) {
          J[d][0] = cell.vertex(1)(d) - cell.vertex(0)(d);
          J[d][1] = cell.vertex(2)(d) - cell.vertex(0)(d);
        }
        for (const auto& xi_arr : ref_points) {
          for (unsigned int d = 0; d < dim; ++d) {
            double xd = x0[d];
            for (unsigned int e = 0; e < dim; ++e)
              xd += J[d][e] * xi_arr[e];
            out << xd << " ";
          }
          for (unsigned int d = dim; d < 3; ++d)
            out << "0.0 ";
          out << "\n";
        }
      }
      out << "        </DataArray>\n"
          << "      </Points>\n";

      // Cells
      out << "      <Cells>\n";
      out << "        <DataArray type=\"UInt32\" Name=\"connectivity\""
          << " format=\"ascii\">\n";
      for (uint32_t k = 0; k < n_cells; ++k)
        for (const auto& tri : ref_tris)
          out << k * n_ref_points + tri[0] << " " << k * n_ref_points + tri[1]
              << " " << k * n_ref_points + tri[2] << "\n";
      out << "        </DataArray>\n";
      out << "        <DataArray type=\"UInt32\" Name=\"offsets\""
          << " format=\"ascii\">\n";
      for (uint32_t k = 0; k < total_cells; ++k)
        out << (k + 1) * 3 << " ";
      out << "\n        </DataArray>\n";
      out << "        <DataArray type=\"UInt8\" Name=\"types\""
          << " format=\"ascii\">\n";
      for (uint32_t k = 0; k < total_cells; ++k)
        out << "5 ";
      out << "\n        </DataArray>\n";
      out << "      </Cells>\n";

      // Point data
      out << "      <PointData>\n";
      for (const auto& e : entries_) {
        out << "        <DataArray type=\"Float64\" Name=\"" << e.name
            << "\" format=\"ascii\">\n";
        for (auto cell : dh_->active_cell_range()) {
          cell.get_dof_indices(local_dof_indices);
          for (const auto& xi_arr : ref_points) {
            Tensor<1, dim, double> xi;
            for (unsigned int d = 0; d < dim; ++d)
              xi(d) = xi_arr[d];
            double val = 0.0;
            for (unsigned int i = 0; i < ndpc; ++i)
              val +=
                e.data[local_dof_indices[i]] * dh_->fe().shape_value(i, xi);
            out << val << "\n";
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

  const Triangulation<dim>* tria_ = nullptr;
  const DoFHandler<dim, double>* dh_ = nullptr;

  std::vector<DataEntry> entries_;
  double time_ = 0.0;
  unsigned int cycle_ = 0;
  bool has_time_ = false;

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
