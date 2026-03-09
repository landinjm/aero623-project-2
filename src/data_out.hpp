#pragma once

#include <dof_handler.hpp>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector.hpp>
#include <vector>

template<unsigned int dim>
class DataOut
{
public:
  using Topo = SimplexTopology<dim>;

private:
  struct DataEntry
  {
    std::string name;
    std::vector<double> data;
  };

  const Triangulation<dim>* tria_ = nullptr;
  std::vector<DataEntry> data_entries_;
  double time_ = 0.0;
  unsigned int cycle_ = 0;
  bool has_time_ = false;

public:
  void attach_triangulation(const Triangulation<dim>& tria) { tria_ = &tria; }
  void set_time(double t)
  {
    time_ = t;
    has_time_ = true;
  }
  void set_cycle(unsigned int c) { cycle_ = c; }

  void clear()
  {
    data_entries_.clear();
    tria_ = nullptr;
    has_time_ = false;
    time_ = 0.0;
    cycle_ = 0;
  }

  template<typename RealType, typename MemorySpace>
  void add_data_vector(const Vector<RealType, MemorySpace>& vec,
                       const std::string& name)
  {
    Vector<RealType, HostMemSpace> host_vec(vec.size());
    host_vec.import(vec, VectorOperation::insert);

    DataEntry e;
    e.name = name;
    e.data.resize(vec.size());
    for (size_t i = 0; i < vec.size(); ++i)
      e.data[i] = static_cast<double>(host_vec[i]);

    data_entries_.push_back(std::move(e));
  }

  void write_vtu(const std::string& filename) const
  {
    if (!tria_)
      throw std::runtime_error("DataOut: no triangulation attached");

    std::ofstream out(filename);
    if (!out.is_open())
      throw std::runtime_error("DataOut: cannot open " + filename);

    auto active = get_active_cells();
    const uint32_t n_pts = tria_->n_vertices();
    const uint32_t n_cells = static_cast<uint32_t>(active.size());

    write_header(out);
    write_piece_open(out, n_pts, n_cells);
    write_points(out);
    write_cells(out, active);
    write_cell_data(out, active);
    write_piece_close(out);
    write_footer(out);
  }

private:
  std::vector<CellIndexType> get_active_cells() const
  {
    std::vector<CellIndexType> active;
    active.reserve(tria_->n_active_cells());
    for (auto cell : tria_->active_cell_range())
      active.push_back(cell.index);
    return active;
  }

  static constexpr uint8_t vtk_cell_type()
  {
    if constexpr (dim == 1)
      return 3;
    if constexpr (dim == 2)
      return 5;
    if constexpr (dim == 3)
      return 10;
  }

  void write_header(std::ofstream& out) const
  {
    out << "<?xml version=\"1.0\"?>\n";
    // Embed time as field data if set
    out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" "
        << "byte_order=\"LittleEndian\">\n"
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

  void write_piece_open(std::ofstream& out,
                        uint32_t n_pts,
                        uint32_t n_cells) const
  {
    out << "    <Piece NumberOfPoints=\"" << n_pts << "\" NumberOfCells=\""
        << n_cells << "\">\n";
  }

  void write_piece_close(std::ofstream& out) const { out << "    </Piece>\n"; }

  void write_points(std::ofstream& out) const
  {
    out << "      <Points>\n"
        << "        <DataArray type=\"Float64\""
        << " NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (uint32_t i = 0; i < tria_->n_vertices(); ++i) {
      for (int d = 0; d < dim; ++d)
        out << tria_->vertices(i, d) << " ";
      for (int d = dim; d < 3; ++d)
        out << "0.0 ";
      out << "\n";
    }
    out << "        </DataArray>\n"
        << "      </Points>\n";
  }

  void write_cells(std::ofstream& out,
                   const std::vector<CellIndexType>& active) const
  {
    out << "      <Cells>\n";

    out << "        <DataArray type=\"UInt32\" Name=\"connectivity\" "
           "format=\"ascii\">\n";
    for (CellIndexType c : active) {
      for (uint8_t v = 0; v < Topo::verts_per_cell; ++v)
        out << tria_->cell_verts(c, v) << " ";
      out << "\n";
    }
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"UInt32\" Name=\"offsets\" "
           "format=\"ascii\">\n";
    uint32_t offset = 0;
    for (size_t i = 0; i < active.size(); ++i) {
      offset += Topo::verts_per_cell;
      out << offset << " ";
    }
    out << "\n        </DataArray>\n";

    out
      << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (size_t i = 0; i < active.size(); ++i)
      out << static_cast<int>(vtk_cell_type()) << " ";
    out << "\n        </DataArray>\n";

    out << "      </Cells>\n";
  }

  void write_cell_data(std::ofstream& out,
                       const std::vector<CellIndexType>& active) const
  {
    // Always write built-in topology data
    out << "      <CellData>\n";

    out << "        <DataArray type=\"UInt32\" Name=\"level\" "
           "format=\"ascii\">\n";
    for (CellIndexType c : active)
      out << tria_->cell_level(c) << " ";
    out << "\n        </DataArray>\n";

    out << "        <DataArray type=\"UInt8\" Name=\"material_id\" "
           "format=\"ascii\">\n";
    for (CellIndexType c : active)
      out << static_cast<int>(tria_->material_ids(c)) << " ";
    out << "\n        </DataArray>\n";

    out << "        <DataArray type=\"UInt8\" Name=\"boundary_id\" "
           "format=\"ascii\">\n";
    for (CellIndexType c : active) {
      uint8_t bid = 255;
      for (uint8_t lf = 0; lf < Topo::faces_per_cell; ++lf) {
        FaceIndexType fi = tria_->cell_faces(c, lf);
        if (tria_->face_flags(fi) & FaceFlags::Boundary) {
          bid = tria_->boundary_ids(fi);
          break;
        }
      }
      out << static_cast<int>(bid) << " ";
    }
    out << "\n        </DataArray>\n";

    // User-attached cell data
    for (auto& e : data_entries_) {
      out << "        <DataArray type=\"Float64\" Name=\"" << e.name
          << "\" format=\"ascii\">\n";
      for (CellIndexType c : active)
        out << e.data[c] << " ";
      out << "\n        </DataArray>\n";
    }

    out << "      </CellData>\n";
  }
};
