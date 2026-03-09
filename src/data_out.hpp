#pragma once

#include <cstdint>
#include <dof_handler.hpp>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

template<unsigned int dim>
class DataOut
{
public:
  using Topo = SimplexTopology<dim>;

  static constexpr uint8_t vtk_cell_type()
  {
    if constexpr (dim == 1)
      return 3;
    if constexpr (dim == 2)
      return 5;
    if constexpr (dim == 3)
      return 10;
  }

  static void write(const Triangulation<dim>& tria, const std::string& filename)
  {
    std::ofstream out(filename);
    if (!out.is_open())
      throw std::runtime_error("VTUExporter: cannot open " + filename);

    // Collect active cell indices once
    std::vector<CellIndexType> active;
    active.reserve(tria.n_active_cells());
    for (auto cell : tria.active_cell_range())
      active.push_back(cell.index);

    const uint32_t n_pts = tria.n_vertices();
    const uint32_t n_cells = static_cast<uint32_t>(active.size());

    write_header(out);
    write_piece_open(out, n_pts, n_cells);
    write_points(out, tria);
    write_cells(out, tria, active);
    write_cell_data(out, tria, active);
    write_piece_close(out);
    write_footer(out);
  }

private:
  static void write_header(std::ofstream& out)
  {
    out << "<?xml version=\"1.0\"?>\n"
        << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" "
        << "byte_order=\"LittleEndian\">\n"
        << "  <UnstructuredGrid>\n";
  }

  static void write_footer(std::ofstream& out)
  {
    out << "  </UnstructuredGrid>\n"
        << "</VTKFile>\n";
  }

  static void write_piece_open(std::ofstream& out,
                               uint32_t n_pts,
                               uint32_t n_cells)
  {
    out << "    <Piece NumberOfPoints=\"" << n_pts << "\" NumberOfCells=\""
        << n_cells << "\">\n";
  }

  static void write_piece_close(std::ofstream& out) { out << "    </Piece>\n"; }

  static void write_points(std::ofstream& out, const Triangulation<dim>& tria)
  {
    out << "      <Points>\n"
        << "        <DataArray type=\"Float64\""
        << " NumberOfComponents=\"3\" format=\"ascii\">\n";

    for (uint32_t i = 0; i < tria.n_vertices(); ++i) {
      for (int d = 0; d < dim; ++d)
        out << tria.vertices(i, d) << " ";
      for (int d = dim; d < 3; ++d) // VTK always needs 3 components
        out << "0.0 ";
      out << "\n";
    }

    out << "        </DataArray>\n"
        << "      </Points>\n";
  }

  static void write_cells(std::ofstream& out,
                          const Triangulation<dim>& tria,
                          const std::vector<CellIndexType>& active)
  {
    out << "      <Cells>\n";

    // connectivity
    out << "        <DataArray type=\"UInt32\" Name=\"connectivity\""
        << " format=\"ascii\">\n";
    for (CellIndexType c : active) {
      for (uint8_t v = 0; v < Topo::verts_per_cell; ++v)
        out << tria.cell_verts(c, v) << " ";
      out << "\n";
    }
    out << "        </DataArray>\n";

    // offsets
    out << "        <DataArray type=\"UInt32\" Name=\"offsets\""
        << " format=\"ascii\">\n";
    uint32_t offset = 0;
    for (size_t i = 0; i < active.size(); ++i) {
      offset += Topo::verts_per_cell;
      out << offset << " ";
    }
    out << "\n        </DataArray>\n";

    // types — uniform, so just repeat
    out << "        <DataArray type=\"UInt8\" Name=\"types\""
        << " format=\"ascii\">\n";
    for (size_t i = 0; i < active.size(); ++i)
      out << static_cast<int>(vtk_cell_type()) << " ";
    out << "\n        </DataArray>\n";

    out << "      </Cells>\n";
  }

  static void write_cell_data(std::ofstream& out,
                              const Triangulation<dim>& tria,
                              const std::vector<CellIndexType>& active)
  {
    out << "      <CellData>\n";

    out << "        <DataArray type=\"UInt32\" Name=\"level\""
        << " format=\"ascii\">\n";
    for (CellIndexType c : active)
      out << tria.cell_level(c) << " ";
    out << "\n        </DataArray>\n";

    out << "        <DataArray type=\"UInt8\" Name=\"material_id\""
        << " format=\"ascii\">\n";
    for (CellIndexType c : active)
      out << static_cast<int>(tria.material_ids(c)) << " ";
    out << "\n        </DataArray>\n";

    out << "        <DataArray type=\"UInt8\" Name=\"boundary_id\""
        << " format=\"ascii\">\n";
    for (CellIndexType c : active) {
      // Report the boundary id of any boundary face on this cell,
      // or 255 if the cell is fully interior. Useful for debugging
      // boundary colorization without a separate face export.
      uint8_t bid = 255;
      for (uint8_t lf = 0; lf < Topo::faces_per_cell; ++lf) {
        FaceIndexType fi = tria.cell_faces(c, lf);
        if (tria.face_flags(fi) & FaceFlags::Boundary) {
          bid = tria.boundary_ids(fi);
          break;
        }
      }
      out << static_cast<int>(bid) << " ";
    }
    out << "\n        </DataArray>\n";

    out << "      </CellData>\n";
  }
};
