#pragma once

#include <Kokkos_Core.hpp>
#include <config.hpp>
#include <fe.hpp>
#include <triangulation.hpp>

/**
 * @brief Collection of all degrees of freedom across the mesh.
 *
 * Since we have a DG scheme, each cell owns its own dofs.
 */
template<unsigned int dim, typename RealType>
class DoFHandler
{
public:
  using size_type = uint32_t;
  using IndexView = typename MatrixViewTrait<size_type, DeviceMemSpace>::type;
  using HostIndexView = typename IndexView::HostMirror;

  struct CellAccessor
  {
    ::CellAccessor<dim> tria_cell; // your existing accessor
    const DoFHandler* handler;

    // Forward all geometry queries to the triangulation accessor
    CellIndexType index() const { return tria_cell.index; }
    bool is_active() const { return tria_cell.is_active(); }
    double measure() const { return tria_cell.measure(); }
    double diameter() const { return tria_cell.diameter(); }

    // Vertex coordinates, face accessors, etc. — all delegated
    auto vertex(unsigned int v) const { return tria_cell.vertex(v); }
    bool face_at_boundary(uint8_t local_f) const
    {
      return tria_cell.face_at_boundary(local_f);
    }
    BoundaryIdType face_boundary_id(uint8_t local_f) const
    {
      return tria_cell.face_boundary_id(local_f);
    }
    CellIndexType neighbor_index(uint8_t local_f) const
    {
      FaceIndexType fi = tria_cell.face_index(local_f);
      auto c0 = tria_cell.tria->face_cells(fi, 0);
      auto c1 = tria_cell.tria->face_cells(fi, 1);
      // Return whichever side is not this cell
      return (c0 == tria_cell.index) ? c1 : c0;
    }

    // DOF query — the only thing DoFHandler adds
    void get_dof_indices(std::vector<uint32_t>& indices) const
    {
      const size_type k = tria_cell.index;
      const size_type ndpc = handler->n_dofs_per_cell_;
      indices.resize(ndpc);
      for (size_type i = 0; i < ndpc; ++i)
        indices[i] = handler->cell_dof_indices_host_(k, i);
    }

    uint32_t dof_index(unsigned int local) const
    {
      return handler->cell_dof_indices_host_(tria_cell.index, local);
    }
  };

  struct ActiveCellRange
  {
    const DoFHandler* handler;

    struct Iterator
    {
      typename Triangulation<dim>::ActiveCellRange::Iterator tria_it;
      const DoFHandler* handler;

      CellAccessor operator*() const { return { *tria_it, handler }; }
      Iterator& operator++()
      {
        ++tria_it;
        return *this;
      }
      bool operator!=(const Iterator& o) const { return tria_it != o.tria_it; }
    };

    Iterator begin() const
    {
      auto r = handler->tria_.active_cell_range();
      return { r.begin(), handler };
    }
    Iterator end() const
    {
      auto r = handler->tria_.active_cell_range();
      return { r.end(), handler };
    }
  };

  DoFHandler(const Triangulation<dim>& tria,
             const FE_DGQLegendre<dim, RealType>& fe)
    : tria_(tria)
    , fe_(fe)
    , n_cells_(tria.n_cells())
    , n_dofs_per_cell_(fe.n_dofs())
    , n_dofs_total_(0)
  {
    ASSERT(tria.n_cells() > 0, "Triangulation must have at least one cell");
    distribute_dofs();
  }

  void distribute_dofs()
  {
    n_cells_ = tria_.n_active_cells();
    n_dofs_total_ = n_cells_ * n_dofs_per_cell_;

    cell_dof_indices_device_ =
      IndexView("cell_dof_indices", tria_.n_cells(), n_dofs_per_cell_);
    cell_dof_indices_host_ =
      Kokkos::create_mirror_view(cell_dof_indices_device_);

    // DG: each active cell gets a contiguous block of global DOF indices.
    // Inactive cells (refined away) are assigned invalid sentinel values.
    size_type next_dof = 0;
    for (auto cell : tria_.active_cell_range()) {
      const size_type k = cell.index;
      for (size_type i = 0; i < n_dofs_per_cell_; ++i)
        cell_dof_indices_host_(k, i) = next_dof++;
    }

    // Fill inactive cells with sentinel so bugs surface immediately
    constexpr size_type invalid = std::numeric_limits<size_type>::max();
    for (size_type k = 0; k < tria_.n_cells(); ++k) {
      auto c = tria_.get_cell(k);
      if (!c.is_active())
        for (size_type i = 0; i < n_dofs_per_cell_; ++i)
          cell_dof_indices_host_(k, i) = invalid;
    }

    Kokkos::deep_copy(cell_dof_indices_device_, cell_dof_indices_host_);

    ASSERT(next_dof == n_dofs_total_, "DOF count mismatch after distribution");
  }

  unsigned int n_cells() const { return n_cells_; }
  unsigned int n_dofs_per_cell() const { return n_dofs_per_cell_; }
  unsigned int n_dofs() const { return n_dofs_total_; }

  const FE_DGQLegendre<dim, RealType>& fe() const { return fe_; }
  const Triangulation<dim>& tria() const { return tria_; }

  CellAccessor cell(CellIndexType k) const
  {
    return { tria_.get_cell(k), this };
  }

  IndexView cell_dof_indices() const { return cell_dof_indices_device_; }

  ActiveCellRange active_cell_range() const { return { this }; }

private:
  const Triangulation<dim>& tria_;
  const FE_DGQLegendre<dim, RealType>& fe_;

  size_type n_cells_;
  size_type n_dofs_per_cell_;
  size_type n_dofs_total_;

  IndexView cell_dof_indices_device_;
  HostIndexView cell_dof_indices_host_;
};
