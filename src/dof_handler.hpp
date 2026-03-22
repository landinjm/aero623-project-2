#pragma once

#include <Kokkos_Core.hpp>
#include <config.hpp>
#include <fe.hpp>
#include <iostream>
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
  using HostIndexView = typename IndexView::host_mirror_type;

  struct CellAccessor
  {
    ::CellAccessor<dim> tria_cell;
    const DoFHandler* handler;

    CellIndexType index() const { return tria_cell.index; }
    double measure() const { return tria_cell.measure(); }

    auto n_vertices() const { return tria_cell.n_vertices(); }

    auto vertex(LocalIndexType v) const { return tria_cell.vertex(v); }

    unsigned int n_geometry_nodes() const
    {
      return tria_cell.n_geometry_nodes();
    }

    auto geometry_node(LocalIndexType i) const
    {
      return tria_cell.geometry_node(i);
    }

    ::FaceAccessor<dim> face(LocalIndexType local_f) const
    {
      return tria_cell.face(local_f);
    }

    bool face_at_boundary(LocalIndexType local_f) const
    {
      return tria_cell.face_at_boundary(local_f);
    }

    bool face_is_periodic(LocalIndexType local_f) const
    {
      return tria_cell.face_is_periodic(local_f);
    }

    BoundaryIdType face_boundary_id(LocalIndexType local_f) const
    {
      return tria_cell.face_boundary_id(local_f);
    }

    CellIndexType neighbor_index(LocalIndexType local_f) const
    {
      return tria_cell.neighbor_index(local_f);
    }

    CellAccessor neighbor(LocalIndexType local_f) const
    {
      return CellAccessor{ tria_cell.neighbor(local_f), handler };
    }

    LocalIndexType neighbor_face_index(LocalIndexType local_f) const
    {
      const auto global_f = tria_cell.face_index(local_f);
      const auto neighbor = tria_cell.neighbor(local_f);
      for (LocalIndexType lf = 0; lf < ::SimplexTopology<dim>::faces_per_cell;
           ++lf)
        if (neighbor.face_index(lf) == global_f)
          return lf;
      ASSERT(false, "Could not find neighbor face index");
      return 0;
    }

    CellAccessor periodic_neighbor(LocalIndexType local_f) const
    {
      ASSERT(tria_cell.face_is_periodic(local_f), "Face is not periodic");
      const auto neighbor_face = tria_cell.face(local_f).periodic_neighbor();
      const auto neighbor_cell_idx =
        tria_cell.tria->face_cells(neighbor_face.index, 0);
      return CellAccessor{ tria_cell.tria->get_cell(neighbor_cell_idx),
                           handler };
    }

    void get_dof_indices(std::vector<size_type>& indices) const
    {
      const size_type k = tria_cell.index;
      const size_type ndpc = handler->n_dofs_per_cell_;
      indices.resize(ndpc);
      for (size_type i = 0; i < ndpc; ++i) {
        indices[i] = handler->cell_dof_indices_host_(k, i);
      }
    }

    size_type dof_index(size_type local) const
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
    , n_cells_(0)
    , n_dofs_per_cell_(fe.n_dofs())
    , n_dofs_total_(0)
  {
    ASSERT(tria.n_cells() > 0, "Triangulation must have at least one cell");
    distribute_dofs();
  }

  void distribute_dofs()
  {
    n_cells_ = tria_.n_cells();
    n_dofs_total_ = n_cells_ * n_dofs_per_cell_;

    const size_type total_cells = tria_.n_cells();

    cell_dof_indices_device_ =
      IndexView("cell_dof_indices", total_cells, n_dofs_per_cell_);
    cell_dof_indices_host_ =
      Kokkos::create_mirror_view(cell_dof_indices_device_);

    // Fill all entries with invalid entry first
    constexpr size_type invalid = std::numeric_limits<size_type>::max();
    for (size_type k = 0; k < total_cells; ++k) {
      for (size_type i = 0; i < n_dofs_per_cell_; ++i) {
        cell_dof_indices_host_(k, i) = invalid;
      }
    }

    // Assign contiguous DOF blocks to active cells only
    size_type next_dof = 0;
    for (auto cell : tria_.active_cell_range()) {
      const size_type k = cell.index;
      for (size_type i = 0; i < n_dofs_per_cell_; ++i) {
        cell_dof_indices_host_(k, i) = next_dof++;
      }
    }

    ASSERT(next_dof == n_dofs_total_, "DOF count mismatch after distribution");

    Kokkos::deep_copy(cell_dof_indices_device_, cell_dof_indices_host_);
  }

  size_type n_cells() const { return n_cells_; }
  size_type n_dofs_per_cell() const { return n_dofs_per_cell_; }
  size_type n_dofs() const { return n_dofs_total_; }

  const FE_DGQLegendre<dim, RealType>& fe() const { return fe_; }
  const Triangulation<dim>& tria() const { return tria_; }

  CellAccessor cell(CellIndexType k) const
  {
    ASSERT(k < tria_.n_cells(), "Cell index out of range");
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
