#pragma once

#include <Kokkos_Core.hpp>

/**
 * @brief Host and Device memory spaces.
 */
#ifdef KOKKOS_ENABLE_CUDA
#define DeviceMemSpace Kokkos::CudaSpace
#elif defined(KOKKOS_ENABLE_HIP)
#define DeviceMemSpace Kokkos::HIPSpace
#else
#define DeviceMemSpace Kokkos::HostSpace
#endif

using HostMemSpace = Kokkos::HostSpace;

/**
 * @brief Host and Device execution spaces.
 */
using HostExecSpace = HostMemSpace::execution_space;
using DeviceExecSpace = DeviceMemSpace::execution_space;

/**
 * @brief Host and Device range policies.
 */
using host_range_policy = Kokkos::RangePolicy<HostExecSpace>;
using device_range_policy = Kokkos::RangePolicy<DeviceExecSpace>;

/**
 * @brief Layout of views.
 */
using Layout = Kokkos::LayoutLeft;

/**
 * @brief Vector view.
 */
template<typename RealType, typename MemorySpace>
struct VectorViewTrait
{
  using type = Kokkos::View<RealType*, Layout, MemorySpace>;
};

/**
 * @brief Matrix view.
 */
template<typename RealType, typename MemorySpace>
struct MatrixViewTrait
{
  using type = Kokkos::View<RealType**, Layout, MemorySpace>;
};
