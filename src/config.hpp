#pragma once

#include <Kokkos_Core.hpp>

/**
 * @brief Host and Device memory spaces.
 */
#ifdef KOKKOS_ENABLE_CUDA
#define DeviceMemSpace Kokkos::CudaSpace
#endif

#ifndef DeviceMemSpace
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
