#pragma once

#include <Kokkos_Core.hpp>
#include <string>

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

/**
 * @brief Assertion message
 */
struct AssertMessage
{
public:
  AssertMessage(const char* m)
    : msg(m) {};

  AssertMessage(std::string m)
    : msg(std::move(m)) {};

  const char* c_str() const { return msg.c_str(); }

private:
  std::string msg;
};

/**
 * @brief Assertion that always runs
 */
#define ASSERT_THROW(cond, msg)                                                \
  do {                                                                         \
    if (!bool(cond)) {                                                         \
      KOKKOS_IF_ON_HOST(                                                       \
        (::Kokkos::abort(                                                      \
           (std::string("Kokkos contract violation:\n"                         \
                        "  Asserted condition `" #cond "` evaluated false.\n"  \
                        "  Message: ") +                                       \
            AssertMessage(msg).c_str() +                                       \
            "\n  Error at " __FILE__ ":" KOKKOS_IMPL_TOSTRING(__LINE__) "\n")  \
             .c_str());))                                                      \
      KOKKOS_IF_ON_DEVICE(                                                     \
        (::Kokkos::abort("Kokkos contract violation:\n"                        \
                         "  Asserted condition `" #cond "` evaluated false.\n" \
                         "  Error at " __FILE__                                \
                         ":" KOKKOS_IMPL_TOSTRING(__LINE__) "\n");))           \
    }                                                                          \
  } while (false)

/**
 * @brief Assertion that only runs in DEBUG mode
 */
#ifndef NDEBUG
#define ASSERT(cond, msg) ASSERT_THROW(cond, msg)
#else
#define ASSERT(cond, msg) ((void)0)
#endif
