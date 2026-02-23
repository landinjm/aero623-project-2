#include <Kokkos_Core.hpp>
#include <config.hpp>
#include <timer.hpp>

template<typename RealType, typename MemorySpace>
struct VectorViewTrait;

template<>
struct VectorViewTrait<double, HostMemSpace>
{
  using type = ViewHostVectorType;
};

template<>
struct VectorViewTrait<double, DeviceMemSpace>
{
  using type = ViewDeviceVectorType;
};

enum class VectorOperation
{
  insert,
  add,
  min,
  max
};

template<typename RealType, typename MemorySpace>
class Vector
{
public:
  using ViewType = typename VectorViewTrait<RealType, MemorySpace>::type;
  using value_type = RealType;
  using size_type = typename ViewType::size_type;

  /**
   * @brief Default constructor.
   */
  Vector() = default;

  /**
   * @brief Constructor with specific size
   */
  explicit Vector(const size_type size)
    : data("vector", size) {};

  /**
   * @brief Copy constructor.
   */
  Vector(const Vector<RealType, MemorySpace>& vector)
    : data("vector", vector.size())
  {
    Kokkos::deep_copy(data, vector.data);
  };

  /**
   * @brief Move constructor.
   */
  Vector(Vector<RealType, MemorySpace>&& vector) noexcept
    : data(std::move(vector.data)) {};

  /**
   * @brief Copy assignment.
   */
  Vector<RealType, MemorySpace>& operator=(
    const Vector<RealType, MemorySpace>& vector) = delete;

  /**
   * @brief Move assignment.
   */
  Vector<RealType, MemorySpace>& operator=(
    Vector<RealType, MemorySpace>&& vector) noexcept
  {
    if (this == &vector) {
      return *this;
    }
    data = std::move(vector.data);
    return *this;
  }

  /**
   * @brief Reinitialize the vector.
   */
  void reinit(size_type size)
  {
    if (size == data.size()) {
      return;
    }

    Kokkos::resize(data, size);
  }

  /**
   * @brief Swap two vectors.
   */
  void swap(Vector<RealType, MemorySpace>& vector) noexcept
  {
    using std::swap;
    swap(data, vector.data);
  }

  /**
   * @brief Swap two vectors.
   */
  friend void swap(Vector<RealType, MemorySpace>& vector_1,
                   Vector<RealType, MemorySpace>& vector_2) noexcept
  {
    vector_1.swap(vector_2);
  }

  /**
   * @brief Import data from one memory space to another.
   */
  template<typename MemorySpace2>
  void import(const Vector<RealType, MemorySpace2>& src,
              VectorOperation operation)
  {
    KOKKOS_ASSERT(size() == src.size());

    if (operation == VectorOperation::insert) {
      Kokkos::deep_copy(data, src.data);
      return;
    }

    ViewType tmp(Kokkos::view_alloc("tmp", Kokkos::WithoutInitializing),
                 src.size());
    Kokkos::deep_copy(tmp, src.data);
    auto dst = data;

    if (operation == VectorOperation::add) {
      Kokkos::parallel_for(
        Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, size()),
        KOKKOS_LAMBDA(const size_type i) { dst(i) += tmp(i); });
    } else if (operation == VectorOperation::min) {
      Kokkos::parallel_for(
        Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, size()),
        KOKKOS_LAMBDA(const size_type i) {
          dst(i) = Kokkos::min(dst(i), tmp(i));
        });
    } else if (operation == VectorOperation::max) {
      Kokkos::parallel_for(
        Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, size()),
        KOKKOS_LAMBDA(const size_type i) {
          dst(i) = Kokkos::max(dst(i), tmp(i));
        });
    }
  }

  /**
   * @brief Default destructor.
   */
  ~Vector() = default;

  /**
   * @brief Size.
   */
  std::size_t size() const { return data.size(); };

private:
  template<typename RealType2, typename MemorySpace2>
  friend class Vector;

  ViewType data;
};

int
main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  {
    // Kokkos testing
    int n_rows = 1000;
    int n_columns = 10000;
    int total_dofs = n_rows * n_columns;
    int n_iterations = 100;

    // Create arrays and set the values
    Timer::instance().begin_section("Allocation - Kokkos");
    ViewDeviceVectorType x("X", n_columns);
    ViewDeviceVectorType y("Y", n_rows);
    ViewDeviceMatrixType A("A", n_rows, n_columns);

    // Copy the data to the device
    Kokkos::deep_copy(x, 1.0);
    Kokkos::deep_copy(y, 1.0);
    Kokkos::deep_copy(A, 1.0);

    Timer::instance().end_section("Allocation - Kokkos");

    // Matrix multiplication some number of iterations
    Timer::instance().begin_section("Multiplication - Kokkos");
    for (int iteration = 0; iteration < n_iterations; ++iteration) {
      double result = 0.0;
      Kokkos::parallel_reduce(
        "yAx",
        device_range_policy(0, n_rows),
        KOKKOS_LAMBDA(int j, double& update) {
          double temp2 = 0;

          for (int i = 0; i < n_columns; ++i) {
            temp2 += A(j, i) * x(i);
          }

          update += y(j) * temp2;
        },
        result);
    }
    Timer::instance().end_section("Multiplication - Kokkos");
  }
  Kokkos::finalize();

  return 0;
}
