#pragma once

#include <Kokkos_Core.hpp>
#include <config.hpp>
#include <iostream>

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
  using iterator = RealType*;
  using const_iterator = const RealType*;

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
   * @brief Copy constructor.
   *
   * Note that this may lose or gain accuracy depending on what the template
   * parameters are.
   */
  template<typename OtherRealType>
  Vector(const Vector<OtherRealType, MemorySpace>& vector)
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
  template<typename OtherMemorySpace>
  void import(const Vector<RealType, OtherMemorySpace>& src,
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
  size_type size() const { return data.size(); };

  /**
   * @brief Whether the vector is empty.
   */
  bool empty() const { return size() == 0; };

  /**
   * @brief Set all components equal to some scalar.
   */
  Vector<RealType, MemorySpace>& operator=(const RealType value)
  {
    Kokkos::deep_copy(data, value);
    return *this;
  }

  /**
   * @brief Equality test between two vectors.
   */
  bool operator==(const Vector<RealType, MemorySpace>& vector)
  {
    if (size() != vector.size()) {
      return false;
    }

    unsigned int difference = 0;
    auto dst = data;
    auto src = vector.data;
    Kokkos::parallel_reduce(
      Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, size()),
      KOKKOS_LAMBDA(const size_type i, unsigned int& local) {
        local += (dst(i) != src(i)) ? 1 : 0;
      },
      difference);

    return difference == 0;
  }

  /**
   * @brief Inequality test between two vectors.
   */
  bool operator!=(const Vector<RealType, MemorySpace>& vector)
  {
    return !(*this == vector);
  }

  /**
   * @brief Scalar product of two vectors.
   */
  RealType operator*(const Vector<RealType, MemorySpace>& vector) const
  {
    KOKKOS_ASSERT(size() == vector.size());

    RealType result = 0;
    auto dst = data;
    auto src = vector.data;
    Kokkos::parallel_reduce(
      Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, size()),
      KOKKOS_LAMBDA(const size_type i, RealType& local) {
        local += dst(i) * src(i);
      },
      result);

    return result;
  }

  /**
   * @brief Mean value.
   */
  RealType mean() const
  {
    RealType result = 0;
    auto src = data;
    Kokkos::parallel_reduce(
      Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, size()),
      KOKKOS_LAMBDA(const size_type i, RealType& local) { local += src(i); },
      result);
    return result / static_cast<RealType>(size());
  }

  /**
   * @brief Square of L2-norm.
   */
  RealType norm_square() const
  {
    RealType result = 0;
    auto src = data;
    Kokkos::parallel_reduce(
      Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, size()),
      KOKKOS_LAMBDA(const size_type i, RealType& local) {
        local += src(i) * src(i);
      },
      result);
    return result;
  }

  /**
   * @brief L1-norm.
   */
  RealType l1_norm() const
  {
    RealType result = 0;
    auto src = data;
    Kokkos::parallel_reduce(
      Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, size()),
      KOKKOS_LAMBDA(const size_type i, RealType& local) {
        local += Kokkos::abs(src(i));
      },
      result);
    return result;
  }

  /**
   * @brief L2-norm.
   */
  RealType l2_norm() const { return Kokkos::sqrt(norm_square()); }

  /**
   * @brief Add and dot.
   */
  RealType add_and_dot(const RealType value,
                       const Vector<RealType, MemorySpace>& vector,
                       const Vector<RealType, MemorySpace>& other_vector) const
  {
    KOKKOS_ASSERT(size() == vector.size());
    KOKKOS_ASSERT(size() == other_vector.size());

    RealType result = 0;
    auto dst = data;
    auto src = vector.data;
    auto other = other_vector.data;
    Kokkos::parallel_reduce(
      Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, size()),
      KOKKOS_LAMBDA(const size_type i, RealType& local) {
        local += (dst(i) + value * src(i)) * other(i);
      },
      result);

    return result;
  };

  /**
   * @brief Index.
   */
  RealType operator[](const size_type i) const
  {
    static_assert(
      Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemorySpace>::accessible,
      "[] operators are only valid for host-accessible memory spaces.");
    return data(i);
  }

  /**
   * @brief Index with writeable reference.
   */
  RealType& operator[](const size_type i)
  {
    static_assert(
      Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemorySpace>::accessible,
      "[] operators are only valid for host-accessible memory spaces.");
    return data(i);
  }

  /**
   * @brief Index — device and host accessible.
   */
  KOKKOS_INLINE_FUNCTION
  RealType operator()(const size_type i) const { return data(i); }

  /**
   * @brief Index with writeable reference — device and host accessible.
   */
  KOKKOS_INLINE_FUNCTION
  RealType& operator()(const size_type i) { return data(i); }

  KOKKOS_INLINE_FUNCTION
  ViewType& view() { return data; }

  KOKKOS_INLINE_FUNCTION
  const ViewType& view() const { return data; }

  /**
   * @brief Iterator to beginning of range.
   */
  iterator begin()
  {
    static_assert(
      Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemorySpace>::accessible,
      "Vector iterators are only allowed for host-accessible memory spaces.");
    return data.data();
  }

  /**
   * @brief Const iterator to beginning of range.
   */
  const_iterator begin() const
  {
    static_assert(
      Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemorySpace>::accessible,
      "Vector iterators are only allowed for host-accessible memory spaces.");
    return data.data();
  }

  /**
   * @brief Const iterator to beginning of range.
   */
  const_iterator cbegin() const
  {
    static_assert(
      Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemorySpace>::accessible,
      "Vector iterators are only allowed for host-accessible memory spaces.");
    return data.data();
  }

  /**
   * @brief Iterator to end of range.
   */
  iterator end()
  {
    static_assert(
      Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemorySpace>::accessible,
      "Vector iterators are only allowed for host-accessible memory spaces.");
    return data.data() + data.size();
  }

  /**
   * @brief Const iterator to end of range.
   */
  const_iterator end() const
  {
    static_assert(
      Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemorySpace>::accessible,
      "Vector iterators are only allowed for host-accessible memory spaces.");
    return data.data() + data.size();
  }

  /**
   * @brief Const iterator to end of range.
   */
  const_iterator cend() const
  {
    static_assert(
      Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemorySpace>::accessible,
      "Vector iterators are only allowed for host-accessible memory spaces.");
    return data.data() + data.size();
  }

  /**
   * @brief Add vector to current one.
   */
  Vector<RealType, MemorySpace>& operator+=(
    const Vector<RealType, MemorySpace>& vector)
  {
    KOKKOS_ASSERT(size() == vector.size());
    auto dst = data;
    auto src = vector.data;
    Kokkos::parallel_for(
      Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, size()),
      KOKKOS_LAMBDA(const size_type i) { dst(i) += src(i); });
    return *this;
  }

  /**
   * @brief Add two vectors.
   */
  friend Vector<RealType, MemorySpace> operator+(
    Vector<RealType, MemorySpace> lhs,
    const Vector<RealType, MemorySpace>& rhs)
  {
    lhs += rhs;
    return lhs;
  }

  /**
   * @brief Subtract vector from current one.
   */
  Vector<RealType, MemorySpace>& operator-=(
    const Vector<RealType, MemorySpace>& vector)
  {
    KOKKOS_ASSERT(size() == vector.size());
    auto dst = data;
    auto src = vector.data;
    Kokkos::parallel_for(
      Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, size()),
      KOKKOS_LAMBDA(const size_type i) { dst(i) -= src(i); });
    return *this;
  }

  /**
   * @brief Subtract two vectors.
   */
  friend Vector<RealType, MemorySpace> operator-(
    Vector<RealType, MemorySpace> lhs,
    const Vector<RealType, MemorySpace>& rhs)
  {
    lhs -= rhs;
    return lhs;
  }

  /**
   * @brief Multiply current vector by some scalar value.
   */
  Vector<RealType, MemorySpace>& operator*=(const RealType value)
  {
    auto dst = data;
    Kokkos::parallel_for(
      Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, size()),
      KOKKOS_LAMBDA(const size_type i) { dst(i) *= value; });
    return *this;
  }

  /**
   * @brief Multiply vector by scalar.
   */
  friend Vector<RealType, MemorySpace> operator*(
    Vector<RealType, MemorySpace> lhs,
    const RealType value)
  {
    lhs *= value;
    return lhs;
  }

  /**
   * @brief Divide current vector by some scalar value.
   */
  Vector<RealType, MemorySpace>& operator/=(const RealType value)
  {
    auto dst = data;
    Kokkos::parallel_for(
      Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, size()),
      KOKKOS_LAMBDA(const size_type i) { dst(i) /= value; });
    return *this;
  }

  /**
   * @brief Divide vector by scalar.
   */
  friend Vector<RealType, MemorySpace> operator/(
    Vector<RealType, MemorySpace> lhs,
    const RealType value)
  {
    lhs /= value;
    return lhs;
  }

  /**
   * @brief Add scalar value to current vector.
   */
  void add(const RealType value)
  {
    auto dst = data;
    Kokkos::parallel_for(
      Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, size()),
      KOKKOS_LAMBDA(const size_type i) { dst(i) += value; });
  }

  /**
   * @brief Add vector to current one.
   */
  void add(const Vector<RealType, MemorySpace>& vector)
  {
    KOKKOS_ASSERT(size() == vector.size());
    *this += vector;
  }

  /**
   * @brief Add scaled vector to current one.
   */
  void add(const RealType value, const Vector<RealType, MemorySpace>& vector)
  {
    KOKKOS_ASSERT(size() == vector.size());
    auto dst = data;
    auto src = vector.data;
    Kokkos::parallel_for(
      Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, size()),
      KOKKOS_LAMBDA(const size_type i) { dst(i) += value * src(i); });
  }

  /**
   * @brief Add vector to a scaled version of the current one.
   */
  void sadd(const RealType value, const Vector<RealType, MemorySpace>& vector)
  {
    KOKKOS_ASSERT(size() == vector.size());
    auto dst = data;
    auto src = vector.data;
    Kokkos::parallel_for(
      Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, size()),
      KOKKOS_LAMBDA(const size_type i) { dst(i) = value * dst(i) + src(i); });
  }

  /**
   * @brief Add scaled vector to a scaled version of the current one.
   */
  void sadd(const RealType value,
            const RealType value_2,
            const Vector<RealType, MemorySpace>& vector)
  {
    KOKKOS_ASSERT(size() == vector.size());
    auto dst = data;
    auto src = vector.data;
    Kokkos::parallel_for(
      Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, size()),
      KOKKOS_LAMBDA(const size_type i) {
        dst(i) = value * dst(i) + value_2 * src(i);
      });
  }

  /**
   * @brief Scale each element of the current vector by those in the scaling
   * vector.
   */
  void scale(const Vector<RealType, MemorySpace>& vector)
  {
    KOKKOS_ASSERT(size() == vector.size());
    auto dst = data;
    auto src = vector.data;
    Kokkos::parallel_for(
      Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, size()),
      KOKKOS_LAMBDA(const size_type i) { dst(i) *= src(i); });
  }

  /**
   * @brief Print vector to some output stream
   */
  void print(std::ostream& out,
             const unsigned int precision = 3,
             const bool scientific = true,
             const bool across = true) const
  {
    static_assert(
      Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemorySpace>::accessible,
      "print() is only allowed for host-accessible memory spaces.");

    std::ios::fmtflags old_flags = out.flags();
    unsigned int old_precision = out.precision();

    out.precision(precision);
    out << (scientific ? std::scientific : std::fixed);

    if (across) {
      for (size_type i = 0; i < data.size(); ++i) {
        out << data(i);
        if (i < data.size() - 1)
          out << ' ';
      }
      out << '\n';
    } else {
      for (size_type i = 0; i < data.size(); ++i)
        out << data(i) << '\n';
    }

    out.flags(old_flags);
    out.precision(old_precision);
  }

  /**
   * @brief Estimate the memory usage of the vector.
   */
  std::size_t memory_consumption() const
  {
    return sizeof(*this) + data.size() * sizeof(RealType);
  }

public:
  template<typename RealType2, typename MemorySpace2>
  friend class Vector;
  
  RealType* raw_data() { return data.data(); }
  const RealType* raw_data() const { return data.data(); }

private:
  ViewType data;
};
