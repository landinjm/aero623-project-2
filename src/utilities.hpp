#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

template<typename RealType>
inline RealType
l1_norm(const std::vector<RealType>& vec)
{
  RealType error = RealType{ 0 };
  for (const auto& val : vec) {
    error += std::abs(val);
  }
  return error;
}

template<typename RealType>
inline RealType
l2_norm(const std::vector<RealType>& vec)
{
  RealType error = RealType{ 0 };
  for (const auto& val : vec) {
    error += val * val;
  }
  return std::sqrt(error);
}

template<typename RealType>
inline void
zero(std::vector<RealType>& vec)
{
  std::fill(vec.begin(), vec.end(), RealType{ 0 });
}

template<typename RealType>
inline void
set(std::vector<RealType>& vec, RealType val)
{
  std::fill(vec.begin(), vec.end(), val);
}

template<typename RealType>
inline RealType
min(const std::vector<RealType>& vec)
{
  return *std::min_element(vec.begin(), vec.end());
}

template<typename RealType>
inline RealType
max(const std::vector<RealType>& vec)
{
  return *std::max_element(vec.begin(), vec.end());
}
