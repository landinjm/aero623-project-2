cmake_minimum_required(VERSION 3.18)
set(PROJECT_NAME proj-2)
project(${PROJECT_NAME} VERSION 1.0)

# Config options
option(BUILD_EXAMPLES "Build example applications" OFF)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_SHARED_LIBS "Use shared libraries" ON)

# Global settings
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Vendor dependencies
include(FetchContent)

# libassert
FetchContent_Declare(
  libassert
  GIT_REPOSITORY https://github.com/jeremy-rifkin/libassert.git
  GIT_TAG v2.2.1
)

# Kokkos
find_package(
  Kokkos
  QUIET
  HINTS
    ${Kokkos_DIR}
    $ENV{Kokkos_DIR}
)

if(NOT ${Kokkos_FOUND})
  FetchContent_Declare(
    Kokkos
    URL
      https://github.com/kokkos/kokkos/releases/download/4.5.01/kokkos-4.5.01.tar.gz
    URL_HASH
      SHA256=52d003ffbbe05f30c89966e4009c017efb1662b02b2b73190670d3418719564c
  )
  FetchContent_MakeAvailable(Kokkos)
endif()

# Google Test suite
FetchContent_Declare(
  googletest
  URL
    https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

FetchContent_MakeAvailable(
  libassert
  googletest
)

# The source and header files
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "src/*.hpp")

# Add the main file
list(APPEND SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

# Add executable and link libraries
add_executable(
  ${PROJECT_NAME}
  ${SOURCES}
  ${HEADERS}
)
target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(
  ${PROJECT_NAME}
  PRIVATE
    libassert::assert
    Kokkos::kokkos
)

# Tests
if(BUILD_TESTS)
  enable_testing()

  # Grab the test files
  file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp")

  # Test executable
  add_executable(${PROJECT_NAME}_tests ${TEST_SOURCES})

  # Link our sources (except for main.cpp)
  list(FILTER SOURCES EXCLUDE REGEX ".*main\\.cpp$")
  target_sources(${PROJECT_NAME}_tests PRIVATE ${SOURCES})
  target_include_directories(
    ${PROJECT_NAME}_tests
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  target_link_libraries(
    ${PROJECT_NAME}_tests
    PRIVATE
      libassert::assert
      Kokkos::kokkos
      GTest::gtest_main
  )

  include(GoogleTest)
  gtest_discover_tests(${PROJECT_NAME}_tests)
endif()
