cmake_minimum_required(VERSION 3.18)
set(PROJECT_NAME proj-2)
project(${PROJECT_NAME} VERSION 1.0)

# Config options
option(BUILD_EXAMPLES "Build example applications" OFF)

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
  GIT_TAG        v2.2.1
)

# Google Test suite
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

FetchContent_MakeAvailable(libassert googletest)

# Add executable and link libraries
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(
	${PROJECT_NAME}
	PRIVATE
	libassert::assert
)


