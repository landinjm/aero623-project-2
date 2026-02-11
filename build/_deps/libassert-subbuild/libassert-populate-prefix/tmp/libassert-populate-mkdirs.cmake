# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/tjmellema/aero623-project-2/build/_deps/libassert-src"
  "/home/tjmellema/aero623-project-2/build/_deps/libassert-build"
  "/home/tjmellema/aero623-project-2/build/_deps/libassert-subbuild/libassert-populate-prefix"
  "/home/tjmellema/aero623-project-2/build/_deps/libassert-subbuild/libassert-populate-prefix/tmp"
  "/home/tjmellema/aero623-project-2/build/_deps/libassert-subbuild/libassert-populate-prefix/src/libassert-populate-stamp"
  "/home/tjmellema/aero623-project-2/build/_deps/libassert-subbuild/libassert-populate-prefix/src"
  "/home/tjmellema/aero623-project-2/build/_deps/libassert-subbuild/libassert-populate-prefix/src/libassert-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/tjmellema/aero623-project-2/build/_deps/libassert-subbuild/libassert-populate-prefix/src/libassert-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/tjmellema/aero623-project-2/build/_deps/libassert-subbuild/libassert-populate-prefix/src/libassert-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
