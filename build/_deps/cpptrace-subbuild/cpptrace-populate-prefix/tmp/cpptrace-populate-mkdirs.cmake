# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/tjmellema/aero623-project-2/build/_deps/cpptrace-src"
  "/home/tjmellema/aero623-project-2/build/_deps/cpptrace-build"
  "/home/tjmellema/aero623-project-2/build/_deps/cpptrace-subbuild/cpptrace-populate-prefix"
  "/home/tjmellema/aero623-project-2/build/_deps/cpptrace-subbuild/cpptrace-populate-prefix/tmp"
  "/home/tjmellema/aero623-project-2/build/_deps/cpptrace-subbuild/cpptrace-populate-prefix/src/cpptrace-populate-stamp"
  "/home/tjmellema/aero623-project-2/build/_deps/cpptrace-subbuild/cpptrace-populate-prefix/src"
  "/home/tjmellema/aero623-project-2/build/_deps/cpptrace-subbuild/cpptrace-populate-prefix/src/cpptrace-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/tjmellema/aero623-project-2/build/_deps/cpptrace-subbuild/cpptrace-populate-prefix/src/cpptrace-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/tjmellema/aero623-project-2/build/_deps/cpptrace-subbuild/cpptrace-populate-prefix/src/cpptrace-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
