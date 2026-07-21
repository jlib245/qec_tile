cmake_minimum_required(VERSION 3.15...3.30)

# C++20을 쓰기 위해 최신 g++(>=11)를 우선 탐색. 없으면 기본 컴파일러로 폴백.
# 외부에서 CXX / CMAKE_CXX_COMPILER를 지정했다면 그쪽을 존중한다.
if(NOT CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX})
  find_program(_QEC_GXX NAMES g++-14 g++-13 g++-12 g++-11)
  if(_QEC_GXX)
    set(CMAKE_CXX_COMPILER "${_QEC_GXX}")
  endif()
endif()

project(qec_tile LANGUAGES CXX)
message(STATUS "qec_tile: C++ compiler = ${CMAKE_CXX_COMPILER}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

# nanobind의 cmake config 위치 (빌드 의존성으로 설치됨)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE nanobind_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH "${nanobind_DIR}")
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core src/bindings/module.cpp)
target_include_directories(_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp)

install(TARGETS _core LIBRARY DESTINATION qec_tile)
