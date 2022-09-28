
cmake_minimum_required(VERSION 3.6 FATAL_ERROR)
project(onnx2flt)

include(cmake/onnx.cmake)

find_package(protobuf REQUIRED)

add_subdirectory(fbs)

add_executable(onnx2flt main.cc)
configure_onnx()

target_link_libraries(onnx2flt
    flatbuffers
    protobuf::libprotobuf
    onnx)

set_property(GLOBAL PROPERTY CXX_STANDARD 2b)


