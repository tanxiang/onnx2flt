
cmake_minimum_required(VERSION 3.6 FATAL_ERROR)
project(onnx2flt)


find_package(FlatBuffers REQUIRED)
find_package(protobuf REQUIRED)

add_executable(onnx2flt main.cc)

set_property(GLOBAL PROPERTY CXX_STANDARD 2b)


