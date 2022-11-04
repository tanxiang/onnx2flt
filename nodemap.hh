#pragma once
#include "context.hh"

struct OpToFuncMap
    : public std::map<
          std::string,
          std::function<std::pair<nn::Layer, flatbuffers::Offset<void>>(
              flatbuffers::FlatBufferBuilder &,  std::vector<const onnx::NodeProto *> &,
              mapContext &)>> {
  OpToFuncMap() ;
};
