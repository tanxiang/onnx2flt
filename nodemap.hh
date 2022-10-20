#pragma once
#include "context.hh"

struct OpToFuncMap
    : public std::map<
          std::string,
          std::function<std::pair<uint8_t, flatbuffers::Offset<void>>(
              flatbuffers::FlatBufferBuilder &,  std::vector<const onnx::NodeProto *> &,
              mapContext &)>> {
  OpToFuncMap() ;
};
