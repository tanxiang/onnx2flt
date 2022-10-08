#pragma once
#include "context.hh"

struct OpToFuncMap
    : public std::map<
          std::string,
          std::function<std::pair<uint8_t, flatbuffers::Offset<void>>(
              flatbuffers::FlatBufferBuilder &, const onnx::NodeProto &,
              mapContext &)>> {
  OpToFuncMap() ;
};
