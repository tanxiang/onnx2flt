#pragma once
#include "context.hh"

struct OpToReMap
    : public std::map<
          std::string,
          std::function<void(flatbuffers::FlatBufferBuilder &,
                             const onnx::NodeProto &, mapContext &)>> {
  OpToReMap();
};
std::string nodeID(const onnx::NodeProto &node);

std::vector<std::vector<const onnx::NodeProto *>>
createNodeVVFromInput(const onnx::ValueInfoProto &, mapContext &);
