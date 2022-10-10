#pragma once
#include "context.hh"


struct OpToReMap
    : public std::map<
          std::string,
          std::function<void(flatbuffers::FlatBufferBuilder &,
                             const onnx::NodeProto &, mapContext &)>> {
  OpToReMap() ;
};

std::vector<std::vector<onnx::NodeProto*>> createNodeVVFromInput(const onnx::ValueInfoProto&,mapContext&);

