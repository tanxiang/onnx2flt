#pragma once
#include "context.hh"

std::string nodeID(const onnx::NodeProto &node);

std::vector<std::vector<const onnx::NodeProto *>>
createNodeVVFromInput(const onnx::ValueInfoProto &, mapContext &);

std::vector<std::vector<const onnx::NodeProto *>>
createNodeVVFromOutputs(const std::vector<std::string> output,
                        mapContext &context);

struct uLayerData //: public flatbuffers::Offset<void> , nn::Layer
{
  nn::Layer type;
  flatbuffers::Offset<void> data;
  /* data */
};

std::map<int, uLayerData>
writeFlNodeFromOutputs(flatbuffers::FlatBufferBuilder &builder,
                       const std::vector<std::string> outputs,
                       mapContext &context);