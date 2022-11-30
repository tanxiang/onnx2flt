#pragma once
#include "context.hh"

std::string nodeID(const onnx::NodeProto &node);

std::vector<std::vector<const onnx::NodeProto *>>
createNodeVVFromInput(const onnx::ValueInfoProto &, mapContext &);

std::vector<std::vector<const onnx::NodeProto *>>
createNodeVVFromOutputs(const std::vector<std::string> output,
                        mapContext &context);



struct kLayerData //: public flatbuffers::Offset<void> , nn::Layer
{
    size_t key;
  nn::Layer type;
  flatbuffers::Offset<void> data;
  /* data */
};

inline bool operator<(const kLayerData& l, const kLayerData& r) { return l.key < r.key; }
inline bool operator<(const kLayerData& l, const size_t& r) { return l.key < r; }
inline bool operator<(const size_t& l, const kLayerData& r) { return l < r.key; }

std::set<kLayerData,std::less<>>
writeFlNodeFromOutputs(flatbuffers::FlatBufferBuilder &builder,
                       const std::vector<std::string> outputs,
                       mapContext &context);