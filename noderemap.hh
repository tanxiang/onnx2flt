#pragma once
#include "context.hh"


std::string nodeID(const onnx::NodeProto &node);

std::vector<std::vector<const onnx::NodeProto *>>
createNodeVVFromInput(const onnx::ValueInfoProto &, mapContext &);
