#pragma once
#include "fbs/gnt_generated.h"
#include <map>
#include <onnx/onnx_pb.h>

struct mapContext {
  std::map<std::string, const onnx::TensorProto &> tensorMap;
  // std::map<std::string, const onnx::SparseTensorProto &> sparseTensorMap;
  std::map<std::string, const onnx::NodeProto &> nodeMap;
  std::multimap<std::string, const onnx::NodeProto &> inputNodeMap;
  std::map<std::string, const onnx::NodeProto &> outputNodeMap;
  std::map<std::string,const onnx::ValueInfoProto &> graphsInputs;
};

